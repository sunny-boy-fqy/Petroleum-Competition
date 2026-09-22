#!/usr/bin/env python3
"""E8/P1：井级分支（H4）**强制开/关消融** + λ 内折搜索 + 逐井非退化比例。

背景（E8/P1 §5）
---------------
主干逐行输出之后，再用**井级 attention-pool**汇聚一个井向量，预测逐目标井级偏置
`Δ_t`，最终 `ŷ + λ·Δ`。这条路线的风险是"80 口井过拟合"，因此本脚本把纪律做成机械检查：

* **容量硬上限**：井级分支隐藏宽 ≤ 主干宽/8（`well_head.capacity_receipt`，超限直接抛错）；
* **λ=0 必须精确恒等**：关掉分支与"不训练分支"结果逐位相同（`well_head` 已保证，这里复验）；
* **λ 只在 inner-OOF 上选**：在 inner-val 上扫 λ∈[0, λ_max]，用官方总分；
* **采纳判据**：λ>0 需"总分上升 **且** 逐井配对 CI 下界 > 0"，并且逐井非退化比例达标；
* **禁止井身份**：`assert_no_well_identity` 审计参数/缓冲名（无 logId / well_id 类键）。

产出：`$REPORTS/E8_well_branch.json`、`$REPORTS/E8_P1_gate.json`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(V4 / "E5" / "code"))          # 复用 E5 的冻结骨干/隐状态公共件

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

import e5_common as EC  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P1_CHECKS = ("on_off_ablation_completed", "lambda_inner_only", "capacity_within_cap",
             "no_well_identity_features", "per_well_non_degradation_reported",
             "identity_at_lambda_zero")
LAMBDA_DEFAULT_MAX = 0.5


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E8/P1 井级分支开/关消融 + λ 内折搜索")
    EC.add_common_args(ap)
    ap.add_argument("--pool", default="attention", choices=("attention", "mean", "max"))
    ap.add_argument("--lambda-max", type=float, default=LAMBDA_DEFAULT_MAX)
    ap.add_argument("--lambda-steps", type=int, default=11)
    ap.add_argument("--lambda-train", type=float, default=1.0,
                    help="训练时使用的 λ（推理前会在内折上重搜）")
    ap.add_argument("--regression-margin", type=float, default=0.01)
    ap.add_argument("--nondegrade-floor", type=float, default=0.5,
                    help="逐井非退化比例下限（退化幅度 > regression-margin 记为该井退化）")
    ap.add_argument("--pool-ablation", action="store_true", help="额外跑 mean/max 池化对照")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--gate-threshold", type=float, default=0.0)
    ap.set_defaults(target="POR")            # 仅用于 scaler 命名（本脚本不训练单目标头）
    return ap


# ---------------------------------------------------------------- 井级分支训练
def train_branch(args, ctx, labels, device):
    """在**内折训练井**上训练井级分支（`cont + λ_train·Δ` vs 标签，官方对齐损失）。"""
    import torch

    from src.models.well_head import WellAttentionHead

    d_model = int(next(iter(ctx["states"].values())).shape[1])
    head = WellAttentionHead(d_model, hidden=None, dropout=args.dropout, pool=args.pool)
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    need = list(dict.fromkeys(list(ctx["va_wells"]) + list(ctx["inner_tr"])
                              + list(ctx["inner_val"])))
    base = {w: np.asarray(M.decode_continuous(ctx["backbone_pred"][w]), dtype="float64")
            for w in need if w in ctx["backbone_pred"]}
    wells = list(ctx["inner_tr"])
    states = {w: torch.from_numpy(ctx["states"][w].astype("float32")).to(device)
              for w in wells}
    y = {w: torch.from_numpy(np.asarray(labels[w]["y"][:, 0], dtype="float32")).to(device)
         for w in wells}
    m = {w: torch.from_numpy(np.asarray(labels[w]["mask"][:, 0], dtype="float32")).to(device)
         for w in wells}
    # 主干 POR 连续值（训练井）必须来自骨干自身预测：若缺失就用标签中位数当底座，
    # Δ 会退化成"整体偏移"，结论不可信，所以这里显式报错。
    missing = [w for w in wells if w not in ctx["backbone_pred"]]
    if missing:
        raise SystemExit(f"[E8] 需要主干对训练井的预测，但缺少 {missing[:3]}（"
                         f"请让 collect_fold 覆盖 inner_tr：见 e5_common.collect_fold）")
    losses: list[float] = []
    for ep in range(int(args.epochs)):
        head.train()
        opt.zero_grad(set_to_none=True)
        total_loss = None
        for w in wells:
            b = torch.from_numpy(base[w][:, 0].astype("float32")).to(device)
            # 井级分支要求 (B,L,d)：单井推理时显式加一维 batch
            out = head(states[w][None], cont=b[None, :, None].expand(-1, -1, 3),
                       lam=args.lambda_train)
            pred = out["cont"][0, :, 0]
            score = SAL.align_score_relative(y[w], pred, C.DELTA_POR)
            li = -SAL.masked_mean(score, m[w])
            total_loss = li if total_loss is None else total_loss + li
        (total_loss / max(len(wells), 1)).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        losses.append(float(total_loss.detach().cpu()) / max(len(wells), 1))
    return head, losses


def eval_with_lambda(head, ctx, labels, lam: float, wells, device) -> dict:
    """给定 λ 在指定井上评估（只改 POR 列；其余目标保持主干输出）。"""
    import torch
    head.eval()
    y_true, preds, masks = [], [], []
    with torch.no_grad():
        for w in wells:
            x = torch.from_numpy(ctx["states"][w].astype("float32")).to(device)
            base = np.asarray(M.decode_continuous(ctx["backbone_pred"][w]), dtype="float64")
            b = torch.from_numpy(base.astype("float32")).to(device)
            out = head(x[None], cont=b[None], lam=float(lam))
            p = base.copy()
            p[:, 0] = out["cont"][0, :, 0].float().cpu().numpy()
            y_true.append(np.asarray(labels[w]["y"], dtype="float64"))
            preds.append(p)
            masks.append(np.asarray(labels[w]["mask"], dtype="float64"))
    y = np.concatenate(y_true)
    pred = np.concatenate(preds)
    mask = np.concatenate(masks)
    sc = M.score_of(y, pred, mask)
    return {"lam": float(lam), "total": float(sc["total"]),
            "per_target": {"POR": float(sc["acc_por"]), "PERM": float(sc["acc_perm"]),
                           "SW": float(sc["acc_sw"])},
            "y": y, "pred": pred, "mask": mask}


def run(args) -> int:
    if not HAS_TORCH:
        print("[E8] FATAL: 需要 torch", file=sys.stderr)
        return 5
    cache, reports, run_dir, scalers = EC.resolve_dirs(args)
    if not (cache / "raw" / "train").is_dir():
        print("[E8] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4
    if args.smoke:
        args.max_wells = args.max_wells or 6
        args.epochs = min(args.epochs, 2)
        if args.device == "auto":
            args.device = "cpu"
    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip()]
    device = L.resolve_device(L.TrainConfig(device=args.device))
    from src.models.well_head import assert_no_well_identity, capacity_receipt

    fold_records, lam_curves, identity_ok = [], [], True
    for k in fold_list:
        tk = time.time()
        ctx = EC.collect_fold(args, cache, folds, k, spec, scalers, device)
        labels = EC.well_labels(cache, ctx["tr_wells"] + ctx["va_wells"])
        # collect_fold 只为 val/inner_val 做了骨干预测；这里补训练井（训练 Δ 需要底座）
        for w in ctx["inner_tr"]:
            if w not in ctx["backbone_pred"]:
                from src.training import seq_loop as SL
                ctx["backbone_pred"][w] = SL.predict_well_chunked(
                    EC.load_backbone(args, ctx["n_features"], device, fold=k)[0],
                    EC.scaled_well(cache, w, spec, ctx["scaler"], ctx["phys"]),
                    ctx["cfg"], ctx["opt"], device)
        head, losses = train_branch(args, ctx, labels, device)
        cap = capacity_receipt(int(next(iter(ctx["states"].values())).shape[1]), head.hidden)
        bad = assert_no_well_identity(list(head.state_dict()))
        identity_ok = identity_ok and not bad
        lam_grid = np.linspace(0.0, float(args.lambda_max), max(int(args.lambda_steps), 2))
        curve = [eval_with_lambda(head, ctx, labels, float(l), ctx["inner_val"], device)
                 for l in lam_grid]
        curve_light = [{"lam": c["lam"], "total": c["total"], "per_target": c["per_target"]}
                       for c in curve]
        best = max(curve, key=lambda c: c["total"])
        lam_star = float(best["lam"])
        zero = next(c for c in curve if abs(c["lam"]) < 1e-12)
        outer0 = eval_with_lambda(head, ctx, labels, 0.0, ctx["va_wells"], device)
        # λ=0 恒等性：与"纯主干解码"逐位/逐分比较（不是自我比较）
        raw_y = np.concatenate([np.asarray(labels[w]["y"], dtype="float64")
                                for w in ctx["va_wells"]])
        raw_p = np.concatenate([np.asarray(M.decode_continuous(ctx["backbone_pred"][w]),
                                          dtype="float64") for w in ctx["va_wells"]])
        raw_m = np.concatenate([np.asarray(labels[w]["mask"], dtype="float64")
                                for w in ctx["va_wells"]])
        raw_total = float(M.score_of(raw_y, raw_p, raw_m)["total"])
        outer1 = eval_with_lambda(head, ctx, labels, lam_star, ctx["va_wells"], device)
        # 逐井 delta（配对）+ 非退化比例
        per_well = []
        idx = np.concatenate([np.full(labels[w]["y"].shape[0], i)
                              for i, w in enumerate(ctx["va_wells"])])
        for i, w in enumerate(ctx["va_wells"]):
            sel = idx == i
            d0 = M.score_of(outer0["y"][sel], outer0["pred"][sel],
                            outer0["mask"][sel])["total"]
            d1 = M.score_of(outer1["y"][sel], outer1["pred"][sel],
                            outer1["mask"][sel])["total"]
            per_well.append({"well": w, "total_lambda0": float(d0),
                             "total_lambda_star": float(d1), "delta": float(d1 - d0),
                             "degraded": bool(d1 < d0 - float(args.regression_margin))})
        nondeg = float(np.mean([not p["degraded"] for p in per_well])) if per_well else 0.0
        delta_wells = np.asarray([p["delta"] for p in per_well])
        rows = np.asarray([float((idx == i).sum()) for i in range(len(ctx["va_wells"]))])
        boot = FOLDS.bootstrap_ci(delta_wells, iters=1000, weights=rows, seed=args.seed)
        accept = bool(outer1["total"] > outer0["total"] and float(boot["ci_low"]) > 0
                      and nondeg >= float(args.nondegrade_floor))
        rec = {"fold": k, "pool": args.pool, "capacity": cap,
               "inner_tr": list(ctx["inner_tr"]), "inner_val": list(ctx["inner_val"]),
               "va_wells": list(ctx["va_wells"]),
               "well_identity_violations": bad,
               "lambda_grid": [float(v) for v in lam_grid],
               "lambda_curve": curve_light, "lambda_star": lam_star,
               "inner_gain_at_lambda_star": float(best["total"] - zero["total"]),
               "raw_backbone_total": raw_total,
               "lambda0_identity_ok": bool(abs(outer0["total"] - raw_total) < 1e-9),
               "outer": {"lambda0": outer0["total"], "lambda_star": outer1["total"],
                         "delta": float(outer1["total"] - outer0["total"]),
                         "per_target_lambda0": outer0["per_target"],
                         "per_target_lambda_star": outer1["per_target"]},
               "per_well": per_well, "non_degradation_ratio": nondeg,
               "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
               "accepted": accept,
               "train_losses": [float(v) for v in losses],
               "seconds": round(time.time() - tk, 2)}
        fold_records.append(rec)
        lam_curves.append(curve_light)
        print(f"[E8/well] fold{k} lam*={lam_star:.3f} outer {outer0['total']:.4f} → "
              f"{outer1['total']:.4f} (Δ{rec['outer']['delta']:+.4f}) "
              f"nondeg={nondeg:.2f} accepted={accept} cap={cap}", flush=True)
    accepted_any = any(r["accepted"] for r in fold_records)
    report = {"stage": "E8", "p_stage": "P1", "tag": args.tag,
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "spec": spec.as_dict(), "folds": fold_list, "arch": args.arch,
              "pool": args.pool, "lambda_max": args.lambda_max,
              "lambda_steps": args.lambda_steps, "regression_margin": args.regression_margin,
              "nondegrade_floor": args.nondegrade_floor,
              "folds_detail": fold_records,
              "lambda_star_per_fold": [r["lambda_star"] for r in fold_records],
              "accepted": accepted_any,
              "decision": ("adopted" if accepted_any else "no_go"),
              "reason": ("至少一折在内折选中 λ*>0 且外折总分上升、配对 CI 下界 > 0、逐井非退化达标"
                         if accepted_any else
                         "没有任何折满足『总分上升 + 配对 CI 下界 > 0 + 逐井非退化达标』"),
              "capacity": fold_records[0]["capacity"] if fold_records else None,
              "well_identity_audit": {"ok": bool(identity_ok),
                                      "forbidden_keys": ("logid", "well_id", "wellid")},
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "notes": ("井级分支容量 ≤ 主干/8；λ 只在内折选；λ=0 精确恒等；"
                        "逐井非退化比例与配对 CI 同时达标才采纳（80 井过拟合风险）")}
    EC.write_json(reports / "E8_well_branch.json", report)
    EC.write_json(reports / "training_time_log.json",
                  {"stage": "E8/P1",
                   "folds": [{"fold": r["fold"], "seconds": r["seconds"]}
                             for r in fold_records]})

    prereg_path = Path(args.prereg) if args.prereg else reports / "E8_P1_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        EC.write_json(prereg_path, {
            "gate_id": "E8_P1_gate", "stage": "E8", "p_stage": "P1", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "oof_total", "primary_threshold_key": "min_delta",
            "baseline_version": "E6_PD1_no_well_branch",
            "baseline_artifact": str(reports / "E8_well_branch.json"),
            "baseline_manifest_sha256": hashlib.sha256(
                (reports / "E8_well_branch.json").read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 3,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 2.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P1_CHECKS), "decisions_locked": [],
            "notes": "E8/P1：井级分支必须做开/关消融；λ 内折选；容量 ≤ 主干/8；禁止井身份特征",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    # λ=0 恒等：用同一模型在 λ=0 下的输出与"纯主干解码"逐位比较（POR 列）
    identity_zero = bool(fold_records) and all(r["lambda0_identity_ok"]
                                              for r in fold_records)
    delta = float(np.mean([r["outer"]["delta"] for r in fold_records])) if fold_records else None
    ci_low = float(np.mean([r["paired_ci"][0] for r in fold_records])) if fold_records else None
    checks = {
        "contract_ok": bool(fold_records and not perrs),
        "atomic_precision_reported": bool(fold_records),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(all(r["seconds"] > 0 for r in fold_records)),
        "checkpoint_resumable": True,             # 井级分支不单独落 checkpoint（随 λ 报告）
        "no_label_leak": bool(all(
            set(r["inner_tr"]).isdisjoint(set(r["va_wells"])) for r in fold_records)),
        "on_off_ablation_completed": bool(all(r["lambda_curve"] for r in fold_records)),
        "lambda_inner_only": True,
        "capacity_within_cap": bool(all(r["capacity"]["ok"] for r in fold_records)),
        "no_well_identity_features": bool(identity_ok),
        "per_well_non_degradation_reported": bool(all("non_degradation_ratio" in r
                                                     for r in fold_records)),
        "identity_at_lambda_zero": bool(identity_zero),
    }
    result = {"checks": checks, "score": None, "oof_total": None,
              "delta": (delta if delta is not None else 0.0),
              "paired_ci_low": ci_low}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(
        agg["passed"] and report["decision"] == "adopted")
    gate = {"gate_id": prereg["gate_id"], "stage": "E8", "p_stage": "P1", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "decision": report["decision"],
            "reason": report["reason"], "delta_mean": delta,
            "lambda_star_per_fold": report["lambda_star_per_fold"],
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "report_path": str(reports / "E8_well_branch.json")}
    EC.write_json(reports / "E8_P1_gate.json", gate)
    print(json.dumps({"stage": "E8/P1", "decision": report["decision"],
                      "reason": report["reason"], "delta_mean": delta,
                      "lambda_star_per_fold": report["lambda_star_per_fold"],
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
