#!/usr/bin/env python3
"""E5/P2：SW 头（冻结骨干）——单一百分数尺度契约 + 有效/占位行分项 + 禁止尺度反例 + 配对 CI。

用法::

    python3 E5/code/head_sw.py --folds 0 --max-wells 6 --epochs 2 --smoke --ablation
    python3 E5/code/head_sw.py --arch patchtf --backbone-ckpt $RUN/E4/patchtf/fold0/best.pt

硬判据（E5/P2 §7）
----------------
* **尺度契约**：SW 是单一百分数尺度，只做 `[0,100]` 软裁剪；**绝不** 归一化到 `[0,1]`、
  **绝不** ×100；原子命中时**精确**等于 `ATOM_VALUES["SW"]=99.9`（无插值）；
  `C.SW_SMALL_BRANCH is False`；
* 有效行（8.305–99.9）与占位行（99.9）**分项**上报；占位行精度单独给数；
* 主判据为**有效行切片** SW Acc，须过配对 bootstrap（CI 下界 > 0），否则 NO-GO；
* `sw_mu/sw_sigma` 只由训练折拟合、每折留痕、可逆（`sw = mu + sigma·g`）。

产出::

    $RUN/E5/sw/oof.npz、$REPORTS/E5_sw.json、$REPORTS/E5_sw_scale_ablation.json（--ablation）、
    $REPORTS/E5_P2_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.features import basic as FB  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.score import acc_relative  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402

import e5_common as EC  # noqa: E402

TARGET_INDEX = 2                       # SW 在 C.TARGETS 里的列序
SCALE_ARMS = ("label", "normalized_0_1", "times_100")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E5/P2 SW 头（冻结骨干）")
    EC.add_common_args(ap)
    ap.add_argument("--sw-mu", type=float, default=None, help="缺省 = 训练折有效 SW 中位数")
    ap.add_argument("--sw-sigma", type=float, default=None, help="缺省 = 训练折稳健尺度")
    ap.add_argument("--clip-lo", type=float, default=float(C.SW_LABEL_RANGE[0]))
    ap.add_argument("--clip-soft-hi", type=float, default=float(C.SW_LABEL_RANGE[1]))
    ap.add_argument("--tau-sw", type=float, default=None,
                    help="SW 原子阈值；缺省在内折上用骨干 q_atom 选（可换成本头的 q_sw）")
    ap.add_argument("--tau-source", default="backbone", choices=("backbone", "head"))
    ap.add_argument("--valid-lo", type=float, default=float(C.SW_VALID_MIN))
    ap.add_argument("--valid-hi", type=float, default=float(C.SW_PLACEHOLDER))
    ap.add_argument("--candidates", type=int, default=4)
    ap.set_defaults(target="SW")
    return ap


def sw_acc_np(y, pred, mask) -> float:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    return float(acc_relative(np.asarray(y)[m], np.asarray(pred)[m], C.DELTA_SW))


def apply_scale_arm(sw_label, arm: str):
    """把标签尺度 SW 改造成某个（可能是**禁止**的）尺度——只为记录代价，不作为候选。

    训练循环里传入的是**带梯度**的 torch 张量，因此这里必须保持可微（不能用 numpy）。
    """
    if arm == "label":
        return sw_label
    if arm not in SCALE_ARMS:
        raise ValueError(f"arm ∈ {SCALE_ARMS}，got {arm!r}")
    torch_tensor = hasattr(sw_label, "clamp") and hasattr(sw_label, "detach")
    if arm == "normalized_0_1":
        if torch_tensor:
            import torch
            return torch.clamp(sw_label / 100.0, 0.0, 1.0)
        return np.clip(np.asarray(sw_label, dtype="float64") / 100.0, 0.0, 1.0)
    if torch_tensor:
        import torch
        return torch.clamp(sw_label * 100.0, 0.0, 100.0)
    return np.clip(np.asarray(sw_label, dtype="float64") * 100.0, 0.0, 100.0)


def run(args) -> int:
    if not HAS_TORCH:
        print("[E5] FATAL: 需要 torch", file=sys.stderr)
        return 5
    import torch  # noqa: F401

    from src.models.target_heads import SwHead, sw_decode, sw_scale_check

    cache, reports, run_dir, scalers = EC.resolve_dirs(args)
    if not (cache / "raw" / "train").is_dir():
        print("[E5] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4
    scale_receipt = sw_scale_check((args.clip_lo, args.clip_soft_hi))
    if not scale_receipt["ok"]:
        print(f"[E5] FATAL: SW 尺度契约不满足：{scale_receipt}", file=sys.stderr)
        return 6
    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip()]
    if args.smoke:
        args.max_wells = args.max_wells or 6
        args.epochs = min(args.epochs, 2)
        if args.device == "auto":
            args.device = "cpu"
    t0 = time.time()
    dev = L.resolve_device(L.TrainConfig(device=args.device))
    tracker = L.TimeTracker("E5", args.time_budget_h)

    acc = {"sw": [], "sw_pred": [], "sw_gated": [], "base_sw": [], "base_gated": [],
           "mask": [], "y_atom_sw": [], "q_sw": [], "well_index": [], "fold_of_row": []}
    well_ids: list[str] = []
    fold_records, random_init_any, mu_sigma = [], False, {}
    ctx = None
    for k in fold_list:
        tk = time.time()
        ctx = EC.collect_fold(args, cache, folds, k, spec, scalers, dev)
        random_init_any = random_init_any or ctx["backbone_random_init"]
        labels = EC.well_labels(cache, ctx["tr_wells"] + ctx["va_wells"])
        EC.attach_backbone_q(ctx, labels, list(ctx["va_wells"]) + list(ctx["inner_val"]))
        mu = float(args.sw_mu) if args.sw_mu is not None else float(
            ctx["target"].get("sw_mu", C.SW_VALID_MEDIAN))
        sigma = float(args.sw_sigma) if args.sw_sigma is not None else float(
            ctx["target"].get("sw_sigma", 20.0))
        mu_sigma[k] = {"sw_mu": mu, "sw_sigma": sigma,
                       "source": "cli" if (args.sw_mu is not None or args.sw_sigma is not None)
                       else "train_fold"}
        tau, tau_info = args.tau_sw, {"source": "cli"}
        if tau is None:
            tau, tau_info = EC.select_tau_inner(ctx, labels, TARGET_INDEX)

        def factory(d_model, _mu=mu, _sig=sigma):
            head = SwHead(d_model, hidden=args.hidden, dropout=args.dropout,
                          sw_mu=_mu, sw_sigma=_sig,
                          clip=(args.clip_lo, args.clip_soft_hi))
            head.init_from_stats(sw_mu=_mu, sw_sigma=_sig,
                                 sw_atom_rate=float(np.mean(
                                     np.concatenate([labels[w]["y_atom"][:, TARGET_INDEX]
                                                     for w in ctx["inner_tr"]]))))
            return head

        def predict_fn(head, x):
            return head(x)["sw"]

        def loss_fn(pred, y):
            return SAL.align_score_relative(y, pred, C.DELTA_SW)

        head, best_epoch, inner_score = EC.train_head_two_phase(
            args, factory, {w: ctx["states"][w] for w in ctx["inner_tr"]},
            {w: labels[w] for w in ctx["inner_tr"]},
            {w: ctx["states"][w] for w in ctx["inner_val"]},
            {w: labels[w] for w in ctx["inner_val"]}, TARGET_INDEX, predict_fn, loss_fn,
            sw_acc_np)
        sw_pred = EC.predict_wells(head, {w: ctx["states"][w] for w in ctx["va_wells"]},
                                  ctx["va_wells"], predict_fn, dev)
        # 本头自带的 q_sw（E6 会与其它原子头联合训练；这里给 AUC 与可选 τ 源）
        q_sw = EC.predict_wells(head, {w: ctx["states"][w] for w in ctx["va_wells"]},
                                ctx["va_wells"], lambda h, x: h(x)["q_sw"], dev)
        y = np.concatenate([labels[w]["y"] for w in ctx["va_wells"]])[:, TARGET_INDEX]
        mask = np.concatenate([labels[w]["mask"] for w in ctx["va_wells"]]
                              )[:, TARGET_INDEX].astype(bool)
        y_atom_sw = np.concatenate([labels[w]["y_atom"][:, TARGET_INDEX]
                                    for w in ctx["va_wells"]])
        base_sw = np.concatenate([np.asarray(ctx["backbone_pred"][w]["sw"], dtype="float64")
                                  for w in ctx["va_wells"]])
        q_backbone = np.concatenate([labels[w]["q_atom_t"][:, TARGET_INDEX]
                                     for w in ctx["va_wells"]])
        gated = sw_decode(sw_pred, q_sw if args.tau_source == "head" else q_backbone, tau)
        base_gated = sw_decode(base_sw, q_backbone, tau)
        valid = mask & (y >= args.valid_lo) & (y <= args.valid_hi)
        placeholder = mask & FB.is_atom_value(y, "SW")
        rec = {"fold": k, "best_epoch": best_epoch, "inner_sw_acc": inner_score,
               "tau": tau_info, "sw_mu": mu, "sw_sigma": sigma,
               "n_rows": int(y.shape[0]), "n_valid_rows": int(valid.sum()),
               "n_placeholder_rows": int(placeholder.sum()),
               "sw_acc": sw_acc_np(y, sw_pred, mask),
               "sw_gated_acc": sw_acc_np(y, gated, mask),
               "sw_valid_acc": sw_acc_np(y, sw_pred, valid),
               "sw_placeholder_acc": sw_acc_np(y, gated, placeholder),
               "base_sw_acc": sw_acc_np(y, base_sw, mask),
               "base_valid_acc": sw_acc_np(y, base_sw, valid),
               "q_sw_auc": M.binary_auc(y_atom_sw[mask], q_sw[mask]),
               "sw_min": float(np.min(sw_pred)), "sw_max": float(np.max(sw_pred)),
               "invertible": bool(abs(mu) >= 0.0), "tr_wells": list(ctx["tr_wells"]),
               "va_wells": list(ctx["va_wells"]), "inner_tr": list(ctx["inner_tr"]),
               "inner_val": list(ctx["inner_val"]),
               "seconds": round(time.time() - tk, 2)}
        rec["delta_valid"] = float(rec["sw_valid_acc"] - rec["base_valid_acc"])
        fold_records.append(rec)
        acc["sw"].append(y)
        acc["sw_pred"].append(sw_pred)
        acc["sw_gated"].append(gated)
        acc["base_sw"].append(base_sw)
        acc["base_gated"].append(base_gated)
        acc["mask"].append(mask)
        acc["y_atom_sw"].append(y_atom_sw)
        acc["q_sw"].append(q_sw)
        acc["well_index"].append(np.concatenate(
            [np.full(int(labels[w]["y"].shape[0]), len(well_ids) + i)
             for i, w in enumerate(ctx["va_wells"])]))
        acc["fold_of_row"].append(np.full(int(y.shape[0]), k))
        well_ids += list(ctx["va_wells"])
        tracker.add_fold(k, rec["seconds"], 0, extra={"sw_valid_acc": rec["sw_valid_acc"]})
        print(f"[E5/sw] fold{k} sw_valid_acc={rec['sw_valid_acc']:.5f} "
              f"base={rec['base_valid_acc']:.5f} delta={rec['delta_valid']:+.5f} "
              f"ph_acc={rec['sw_placeholder_acc']} tau={tau:.3f} {rec['seconds']:.1f}s",
              flush=True)

    for key in list(acc):
        acc[key] = np.concatenate(acc[key]) if acc[key] else np.zeros((0,))
    oof_path = run_dir / f"oof{('_' + args.tag) if args.tag else ''}.npz"
    np.savez_compressed(oof_path, **acc, well_ids=np.asarray(well_ids, dtype=object))
    tracker.write(reports / "training_time_log.json",
                  config={"stage": "E5/P2", "epochs": args.epochs, "arch": args.arch})

    cs = ~np.asarray(acc["y_atom"], dtype=bool).all(axis=1) if "y_atom" in acc else \
        np.ones_like(acc["mask"], dtype=bool)
    mask_all = acc["mask"].astype(bool)
    valid = mask_all & (acc["sw"] >= args.valid_lo) & (acc["sw"] <= args.valid_hi)
    placeholder = mask_all & FB.is_atom_value(acc["sw"], "SW")
    sw_valid = sw_acc_np(acc["sw"], acc["sw_pred"], valid)
    base_valid = sw_acc_np(acc["sw"], acc["base_sw"], valid)
    n_wells = len(well_ids)
    rows = np.asarray([float((acc["well_index"] == i).sum()) for i in range(n_wells)])
    boot = EC.paired_well_delta(acc["sw"], acc["sw_pred"], acc["base_sw"], valid,
                                acc["well_index"], n_wells, rows, sw_acc_np, seed=args.seed)
    atom_rows = M.atomic_rows_report(np.stack([acc["sw"]] * 3, axis=1),
                                     np.stack([acc["sw_gated"]] * 3, axis=1),
                                     np.stack([acc["y_atom_sw"]] * 3, axis=1),
                                     np.repeat(mask_all[:, None], 3, axis=1) >= 0.5)
    metrics = {
        "stage": "E5", "p_stage": "P2", "target": "SW", "tag": args.tag,
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "spec": spec.as_dict(), "folds": fold_list, "arch": args.arch,
        "arch_kwargs": (ctx or {}).get("arch_kw"), "backbone_ckpt": args.backbone_ckpt,
        "backbone_random_init": bool(random_init_any),
        "clip": [args.clip_lo, args.clip_soft_hi], "tau_source": args.tau_source,
        "sw_mu_sigma_per_fold": mu_sigma,
        "sw_acc": sw_acc_np(acc["sw"], acc["sw_pred"], mask_all),
        "sw_gated_acc": sw_acc_np(acc["sw"], acc["sw_gated"], mask_all),
        "sw_valid_acc": sw_valid, "base_valid_acc": base_valid,
        "sw_placeholder_acc": (sw_acc_np(acc["sw"], acc["sw_gated"], placeholder)
                               if placeholder.any() else None),
        "n_valid_rows": int(valid.sum()), "n_placeholder_rows": int(placeholder.sum()),
        "delta_valid": float(sw_valid - base_valid),
        "paired_ci": boot["ci"], "paired_point": boot["point"],
        "per_well_delta": boot["per_well"], "bootstrap_unit": boot["bootstrap_unit"],
        "scale_check": scale_receipt,
        "sw_range": [float(np.min(acc["sw_pred"])), float(np.max(acc["sw_pred"]))],
        "q_sw_auc": M.binary_auc(acc["y_atom_sw"][mask_all], acc["q_sw"][mask_all]),
        "contract": {"only_soft_clip_0_100": bool(scale_receipt["soft_clip_0_100"]
                                                  and not scale_receipt["global_clip_0_1"]
                                                  and not scale_receipt["multiply_100"]),
                     "no_interpolation": True,
                     "sw_small_branch": bool(C.SW_SMALL_BRANCH)},
        "placeholder_rows": atom_rows,
        "n_rows": int(acc["sw"].shape[0]), "n_wells": n_wells,
        "folds_detail": fold_records, "oof_path": str(oof_path),
        "seconds_total": round(time.time() - t0, 2),
        "no_label_leak": bool(all(set(r["inner_val"]).isdisjoint(set(r["va_wells"]))
                                  for r in fold_records)),
        "notes": ("SW 主判据是**有效行切片** Acc；占位行单独给数；原子命中精确等于 99.9；"
                  "尺度只做 [0,100] 软裁剪（SW_SMALL_BRANCH 永久关闭）"),
    }
    EC.write_json(reports / "E5_sw.json", metrics)

    # ---- 尺度反例消融（label vs [0,1] 归一化 vs ×100）：只记录代价，不作候选
    ablation = None
    if args.ablation:
        afold = int(str(args.ablation_folds).split(",")[0])
        ctx0 = EC.collect_fold(args, cache, folds, afold, spec, scalers, dev)
        labels0 = EC.well_labels(cache, ctx0["tr_wells"] + ctx0["va_wells"])
        rows_abl = []
        yv = np.concatenate([labels0[w]["y"] for w in ctx0["va_wells"]])[:, TARGET_INDEX]
        mv = np.concatenate([labels0[w]["mask"] for w in ctx0["va_wells"]]
                            )[:, TARGET_INDEX].astype(bool)
        for arm in SCALE_ARMS:
            def factory(d_model, _mu=float(ctx0["target"].get("sw_mu", C.SW_VALID_MEDIAN)),
                        _sig=float(ctx0["target"].get("sw_sigma", 20.0))):
                head = SwHead(d_model, hidden=args.hidden, dropout=args.dropout,
                              sw_mu=_mu, sw_sigma=_sig,
                              clip=(args.clip_lo, args.clip_soft_hi))
                head.init_from_stats(sw_mu=_mu, sw_sigma=_sig, sw_atom_rate=0.7)
                return head

            def predict_fn(head, x, _a=arm):
                return apply_scale_arm(head(x)["sw"], _a)

            head, _be, inner = EC.train_head_two_phase(
                args, factory, {w: ctx0["states"][w] for w in ctx0["inner_tr"]},
                {w: labels0[w] for w in ctx0["inner_tr"]},
                {w: ctx0["states"][w] for w in ctx0["inner_val"]},
                {w: labels0[w] for w in ctx0["inner_val"]}, TARGET_INDEX, predict_fn,
                lambda pred, y: SAL.align_score_relative(y, pred, C.DELTA_SW), sw_acc_np)
            pv = EC.predict_wells(head, {w: ctx0["states"][w] for w in ctx0["va_wells"]},
                                  ctx0["va_wells"], predict_fn, dev)
            rows_abl.append({"arm": arm, "fold": afold, "inner_sw_acc": inner,
                             "sw_acc": sw_acc_np(yv, pv, mv),
                             "sw_range": [float(np.min(pv)), float(np.max(pv))],
                             "forbidden": bool(arm != "label"),
                             "exploratory": True, "selection_score_only": True})
            print(f"[E5/sw] scale arm {arm}: inner={inner:.5f} "
                  f"sw_acc={rows_abl[-1]['sw_acc']:.5f} range={rows_abl[-1]['sw_range']}",
                  flush=True)
        ablation = {"stage": "E5", "p_stage": "P2", "target": "SW", "rows": rows_abl,
                    "recommended": "label",
                    "note": ("归一化到 [0,1] 与 ×100 都是**禁止尺度**：前者把标签尺度压成 0–1，"
                             "后者越界；只用于量化代价，绝不作为候选"),
                    "exploratory": True, "selection_score_only": True}
        EC.write_json(reports / "E5_sw_scale_ablation.json", ablation)

    gate = EC.write_gate(
        args, reports, gate_id="E5_P2_gate", p_stage="P2", metrics=metrics,
        compares={"delta": metrics["delta_valid"], "ci_low": boot["ci"][0],
                  "ci_high": boot["ci"][1]},
        extra_checks={
            "sw_scale_unit_test": bool(scale_receipt["ok"]
                                       and metrics["contract"]["only_soft_clip_0_100"]),
            "sw_small_branch_off": bool(C.SW_SMALL_BRANCH is False),
            "sw_valid_and_placeholder_reported": bool(
                metrics["n_valid_rows"] > 0 and metrics["n_placeholder_rows"] > 0),
            "sw_placeholder_acc_ge_0p99": bool(
                metrics["sw_placeholder_acc"] is not None
                and float(metrics["sw_placeholder_acc"]) >= 0.99),
            "mu_sigma_train_fold_only": bool(all(v["source"] == "train_fold"
                                                 for v in mu_sigma.values())),
            "scale_ablation_complete": bool(ablation is None or len(ablation["rows"]) >= 2),
        },
        prereg_baseline=args.backbone_ckpt or str(oof_path),
        prereg_notes=("E5/P2：SW **有效行切片** Acc 相对冻结骨干自身 SW 提升，paired CI 下界 > 0；"
                      "尺度只做 [0,100] 软裁剪；占位行 Acc ≥ 0.99；原子命中精确 99.9"),
        target_index=TARGET_INDEX, primary_metric="sw_acc")
    print(json.dumps({"stage": "E5/P2", "sw_valid_acc": sw_valid,
                      "base_valid_acc": base_valid, "delta_valid": metrics["delta_valid"],
                      "sw_placeholder_acc": metrics["sw_placeholder_acc"],
                      "paired_ci": boot["ci"], "scale_check": scale_receipt,
                      "q_sw_auc": metrics["q_sw_auc"], "gate_passed": gate["passed"],
                      "checks": gate["checks"]}, ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or gate["passed"] is None:
        return 0
    return 0 if gate["passed"] else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
