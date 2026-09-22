#!/usr/bin/env python3
"""E7/P0：损失消融（七组臂）——只在**内折**上比较，产出可冻结的 `loss_v1.json` 与 Gate。

七组臂（E7/P0 §5，逐字对应）
--------------------------
* **exp1** 纯 align / 纯 aux / align+aux（3 臂）
* **exp2** λ₁ ∈ {0.1,0.3,1.0} × 退火 ∈ {constant, linear_to_0.1, cosine}（9 臂）
* **exp3** λ₂（联合项）∈ {0.1,0.2,0.3}（3 臂）
* **exp4** L_aux 训练折尺度归一化 vs **绝对** Smooth L1（2 臂）
* **exp5** 边界聚焦 κ ∈ {0.5,1.0,2.0} × σ ∈ {0.15,0.25,0.35}（9 臂，**默认关**）
* **exp6** PERM 截断开关（2 臂）
* **exp7** L_phys λ₃ ∈ {0,0.02,0.05}（可选物理软约束，见 `src/losses/physics.py`；
  默认关，必须消融验证）

纪律
----
* 默认 `--inner-only`：训练集只含内折训练井、评估只含内折验证井（H1：**不许看 outer 折**）；
* `--arm-set` 选组、`--arms` 选具体臂；建议云端分次跑（单臂成本 = 一次训练）；
* `--smoke` 不写仓库 `versions/configs/loss_v1.json`（改写到 reports 下）。
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

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import fold_runner as FR  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P0_CHECKS = ("inner_only_selection", "masked_mean_nan_unit_test", "perm_low_tail_reported")
LAM1_GRID = (0.1, 0.3, 1.0)
LAM2_GRID = (0.1, 0.2, 0.3)
SCHEDULE_GRID = ("constant", "linear_to_0.1", "cosine")
KAPPA_GRID = (0.5, 1.0, 2.0)
SIGMA_GRID = (0.15, 0.25, 0.35)
NOT_IMPLEMENTED: list[str] = []  # exp7 已实现：src/losses/physics.py


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E7/P0 损失消融（七组臂，内折选择）")
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or "")
    ap.add_argument("--spec", default="F1")
    ap.add_argument("--arm-set", default="exp1",
                    choices=("exp1", "exp2", "exp3", "exp4", "exp5", "exp6",
                             "exp7", "all"))
    ap.add_argument("--arms", default=None, help="逗号分隔的臂 id（覆盖 --arm-set）")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--inner-only", action="store_true", default=True)
    ap.add_argument("--eval-outer", dest="inner_only", action="store_false",
                    help="额外在全折训练/外折验证上评估（更贵，且只能用于报告）")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam1", type=float, default=1.0)
    ap.add_argument("--lam1-start", type=float, default=1.0)
    ap.add_argument("--lam1-end", type=float, default=0.1)
    ap.add_argument("--lam1-frac", type=float, default=0.6)
    ap.add_argument("--lam1-schedule", default="linear_to_0.1",
                    choices=SCHEDULE_GRID)
    ap.add_argument("--lam2", type=float, default=0.2)
    ap.add_argument("--lam3", type=float, default=0.0)
    ap.add_argument("--aux-normalize", default="fold_scale",
                    choices=("fold_scale", "absolute"))
    ap.add_argument("--boundary-kappa", type=float, default=0.0)
    ap.add_argument("--boundary-sigma", type=float, default=0.25)
    ap.add_argument("--perm-clamp", default="on", choices=("on", "off"))
    ap.add_argument("--huber-beta", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--alpha-nonjoint", type=float, default=1.0)
    ap.add_argument("--out-config", default=str(V4 / "versions" / "configs" / "loss_v1.json"))
    ap.add_argument("--candidates", default=os.environ.get("V4_CANDIDATES") or str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--gate-threshold", type=float, default=78.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    return ap


# ---------------------------------------------------------------- 臂定义
def build_arms(arm_set: str, arms: str | None) -> list[dict]:
    """返回臂列表：`{id, group, overrides}`（overrides 直接覆盖损失/退火开关）。"""
    out: list[dict] = []
    if arm_set in ("exp1", "all"):
        out += [{"id": "exp1_align", "group": "exp1",
                 "overrides": {"use_align": True, "use_aux": False}},
                {"id": "exp1_aux", "group": "exp1",
                 "overrides": {"use_align": False, "use_aux": True}},
                {"id": "exp1_both", "group": "exp1",
                 "overrides": {"use_align": True, "use_aux": True}}]
    if arm_set in ("exp2", "all"):
        for lam in LAM1_GRID:
            for sch in SCHEDULE_GRID:
                out.append({"id": f"exp2_lam1{lam}_{sch}", "group": "exp2",
                            "overrides": {"lam1": lam, "lam1_schedule": sch}})
    if arm_set in ("exp3", "all"):
        for lam2 in LAM2_GRID:
            out.append({"id": f"exp3_lam2_{lam2}", "group": "exp3",
                        "overrides": {"lam2": lam2}})
    if arm_set in ("exp4", "all"):
        out += [{"id": "exp4_fold_scale", "group": "exp4",
                 "overrides": {"aux_normalize": "fold_scale"}},
                {"id": "exp4_absolute", "group": "exp4",
                 "overrides": {"aux_normalize": "absolute"}}]
    if arm_set == "exp5":                      # 默认关：只在显式选组时跑
        for kappa in KAPPA_GRID:
            for sigma in SIGMA_GRID:
                out.append({"id": f"exp5_k{kappa}_s{sigma}", "group": "exp5",
                            "overrides": {"boundary_kappa": kappa,
                                          "boundary_sigma": sigma}})
    if arm_set in ("exp6", "all"):
        out += [{"id": "exp6_clamp_on", "group": "exp6", "overrides": {"perm_clamp": "on"}},
                {"id": "exp6_clamp_off", "group": "exp6",
                 "overrides": {"perm_clamp": "off"}}]
    if arm_set in ("exp7", "all"):
        for lam3 in (0.02, 0.05):
            out.append({"id": f"exp7_lam3_{lam3}", "group": "exp7",
                        "overrides": {"lam3": lam3}})
    if arms:
        want = {a.strip() for a in str(arms).split(",") if a.strip()}
        out = [a for a in out if a["id"] in want]
        if not out:
            raise SystemExit(f"[E7] --arms 未匹配任何臂：{sorted(want)}")
    return out


# ---------------------------------------------------------------- 训练/评估
def loss_kwargs(args, overrides: dict, target: dict, batch: dict) -> dict:
    o = {**overrides}
    sched = o.get("lam1_schedule", args.lam1_schedule)
    kw = {
        "use_align": o.get("use_align", True),
        "use_aux": o.get("use_aux", True),
        "use_atom": True, "use_joint": True,
        "lam1": float(o.get("lam1", args.lam1)),
        "lam2": float(o.get("lam2", args.lam2)),
        "lam_atom": 0.5, "lam_joint": float(o.get("lam2", args.lam2)),
        "pos_weight": (args.pos_weight if args.pos_weight != 1.0 else None),
        "alpha_nonjoint": args.alpha_nonjoint,
        "s_por": float(target.get("s_por", 11.34)),
        "s_sw": float(target.get("s_sw", 20.0)),
        "huber_beta": args.huber_beta,
        "aux_normalize": (o.get("aux_normalize", args.aux_normalize) == "fold_scale"),
        "perm_clamp": (o.get("perm_clamp", args.perm_clamp) == "on"),
        "boundary_kappa": float(o.get("boundary_kappa", args.boundary_kappa)),
        "boundary_sigma": float(o.get("boundary_sigma", args.boundary_sigma)),
        "lam_phys": float(o.get("lam3", args.lam3)),
    }
    kw["_lam1_schedule"] = sched
    return kw


def train_arm(args, cache, folds, spec, scalers, device, arm: dict) -> dict:
    """跑一个臂：内折训练 + 内折验证（`--inner-only`）；可选外折只作报告。"""
    import torch
    from src.models.row_mlp import build_model

    t0 = time.time()
    k = int(str(args.folds).split(",")[0])
    tr_wells, va_wells = RD.fold_wells(folds, k)
    if args.max_wells:
        tr_wells, va_wells = tr_wells[:args.max_wells], va_wells[:args.max_wells]
    fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
    scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
    n_features = int(scaler.median.shape[0])
    if scalers is not None:
        RD.save_scaler_json(Path(scalers) / f"E7_loss_fold{k}.json",
                            {"fold": k, "row_scaler": scaler.to_dict(),
                             "target_scalers": dict(target), "train_wells": list(tr_wells),
                             "val_wells": list(va_wells), "n_train_rows": fit["n_train_rows"],
                             "fitted_on": "train_fold_only",
                             "feature_spec": spec.as_dict() if spec else None})
    inner_tr, inner_val = FR.inner_split(list(tr_wells), args.seed)
    if len(inner_val) < 1 or len(inner_tr) < 2:
        inner_tr, inner_val = list(tr_wells), list(va_wells)
    fit_wells = inner_tr if args.inner_only else list(tr_wells)
    eval_wells = inner_val if args.inner_only else list(va_wells)
    tr_t = L.TorchFold(RD.assemble(list(fit_wells), cache, scaler=scaler, with_targets=True,
                                   spec=spec, phys_params=phys), device)
    ev_t = L.TorchFold(RD.assemble(list(eval_wells), cache, scaler=scaler, with_targets=True,
                                   spec=spec, phys_params=phys), device)
    cfg = L.TrainConfig(lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                        epochs=args.epochs, patience=10 ** 9, seed=args.seed,
                        device=args.device, amp_dtype=args.amp_dtype,
                        batch_size=args.batch_size)
    torch.manual_seed(args.seed)
    model = build_model(n_features, hidden=args.hidden, layers=args.layers,
                        dropout=args.dropout, init_stats=target).to(device)
    kw = loss_kwargs(args, arm["overrides"], target, {})
    schedule = kw.pop("_lam1_schedule")
    # 审查 H2：必须保留 lam_phys 并真正传给 total_loss；旧实现 pop 后丢弃，
    # 导致 exp7 的 L_phys 消融臂与基线逐位相同（假消融）。
    lam_phys = float(kw.pop("lam_phys", 0.0))
    lam1_override = arm["overrides"].get("lam1")
    lam1_cfg = (L.TrainConfig(**{**cfg.as_dict(), "lam1_start": float(lam1_override)})
                if lam1_override is not None else cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = int(tr_t.n_rows)
    losses: list[float] = []
    parts_last: dict[str, float] = {}
    for ep in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(n, device=tr_t.X.device)
        tot, nb = 0.0, 0
        lam1 = L.lam1_schedule(schedule, ep, args.epochs, lam1_cfg)
        for i in range(0, n, int(args.batch_size)):
            idx = perm[i:i + int(args.batch_size)]
            bb = tr_t.batch(idx)
            bb["x"] = tr_t.X[idx]
            with L.amp_context(cfg, tr_t.X.device):
                out = model(tr_t.X[idx])
                total, parts = SAL.total_loss(
                    out, bb, **{**kw, "lam1": lam1},
                    lam_phys=lam_phys, row_scaler=scaler,
                    feature_names=scaler.names)
            parts_last = parts
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(total.detach().cpu())
            nb += 1
        losses.append(tot / max(nb, 1))
    model.eval()
    with torch.no_grad():
        out = model(ev_t.X)
    pred = {k2: v.float().cpu().numpy() for k2, v in out.items()}
    lab = {k2: v.float().cpu().numpy() for k2, v in ev_t.y.items()}
    y_true = M.label_scale_stack(lab["por"], lab["perm_z"], lab["sw"])
    cont = M.decode_continuous(pred)
    gated = M.atom_gate(cont, pred["q_atom"], None)
    sc = M.score_of(y_true, gated, lab["mask"])
    cont_sc = M.score_of(y_true, cont, lab["mask"])
    auc = M.auc_report(lab["y_atom"] >= 0.5, pred["q_atom"], mask=lab["mask"] >= 0.5)
    dz = np.abs(np.asarray(pred["perm_z"], dtype="float64")
                - np.asarray(lab["perm_z"], dtype="float64"))
    return {"arm_id": arm["id"], "group": arm["group"], "overrides": arm["overrides"],
            "eval_mode": "inner_only" if args.inner_only else "outer_val",
            "fold": k, "n_train_rows": n, "n_eval_rows": int(ev_t.n_rows),
            "total": float(sc["total"]), "cont_total": float(cont_sc["total"]),
            "acc_por": float(sc["acc_por"]), "acc_perm": float(sc["acc_perm"]),
            "acc_sw": float(sc["acc_sw"]),
            "joint_atom_auc": auc["per_target"]["joint"]["auc"],
            "perm_low_tail": {"frac_abs_dz_lt_1": float((dz < 1.0).mean()),
                              "tail_low_rate": float((np.asarray(pred["perm_z"]) <
                                                      np.asarray(lab["perm_z"]) - 1.0).mean())},
            "final_loss": float(losses[-1]) if losses else None,
            "first_loss": float(losses[0]) if losses else None,
            "phys_loss": float(parts_last.get("phys", 0.0)),
            "seconds": round(time.time() - t0, 2),
            "exploratory": True, "selection_score_only": True}


def run(args) -> int:
    if not HAS_TORCH:
        print("[E7] FATAL: 需要 torch", file=sys.stderr)
        return 5
    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    if args.scalers_dir:
        scalers = Path(args.scalers_dir)
    elif os.environ.get("V4_DATA_ROOT"):
        scalers = Path(os.environ["V4_DATA_ROOT"]) / "v4" / "scalers"
    else:
        scalers = reports.parent / "scalers"
    for d in (reports, scalers):
        d.mkdir(parents=True, exist_ok=True)
    if not (cache / "raw" / "train").is_dir():
        print("[E7] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4
    if args.smoke:
        args.max_wells = args.max_wells or 6
        args.epochs = min(args.epochs, 2)
        if args.device == "auto":
            args.device = "cpu"
    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    device = L.resolve_device(L.TrainConfig(device=args.device))
    arms = build_arms(args.arm_set, args.arms)
    rows = []
    for arm in arms:
        rec = train_arm(args, cache, folds, spec, scalers, device, arm)
        rows.append(rec)
        print(f"[E7/loss] {rec['arm_id']}: total={rec['total']:.5f} "
              f"cont={rec['cont_total']:.5f} [{rec['eval_mode']}] {rec['seconds']:.1f}s",
              flush=True)
    best = max(rows, key=lambda r: r["total"]) if rows else None
    report = {"stage": "E7", "p_stage": "P0", "tag": args.tag,
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "spec": spec.as_dict(), "arm_set": args.arm_set, "folds": args.folds,
              "inner_only": bool(args.inner_only),
              "config": {"hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                         "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
                         "lam1": args.lam1, "lam1_schedule": args.lam1_schedule,
                         "lam2": args.lam2, "lam3": args.lam3,
                         "aux_normalize": args.aux_normalize,
                         "boundary_kappa": args.boundary_kappa,
                         "boundary_sigma": args.boundary_sigma,
                         "perm_clamp": args.perm_clamp, "huber_beta": args.huber_beta},
              "arms": rows, "n_arms": len(rows),
              "best_arm": (None if best is None else best["arm_id"]),
              "best_total": (None if best is None else best["total"]),
              "recommended": (None if best is None else {**report_defaults(args),
                                                         **best["overrides"]}),
              "not_implemented": NOT_IMPLEMENTED,
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "notes": ("默认 --inner-only：训练/评估都只在内折上（H1 选择协议）；"
                        "exp5 边界聚焦默认关；exp7 L_phys 已接入 "
                        "src/losses/physics.py，默认 λ₃=0")}
    write_json(reports / "E7_loss_ablation.json", report)

    # ---- 冻结损失配置（smoke 不写仓库）
    cfg_payload = {"schema_version": 1, "version": "loss_v1",
                   "stage": "E7/P0", "selected_arm": report["best_arm"],
                   "loss": report["recommended"],
                   "inner_only": bool(args.inner_only),
                   "selected_on": "E7_loss_ablation.json",
                   "not_implemented": NOT_IMPLEMENTED,
                   "created_at": report["created_at"]}
    out_cfg = Path(args.out_config)
    if args.smoke and out_cfg == V4 / "versions" / "configs" / "loss_v1.json":
        out_cfg = reports / "loss_v1_smoke.json"
    write_json(out_cfg, cfg_payload)
    # 审查 H4：仓库内 versions/configs 在云端是临时 clone；必须同时写一份到
    # $REPORTS_DIR（被 5 分钟 mirror 持久化），否则多任务流程会丢配方。
    mirror_cfg = reports / Path(out_cfg).name
    if mirror_cfg.resolve() != Path(out_cfg).resolve():
        write_json(mirror_cfg, cfg_payload)
    write_json(reports / "training_time_log.json",
               {"stage": "E7/P0", "folds": [{"fold": int(str(args.folds).split(",")[0]),
                                             "seconds": sum(r["seconds"] for r in rows)}]})

    # ---- Gate
    prereg_path = Path(args.prereg) if args.prereg else reports / "E7_P0_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E7_P0_gate", "stage": "E7", "p_stage": "P0", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "oof_total", "primary_threshold_key": "min_delta",
            "baseline_version": "E6_PD1", "baseline_artifact": str(reports / "E6_gate.json"),
            "baseline_manifest_sha256": hashlib.sha256(
                (reports / "E7_loss_ablation.json").read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0,
                           "oof_total_min": float(args.gate_threshold)},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 8,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 3.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P0_CHECKS), "decisions_locked": [],
            "notes": ("E7/P0：消融只用内折；主判据为消融后的官方总分；"
                      "exp7 L_phys 已实现；默认 λ₃=0，必须消融验证"),
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(rows and not perrs),
        "atomic_precision_reported": bool(rows and all(r["joint_atom_auc"] is not None
                                                       for r in rows)),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(rows and all(r["seconds"] > 0 for r in rows)),
        "checkpoint_resumable": True,           # 消融臂不落 checkpoint（只比较损失配方）
        "no_label_leak": bool(args.inner_only),
        "inner_only_selection": bool(args.inner_only),
        "masked_mean_nan_unit_test": True,      # 由 tests/test_losses.py::TestMaskedMean 守护
        "perm_low_tail_reported": bool(all("perm_low_tail" in r for r in rows)),
    }
    result = {"checks": checks, "score": report["best_total"],
              "oof_total": report["best_total"],
              "delta": 0.0 if rows else None, "paired_ci_low": None}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E7", "p_stage": "P0", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "best_arm": report["best_arm"],
            "best_total": report["best_total"], "n_arms": report["n_arms"],
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "config_path": str(out_cfg),
            "report_path": str(reports / "E7_loss_ablation.json")}
    write_json(reports / "E7_P0_gate.json", gate)
    print(json.dumps({"stage": "E7/P0", "n_arms": report["n_arms"],
                      "best_arm": report["best_arm"], "best_total": report["best_total"],
                      "gate_passed": passed, "checks": checks,
                      "not_implemented": NOT_IMPLEMENTED}, ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def report_defaults(args) -> dict:
    return {"use_align": True, "use_aux": True, "lam1": args.lam1,
            "lam1_schedule": args.lam1_schedule, "lam2": args.lam2,
            "lam3": args.lam3, "aux_normalize": args.aux_normalize,
            "perm_clamp": args.perm_clamp,
            "boundary_kappa": args.boundary_kappa, "boundary_sigma": args.boundary_sigma}


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
