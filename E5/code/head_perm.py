#!/usr/bin/env python3
"""E5/P1：PERM 头（冻结骨干）——三档 z 输出 + 桶头对照 + 尾部一致性 + 配对 CI Gate。

用法::

    python3 E5/code/head_perm.py --folds 0 --max-wells 6 --epochs 2 --smoke --ablation
    python3 E5/code/head_perm.py --arch patchtf --backbone-ckpt $RUN/E4/patchtf/fold0/best.pt

硬判据（E5/P1 §7）
----------------
* 主判据是**连续切片** PERM Acc（官方口径 `1 − |max(ẑ−z, log10(ε))|`，极端低估被截断保护）；
* 输出必须**有限且 PERM > 0**（`10**perm_z`），不允许 ≤0；`init` 用训练折 `perm_z` 中位数（不是 0）；
* 与骨干自身 PERM 的配对 bootstrap CI 下界 > 0 才采纳，否则 NO-GO；
* `E5_perm_tail.json` 必须给出**按真值量级分桶**的单调性结论（官方 Acc 与对齐损失反号）。

产出::

    $RUN/E5/perm/oof.npz、$REPORTS/E5_perm.json、$REPORTS/E5_perm_tail.json、
    $REPORTS/E5_perm_param_ablation.json（--ablation）、$REPORTS/E5_P1_gate.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.models.target_heads import tail_consistency_report  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402

import e5_common as EC  # noqa: E402

Z_OUTPUTS = ("tanh", "clip", "linear")
AUX_KINDS = ("smooth_l1", "soft_ce_bucket", "quantile")
TARGET_INDEX = 1                       # PERM 在 C.TARGETS 里的列序


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E5/P1 PERM 头（冻结骨干）")
    EC.add_common_args(ap)
    ap.add_argument("--z-output", default="tanh", choices=Z_OUTPUTS)
    ap.add_argument("--clip-min", type=float, default=C.PERM_LOG_MIN)
    ap.add_argument("--clip-max", type=float, default=C.PERM_LOG_MAX)
    ap.add_argument("--init-z-median", type=float, default=None,
                    help="缺省 = 训练折 perm_z 中位数（不是 0）")
    ap.add_argument("--aux", default="smooth_l1", choices=AUX_KINDS)
    ap.add_argument("--n-buckets", type=int, default=0, help=">0 时启用 soft-CE 桶头（对照臂）")
    ap.add_argument("--quantile-heads", type=int, default=0, help="≥3 时启用分位辅助头")
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--tail-bins", type=int, default=8)
    ap.add_argument("--tau-perm", type=float, default=None,
                    help="PERM 原子阈值；缺省在内折上用骨干 q_atom 选")
    ap.set_defaults(target="PERM")
    return ap


def perm_acc_np(z_true, z_pred, mask) -> float:
    """官方 PERM 命中率（截断到 `log10(eps)`）——与 `score.acc_perm` 同构。"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    z = np.asarray(z_true, dtype="float64")[m]
    p = np.asarray(z_pred, dtype="float64")[m]
    d = np.maximum(p - z, math.log10(C.EPS))
    return float(np.clip(1.0 - np.abs(d), 0.0, 1.0).mean())


def gate_perm(z, z_pred_linear_ok: bool, tau, q) -> np.ndarray:
    """用骨干 `q_atom` 的 PERM 列做硬切换（命中 → 线性 0.01）。"""
    if tau is None or q is None:
        return z_pred_linear_ok
    return np.where(q > float(tau), float(C.ATOM_VALUES["PERM"]), z_pred_linear_ok)


def run(args) -> int:
    if not HAS_TORCH:
        print("[E5] FATAL: 需要 torch", file=sys.stderr)
        return 5
    import torch  # noqa: F401

    from src.models.target_heads import PermHead   # torch 门控：--help 不应需要 torch

    cache, reports, run_dir, scalers = EC.resolve_dirs(args)
    if not (cache / "raw" / "train").is_dir():
        print("[E5] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4
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

    acc = {"z": [], "perm_z": [], "perm_z_gated": [], "base_z": [], "base_gated": [],
           "mask": [], "y_atom": [], "well_index": [], "fold_of_row": []}
    well_ids: list[str] = []
    fold_records, random_init_any, z_median = [], False, None
    ctx = None
    for k in fold_list:
        tk = time.time()
        ctx = EC.collect_fold(args, cache, folds, k, spec, scalers, dev)
        random_init_any = random_init_any or ctx["backbone_random_init"]
        labels = EC.well_labels(cache, ctx["tr_wells"] + ctx["va_wells"])
        EC.attach_backbone_q(ctx, labels, list(ctx["va_wells"]) + list(ctx["inner_val"]))
        z_median = float(ctx["target"].get("perm_z_median", -0.08))
        tau, tau_info = args.tau_perm, {"source": "cli"}
        if tau is None:
            tau, tau_info = EC.select_tau_inner(ctx, labels, TARGET_INDEX)
        use_bucket = int(args.n_buckets) > 0

        def factory(d_model, _z=z_median):
            head = PermHead(d_model, hidden=args.hidden, dropout=args.dropout,
                            z_output=args.z_output,
                            clip=(args.clip_min, args.clip_max),
                            n_buckets=int(args.n_buckets) or None,
                            quantile_heads=int(args.quantile_heads))
            # 初值用**训练折** perm_z 中位数（不是 0）；--init-z-median 只用于对照实验
            head.init_from_stats(perm_z_median=float(
                _z if args.init_z_median is None else args.init_z_median))
            return head

        def predict_fn(head, x, _bucket=use_bucket):
            out = head(x)
            return out["perm_z_bucket"] if _bucket else out["perm_z"]

        def loss_fn(pred, y):
            return SAL.align_score_log(y, pred)

        def metric_fn(y_true, y_pred, mask):
            return perm_acc_np(y_true, y_pred, mask)

        head, best_epoch, inner_score = EC.train_head_two_phase(
            args, factory, {w: ctx["states"][w] for w in ctx["inner_tr"]},
            {w: labels[w] for w in ctx["inner_tr"]},
            {w: ctx["states"][w] for w in ctx["inner_val"]},
            {w: labels[w] for w in ctx["inner_val"]}, TARGET_INDEX, predict_fn, loss_fn,
            metric_fn)
        z_pred = EC.predict_wells(head, {w: ctx["states"][w] for w in ctx["va_wells"]},
                                 ctx["va_wells"], predict_fn, dev)
        y = np.concatenate([labels[w]["y"] for w in ctx["va_wells"]])[:, TARGET_INDEX]
        mask = np.concatenate([labels[w]["mask"] for w in ctx["va_wells"]])[:, TARGET_INDEX]
        base_z = np.concatenate([np.asarray(ctx["backbone_pred"][w]["perm_z"],
                                           dtype="float64") for w in ctx["va_wells"]])
        q = np.concatenate([labels[w]["q_atom_t"][:, TARGET_INDEX] for w in ctx["va_wells"]])
        gated = gate_perm(z_pred, z_pred, tau, q)
        base_gated = gate_perm(base_z, base_z, tau, q)
        m = mask.astype(bool)
        dz = np.abs(z_pred[m] - y[m]) if m.any() else np.zeros(0)
        rec = {"fold": k, "best_epoch": best_epoch, "inner_perm_acc": inner_score,
               "tau": tau_info, "n_rows": int(y.shape[0]),
               "perm_cont_acc": perm_acc_np(y, z_pred, m),
               "perm_gated_acc": perm_acc_np(y, gated, m),
               "base_cont_acc": perm_acc_np(y, base_z, m),
               "base_gated_acc": perm_acc_np(y, base_gated, m),
               "z_sigma": float(np.std(z_pred[m])) if m.any() else None,
               "frac_abs_dz_lt_1": float((dz < 1.0).mean()) if dz.size else None,
               "tail_low_rate": float((z_pred[m] < y[m] - 1.0).mean()) if m.any() else None,
               "tail_high_rate": float((z_pred[m] > y[m] + 1.0).mean()) if m.any() else None,
               "perm_positive": bool(np.all(np.power(10.0, z_pred) > 0)),
               "finite": bool(np.isfinite(z_pred).all()),
               "z_output": args.z_output, "n_buckets": int(args.n_buckets),
               "quantile_heads": int(args.quantile_heads), "z_init": z_median,
               "tr_wells": list(ctx["tr_wells"]), "va_wells": list(ctx["va_wells"]),
               "inner_tr": list(ctx["inner_tr"]), "inner_val": list(ctx["inner_val"]),
               "seconds": round(time.time() - tk, 2)}
        rec["delta_cont"] = float(rec["perm_cont_acc"] - rec["base_cont_acc"])
        fold_records.append(rec)
        acc["z"].append(y)
        acc["perm_z"].append(z_pred)
        acc["perm_z_gated"].append(gated)
        acc["base_z"].append(base_z)
        acc["base_gated"].append(base_gated)
        acc["mask"].append(mask)
        acc["y_atom"].append(np.concatenate([labels[w]["y_atom"] for w in ctx["va_wells"]]))
        acc["well_index"].append(np.concatenate(
            [np.full(int(labels[w]["y"].shape[0]), len(well_ids) + i)
             for i, w in enumerate(ctx["va_wells"])]))
        acc["fold_of_row"].append(np.full(int(y.shape[0]), k))
        well_ids += list(ctx["va_wells"])
        tracker.add_fold(k, rec["seconds"], 0, extra={"perm_cont_acc": rec["perm_cont_acc"]})
        print(f"[E5/perm] fold{k} z_output={args.z_output} perm_cont_acc="
              f"{rec['perm_cont_acc']:.5f} base={rec['base_cont_acc']:.5f} "
              f"delta={rec['delta_cont']:+.5f} tau={tau:.3f} {rec['seconds']:.1f}s",
              flush=True)

    for key in list(acc):
        acc[key] = np.concatenate(acc[key]) if acc[key] else np.zeros((0,))
    oof_path = run_dir / f"oof{('_' + args.tag) if args.tag else ''}.npz"
    np.savez_compressed(oof_path, **acc, well_ids=np.asarray(well_ids, dtype=object))
    tracker.write(reports / "training_time_log.json",
                  config={"stage": "E5/P1", "epochs": args.epochs, "arch": args.arch})

    cs = ~np.asarray(acc["y_atom"], dtype=bool).all(axis=1)
    ms = acc["mask"].astype(bool) & cs
    perm_cont = perm_acc_np(acc["z"], acc["perm_z"], ms)
    perm_gated = perm_acc_np(acc["z"], acc["perm_z_gated"], ms)
    base_cont = perm_acc_np(acc["z"], acc["base_z"], ms)
    base_gated = perm_acc_np(acc["z"], acc["base_gated"], ms)
    n_wells = len(well_ids)
    rows = np.asarray([float((acc["well_index"] == i).sum()) for i in range(n_wells)])
    boot = EC.paired_well_delta(acc["z"], acc["perm_z"], acc["base_z"], ms,
                                acc["well_index"], n_wells, rows, perm_acc_np,
                                seed=args.seed)
    dz = np.abs(acc["perm_z"][ms] - acc["z"][ms]) if ms.any() else np.zeros(0)
    tail = tail_consistency_report(acc["z"][ms], acc["perm_z"][ms], n_bins=args.tail_bins)
    EC.write_json(reports / "E5_perm_tail.json",
                  {"stage": "E5", "p_stage": "P1", "target": "PERM",
                   "bins": tail["bins"], "monotone_consistent": tail["monotone_consistent"],
                   "spearman_acc_vs_z": tail["spearman_acc_vs_z"],
                   "spearman_loss_vs_z": tail["spearman_loss_vs_z"],
                   "note": tail["note"], "exploratory": bool(args.exploratory)})
    atom_rows = M.atomic_rows_report(np.stack([acc["z"], acc["z"], acc["z"]], axis=1),
                                     np.stack([acc["perm_z"], acc["perm_z"],
                                               acc["perm_z"]], axis=1),
                                     acc["y_atom"],
                                     np.repeat(acc["mask"][:, None], 3, axis=1) >= 0.5)
    metrics = {
        "stage": "E5", "p_stage": "P1", "target": "PERM", "tag": args.tag,
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "spec": spec.as_dict(), "folds": fold_list, "arch": args.arch,
        "arch_kwargs": (ctx or {}).get("arch_kw"), "backbone_ckpt": args.backbone_ckpt,
        "backbone_random_init": bool(random_init_any),
        "z_output": args.z_output, "n_buckets": int(args.n_buckets),
        "quantile_heads": int(args.quantile_heads), "aux": args.aux,
        "clip": [args.clip_min, args.clip_max], "z_init": z_median,
        "perm_cont_acc": perm_cont, "perm_gated_acc": perm_gated,
        "base_cont_acc": base_cont, "base_gated_acc": base_gated,
        "delta_cont": float(perm_cont - base_cont),
        "paired_ci": boot["ci"], "paired_point": boot["point"],
        "per_well_delta": boot["per_well"], "bootstrap_unit": boot["bootstrap_unit"],
        "z_sigma_mean": float(np.mean([r["z_sigma"] for r in fold_records
                                       if r["z_sigma"] is not None] or [0.0])),
        "frac_abs_dz_lt_1": float((dz < 1.0).mean()) if dz.size else None,
        "tail_low_rate": float((acc["perm_z"][ms] < acc["z"][ms] - 1.0).mean())
        if ms.any() else None,
        "tail_high_rate": float((acc["perm_z"][ms] > acc["z"][ms] + 1.0).mean())
        if ms.any() else None,
        "tail_monotone_consistent": tail["monotone_consistent"],
        "contract": {"perm_positive": bool(np.all(np.power(10.0, acc["perm_z"]) > 0)),
                     "finite": bool(np.isfinite(acc["perm_z"]).all()),
                     "no_le_scores": True},
        "placeholder_rows": atom_rows, "cont_slice_rows": int(cs.sum()),
        "n_rows": int(acc["z"].shape[0]), "n_wells": n_wells,
        "folds_detail": fold_records, "oof_path": str(oof_path),
        "seconds_total": round(time.time() - t0, 2),
        "no_label_leak": bool(all(set(r["inner_val"]).isdisjoint(set(r["va_wells"]))
                                  for r in fold_records)),
        "notes": ("PERM 主判据是连续切片 Acc；z 截断口径 log10(eps)=%.1f；"
                  "桶头（n_buckets>0）与 tanh/clip/linear 构成对照" % math.log10(C.EPS)),
    }
    EC.write_json(reports / "E5_perm.json", metrics)

    # ---- 消融：z 输出三臂（+ 桶头臂）
    ablation = None
    if args.ablation:
        afold = int(str(args.ablation_folds).split(",")[0])
        ctx0 = EC.collect_fold(args, cache, folds, afold, spec, scalers, dev)
        labels0 = EC.well_labels(cache, ctx0["tr_wells"] + ctx0["va_wells"])
        rows_abl = []
        arms = [{"z_output": z, "n_buckets": 0} for z in Z_OUTPUTS]
        arms.append({"z_output": "tanh", "n_buckets": 6})
        for arm in arms:
            label = arm["z_output"] + ("+bucket" if arm["n_buckets"] else "")

            def factory(d_model, _a=arm):
                return PermHead(d_model, hidden=args.hidden, dropout=args.dropout,
                                z_output=_a["z_output"],
                                clip=(args.clip_min, args.clip_max),
                                n_buckets=int(_a["n_buckets"]) or None)

            def predict_fn(head, x, _a=arm):
                out = head(x)
                return out["perm_z_bucket"] if _a["n_buckets"] else out["perm_z"]

            head, _be, inner = EC.train_head_two_phase(
                args, factory, {w: ctx0["states"][w] for w in ctx0["inner_tr"]},
                {w: labels0[w] for w in ctx0["inner_tr"]},
                {w: ctx0["states"][w] for w in ctx0["inner_val"]},
                {w: labels0[w] for w in ctx0["inner_val"]}, TARGET_INDEX, predict_fn,
                lambda pred, y: SAL.align_score_log(y, pred), perm_acc_np)
            zp = EC.predict_wells(head, {w: ctx0["states"][w] for w in ctx0["va_wells"]},
                                  ctx0["va_wells"], predict_fn, dev)
            yv = np.concatenate([labels0[w]["y"] for w in ctx0["va_wells"]])[:, TARGET_INDEX]
            mv = np.concatenate([labels0[w]["mask"] for w in ctx0["va_wells"]]
                                )[:, TARGET_INDEX].astype(bool)
            rows_abl.append({"arm": label, "fold": afold, "inner_perm_acc": inner,
                             "perm_cont_acc": perm_acc_np(yv, zp, mv),
                             "within_clip": bool(arm["z_output"] != "linear"
                                                 or np.abs(zp).max() <= abs(args.clip_max)),
                             "exploratory": True, "selection_score_only": True})
            print(f"[E5/perm] ablation {label}: inner={inner:.5f} "
                  f"cont={rows_abl[-1]['perm_cont_acc']:.5f}", flush=True)
        ablation = {"stage": "E5", "p_stage": "P1", "target": "PERM", "rows": rows_abl,
                    "recommended": args.z_output,
                    "note": "linear 臂无界（可越 [-6,6]，但 PERM>0 仍成立）；桶头是单调期望解码臂",
                    "exploratory": True, "selection_score_only": True}
        EC.write_json(reports / "E5_perm_param_ablation.json", ablation)

    gate = EC.write_gate(
        args, reports, gate_id="E5_P1_gate", p_stage="P1", metrics=metrics,
        compares={"delta": metrics["delta_cont"], "ci_low": boot["ci"][0],
                  "ci_high": boot["ci"][1]},
        extra_checks={
            "perm_positive_ok": bool(metrics["contract"]["perm_positive"]),
            "perm_finite_ok": bool(metrics["contract"]["finite"]),
            "z_init_not_zero": bool(abs(float(z_median)) > 1e-9),
            "tail_report_written": bool((reports / "E5_perm_tail.json").is_file()),
            "tail_monotone_consistent": bool(tail["monotone_consistent"]),
            "ablation_table_complete": bool(ablation is None or len(ablation["rows"]) >= 3),
        },
        prereg_baseline=args.backbone_ckpt or str(oof_path),
        prereg_notes=("E5/P1：PERM 连续切片 Acc 相对冻结骨干自身 PERM 提升，paired CI 下界 > 0；"
                      "输出必须有限且 PERM>0；尾部单调性单独出报告"),
        target_index=TARGET_INDEX, primary_metric="perm_acc")
    print(json.dumps({"stage": "E5/P1", "z_output": args.z_output,
                      "perm_cont_acc": perm_cont, "base_cont_acc": base_cont,
                      "delta_cont": metrics["delta_cont"], "paired_ci": boot["ci"],
                      "tail_monotone": tail["monotone_consistent"],
                      "gate_passed": gate["passed"], "checks": gate["checks"]},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or gate["passed"] is None:
        return 0
    return 0 if gate["passed"] else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
