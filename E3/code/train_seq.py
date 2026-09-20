#!/usr/bin/env python3
"""E3/P2：序列主干的 5 折 OOF + 感受野消融 + 行级受控对照 + 硬 Gate（≥81.0）。

用法::

    # 本机小规模预检（结论标 exploratory，不进 Gate 数值）
    python3 E3/code/train_seq.py --arch tcn --folds 0 --max-wells 6 --epochs 2 --exploratory

    # 云端正式（全折）
    python3 E3/code/train_seq.py --arch unet --folds all
    python3 E3/code/train_seq.py --arch tcn  --folds all

产出（E3/P2 §4）::

    $RUN/E3/oof_{arch}.npz               逐行 OOF（含 well_index/fold_of_row/tau）
    $REPORTS/E3_metrics_{arch}.json      逐折/逐目标/占位行/连续切片/配对 CI
    $REPORTS/E3_param_budget.json        参数量/感受野上界/单折耗时
    $REPORTS/E3_boundary_report.json     井首尾 10 m vs 中段 + 拼缝体检
    $REPORTS/E3_gate.json                硬 Gate（OOF ≥ 81.0 + mandatory checks）

感受野消融与行级对照分别在 `rf_ablation.py` / `compare_row_vs_seq.py`。
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
from src.data import seq_dataset as SD  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import fold_runner as FR  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import seq_loop as SL  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
ARCH_KW = {
    "unet": {"base_ch": 64, "depth": 5, "k": 5},
    "tcn": {"channels": 128, "n_blocks": 9, "k": 3, "dilation_max": 512},
}


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def default_cache_root() -> Path:
    return env_path("V4_CACHE_ROOT", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))


def default_reports_dir() -> Path:
    return env_path("V4_REPORTS_DIR", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))


def default_run_root() -> Path:
    return env_path("V4_RUN_ROOT", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v4 E3 sequence backbone runner")
    ap.add_argument("--arch", default="unet", choices=("unet", "tcn"))
    ap.add_argument("--cache-root", default=str(default_cache_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--run-root", default=str(default_run_root()))
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or "",
                    help="scaler JSON 目录；缺省时：有 V4_DATA_ROOT 用它，否则与 reports 同级")
    ap.add_argument("--spec", default="F1", help="特征版本：F1 / F1+phys / F1+win / F1+well / F2")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-chunks", type=int, default=4)
    ap.add_argument("--weight-kind", default="triangular",
                    choices=("triangular", "hann", "equal"))
    ap.add_argument("--arch-kwargs", default=None, help="JSON：覆盖 ARCH_KW（消融用）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=C.DISK_MIN_FREE_GB)
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=81.0)
    ap.add_argument("--tag", default="", help="产物后缀（消融用，避免互相覆盖）")
    return ap


def run_arch(args) -> int:
    if not HAS_TORCH:
        print("[E3] FATAL: 需要 torch", file=sys.stderr)
        return 5
    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    run_dir = Path(args.run_root) / "E3"
    if args.scalers_dir:
        scalers = Path(args.scalers_dir)
    elif os.environ.get("V4_DATA_ROOT"):
        scalers = Path(os.environ["V4_DATA_ROOT"]) / "v4" / "scalers"
    else:
        # 本机预检：不要往 /data 写（无权限），放在 reports 同级
        scalers = reports.parent / "scalers"
    for d in (reports, run_dir, scalers):
        d.mkdir(parents=True, exist_ok=True)
    if not (cache / "raw" / "train").is_dir():
        print("[E3] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4

    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip()]
    kwargs = dict(ARCH_KW[args.arch])
    if args.arch_kwargs:
        kwargs.update(json.loads(args.arch_kwargs))

    cfg = L.TrainConfig(lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                        epochs=args.epochs, patience=args.patience, seed=args.seed,
                        device=args.device, amp_dtype=args.amp_dtype, batch_size=1024,
                        time_budget_h=args.time_budget_h, min_free_gb=args.min_free_gb,
                        disk_path=args.disk_path)
    if args.smoke:
        args.max_wells = args.max_wells or 8
        args.epochs = min(args.epochs, 2)
        args.patience = 10 ** 9
        if args.device == "auto":
            args.device = "cpu"
        cfg = L.TrainConfig(**{**cfg.as_dict(), "epochs": args.epochs,
                               "patience": args.patience, "device": args.device})

    t0 = time.time()
    tracker = L.TimeTracker("E3", args.time_budget_h)
    results = []
    # worker 内存预算（E3/P0 §7）的**正确语义**：300 MB 是"每个 worker 的增量"预算，
    # 而训练进程本身要装 torch/模型/优化器/缓存分配器（实测 Python+torch 首次前向后
    # RSS 会涨 ~400 MB）。因此基线取**首个折跑完后的稳态 RSS**，检查的是
    # "后续折是否持续增长" —— 这才等价于"没有把分片/井矩阵攒在内存里"。
    base_rss = None
    for k in fold_list:
        opt = SL.SeqOptions(spec=spec, chunk=args.chunk, overlap=args.overlap,
                            batch_chunks=args.batch_chunks, weight_kind=args.weight_kind,
                            max_wells=args.max_wells, smoke=args.smoke, resume=args.resume,
                            save_checkpoints=True, select_tau=not args.smoke,
                            scaler_prefix=f"E3_{args.arch}", arch=args.arch,
                            run_dir=run_dir / args.arch, scalers_dir=scalers,
                            tb_run_name=f"E3_{args.arch}_fold{k}")
        r = SL.run_two_phase_seq_fold(k, folds, cache, cfg, opt, arch_kwargs=kwargs)
        tracker.add_fold(k, r.seconds, r.hist1["n_epochs_run"] + r.hist2["n_epochs_run"],
                         extra={"best_epoch": r.best_epoch, "tau": r.tau["tau"],
                                "inner_oof_total": r.inner_oof_total, "arch": args.arch})
        results.append(r)
        if base_rss is None:
            base_rss = SD.WellShardReader.rss_mb()          # 稳态基线（首折后）
        print(f"[E3/{args.arch}] fold{k} done {r.seconds:.1f}s inner={r.inner_oof_total} "
              f"rss={SD.WellShardReader.rss_mb()}MiB", flush=True)
    rss_end = SD.WellShardReader.rss_mb()
    SD.WellShardReader.assert_worker_budget(limit_mb=300.0, baseline_mb=float(base_rss or 0.0))
    mem = {"rss_steady_mb": float(base_rss or 0.0), "rss_end_mb": rss_end,
           "growth_mb": round(rss_end - float(base_rss or 0.0), 1), "limit_mb": 300.0,
           "note": "稳态基线取首折后 RSS；检查的是后续折的持续增长（泄漏检测）"}

    oof = SL.assemble_oof_seq(results, cache)
    tag = args.tag or args.arch
    oof_path = run_dir / f"oof_{tag}.npz"
    np.savez_compressed(oof_path, **{k: v for k, v in oof.items() if k != "well_ids"},
                        well_ids=oof["well_ids"])

    gated = M.score_of(oof["y_true"], oof["y_pred"], oof["mask"])
    cont = M.score_of(oof["y_true"], oof["cont"], oof["mask"])
    atom_rows = M.atomic_rows_report(oof["y_true"], oof["y_pred"], oof["y_atom"] >= 0.5,
                                     oof["mask"] >= 0.5)
    cont_rows = ~(oof["y_atom"] >= 0.5).all(axis=1)
    cont_slice = M.score_of(oof["y_true"][cont_rows], oof["y_pred"][cont_rows],
                            oof["mask"][cont_rows])
    well_tot, well_rows = FR.per_well_totals(oof)
    const = np.tile([C.ATOM_VALUES[t] for t in C.TARGETS], (oof["y_true"].shape[0], 1))
    well_const, _ = FR.per_well_totals(oof, y_pred=const)
    boot = FOLDS.bootstrap_ci(well_tot - well_const, iters=1000, weights=well_rows,
                              seed=cfg.seed)
    all_preds = {w: r.pred[w] for r in results for w in r.va_wells}
    boundary = SL.boundary_report(all_preds, cache)

    metrics = {
        "stage": "E3", "arch": args.arch, "tag": tag, "exploratory": bool(args.exploratory),
        "spec": spec.as_dict(), "folds": fold_list, "chunk": args.chunk,
        "overlap": args.overlap, "weight_kind": args.weight_kind,
        "arch_kwargs": kwargs, "config": cfg.as_dict(),
        "n_rows": int(oof["y_true"].shape[0]), "n_wells": int(oof["n_wells"]),
        "oof_total": float(gated["total"]), "oof_cont_only": float(cont["total"]),
        "acc_por": float(gated["acc_por"]), "acc_perm": float(gated["acc_perm"]),
        "acc_sw": float(gated["acc_sw"]),
        "delta_vs_const": float(gated["total"] - C.CONSTANT_BASELINE_OOF),
        "paired_ci_vs_const": [float(boot["ci_low"]), float(boot["ci_high"])],
        "placeholder_rows": atom_rows, "cont_slice": cont_slice,
        "folds_detail": [SL.fold_metrics_seq(r, cache, SL.SeqOptions(spec=spec)) for r in results],
        "model": results[0].model_summary if results else {},
        "coverage": [r.coverage for r in results],
        "seconds_total": round(time.time() - t0, 2),
        "memory": mem,
        "oof_path": str(oof_path),
    }
    write_json(reports / f"E3_metrics_{tag}.json", metrics)

    budget_path = reports / "E3_param_budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.is_file() else {}
    budget[tag] = {"arch": args.arch, "arch_kwargs": kwargs,
                   "n_params": (results[0].model_summary.get("n_params") if results else None),
                   "chunk": args.chunk, "overlap": args.overlap, "rows_oof": metrics["n_rows"],
                   "seconds_total": metrics["seconds_total"],
                   "seconds_per_fold": [round(r.seconds, 2) for r in results]}
    write_json(budget_path, budget)
    write_json(reports / "E3_boundary_report.json",
               {"arch": args.arch, "tag": tag, **boundary})

    # ---- Gate（delta：OOF 增量 + 绝对门槛 81.0；exploratory 不判）
    prereg_path = reports / "E3_P2_gate_prereg.json"
    if not prereg_path.is_file():
        const_ref = reports / "E1_const_baseline.json"
        if not const_ref.is_file():
            write_json(const_ref, {"version": "CONST", "missing_mode": C.SCORE_MISSING_MODE,
                                   "oof_total": C.CONSTANT_BASELINE_OOF})
        import hashlib
        sha = hashlib.sha256(const_ref.read_bytes()).hexdigest()
        write_json(prereg_path, {
            "gate_id": "E3_P2_gate", "stage": "E3", "p_stage": "P2", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "oof_total", "primary_threshold_key": "min_delta",
            "baseline_version": "CONST", "baseline_artifact": str(const_ref),
            "baseline_manifest_sha256": sha,
            "thresholds": {"min_delta": 10.5, "min_effect_floor": 0.0,
                           "oof_total_min": float(args.gate_threshold)},
            "alpha": 0.05, "multiplicity": "none", "candidate_budget": 1,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 20.0,
            "mandatory_checks": list(REQUIRED_CHECKS), "decisions_locked": [],
            "notes": "E3/P2 硬 Gate：OOF ≥ 81.0（= 常数基线 + 10.5）；序列必须优于同头行级",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(cfg.disk_path)
    except Exception as exc:                      # 不静默：记录失败原因
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(not perrs and metrics["n_rows"] > 0),
        "atomic_precision_reported": bool(atom_rows["hit_rate"]),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(all(r.seconds > 0 for r in results)),
        "checkpoint_resumable": bool(all(r.resumable.get("ok") for r in results)),
        "no_label_leak": bool(all(set(r.tr_wells).isdisjoint(set(r.va_wells))
                                  and set(r.fit_wells).issubset(set(r.tr_wells))
                                  and set(r.inner_val_wells).isdisjoint(set(r.va_wells))
                                  for r in results)),
    }
    result = {"checks": checks, "score": metrics["oof_total"], "oof_total": metrics["oof_total"],
              "delta": metrics["delta_vs_const"],
              "paired_ci_low": float(boot["ci_low"]),
              "por_acc": metrics["acc_por"], "perm_acc": metrics["acc_perm"],
              "sw_acc": metrics["acc_sw"]}
    agg = GATES.aggregate_gate(prereg, result)
    gate = {"gate_id": prereg["gate_id"], "stage": "E3", "arch": args.arch, "tag": tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory),
            "passed": None if args.exploratory else bool(agg["passed"]),
            "oof_total": metrics["oof_total"], "threshold": args.gate_threshold,
            "delta_vs_const": metrics["delta_vs_const"],
            "paired_ci_low": result["paired_ci_low"], "checks": checks,
            "prereg_errors": perrs, "aggregate": agg,
            "disk": disk,
            "boundary_summary": {"max_edge_gap": boundary["max_edge_gap"],
                                 "within_threshold": boundary["within_threshold"]},
            "metrics_path": str(reports / f"E3_metrics_{tag}.json"),
            "oof_path": str(oof_path)}
    write_json(reports / f"E3_gate_{tag}.json" if tag != "unet" and tag != "tcn"
               else reports / "E3_gate.json", gate)
    print(json.dumps({"arch": args.arch, "tag": tag, "oof_total": metrics["oof_total"],
                      "gate_passed": gate["passed"], "checks": checks,
                      "boundary_max_gap": boundary["max_edge_gap"],
                      "n_params": budget[tag]["n_params"], "exploratory": gate["exploratory"],
                      "rss_growth_mb": mem["growth_mb"]},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke:
        return 0
    return 0 if gate["passed"] else 3


def main(argv: list[str] | None = None) -> int:
    return run_arch(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
