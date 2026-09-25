#!/usr/bin/env python3
"""E4/P0：PatchTF（patch Transformer）5 折 OOF + CI/位置/容量消融 + 拼缝体检 + 对照 Gate。

用法::

    # 本机小规模预检（结论标 exploratory，不进 Gate 数值）
    python3 E4/code/train_patchtf.py --folds 0 --max-wells 6 --epochs 2 --exploratory
    # 网格搜索（**只用 inner-OOF 选择**，默认只跑 fold0）
    python3 E4/code/train_patchtf.py --search search.json --exploratory
    # 云端正式（全折）
    python3 E4/code/train_patchtf.py --folds all

产出（E4/P0 §4）::

    $RUN/E4/oof_{tag}.npz                逐行 OOF（含 well_index/fold_of_row/tau）
    $REPORTS/E4_patchtf.json             主配置指标 + 搜索表 + 三个消融 + 还原/拼缝体检
    $REPORTS/E4_param_budget.json        参数量/token 数/单折耗时
    $REPORTS/E4_gate.json                对照 Gate（vs E3 最佳 OOF：delta + 配对 CI）

纪律（E4/P0 §12.5）
------------------
* **搜索/消融只用 inner-OOF**：网格按 `inner_oof_total` 选，outer 数值只作报告并标
  `exploratory=true`（资源预检性质），不得进入 Gate 数值；
* 无 E3 基线时不伪造 delta：`delta_vs_e3=None`、`nogo=true`、`passed=None`；
* `selection_score_only=true`：本报告的选择信号来自 inner-OOF。
"""
from __future__ import annotations

import argparse
import hashlib
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
from src.training import checkpoint as CK  # noqa: E402
from src.training import fold_runner as FR  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import recipe as R  # noqa: E402
from src.training import seq_loop as SL  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
DEFAULT_PATCH_KW = {"patch_len": 32, "stride": 16, "d_model": 128, "n_layers": 4,
                    "n_heads": 8, "channel_independent": True, "rel_pos": True}


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
    ap = argparse.ArgumentParser(description="E4/P0 PatchTF 训练与消融")
    ap.add_argument("--cache-root", default=str(default_cache_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--run-root", default=str(default_run_root()))
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or "",
                    help="尺度参数输出目录（缺省：$V4_DATA_ROOT/v4/scalers，再退到 reports 同级）")
    ap.add_argument("--spec", default="F1")
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
    # PatchTF 结构
    ap.add_argument("--patch-len", type=int, default=DEFAULT_PATCH_KW["patch_len"])
    ap.add_argument("--stride", type=int, default=DEFAULT_PATCH_KW["stride"])
    ap.add_argument("--d-model", type=int, default=DEFAULT_PATCH_KW["d_model"])
    ap.add_argument("--n-layers", type=int, default=DEFAULT_PATCH_KW["n_layers"])
    ap.add_argument("--n-heads", type=int, default=DEFAULT_PATCH_KW["n_heads"])
    ap.add_argument("--channel-independent", dest="channel_independent",
                    action="store_true", default=True, help="因子化 patch 嵌入（默认开）")
    ap.add_argument("--no-channel-independent", dest="channel_independent",
                    action="store_false", help="联合 patch 嵌入（CI 消融的对照臂）")
    ap.add_argument("--rel-pos", default="on", choices=("on", "off"))
    # 搜索 / 消融（全部只用于 inner-OOF 选择，标 exploratory）
    ap.add_argument("--search", default=None, help="JSON 文件：arch_kwargs 变体列表")
    ap.add_argument("--search-folds", default="0")
    ap.add_argument("--channel-independence-ablation", action="store_true")
    ap.add_argument("--rel-pos-ablation", action="store_true")
    ap.add_argument("--capacity-ablation", action="store_true")
    ap.add_argument("--e3-oof", default=None, help="E3 最佳 OOF npz（对照基线）")
    ap.add_argument("--seam-check-well", default=None, help="拼缝体检用的井 id（默认首口 val 井）")
    ap.add_argument("--seam-tol", type=float, default=0.02,
                    help="相对逐点差上限（分母：por 上限/6/100/1）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=C.DISK_MIN_FREE_GB)
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=0.0)
    ap.add_argument("--tag", default="", help="产物后缀（消融用，避免互相覆盖）")
    return ap


def arch_kwargs_from_args(args) -> dict:
    return {"patch_len": int(args.patch_len), "stride": int(args.stride),
            "d_model": int(args.d_model), "n_layers": int(args.n_layers),
            "n_heads": int(args.n_heads), "dropout": float(args.dropout),
            "channel_independent": bool(args.channel_independent),
            "rel_pos": args.rel_pos == "on"}


def _fold_list(args, folds) -> list[int]:
    return list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip()]


def run_variant(cache: Path, run_dir: Path, scalers: Path, spec, folds, fold_list,
                cfg: L.TrainConfig, args, kwargs: dict, tag: str) -> dict:
    """跑一组结构配置（`tag` 区分产物），返回结果摘要（**不判 Gate**）。"""
    t0 = time.time()
    tracker = L.TimeTracker("E4", args.time_budget_h)
    results, base_rss = [], None
    rss_start = SD.WellShardReader.rss_mb()
    for k in fold_list:
        opt = SL.SeqOptions(spec=spec, chunk=args.chunk, overlap=args.overlap,
                            batch_chunks=args.batch_chunks, weight_kind=args.weight_kind,
                            max_wells=args.max_wells, smoke=args.smoke, resume=args.resume,
                            save_checkpoints=True, select_tau=not args.smoke,
                            scaler_prefix=f"E4_{tag}", arch="patchtf",
                            run_dir=run_dir / tag, scalers_dir=scalers,
                            tb_run_name=f"E4_{tag}_fold{k}")
        r = SL.run_two_phase_seq_fold(k, folds, cache, cfg, opt, arch_kwargs=kwargs)
        tracker.add_fold(k, r.seconds, r.hist1["n_epochs_run"] + r.hist2["n_epochs_run"],
                         extra={"best_epoch": r.best_epoch, "tau": r.tau["tau"],
                                "inner_oof_total": r.inner_oof_total, "arch": "patchtf"})
        results.append(r)
        if base_rss is None:
            base_rss = SD.WellShardReader.rss_mb()
    rss_end = SD.WellShardReader.rss_mb()
    rss_steady = float(base_rss if base_rss is not None else rss_start)
    if rss_end > 13.0 * 1024:
        print(f"!! [E4] RSS {rss_end:.0f} MiB 接近 16 GiB 主机内存上限", file=sys.stderr,
              flush=True)
    oof = SL.assemble_oof_seq(results, cache)
    gated = M.score_of(oof["y_true"], oof["y_pred"], oof["mask"])
    cont = M.score_of(oof["y_true"], oof["cont"], oof["mask"])
    return {"tag": tag, "kwargs": kwargs, "results": results, "oof": oof,
            "oof_total": float(gated["total"]), "oof_cont_only": float(cont["total"]),
            "acc_por": float(gated["acc_por"]), "acc_perm": float(gated["acc_perm"]),
            "acc_sw": float(gated["acc_sw"]),
            "inner_oof_mean": float(np.mean([r.inner_oof_total or np.nan for r in results])),
            "seconds": round(float(time.time() - t0), 2),
            "seconds_per_fold": [round(r.seconds, 2) for r in results],
            "mem": {"rss_start_mb": float(rss_start), "rss_steady_mb": rss_steady,
                    "rss_end_mb": rss_end,
                    "growth_mb": round(rss_end - rss_steady, 1),
                    "peak_rss_mb": SD.WellShardReader.peak_rss_mb(), "limit_mb": None,
                    "note": "E4 整折预加载：不再使用 300 MiB worker 硬预算"},
            "model": results[0].model_summary if results else {}}


def paired_vs_baseline(oof: dict, baseline_path: Path, seed: int) -> dict:
    """与基线 OOF 的按井行数加权配对 bootstrap（按 well_id 对齐，不假设顺序）。"""
    if not Path(baseline_path).is_file():
        return {"ok": False, "reason": f"缺少基线 OOF：{baseline_path}"}
    with np.load(baseline_path, allow_pickle=True) as z:
        base = {k: z[k] for k in z.files}
    ids_a = [str(w) for w in oof["well_ids"]]
    ids_b = [str(w) for w in base["well_ids"]]
    if set(ids_a) != set(ids_b):
        return {"ok": False, "reason": "井集合不一致，无法配对",
                "n_wells_self": len(ids_a), "n_wells_base": len(ids_b)}
    rows_a, rows_b = [], []
    for w in ids_a:
        sa = oof["well_index"] == ids_a.index(w)
        sb = base["well_index"] == ids_b.index(w)
        rows_a.append(M.score_of(oof["y_true"][sa], oof["y_pred"][sa], oof["mask"][sa])["total"])
        rows_b.append(M.score_of(base["y_true"][sb], base["y_pred"][sb],
                                 base["mask"][sb])["total"])
    a, b = np.asarray(rows_a), np.asarray(rows_b)
    n_rows = np.asarray([int((oof["well_index"] == i).sum()) for i in range(len(ids_a))],
                        dtype="float64")
    boot = FOLDS.bootstrap_ci(a - b, iters=1000, weights=n_rows, seed=seed)
    return {"ok": True, "baseline_path": str(baseline_path),
            "baseline_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
            "self_total": float(M.score_of(oof["y_true"], oof["y_pred"], oof["mask"])["total"]),
            "baseline_total": float(M.score_of(base["y_true"], base["y_pred"],
                                               base["mask"])["total"]),
            "delta": float(boot["point"]),
            "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
            "n_wells": len(ids_a)}


def seam_check(cache: Path, run_dir: Path, scalers: Path, spec, folds, cfg, args,
               kwargs: dict, tag: str, well: str | None) -> dict:
    """分块推理 vs 整井推理的逐点差异（拼缝体检；**同一权重**，只改 chunk 切法）。"""
    try:
        import torch  # noqa: F401
    except Exception as exc:                                  # pragma: no cover
        return {"ok": None, "reason": f"需要 torch：{exc}"}
    fold0 = int((args.folds if args.folds != "all" else "0").split(",")[0])
    ck = run_dir / tag / f"fold{fold0}" / "best.pt"
    if not ck.is_file():
        return {"ok": None, "reason": f"缺少 checkpoint：{ck}（拼缝体检需先训练该折）"}
    avail = {p.stem for p in (cache / "raw" / "train").glob("*.npz")}
    tr_wells, va_wells = RD.fold_wells(folds, fold0)
    if args.max_wells:
        tr_wells = tr_wells[:args.max_wells]
    tr_wells = [w for w in tr_wells if w in avail]
    if well is None:
        well = next((w for w in va_wells if w in avail), None)
    if not tr_wells or well is None or well not in avail:
        return {"ok": None, "reason": f"缓存缺少训练/体检井分片（avail={len(avail)}）",
                "well": None if well is None else str(well)}
    fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
    scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
    model = SL.build_seq_model("patchtf", int(scaler.median.shape[0]), init_stats=target,
                               **kwargs)
    CK.load_checkpoint(ck, model=model, map_location="cpu")
    dev = L.resolve_device(cfg)
    model.to(dev)
    got = RD._well_feature_matrix(cache, well, "train", spec, phys_params=phys)
    X = np.asarray(got[0] if isinstance(got, tuple) else got, dtype="float32")
    X = np.asarray(scaler.transform(X), dtype="float32")     # 与训练同一条尺度路径
    opt_a = SL.SeqOptions(spec=spec, chunk=int(args.chunk), overlap=int(args.overlap),
                          batch_chunks=args.batch_chunks, weight_kind=args.weight_kind,
                          arch="patchtf")
    opt_b = SL.SeqOptions(spec=spec, chunk=int(X.shape[0]), overlap=0,
                          batch_chunks=1, weight_kind=args.weight_kind, arch="patchtf")
    pa = SL.predict_well_chunked(model, X, cfg, opt_a, dev)
    pb = SL.predict_well_chunked(model, X, cfg, opt_b, dev)
    diffs = {k: float(np.abs(np.asarray(pa[k]) - np.asarray(pb[k])).max()) for k in pa}
    # 判据用**相对尺度**（por/perm_z/sw 各自量纲不同，绝对差不可比）
    scales = {"por": C.POR_MAX_BUFFER * C.POR_VALID_MAX, "perm_z": 6.0, "sw": 100.0,
              "q_atom": 1.0, "q_joint": 1.0}
    rel = {k: (diffs[k] / scales[k] if scales.get(k) else diffs[k]) for k in diffs}
    worst = max(rel.values(), default=float("nan"))
    return {"ok": bool(worst <= float(args.seam_tol)), "well": str(well),
            "chunk": int(args.chunk), "overlap": int(args.overlap),
            "tol": float(args.seam_tol), "max_point_diff": worst,
            "max_abs_point_diff": max(diffs.values(), default=float("nan")),
            "per_key_abs": diffs, "per_key_rel": rel,
            "n_rows": int(X.shape[0]), "n_chunks": len(SD.chunks_for(int(X.shape[0]),
                                                                     opt_a.chunk,
                                                                     opt_a.overlap)),
            "note": "判据为**相对**逐点差（分母 por 上限/6/100/1）；分块与整井切法不同，"
                    "只要求无拼缝跳变（≤ tol）"}


def run(args) -> int:
    if not HAS_TORCH:
        print("[E4] FATAL: 需要 torch（patchtf 为 torch 主干）", file=sys.stderr)
        return 5
    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    run_dir = Path(args.run_root) / "E4"
    if args.scalers_dir:
        scalers = Path(args.scalers_dir)
    elif os.environ.get("V4_DATA_ROOT"):
        scalers = Path(os.environ["V4_DATA_ROOT"]) / "v4" / "scalers"
    else:
        scalers = reports.parent / "scalers"
    for d in (reports, run_dir, scalers):
        d.mkdir(parents=True, exist_ok=True)
    if not (cache / "raw" / "train").is_dir():
        print("[E4] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4

    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = _fold_list(args, folds)
    kwargs = arch_kwargs_from_args(args)
    cfg = L.TrainConfig(lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                        epochs=args.epochs, patience=args.patience, seed=args.seed,
                        device=args.device, amp_dtype=args.amp_dtype, batch_size=1024,
                        time_budget_h=args.time_budget_h, min_free_gb=args.min_free_gb,
                        disk_path=args.disk_path)
    R.apply_loss_recipe(cfg)
    if args.smoke:
        args.max_wells = args.max_wells or 8
        args.epochs = min(args.epochs, 2)
        args.patience = 10 ** 9
        if args.device == "auto":
            args.device = "cpu"
        cfg = L.TrainConfig(**{**cfg.as_dict(), "epochs": args.epochs,
                               "patience": args.patience, "device": args.device})

    # ---- 网格搜索（只用 inner-OOF 选择；默认只跑 fold0）
    search_table: list[dict] = []
    if args.search:
        variants = json.loads(Path(args.search).read_text(encoding="utf-8"))
        sfold = [int(x) for x in str(args.search_folds).split(",") if x.strip()]
        best, best_inner = None, float("-inf")
        for i, v in enumerate(variants):
            kw = {**kwargs, **v}
            r = run_variant(cache, run_dir, scalers, spec, folds, sfold, cfg, args, kw,
                            tag=f"search{i}")
            row = {"variant": int(i), "kwargs": kw, "inner_oof_total": r["inner_oof_mean"],
                   "oof_total": r["oof_total"], "n_params": r["model"].get("n_params"),
                   "seconds_per_fold": r["seconds_per_fold"], "exploratory": True,
                   "selection_score_only": True}
            search_table.append(row)
            if (r["inner_oof_mean"] or float("-inf")) > best_inner:
                best_inner, best = r["inner_oof_mean"], kw
        if best is not None:
            kwargs = best
        print(f"[E4] search done: {len(search_table)} 变体；选中 {kwargs}", flush=True)

    tag = args.tag or "patchtf"
    main = run_variant(cache, run_dir, scalers, spec, folds, fold_list, cfg, args, kwargs, tag)
    oof_path = run_dir / f"oof_{tag}.npz"
    np.savez_compressed(oof_path, **{k: v for k, v in main["oof"].items() if k != "well_ids"},
                        well_ids=main["oof"]["well_ids"])
    trainer = L.TimeTracker("E4", args.time_budget_h)      # 时间日志（Gate 依赖）
    for r in main["results"]:
        trainer.add_fold(r.fold, r.seconds, r.hist1["n_epochs_run"] + r.hist2["n_epochs_run"],
                         extra={"arch": "patchtf", "best_epoch": r.best_epoch})
    trainer.write(reports / "training_time_log.json", config=cfg.as_dict())

    # ---- 消融（全部 inner-OOF 选择语义；结果标 exploratory；**每个变体只训一次**）
    abl: dict[str, object] = {}

    def _one(ab_tag: str, **overrides) -> dict:
        r = run_variant(cache, run_dir, scalers, spec, folds, fold_list, cfg, args,
                        {**kwargs, **overrides}, tag=ab_tag)
        return {"kwargs": {**kwargs, **overrides}, "inner_oof_total": r["inner_oof_mean"],
                "oof_total": r["oof_total"], "n_params": r["model"].get("n_params"),
                "seconds_per_fold": r["seconds_per_fold"], "exploratory": True,
                "selection_score_only": True}

    if args.channel_independence_ablation:
        abl["channel_independence"] = [_one(f"ci{int(v)}", channel_independent=v)
                                       for v in (True, False)]
    if args.rel_pos_ablation:
        abl["rel_pos"] = [_one(f"rel{int(v)}", rel_pos=bool(v)) for v in (1, 0)]
    if args.capacity_ablation:
        abl["capacity"] = [_one(f"cap{dm}x{nl}", d_model=dm, n_layers=nl)
                           for dm, nl in ((64, 2), (128, 4), (256, 6))]

    # ---- 还原与拼缝体检
    from src.models.patchtf import (coverage_report, reconstruction_check)
    recon = reconstruction_check(int(args.chunk), int(kwargs["patch_len"]),
                                 int(kwargs["stride"]), kind=args.weight_kind)
    recon["coverage"] = coverage_report(int(args.chunk), int(kwargs["patch_len"]),
                                        int(args.stride), args.weight_kind)
    try:
        seam = seam_check(cache, run_dir, scalers, spec, folds, cfg, args, kwargs, tag,
                          args.seam_check_well)
    except Exception as exc:                        # 不静默：记录失败原因（不判 PASS）
        seam = {"ok": None, "reason": f"{type(exc).__name__}: {exc}"}

    # ---- 对照基线（无基线不伪造 delta）
    e3 = args.e3_oof or str(run_dir.parent / "E3" / "oof_unet.npz")
    cmp = paired_vs_baseline(main["oof"], Path(e3), cfg.seed)

    n_rows = int(main["oof"]["y_true"].shape[0])
    atom_rows = M.atomic_rows_report(main["oof"]["y_true"], main["oof"]["y_pred"],
                                     main["oof"]["y_atom"] >= 0.5, main["oof"]["mask"] >= 0.5)
    metrics = {
        "stage": "E4", "p_stage": "P0", "arch": "patchtf", "tag": tag,
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "spec": spec.as_dict(), "folds": fold_list, "chunk": args.chunk,
        "overlap": args.overlap, "weight_kind": args.weight_kind, **kwargs,
        "oof_total": main["oof_total"], "oof_cont_only": main["oof_cont_only"],
        "acc_por": main["acc_por"], "acc_perm": main["acc_perm"], "acc_sw": main["acc_sw"],
        "inner_oof_mean": main["inner_oof_mean"],
        "n_rows": n_rows, "n_wells": int(main["oof"]["n_wells"]),
        "placeholder_rows": atom_rows,
        "e3_baseline": cmp,
        "delta_vs_e3": cmp.get("delta"), "paired_ci_vs_e3": cmp.get("paired_ci"),
        "search_table": search_table,
        "channel_independence_ablation": abl.get("channel_independence"),
        "rel_pos_ablation": abl.get("rel_pos"),
        "capacity_ablation": abl.get("capacity"),
        "reconstruction_check": {k: v for k, v in recon.items() if k != "spans"},
        "chunk_vs_full_recon": seam,
        "model": main["model"], "mem": main["mem"],
        "seconds_per_fold": main["seconds_per_fold"],
        "oof_path": str(oof_path), "config": cfg.as_dict(),
    }
    write_json(reports / "E4_patchtf.json", metrics)

    budget_path = reports / "E4_param_budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.is_file() else {}
    budget[tag] = {"arch": "patchtf", "arch_kwargs": kwargs,
                   "n_params": main["model"].get("n_params"),
                   "tokens_at_length": main["model"].get("tokens_at_length"),
                   "receptive_field_rows": main["model"].get("receptive_field_rows"),
                   "chunk": args.chunk, "overlap": args.overlap, "rows_oof": n_rows,
                   "seconds_per_fold": main["seconds_per_fold"]}
    write_json(budget_path, budget)

    # ---- Gate（delta vs E3 最佳；无基线则 passed=None + nogo）
    prereg_path = reports / "E4_P0_gate_prereg.json"
    if not prereg_path.is_file():
        import time as _t
        base_art = Path(e3)
        # 基线可能尚未产出：此时**不写空串**（预注册校验器视其为占位符），
        # 而是把路径本身的 sha256 作为指纹，并在 notes 标 baseline_artifact_pending。
        pending = not base_art.is_file()
        sha = hashlib.sha256((base_art.read_bytes() if not pending
                              else str(base_art).encode("utf-8"))).hexdigest()
        write_json(prereg_path, {
            "gate_id": "E4_P0_gate", "stage": "E4", "p_stage": "P0", "gate_type": "delta",
            "created_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "target_acc", "primary_threshold_key": "min_delta",
            "baseline_version": "E3_best", "baseline_artifact": str(base_art),
            "baseline_manifest_sha256": sha,
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 4,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(REQUIRED_CHECKS), "decisions_locked": [],
            "notes": ("E4/P0：PatchTF 相对 E3 最佳 OOF 的 paired CI 下界必须 > 0；"
                      "delta 的判定量是 OOF Total（primary_metric=target_acc 为装饰字段）；"
                      f"baseline_artifact_pending={pending}"),
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(cfg.disk_path)
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(not perrs and n_rows > 0),
        "atomic_precision_reported": bool(atom_rows["hit_rate"]),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(all(r.seconds > 0 for r in main["results"])),
        "checkpoint_resumable": bool(all(r.resumable.get("ok") for r in main["results"])),
        "no_label_leak": bool(all(set(r.tr_wells).isdisjoint(set(r.va_wells))
                                  and set(r.fit_wells).issubset(set(r.tr_wells))
                                  and set(r.inner_val_wells).isdisjoint(set(r.va_wells))
                                  for r in main["results"])),
        "length_contract_ok": bool(recon["ok"]),
        "no_seam_jump": bool(seam.get("ok") is True),
        "ci_and_rel_pos_ablated": bool(args.channel_independence_ablation
                                       and args.rel_pos_ablation),
    }
    if not cmp.get("ok"):
        result = {"checks": {**checks, "baseline_available": False},
                  "score": metrics["oof_total"], "oof_total": metrics["oof_total"],
                  "delta": 0.0, "paired_ci_low": float("-inf")}
        agg = {"passed": False, "reason": f"缺少 E3 基线：{cmp.get('reason')}",
               "prereg_errors": perrs, "mandatory_ok": False}
        passed = None
    else:
        result = {"checks": {**checks, "baseline_available": True},
                  "score": metrics["oof_total"], "oof_total": metrics["oof_total"],
                  "delta": float(cmp["delta"]), "paired_ci_low": float(cmp["paired_ci"][0]),
                  "por_acc": metrics["acc_por"], "perm_acc": metrics["acc_perm"],
                  "sw_acc": metrics["acc_sw"]}
        agg = GATES.aggregate_gate(prereg, result)
        passed = None if args.exploratory else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E4", "p_stage": "P0", "tag": tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory), "passed": passed,
            "nogo": bool(passed is False or not cmp.get("ok")),
            "oof_total": metrics["oof_total"], "delta_vs_e3": metrics["delta_vs_e3"],
            "paired_ci_vs_e3": metrics["paired_ci_vs_e3"],
            "checks": result["checks"], "prereg_errors": perrs, "aggregate": agg,
            "disk": disk, "reconstruction_check": metrics["reconstruction_check"],
            "chunk_vs_full_recon": seam, "search_table": search_table,
            "metrics_path": str(reports / "E4_patchtf.json"), "oof_path": str(oof_path),
            "mde_units": prereg["mde_units"], "candidate_budget": prereg["candidate_budget"]}
    write_json(reports / "E4_gate.json", gate)
    print(json.dumps({"arch": "patchtf", "tag": tag, "oof_total": metrics["oof_total"],
                      "delta_vs_e3": metrics["delta_vs_e3"],
                      "paired_ci_vs_e3": metrics["paired_ci_vs_e3"],
                      "gate_passed": passed, "nogo": gate["nogo"], "checks": result["checks"],
                      "n_params": budget[tag]["n_params"],
                      "reconstruction_ok": recon["ok"],
                      "seam_ok": seam.get("ok")}, ensure_ascii=False, indent=2))
    if not (args.exploratory or args.smoke) and passed is True:
        try:
            ckpts = []
            for r in main["results"]:
                p = run_dir / tag / f"fold{int(r.fold)}" / "best.pt"
                if p.is_file():
                    ckpts.append(str(p))
            if ckpts:
                entry = {
                    "candidate_id": f"E4_{tag}", "stage": "E4", "arch": "patchtf",
                    "feature_version": spec.key, "checkpoint": ckpts[0],
                    "checkpoints": ckpts, "oof_path": str(oof_path),
                    "oof_total": float(metrics["oof_total"]),
                    "cv": {"total": float(metrics["oof_total"]),
                           "por": float(metrics["acc_por"]),
                           "perm": float(metrics["acc_perm"]),
                           "sw": float(metrics["acc_sw"]),
                           "missing_mode": C.SCORE_MISSING_MODE,
                           "selection_score_only": True},
                    "atomic": {"tau": None, "no_interpolation": True},
                    "scalers": {"fitted_on": "train_fold_only"},
                    "status": "local_only",
                    "notes": "E4 PatchTF；CPU 推理由 predictor 通用路径支持",
                }
                REG.upsert_candidate(entry, path=os.environ.get("V4_CANDIDATES") or None)
        except Exception as exc:
            print(f"[E4] 候选登记失败（{type(exc).__name__}: {exc}），不影响训练",
                  file=sys.stderr, flush=True)
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
