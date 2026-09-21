#!/usr/bin/env python3
"""E1 行级基线：5 折 OOF + 真实评分早停 + 硬 Gate（OOF Total ≥ 78.0）。

用法（云端由 `run_train.sh --mode stage --stage E1` 调用）::

    python3 E1/code/train_row.py --train-dir /data/v4/data/train --test-dir /data/v4/data/test \\
        --cache-root /data/v4/cache --out-dir /data/v4/runs/E1

    # 本机冒烟（8 井 / 2 epoch / fold0，不判 Gate）
    python3 E1/code/train_row.py --train-dir ../data/train --test-dir ../data/test \\
        --cache-root /tmp/v4cache --out-dir /tmp/v4runs/E1 --smoke --device cpu

协议（E1/P0 §12.5 + E1/P1 §5）：每个 outer 折两阶段，**outer 折不参与任何选择**
--------------------------------------------------------------------------
1. **选择阶段**：把 outer-train 井切成 `C.N_INNER_FOLDS=3` 个 inner 折；
   `inner0` 作验证、`inner1+2` 作训练；每个 epoch 用真实 `score.py` 打分（早停依据是**分数**
   而不是 loss），训练结束时把**最优 epoch 权重**写回模型（`keep_best=True`）；
   再在该模型的 `inner0` 预测上选逐目标原子阈值 `tau`（平台中点规则、官方目标函数）。
2. **终训阶段**：用**全部** outer-train 井按同一配方训练 `best_epoch` 个 epoch；
   对 outer-val 折**只推理一次**，用阶段 1 选出的 `tau` 解码。

产出（E1/P1 §4）::

    $OUT/fold{k}/{best,last,last_prev}.pt + *.manifest.json
    $OUT/oof.npz
    $REPORTS/E1_row_features.json、E1_loss_curve.csv、E1_metrics.json、E1_gate.json、
             E1_P1_gate_prereg.json、training_time_log.json
    $SCALERS/E1_fold{k}.json
    versions/candidates.json::E1_PD0

退出码：0 通过/冒烟；3 Gate 未过；4 数据/Cache 缺失；5 无 torch；6 预注册非法。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.data import dataset as D  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import basic as F  # noqa: E402
from src.inference import atomic_gate as AG  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.score import score_arrays  # noqa: E402
from src.training import checkpoint as CK  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as G  # noqa: E402

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
# 单折协议与标签/掩码取用统一走 fold_runner（E1/E2/E3 共用一份实现）
from src.training.fold_runner import labels_of, mask_of, y_atom_of  # noqa: E402
CONST_TILE = np.asarray([C.ATOM_VALUES[t] for t in C.TARGETS], dtype="float64")


# ---------------------------------------------------------------- 路径小工具
def _env_root(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def default_reports_dir() -> Path:
    return _env_root("V4_REPORTS_DIR", str(_env_root("V4_DATA_ROOT", "/data") / "v4" / "reports"))


def default_run_root() -> Path:
    return _env_root("V4_RUN_ROOT", str(_env_root("V4_DATA_ROOT", "/data") / "v4" / "runs"))


def default_cache_root() -> Path:
    return _env_root("V4_CACHE_ROOT", str(_env_root("V4_DATA_ROOT", "/data") / "v4" / "cache"))


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: str | Path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p



# ---------------------------------------------------------------- cache
def ensure_cache(args) -> dict:
    cache = Path(args.cache_root)
    have_raw = (cache / "raw" / "train").is_dir() and any((cache / "raw" / "train").glob("*.npz"))
    info: dict = {"cache_root": str(cache), "raw_present": bool(have_raw)}
    if not have_raw:
        if not args.train_dir:
            raise SystemExit("[E1] raw 分片缺失且未提供 --train-dir，无法构建缓存")
        t0 = time.time()
        man = D.build_cache(args.train_dir, args.test_dir or args.train_dir, cache)
        info["build_cache"] = {"seconds": round(time.time() - t0, 2),
                               "counts": man.get("counts", {})}
    wells = sorted(p.stem for p in (cache / "raw" / "train").glob("*.npz"))
    info["build_row_cache"] = RD.build_row_cache(cache, wells, "train", verbose=False)
    info["n_train_wells_cached"] = len(wells)
    info["row_cache_bytes"] = RD.row_dir(cache, "train").stat().st_size \
        if RD.row_dir(cache, "train").is_dir() else 0
    return info


# ---------------------------------------------------------------- 单折
def run_fold(fold: int, folds: dict, cache: Path, run_dir: Path, scalers_dir: Path,
             cfg: L.TrainConfig, args) -> dict:
    """单折 = `fold_runner.run_two_phase_fold` + E1 的曲线/日志接线。

    协议实现只有一份（`src/training/fold_runner.py`），E1 只负责：
      * 把每 epoch 的训练/验证指标写成 `E1_loss_curve.csv` 的行；
      * 把 `FoldResult` 映射成 main() 汇总 OOF 所需的形状。
    """
    from src.training import fold_runner as FR

    curve: list[dict] = []

    def _row(phase: str, epoch: int, rec: dict) -> dict:
        val = rec.get("val") or {}
        return {"fold": fold, "phase": phase, "epoch": epoch,
                "seconds": rec.get("seconds"), "lr": rec.get("lr"),
                "lam1": rec.get("lam1"), "loss_total": rec.get("total"),
                "loss_align": rec.get("align"), "loss_aux": rec.get("aux"),
                "loss_joint": rec.get("joint"), "loss_atom": rec.get("atom"),
                "val_total_cont": val.get("total"), "val_por": val.get("acc_por"),
                "val_perm": val.get("acc_perm"), "val_sw": val.get("acc_sw"),
                "disk_free_gb": rec.get("disk_free_gb")}

    def on_select(epoch: int, rec: dict) -> None:
        curve.append(_row("select", epoch, rec))
        if (epoch + 1) % max(cfg.log_every, 1) == 0 or epoch < 2:
            t = (rec.get("val") or {}).get("total", float("nan"))
            print(f"[E1] fold{fold} ep{epoch:3d} loss={rec.get('total', float('nan')):.4f} "
                  f"innerOOF={t:.4f}", flush=True)

    def on_final(epoch: int, rec: dict) -> None:
        curve.append(_row("final", epoch, rec))

    opt = FR.FoldOptions(spec=None, max_wells=args.max_wells, smoke=args.smoke,
                         resume=args.resume, save_checkpoints=True, select_tau=True,
                         scaler_prefix="E1", run_dir=run_dir, scalers_dir=scalers_dir,
                         tb_run_name=f"E1_pd0_fold{fold}",
                         on_select_epoch=on_select, on_final_epoch=on_final)
    res = FR.run_two_phase_fold(fold, folds, cache, cfg, opt)
    print(f"[E1] fold{fold}: train={len(res.tr_wells)} val={len(res.va_wells)} | "
          f"por_max={res.target['por_max']:.3f} sw_mu={res.target['sw_mu']:.3f} "
          f"sw_sigma={res.target['sw_sigma']:.3f} s_por={res.target['s_por']:.3f} "
          f"s_sw={res.target['s_sw']:.3f}; innerOOF={res.inner_oof_total} "
          f"best_epoch={res.best_epoch} tau={['%.2f' % t for t in res.tau['tau']]}",
          flush=True)
    return {"fold": res.fold, "tau": res.tau, "best_epoch": res.best_epoch,
            "inner_oof_total": res.inner_oof_total, "pred": res.pred, "va": res.va,
            "scaler": res.scaler, "target": res.target, "seconds": res.seconds,
            "hist1": res.hist1, "hist2": res.hist2, "curve": curve,
            "resumable": res.resumable, "tr_wells": res.tr_wells,
            "va_wells": res.va_wells, "fit_wells": res.fit_wells,
            "inner_tr_wells": res.inner_tr_wells,
            "inner_val_wells": res.inner_val_wells}


# ---------------------------------------------------------------- 契约
def oof_contract(oof: dict) -> dict:
    """E1 的 OOF 解码契约：PERM>0 / SW 量纲守卫 / 深度对齐 / 无插值 / 有限值。

    （提交级契约 `inference.contract.validate_payload` 留给 E10；E1 没有测试集权重，
    所以这里校验的是**同一套解码纪律**在 OOF 上确实成立。）
    """
    y_pred = np.asarray(oof["y_pred"], dtype="float64")
    cont = np.asarray(oof["cont"], dtype="float64")
    q_atom = np.asarray(oof["q_atom"], dtype="float64")
    tau = np.asarray(oof["tau_per_row"], dtype="float64")
    checks: dict[str, bool] = {}
    checks["perm_positive"] = bool((y_pred[:, 1] > 0).all())
    frac_low = float(np.mean(y_pred[:, 2] < C.SW_LOW_GUARD_ABS))
    checks["sw_not_unit_scale"] = bool(frac_low <= C.SW_LOW_GUARD_FRAC_MAX)
    non_atom = ~np.isclose(y_pred[:, 2], C.SW_PLACEHOLDER, atol=1e-9)
    if int(non_atom.sum()) < C.SW_LOW_GUARD_MIN_NONATOM:
        checks["sw_nonatom_p05_ok"] = True
    else:
        checks["sw_nonatom_p05_ok"] = bool(
            float(np.percentile(y_pred[non_atom, 2], 5)) >= C.SW_LOW_GUARD_NONATOM_P05_MIN)
    hit = q_atom >= tau
    atom_ok = np.ones_like(y_pred, dtype=bool)
    for t, name in enumerate(C.TARGETS):
        atom_ok[:, t] = (~hit[:, t]) | np.isclose(y_pred[:, t], C.ATOM_VALUES[name], atol=1e-9)
    checks["atom_no_interpolation"] = bool(atom_ok.all())
    checks["depth_alignment"] = bool(np.allclose(oof["depth"], oof["depth_in"], atol=1e-6))
    checks["finite"] = bool(np.isfinite(y_pred).all())
    checks["cont_finite"] = bool(np.isfinite(cont).all())
    return {"ok": bool(all(checks.values())), "checks": checks,
            "sw_frac_below_guard": frac_low,
            "note": "E1 无测试集权重；此处校验 OOF 上的解码纪律（PERM>0/SW 量纲/无插值/深度对齐）"}


def per_well_totals(y_true, y_pred, mask_bool, well_index, n_wells):
    """逐井官方 Total（cluster bootstrap 的单元）与逐井行数（权重）。"""
    tot = np.full(n_wells, np.nan)
    rows = np.zeros(n_wells, dtype="int64")
    for w in range(n_wells):
        sel = well_index == w
        rows[w] = int(sel.sum())
        if rows[w]:
            tot[w] = score_arrays(y_true[sel], y_pred[sel], missing=~mask_bool[sel],
                                  missing_mode=C.SCORE_MISSING_MODE)["total"]
    return tot, rows


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 E1 row-level baseline (5-fold OOF)")
    ap.add_argument("--train-dir", default=os.environ.get("V4_TRAIN_DIR"))
    ap.add_argument("--test-dir", default=os.environ.get("V4_TEST_DIR"))
    ap.add_argument("--cache-root", default=str(default_cache_root()))
    ap.add_argument("--out-dir", default=str(default_run_root() / "E1"))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    # 默认优先 `V4_SCALERS_DIR`（与 E4–E6 同一约定），否则退到 $V4_DATA_ROOT/v4/scalers
    ap.add_argument("--scalers-dir",
                    default=os.environ.get("V4_SCALERS_DIR")
                    or str(_env_root("V4_DATA_ROOT", "/data") / "v4" / "scalers"))
    ap.add_argument("--candidates", default=os.environ.get("V4_CANDIDATES") or str(V4 / "versions" / "candidates.json"),
                    help="候选注册表路径（测试必须指向 tmp，避免污染仓库）")
    ap.add_argument("--folds", default="all", help="all | 0 | 0,3")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=C.DISK_MIN_FREE_GB)
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args(argv)

    if not HAS_TORCH:
        print("[E1] FATAL: 需要 torch（本机契约层请跑 tests/run_all.py）", file=sys.stderr)
        return 5
    if args.smoke:
        args.folds = "0"
        args.max_wells = args.max_wells or 8
        args.epochs = min(args.epochs, 2)
        args.patience = 10 ** 9
        if args.device == "auto":
            args.device = "cpu"

    reports, run_dir, scalers_dir = Path(args.reports_dir), Path(args.out_dir), Path(args.scalers_dir)
    for d in (reports, run_dir, scalers_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = L.TrainConfig(hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                        epochs=args.epochs, patience=args.patience, seed=args.seed,
                        time_budget_h=args.time_budget_h, device=args.device,
                        amp_dtype=args.amp_dtype, min_free_gb=args.min_free_gb,
                        disk_path=args.disk_path)
    t_start = time.time()
    cache_info = {"skipped": True} if args.skip_cache else ensure_cache(args)
    cache = Path(args.cache_root)

    folds = FOLDS.load_folds()
    fold_list = list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip() != ""]
    print(f"[E1] cfg={cfg.as_dict()}", flush=True)
    print(f"[E1] folds={fold_list} cache={cache} run={run_dir} reports={reports} "
          f"folds_file={folds['source_path']} sha={folds['source_sha256'][:12]}", flush=True)

    # ---- 预注册（实验前写入；已存在则只校验，绝不改写）
    const_baseline_path = reports / "E1_const_baseline.json"
    write_json(const_baseline_path, {"version": "CONST", "missing_mode": C.SCORE_MISSING_MODE,
                                     "oof_total": C.CONSTANT_BASELINE_OOF,
                                     "note": "常数基线 (0.1,0.01,99.9)，E0 复算锚点 70.4907"})
    prereg_path = reports / "E1_P1_gate_prereg.json"
    if prereg_path.is_file():
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    else:
        prereg = {
            "gate_id": "E1_P1_gate", "stage": "E1", "p_stage": "P1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gate_type": "delta", "primary_metric": "oof_total",
            "primary_threshold_key": "min_delta",
            "baseline_version": "CONST", "baseline_artifact": str(const_baseline_path),
            "baseline_manifest_sha256": sha256_file(const_baseline_path),
            "thresholds": {"min_delta": 7.5, "min_effect_floor": 0.0, "oof_total_min": 78.0},
            "alpha": 0.05, "multiplicity": "none", "candidate_budget": 1,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(REQUIRED_CHECKS), "decisions_locked": [],
            "notes": "E1/P1 硬 Gate：OOF Total ≥ 78.0 且 5 折同向；tau 只在 inner-OOF 选",
        }
        write_json(prereg_path, prereg)
    errs = G.validate_prereg(prereg)
    if errs:
        print("[E1] PREREG INVALID:", errs, file=sys.stderr)
        return 6

    # ---- 逐折
    tracker = L.TimeTracker("E1", args.time_budget_h)
    results = []
    for k in fold_list:
        r = run_fold(k, folds, cache, run_dir, scalers_dir, cfg, args)
        tracker.add_fold(k, r["seconds"], len(r["hist1"]["epochs"]) + len(r["hist2"]["epochs"]),
                         extra={"best_epoch": r["best_epoch"], "tau": r["tau"]["tau"],
                                "inner_oof_total": r["inner_oof_total"]})
        results.append(r)
        print(f"[E1] fold{k} done in {r['seconds']:.1f}s", flush=True)

    # ---- 汇总 OOF（well_index 必须做全局偏移，否则跨折井号会撞车）
    well_ids: list[str] = []
    y_true_l, y_pred_l, cont_l, q_atom_l, q_joint_l, mask_l = [], [], [], [], [], []
    depth_l, depth_in_l, fold_l, tau_l, y_atom_l, well_index_l = [], [], [], [], [], []
    offset = 0
    for r in results:
        va = r["va"]
        pred = r["pred"]
        c = M.decode_continuous(pred)
        tau = np.asarray(r["tau"]["tau"], dtype="float64")
        g = M.atom_gate(c, pred["q_atom"], tau)
        d_in = np.concatenate([D.read_well_shard(cache, w, "train")["depth"] for w in va.well_ids])
        well_ids += list(va.well_ids)
        y_true_l.append(labels_of(va))
        y_pred_l.append(g)
        cont_l.append(c)
        q_atom_l.append(np.asarray(pred["q_atom"]))
        q_joint_l.append(np.asarray(pred["q_joint"]))
        mask_l.append(mask_of(va))
        depth_l.append(np.asarray(va.depth, dtype="float64"))
        depth_in_l.append(np.asarray(d_in, dtype="float64"))
        fold_l.append(np.full(va.n_rows, r["fold"], dtype="int32"))
        tau_l.append(np.tile(tau[None, :], (va.n_rows, 1)))
        y_atom_l.append(y_atom_of(va))
        well_index_l.append(np.asarray(va.well_index, dtype="int64") + offset)
        offset += va.n_wells
    oof = {
        "well_ids": np.array(well_ids, dtype=object),
        "well_index": np.concatenate(well_index_l),
        "depth": np.concatenate(depth_l), "depth_in": np.concatenate(depth_in_l),
        "y_true": np.concatenate(y_true_l), "y_pred": np.concatenate(y_pred_l),
        "cont": np.concatenate(cont_l), "q_atom": np.concatenate(q_atom_l),
        "q_joint": np.concatenate(q_joint_l), "mask": np.concatenate(mask_l),
        "y_atom": np.concatenate(y_atom_l),
        "fold_of_row": np.concatenate(fold_l), "tau_per_row": np.concatenate(tau_l),
    }
    oof_path = run_dir / "oof.npz"
    np.savez_compressed(oof_path, **oof)

    # ---- 指标
    m_bool = oof["mask"] >= 0.5
    a_bool = oof["y_atom"] >= 0.5
    cont_score = M.score_of(oof["y_true"], oof["cont"], oof["mask"])
    gated_score = M.score_of(oof["y_true"], oof["y_pred"], oof["mask"])
    atom_rows = M.atomic_rows_report(oof["y_true"], oof["y_pred"], a_bool, m_bool)
    head: dict[str, dict] = {}
    for t, name in enumerate(C.TARGETS):
        ps, rs, f1s = [], [], []
        for r in results:
            pr = M.atomic_precision_recall(y_atom_of(r["va"]) >= 0.5, r["pred"]["q_atom"],
                                           r["tau"]["tau"])[name]
            ps.append(pr["precision"]); rs.append(pr["recall"]); f1s.append(pr["f1"])
        head[name] = {"precision": float(np.nanmean(ps)), "recall": float(np.nanmean(rs)),
                      "f1": float(np.nanmean(f1s)),
                      "tau": [float(r["tau"]["tau"][t]) for r in results]}
    atom_slices = RD.atom_slice_report(oof["y_true"], oof["y_pred"], m_bool)
    # 连续切片 = **排除联合占位行**（三目标同时为原子值）后的行（E1/P1 §7 的 CONTIN 口径）
    cont_only_rows = ~np.asarray(a_bool).all(axis=1)
    cont_slice = M.score_of(oof["y_true"][cont_only_rows], oof["y_pred"][cont_only_rows],
                            oof["mask"][cont_only_rows])

    fold_rows = []
    for r in results:
        va = r["va"]
        yt, mt = labels_of(va), mask_of(va)
        pk = M.atom_gate(M.decode_continuous(r["pred"]), r["pred"]["q_atom"], r["tau"]["tau"])
        s = score_arrays(yt, pk, missing=~mt.astype(bool), missing_mode=C.SCORE_MISSING_MODE)
        sc = score_arrays(yt, np.tile(CONST_TILE, (yt.shape[0], 1)), missing=~mt.astype(bool),
                          missing_mode=C.SCORE_MISSING_MODE)
        fold_rows.append({"fold": r["fold"], "n_rows": int(va.n_rows),
                          "total": float(s["total"]), "por": float(s["acc_por"]),
                          "perm": float(s["acc_perm"]), "sw": float(s["acc_sw"]),
                          "const_total": float(sc["total"]),
                          "delta_vs_const": float(s["total"] - sc["total"]),
                          "tau": [float(x) for x in r["tau"]["tau"]],
                          "best_epoch": int(r["best_epoch"]),
                          "inner_oof_total": r["inner_oof_total"],
                          "seconds": round(float(r["seconds"]), 2)})

    const_arrays = np.tile(CONST_TILE, (oof["y_true"].shape[0], 1))
    well_tot, well_rows = per_well_totals(oof["y_true"], oof["y_pred"], m_bool,
                                          oof["well_index"], len(well_ids))
    well_tot_const, _ = per_well_totals(oof["y_true"], const_arrays, m_bool,
                                        oof["well_index"], len(well_ids))
    delta_well = well_tot - well_tot_const
    boot = FOLDS.bootstrap_ci(delta_well, iters=int(prereg["bootstrap_iters"]),
                              weights=well_rows, seed=cfg.seed)
    folds_all_same_direction = all(f["delta_vs_const"] > 0 for f in fold_rows)

    metrics = {
        "stage": "E1", "candidate_id": "E1_PD0",
        "n_folds": len(results), "n_wells": len(well_ids), "n_rows": int(oof["y_true"].shape[0]),
        "scoring": {"missing_mode": C.SCORE_MISSING_MODE,
                    "const_baseline": C.CONSTANT_BASELINE_OOF},
        "oof_cont_only": cont_score, "oof_gated": gated_score,
        "oof_total": float(gated_score["total"]),
        "delta_vs_const": float(gated_score["total"] - C.CONSTANT_BASELINE_OOF),
        "folds": fold_rows, "folds_all_same_direction": bool(folds_all_same_direction),
        "cont_slice": cont_slice, "atom_slices": atom_slices,
        "placeholder_rows": atom_rows, "atomic_head": head,
        "paired_bootstrap": boot,
        "scalers": {str(r["fold"]): dict(r["target"]) for r in results},
        "scalers_fitted_on": "train_fold_only",
        "tau_selected_on": "inner_oof_official_total", "no_interpolation": True,
        "train_strategy": {"two_stage": True,
                           "stage1": "inner 折选 best_epoch + tau（真实 score.py）",
                           "stage2": "全部 outer-train 重训 best_epoch，outer-val 只推理一次"},
        "cache": cache_info, "config": cfg.as_dict(),
        "oof_path": str(oof_path), "oof_sha256": sha256_file(oof_path),
        "seconds_total": round(time.time() - t_start, 2),
    }
    write_json(reports / "E1_metrics.json", metrics)

    with (reports / "E1_loss_curve.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["fold", "phase", "epoch", "seconds", "lr", "lam1", "loss_total", "loss_align",
                "loss_aux", "loss_joint", "loss_atom", "val_total_cont", "val_por", "val_perm",
                "val_sw", "disk_free_gb"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in results:
            for row in r["curve"]:
                wr.writerow({k: row.get(k) for k in cols})

    tr0 = RD.assemble(results[0]["tr_wells"], cache, scaler=results[0]["scaler"], with_targets=True)
    feat_rep = RD.feature_report(tr0, results[0]["scaler"])
    feat_rep.update({"cache": cache_info, "per_fold_scalers": metrics["scalers"],
                     "n_features": int(F.N_FEATURES), "feature_names": list(F.FEATURE_NAMES),
                     "inner": {"n_inner_folds": C.N_INNER_FOLDS,
                               "stage1_train_wells": len(results[0]["inner_tr_wells"]),
                               "stage1_val_wells": len(results[0]["inner_val_wells"])}})
    write_json(reports / "E1_row_features.json", feat_rep)
    del tr0

    # ---- Gate
    time_log = tracker.write(reports / "training_time_log.json", config=cfg.as_dict())
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(cfg.disk_path)
    except Exception as exc:                                   # 不静默：记录原因
        disk = {"level": "unknown", "error": str(exc)}
    contract = oof_contract(oof)
    min_ph = M.placeholder_min_acc({"atomic_rows": atom_rows})
    checks = {
        "contract_ok": bool(contract["ok"]),
        # "reported" = 指标字段真实存在；NaN 可以表示"该折无正例"，
        # 不能像 H5 那样直接写 True。
        "atomic_precision_reported": bool(head) and all(
            isinstance(v, dict) and "precision" in v and "recall" in v
            for v in head.values()),
        "disk_budget_ok": bool((disk or {}).get("level") == "ok"),
        "training_time_log_valid": bool(json.loads(time_log.read_text(encoding="utf-8"))["valid"]),
        "checkpoint_resumable": bool(results and results[0]["resumable"]["ok"]),
        "no_label_leak": bool(all(set(r["tr_wells"]).isdisjoint(set(r["va_wells"]))
                                  and set(r["fit_wells"]).issubset(set(r["tr_wells"]))
                                  and set(r["inner_val_wells"]).issubset(set(r["tr_wells"]))
                                  for r in results)),
    }
    result = {"checks": checks, "score": float(gated_score["total"]),
              "oof_total": float(gated_score["total"]),
              "delta": float(gated_score["total"] - C.CONSTANT_BASELINE_OOF),
              "paired_ci_low": float(boot["ci_low"]),
              "por_acc": float(gated_score["acc_por"]),
              "perm_acc": float(gated_score["acc_perm"]),
              "sw_acc": float(gated_score["acc_sw"]),
              "atomic_precision": float(np.nanmean([v["precision"] for v in head.values()])),
              "atomic_recall": float(np.nanmean([v["recall"] for v in head.values()]))}
    agg = G.aggregate_gate(prereg, result)
    gate = {"gate_id": prereg["gate_id"], "stage": "E1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "passed": bool(agg["passed"]), "gate_type": agg["details"].get("gate_type"),
            "primary_metric": prereg["primary_metric"],
            "oof_total": result["oof_total"], "delta_vs_const": result["delta"],
            "paired_ci_low": result["paired_ci_low"],
            "folds_all_same_direction": bool(folds_all_same_direction),
            "thresholds": prereg["thresholds"], "checks": checks,
            "placeholder_min_acc": min_ph, "placeholder_min_acc_required": 0.98,
            "placeholder_ok": bool(min_ph == min_ph and min_ph >= 0.98),
            "atomic_slices": atom_slices, "cont_slice_total": cont_slice["total"],
            "folds": fold_rows, "oof_path": str(oof_path), "oof_sha256": metrics["oof_sha256"],
            "metrics_path": str(reports / "E1_metrics.json"),
            "prereg_path": str(prereg_path), "prereg_sha256": sha256_file(prereg_path),
            "training_time_log": str(time_log), "disk": disk, "contract": contract,
            "aggregate": agg, "smoke": bool(args.smoke)}
    gate_path = reports / "E1_gate.json"
    write_json(gate_path, gate)

    # ---- 候选登记（schema 见 versions/candidates.json::fields）
    # 冒烟运行（--smoke）**不登记候选**：它不产出任何 Gate 数值。
    # `--candidates` 指向不存在的路径时用空骨架初始化（测试指向 tmp，绝不写仓库）。
    cand_path = Path(args.candidates)
    if cand_path.is_file():
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
    else:
        cand = {"schema_version": 1, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "note": "由 E1/code/train_row.py 创建（--candidates 指到仓库外时的空骨架）",
                "candidates": []}
    entry = {
        "candidate_id": "E1_PD0", "stage": "E1", "base": "CONST", "arch": "RowMLP",
        "loss": "score_aligned_v1", "feature_version": "F1",
        "decode": "atom_hard_switch_inner_oof_tau",
        "checkpoint": str(run_dir / "fold0" / "best.pt"),
        "result_zip": None, "result_zip_sha256": None,
        "cv": {"total": float(gated_score["total"]), "por": float(gated_score["acc_por"]),
               "perm": float(gated_score["acc_perm"]), "sw": float(gated_score["acc_sw"]),
               "fold_deltas": [f["delta_vs_const"] for f in fold_rows],
               "paired_bootstrap_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
               "missing_mode": C.SCORE_MISSING_MODE, "selection_score_only": True},
        "atomic": {"por_acc": atom_rows["hit_rate"]["POR"], "perm_acc": atom_rows["hit_rate"]["PERM"],
                   "sw_acc": atom_rows["hit_rate"]["SW"],
                   "por_precision": head["POR"]["precision"],
                   "perm_precision": head["PERM"]["precision"],
                   "sw_precision": head["SW"]["precision"],
                   "por_recall": head["POR"]["recall"], "perm_recall": head["PERM"]["recall"],
                   "sw_recall": head["SW"]["recall"],
                   "por_f1": head["POR"]["f1"], "perm_f1": head["PERM"]["f1"],
                   "sw_f1": head["SW"]["f1"],
                   "joint_atom_acc": None, "joint_atom_auc": None, "joint_atom_ap": None,
                   "tau": [float(np.mean([r["tau"]["tau"][t] for r in results])) for t in range(3)],
                   "tau_selected_on": "inner_oof_official_total",
                   "tau_plateau_midpoint": True,
                   "joint_guard": {"enabled": False, "tau_high": None, "inner_oof_delta": None,
                                   "ci_low": None},
                   "no_interpolation": True},
        "a_board_score": None, "a_board_delta_vs_b0": None,
        "status": "local_only", "parent": None,
        "notes": f"E1 行级基线（{len(results)} 折 OOF）；placeholder_min_acc={min_ph:.4f}；"
                 f"gate_passed={gate['passed']}；folds_same_direction={folds_all_same_direction}",
        "scalers": {**{k: float(results[0]["target"][k]) for k in
                       ("por_max", "por_median", "sw_mu", "sw_sigma", "perm_z_median",
                        "s_por", "s_sw")},
                    "fitted_on": "train_fold_only"},
        "train_strategy": {"two_stage": True,
                           "stage1": "inner 折选 best_epoch + tau",
                           "stage2": f"outer-train 重训 best_epoch（fold0={results[0]['best_epoch']}）",
                           "ema": {"enabled": False, "decay": None},
                           "swa": {"enabled": False},
                           "snapshot_topk": {"k": 0, "fusion_on": "inner_oof"}},
    }
    cand["candidates"] = [c for c in cand.get("candidates", [])
                          if c.get("candidate_id") != "E1_PD0"] + [entry]
    if not args.smoke:
        write_json(cand_path, cand)
        print(f"[E1] candidate E1_PD0 registered -> {cand_path}", flush=True)

    print(json.dumps({"gate_passed": gate["passed"], "oof_total": gate["oof_total"],
                      "delta_vs_const": gate["delta_vs_const"],
                      "paired_ci_low": gate["paired_ci_low"], "checks": checks,
                      "folds_all_same_direction": folds_all_same_direction,
                      "placeholder_min_acc": min_ph, "smoke": bool(args.smoke),
                      "folds": [{"fold": f["fold"], "total": round(f["total"], 4),
                                 "delta": round(f["delta_vs_const"], 4)} for f in fold_rows],
                      "report": str(gate_path)}, ensure_ascii=False, indent=2))
    if args.smoke:
        return 0
    return 0 if gate["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
