#!/usr/bin/env python3
"""WP11：表格/链式一阶成员（GBDT / Chained）——参考 SPWLA 2021 第 2/4/5 名。

产出与 E8 集成兼容的 OOF：

    $RUN_ROOT/E8/oof_{tag}.npz
        cont, q_atom, y_true, mask, y_atom, y_pred, well_index,
        fold_of_row, depth, well_ids, n_wells, spec_key

其中 ``cont`` 是标签尺度连续预测；GBDT/链式成员是**连续成员**，q_atom 固定 0.5
（只作为占位，不参与原子硬切换；E8 融合时以其他成员的 q_atom 为准）。

外部依赖：scikit-learn（HistGB）或 lightgbm/xgboost/catboost；缺失时显式失败，
不静默假装完成。链式模型在没有 sklearn 时自动降级为 numpy Ridge。
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

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.score import score_arrays  # noqa: E402


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WP11 表格/链式一阶成员")
    ap.add_argument("--kind", default="gbdt", choices=("gbdt", "chained"))
    ap.add_argument("--gbdt-kind", default="histgb",
                    choices=("histgb", "lgbm", "xgboost", "catboost"))
    ap.add_argument("--estimator-kind", default="histgb",
                    choices=("histgb", "lgbm", "ridge"),
                    help="链式模型每步使用的估计器")
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--spec", default="F1")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help=">0 时只用前 (1-frac) 井训练、后 frac 井做内部快速验证（smoke 用）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def _check_deps(kind: str) -> tuple[bool, str]:
    """返回 (sklearn_available, message)。"""
    try:
        import sklearn  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, f"scikit-learn 不可用：{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.max_wells = args.max_wells or 12
    reports, run_dir = Path(args.reports_dir), Path(args.run_root) / "E8"
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.kind
    sklearn_ok, sk_msg = _check_deps(args.kind)
    if args.kind == "gbdt" and not sklearn_ok and args.gbdt_kind == "histgb":
        print(f"[E8/tabular] FATAL: {sk_msg}；无法训练 HistGB。"
              "请安装 scikit-learn 或改 --gbdt-kind lgbm/xgboost/catboost。", file=sys.stderr)
        return 5
    if args.kind == "chained" and not sklearn_ok and args.estimator_kind == "histgb":
        print(f"[E8/tabular] WARN: {sk_msg}；链式模型降级为 numpy Ridge", file=sys.stderr)
        args.estimator_kind = "ridge"

    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = (list(range(int(folds["n_folds"]))) if args.folds == "all"
                 else [int(x) for x in str(args.folds).split(",") if x.strip()])

    acc = {k: [] for k in ("cont", "q_atom", "y_true", "mask", "y_atom",
                           "y_pred", "well_index", "fold_of_row", "depth")}
    well_ids: list[str] = []
    offset = 0
    t0 = time.time()
    per_fold = []
    for k in fold_list:
        tr, va = RD.fold_wells(folds, k)
        if args.max_wells:
            tr, va = tr[:args.max_wells], va[:args.max_wells]
        fit = RD.fit_scalers_from_wells(tr, args.cache_root, spec=spec)
        scaler, phys = fit["scaler"], fit.get("phys_params")
        tr_t = RD.assemble(tr, args.cache_root, scaler=scaler, with_targets=True,
                           spec=spec, phys_params=phys)
        va_t = RD.assemble(va, args.cache_root, scaler=scaler, with_targets=True,
                           spec=spec, phys_params=phys)
        Xtr = np.asarray(tr_t.X, dtype="float64")
        Xva = np.asarray(va_t.X, dtype="float64")
        Ytr = M.label_scale_stack(tr_t.y_por, tr_t.y_perm_z, tr_t.y_sw)
        mtr = np.asarray(tr_t.mask, dtype="float64")
        if args.kind == "gbdt":
            from src.models.gbdt import MultiTargetGBDT
            model = MultiTargetGBDT(kind=args.gbdt_kind, params={})
            model.fit(Xtr, Ytr, mask=mtr)
        else:
            from src.training.chained import ChainedRegressor
            model = ChainedRegressor(estimator_kind=args.estimator_kind,
                                     n_splits=5, seed=args.seed,
                                     estimator_params=({"alpha": 1.0}
                                                       if args.estimator_kind == "ridge"
                                                       else {}))
            model.fit(Xtr, Ytr, mask=mtr)
        cont_va = np.asarray(model.predict(Xva), dtype="float64")
        # 连续成员：q_atom 只作为占位；E8 融合时会以其他成员 q_atom 为准
        q_va = np.full_like(cont_va, 0.5)
        yt = M.label_scale_stack(va_t.y_por, va_t.y_perm_z, va_t.y_sw)
        mask_va = np.asarray(va_t.mask, dtype="float64")
        acc["cont"].append(cont_va)
        acc["q_atom"].append(q_va)
        acc["y_true"].append(yt)
        acc["mask"].append(mask_va)
        acc["y_atom"].append(np.asarray(va_t.y_atom, dtype="float64"))
        acc["y_pred"].append(cont_va)
        acc["well_index"].append(np.asarray(va_t.well_index, dtype="int64") + offset)
        acc["fold_of_row"].append(np.full(yt.shape[0], int(k), dtype="int32"))
        acc["depth"].append(np.asarray(va_t.depth, dtype="float64"))
        offset += int(va_t.n_wells)
        well_ids += list(va_t.well_ids)
        sc = score_arrays(yt, cont_va, missing=~mask_va.astype(bool),
                          missing_mode=C.SCORE_MISSING_MODE)
        per_fold.append({"fold": int(k), "total": float(sc["total"]),
                         "por": float(sc["acc_por"]), "perm": float(sc["acc_perm"]),
                         "sw": float(sc["acc_sw"]), "n_rows": int(yt.shape[0]),
                         "n_wells": int(va_t.n_wells)})
        print(f"[E8/{tag}] fold{k} total={sc['total']:.4f} "
              f"por={sc['acc_por']:.4f} perm={sc['acc_perm']:.4f} sw={sc['acc_sw']:.4f}",
              flush=True)

    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["well_ids"] = np.asarray(well_ids, dtype=object)
    out["n_wells"] = len(well_ids)
    out["spec_key"] = spec.key
    oof_path = run_dir / f"oof_{tag}.npz"
    np.savez_compressed(oof_path, **out)
    sc_all = score_arrays(out["y_true"], out["cont"],
                          missing=~out["mask"].astype(bool),
                          missing_mode=C.SCORE_MISSING_MODE)
    report = {
        "stage": "E8", "p_stage": "WP11/tabular-member", "tag": tag,
        "kind": args.kind, "gbdt_kind": args.gbdt_kind if args.kind == "gbdt" else None,
        "estimator_kind": args.estimator_kind if args.kind == "chained" else None,
        "spec": spec.as_dict(), "folds": fold_list, "seed": args.seed,
        "scientific": not (args.smoke or args.exploratory),
        "oof_total": float(sc_all["total"]), "oof_por": float(sc_all["acc_por"]),
        "oof_perm": float(sc_all["acc_perm"]), "oof_sw": float(sc_all["acc_sw"]),
        "n_rows": int(out["y_true"].shape[0]), "n_wells": int(out["n_wells"]),
        "oof_path": str(oof_path), "per_fold": per_fold,
        "seconds": round(time.time() - t0, 2),
        "note": ("GBDT/链式是连续成员；q_atom=0.5 仅作占位，"
                 "E8 融合时原子切换以其他成员 q_atom 为准。"),
    }
    rep_path = reports / f"E8_tabular_{tag}.json"
    write_json(rep_path, report)
    print(json.dumps({"kind": args.kind, "tag": tag, "oof_total": report["oof_total"],
                      "oof_path": str(oof_path), "report": str(rep_path)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
