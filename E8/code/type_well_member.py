#!/usr/bin/env python3
"""WP8：类型井选择 + 井间自适应的一阶成员（参考 SPWLA 2021 冠军 UTFE）。

流程（逐 outer 折、逐验证井）：
  1. 只用**训练井输入**为验证井选 top-k 类型井（KL / DTW / signature）；
  2. 在类型井子集上 fit scaler + 训练 GBDT/链式连续模型（不碰验证标签）；
  3. 把验证井的**输入曲线**用线性/分位匹配到 top-1 类型井（只使用输入，不涉及标签）；
  4. 用同一 scaler 变换适配后的验证特征并预测。

产出：
  ``$RUN_ROOT/E8/oof_typewell_{kind}.npz``（与 E8 集成兼容）
  ``$REPORTS/E8_type_well_member_{kind}.json``

向后兼容：本脚本是独立新增入口，不修改任何既有训练脚本；默认只有显式调用才运行。
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
from src.data import dataset as D  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.data import type_well as TW  # noqa: E402
from src.data import well_adapt as WA  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.score import score_arrays  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402


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
    ap = argparse.ArgumentParser(description="WP8 类型井 + 井间自适应成员")
    ap.add_argument("--kind", default="gbdt", choices=("gbdt", "chained"))
    ap.add_argument("--gbdt-kind", default="histgb",
                    choices=("histgb", "lgbm", "xgboost", "catboost"))
    ap.add_argument("--estimator-kind", default="histgb",
                    choices=("histgb", "lgbm", "ridge"))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--select-method", default="kl", choices=("kl", "dtw", "signature"))
    ap.add_argument("--adapt-method", default="quantile",
                    choices=("none", "linear", "quantile"))
    ap.add_argument("--adapt-alpha", type=float, default=1.0)
    ap.add_argument("--fallback-all-train", action="store_true", default=True,
                    help="类型井不足/选择失败时回退到全部训练井（默认开）")
    ap.add_argument("--no-fallback-all-train", dest="fallback_all_train",
                    action="store_false")
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
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def _train_on_wells(kind: str, args, wells, cache, spec, fit):
    scaler = fit["scaler"]
    tr_t = RD.assemble(list(wells), cache, scaler=scaler, with_targets=True,
                       spec=spec, phys_params=fit.get("phys_params"))
    X = np.asarray(tr_t.X, dtype="float64")
    Y = M.label_scale_stack(tr_t.y_por, tr_t.y_perm_z, tr_t.y_sw)
    m = np.asarray(tr_t.mask, dtype="float64")
    if kind == "gbdt":
        from src.models.gbdt import MultiTargetGBDT
        model = MultiTargetGBDT(kind=args.gbdt_kind, params={})
        model.fit(X, Y, mask=m)
    else:
        from src.training.chained import ChainedRegressor
        est = args.estimator_kind
        if est == "histgb":
            try:
                import sklearn  # noqa: F401
            except Exception:
                est = "ridge"
        model = ChainedRegressor(estimator_kind=est, n_splits=5, seed=args.seed,
                                 estimator_params=({"alpha": 1.0} if est == "ridge" else {}))
        model.fit(X, Y, mask=m)
    return model, scaler


def _adapt_val_features(args, cache, spec, fit, ref_well, val_well):
    """把 val_well 输入曲线适配到 ref_well，再构建特征并 scaler.transform。"""
    va_sh = D.read_well_shard(cache, val_well, "train")
    ref_sh = D.read_well_shard(cache, ref_well, "train")
    if args.adapt_method == "none":
        adapted = {"inputs": np.asarray(va_sh["inputs"], dtype="float64"),
                   "missing": np.asarray(va_sh["missing"], dtype="int8")}
    else:
        adapted = WA.adapt_shard_inputs(va_sh, ref_sh, method=args.adapt_method,
                                        alpha=args.adapt_alpha)
    shard = {"inputs": adapted["inputs"], "missing": adapted["missing"],
             "depth": np.asarray(va_sh["depth"], dtype="float64")}
    Xraw, _names = G.build_matrix(shard, spec, phys_params=fit.get("phys_params"))
    return fit["scaler"].transform(Xraw)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.tag = args.tag or args.kind
    if args.smoke:
        args.topk = min(args.topk, 3)
        args.max_wells = args.max_wells or 12
    reports, run_dir = Path(args.reports_dir), Path(args.run_root) / "E8"
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = (list(range(int(folds["n_folds"]))) if args.folds == "all"
                 else [int(x) for x in str(args.folds).split(",") if x.strip()])
    acc = {k: [] for k in ("cont", "q_atom", "y_true", "mask", "y_atom",
                           "y_pred", "well_index", "fold_of_row", "depth")}
    well_ids: list[str] = []
    offset = 0
    per_fold: list[dict] = []
    selections: dict[str, dict] = {}
    t0 = time.time()
    for k in fold_list:
        tr, va = RD.fold_wells(folds, k)
        if args.max_wells:
            tr, va = tr[:args.max_wells], va[:args.max_wells]
        train_shards = {w: D.read_well_shard(args.cache_root, w, "train") for w in tr}
        val_shards = {w: D.read_well_shard(args.cache_root, w, "train") for w in va}
        sel = TW.select_type_wells_batch(val_shards, train_shards, topk=args.topk,
                                         method=args.select_method)
        fold_preds, fold_ys, fold_ms, fold_ya, fold_depth, fold_widx =             [], [], [], [], [], []
        for w in va:
            cands = sel.get(w, {}).get("candidates") or []
            type_wells = [c["well"] for c in cands if c.get("well") in train_shards]
            fallback = False
            if not type_wells:
                if not args.fallback_all_train:
                    raise RuntimeError(f"fold{k} val={w}: 无类型井且 fallback 关闭")
                type_wells = list(tr)
                fallback = True
            fit = RD.fit_scalers_from_wells(type_wells, args.cache_root, spec=spec)
            model, _scaler = _train_on_wells(args.kind, args, type_wells,
                                             args.cache_root, spec, fit)
            ref = type_wells[0] if not fallback else type_wells[0]
            Xva = _adapt_val_features(args, args.cache_root, spec, fit, ref, w)
            cont = np.asarray(model.predict(Xva), dtype="float64")
            sh = val_shards[w]
            # 标签只用于 OOF 记录，不参与类型井选择/适配
            from src.features.basic import build_labels
            lab = build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
            yt = M.label_scale_stack(lab["por"], lab["perm_z"], lab["sw"])
            mask = np.asarray(lab["mask"], dtype="float64")
            fold_preds.append(cont)
            fold_ys.append(yt)
            fold_ms.append(mask)
            fold_ya.append(np.asarray(lab["y_atom"], dtype="float64"))
            fold_depth.append(np.asarray(sh["depth"], dtype="float64"))
            fold_widx.append(np.full(cont.shape[0], offset + len(fold_widx),
                                     dtype="int64"))
            selections[f"{k}:{w}"] = {"type_wells": type_wells, "fallback": fallback,
                                      "candidates": cands,
                                      "adapt_method": args.adapt_method,
                                      "adapt_alpha": args.adapt_alpha}
        cont = np.concatenate(fold_preds, axis=0) if fold_preds else np.zeros((0, 3))
        yt = np.concatenate(fold_ys, axis=0) if fold_ys else np.zeros((0, 3))
        mask = np.concatenate(fold_ms, axis=0) if fold_ms else np.zeros((0, 3))
        acc["cont"].append(cont)
        acc["q_atom"].append(np.full_like(cont, 0.5))
        acc["y_true"].append(yt)
        acc["mask"].append(mask)
        acc["y_atom"].append(np.concatenate(fold_ya, axis=0) if fold_ya else np.zeros((0, 3)))
        acc["y_pred"].append(cont)
        acc["well_index"].append(np.concatenate(fold_widx) if fold_widx
                                 else np.zeros(0, dtype="int64"))
        acc["fold_of_row"].append(np.full(yt.shape[0], int(k), dtype="int32"))
        acc["depth"].append(np.concatenate(fold_depth, axis=0) if fold_depth else np.zeros(0))
        offset += len(va)
        well_ids += list(va)
        sc = score_arrays(yt, cont, missing=~mask.astype(bool),
                          missing_mode=C.SCORE_MISSING_MODE)
        per_fold.append({"fold": int(k), "total": float(sc["total"]),
                         "por": float(sc["acc_por"]), "perm": float(sc["acc_perm"]),
                         "sw": float(sc["acc_sw"]), "n_wells": len(va)})
        print(f"[E8/typewell-{args.tag}] fold{k} total={sc['total']:.4f}", flush=True)
    out = {key: np.concatenate(val, axis=0) if val else np.zeros((0, 3))
           for key, val in acc.items()}
    out["well_ids"] = np.asarray(well_ids, dtype=object)
    out["n_wells"] = len(well_ids)
    out["spec_key"] = spec.key
    oof_path = run_dir / f"oof_typewell_{args.tag}.npz"
    np.savez_compressed(oof_path, **out)
    sc_all = score_arrays(out["y_true"], out["cont"], missing=~out["mask"].astype(bool),
                          missing_mode=C.SCORE_MISSING_MODE)
    report = {"stage": "E8", "p_stage": "WP8/type-well-member", "tag": args.tag,
              "kind": args.kind, "topk": args.topk, "select_method": args.select_method,
              "adapt_method": args.adapt_method, "adapt_alpha": args.adapt_alpha,
              "spec": spec.as_dict(), "folds": fold_list, "seed": args.seed,
              "scientific": not (args.smoke or args.exploratory),
              "oof_total": float(sc_all["total"]), "oof_por": float(sc_all["acc_por"]),
              "oof_perm": float(sc_all["acc_perm"]), "oof_sw": float(sc_all["acc_sw"]),
              "n_rows": int(out["y_true"].shape[0]), "n_wells": int(out["n_wells"]),
              "oof_path": str(oof_path), "per_fold": per_fold,
              "selections": selections, "seconds": round(time.time() - t0, 2),
              "note": ("类型井只由训练井输入选择；验证井输入适配不涉及任何标签。"
                       "q_atom=0.5 仅占位；E8 融合时原子切换以其他成员 q_atom 为准。")}
    rep_path = reports / f"E8_type_well_member_{args.tag}.json"
    write_json(rep_path, report)
    print(json.dumps({"tag": args.tag, "oof_total": report["oof_total"],
                      "oof_path": str(oof_path), "report": str(rep_path)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
