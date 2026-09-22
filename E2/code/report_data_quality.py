#!/usr/bin/env python3
"""WP9：数据质量报告（缺失/插补/异常/代表采样），只读输入，不训练标签模型。

产出 `$REPORTS/E2_data_quality.json`：
  * 逐曲线缺失率、逐行缺失数分布；
  * median / KNN / MICE 三种插补在样本上的耗时与缺失率；
  * IQR / IsolationForest 异常权重摘要（若有 sklearn）；
  * Kennard-Stone 选出的代表 inner-val 井（只用于模型选择，不改变 outer 折）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.data import dataset as D  # noqa: E402
from src.data import impute as IM  # noqa: E402
from src.data import outliers as OUT  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import representative as REP  # noqa: E402


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WP9 数据质量报告")
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    reports = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_root)
    folds = FOLDS.load_folds()
    wells = list(folds["well_list"])
    if args.max_wells:
        wells = wells[:args.max_wells]
    if args.smoke:
        wells = wells[:8]
    t0 = time.time()
    miss_counts = np.zeros(len(C.INPUT_COLUMNS), dtype="int64")
    total_rows = 0
    per_well_sign = {}
    Xs = []
    for w in wells:
        sh = D.read_well_shard(cache, w, "train")
        X = np.asarray(sh["inputs"], dtype="float64")
        miss = ~np.isfinite(X)
        miss_counts += miss.sum(axis=0)
        total_rows += X.shape[0]
        per_well_sign[w] = np.concatenate([np.nanmean(X, axis=0), np.nanstd(X, axis=0)])
        if sum(x.shape[0] for x in Xs) < int(args.max_rows):
            Xs.append(X)
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(C.INPUT_COLUMNS)))
    if X.shape[0] > int(args.max_rows):
        rng = np.random.default_rng(args.seed)
        X = X[rng.choice(X.shape[0], size=int(args.max_rows), replace=False)]
    missing_report = {c: {"missing": int(miss_counts[i]),
                          "total": int(total_rows),
                          "rate": float(miss_counts[i] / max(total_rows, 1))}
                      for i, c in enumerate(C.INPUT_COLUMNS)}
    impute = {}
    for method, kw in (("median", {}), ("knn", {"k": 5}),
                       ("mice", {"max_iter": 5, "seed": args.seed})):
        t = time.time()
        try:
            res = IM.impute_matrix(X, method=method, **kw)
            impute[method] = {"method": res.get("method"), "seconds": round(time.time() - t, 3),
                              "n_missing_before": int((~np.isfinite(X)).sum()),
                              "n_missing_after": int((~np.isfinite(res["X"])).sum())}
        except Exception as exc:
            impute[method] = {"error": f"{type(exc).__name__}: {exc}"}
    outlier = {}
    for method, kw in (("iqr", {}), ("iforest", {"contamination": 0.05, "seed": args.seed})):
        try:
            w, rep = OUT.sample_weights_from_outliers(X, method=method, **kw)
            outlier[method] = rep
        except Exception as exc:
            outlier[method] = {"error": f"{type(exc).__name__}: {exc}"}
    ks_tr = ks_va = []
    if len(wells) >= 5:
        try:
            ks_tr, ks_va = REP.representative_inner_split(wells, per_well_sign, n_val=max(2, len(wells)//5),
                                                          seed=args.seed)
        except Exception as exc:
            ks_tr, ks_va = [], []
            outlier["ks_error"] = f"{type(exc).__name__}: {exc}"
    report = {"stage": "E2", "p_stage": "WP9/data-quality", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "n_wells": len(wells), "n_rows_sampled": int(X.shape[0]),
              "curves": list(C.INPUT_COLUMNS), "missing": missing_report,
              "impute": impute, "outliers": outlier,
              "ks_inner_val_wells": ks_va, "ks_inner_train_wells": ks_tr,
              "seconds": round(time.time() - t0, 2),
              "note": "只读输入曲线；插补/异常/KS 参数只允许在训练折上拟合。"}
    path = reports / "E2_data_quality.json"
    write_json(path, report)
    print(json.dumps({"report": str(path), "n_wells": len(wells),
                      "n_rows_sampled": int(X.shape[0])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
