#!/usr/bin/env python3
"""WP10：扩展岩石物理特征报告（只读输入，不训练模型）。

产出 `$REPORTS/E2_petro_features.json`：PetroParams（训练折拟合）、18 列扩展特征的
min/max/mean/std、缺失率、与标签的粗相关；并保存一份 512 行的样例矩阵，便于后续
接入 FeatureSpec。
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
from src.features import physics_ext as PE  # noqa: E402
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WP10 扩展岩石物理特征报告")
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    reports, run_dir = Path(args.reports_dir), Path(args.run_root) / "E2" / "petro"
    reports.mkdir(parents=True, exist_ok=True); run_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_root)
    folds = FOLDS.load_folds()
    wells = list(folds["well_list"])
    if args.smoke:
        wells = wells[:8]
    if args.max_wells:
        wells = wells[:args.max_wells]
    Xs, ms, ys = [], [], []
    t0 = time.time()
    for w in wells:
        sh = D.read_well_shard(cache, w, "train")
        X = np.asarray(sh["inputs"], dtype="float64")
        m = np.asarray(sh.get("missing"), dtype="bool")
        F, names = PE.build_petro_features(X, m)
        Xs.append(F)
        ms.append(m)
        # 标签尺度
        from src.features.basic import build_labels
        lab = build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
        ys.append(np.stack([lab["por"], np.power(10.0, lab["perm_z"]), lab["sw"]], axis=1))
        if sum(x.shape[0] for x in Xs) >= int(args.max_rows):
            break
    F = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(PE.PETRO_FEATURES)))
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0, 3))
    if F.shape[0] > int(args.max_rows):
        idx = np.linspace(0, F.shape[0] - 1, int(args.max_rows)).astype(int)
        F, y = F[idx], y[idx]
    params = PE.fit_petro_params(np.concatenate([np.asarray(D.read_well_shard(cache, w, "train")["inputs"],
                                                           dtype="float64")
                                                 for w in wells[:max(1, min(len(wells), 8))]], axis=0))
    stats = {}
    for j, name in enumerate(PE.PETRO_FEATURES):
        col = F[:, j]
        finite = np.isfinite(col)
        corr = {}
        for t, tname in enumerate(C.TARGETS):
            v = np.isfinite(col) & np.isfinite(y[:, t])
            try:
                corr[tname] = (None if v.sum() < 3 else float(np.corrcoef(col[v], y[v, t])[0, 1]))
            except Exception:
                corr[tname] = None
        stats[name] = {"missing_rate": float(1.0 - finite.mean()) if col.size else None,
                       "min": (None if not finite.any() else float(np.nanmin(col))),
                       "max": (None if not finite.any() else float(np.nanmax(col))),
                       "mean": (None if not finite.any() else float(np.nanmean(col))),
                       "std": (None if not finite.any() else float(np.nanstd(col))),
                       "corr": corr}
    sample_path = run_dir / "petro_sample.npz"
    np.savez_compressed(sample_path, X=F[:512], y=y[:512],
                        feature_names=np.asarray(PE.PETRO_FEATURES, dtype=object))
    report = {"stage": "E2", "p_stage": "WP10/petro",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "n_wells": len(wells), "n_rows": int(F.shape[0]),
              "params": params.as_dict(), "features": list(PE.PETRO_FEATURES),
              "stats": stats, "sample_path": str(sample_path),
              "seconds": round(time.time() - t0, 2),
              "note": "参数只在训练折/全体训练井上拟合；接入 FeatureSpec 前必须做 inner-OOF 消融。"}
    path = reports / "E2_petro_features.json"
    write_json(path, report)
    print(json.dumps({"report": str(path), "features": len(PE.PETRO_FEATURES),
                      "n_rows": int(F.shape[0])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
