"""按井折协议的读取、指纹与 inner 折切分（E0 冻结）。

主判据折 = `v1_well_folds.json`（80 井按井 5 折），必须**逐字节复用**，
使 v4 的 OOF 与历史锚点（B0=80.382479 / C1W=80.180944）同折可比。

该文件结构（E0 实测）：
    {"seed":42, "n_folds":5, "well_list":[...80...], "fold_of_well":{well:fold},
     "fold_mapping":{...}, "hash":"..."}

本模块只依赖标准库 + numpy（可选）。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .. import constants as C
from ..portability import HAS_NUMPY


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_folds_file(v4_root: str | Path | None = None) -> Path:
    """按优先级定位折文件。

    1. `$V4_DATA_ROOT/v4/data/folds/v1_well_folds.json`（云端 /data 挂载）
    2. `<v4>/versions/reference/v1_well_folds.json`（**随 git 仓库冻结的分片**，自包含）
    3. `../v1/src/well_folds.json`、`../v2/versions/reference/v1_well_folds.json`（本机）
    """
    root = Path(v4_root) if v4_root else Path(__file__).resolve().parents[2]
    env = os.environ.get("V4_DATA_ROOT")
    cands: list[Path] = []
    if env:
        cands.append(Path(env) / "v4" / "data" / "folds" / C.WELL_FOLDS_SOURCE)
    cands += [
        root / "versions" / "reference" / C.WELL_FOLDS_SOURCE,
        root.parent / "v1" / "src" / C.WELL_FOLDS_SOURCE,
        root.parent / "v2" / "versions" / "reference" / C.WELL_FOLDS_SOURCE,
    ]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "找不到按井折文件。已尝试：\n  " + "\n  ".join(str(c) for c in cands)
        + "\n请运行 `bash v4/tools/bootstrap_data.sh` 或设置 V4_DATA_ROOT。"
    )


def load_folds(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else find_folds_file()
    raw = json.loads(p.read_text(encoding="utf-8"))
    well_list = list(raw["well_list"])
    fold_of = {w: int(raw["fold_of_well"][w]) for w in well_list}
    n_folds = int(raw.get("n_folds", C.N_FOLDS))
    return {
        "source_path": str(p),
        "source_sha256": sha256_file(p),
        "seed": raw.get("seed"),
        "n_folds": n_folds,
        "well_list": well_list,
        "fold_of_well": fold_of,
        "raw_hash": raw.get("hash"),
    }


def make_inner_folds(wells: list[str], n_inner: int = C.N_INNER_FOLDS,
                     seed: int = 42, n_folds: int | None = None) -> dict[str, list[str]]:
    """把 outer-train 的井切成 n_inner 个 inner 折（井维度，确定性）。

    排序后按 `hash(seed, well)` 做稳定散列分配，保证同一 outer 折内可复算。
    """
    k = n_folds or n_inner
    buckets: list[list[str]] = [[] for _ in range(k)]
    for w in sorted(wells):
        h = hashlib.sha256(f"{seed}:{w}".encode()).hexdigest()
        buckets[int(h[:8], 16) % k].append(w)
    return {f"inner{i}": b for i, b in enumerate(buckets)}


def fold_manifest(folds: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_path": folds["source_path"],
        "source_sha256": folds["source_sha256"],
        "seed": folds["seed"],
        "n_folds": folds["n_folds"],
        "n_wells": len(folds["well_list"]),
        "fold_sizes": {
            str(f): sum(1 for v in folds["fold_of_well"].values() if v == f)
            for f in range(folds["n_folds"])
        },
    }


def bootstrap_ci(values: Any, iters: int = 1000, alpha: float = 0.05,
                 weights: Any = None, seed: int = 42) -> dict[str, float]:
    """按井的非参数 cluster bootstrap（可选行数加权）。

    values : (n_wells,) 每口井的 delta（或分数）
    weights: (n_wells,) 井行数（若给出则做加权点估计）
    """
    if not HAS_NUMPY:
        raise RuntimeError("bootstrap_ci requires numpy")
    import numpy as np  # noqa: PLC0415

    v = np.asarray(values, dtype="float64")
    w = None if weights is None else np.asarray(weights, dtype="float64")
    rng = np.random.default_rng(seed)
    n = v.shape[0]
    stats = np.empty(iters, dtype="float64")
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        if w is None:
            stats[i] = v[idx].mean()
        else:
            ww = w[idx]
            stats[i] = float((v[idx] * ww).sum() / max(ww.sum(), 1e-12))
    point = float(v.mean()) if w is None else float((v * w).sum() / max(w.sum(), 1e-12))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_units": int(n),
        "iters": int(iters),
        "weighted": w is not None,
    }
