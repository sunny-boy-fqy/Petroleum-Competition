"""按井分片的张量缓存（内存纪律：不把全部井常驻内存）。

云端真实约束：**16 GiB 系统内存** + 80 GiB 显存 → 瓶颈永远在内存。
因此 v4 的数据访问统一走"按井分片落盘 + 按需读取"：

    cache/raw/<well_id>.npz     输入曲线 + 缺失指示 + 深度（float32）
    cache/labels/<well_id>.npz  三目标 + 缺测掩码（float32/int8）

本模块只依赖 numpy（不需要 torch），因此本机也能生成与校验缓存。

设计要点
--------
1. 分片内容固定为**规范 17 列对齐后**的数据（见 `parse.py` 的按表头名解析）；
2. 输入曲线保留 NaN（不填 0），由下游特征层决定如何编码；
3. 额外的缺失指示位由 `features/basic.py` 生成，不在分片里冗余存储；
4. 单井分片 ~0.5 MB，90 井合计 ~50 MB，远低于 30 GB 磁盘预算。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .. import constants as C
from ..portability import HAS_NUMPY
from . import labels as L
from . import parse as P

if HAS_NUMPY:
    import numpy as np


@dataclass
class ShardInfo:
    well_id: str
    split: str
    n_rows: int
    n_cols_in: int
    with_targets: bool


def shard_paths(cache_root: str | Path, well_id: str, split: str = "train") -> Path:
    d = Path(cache_root) / "raw" / split
    return d / f"{well_id}.npz"


def labels_path(cache_root: str | Path, well_id: str) -> Path:
    d = Path(cache_root) / "labels"
    return d / f"{well_id}.npz"


def write_well_shard(rec: P.WellRecord, cache_root: str | Path, split: str) -> ShardInfo:
    """把一口井写入分片（原子写：先写 tmp 再 rename）。"""
    if not HAS_NUMPY:
        raise RuntimeError("write_well_shard requires numpy")
    cache_root = Path(cache_root)
    sp = shard_paths(cache_root, rec.well_id, split)
    sp.parent.mkdir(parents=True, exist_ok=True)

    inputs = np.asarray(rec.inputs, dtype="float32")
    depth = np.asarray(rec.depth, dtype="float32")
    miss = (~np.isfinite(inputs)).astype("int8")

    tmp = sp.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        depth=depth,
        inputs=inputs,
        missing=miss,
        n_rows=np.int64(rec.n_rows),
        header=np.array(rec.header, dtype=object),
        missing_columns=np.array(rec.missing_columns, dtype=object),
        extra_columns=np.array(rec.extra_columns, dtype=object),
    )
    tmp.replace(sp)

    if rec.targets is not None:
        t = np.asarray(rec.targets, dtype="float32")
        tm = L.missing_masks(np.asarray(rec.targets, dtype="float64")).astype("int8")
        ph = L.placeholder_flags(np.asarray(rec.targets, dtype="float64")).astype("int8")
        lp = labels_path(cache_root, rec.well_id)
        lp.parent.mkdir(parents=True, exist_ok=True)
        tmp2 = lp.with_suffix(".tmp.npz")
        np.savez_compressed(tmp2, targets=t, target_missing=tm, placeholder=ph)
        tmp2.replace(lp)

    return ShardInfo(well_id=rec.well_id, split=split, n_rows=rec.n_rows,
                     n_cols_in=inputs.shape[1], with_targets=rec.targets is not None)


def read_well_shard(cache_root: str | Path, well_id: str, split: str = "train") -> dict:
    """读取一口井的分片（返回 numpy 数组字典；调用方用完即释放）。"""
    if not HAS_NUMPY:
        raise RuntimeError("read_well_shard requires numpy")
    sp = shard_paths(cache_root, well_id, split)
    if not sp.is_file():
        raise FileNotFoundError(f"shard missing: {sp}（先运行 build_cache.py）")
    out: dict = {}
    with np.load(sp, allow_pickle=True) as z:
        out["depth"] = z["depth"].astype("float32")
        out["inputs"] = z["inputs"].astype("float32")
        out["missing"] = z["missing"].astype("int8")
        out["n_rows"] = int(z["n_rows"])
        out["header"] = list(z["header"])
        out["missing_columns"] = list(z["missing_columns"])
        out["extra_columns"] = list(z["extra_columns"])
    lp = labels_path(cache_root, well_id)
    if lp.is_file():
        with np.load(lp, allow_pickle=True) as z:
            out["targets"] = z["targets"].astype("float32")
            out["target_missing"] = z["target_missing"].astype("int8")
            out["placeholder"] = z["placeholder"].astype("int8")
    else:
        out["targets"] = None
        out["target_missing"] = None
        out["placeholder"] = None
    return out


def build_cache(train_dir: str | Path, test_dir: str | Path, cache_root: str | Path,
                limit: int | None = None, verbose: bool = True) -> dict:
    """一次性构建全部缓存，并写 `cache/manifest.json`。"""
    cache_root = Path(cache_root)
    infos: list[ShardInfo] = []
    counts = {"train_wells": 0, "test_wells": 0, "train_rows": 0, "test_rows": 0,
              "placeholder_rows": 0, "missing_rows": 0, "valid_rows": 0}

    for split, d, with_t in (("train", train_dir, True), ("test", test_dir, False)):
        files = sorted(Path(d).glob("*.txt"))
        if limit:
            files = files[:limit]
        for f in files:
            rec = P.parse_well(f, with_targets=with_t)
            infos.append(write_well_shard(rec, cache_root, split))
            counts[f"{split}_wells"] += 1
            counts[f"{split}_rows"] += rec.n_rows
            if with_t:
                counts["placeholder_rows"] += rec.n_placeholder_rows
                counts["missing_rows"] += rec.n_missing_rows
                counts["valid_rows"] += rec.n_valid_rows
            if verbose and counts[f"{split}_wells"] % 20 == 0:
                print(f"  [{split}] {counts[f'{split}_wells']} wells, "
                      f"{counts[f'{split}_rows']} rows", flush=True)

    man = {
        "schema_version": 1,
        "constants": {"columns": list(C.COLUMNS), "placeholder": C.PLACEHOLDER,
                      "sentinels": list(C.SENTINELS), "missing_lt": C.MISSING_LT},
        "counts": counts,
        "shards": [i.__dict__ for i in infos],
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return man


def iter_wells(cache_root: str | Path, split: str = "train") -> Iterator[str]:
    d = Path(cache_root) / "raw" / split
    for f in sorted(d.glob("*.npz")):
        yield f.stem


def shard_bytes(cache_root: str | Path) -> int:
    total = 0
    for f in Path(cache_root).rglob("*.npz"):
        total += f.stat().st_size
    return total
