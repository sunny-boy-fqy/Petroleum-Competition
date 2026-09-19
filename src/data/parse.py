"""测井 txt 解析（v4 自写，标准库优先，**按表头名对齐**）。

文件格式（rules.md §5.1）：
    第 1 行：曲线名称
    第 2 行：单位（丢弃）
    第 3 行起：逗号分隔数据，0.1 m 采样

⚠️ **E0 实测的关键事实（本机复算，2026-09-19）**：本地 80 口训练井中
**有 3 口井的表头与官方 17 列不一致**，共 27,080 行：

| 井（前 8 位） | 列数 | 差异 |
|---|---:|---|
| `42f2870b` | 20 | 多 `K, U, CGR`；**缺 `CASE`** |
| `b7eb1274` | 21 | 多 `TH, K, U, CGR`；**缺 `CASE`** |
| `c7611b01` | 16 | **缺 `CASE`** |

因此 **绝不能按列位置解析**（按位置会把 GR 之后的曲线整体错位，
或直接丢弃这 27,080 行）。本模块改为：

1. 读表头 → 建立 `表头名 -> 列号` 映射（大写、去空格、去 BOM）；
2. 按**名字**把每行数据填入规范 17 列；
3. 规范表中**缺失的列填 NaN**（并在 `missing_columns` 中记录）；
4. **多余列（K/U/TH/CGR）忽略**（记录在 `extra_columns`，供未来使用）；
5. 若某行字段数与表头不一致，按较短长度填充、并计入 `malformed_rows` 统计。

规范列顺序见 `src/constants.py::COLUMNS`。仅依赖标准库 + 可选 numpy。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .. import constants as C
from ..portability import HAS_NUMPY

if HAS_NUMPY:
    import numpy as np

# 允许出现在数据里但不属于规范输入曲线的附加曲线（记录备用，不参与建模）
OPTIONAL_CURVES: tuple[str, ...] = ("K", "U", "TH", "CGR", "KLOG", "SPT")


@dataclass
class WellRecord:
    well_id: str
    depth: Any                    # (n,)
    inputs: Any                   # (n, 14) 规范顺序，缺列为 NaN
    targets: Any | None           # (n, 3) 或 None
    units: tuple[str, ...] = ()
    header: tuple[str, ...] = ()
    n_rows: int = 0
    n_header_cols: int = 0
    n_missing_rows: int = 0       # 三目标全缺
    n_placeholder_rows: int = 0
    n_valid_rows: int = 0
    missing_columns: tuple[str, ...] = ()   # 规范列中本井缺失的
    extra_columns: tuple[str, ...] = ()     # 本井多出的（非规范）
    malformed_rows: int = 0                 # 字段数与表头不一致的行
    non_monotonic_depth: int = 0
    duplicate_depth: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _to_float(tok: str) -> float:
    t = tok.strip()
    if not t:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def is_missing(value: float) -> bool:
    """冻结哨兵规则：NaN / -99999 / -9999 / 任何 < -1000。"""
    if value != value:
        return True
    if value < C.MISSING_LT:
        return True
    return any(value == s for s in C.SENTINELS)


def parse_well(path: str | Path, with_targets: bool = True) -> WellRecord:
    p = Path(path)
    well_id = p.stem

    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"empty file: {p}") from None
        units = next(reader, [])

        header = [h.strip().upper() for h in raw_header]
        # 去重检查（同名列会导致映射歧义）
        if len(set(header)) != len(header):
            dup = [h for h in set(header) if header.count(h) > 1]
            raise ValueError(f"{p.name}: duplicate column names {dup}")
        idx = {h: i for i, h in enumerate(header)}
        n_hdr = len(header)

        wanted = list(C.COLUMNS) if with_targets else list(C.INPUT_COLUMNS)
        missing_cols = tuple(c for c in wanted if c not in idx)
        extra_cols = tuple(h for h in header if h not in C.COLUMNS)

        if "DEPTH" not in idx:
            raise ValueError(f"{p.name}: no DEPTH column")
        if with_targets:
            absent_targets = [t for t in C.TARGET_COLUMNS if t not in idx]
            if absent_targets:
                raise ValueError(f"{p.name}: missing target columns {absent_targets}")

        # 每行按名字重排；缺列填 NaN
        n_out = len(wanted)
        rows: list[list[float]] = []
        malformed = 0
        take = [idx[c] for c in wanted if c in idx]
        for raw in reader:
            if not raw or (len(raw) == 1 and not raw[0].strip()):
                continue
            if len(raw) != n_hdr:
                malformed += 1
            out = [float("nan")] * n_out
            for j, c in enumerate(wanted):
                k = idx.get(c)
                if k is None or k >= len(raw):
                    continue
                out[j] = _to_float(raw[k])
            rows.append(out)

    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError(f"{p.name}: no data rows")
    if HAS_NUMPY:
        arr = np.asarray(rows, dtype="float64")
    else:
        arr = rows
    # 长度断言：禁止 numpy 切片静默截断（曾导致目标最后一列被丢掉）
    got_cols = arr.shape[1] if HAS_NUMPY else len(arr[0])
    if got_cols != n_out:
        raise AssertionError(f"{p.name}: internal column count {got_cols} != expected {n_out}")

    depth = arr[:, 0] if HAS_NUMPY else [r[0] for r in arr]
    inputs = arr[:, 1:15] if HAS_NUMPY else [r[1:15] for r in arr]
    targets = None
    n_missing = n_ph = n_valid = 0
    if with_targets:
        # 规范列顺序：0=DEPTH, 1..14 = 14 条输入曲线, 最后 3 列 = POR, PERM, SW
        t0 = n_out - len(C.TARGET_COLUMNS)
        targets = arr[:, t0:n_out] if HAS_NUMPY else [r[t0:n_out] for r in arr]
        if (targets.shape[1] if HAS_NUMPY else len(targets[0])) != 3:
            raise AssertionError(f"{p.name}: targets must have 3 columns")
        for i in range(n_rows):
            t = targets[i]
            if all(is_missing(float(t[k])) for k in range(3)):
                n_missing += 1
                continue
            if all(
                abs(float(t[k]) - C.PLACEHOLDER[C.TARGET_COLUMNS[k]]) <= C.PLACEHOLDER_ABS_TOL
                for k in range(3)
            ):
                n_ph += 1
            else:
                n_valid += 1

    # 深度诊断
    non_mono = dup_depth = 0
    if HAS_NUMPY and n_rows > 1:
        d = np.asarray(depth, dtype="float64")
        non_mono = int((np.diff(d) <= 0).sum())
        dup_depth = int((np.diff(d) == 0).sum())

    # 输入侧哨兵 -> NaN
    if HAS_NUMPY and n_rows:
        inputs = np.array(inputs, dtype="float64", copy=True)
        mask = (inputs < C.MISSING_LT) | ~np.isfinite(inputs)
        inputs[mask] = np.nan

    return WellRecord(
        well_id=well_id,
        depth=depth,
        inputs=inputs,
        targets=targets,
        units=tuple(u.strip() for u in units),
        header=tuple(header),
        n_rows=n_rows,
        n_header_cols=n_hdr,
        n_missing_rows=n_missing,
        n_placeholder_rows=n_ph,
        n_valid_rows=n_valid,
        missing_columns=missing_cols,
        extra_columns=extra_cols,
        malformed_rows=malformed,
        non_monotonic_depth=non_mono,
        duplicate_depth=dup_depth,
    )


def load_split(directory: str | Path, with_targets: bool,
               limit: int | None = None) -> Iterator[WellRecord]:
    d = Path(directory)
    files = sorted(d.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    for f in files:
        yield parse_well(f, with_targets=with_targets)


def summarise(split: str, records: list[WellRecord]) -> dict[str, Any]:
    rows = sum(r.n_rows for r in records)
    miss = sum(r.n_missing_rows for r in records)
    ph = sum(r.n_placeholder_rows for r in records)
    val = sum(r.n_valid_rows for r in records)
    lens = sorted(r.n_rows for r in records)
    per_target_missing = {
        t: int(sum(
            sum(1 for i in range(r.n_rows) if is_missing(float(r.targets[i][k])))
            for r in records if r.targets is not None
        ))
        for k, t in enumerate(C.TARGET_COLUMNS)
    } if records and records[0].targets is not None else {}

    odd = [
        {
            "well_id": r.well_id,
            "n_header_cols": r.n_header_cols,
            "missing_columns": list(r.missing_columns),
            "extra_columns": list(r.extra_columns),
            "n_rows": r.n_rows,
        }
        for r in records
        if r.missing_columns or r.extra_columns
    ]
    return {
        "split": split,
        "n_wells": len(records),
        "n_rows": rows,
        "n_missing_rows": miss,
        "n_placeholder_rows": ph,
        "n_valid_rows": val,
        "frac_missing": round(miss / rows, 6) if rows else None,
        "frac_placeholder": round(ph / rows, 6) if rows else None,
        "frac_valid": round(val / rows, 6) if rows else None,
        "rows_per_well": {
            "min": lens[0] if lens else None,
            "median": lens[len(lens) // 2] if lens else None,
            "max": lens[-1] if lens else None,
        },
        "missing_per_target_rows": per_target_missing,
        "non_monotonic_depth_rows": int(sum(r.non_monotonic_depth for r in records)),
        "duplicate_depth_rows": int(sum(r.duplicate_depth for r in records)),
        "malformed_rows": int(sum(r.malformed_rows for r in records)),
        "noncanonical_schema_wells": odd,
        "optional_curve_well_counts": _optional_counts(records),
    }


def _optional_counts(records: list[WellRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        for c in r.extra_columns:
            out[c] = out.get(c, 0) + 1
    return out
