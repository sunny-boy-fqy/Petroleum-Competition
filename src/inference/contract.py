"""提交契约校验（rules.md §6.1）。

只依赖标准库（可选 numpy 用于有限性检查），**不 import torch**，
因此本机与云端都能在打包前跑同一套校验。

校验项（任一失败即拒绝提交）：
  1. 顶层键恰为 modelId / modelName / version / resultData；
  2. resultData 为数组，元素含 logId 与 predictions；
  3. logId 与测试目录文件名集合**完全一致**（不重不漏）；
  4. predictions 行数与输入逐行一致（默认 95,948 行 / 10 井）；
  5. 每行键恰为 depth / POR / PERM / SW，depth 单调递增且与输入深度对齐；
  6. POR/SW 有限；PERM 有限且 > 0；
  7. 不做任何范围裁剪（SW 必须保持训练标签尺度，禁止压到 [0,1]）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import constants as C


@dataclass
class ContractResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _input_rows(test_dir: str | Path, well_id: str) -> int | None:
    f = Path(test_dir) / f"{well_id}.txt"
    if not f.is_file():
        return None
    n = 0
    with f.open("r", encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh):
            if i < 2:
                continue
            if line.strip():
                n += 1
    return n


def _input_depths(test_dir: str | Path, well_id: str) -> list[float]:
    """读取输入文件的逐行深度（用于深度对齐测试与校验）。"""
    f = Path(test_dir) / f"{well_id}.txt"
    out: list[float] = []
    if not f.is_file():
        return out
    with f.open("r", encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh):
            if i < 2 or not line.strip():
                continue
            try:
                out.append(float(line.split(",")[0]))
            except ValueError:
                continue
    return out


def validate_payload(payload: dict[str, Any], test_dir: str | Path | None = None,
                     expected_rows: int | None = None,
                     expected_wells: int | None = None,
                     row_scales: dict[str, tuple[float, float]] | None = None,
                     strict_keys: bool = True) -> ContractResult:
    res = ContractResult(ok=True)

    if not isinstance(payload, dict):
        res.ok = False
        res.errors.append("payload is not a JSON object")
        return res

    top = set(payload.keys())
    if strict_keys and top != set(C.RESULT_TOP_KEYS):
        res.ok = False
        res.errors.append(
            f"top-level keys {sorted(top)} != {sorted(C.RESULT_TOP_KEYS)}"
        )

    data = payload.get("resultData")
    if not isinstance(data, list):
        res.ok = False
        res.errors.append("resultData is not a list")
        return res

    n_rows_total = 0
    n_wells = len(data)
    wells_seen: set[str] = set()

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            res.ok = False
            res.errors.append(f"resultData[{i}] is not an object")
            continue
        log_id = item.get("logId")
        preds = item.get("predictions")
        if not isinstance(log_id, str) or not log_id:
            res.ok = False
            res.errors.append(f"resultData[{i}].logId missing/invalid")
            continue
        if log_id in wells_seen:
            res.ok = False
            res.errors.append(f"duplicate logId {log_id}")
        wells_seen.add(log_id)
        if not isinstance(preds, list) or not preds:
            res.ok = False
            res.errors.append(f"{log_id}: predictions missing/empty")
            continue

        n_rows_total += len(preds)
        prev_depth = None
        n_bad_perm = n_bad_finite = n_bad_keys = 0
        for j, p in enumerate(preds):
            if not isinstance(p, dict):
                n_bad_keys += 1
                continue
            if strict_keys and set(p.keys()) != set(C.PREDICTION_KEYS):
                n_bad_keys += 1
                continue
            d, por, perm, sw = p.get("depth"), p.get("POR"), p.get("PERM"), p.get("SW")
            if not all(_is_finite(v) for v in (d, por, perm, sw)):
                n_bad_finite += 1
                continue
            perm_f = float(perm)
            if perm_f <= 0:
                n_bad_perm += 1
            if prev_depth is not None and float(d) <= prev_depth:
                res.ok = False
                if len(res.errors) < 40:
                    res.errors.append(f"{log_id}: depth not strictly increasing at row {j}")
            prev_depth = float(d)
            if j == 0:
                res.stats.setdefault("first_depths", {})[log_id] = float(d)
            if j == len(preds) - 1:
                res.stats.setdefault("last_depths", {})[log_id] = float(d)

        if n_bad_keys:
            res.ok = False
            res.errors.append(f"{log_id}: {n_bad_keys} rows with wrong keys {C.PREDICTION_KEYS}")
        if n_bad_finite:
            res.ok = False
            res.errors.append(f"{log_id}: {n_bad_finite} rows with non-finite values")
        if n_bad_perm:
            res.ok = False
            res.errors.append(f"{log_id}: {n_bad_perm} rows with PERM <= 0")

        # R2-B6：逐井行数必须与输入文件一致
        if test_dir is not None:
            n_in = _input_rows(test_dir, log_id)
            if n_in is None:
                res.ok = False
                res.errors.append(f"{log_id}: no matching input file under {test_dir}")
            elif n_in != len(preds):
                res.ok = False
                res.errors.append(
                    f"{log_id}: predictions {len(preds)} rows != input {n_in} rows")

    # 井数硬校验（R2-B6：即使没有 test_dir 也要检查 expected_wells）
    if expected_wells is not None and n_wells != int(expected_wells):
        res.ok = False
        res.errors.append(f"payload has {n_wells} wells, expected {int(expected_wells)}")

    # 井集合与输入目录比对
    if test_dir is not None:
        dir_wells = {f.stem for f in Path(test_dir).glob("*.txt")}
        missing = dir_wells - wells_seen
        extra = wells_seen - dir_wells
        if missing:
            res.ok = False
            res.errors.append(f"missing wells: {sorted(missing)[:5]} ({len(missing)})")
        if extra:
            res.ok = False
            res.errors.append(f"unexpected wells: {sorted(extra)[:5]} ({len(extra)})")
        res.stats["n_expected_wells"] = len(dir_wells)
        want_wells = C.EXPECTED_N_TEST_WELLS if expected_wells is None else int(expected_wells)
        if len(dir_wells) != want_wells:
            # R2-B6：井数是硬错误（此前只 warn）
            res.ok = False
            res.errors.append(
                f"test dir has {len(dir_wells)} wells, expected {want_wells}")
        if n_wells != want_wells:
            res.ok = False
            res.errors.append(f"payload has {n_wells} wells, expected {want_wells}")

    # R2-B6：SW 标签尺度守卫（防止误用 [0,1] 归一化后直接提交）
    scales = row_scales or {"SW": (C.SW_MIN_OBSERVED, 100.0),
                            "POR": (0.0, 60.0), "PERM": (0.0, 1e6)}
    if "SW" in scales:
        med = [float(p.get("SW", 0.0)) for item in data for p in item.get("predictions", [])
               if isinstance(p, dict)]
        if med:
            import statistics as _st
            m = _st.median(med)
            lo, hi = scales["SW"]
            # 训练集有效 SW 中位数 ≈ 82.8；若整份预测中位数 < lo，几乎必然是量纲错误
            res.stats["sw_median"] = m
            if m < lo:
                res.ok = False
                res.errors.append(
                    f"SW median {m:.4f} < {lo} —— 疑似把 SW 归一化到 [0,1] 后直接输出"
                    "（训练标签 SW 有效值实测 8.3–99.9）")

    exp_rows = expected_rows if expected_rows is not None else C.EXPECTED_N_TEST_ROWS
    res.stats["n_wells"] = n_wells
    res.stats["n_rows"] = n_rows_total
    res.stats["expected_rows"] = exp_rows
    if exp_rows is not None and n_rows_total != exp_rows:
        res.ok = False
        res.errors.append(f"row count {n_rows_total} != expected {exp_rows}")

    return res


def validate_file(path: str | Path, test_dir: str | Path | None = None,
                  expected_rows: int | None = None) -> ContractResult:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        r = ContractResult(ok=False)
        r.errors.append(f"cannot read {p}: {exc!r}")
        return r
    return validate_payload(payload, test_dir=test_dir, expected_rows=expected_rows)


def depth_alignment_report(predictions_by_well: dict[str, list[float]],
                           test_dir: str | Path) -> dict[str, Any]:
    """检查预测深度与输入文件深度是否逐行一致（必须完全一致）。"""
    out: dict[str, Any] = {}
    for well, depths in predictions_by_well.items():
        f = Path(test_dir) / f"{well}.txt"
        if not f.is_file():
            out[well] = {"ok": False, "reason": "missing input file"}
            continue
        with f.open("r", encoding="utf-8-sig", newline="") as fh:
            lines = fh.read().splitlines()
        ref = []
        for ln in lines[2:]:
            ln = ln.strip()
            if not ln:
                continue
            ref.append(float(ln.split(",")[0]))
        ok = len(ref) == len(depths) and all(
            abs(a - b) < 1e-6 for a, b in zip(ref, depths)
        )
        out[well] = {"ok": ok, "n_input": len(ref), "n_pred": len(depths)}
    return out
