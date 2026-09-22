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
  7. 不做任何范围裁剪（SW 必须保持训练标签尺度，禁止压到 [0,1]）；
  8. **SW 尺度四重守卫（R4-B3）**：`SW<1.0` 占比、`SW<SW_VALID_MIN` 占比、非原子行 p05、全体中位数
     —— 任一越界即拒绝，防止"原子行占多数"把中位数拉高从而掩盖连续分支被错误归一化到 [0,1]。
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


def _percentile_sorted(sorted_vals: list[float], q: float) -> float | None:
    """线性插值分位数（与 numpy 默认 `linear` 口径一致，纯标准库）。"""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


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
    # M3 审查修复：POR/PERM 也有物理范围，不能只查有限与 PERM>0。
    # 缺省上界取模型/标签物理量级（PERM=10**6，POR=60 与现有 row_scales 一致）；
    # 调用方可用 row_scales 覆盖。SW 的 [0,100] 是官方标签软上界，低于 0/高于 100 必拒。
    bounds = row_scales or {"SW": (0.0, 100.0)}
    por_lo, por_hi = (bounds.get("POR") or (0.0, 60.0))
    perm_lo, perm_hi = (bounds.get("PERM") or (0.0, 1e6))
    sw_lo, sw_hi = (bounds.get("SW") or (0.0, 100.0))

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
        n_bad_range = 0
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
            por_f, sw_f = float(por), float(sw)
            if not (float(por_lo) <= por_f <= float(por_hi)):
                n_bad_range += 1
            if not (float(perm_lo) < perm_f <= float(perm_hi)):
                n_bad_range += 1
            if not (float(sw_lo) <= sw_f <= float(sw_hi)):
                n_bad_range += 1
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
        if n_bad_range:
            res.ok = False
            res.errors.append(
                f"{log_id}: {n_bad_range} rows outside physical ranges "
                f"(POR∈[{por_lo},{por_hi}], PERM∈({perm_lo},{perm_hi}], "
                f"SW∈[{sw_lo},{sw_hi}])")

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

    # R2-B6 + R4-B3：SW 标签尺度守卫。
    # 旧实现只用**全体中位数**，可被"原子行占多数"绕过：7 行 SW=99.9（原子）+ 3 行
    # SW=0.8（被错误归一化到 [0,1] 的连续分支）→ median=99.9 > 8.305 → 契约 passed，
    # 而连续分支的量纲错误被完全掩盖。这正是 PD1 架构最可能出事的情形。
    # 现在改为**四重判据**（任一触发即拒绝提交）：
    #   (a) 明确量纲错误：n(SW < SW_LOW_GUARD_ABS=1.0)/n_obs > 1e-3
    #       —— 训练标签里 SW<1 的行数为 0，任何成规模的低值都只能是量纲错误；
    #   (b) 低于有效最小值的量成规模：n(SW < SW_VALID_MIN)/n_obs > 1%
    #       —— 容忍个别边界外推（真值最小 8.305），但整片低于下界必然是尺度错；
    #   (c) 非原子行 p05 < SW_VALID_MIN（非原子行 ≥ 20 时判定）—— 直接盯连续分支的低分位；
    #   (d) 全体中位数 < SW_VALID_MIN —— 保留旧守卫，兜住"全部被归一化"的退化情形。
    scales = row_scales or {"SW": (C.SW_MIN_OBSERVED, 100.0),
                            "POR": (0.0, 60.0), "PERM": (0.0, 1e6)}
    if "SW" in scales:
        med = [float(p.get("SW", 0.0)) for item in data for p in item.get("predictions", [])
               if isinstance(p, dict)]
        if med:
            import statistics as _st
            lo, hi = scales["SW"]
            n_obs = len(med)
            m = _st.median(med)
            n_low_abs = sum(1 for v in med if v < C.SW_LOW_GUARD_ABS)
            n_low_min = sum(1 for v in med if v < C.SW_LOW_GUARD_NONATOM_P05_MIN)
            # 非原子行 = 预测值不在原子值附近的那些行（连续分支的"领地"）
            atom_hi = C.SW_PLACEHOLDER - max(C.PLACEHOLDER_ABS_TOL, 1e-6)
            non_atom = sorted(v for v in med if v < atom_hi)
            p05 = _percentile_sorted(non_atom, 5.0)
            res.stats.update({
                "sw_median": m, "sw_n": n_obs,
                "sw_n_lt_guard": n_low_abs,
                "sw_lt_guard_frac": n_low_abs / n_obs,
                "sw_low_guard_abs": C.SW_LOW_GUARD_ABS,
                "sw_n_lt_valid_min": n_low_min,
                "sw_lt_valid_min_frac": n_low_min / n_obs,
                "sw_valid_min": C.SW_LOW_GUARD_NONATOM_P05_MIN,
                "sw_non_atom_n": len(non_atom),
                "sw_non_atom_p05": p05,
            })
            if n_low_abs / n_obs > C.SW_LOW_GUARD_FRAC_MAX:
                res.ok = False
                res.errors.append(
                    f"SW 低值计数 {n_low_abs}/{n_obs} = {n_low_abs / n_obs:.5f} > "
                    f"{C.SW_LOW_GUARD_FRAC_MAX}（阈值 SW<{C.SW_LOW_GUARD_ABS}）—— "
                    "疑似把 SW 归一化到 [0,1] 后直接输出"
                    "（训练标签 SW<1 的行数实测为 0，有效值 8.305–99.9）")
            if n_low_min / n_obs > C.SW_SUSPECT_FRAC_MAX:
                res.ok = False
                res.errors.append(
                    f"SW 低于有效最小值 {C.SW_LOW_GUARD_NONATOM_P05_MIN} 的行数 "
                    f"{n_low_min}/{n_obs} = {n_low_min / n_obs:.5f} > "
                    f"{C.SW_SUSPECT_FRAC_MAX} —— 连续分支疑似被整体缩小（量纲错误）")
            # p05 只在非原子行足够多时才判定：真实提交有 ~32k 连续行，p05 稳定；
            # 几个行的小样例上 p05 ≈ min，容易把"单点外推"误判成量纲错误。
            if (p05 is not None and len(non_atom) >= C.SW_LOW_GUARD_MIN_NONATOM
                    and p05 < C.SW_LOW_GUARD_NONATOM_P05_MIN):
                res.ok = False
                res.errors.append(
                    f"SW 非原子行 p05 = {p05:.4f} < {C.SW_LOW_GUARD_NONATOM_P05_MIN}"
                    f"（非原子行 {len(non_atom)} 条）—— 连续分支疑似被归一化到 [0,1]")
            if m < lo:
                res.ok = False
                res.errors.append(
                    f"SW median {m:.4f} < {lo} —— 疑似把 SW 归一化到 [0,1] 后直接输出"
                    "（训练标签 SW 有效值实测 8.3–99.9）")

    # 非原子行 p05 需要分位数：契约只依赖标准库 + 可选 numpy
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
