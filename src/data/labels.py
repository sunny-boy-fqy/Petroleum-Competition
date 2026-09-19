"""标签状态判据与标签尺度工具（E0 冻结口径）。

三状态（rules.md §5.3 + 项目决策）：
    缺测       : 三目标同时为哨兵（-99999/-9999/NaN/< -1000）
    联合常量占位: POR=0.1 ∧ PERM=0.01 ∧ SW=99.9
    有效       : 其余
占位行**不剔除**，参与全量评分。

SW 尺度（E0-R2 修正）：单一标签尺度（百分数）。占位 99.9；有效值实测 8.305–99.9。
此前「双尺度 [0,1]」的说法已被数据证伪（有效行 SW<1 的数量为 0）。
本模块提供显式换算，禁止在别处手写 *100。
"""
from __future__ import annotations

from typing import Any

from .. import constants as C
from ..portability import HAS_NUMPY

if HAS_NUMPY:
    import numpy as np


# ---------------------------------------------------------------- 逐点判据
def placeholder_flags(targets: Any) -> Any:
    """返回 (n,) bool：该行是否为联合常量占位。"""
    if HAS_NUMPY:
        t = np.asarray(targets, dtype="float64")
        return (
            (np.abs(t[:, 0] - C.PLACEHOLDER["POR"]) <= C.PLACEHOLDER_ABS_TOL)
            & (np.abs(t[:, 1] - C.PLACEHOLDER["PERM"]) <= C.PLACEHOLDER_ABS_TOL)
            & (np.abs(t[:, 2] - C.PLACEHOLDER["SW"]) <= C.PLACEHOLDER_ABS_TOL)
        )
    out = []
    for row in targets:
        out.append(
            all(
                abs(float(row[i]) - C.PLACEHOLDER[C.TARGET_COLUMNS[i]]) <= C.PLACEHOLDER_ABS_TOL
                for i in range(3)
            )
        )
    return out


def missing_masks(targets: Any) -> Any:
    """返回 (n, 3) bool：每个目标是否缺测（缺测不参与该目标监督与评分）。"""
    if HAS_NUMPY:
        t = np.asarray(targets, dtype="float64")
        return (t < C.MISSING_LT) | ~np.isfinite(t)
    return [[(float(v) < C.MISSING_LT) or (float(v) != float(v)) for v in row] for row in targets]


def state_labels(targets: Any) -> Any:
    """返回 (n,) int8：0=有效, 1=占位, -1=缺测（三目标全缺）。"""
    if HAS_NUMPY:
        t = np.asarray(targets, dtype="float64")
        miss = (t < C.MISSING_LT) | ~np.isfinite(t)
        all_miss = miss.all(axis=1)
        ph = placeholder_flags(t)
        lab = np.where(all_miss, -1, np.where(ph, 1, 0)).astype("int8")
        return lab
    out = []
    for i, row in enumerate(targets):
        miss = [(float(v) < C.MISSING_LT) or (float(v) != float(v)) for v in row]
        if all(miss):
            out.append(-1)
        elif all(
            abs(float(row[k]) - C.PLACEHOLDER[C.TARGET_COLUMNS[k]]) <= C.PLACEHOLDER_ABS_TOL
            for k in range(3)
        ):
            out.append(1)
        else:
            out.append(0)
    return out


# ---------------------------------------------------------------- SW 尺度
def sw_scale_report(sw: Any) -> dict:
    """报告 SW 的实测范围（用于数据卡与单测），证明它是单一标签尺度。"""
    if HAS_NUMPY:
        v = np.asarray(sw, dtype="float64")
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"n": 0}
        return {
            "n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
            "max": float(v.max()), "p01": float(np.percentile(v, 1)),
            "p99": float(np.percentile(v, 99)),
            "n_lt_1": int((v < 1).sum()), "n_lt_10": int((v < 10).sum()),
            "label_scale": "percent_0_100",
            "small_branch_enabled": C.SW_SMALL_BRANCH,
        }
    vals = sorted(float(x) for x in sw if float(x) == float(x))
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1],
            "label_scale": "percent_0_100", "small_branch_enabled": C.SW_SMALL_BRANCH}


def sw_decode(q_placeholder: Any, f_valid: Any, small_branch: bool | None = None,
              scale: float | None = None) -> Any:
    """SW 混合解码。

    **默认（E0-R2 起）**：SW 是单一标签尺度，故取 `SW = q·99.9 + (1−q)·f_valid`，
    其中 `f_valid` 已是标签尺度的有效分支输出。
    仅当显式 `small_branch=True`（且 `constants.SW_SMALL_BRANCH=True`）时，
    才把 `f_valid` 视为 [0,1] 并乘以 `SW_SMALL_BRANCH_SCALE`（保留旧路径仅供对照；默认关闭）。

    注意：E6 的提交路径使用**硬切换**（q>τ 直接输出 99.9）；本函数用于训练期监督与诊断。
    """
    use_small = C.SW_SMALL_BRANCH if small_branch is None else small_branch
    sc = C.SW_SMALL_BRANCH_SCALE if scale is None else scale
    if HAS_NUMPY:
        q = np.asarray(q_placeholder, dtype="float64")
        f = np.asarray(f_valid, dtype="float64")
        valid = f * sc if use_small else f
        return q * C.SW_PLACEHOLDER + (1.0 - q) * valid
    return [
        float(qq) * C.SW_PLACEHOLDER + (1.0 - float(qq)) * (float(ff) * sc if use_small else float(ff))
        for qq, ff in zip(q_placeholder, f_valid)
    ]


# ---------------------------------------------------------------- 物理约束
def perm_from_log10(z: Any) -> Any:
    """log10 域 -> 线性的 PERM，并强制正且有限（契约要求 PERM > 0）。"""
    if HAS_NUMPY:
        zc = np.clip(np.asarray(z, dtype="float64"), C.PERM_LOG_MIN, C.PERM_LOG_MAX)
        return np.power(10.0, zc)
    return [10.0 ** max(C.PERM_LOG_MIN, min(C.PERM_LOG_MAX, float(v))) for v in z]


def perm_to_log10(perm: Any) -> Any:
    if HAS_NUMPY:
        p = np.maximum(np.asarray(perm, dtype="float64"), 1e-12)
        return np.log10(p)
    import math

    return [math.log10(max(float(v), 1e-12)) for v in perm]
