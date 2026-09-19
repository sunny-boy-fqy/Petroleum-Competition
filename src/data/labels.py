"""标签状态判据与标签尺度工具（E0 冻结口径）。

三状态（rules.md §5.3 + 项目决策）：
    缺测       : 三目标同时为哨兵（-99999/-9999/NaN/< -1000）
    联合常量占位: POR=0.1 ∧ PERM=0.01 ∧ SW=99.9
    有效       : 其余
占位行**不剔除**，参与全量评分。

SW 双尺度警示：同一列里占位是 99.9（百分数），有效值是 [0,1]（小数）。
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
def sw_valid_to_label(sw_small: Any) -> Any:
    """把 [0,1] 的有效分支输出换算为与标签同尺度的百分数。"""
    if HAS_NUMPY:
        return np.asarray(sw_small, dtype="float64") * C.SW_VALID_SCALE
    return [float(v) * C.SW_VALID_SCALE for v in sw_small]


def sw_label_to_valid(sw_label: Any) -> Any:
    """把标签尺度的有效值换算回 [0,1]（仅用于有效分支的分析/监督）。"""
    if HAS_NUMPY:
        return np.asarray(sw_label, dtype="float64") / C.SW_VALID_SCALE
    return [float(v) / C.SW_VALID_SCALE for v in sw_label]


def sw_decode(q_placeholder: Any, f_valid_logit: Any, scale: float | None = None) -> Any:
    """SW 双分支混合解码：q*99.9 + (1-q)*sigmoid(f)*100。

    注意：这是**连续**混合；E6 的实际提交路径使用**硬切换**（q>τ 时直接输出 99.9），
    混合式仅用于训练期监督与诊断对比。
    """
    s = C.SW_VALID_SCALE if scale is None else scale
    if HAS_NUMPY:
        q = np.asarray(q_placeholder, dtype="float64")
        f = np.asarray(f_valid_logit, dtype="float64")
        valid = 1.0 / (1.0 + np.exp(-f)) * s
        return q * C.SW_PLACEHOLDER + (1.0 - q) * valid
    import math

    return [
        float(qq) * C.SW_PLACEHOLDER
        + (1.0 - float(qq)) * (1.0 / (1.0 + math.exp(-float(ff)))) * s
        for qq, ff in zip(q_placeholder, f_valid_logit)
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
