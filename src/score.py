"""官方评分复算（rules.md §7.3 / §7.4）。

    Acc_POR  = mean_i max(0, 1 - |ŷ-y| / (0.08*(|y|+eps)))
    Acc_PERM = mean_i max(0, 1 - |log10(max(ŷ/y, eps))|)
    Acc_SW   = mean_i max(0, 1 - |ŷ-y| / (0.05*(|y|+eps)))
    Total    = (0.30*Acc_POR + 0.35*Acc_PERM + 0.35*Acc_SW) * 100

**分母口径（E0 冻结，已由常数基线复算确定）**
`missing_mode`：
  - `"drop"`（**默认，冻结口径**）：分母只算该目标非缺测的行数。
    依据：常数基线 (0.1, 0.01, 99.9) 在此口径下 = **70.490735**，命中公开锚点 70.4907 ±1e-4。
  - `"mask"`：分母为全部行 N（rules 公式字面写法），缺测行该目标得 0；
    此口径下常数基线 = 69.843218，**比锚点低 0.65 分**，故官方不是这个口径。

E0/P2 Gate：两者都必须能被复算出来并写入数据卡；`drop` 必须命中锚点。

只依赖 numpy；无 numpy 时自动退化到纯 Python（慢但可用于小样本单测）。
"""
from __future__ import annotations

import math
from typing import Any

from . import constants as C
from .portability import HAS_NUMPY

if HAS_NUMPY:
    import numpy as np


def _as_array(x: Any) -> Any:
    return np.asarray(x, dtype="float64") if HAS_NUMPY else [float(v) for v in x]


def acc_relative(y: Any, yhat: Any, delta: float, eps: float = C.EPS,
                 missing_mask: Any = None, missing_mode: str = C.SCORE_MISSING_MODE) -> float:
    """POR / SW：相对误差线性得分。"""
    if HAS_NUMPY:
        y = _as_array(y)
        p = _as_array(yhat)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = 1.0 - np.abs(p - y) / (delta * (np.abs(y) + eps))
        s = np.clip(s, 0.0, 1.0)
        if missing_mask is not None:
            m = np.asarray(missing_mask, dtype=bool)
            if missing_mode == "mask":
                s = np.where(m, 0.0, s)
                return float(s.mean())
            keep = ~m
            return float(s[keep].mean()) if keep.any() else 0.0
        return float(s.mean())

    s_list = []
    keep_flags = []
    for i, (yy, pp) in enumerate(zip(y, yhat)):
        yy = float(yy)
        pp = float(pp)
        sc = 1.0 - abs(pp - yy) / (delta * (abs(yy) + eps))
        sc = max(0.0, min(1.0, sc))
        miss = bool(missing_mask[i]) if missing_mask is not None else False
        s_list.append(0.0 if (miss and missing_mode == "mask") else sc)
        keep_flags.append(not miss)
    if missing_mask is not None and missing_mode == "drop":
        kept = [s for s, k in zip(s_list, keep_flags) if k]
        return sum(kept) / len(kept) if kept else 0.0
    return sum(s_list) / len(s_list) if s_list else 0.0


def acc_perm(y: Any, yhat: Any, eps: float = C.EPS, missing_mask: Any = None,
             missing_mode: str = C.SCORE_MISSING_MODE) -> float:
    """PERM：log10 量级误差得分（相差 10 倍得 0）。"""
    if HAS_NUMPY:
        y = _as_array(y)
        p = _as_array(yhat)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.maximum(p / np.maximum(y, 1e-300), eps)
            s = 1.0 - np.abs(np.log10(ratio))
        s = np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0)
        if missing_mask is not None:
            m = np.asarray(missing_mask, dtype=bool)
            if missing_mode == "mask":
                return float(np.where(m, 0.0, s).mean())
            keep = ~m
            return float(s[keep].mean()) if keep.any() else 0.0
        return float(s.mean())

    s_list, keep_flags = [], []
    for i, (yy, pp) in enumerate(zip(y, yhat)):
        yy = float(yy)
        pp = float(pp)
        ratio = max(pp / yy if yy > 0 else 0.0, eps)
        sc = 1.0 - abs(math.log10(ratio))
        sc = max(0.0, min(1.0, sc))
        miss = bool(missing_mask[i]) if missing_mask is not None else False
        s_list.append(0.0 if (miss and missing_mode == "mask") else sc)
        keep_flags.append(not miss)
    if missing_mask is not None and missing_mode == "drop":
        kept = [s for s, k in zip(s_list, keep_flags) if k]
        return sum(kept) / len(kept) if kept else 0.0
    return sum(s_list) / len(s_list) if s_list else 0.0


def score_arrays(y_true: Any, y_pred: Any, missing: Any = None,
                 missing_mode: str = C.SCORE_MISSING_MODE) -> dict[str, float]:
    """对一个 split（或多井拼接）打分。

    y_true / y_pred : (n, 3) 顺序为 POR, PERM, SW（标签尺度）
    missing         : (n, 3) bool，True 表示该目标缺测
    """
    if HAS_NUMPY:
        yt = np.asarray(y_true, dtype="float64")
        yp = np.asarray(y_pred, dtype="float64")
        m = None if missing is None else np.asarray(missing, dtype=bool)
    else:
        yt = [list(map(float, r)) for r in y_true]
        yp = [list(map(float, r)) for r in y_pred]
        m = missing

    def col(a, j):
        return a[:, j] if HAS_NUMPY else [r[j] for r in a]

    def mcol(j):
        if m is None:
            return None
        return m[:, j] if HAS_NUMPY else [r[j] for r in m]

    a_por = acc_relative(col(yt, 0), col(yp, 0), C.DELTA_POR, missing_mask=mcol(0),
                         missing_mode=missing_mode)
    a_perm = acc_perm(col(yt, 1), col(yp, 1), missing_mask=mcol(1),
                      missing_mode=missing_mode)
    a_sw = acc_relative(col(yt, 2), col(yp, 2), C.DELTA_SW, missing_mask=mcol(2),
                        missing_mode=missing_mode)
    total = 100.0 * (
        C.SCORE_WEIGHTS["POR"] * a_por
        + C.SCORE_WEIGHTS["PERM"] * a_perm
        + C.SCORE_WEIGHTS["SW"] * a_sw
    )
    n = len(yt)
    return {
        "n_rows": n,
        "acc_por": a_por,
        "acc_perm": a_perm,
        "acc_sw": a_sw,
        "total": total,
        "missing_mode": missing_mode,
    }


def constant_prediction(n: int, what: str = "placeholder") -> Any:
    """生成常数预测 (n, 3)。"""
    if what == "placeholder":
        vals = [C.PLACEHOLDER[t] for t in C.TARGET_COLUMNS]
    elif what == "zeros":
        vals = [0.0, 0.0001, 0.0]
    else:
        raise ValueError(what)
    if HAS_NUMPY:
        return np.tile(np.asarray(vals, dtype="float64"), (n, 1))
    return [list(vals) for _ in range(n)]


def slice_scores(y_true: Any, y_pred: Any, idx: Any, missing: Any = None) -> dict[str, float]:
    """子切片评分（如"仅连续行"，排除原子行）——用于 Gate 的连续切片判据。"""
    if HAS_NUMPY:
        yt = np.asarray(y_true, dtype="float64")[idx]
        yp = np.asarray(y_pred, dtype="float64")[idx]
        mm = None if missing is None else np.asarray(missing, dtype=bool)[idx]
    else:
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        mm = None if missing is None else [missing[i] for i in idx]
    return score_arrays(yt, yp, mm)
