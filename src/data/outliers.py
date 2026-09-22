"""WP9：异常检测与样本权重（参考 SPWLA 2021 的 IsolationForest / 3σ / IQR）。

原则：不直接删除测试行（官方评分包含全部行）；训练期用**样本权重**降低异常行影响，
或把异常标志作为特征。所有阈值只在训练折上拟合。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def robust_zscore(X: Any) -> np.ndarray:
    X = np.asarray(X, dtype="float64")
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    mad = np.where(mad < 1e-12, 1.0, mad)
    return (X - med) / (1.4826 * mad)


def iqr_outlier_mask(X: Any, factor: float = 3.0,
                     ignore_nan: bool = True) -> np.ndarray:
    """逐列 IQR 异常：返回 (N, d) bool。"""
    X = np.asarray(X, dtype="float64")
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = np.where((q3 - q1) < 1e-12, 1.0, q3 - q1)
    lo = q1 - factor * iqr
    hi = q3 + factor * iqr
    mask = (X < lo[None, :]) | (X > hi[None, :])
    if ignore_nan:
        mask = mask & np.isfinite(X)
    return mask


def iforest_outlier_mask(X: Any, contamination: float = 0.05,
                         seed: int = 0) -> np.ndarray:
    """IsolationForest 逐行异常（sklearn 可选）；缺失时降级 IQR 行聚合。"""
    X = np.asarray(X, dtype="float64")
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        return iqr_outlier_mask(X, factor=3.0).any(axis=1)
    Xf = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0)[None, :])
    model = IsolationForest(contamination=float(contamination), random_state=seed)
    pred = model.fit_predict(Xf)
    return pred < 0


def sample_weights_from_outliers(X: Any, method: str = "iqr",
                                 max_downweight: float = 0.2,
                                 factor: float = 3.0, contamination: float = 0.05,
                                 seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """把异常行降权到 ``[max_downweight, 1]``；返回 (weights, report)。"""
    X = np.asarray(X, dtype="float64")
    n = X.shape[0]
    m = str(method).lower()
    if m in ("none", "off"):
        w = np.ones(n, dtype="float64")
        bad = np.zeros(n, dtype=bool)
    elif m in ("iqr", "robust"):
        per = iqr_outlier_mask(X, factor=factor)
        bad = per.any(axis=1)
        frac = per.mean(axis=1)
        w = 1.0 - (1.0 - float(max_downweight)) * np.clip(frac, 0.0, 1.0)
    elif m in ("iforest", "isolationforest"):
        bad = iforest_outlier_mask(X, contamination=contamination, seed=seed)
        w = np.where(bad, float(max_downweight), 1.0)
    else:
        raise ValueError(f"unknown outlier method {method!r}")
    w = np.clip(w, float(max_downweight), 1.0)
    return w, {"method": m, "n_rows": int(n), "n_flagged": int(bad.sum()),
               "flagged_frac": float(bad.mean()), "max_downweight": float(max_downweight)}
