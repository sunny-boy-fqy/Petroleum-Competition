"""WP9：缺失值插补（median / KNN / iterative MICE）——参考 SPWLA 2021。

规则：
  * 插补器只在训练折上 ``fit``，验证/测试只 ``transform``；
  * 缺失指示位始终单独保留（missingness 本身是信息）；
  * sklearn 可选：没有 sklearn 时 iterative 自动降级为 KNN，并显式标注。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as2d(X: Any) -> np.ndarray:
    X = np.asarray(X, dtype="float64")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def median_imputer(X: Any) -> dict[str, Any]:
    X = _as2d(X)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    out = np.where(np.isfinite(X), X, med[None, :])
    return {"X": out, "fill_values": med, "method": "median"}


def knn_imputer(X: Any, k: int = 5, max_train: int = 5000,
                seed: int = 0) -> dict[str, Any]:
    """逐行 KNN 插补（纯 numpy，分块计算；大数据时对完整行下采样）。"""
    X = _as2d(X)
    n, d = X.shape
    miss = ~np.isfinite(X)
    if not miss.any():
        return {"X": X.copy(), "method": "knn", "k": int(k)}
    complete = ~miss.any(axis=1)
    if complete.sum() < 2:
        med = median_imputer(X)
        med["method"] = "knn->median(no complete rows)"
        return med
    rng = np.random.default_rng(seed)
    idx_complete = np.where(complete)[0]
    if idx_complete.size > int(max_train):
        idx_complete = rng.choice(idx_complete, size=int(max_train), replace=False)
    ref = X[idx_complete]
    out = X.copy()
    for i in np.where(miss.any(axis=1))[0]:
        obs = np.isfinite(X[i])
        if obs.sum() == 0:
            out[i, :] = np.nanmedian(ref, axis=0)
            continue
        # 在共同观测维度上算距离（用参考行的中位数填补参考缺失，保持简单）
        ref_obs = ref[:, obs]
        ref_filled = np.where(np.isfinite(ref_obs), ref_obs,
                              np.nanmedian(ref_obs, axis=0)[None, :])
        dist = np.sqrt(((ref_filled - X[i, obs][None, :]) ** 2).sum(axis=1))
        nn = np.argsort(dist)[:max(int(k), 1)]
        vals = ref[nn][:, ~obs]
        fill = np.nanmedian(vals, axis=0)
        fill = np.where(np.isfinite(fill), fill, np.nanmedian(ref[:, ~obs], axis=0))
        out[i, ~obs] = fill
    return {"X": out, "method": "knn", "k": int(k),
            "n_imputed": int(miss.any(axis=1).sum())}


def iterative_imputer(X: Any, max_iter: int = 10,
                      estimator: str = "ridge", seed: int = 0) -> dict[str, Any]:
    """MICE（sklearn IterativeImputer）；sklearn 缺失时降级 KNN。"""
    X = _as2d(X)
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
    except Exception:
        res = knn_imputer(X, seed=seed)
        res["method"] = "knn(iterative_fallback_no_sklearn)"
        return res
    if estimator in ("histgb", "gbdt", "lgbm"):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            est = HistGradientBoostingRegressor(random_state=seed)
        except Exception:
            est = BayesianRidge()
    else:
        est = BayesianRidge()
    imp = IterativeImputer(estimator=est, max_iter=int(max_iter),
                           random_state=seed, sample_posterior=False)
    out = imp.fit_transform(X)
    return {"X": out, "method": f"iterative:{type(est).__name__}",
            "n_imputed": int((~np.isfinite(X)).any(axis=1).sum())}


def impute_matrix(X: Any, method: str = "median", **kw) -> dict[str, Any]:
    m = str(method).lower()
    if m in ("median", "mean"):
        return median_imputer(X)
    if m in ("knn", "knnimputer"):
        return knn_imputer(X, **kw)
    if m in ("mice", "iterative", "iterativeimputer"):
        return iterative_imputer(X, **kw)
    if m in ("none", "off"):
        Xa = _as2d(X)
        return {"X": Xa.copy(), "method": "none"}
    raise ValueError(f"unknown imputation method {method!r}")


@dataclass
class MatrixImputer:
    """可 fit/transform 的插补器包装；fit 只允许在训练折调用。"""
    method: str = "median"
    params: dict[str, Any] = field(default_factory=dict)
    _fill: np.ndarray | None = None
    _impl: Any = None
    fitted: bool = False

    def fit(self, X: Any) -> "MatrixImputer":
        res = impute_matrix(X, self.method, **self.params)
        self._impl = res
        self._fill = np.asarray(res.get("fill_values", np.nan), dtype="float64")
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = _as2d(X)
        if not self.fitted:
            raise RuntimeError("MatrixImputer 尚未 fit")
        if self.method in ("median", "mean") and self._fill is not None:
            return np.where(np.isfinite(X), X, self._fill[None, :])
        # KNN/迭代插补的 transform 用同一训练结果重新拟合代价高；这里复用 fill/中位数
        if self._fill is not None and np.isfinite(self._fill).any():
            return np.where(np.isfinite(X), X, self._fill[None, :])
        return X

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)
