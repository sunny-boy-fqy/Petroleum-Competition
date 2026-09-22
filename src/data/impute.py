"""WP9：缺失值插补（median / KNN / iterative MICE）——参考 SPWLA 2021。

规则：
  * 插补器只在训练折上 ``fit``，验证/测试只 ``transform``；
  * 缺失指示位始终单独保留（missingness 本身是信息）；
  * sklearn 可选：没有 sklearn 时 iterative 自动降级为 KNN，并显式标注。

审计修复（P1）：``MatrixImputer`` 的 KNN/MICE 分支不再静默返回原始 NaN；
``fit`` 会保存真正的 KNN/MICE 模型，``transform`` 必须使用该模型。
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


def _column_medians(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def median_imputer(X: Any) -> dict[str, Any]:
    X = _as2d(X)
    med = _column_medians(X)
    out = np.where(np.isfinite(X), X, med[None, :])
    return {"X": out, "fill_values": med, "method": "median"}


class KNNImputerModel:
    """可 fit/transform 的纯 numpy KNN 插补器（训练折保存参考行）。"""

    def __init__(self, k: int = 5, max_train: int = 5000, seed: int = 0):
        self.k = max(int(k), 1)
        self.max_train = int(max_train)
        self.seed = int(seed)
        self.ref_: np.ndarray | None = None
        self.fill_values_: np.ndarray | None = None

    def fit(self, X: Any) -> "KNNImputerModel":
        X = _as2d(X)
        self.fill_values_ = _column_medians(X)
        miss = ~np.isfinite(X)
        complete = ~miss.any(axis=1)
        if complete.sum() < 2:                  # 没有足够完整行 -> 只用列中位数兜底
            self.ref_ = None
            return self
        rng = np.random.default_rng(self.seed)
        idx = np.where(complete)[0]
        if idx.size > self.max_train:
            idx = rng.choice(idx, size=self.max_train, replace=False)
        self.ref_ = X[idx].astype("float64", copy=True)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.ref_ is None:
            X = _as2d(X)
            if self.fill_values_ is None:
                raise RuntimeError("KNNImputerModel 尚未 fit")
            return np.where(np.isfinite(X), X, self.fill_values_[None, :])
        X = _as2d(X)
        if self.fill_values_ is None:
            raise RuntimeError("KNNImputerModel 尚未 fit")
        out = X.copy()
        ref = self.ref_
        for i in range(out.shape[0]):
            obs = np.isfinite(out[i])
            if not obs.any():
                out[i, :] = self.fill_values_
                continue
            if obs.all():
                continue
            ref_obs = ref[:, obs]
            dist = np.sqrt(((ref_obs - out[i, obs][None, :]) ** 2).sum(axis=1))
            nn = np.argsort(dist)[:self.k]
            vals = ref[nn][:, ~obs]
            if vals.size == 0:
                out[i, ~obs] = self.fill_values_[~obs]
                continue
            fill = np.nanmedian(vals, axis=0)
            fill = np.where(np.isfinite(fill), fill, self.fill_values_[~obs])
            out[i, ~obs] = fill
        return out

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)


def knn_imputer(X: Any, k: int = 5, max_train: int = 5000,
                seed: int = 0) -> dict[str, Any]:
    """逐行 KNN 插补（纯 numpy）；返回 X 与可复用的 ``model``。"""
    X = _as2d(X)
    model = KNNImputerModel(k=k, max_train=max_train, seed=seed).fit(X)
    out = model.transform(X)
    return {"X": out, "method": "knn", "k": int(k),
            "n_imputed": int((~np.isfinite(X)).any(axis=1).sum()),
            "model": model}


def _make_mice_model(X: np.ndarray, max_iter: int, estimator: str, seed: int):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    if estimator in ("histgb", "gbdt", "lgbm"):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            est = HistGradientBoostingRegressor(random_state=seed)
        except Exception:
            est = BayesianRidge()
    else:
        est = BayesianRidge()
    return IterativeImputer(estimator=est, max_iter=int(max_iter),
                            random_state=seed, sample_posterior=False)


def iterative_imputer(X: Any, max_iter: int = 10,
                      estimator: str = "ridge", seed: int = 0) -> dict[str, Any]:
    """MICE（sklearn IterativeImputer）；sklearn 缺失时降级 KNN。"""
    X = _as2d(X)
    try:
        model = _make_mice_model(X, max_iter=max_iter, estimator=estimator, seed=seed)
        out = model.fit_transform(X)
        return {"X": out, "method": f"iterative:{type(model.estimator).__name__}",
                "n_imputed": int((~np.isfinite(X)).any(axis=1).sum()), "model": model}
    except Exception:
        res = knn_imputer(X, seed=seed)
        res["method"] = "knn(iterative_fallback_no_sklearn)"
        return res


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
    _model: Any = None
    fitted: bool = False

    def fit(self, X: Any) -> "MatrixImputer":
        X = _as2d(X)
        m = str(self.method).lower()
        if m in ("median", "mean"):
            self._fill = _column_medians(X)
            self._model = None
        elif m in ("knn", "knnimputer"):
            self._model = KNNImputerModel(
                k=int(self.params.get("k", 5)),
                max_train=int(self.params.get("max_train", 5000)),
                seed=int(self.params.get("seed", 0))).fit(X)
            self._fill = self._model.fill_values_
        elif m in ("mice", "iterative", "iterativeimputer"):
            try:
                self._model = _make_mice_model(
                    X, max_iter=int(self.params.get("max_iter", 10)),
                    estimator=str(self.params.get("estimator", "ridge")),
                    seed=int(self.params.get("seed", 0)))
                self._fill = _column_medians(X)     # 兜底/诊断
            except Exception:
                # sklearn 缺失 -> KNN 模型，绝不静默返回原始 NaN
                self._model = KNNImputerModel(
                    k=int(self.params.get("k", 5)),
                    max_train=int(self.params.get("max_train", 5000)),
                    seed=int(self.params.get("seed", 0))).fit(X)
                self._fill = self._model.fill_values_
        elif m in ("none", "off"):
            self._model = None
            self._fill = None
        else:
            raise ValueError(f"unknown imputation method {self.method!r}")
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MatrixImputer 尚未 fit")
        X = _as2d(X)
        m = str(self.method).lower()
        if m in ("none", "off"):
            return X
        if m in ("median", "mean"):
            if self._fill is None:
                raise RuntimeError("median MatrixImputer 缺少 fill_values")
            return np.where(np.isfinite(X), X, self._fill[None, :])
        if m in ("knn", "knnimputer"):
            if self._model is None:
                raise RuntimeError("KNN MatrixImputer 缺少 model")
            return self._model.transform(X)
        # MICE：sklearn 模型有 transform；降级 KNN 模型也有 transform
        if self._model is not None and hasattr(self._model, "transform"):
            return self._model.transform(X)
        if self._fill is None:
            raise RuntimeError("MICE MatrixImputer 缺少可用模型")
        return np.where(np.isfinite(X), X, self._fill[None, :])

    def fit_transform(self, X: Any) -> np.ndarray:
        X = _as2d(X)
        self.fit(X)
        if str(self.method).lower() in ("none", "off"):
            return X
        if self._model is not None and hasattr(self._model, "fit_transform"):
            # MICE 的 fit_transform 走 sklearn；KNN 的 fit_transform 也可用
            return self._model.fit_transform(X)
        return self.transform(X)
