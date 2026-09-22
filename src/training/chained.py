"""WP11：链式目标回归（POR → PERM → SW）——参考 SPWLA 2021 MoLPhy/Atwah。

核心纪律：
  * 后续目标的“前序目标预测”必须是 **out-of-fold** 预测，不能用 in-fold 拟合值，
    否则会标签泄漏；
  * 每步只在观测到的行上拟合（mask=1）；
  * 支持 sklearn 估计器；无 sklearn 时使用内置 numpy Ridge。
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .. import constants as C


class NumpyRidge:
    """极简 numpy Ridge（闭式解，可选非负）。"""

    def __init__(self, alpha: float = 1.0, fit_bias: bool = True, nonneg: bool = False):
        self.alpha = float(alpha)
        self.fit_bias = bool(fit_bias)
        self.nonneg = bool(nonneg)
        self.coef_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64").reshape(-1)
        if self.fit_bias:
            Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        else:
            Xa = X
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype="float64").reshape(-1)[:, None]
            Xw = Xa * w
        else:
            Xw = Xa
        A = Xa.T @ Xw + self.alpha * np.eye(Xa.shape[1])
        b = Xw.T @ y
        coef = np.linalg.solve(A, b)
        if self.fit_bias:
            self.coef_, self.bias_ = coef[:-1], float(coef[-1])
        else:
            self.coef_, self.bias_ = coef, 0.0
        if self.nonneg:
            self.coef_ = np.clip(self.coef_, 0.0, None)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("NumpyRidge 尚未 fit")
        X = np.asarray(X, dtype="float64")
        return X @ self.coef_ + self.bias_


def make_estimator(kind: str = "ridge", **kw):
    k = str(kind).lower()
    if k in ("ridge", "linear"):
        return NumpyRidge(**kw)
    if k in ("histgb", "hist_gb", "gbdt", "sklearn"):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except Exception as exc:
            raise RuntimeError(f"HistGB 需要 scikit-learn：{exc}")
        return HistGradientBoostingRegressor(**kw)
    if k == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError(f"LGBM 需要 lightgbm：{exc}")
        return LGBMRegressor(**kw)
    raise ValueError(f"unknown estimator kind {kind!r}")


def _kfold_indices(n: int, n_splits: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::n_splits] for i in range(int(n_splits))]


def _oof_predictions(X: np.ndarray, y: np.ndarray, obs: np.ndarray,
                     estimator_factory: Callable[[], Any],
                     folds: np.ndarray | None = None, n_splits: int = 5,
                     seed: int = 42) -> np.ndarray:
    """对可观测行做 K-fold OOF 预测；返回长度 n 的向量（含不可观测行）。"""
    n = X.shape[0]
    pred = np.full(n, np.nan, dtype="float64")
    idx_obs = np.where(obs)[0]
    if idx_obs.size == 0:
        return pred
    if folds is not None:
        fold_arr = np.asarray(folds).reshape(-1)
        uniq = sorted(set(int(v) for v in fold_arr[idx_obs]))
        splits = [idx_obs[fold_arr[idx_obs] == k] for k in uniq]
    else:
        splits = [idx_obs[s] for s in _kfold_indices(idx_obs.size, max(int(n_splits), 2), seed)]
    for va in splits:
        tr = np.setdiff1d(idx_obs, va, assume_unique=False)
        if tr.size == 0 or va.size == 0:
            continue
        est = estimator_factory()
        est.fit(X[tr], y[tr])
        pred[va] = np.asarray(est.predict(X[va]), dtype="float64").reshape(-1)
    # 对未覆盖的行用全量拟合兜底
    left = idx_obs[~np.isfinite(pred[idx_obs])]
    if left.size:
        est = estimator_factory()
        est.fit(X[idx_obs], y[idx_obs])
        pred[left] = np.asarray(est.predict(X[left]), dtype="float64").reshape(-1)
    return pred


class ChainedRegressor:
    """顺序预测多个目标；每步把前序目标的 OOF 预测追加为特征。"""

    def __init__(self, estimator_kind: str = "ridge", n_splits: int = 5,
                 seed: int = 42, estimator_params: dict[str, Any] | None = None):
        self.estimator_kind = str(estimator_kind)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.estimator_params = dict(estimator_params or {})
        self.models_: list[Any] = []
        self.feature_names_: list[str] = []
        self.oof_: dict[str, np.ndarray] = {}

    def _factory(self):
        kw = dict(self.estimator_params)
        return make_estimator(self.estimator_kind, **kw)

    def fit(self, X: Any, Y: Any, mask: Any | None = None,
            folds: Any | None = None) -> "ChainedRegressor":
        X = np.asarray(X, dtype="float64")
        Y = np.asarray(Y, dtype="float64")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n, d = X.shape
        if Y.shape[0] != n:
            raise ValueError("X/Y 行数不一致")
        m = np.ones_like(Y, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self.models_ = []
        self.feature_names_ = [f"x{j}" for j in range(d)]
        X_cur = X.copy()
        for t in range(Y.shape[1]):
            obs = m[:, t]
            if obs.sum() < 2:
                raise ValueError(f"目标 {t} 的可观测行不足：{int(obs.sum())}")
            est = self._factory()
            est.fit(X_cur[obs], Y[obs, t])
            self.models_.append(est)
            # 生成该目标的 OOF 预测，作为下一个目标的特征
            oof = _oof_predictions(X_cur, Y[:, t], obs, self._factory,
                                   folds=folds, n_splits=self.n_splits, seed=self.seed)
            self.oof_[f"y{t}_oof"] = oof
            X_cur = np.concatenate([X_cur, oof.reshape(-1, 1)], axis=1)
            self.feature_names_.append(f"y{t}_oof")
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("ChainedRegressor 尚未 fit")
        X_cur = np.asarray(X, dtype="float64")
        outs = []
        for t, est in enumerate(self.models_):
            p = np.asarray(est.predict(X_cur), dtype="float64").reshape(-1)
            outs.append(p)
            X_cur = np.concatenate([X_cur, p.reshape(-1, 1)], axis=1)
        return np.stack(outs, axis=1)

    def fit_predict(self, X: Any, Y: Any, mask: Any | None = None,
                    folds: Any | None = None) -> np.ndarray:
        return self.fit(X, Y, mask=mask, folds=folds).predict(X)
