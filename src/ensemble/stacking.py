"""WP6：二阶 stacking（纯 numpy 优先，可选 sklearn HistGB）。

用途：把多个一阶成员的 OOF 预测（连续头、q_atom、q_joint）作为特征，
在内层 OOF 上训练一个**受限二阶模型**，再在 outer/confirm 上评估。

规则：
  * 二阶模型只允许在 inner-OOF 上拟合；outer 折只推理一次；
  * 报告必须给出相对最佳单成员/一阶融合的 paired CI；
  * 若竞赛规则不允许非一阶模型，调用方必须显式禁用本模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _as2d(X: Any) -> np.ndarray:
    return np.asarray(X, dtype="float64")


class RidgeStacker:
    """非负（可选）岭回归：``y ≈ X w + b``，闭式解 + 非负截断。"""

    def __init__(self, l2: float = 1.0, nonneg: bool = True, fit_bias: bool = True,
                 normalize: bool = False):
        self.l2 = float(l2)
        self.nonneg = bool(nonneg)
        self.fit_bias = bool(fit_bias)
        self.normalize = bool(normalize)
        self.coef_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> "RidgeStacker":
        X = _as2d(X)
        y = np.asarray(y, dtype="float64").reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y 行数不一致：{X.shape[0]} vs {y.shape[0]}")
        if self.normalize:
            self.mu_ = X.mean(axis=0)
            self.sd_ = X.std(axis=0)
            self.sd_[self.sd_ < 1e-12] = 1.0
            X = (X - self.mu_) / self.sd_
        w = np.ones(X.shape[0]) if sample_weight is None else np.asarray(
            sample_weight, dtype="float64").reshape(-1)
        w = np.clip(w, 1e-12, None)
        Xw = X * w[:, None]
        if self.fit_bias:
            Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
            Xw = Xa * w[:, None]
            A = Xa.T @ Xw + self.l2 * np.eye(Xa.shape[1])
            b = Xw.T @ y
            coef = np.linalg.solve(A, b)
            self.coef_ = coef[:-1]
            self.bias_ = float(coef[-1])
        else:
            A = X.T @ Xw + self.l2 * np.eye(X.shape[1])
            b = Xw.T @ y
            self.coef_ = np.linalg.solve(A, b)
            self.bias_ = 0.0
        if self.nonneg:
            self.coef_ = np.clip(self.coef_, 0.0, None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeStacker 尚未 fit")
        X = _as2d(X)
        if self.normalize:
            X = (X - self.mu_) / self.sd_
        return X @ self.coef_ + self.bias_


class LogisticStacker:
    """L2 正则逻辑回归（梯度下降，用于原子概率融合）。"""

    def __init__(self, l2: float = 1.0, lr: float = 0.5, iters: int = 300,
                 fit_bias: bool = True):
        self.l2 = float(l2)
        self.lr = float(lr)
        self.iters = int(iters)
        self.fit_bias = bool(fit_bias)
        self.coef_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(x):
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> "LogisticStacker":
        X = _as2d(X)
        y = np.asarray(y, dtype="float64").reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X/y 行数不一致")
        w = np.ones(X.shape[0]) if sample_weight is None else np.asarray(
            sample_weight, dtype="float64").reshape(-1)
        w = np.clip(w, 1e-12, None)
        n, d = X.shape
        self.coef_ = np.zeros(d, dtype="float64")
        self.bias_ = 0.0
        for _ in range(self.iters):
            z = X @ self.coef_ + self.bias_
            p = self._sigmoid(z)
            g = (p - y) * w
            grad_w = X.T @ g / max(w.sum(), 1e-12) + self.l2 * self.coef_ / n
            grad_b = float(g.sum() / max(w.sum(), 1e-12)) if self.fit_bias else 0.0
            self.coef_ -= self.lr * grad_w
            if self.fit_bias:
                self.bias_ -= self.lr * grad_b
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticStacker 尚未 fit")
        return self._sigmoid(_as2d(X) @ self.coef_ + self.bias_)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= float(threshold)).astype("int64")


@dataclass
class StackingReport:
    kind: str
    n_features: int
    n_train: int
    available: bool = True
    notes: str = ""
    coef: list[float] = field(default_factory=list)
    bias: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "n_features": int(self.n_features),
                "n_train": int(self.n_train), "available": bool(self.available),
                "coef": [float(x) for x in self.coef], "bias": float(self.bias),
                "notes": self.notes}


def make_stacker(kind: str = "ridge", **kw):
    k = str(kind).lower()
    if k in ("ridge", "linear"):
        return RidgeStacker(**kw)
    if k in ("logistic", "lr"):
        return LogisticStacker(**kw)
    if k in ("histgb", "hist_gb", "gbdt", "sklearn"):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except Exception as exc:                       # 显式降级，不静默
            raise RuntimeError(f"HistGB 需要 scikit-learn：{type(exc).__name__}: {exc}")
        return HistGradientBoostingRegressor(**kw)
    raise ValueError(f"unknown stacker kind: {kind!r}")


def build_stacking_features(members: Mapping[str, Mapping[str, Any]],
                            keys: Sequence[str] = ("por", "perm_z", "sw", "q_atom")
                            ) -> tuple[list[str], np.ndarray]:
    """把成员预测拼成 (N, n_members*n_keys) 特征；返回 (feature_names, X)。"""
    names: list[str] = []
    cols: list[np.ndarray] = []
    ms = sorted(members)
    if not ms:
        return [], np.zeros((0, 0))
    n = None
    for m in ms:
        for key in keys:
            if key in members[m]:
                v = np.asarray(members[m][key], dtype="float64")
                v2 = v.reshape(-1, 1) if v.ndim == 1 else v
                if n is None:
                    n = v2.shape[0]
                if v2.shape[0] != n:
                    raise ValueError("成员行数不一致")
                for j in range(v2.shape[1]):
                    names.append(f"{m}:{key}:{j}")
                cols.append(v2)
    X = np.concatenate(cols, axis=1) if cols else np.zeros((n or 0, 0))
    return names, X


def cv_predict_stacker(X: Any, y: Any, folds: Any, kind: str = "ridge",
                       l2: float = 1.0) -> dict[str, Any]:
    """折内交叉拟合：用 K-1 折训练、留出折预测，返回 OOF 预测与每折模型系数。"""
    X = _as2d(X)
    y = np.asarray(y, dtype="float64").reshape(-1)
    folds = np.asarray(folds, dtype="int64").reshape(-1)
    if X.shape[0] != y.shape[0] or folds.shape[0] != y.shape[0]:
        raise ValueError("X/y/folds 行数不一致")
    oof = np.full(y.shape[0], np.nan, dtype="float64")
    coefs = []
    info = []
    for k in sorted(set(int(v) for v in folds)):
        tr, va = folds != k, folds == k
        model = make_stacker(kind, l2=l2).fit(X[tr], y[tr])
        oof[va] = model.predict(X[va]) if hasattr(model, "predict") else np.nan
        coef = getattr(model, "coef_", None)
        coefs.append(None if coef is None else np.asarray(coef).tolist())
        info.append({"fold": int(k), "n_train": int(tr.sum()), "n_val": int(va.sum())})
    return {"oof": oof, "coefs": coefs, "folds": info}
