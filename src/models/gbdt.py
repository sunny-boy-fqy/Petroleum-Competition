"""WP11：GBDT 一阶成员（HistGB / LightGBM / XGBoost，全部可选）。

参考 SPWLA 2021 第 2/4/5 名：ExtraTrees/XGBoost/LightGBM/CatBoost 在测井回归上极强。
若规则允许非 DL 成员，本模块提供可直接加入集成的 GBDT 候选；缺失库时显式报错。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


def available_kinds() -> dict[str, bool]:
    out = {"histgb": False, "lgbm": False, "xgboost": False, "catboost": False}
    try:
        import sklearn  # noqa: F401
        out["histgb"] = True
    except Exception:
        pass
    for key, mod in (("lgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost")):
        try:
            __import__(mod)
            out[key] = True
        except Exception:
            pass
    return out


def make_gbdt(kind: str = "histgb", **kw):
    k = str(kind).lower()
    if k in ("histgb", "hist_gb", "sklearn"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**kw)
    if k in ("lgbm", "lightgbm"):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**kw)
    if k in ("xgb", "xgboost"):
        from xgboost import XGBRegressor
        return XGBRegressor(**kw)
    if k in ("cat", "catboost"):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(verbose=False, **kw)
    raise ValueError(f"unknown GBDT kind {kind!r}; available={available_kinds()}")


class MultiTargetGBDT:
    """逐目标独立 GBDT（简单、可解释、易集成）。"""

    def __init__(self, kind: str = "histgb", params: Mapping[str, Any] | None = None):
        self.kind = str(kind)
        self.params = dict(params or {})
        self.models_: list[Any] = []
        self.fitted = False

    def fit(self, X: Any, Y: Any, mask: Any | None = None,
            sample_weight: Any | None = None) -> "MultiTargetGBDT":
        X = np.asarray(X, dtype="float64")
        Y = np.asarray(Y, dtype="float64")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        m = np.ones_like(Y, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self.models_ = []
        for t in range(Y.shape[1]):
            obs = m[:, t]
            if obs.sum() < 2:
                raise ValueError(f"目标 {t} 可观测行不足")
            model = make_gbdt(self.kind, **self.params)
            sw = None if sample_weight is None else np.asarray(sample_weight)[obs]
            try:
                model.fit(X[obs], Y[obs, t], sample_weight=sw)
            except TypeError:
                model.fit(X[obs], Y[obs, t])
            self.models_.append(model)
        self.fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MultiTargetGBDT 尚未 fit")
        X = np.asarray(X, dtype="float64")
        return np.stack([np.asarray(m.predict(X), dtype="float64").reshape(-1)
                         for m in self.models_], axis=1)


def fit_multi_target(X: Any, Y: Any, mask: Any | None = None,
                     kind: str = "histgb", params: Mapping[str, Any] | None = None,
                     sample_weight: Any | None = None) -> MultiTargetGBDT:
    return MultiTargetGBDT(kind=kind, params=params).fit(
        X, Y, mask=mask, sample_weight=sample_weight)
