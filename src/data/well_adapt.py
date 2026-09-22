"""WP8：井间自适应（输入特征分布匹配）——参考 SPWLA 2021 UTFE/MoLPhy。

方法：
  * ``quantile_match``：把源井每个特征单调映射到参考井的经验分位（101 分位 + 线性插值）；
  * ``linear_match``：仅用均值/标准差做线性平移缩放（MoLPhy 的 histogram matching 简化版）；
  * ``blend_match``：输出 = (1-α)·原始 + α·匹配，控制适配强度。

只允许在**测试输入**上做适配，不得使用测试标签；参数必须从训练/参考井获得。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _col_finite(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float64")
    return v[np.isfinite(v)]


def linear_match(src: Any, ref: Any, alpha: float = 1.0) -> np.ndarray:
    """逐列 mean/std 匹配：``(x-μs)/σs * σr + μr``。"""
    s = np.asarray(src, dtype="float64")
    r = np.asarray(ref, dtype="float64")
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if s.shape[1] != r.shape[1]:
        raise ValueError(f"src/ref 列数不一致：{s.shape[1]} vs {r.shape[1]}")
    out = s.copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    for j in range(s.shape[1]):
        rv = _col_finite(r[:, j])
        if rv.size < 2:
            continue
        mu_r, sd_r = float(rv.mean()), float(rv.std())
        sv = s[:, j]
        finite = np.isfinite(sv)
        if finite.sum() < 2:
            continue
        mu_s, sd_s = float(sv[finite].mean()), float(sv[finite].std())
        if sd_s < 1e-12:
            continue
        matched = (sv[finite] - mu_s) / sd_s * max(sd_r, 1e-12) + mu_r
        out[finite, j] = (1.0 - a) * sv[finite] + a * matched
    return out


def quantile_match(src: Any, ref: Any, alpha: float = 1.0,
                   n_quantiles: int = 101) -> np.ndarray:
    """逐列经验分位匹配（单调）。"""
    s = np.asarray(src, dtype="float64")
    r = np.asarray(ref, dtype="float64")
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if s.shape[1] != r.shape[1]:
        raise ValueError(f"src/ref 列数不一致：{s.shape[1]} vs {r.shape[1]}")
    out = s.copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    qs = np.linspace(0.0, 1.0, int(n_quantiles))
    for j in range(s.shape[1]):
        rv = _col_finite(r[:, j])
        sv = s[:, j]
        finite = np.isfinite(sv)
        if rv.size < 3 or finite.sum() < 3:
            continue
        q_ref = np.quantile(rv, qs)
        order = np.argsort(np.argsort(sv[finite], kind="mergesort"))
        u = (order + 0.5) / max(finite.sum(), 1)
        matched = np.interp(u, qs, q_ref)
        out[finite, j] = (1.0 - a) * sv[finite] + a * matched
    return out


def match_matrix(src: Any, ref: Any, method: str = "quantile",
                 alpha: float = 1.0) -> np.ndarray:
    m = str(method).lower()
    if m == "linear":
        return linear_match(src, ref, alpha=alpha)
    if m in ("quantile", "quantile_match", "hist"):
        return quantile_match(src, ref, alpha=alpha)
    if m in ("none", "off"):
        return np.asarray(src, dtype="float64").copy()
    raise ValueError(f"unknown method {method!r}; expected none/linear/quantile")


def adapt_shard_inputs(shard: Mapping[str, Any], ref_shard: Mapping[str, Any],
                       method: str = "quantile", alpha: float = 1.0
                       ) -> dict[str, np.ndarray]:
    """返回适配后的 ``inputs`` 与新的 ``missing``；只改输入，不动 depth/targets。"""
    src = np.asarray(shard["inputs"], dtype="float64")
    ref = np.asarray(ref_shard["inputs"], dtype="float64")
    if src.shape[1] != ref.shape[1]:
        raise ValueError("shard/ref inputs 列数不一致")
    out = match_matrix(src, ref, method=method, alpha=alpha)
    return {"inputs": out.astype("float32"),
            "missing": (~np.isfinite(out)).astype("int8")}


def adapt_features(X: Any, ref_X: Any, method: str = "quantile",
                   alpha: float = 1.0) -> np.ndarray:
    """对已经构造好的特征矩阵做同样的列匹配（需同宽）。"""
    return match_matrix(X, ref_X, method=method, alpha=alpha).astype("float32")
