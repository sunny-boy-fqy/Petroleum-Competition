"""原子概率校准（纯 numpy，不依赖 torch）：

- ``fit_temperature`` / ``apply_temperature``：对 logits 做温度缩放（1-D 凸问题，
  用黄金分割搜索，避免 scipy 依赖）。
- ``fit_isotonic_binned`` / ``apply_isotonic_binned``：分箱 + PAV 单调校准，
  对 73 万行也能在秒级完成。
- ``reliability_report``：ECE / Brier / 分箱可靠性，供 Gate 收据使用。

纪律：校准参数只允许在 **inner-OOF** 上拟合；本模块只做数学，不接触任何测试标签。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

EPS = 1e-12


def sigmoid(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p: Any, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def bce_with_logits(logits: Any, y: Any, mask: Any | None = None,
                    sample_weight: Any | None = None) -> float:
    """数值稳定的 BCE（返回标量，越小越好）。"""
    z = np.asarray(logits, dtype="float64")
    t = np.asarray(y, dtype="float64")
    # log(1+exp(z)) - y*z 的稳定写法
    loss = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - t * z
    w = np.ones_like(loss)
    if mask is not None:
        w = w * (np.asarray(mask, dtype="float64") > 0)
    if sample_weight is not None:
        w = w * np.asarray(sample_weight, dtype="float64")
    denom = float(w.sum())
    return float((loss * w).sum() / denom) if denom > 0 else float("nan")


def _golden_section(f, lo: float, hi: float, iters: int = 80) -> float:
    """1-D 单峰函数的最小值（用于温度 T 的搜索）。"""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(int(iters)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def fit_temperature(logits: Any, y: Any, mask: Any | None = None,
                    bounds: tuple[float, float] = (0.05, 20.0)) -> dict[str, float]:
    """温度缩放：最小化 ``BCE(sigmoid(z/T), y)``，返回 ``{"temperature": T}``。"""
    z = np.asarray(logits, dtype="float64").reshape(-1)
    t = np.asarray(y, dtype="float64").reshape(-1)
    if z.shape != t.shape:
        raise ValueError(f"logits/y shape mismatch: {z.shape} vs {t.shape}")
    m = None if mask is None else np.asarray(mask, dtype="float64").reshape(-1)
    if m is not None and m.shape != z.shape:
        raise ValueError("mask shape mismatch")

    def obj(temp: float) -> float:
        return bce_with_logits(z / max(float(temp), EPS), t, m)

    T = _golden_section(obj, *bounds)
    return {"temperature": float(T), "nll": float(obj(T)),
            "n": int(z.size if m is None else (m > 0).sum())}


def fit_temperature_from_probs(probs: Any, y: Any, mask: Any | None = None,
                               **kw) -> dict[str, float]:
    return fit_temperature(logit(probs), y, mask=mask, **kw)


def apply_temperature(logits: Any, temperature: float) -> np.ndarray:
    return sigmoid(np.asarray(logits, dtype="float64") / max(float(temperature), EPS))


def _pav_weighted(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """标准 PAV（支持逐点权重），返回与输入同长的非降拟合。"""
    v = np.asarray(values, dtype="float64").reshape(-1)
    n = v.size
    if n == 0:
        return v.copy()
    w = np.ones(n) if weights is None else np.asarray(weights, dtype="float64").reshape(-1)
    if w.shape != v.shape:
        raise ValueError("weights shape mismatch")
    # 栈元素: [block_value, block_weight, start, end]
    stack: list[list[float]] = []
    for i in range(n):
        cur = [float(v[i]), float(w[i]), float(i), float(i)]
        while stack and stack[-1][0] > cur[0]:
            pv, pw, ps, _pe = stack.pop()
            tw = pw + cur[1]
            cur = [(pv * pw + cur[0] * cur[1]) / max(tw, EPS), tw, ps, cur[3]]
        stack.append(cur)
    out = np.empty(n, dtype="float64")
    for val, _wt, ps, pe in stack:
        out[int(ps):int(pe) + 1] = val
    return out


def fit_isotonic_binned(scores: Any, y: Any, mask: Any | None = None,
                        n_bins: int = 50) -> dict[str, Any]:
    """分箱 + PAV 单调校准；返回可 JSON 化的 ``{bin_edges, bin_values, ...}``。

    ``scores`` 是未校准概率/分数，``y`` 是 0/1。分箱用等频（quantile）边界；
    空箱会被合并。``bin_values`` 单调非降。
    """
    s = np.asarray(scores, dtype="float64").reshape(-1)
    t = np.asarray(y, dtype="float64").reshape(-1)
    if s.shape != t.shape:
        raise ValueError("scores/y shape mismatch")
    m = None if mask is None else np.asarray(mask, dtype="float64").reshape(-1) > 0
    keep = np.isfinite(s) & np.isfinite(t)
    if m is not None:
        keep &= m
    s, t = s[keep], t[keep]
    if s.size == 0:
        return {"bin_edges": [0.0, 1.0], "bin_values": [0.5, 0.5], "n": 0,
                "base_rate": 0.5}
    n_bins = max(int(n_bins), 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(s, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.asarray([float(s.min()) - 1e-9, float(s.max()) + 1e-9])
    # 保证严格递增
    edges = edges + np.linspace(0, 1e-12, edges.size)
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, edges.size - 2)
    vals = np.full(edges.size - 1, np.nan, dtype="float64")
    cnts = np.zeros(edges.size - 1, dtype="float64")
    for b in range(edges.size - 1):
        sel = idx == b
        if sel.any():
            vals[b] = float(t[sel].mean())
            cnts[b] = float(sel.sum())
    good = np.isfinite(vals)
    if not good.any():
        return {"bin_edges": edges.tolist(), "bin_values": [0.5] * (edges.size - 1),
                "n": int(s.size), "base_rate": 0.5}
    # 用全局率填补空箱，再做 PAV
    base = float(t.mean())
    vals[~good] = base
    fitted = _pav_weighted(vals, weights=np.maximum(cnts, 1.0))
    return {"bin_edges": [float(x) for x in edges],
            "bin_values": [float(x) for x in fitted],
            "counts": [int(x) for x in cnts],
            "n": int(s.size), "base_rate": float(base),
            "kind": "isotonic_binned"}


def apply_isotonic_binned(scores: Any, fit: Mapping[str, Any]) -> np.ndarray:
    s = np.asarray(scores, dtype="float64")
    edges = np.asarray(fit["bin_edges"], dtype="float64")
    vals = np.asarray(fit["bin_values"], dtype="float64")
    if edges.size < 2 or vals.size == 0:
        return np.full_like(s, float(fit.get("base_rate", 0.5)))
    # 落在箱内取箱值；两端用最近箱值（np.interp 要求 vals 与 edges 等长）
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(s, centers, vals, left=float(vals[0]), right=float(vals[-1]))


def reliability_report(q: Any, y: Any, mask: Any | None = None,
                       n_bins: int = 15) -> dict[str, Any]:
    """ECE / Brier / 分箱可靠性（用于判断 q_atom 是否校准）。"""
    p = np.asarray(q, dtype="float64").reshape(-1)
    t = np.asarray(y, dtype="float64").reshape(-1)
    m = None if mask is None else np.asarray(mask, dtype="float64").reshape(-1) > 0
    keep = np.isfinite(p) & np.isfinite(t)
    if m is not None:
        keep &= m
    p, t = p[keep], t[keep]
    if p.size == 0:
        return {"n": 0, "ece": None, "brier": None, "bins": []}
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi if i < int(n_bins) - 1 else p <= hi)
        if not sel.any():
            rows.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "mean_q": None, "rate": None})
            continue
        mq, rate, n = float(p[sel].mean()), float(t[sel].mean()), int(sel.sum())
        ece += (n / p.size) * abs(mq - rate)
        rows.append({"lo": float(lo), "hi": float(hi), "n": n,
                     "mean_q": mq, "rate": rate})
    return {"n": int(p.size), "ece": float(ece),
            "brier": float(np.mean((p - t) ** 2)), "bins": rows,
            "mean_q": float(p.mean()), "rate": float(t.mean())}
