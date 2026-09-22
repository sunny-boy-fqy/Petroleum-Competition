"""期望分数原子决策（纯 numpy，不 import torch）。

背景
----
占位行占训练集 ~66.7%，官方总分又对"精确写出原子值"给满分，因此原子决策是最大杠杆。
旧实现是**逐目标一个全局 τ**；本模块把它升级为：

    q_atom（可先做温度/等渗校准） → 按 q 分桶 → 桶内比较
    "输出原子值" vs "输出连续值" 的官方平均分 → 选择期望分更高的动作

桶间用 PAV 单调约束（q 越高越倾向 atom），并用最小增益/最小样本数抑制过拟合。
**所有参数只允许在 inner-OOF 上拟合**；本模块只做数学，不接触测试标签。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .. import constants as C


def _official_score_fn(target: str):
    """返回逐目标官方 score 函数 ``f(y_true, y_pred, mask) -> float``。"""
    from ..score import acc_perm, acc_relative

    t = str(target).upper()

    def _fn(y, pred, mask=None) -> float:
        miss = None if mask is None else (np.asarray(mask, dtype="float64") <= 0)
        if t == "POR":
            return float(acc_relative(y, pred, C.DELTA_POR, missing_mask=miss))
        if t == "SW":
            return float(acc_relative(y, pred, C.DELTA_SW, missing_mask=miss))
        if t == "PERM":
            return float(acc_perm(y, pred, missing_mask=miss))
        raise ValueError(f"unknown target {target!r}")

    return _fn


def default_score_fns(targets: Sequence[str] = C.TARGETS) -> dict[str, Callable]:
    return {t: _official_score_fn(t) for t in targets}


def _pav_nondecreasing(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """PAV 非降拟合（逐点权重）。"""
    v = np.asarray(values, dtype="float64").reshape(-1)
    n = v.size
    if n == 0:
        return v.copy()
    w = np.ones(n) if weights is None else np.asarray(weights, dtype="float64").reshape(-1)
    stack: list[list[float]] = []
    for i in range(n):
        cur = [float(v[i]), float(w[i]), float(i), float(i)]
        while stack and stack[-1][0] > cur[0]:
            pv, pw, ps, _pe = stack.pop()
            tw = pw + cur[1]
            cur = [(pv * pw + cur[0] * cur[1]) / max(tw, 1e-12), tw, ps, cur[3]]
        stack.append(cur)
    out = np.empty(n, dtype="float64")
    for val, _wt, ps, pe in stack:
        out[int(ps):int(pe) + 1] = val
    return out


def _quantile_edges(scores: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(scores, qs))
    if edges.size < 2:
        edges = np.asarray([float(scores.min()) - 1e-9, float(scores.max()) + 1e-9])
    # 去重后若不足 n_bins+1，补足（不引入非法重复边界）
    if edges.size < 2:
        edges = np.asarray([0.0, 1.0])
    return edges


def _bin_index(q: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, q, side="right") - 1
    return np.clip(idx, 0, max(edges.size - 2, 0))


def fit_decision_table(cont: Any, q_atom: Any, y: Any, mask: Any,
                       score_fns: Mapping[str, Callable] | None = None,
                       n_bins: int = 20, min_bin: int = 200,
                       min_gain: float = 0.0, atom_values: Mapping[str, float] | None = None,
                       ) -> dict[str, Any]:
    """在 inner-OOF 上拟合逐目标期望分数动作表。

    Parameters
    ----------
    cont, q_atom, y, mask : (N,3)
    score_fns : {target: f(y,pred,mask)}；缺省用官方 POR/PERM/SW 分数
    n_bins : 初始分箱数；若某箱样本 < min_bin，会减少箱数直到满足
    min_gain : 切 atom 所需的 ``score_atom - score_cont`` 最小增益（安全边际）
    """
    c = np.asarray(cont, dtype="float64")
    q = np.asarray(q_atom, dtype="float64")
    yy = np.asarray(y, dtype="float64")
    m = np.asarray(mask, dtype="float64") > 0
    if not (c.shape == q.shape == yy.shape == m.shape) or c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"cont/q_atom/y/mask 必须同形 (N,3)，got "
                         f"{c.shape}/{q.shape}/{yy.shape}/{m.shape}")
    fns = dict(default_score_fns()) if score_fns is None else dict(score_fns)
    av = {t: float(C.ATOM_VALUES[t]) for t in C.TARGETS}
    if atom_values is not None:
        av.update({k: float(v) for k, v in atom_values.items()})

    out: dict[str, Any] = {
        "kind": "expected_score_binned",
        "n_bins": int(n_bins), "min_bin": int(min_bin), "min_gain": float(min_gain),
        "targets": {},
    }
    for j, t in enumerate(C.TARGETS):
        obs = m[:, j] & np.isfinite(yy[:, j]) & np.isfinite(c[:, j]) & np.isfinite(q[:, j])
        n_obs = int(obs.sum())
        atom = av[t]
        atom_pred = np.full(n_obs, atom, dtype="float64")
        yv = yy[obs, j]
        cv = c[obs, j]
        qv = np.clip(q[obs, j], 0.0, 1.0)
        fn = fns.get(t) or default_score_fns([t])[t]
        global_atom = float(fn(yv, atom_pred)) if n_obs else float("nan")
        global_cont = float(fn(yv, cv)) if n_obs else float("nan")

        # 箱数自适应：样本太少就减少箱数
        nb = int(n_bins)
        while nb > 1 and n_obs / nb < max(int(min_bin), 1):
            nb -= 1
        if n_obs == 0:
            edges = np.asarray([0.0, 1.0])
            actions = np.asarray([False], dtype=bool)
            deltas = np.asarray([0.0])
            raw = np.asarray([0.0])
            counts = np.asarray([0])
        else:
            edges = _quantile_edges(qv, nb) if nb > 1 else np.asarray(
                [float(qv.min()) - 1e-9, float(qv.max()) + 1e-9])
            bidx = _bin_index(qv, edges)
            n_blocks = edges.size - 1
            raw = np.zeros(n_blocks, dtype="float64")
            counts = np.zeros(n_blocks, dtype="float64")
            for b in range(n_blocks):
                sel = bidx == b
                n_b = int(sel.sum())
                counts[b] = n_b
                if n_b == 0:
                    raw[b] = np.nan
                    continue
                s_atom = float(fn(yv[sel], atom_pred[sel]))
                s_cont = float(fn(yv[sel], cv[sel]))
                raw[b] = s_atom - s_cont
            base = float(np.nanmean(raw)) if np.isfinite(raw).any() else 0.0
            raw = np.where(np.isfinite(raw), raw, base)
            # 单调约束：q 越大，delta 非降；再按 min_gain 决定动作
            smoothed = _pav_nondecreasing(raw, weights=np.maximum(counts, 1.0))
            deltas = smoothed
            actions = smoothed > float(min_gain)
        out["targets"][t] = {
            "bin_edges": [float(x) for x in edges],
            "bin_actions": [bool(x) for x in actions],
            "bin_delta": [float(x) for x in deltas],
            "bin_delta_raw": [float(x) for x in (raw if n_obs else [0.0])],
            "bin_n": [int(x) for x in (counts if n_obs else [0])],
            "atom_value": float(atom),
            "score_atom_global": global_atom,
            "score_cont_global": global_cont,
            "n_observed": n_obs,
        }
    return out


def apply_decision_table(cont: Any, q_atom: Any, table: Mapping[str, Any],
                         out: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """按动作表决策；返回 ``(pred, actions)``，actions 形状 (N,3) bool。

    没有表/字段缺失时回退连续值（保守）。
    """
    c = np.asarray(cont, dtype="float64")
    q = np.asarray(q_atom, dtype="float64")
    if c.shape != q.shape or c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"cont/q_atom 必须同形 (N,3)，got {c.shape}/{q.shape}")
    pred = c.copy() if out is None else np.array(out, dtype="float64", copy=True)
    actions = np.zeros(c.shape, dtype=bool)
    targets = (table or {}).get("targets") or {}
    for j, t in enumerate(C.TARGETS):
        spec = targets.get(t)
        if not isinstance(spec, dict):
            continue
        edges = np.asarray(spec.get("bin_edges") or [0.0, 1.0], dtype="float64")
        acts = np.asarray(spec.get("bin_actions") or [], dtype=bool)
        if edges.size < 2 or acts.size == 0:
            continue
        qv = np.clip(q[:, j], 0.0, 1.0)
        bidx = _bin_index(qv, edges)
        a = acts[np.clip(bidx, 0, acts.size - 1)]
        a = a & np.isfinite(q[:, j])
        pred[a, j] = float(spec.get("atom_value", C.ATOM_VALUES[t]))
        actions[a, j] = True
    return pred, actions


@dataclass
class ExpectedScoreDecision:
    """``fit_decision_table`` 的对象封装，便于写回 decode_v1.json。"""
    table: dict[str, Any] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, cont, q_atom, y, mask, **kw) -> "ExpectedScoreDecision":
        self.table = fit_decision_table(cont, q_atom, y, mask, **kw)
        self.fitted = True
        return self

    def predict(self, cont, q_atom) -> tuple[np.ndarray, np.ndarray]:
        return apply_decision_table(cont, q_atom, self.table)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.table)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ExpectedScoreDecision":
        return ExpectedScoreDecision(table=dict(d or {}), fitted=bool(d))
