"""E8/P2 集成融合（**纯 numpy**，不 import torch）：权重选择、同源性报告、显著性与 Gate 判据。

三条纪律
--------
1. **权重只在 inner-OOF 选**（`select_weights_inner`）：outer 折只参与最终评分一次；
   任何"看全折再挑权重"的写法都会把集成变成过拟合（H1 选择协议）。
2. **同源平均不算增益**（`homology_report`）：同折不同 seed / 同结构不同快照高度相关，
   把它们平均后与"最佳单成员"的差可能只是噪声；报告必须给出逐对相关性/一致性，
   并把"高度同源"的成员显式标记（`same_source=true`），警告不得计入增益。
3. **增益必须过显著性**（`paired_bootstrap_delta`）：按井加权的配对 cluster bootstrap，
   CI 下界 ≤ 0 → `decision="no_go"`，**不允许**用点估计宣称提升。

融合对象是**连续头输出**（`por`/`perm_z`/`sw`）与原子概率 `q_atom`；原子硬切换在融合**之后**
统一做（`features.basic.decode_predictions` / `inference.atomic_gate`），否则成员各自的 τ
会被混进平均值里，等于在原子值附近插值（明令禁止）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .. import constants as C

BLEND_KEYS: tuple[str, ...] = ("por", "perm_z", "sw")
PROB_KEYS: tuple[str, ...] = ("q_atom", "q_joint")


@dataclass
class EnsembleReport:
    weights: dict[str, float]
    members: list[str]
    fused_total: float
    best_member: str
    best_member_total: float
    delta: float
    ci: tuple[float, float]
    decision: str
    homology: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "members": list(self.members),
                "fused_total": float(self.fused_total), "best_member": self.best_member,
                "best_member_total": float(self.best_member_total),
                "delta": float(self.delta), "paired_ci": [float(self.ci[0]), float(self.ci[1])],
                "decision": self.decision, "homology": self.homology,
                "rule": "集成 ≥ 最佳单成员 且 CI 下界 > 0 才采纳；同源平均不得计为增益"}


# ---------------------------------------------------------------- 融合
def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """非负归一（和为 1）；全零时抛错（不允许"零权重成员"混进来）。"""
    w = {k: max(float(v), 0.0) for k, v in weights.items()}
    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("weights 之和为 0：无法归一")
    return {k: v / s for k, v in w.items()}


def blend_predictions(preds: Mapping[str, Mapping[str, "np.ndarray"]],
                      weights: Mapping[str, float]) -> dict[str, "np.ndarray"]:
    """按权重融合连续头与门控概率（**原子切换留到融合之后**）。

    `preds[member][key]` 形状 `(n,)` 或 `(n,3)`；所有成员必须行数一致（同折同井序）。
    """
    w = normalize_weights(weights)
    members = [m for m in preds if m in w]
    if not members:
        raise ValueError("没有可用的成员（weights 与 preds 键不匹配）")
    n = np.asarray(preds[members[0]]["por"]).shape[0]
    out: dict[str, Any] = {}
    for key in BLEND_KEYS:
        acc = np.zeros(n, dtype="float64")
        for m in members:
            v = np.asarray(preds[m][key], dtype="float64")
            if v.shape[0] != n:
                raise ValueError(f"{m}:{key} 行数 {v.shape[0]} != {n}（成员未对齐同一折？）")
            acc += w[m] * v
        out[key] = acc
    for key in PROB_KEYS:
        if all(key in preds[m] for m in members):
            shp = np.asarray(preds[members[0]][key]).shape
            if any(np.asarray(preds[m][key]).shape != shp for m in members):
                raise ValueError(f"{key} 形状不一致：{shp}")
            acc = np.zeros(shp, dtype="float64")
            for m in members:
                acc += w[m] * np.asarray(preds[m][key], dtype="float64")
            out[key] = acc
    return out


def score_prediction(pred: Mapping[str, "np.ndarray"], y_true: "np.ndarray",
                     mask: "np.ndarray", tau: Sequence[float] | None = None) -> dict[str, float]:
    """把（连续+门控）预测解码成标签尺度并**用官方口径**打分（`missing_mode="drop"`）。"""
    from ..score import score_arrays
    from ..inference.atomic_gate import per_target_hard_switch

    cont = np.stack([
        np.asarray(pred["por"], dtype="float64"),
        np.power(10.0, np.clip(np.asarray(pred["perm_z"], dtype="float64"),
                               C.PERM_LOG_MIN, C.PERM_LOG_MAX)),
        np.clip(np.asarray(pred["sw"], dtype="float64"), 0.0, 100.0),
    ], axis=1)
    if tau is not None and "q_atom" in pred:
        cont = per_target_hard_switch(cont, np.asarray(pred["q_atom"], dtype="float64"), tau)
    m = ~np.asarray(mask, dtype=bool)
    s = score_arrays(np.asarray(y_true, dtype="float64"), cont, missing=m,
                     missing_mode=C.SCORE_MISSING_MODE)
    return {k: float(v) for k, v in s.items() if k != "missing_mode"}


# ---------------------------------------------------------------- 权重选择
def select_weights_inner(inner_preds: Mapping[str, Mapping[str, "np.ndarray"]],
                         y_true: "np.ndarray", mask: "np.ndarray",
                         tau: Sequence[float] | None = None,
                         grid_step: float = 0.1) -> dict[str, Any]:
    """在 **inner-OOF** 上按官方总分选非负权重（单纯形网格）。

    `grid_step` 越小越贵；默认 0.1（成员 ≤ 5 时组合数可控）。返回
    `{weights, inner_total, candidates}`；**调用方必须保证 `inner_preds` 来自 inner-OOF**。
    """
    members = sorted(inner_preds)
    if not members:
        raise ValueError("inner_preds 为空")
    if len(members) == 1:
        w = {members[0]: 1.0}
        sc = score_prediction(inner_preds[members[0]], y_true, mask, tau)
        return {"weights": w, "inner_total": sc["total"], "candidates": 1}
    steps = max(int(round(1.0 / float(grid_step))), 1)

    def simplex(k: int, total: int) -> list[tuple[int, ...]]:
        if k == 1:
            return [(total,)]
        out = []
        for i in range(total + 1):
            for rest in simplex(k - 1, total - i):
                out.append((i,) + rest)
        return out

    best = {"total": float("-inf"), "weights": None}
    n_cand = 0
    for combo in simplex(len(members), steps):
        w = {m: c / steps for m, c in zip(members, combo)}
        if sum(w.values()) <= 0:
            continue
        n_cand += 1
        fused = blend_predictions(inner_preds, w)
        sc = score_prediction(fused, y_true, mask, tau)
        if sc["total"] > best["total"]:
            best = {"total": sc["total"], "weights": w}
    return {"weights": best["weights"], "inner_total": float(best["total"]),
            "candidates": n_cand, "grid_step": float(grid_step), "members": members}


# ---------------------------------------------------------------- 同源性
def _corr_or_none(x: "np.ndarray", y: "np.ndarray", rel_tol: float = 1e-9) -> float | None:
    """皮尔逊相关；任一侧**数值上近似常数**时返回 None（否则浮点残差会伪造成 corr≈1）。"""
    if x.size < 2:
        return None
    scale = max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if float(np.std(x)) <= rel_tol * scale or float(np.std(y)) <= rel_tol * scale:
        return None
    v = float(np.corrcoef(x, y)[0, 1])
    return None if not np.isfinite(v) else v


def homology_report(preds: Mapping[str, Mapping[str, "np.ndarray"]],
                    threshold: float = 0.99) -> dict[str, Any]:
    """成员两两相关性（连续头逐目标）→ 同源标记。

    `same_source=true` 的成员对高度同源（同折不同 seed / 同结构相邻快照），
    计划明确要求"同源平均不得计为增益"，因此报告必须显式列出。
    """
    members = sorted(preds)
    pairs: dict[str, Any] = {}
    same_source: list[list[str]] = []
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            per_key = {}
            for key in BLEND_KEYS:
                x = np.asarray(preds[a][key], dtype="float64").ravel()
                y = np.asarray(preds[b][key], dtype="float64").ravel()
                per_key[key] = _corr_or_none(x, y)
            vals = [v for v in per_key.values() if v is not None]
            mean_corr = float(np.mean(vals)) if vals else None
            key = f"{a}|{b}"
            pairs[key] = {"per_target": per_key, "mean_corr": mean_corr,
                          "same_source": bool(mean_corr is not None
                                              and mean_corr >= float(threshold))}
            if pairs[key]["same_source"]:
                same_source.append([a, b])
    return {"n_members": len(members), "members": members, "pairs": pairs,
            "same_source_pairs": same_source, "threshold": float(threshold),
            "warning": ("存在高度同源成员：平均它们**不得**计为集成增益，"
                        "必须在报告中标注并单独对照最佳单成员"
                        if same_source else "未发现高度同源成员对")}


# ---------------------------------------------------------------- 显著性与 Gate
def paired_bootstrap_delta(fused_well_totals: "np.ndarray", best_well_totals: "np.ndarray",
                           weights: "np.ndarray | None" = None, iters: int = 1000,
                           alpha: float = 0.05, seed: int = 42) -> dict[str, float]:
    """按井加权的配对 cluster bootstrap（集成分 − 最佳单成员分）。"""
    from ..validation.folds import bootstrap_ci
    d = np.asarray(fused_well_totals, dtype="float64") - np.asarray(best_well_totals,
                                                                   dtype="float64")
    out = bootstrap_ci(d, iters=int(iters), alpha=float(alpha), weights=weights, seed=seed)
    return {k: (float(v) if not isinstance(v, bool) else v) for k, v in out.items()}


def well_totals(pred: Mapping[str, "np.ndarray"], y_true: "np.ndarray", mask: "np.ndarray",
                well_index: "np.ndarray", n_wells: int,
                tau: Sequence[float] | None = None) -> "np.ndarray":
    """逐井官方 Total（bootstrap 的 cluster 单元）。"""
    tot = np.full(int(n_wells), np.nan, dtype="float64")
    for w in range(int(n_wells)):
        sel = np.asarray(well_index) == w
        if not sel.any():
            continue
        sc = score_prediction({k: np.asarray(v)[sel] for k, v in pred.items()
                               if k in (*BLEND_KEYS, *PROB_KEYS)},
                              np.asarray(y_true)[sel], np.asarray(mask)[sel], tau)
        tot[w] = sc["total"]
    return tot


def judge_ensemble(fused: Mapping[str, "np.ndarray"], members: Mapping[str, Mapping[str, "np.ndarray"]],
                   y_true: "np.ndarray", mask: "np.ndarray", well_index: "np.ndarray",
                   n_wells: int, tau: Sequence[float] | None = None,
                   weights: Mapping[str, float] | None = None, iters: int = 1000,
                   seed: int = 42) -> EnsembleReport:
    """完整判定：融合分 vs 最佳单成员 + 配对 CI + 同源性（E8 §7 的硬判据）。"""
    fused_sc = score_prediction(fused, y_true, mask, tau)
    member_sc = {m: score_prediction(members[m], y_true, mask, tau) for m in members}
    best_member = max(member_sc, key=lambda m: member_sc[m]["total"])
    f_tot = well_totals(fused, y_true, mask, well_index, n_wells, tau)
    b_tot = well_totals(members[best_member], y_true, mask, well_index, n_wells, tau)
    ok = ~np.isnan(f_tot) & ~np.isnan(b_tot)
    rows = np.array([int((np.asarray(well_index) == w).sum()) for w in range(int(n_wells))],
                    dtype="float64")[ok]
    boot = paired_bootstrap_delta(f_tot[ok], b_tot[ok], weights=rows, iters=iters, seed=seed)
    delta = float(fused_sc["total"] - member_sc[best_member]["total"])
    adopted = bool(boot["ci_low"] > 0 and delta > 0)
    hom = homology_report(members)
    w = normalize_weights(weights) if weights else {m: 1.0 / len(members) for m in members}
    return EnsembleReport(
        weights=w, members=sorted(members), fused_total=float(fused_sc["total"]),
        best_member=best_member, best_member_total=float(member_sc[best_member]["total"]),
        delta=delta, ci=(boot["ci_low"], boot["ci_high"]),
        decision=("adopted" if adopted else "no_go"),
        homology={**hom, "member_totals": {m: member_sc[m]["total"] for m in member_sc}})


def ema_update(shadow: Mapping[str, "np.ndarray"], current: Mapping[str, "np.ndarray"],
               decay: float = 0.999) -> dict[str, "np.ndarray"]:
    """EMA 影子权重更新（`θ_ema ← decay·θ_ema + (1−decay)·θ`）——纯 numpy，便于单测。"""
    d = float(decay)
    if not (0.0 <= d < 1.0):
        raise ValueError(f"decay 必须在 [0,1)，got {decay}")
    return {k: d * np.asarray(shadow[k], dtype="float64")
            + (1.0 - d) * np.asarray(current[k], dtype="float64") for k in current}


def swa_average(states: Sequence[Mapping[str, "np.ndarray"]]) -> dict[str, "np.ndarray"]:
    """SWA 等权平均（E8 只作对照；BN 统计需在平均后重新估计，报告里必须写明）。"""
    if not states:
        raise ValueError("swa_average: 空输入")
    keys = list(states[0])
    return {k: np.mean([np.asarray(s[k], dtype="float64") for s in states], axis=0)
            for k in keys}
