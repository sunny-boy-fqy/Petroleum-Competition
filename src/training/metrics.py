"""训练期评价：连续解码、原子门解码、占位行/原子行指标（与 `score.py` 同源）。

为什么单独一层
--------------
`loop.py` 只负责"把模型训起来"，`train_row.py` 只负责"编排 5 折与写 Gate"；
把"预测 → 官方分数"的口径集中在这里，避免出现
"训练期早停用一个口径、最终 OOF 用另一个口径"的漂移（E1/P1 §5 步 10/11）。

口径纪律
--------
- 连续预测一律经 `features.basic.decode_predictions` 解码（PERM 走 `10**clip(z,-6,6)`，
  SW 只做 `[0,100]` 软保护裁剪，**永不裁到 [0,1]**）；
- 原子门解码用 `inference.atomic_gate.per_target_hard_switch`（**无插值**）；
- 打分一律 `score_arrays(..., missing_mode="drop")`（E0 冻结口径，常数基线 70.490735）。
"""
from __future__ import annotations

from typing import Any

from .. import constants as C
from ..features import basic as F
from ..portability import HAS_NUMPY

if HAS_NUMPY:
    import numpy as np


def label_scale_stack(y_por, y_perm_z, y_sw) -> "np.ndarray":
    """把 (por, log10(perm), sw) 三列拼成**标签尺度** (N,3)。"""
    p = np.asarray(y_por, dtype="float64")
    z = np.clip(np.asarray(y_perm_z, dtype="float64"), C.PERM_LOG_MIN, C.PERM_LOG_MAX)
    s = np.asarray(y_sw, dtype="float64")
    return np.stack([p, np.power(10.0, z), s], axis=1)


def decode_continuous(pred: dict) -> "np.ndarray":
    """模型输出 → 标签尺度连续预测 (N,3)（不做任何门控）。"""
    return np.stack([
        np.asarray(pred["por"], dtype="float64"),
        np.power(10.0, np.clip(np.asarray(pred["perm_z"], dtype="float64"),
                               C.PERM_LOG_MIN, C.PERM_LOG_MAX)),
        np.clip(np.asarray(pred["sw"], dtype="float64"), 0.0, 100.0),
    ], axis=1)


def atom_gate(cont: "np.ndarray", q_atom, tau, atom_values: dict | None = None) -> "np.ndarray":
    """逐目标硬切换（τ 为标量或 (3,)）。"""
    from ..inference.atomic_gate import per_target_hard_switch
    return per_target_hard_switch(np.asarray(cont, dtype="float64"),
                                  np.asarray(q_atom, dtype="float64"), tau, atom_values)


def _hit_rate(y: "np.ndarray", pred: "np.ndarray", t: int) -> float:
    """官方容差内的**命中率**（score == 1 ⟺ 命中），仅用于原子/占位行判据。"""
    if y.size == 0:
        return float("nan")
    if t == 1:  # PERM：1 个数量级
        ratio = np.maximum(pred / np.maximum(y, 1e-300), C.EPS)
        err = np.abs(np.log10(ratio))
    else:
        delta = C.DELTA_POR if t == 0 else C.DELTA_SW
        err = np.abs(pred - y) / (delta * (np.abs(y) + C.EPS))
    return float(np.mean(err <= 1.0))


def atomic_rows_report(y_true: "np.ndarray", y_pred: "np.ndarray", y_atom: "np.ndarray",
                       mask: "np.ndarray") -> dict[str, Any]:
    """原子（占位）行上的逐目标命中率 + 原子头精确率/召回率。"""
    out: dict[str, Any] = {"rows": {}, "hit_rate": {}}
    for t, name in enumerate(C.TARGETS):
        obs = np.asarray(mask, dtype=bool)[:, t]
        sel = obs & (np.asarray(y_atom, dtype=bool)[:, t])
        out["rows"][name] = int(sel.sum())
        out["hit_rate"][name] = _hit_rate(np.asarray(y_true, float)[sel, t],
                                          np.asarray(y_pred, float)[sel, t], t) if sel.any() else None
    return out


def atomic_precision_recall(y_atom: "np.ndarray", q_atom: "np.ndarray", tau,
                           mask: Any | None = None) -> dict[str, Any]:
    """原子头在阈值 τ 下的逐目标 precision/recall/f1（τ 为标量或 (3,)）。

    `mask` 给出时，只统计对应目标被观测到的行；缺测行既不是正例也不是负例。
    """
    y = np.asarray(y_atom, dtype=bool)
    q = np.asarray(q_atom, dtype="float64")
    if q.shape != y.shape:
        raise ValueError(f"y_atom/q_atom 形状不一致：{y.shape} vs {q.shape}")
    taus = np.asarray(tau, dtype="float64").reshape(-1)
    if taus.size == 1:
        taus = np.repeat(taus, 3)
    mm = None if mask is None else np.asarray(mask, dtype=bool)
    out: dict[str, Any] = {}
    for t, name in enumerate(C.TARGETS):
        if mm is None:
            valid = np.ones(y.shape[0], dtype=bool)
        elif mm.ndim == 1:
            valid = mm
        else:
            valid = mm[:, t]
        yv, qv = y[valid, t], q[valid, t]
        pred = qv >= taus[t]
        tp = float(np.sum(pred & yv))
        fp = float(np.sum(pred & ~yv))
        fn = float(np.sum(~pred & yv))
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else float("nan")
        out[name] = {"precision": prec, "recall": rec, "f1": f1, "tau": float(taus[t]),
                     "n_observed": int(valid.sum())}
    return out


def score_of(y_true_label: "np.ndarray", y_pred_label: "np.ndarray",
             mask: "np.ndarray") -> dict[str, float]:
    """官方口径打分（`missing_mode="drop"`，E0 冻结）。"""
    from ..score import score_arrays
    m = ~np.asarray(mask, dtype=bool)
    return score_arrays(y_true_label, y_pred_label, missing=m,
                        missing_mode=C.SCORE_MISSING_MODE)


def evaluate_predictions(y_true_label: "np.ndarray", mask: "np.ndarray", pred: dict,
                         y_atom: "np.ndarray | None" = None,
                         tau=None) -> dict[str, Any]:
    """一次给出：连续口径分数、原子门口径分数、占位行指标、原子头 P/R。

    `tau is None` 时只算连续口径（训练早期/未选 τ 时使用）。
    """
    cont = decode_continuous(pred)
    res: dict[str, Any] = {
        "cont": score_of(y_true_label, cont, mask),
        "n_rows": int(np.asarray(y_true_label).shape[0]),
    }
    if tau is not None:
        gated = atom_gate(cont, pred["q_atom"], tau)
        res["gated"] = score_of(y_true_label, gated, mask)
    else:
        gated = cont
    if y_atom is not None:
        res["atomic_rows"] = atomic_rows_report(y_true_label, gated, y_atom,
                                               np.asarray(mask, dtype=bool))
        if tau is not None:
            res["atomic_head"] = atomic_precision_recall(
                y_atom, pred["q_atom"], tau, mask=np.asarray(mask, dtype=bool))
    return res


def total_of(res: dict[str, Any], use_gated: bool = True) -> float:
    key = "gated" if (use_gated and "gated" in res) else "cont"
    return float(res[key]["total"])


def placeholder_min_acc(res: dict[str, Any]) -> float:
    """占位行三目标命中率的**最小值**（E1 Gate 要求 ≥ 0.98）。"""
    hr = (res.get("atomic_rows") or {}).get("hit_rate") or {}
    vals = [float(v) for v in hr.values() if v is not None]
    return float(min(vals)) if vals else float("nan")


# ---------------------------------------------------------------- AUC / AP（E6 原子头）
def _average_ranks(x: "np.ndarray") -> "np.ndarray":
    """平均秩（并列取均值），用于 rank-based AUC（不依赖 sklearn/scipy）。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype="float64")
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype="float64")
    xs = x[order]
    i = 0
    while i < xs.shape[0]:
        j = i
        while j + 1 < xs.shape[0] and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def binary_auc(y_true: Any, score: Any) -> float | None:
    """ROC-AUC（Mann–Whitney U）；单类标签或空输入返回 `None`（**不返回 0.5 假装有效**）。"""
    if not HAS_NUMPY:
        raise RuntimeError("binary_auc requires numpy")
    y = np.asarray(y_true, dtype=bool).ravel()
    s = np.asarray(score, dtype="float64").ravel()
    if y.size == 0 or y.shape != s.shape:
        return None
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = _average_ranks(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: Any, score: Any) -> float | None:
    """PR-AUC（平均精度，step 积分）；单类标签或空输入返回 `None`。"""
    if not HAS_NUMPY:
        raise RuntimeError("average_precision requires numpy")
    y = np.asarray(y_true, dtype=bool).ravel()
    s = np.asarray(score, dtype="float64").ravel()
    if y.size == 0 or y.shape != s.shape or y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, ys.size + 1)
    return float((precision * ys).sum() / ys.sum())


def auc_report(y_atom: Any, q_atom: Any, targets: tuple[str, ...] = C.TARGETS,
               mask: Any | None = None) -> dict[str, Any]:
    """逐目标 AUC/AP + 联合原子指标（E6/P0 §7 要求**逐目标**上报，不是只报总分）。

    `mask` 给出时，逐目标 AUC 只用该目标被观测到的行；联合原子 AUC 只用三目标
    均被观测到的行。缺测行不得作为负例参与。
    """
    y = np.asarray(y_atom, dtype=bool)
    q = np.asarray(q_atom, dtype="float64")
    if y.ndim != 2 or q.shape != y.shape:
        raise ValueError(f"y_atom/q_atom 必须同形状 (N,3)，got {y.shape}/{q.shape}")
    mm = None if mask is None else np.asarray(mask, dtype=bool)
    per: dict[str, Any] = {}
    for t, name in enumerate(targets):
        if mm is None:
            valid = np.ones(y.shape[0], dtype=bool)
        elif mm.ndim == 1:
            valid = mm
        else:
            valid = mm[:, t]
        per[name] = {"auc": binary_auc(y[valid, t], q[valid, t]),
                     "average_precision": average_precision(y[valid, t], q[valid, t]),
                     "n_pos": int(y[valid, t].sum()), "n": int(valid.sum())}
    if mm is None:
        joint_valid = np.ones(y.shape[0], dtype=bool)
    elif mm.ndim == 1:
        joint_valid = mm
    else:
        joint_valid = mm.all(axis=1)
    joint = y[joint_valid].all(axis=1)
    per["joint"] = {"auc": binary_auc(joint, q[joint_valid].min(axis=1)),
                    "average_precision": average_precision(joint, q[joint_valid].min(axis=1)),
                    "n_pos": int(joint.sum()), "n": int(joint_valid.sum())}
    vals = [v["auc"] for k, v in per.items() if k != "joint" and v["auc"] is not None]
    return {"per_target": per, "min_auc": (min(vals) if vals else None),
            "note": "AUC 用平均秩（并列取均值）；单类标签返回 None 而不是 0.5；"
                    "mask 给出时缺测行不参与"}


def jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组递归转成 JSON 可写类型。"""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if HAS_NUMPY:
        if isinstance(obj, np.ndarray):
            return [jsonable(v) for v in obj.tolist()]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    return obj
