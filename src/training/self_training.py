"""WP5：Transductive / 自训练的公共件（纯 numpy；只使用测试**输入**，绝不用测试标签）。

三条红线：
  1. 测试侧传入的任何 dict 含 ``y_true/mask/targets/...`` 等标签键 → ``PermissionError``；
  2. 伪标签只允许来自模型预测 + 测试输入分布（逐井分位/均值对齐）；
  3. 每轮伪标签训练必须能在 inner-OOF/确认折上验证；无正增益则回退。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import constants as C

LABEL_LIKE_KEYS: tuple[str, ...] = (
    "y_true", "targets", "target", "target_missing", "mask", "labels",
    "label", "placeholder", "por", "perm", "sw", "porosity", "permeability",
)


def label_key_guard(obj: Mapping[str, Any], context: str = "test-preds") -> dict[str, Any]:
    """测试侧输入含标签键即拒绝；返回审计报告。"""
    keys = {str(k).lower() for k in obj}
    hits = sorted(k for k in keys if k in LABEL_LIKE_KEYS)
    if hits:
        raise PermissionError(
            f"{context} 含标签类键 {hits}：transductive 只允许使用测试输入分布，"
            "不允许任何测试标签。")
    return {"ok": True, "label_like_keys": [], "n_keys": len(keys)}


@dataclass
class SelfTrainingConfig:
    atom_conf: float = 0.9          # q_atom 校准后切原子的置信阈值
    continuous_conf: float = 0.0    # 连续伪标签置信阈值（0 = 全收）
    max_atom_frac: float = 0.95     # 单井/全局原子伪标签比例上限
    per_well: bool = True
    min_rows_per_well: int = 20
    rounds: int = 1
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"atom_conf": float(self.atom_conf),
                "continuous_conf": float(self.continuous_conf),
                "max_atom_frac": float(self.max_atom_frac),
                "per_well": bool(self.per_well),
                "min_rows_per_well": int(self.min_rows_per_well),
                "rounds": int(self.rounds), "notes": self.notes}


def atom_entropy(q: Any) -> "np.ndarray":
    """二值熵（0=确定，1=最不确定），用于一致性权重。"""
    p = np.clip(np.asarray(q, dtype="float64"), 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)


def select_pseudo_labels(cont: Any, q_atom: Any, well_index: Any,
                         config: SelfTrainingConfig | None = None
                         ) -> dict[str, Any]:
    """按校准后 q_atom 选择原子伪标签；连续伪标签 = 模型 cont。

    返回 ``{atom_label, atom_accept, sample_weight, stats}``，形状均为 (N,3)/(N,)。
    """
    cfg = config or SelfTrainingConfig()
    c = np.asarray(cont, dtype="float64")
    q = np.asarray(q_atom, dtype="float64")
    widx = np.asarray(well_index, dtype="int64")
    if c.shape != q.shape or c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"cont/q_atom 必须 (N,3)，got {c.shape}/{q.shape}")
    n = c.shape[0]
    atom_values = np.asarray([C.ATOM_VALUES[t] for t in C.TARGETS], dtype="float64")
    atom_label = np.where(q >= float(cfg.atom_conf),
                          atom_values[None, :], c).astype("float64")
    accept = np.ones((n, 3), dtype=bool)
    # 原子接受：高置信；连续接受：由 continuous_conf 控制（默认全收，权重由熵决定）
    accept = np.where(q >= float(cfg.atom_conf), True, True)
    if cfg.continuous_conf and float(cfg.continuous_conf) > 0:
        accept &= (q >= float(cfg.atom_conf)) | (np.abs(q - 0.5) >= float(cfg.continuous_conf))
    # 逐井原子比例上限：超过则按 q 排序只保留最高的若干行
    stats = {}
    for j, t in enumerate(C.TARGETS):
        if cfg.per_well:
            for w in np.unique(widx):
                sel = widx == w
                if sel.sum() < int(cfg.min_rows_per_well):
                    continue
                n_atom = int((q[sel, j] >= float(cfg.atom_conf)).sum())
                max_atom = int(float(cfg.max_atom_frac) * sel.sum())
                if n_atom > max_atom and n_atom > 0:
                    thr = np.sort(q[sel, j])[::-1][max(max_atom - 1, 0)]
                    accept[sel, j] = q[sel, j] >= max(thr, float(cfg.atom_conf))
        stats[t] = {
            "n_atom": int((q[:, j] >= float(cfg.atom_conf)).sum()),
            "atom_frac": float((q[:, j] >= float(cfg.atom_conf)).mean()),
            "mean_entropy": float(atom_entropy(q[:, j]).mean()),
        }
    # 一致性权重：越确定权重越高（0.2~1.0）
    ent = atom_entropy(q)
    sample_weight = np.clip(1.0 - 0.8 * ent, 0.2, 1.0)
    return {"atom_label": atom_label, "atom_accept": accept,
            "sample_weight": sample_weight, "config": cfg.as_dict(), "stats": stats}


def well_quantile_align(pred: Any, reference: Any, alpha: float = 0.5) -> "np.ndarray":
    """逐井逐目标分位映射：把 pred 单调映射到 reference 的经验分位（只在测试输入上）。

    ``pred``/``reference`` 形状 ``(N,3)``，按井调用（本函数假定单井）。
    """
    out = np.asarray(pred, dtype="float64").copy()
    ref = np.asarray(reference, dtype="float64")
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0 or ref.shape[0] == 0:
        return out
    qs = np.linspace(0.0, 1.0, 101)
    for j in range(out.shape[1]):
        qv = np.quantile(ref[:, j], qs)
        order = np.argsort(np.argsort(out[:, j], kind="mergesort"))
        u = (order + 0.5) / max(out.shape[0], 1)
        out[:, j] = (1.0 - a) * out[:, j] + a * np.interp(u, qs, qv)
    return out


def per_well_align(pred_by_well: Mapping[str, Any], reference_by_well: Mapping[str, Any],
                   alpha: float = 0.5) -> dict[str, "np.ndarray"]:
    """对每口测试井做分位对齐；键不一致时抛错（防错配）。"""
    if set(pred_by_well) != set(reference_by_well):
        raise ValueError("pred_by_well 与 reference_by_well 的井集合不一致")
    return {w: well_quantile_align(pred_by_well[w], reference_by_well[w], alpha=alpha)
            for w in pred_by_well}


def consistency_regularizer(q_logits_a: Any, q_logits_b: Any) -> float:
    """两次前向（如不同 dropout/TTA）的原子概率一致性（MSE）。"""
    a = 1.0 / (1.0 + np.exp(-np.asarray(q_logits_a, dtype="float64")))
    b = 1.0 / (1.0 + np.exp(-np.asarray(q_logits_b, dtype="float64")))
    return float(np.mean((a - b) ** 2))
