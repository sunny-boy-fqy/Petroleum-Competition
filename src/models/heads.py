"""序列头（E3/P1 §5 步 3）：把主干输出 (B,L,d) 变成与 `RowMLP` **同键**的多任务输出。

为什么必须同键：`losses.score_aligned.total_loss` / `features.basic.decode_predictions` /
`inference.atomic_gate` / `training.metrics` 全部按 `RowMLP.forward` 的键约定工作。
序列主干只要复用这套键，损失、解码、τ 选择、契约检查就**一行都不用改**，
也不存在"序列模型偷偷用了另一套口径"的风险。

契约（与 `RowMLP` 完全一致）::

    por (B,L)          = por_max·sigmoid(cont_por)
    perm_z (B,L)       = 6·tanh(cont_perm)
    sw (B,L)           = sw_mu + sw_sigma·cont_sw
    q_atom (B,L,3)        概率（门控专用）
    q_joint (B,L)         概率（门控专用）
    q_atom_logit (B,L,3)  logits（BCEWithLogits 专用）
    q_joint_logit (B,L)   logits（BCEWithLogits 专用）
    ph_logit (B,L)        = q_joint_logit（旧键别名）
"""
from __future__ import annotations

from typing import Any

from .. import constants as C
from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn as nn


if HAS_TORCH:

    class SeqHead(nn.Module):
        """逐行共享的连续头 + 逐目标原子头 + 联合头（对 (B,L,d) 逐点作用）。"""

        def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1,
                     perm_log_abs: float = 6.0, por_max: float | None = None,
                     sw_mu: float | None = None, sw_sigma: float | None = None):
            super().__init__()
            self.perm_log_abs = float(perm_log_abs)
            self.register_buffer("por_max", torch.tensor(float(
                por_max if por_max is not None else C.POR_MAX_BUFFER * C.POR_VALID_MAX)))
            self.register_buffer("sw_mu", torch.tensor(float(
                sw_mu if sw_mu is not None else C.SW_VALID_MEDIAN)))
            self.register_buffer("sw_sigma", torch.tensor(float(
                sw_sigma if sw_sigma is not None else 20.0)))

            def _mlp(out_dim: int) -> nn.Sequential:
                return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, out_dim))

            self.cont_por = _mlp(1)
            self.cont_perm = _mlp(1)
            self.cont_sw = _mlp(1)
            self.q_atom = _mlp(3)
            self.q_joint = _mlp(1)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=0.01)

        def forward(self, x):
            require("torch")
            por = self.por_max * torch.sigmoid(self.cont_por(x).squeeze(-1))
            perm_z = self.perm_log_abs * torch.tanh(self.cont_perm(x).squeeze(-1))
            sw = self.sw_mu + self.sw_sigma * self.cont_sw(x).squeeze(-1)
            q_atom_logit = self.q_atom(x)
            q_joint_logit = self.q_joint(x).squeeze(-1)
            return {
                "por": por, "perm_z": perm_z, "sw": sw,
                "q_atom": torch.sigmoid(q_atom_logit),
                "q_joint": torch.sigmoid(q_joint_logit),
                "q_atom_logit": q_atom_logit,
                "q_joint_logit": q_joint_logit,
                "ph_logit": q_joint_logit,
            }

        @torch.no_grad()
        def init_from_stats(self, por_median: float = 11.34, por_max: float = 39.8,
                            sw_mu: float = 82.805, sw_sigma: float = 20.0,
                            perm_z_median: float = -0.08, joint_atom_rate: float = 0.667,
                            atom_rates: tuple = (0.6674, 0.6773, 0.7097)) -> None:
            """与 `models.row_mlp.RowMLP.init_from_stats` 同语义（**训练折**统计）。"""
            import math as _m
            self.por_max.fill_(float(por_max))
            self.sw_mu.fill_(float(sw_mu))
            self.sw_sigma.fill_(max(float(sw_sigma), 1e-6))

            def _logit(p: float) -> float:
                q = min(max(float(p), 1e-6), 1.0 - 1e-6)
                return _m.log(q / (1.0 - q))

            self.cont_por[-1].bias.fill_(_logit(float(por_median) / max(float(por_max), 1e-9)))
            z_ratio = float(perm_z_median) / max(self.perm_log_abs, 1e-9)
            self.cont_perm[-1].bias.fill_(_m.atanh(min(max(z_ratio, -0.999999), 0.999999)))
            self.cont_sw[-1].bias.fill_(0.0)
            self.q_joint[-1].bias.fill_(_logit(joint_atom_rate))
            with torch.no_grad():
                self.q_atom[-1].bias.copy_(
                    torch.tensor([_logit(r) for r in tuple(atom_rates)]))


INIT_STATS_KEYS: tuple[str, ...] = (
    "por_median", "por_max", "sw_mu", "sw_sigma",
    "perm_z_median", "joint_atom_rate", "atom_rates",
)


def init_head_from_stats(head, init_stats: dict | None):
    """把 `features.basic.fit_target_scalers` 的返回直接喂给序列头（未知名忽略并记录）。"""
    if not init_stats:
        return head
    payload = {k: v for k, v in init_stats.items() if k in INIT_STATS_KEYS}
    head.init_from_stats(**payload)
    head.init_stats_ignored = tuple(sorted(k for k in init_stats if k not in INIT_STATS_KEYS))
    return head


def count_parameters(model) -> int:
    require("torch")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def output_length_report(model, n_features: int, lengths=(64, 257, 1024)) -> dict[str, Any]:
    """**全段 seq2seq** 的硬契约：任意长度输入都必须逐行同长输出（含奇数长度）。"""
    require("torch")
    model.eval()
    out: dict[str, Any] = {}
    with torch.no_grad():
        for L in lengths:
            x = torch.zeros(2, int(L), n_features)
            y = model(x)
            out[str(L)] = {k: list(v.shape) for k, v in y.items()}
    ok = all(v["por"][:2] == [2, int(L)] for L, v in
             ((int(k), vv) for k, vv in out.items()))
    return {"ok": bool(ok), "shapes": out}
