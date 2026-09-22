"""WP2：独立原子分类头（可与行级特征或序列隐状态对接）。

输入：
  * 行级 ``(B, d)``；
  * 序列 ``(B, L, d)``（逐行输出 ``(B, L, 3)``）。
输出：
  ``q_atom_logit`` / ``q_atom``（列序 POR/PERM/SW），以及可选的每目标独立头。
"""
from __future__ import annotations

from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn as nn


if HAS_TORCH:

    class AtomClassifierHead(nn.Module):
        def __init__(self, d_in: int, hidden: int = 128, layers: int = 2,
                     dropout: float = 0.1, n_targets: int = 3,
                     separate_heads: bool = False, perm_log_abs: float = 6.0):
            super().__init__()
            self.d_in = int(d_in)
            self.n_targets = int(n_targets)
            self.separate_heads = bool(separate_heads)
            blocks: list[nn.Module] = []
            d = int(d_in)
            for _ in range(max(int(layers), 1)):
                blocks += [nn.Linear(d, int(hidden)), nn.GELU(), nn.Dropout(float(dropout))]
                d = int(hidden)
            self.trunk = nn.Sequential(*blocks)
            if self.separate_heads:
                self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(self.n_targets)])
            else:
                self.head = nn.Linear(d, self.n_targets)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=0.01)

        def forward(self, x):
            require("torch")
            h = self.trunk(x)
            if self.separate_heads:
                logits = torch.cat([head(h) for head in self.heads], dim=-1)
            else:
                logits = self.head(h)
            return {"q_atom_logit": logits, "q_atom": torch.sigmoid(logits)}


def build_atom_head(d_in: int, hidden: int = 128, layers: int = 2,
                    dropout: float = 0.1, separate_heads: bool = False,
                    seed: int | None = None):
    require("torch")
    if seed is not None:
        import torch
        torch.manual_seed(int(seed))
    from .atom_head import AtomClassifierHead
    return AtomClassifierHead(d_in, hidden=hidden, layers=layers, dropout=dropout,
                              separate_heads=separate_heads)
