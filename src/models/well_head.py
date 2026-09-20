"""E8/P1 井级分支（H4）：主干输出 → 井级 attention-pool → 逐目标井级偏置 Δ_t → `ŷ + λ·Δ_t`。

纪律（E8/P1 §5–§8）
------------------
1. **容量受限**：`hidden ≤ d_model // 8`（默认），超过直接抛错——80 口井极易过拟合。
2. **强正则**：默认 dropout 0.2（weight decay 由优化器侧给）。
3. **禁止井身份特征**：输入只能是主干特征；`assert_no_well_identity()` 审计参数/缓冲名，
   任何 `logId`/`well_id` 类键一律视为违规（井身份泄漏）。
4. **λ 只在 inner-OOF 选**，且 λ=0 必须精确退化为恒等（`forward(..., lam=0)` 返回原值）。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..portability import HAS_TORCH, require

POOLS: tuple[str, ...] = ("mean", "max", "attention")
_FORBIDDEN_KEY_PARTS: tuple[str, ...] = ("logid", "well_id", "wellid", "well_index", "wellname")


def capacity_receipt(d_model: int, hidden: int | None) -> dict[str, Any]:
    """容量收据：`hidden ≤ d_model/8`（下界 4，避免 d_model 很小时无法使用）。"""
    cap = max(int(d_model) // 8, 4)
    h = cap if hidden is None else int(hidden)
    return {"d_model": int(d_model), "hidden": h, "cap": cap, "ok": bool(h <= cap),
            "rule": "井级分支隐藏宽 ≤ 主干 1/8（80 井过拟合风险）"}


def assert_no_well_identity(keys: Sequence[str]) -> list[str]:
    """返回违规键列表（空列表 = 通过）。用户可传入 `model.state_dict().keys()`。"""
    bad = []
    for k in keys:
        low = str(k).lower().replace("-", "_")
        if any(part in low for part in _FORBIDDEN_KEY_PARTS):
            bad.append(str(k))
    return bad


if HAS_TORCH:
    import torch
    import torch.nn as nn

    class WellAttentionHead(nn.Module):
        """`x (B, L, d)` → 井向量 → 逐目标偏置 `Δ (B, 3)`（逐行广播使用）。"""

        def __init__(self, d_model: int, hidden: int | None = None, n_targets: int = 3,
                     dropout: float = 0.2, pool: str = "attention", max_delta: float = 5.0):
            super().__init__()
            if pool not in POOLS:
                raise ValueError(f"pool ∈ {POOLS}，got {pool}")
            rec = capacity_receipt(d_model, hidden)
            if not rec["ok"]:
                raise ValueError(f"井级分支容量超限：hidden={rec['hidden']} > {rec['cap']}"
                                 f"（d_model={d_model}）")
            self.d_model = int(d_model)
            self.hidden = int(rec["hidden"])
            self.pool = pool
            self.max_delta = float(max_delta)
            self.capacity = rec
            if pool == "attention":
                self.attn = nn.Sequential(nn.Linear(self.d_model, self.hidden), nn.Tanh(),
                                          nn.Linear(self.hidden, 1))
            self.mlp = nn.Sequential(nn.Linear(self.d_model, self.hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(self.hidden, int(n_targets)))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=0.01)

        # ------------------------------------------------------------ 池化
        def pool_well(self, x):
            """`x (B,L,d)` → `(pooled (B,d), attn (B,L) 或 None)`。"""
            require("torch")
            if x.dim() != 3:
                raise ValueError(f"x 必须为 (B,L,d)，got {tuple(x.shape)}")
            if self.pool == "mean":
                return x.mean(dim=1), None
            if self.pool == "max":
                return x.max(dim=1).values, None
            logits = self.attn(x).squeeze(-1)                    # (B,L)
            a = torch.softmax(logits, dim=1)
            return torch.einsum("bl,bld->bd", a, x), a

        def forward(self, x, cont=None, lam: float = 0.0):
            """`cont` 为 `(B,3)` 连续头（逐井同值）；`lam=0` 时输出与 `cont` **逐位相同**。"""
            require("torch")
            pooled, a = self.pool_well(x)
            delta = self.max_delta * torch.tanh(self.mlp(pooled))     # 有界，防爆
            out = {"pooled": pooled, "attn": a, "delta": delta,
                   "lambda": float(lam), "capacity": dict(self.capacity)}
            if cont is not None:
                out["cont"] = cont + float(lam) * delta
            return out

        @torch.no_grad()
        def init_from_stats(self, delta_median: Mapping[str, float] | None = None) -> None:
            """把逐目标偏置初值设为训练折中位数（可选；默认零偏置）。"""
            if not delta_median:
                return
            vals = [float(delta_median.get(k, 0.0)) for k in ("POR", "PERM", "SW")]
            self.mlp[-1].bias.copy_(torch.tensor(vals))

        @torch.no_grad()
        def attention_entropy(self, x) -> float | None:
            """attention 池化的归一化熵（1 = 均匀，0 = 只盯一行）。"""
            import math
            _, a = self.pool_well(x)
            if a is None:
                return None
            ent = float(-(a * torch.log(a.clamp_min(1e-12))).sum(1).mean())
            denom = math.log(a.shape[1]) if a.shape[1] > 1 else 1.0
            return ent / denom
