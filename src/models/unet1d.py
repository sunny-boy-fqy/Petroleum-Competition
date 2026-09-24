"""1D U-Net 序列主干（E3/P1 §5 步 1）：**全段 seq2seq**、逐行同长输出。

结构
----
    x (B, L, F)
      └─ 输入投影 Conv1d(F→base_ch)
      └─ depth 级下采样（stride 2）：每级 2×[depthwise-separable Conv1d(k=5) → BN → GELU]
      └─ bottleneck：插入**空洞卷积**（dilation 递增）扩大有效感受野
         （Ascend 反传 pad 约束下，实际 dilation 由 `ascend_safe_dilation` 自动截断）
      └─ depth 级上采样（nearest 到 skip 长度）→ 拼接 skip → 同样的 separable 块
      └─ SeqHead：(B,L,base_ch) → 与 RowMLP 同键的多任务输出

为什么要 depthwise-separable：`资料库/08` §4 建议用它换感受野（参数量 ~1/k 倍）；
为什么 padding 用 `replicate`：zero-padding 会在井首/井尾制造"人为真空"，E3/P2 要求做
边界伪影检查 —— 用 replicate 降低（而不是掩盖）该效应，检查结果仍照实上报。

长度严格保持：每次上采样都用 `F.interpolate(size=skip_len, mode="nearest")` 对齐到 skip
的**实际**长度，因此奇数长度（如 257）也不会错位或丢点。
"""
from __future__ import annotations

from typing import Any

from ..portability import HAS_TORCH, require
from .heads import SeqHead, count_parameters, init_head_from_stats
from .padding import ascend_safe_dilation, nearest_upsample1d, replicate_pad1d

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


if HAS_TORCH:

    class SeparableBlock(nn.Module):
        """depthwise (k=5, groups=C) + pointwise + BN + GELU，可选空洞。"""

        # 不能只把 dilation 卡到 255：反传输入卷积的 pad 随 (k-1)*dilation 增长，
        # U-Net k=5、dilation=255 时 pad 量级 1020，会报 Conv2DBackpropInput
        # "backprop pad value invalid"。这里按 kernel 反推安全上限。
        def __init__(self, ch: int, k: int = 5, dilation: int = 1, dropout: float = 0.0):
            super().__init__()
            self.dilation = ascend_safe_dilation(k, dilation)
            pad = (k - 1) * self.dilation // 2            # 居中（非因果）
            self.pad = pad
            self.dw = nn.Conv1d(ch, ch, k, padding=0, dilation=self.dilation, groups=ch)
            self.pw = nn.Conv1d(ch, ch, 1)
            self.bn = nn.BatchNorm1d(ch)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):                              # (B,C,L)
            x = replicate_pad1d(x, self.pad)
            return self.drop(self.bn(self.pw(F.gelu(self.dw(x)))))


    class Down(nn.Module):
        def __init__(self, ch_in: int, ch_out: int, depth_idx: int, k: int = 5,
                     dropout: float = 0.0, dilation_base: int = 2):
            super().__init__()
            self.b1 = SeparableBlock(ch_in, k, dilation=1, dropout=dropout)
            # 下采样块里插空洞卷积（E3/P1 §5 步 1）
            self.b2 = SeparableBlock(ch_in, k, dilation=dilation_base ** depth_idx,
                                     dropout=dropout)
            self.proj = nn.Conv1d(ch_in, ch_out, 1)
            self.down = nn.AvgPool1d(2, ceil_mode=True)

        def forward(self, x):
            x = self.b2(self.b1(x))
            # skip 与下采样张量**同宽**（ch_out）：解码器按 `chans[min(i+1, d-1)]` 拼接，
            # 二者必须一致，否则 separable 卷积的 groups 会对不上（曾在此处出错）。
            skip = self.proj(x)
            return self.down(skip), skip


    class Up(nn.Module):
        def __init__(self, ch_in: int, ch_skip: int, ch_out: int, k: int = 5,
                     dropout: float = 0.0):
            super().__init__()
            self.b1 = SeparableBlock(ch_in + ch_skip, k, dropout=dropout)
            self.b2 = SeparableBlock(ch_in + ch_skip, k, dropout=dropout)
            self.proj = nn.Conv1d(ch_in + ch_skip, ch_out, 1)

        def forward(self, x, skip):
            x = nearest_upsample1d(x, skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            return self.proj(self.b2(self.b1(x)))


    class UNet1D(nn.Module):
        """`depth` 级 U-Net；`base_ch` 决定宽度，`dilation_max` 影响 bottleneck 感受野。"""

        def __init__(self, n_features: int, base_ch: int = 64, depth: int = 5,
                     dropout: float = 0.1, k: int = 5, dilation_max: int = 512,
                     head_hidden: int = 128, head_dropout: float | None = None,
                     init_stats: dict | None = None, **head_kwargs) -> None:
            super().__init__()
            self.n_features = int(n_features)
            self.depth = int(depth)
            chans = [min(base_ch * (2 ** i), base_ch * 4) for i in range(self.depth)]
            self.stem = nn.Conv1d(self.n_features, chans[0], 1)
            self.downs = nn.ModuleList()
            for i in range(self.depth):
                ch_in = chans[i]
                ch_out = chans[min(i + 1, self.depth - 1)]
                self.downs.append(Down(ch_in, ch_out, i, k=k, dropout=dropout))
            # bottleneck：空洞卷积链，dilation 递增到 dilation_max 附近（上限由 depth 决定）
            dil = 1
            while dil * 2 <= int(dilation_max):
                dil *= 2
            self.bottleneck = nn.Sequential(
                SeparableBlock(chans[-1], k, dilation=1, dropout=dropout),
                SeparableBlock(chans[-1], k, dilation=max(dil // 4, 1), dropout=dropout),
                SeparableBlock(chans[-1], k, dilation=max(dil // 2, 1), dropout=dropout),
                SeparableBlock(chans[-1], k, dilation=max(dil, 1), dropout=dropout),
            )
            self.ups = nn.ModuleList()
            for i in range(self.depth - 1, -1, -1):
                # 解码器第 i 级的输入与 skip **同宽**（= `chans[min(i+1, depth-1)]`），
                # 输出回到 `chans[i]`，于是下一级（i-1）的拼接宽度又是 2·chans[i]。
                ch = chans[min(i + 1, self.depth - 1)]
                self.ups.append(Up(ch, ch, chans[i], k=k, dropout=dropout))
            self.head = init_head_from_stats(
                SeqHead(chans[0], hidden=head_hidden,
                        dropout=dropout if head_dropout is None else head_dropout,
                        **head_kwargs), init_stats)

        def forward(self, x):                              # (B,L,F)
            require("torch")
            h = self.stem(x.transpose(1, 2))                # (B,C,L)
            skips = []
            for d in self.downs:
                h, s = d(h)
                skips.append(s)
            h = self.bottleneck(h)
            for up, s in zip(self.ups, reversed(skips)):
                h = up(h, s)
            return self.head(h.transpose(1, 2))             # (B,L,C) -> 同键输出

        def receptive_field(self) -> int:
            """**理论**感受野（点）：每级下采样 ×2 且块内 k=5 / 空洞递增。"""
            rf = 1
            for i in range(self.depth):
                rf += (5 - 1) * (2 ** i) + (5 - 1) * (2 ** i) * (2 ** i)
            rf += (5 - 1) * (1 + 2 + 4 + 8)
            return int(rf)


def build_unet(n_features: int, init_stats: dict | None = None, **kw) -> Any:
    require("torch")
    return UNet1D(n_features, init_stats=init_stats, **kw)


def model_summary(model) -> dict[str, Any]:
    return {"arch": type(model).__name__, "n_params": count_parameters(model),
            "depth": getattr(model, "depth", None),
            "receptive_field_points": (model.receptive_field()
                                       if hasattr(model, "receptive_field") else None)}
