"""E4/P0 PatchTF：按深度维切 patch 的 Transformer 主干（**与 `SeqHead` 同键**的逐行输出）。

设计要点
--------
1. **patch 化 + 重叠拼接**：`patchify` 沿深度维取 patch，`unpatchify` 用归一化权重
   overlap-add 回逐行；权重归一化保证 **任意长度（含奇数）都精确还原**，
   因此"输出长度 == 输入长度"是硬契约（E4/P0 §7 第 1 条）。
2. **channel-independent（CI）**：`channel_independent=True` 时 patch 嵌入对每个输入通道
   **共享同一个 `Linear(P→d)`**，再按 patch 做一次通道混合投影（因子化嵌入）；
   `False` 时用 `Linear(P·C→d)` 联合嵌入。两者的对照就是 E4/P0 要求的 CI 消融。
3. **位置信息**：`rel_pos=True` 用可学习**相对**位置偏置（加性 attention bias）；
   `False` 则退化为可学习绝对位置嵌入——两者都有位置信息，消融的是"相对 vs 绝对"。
4. **只用 torch 2.4 的 API**：`F.scaled_dot_product_attention`，不依赖编译扩展。
5. **同键**：输出 `por/perm_z/sw/q_atom/q_joint/*_logit/ph_logit`，损失/解码/τ 一行不改；
   `forward_states()` 额外给出逐行主干隐状态（E4/P1 融合与 E5 冻结骨干都用它）。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..portability import HAS_TORCH, require

WEIGHT_KINDS: tuple[str, ...] = ("triangular", "hann", "equal")


# ------------------------------------------------------------------ patch 几何（numpy）
def patch_spans(length: int, patch_len: int, stride: int) -> list[tuple[int, int]]:
    """覆盖 `[0, length)` 的 patch 区间（最后一个 patch **右对齐**，保证尾部被覆盖）。"""
    if patch_len <= 0 or stride <= 0:
        raise ValueError(f"patch_len/stride 必须 >0，got {patch_len}/{stride}")
    if patch_len > length:
        return [(0, length)]
    spans = [(s, s + patch_len) for s in range(0, length - patch_len + 1, stride)]
    if spans[-1][1] < length:
        spans.append((length - patch_len, length))
    return spans


def overlap_weights(n_tokens: int, patch_len: int, kind: str = "triangular") -> np.ndarray:
    """每个 patch 内部的权重窗口 `(n_tokens, patch_len)`（**未**归一化）。"""
    if kind not in WEIGHT_KINDS:
        raise ValueError(f"weight kind ∈ {WEIGHT_KINDS}，got {kind}")
    if kind == "equal":
        return np.ones((n_tokens, patch_len), dtype="float64")
    t = np.linspace(0.0, 1.0, patch_len, dtype="float64")
    if kind == "hann":
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)
    else:                                    # triangular（中心最高）
        w = 1.0 - np.abs(2.0 * t - 1.0)
    w = np.maximum(w, 1e-3)                  # 端点不为 0：否则首/尾行无覆盖
    return np.repeat(w[None, :], n_tokens, axis=0)


def coverage_report(length: int, patch_len: int, stride: int, kind: str = "triangular") -> dict:
    """覆盖收据：每行权重和 > 0（否则拼接会留下未定义行）。"""
    spans = patch_spans(length, patch_len, stride)
    w = overlap_weights(len(spans), spans[0][1] - spans[0][0], kind)
    acc = np.zeros(length, dtype="float64")
    for n, (s, e) in enumerate(spans):
        acc[s:e] += w[n]
    return {"length": int(length), "n_tokens": len(spans), "patch_len": int(patch_len),
            "stride": int(stride), "kind": kind, "min_weight": float(acc.min()),
            "uncovered": int((acc <= 0).sum()), "ok": bool((acc > 0).all()),
            "spans": [[int(s), int(e)] for s, e in spans]}


def reconstruction_check(length: int, patch_len: int, stride: int,
                         kind: str = "triangular", seed: int = 0) -> dict:
    """`patchify → unpatchify` 必须**逐点精确**还原（重叠权重归一化后，几何硬契约）。

    这里折叠/展开的是**逐行**张量（`(B,N,P)`），是 `patchify` 的真逆；
    `PatchTF.forward` 折叠的是逐 token 特征（同一套归一化权重，逐 token 广播到其 P 行）。
    """
    if not HAS_TORCH:
        raise RuntimeError("reconstruction_check requires torch")
    import torch

    cov = coverage_report(length, patch_len, stride, kind)
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(2, length, 3, generator=g, dtype=torch.float64)
    patches, spans = _patchify_torch(x, patch_len, stride)          # (B,N,P,3)
    w = torch.as_tensor(overlap_weights(len(spans), patch_len, kind), dtype=torch.float64)
    idx = torch.cat([torch.arange(s, e) for s, e in spans]).long()
    wflat = w.reshape(-1)
    B, N, P = patches.shape[0], patches.shape[1], patches.shape[2]
    flat = patches.reshape(B, N * P, 3) * wflat.view(1, N * P, 1)
    out = torch.zeros(B, length, 3, dtype=torch.float64)
    out.index_add_(1, idx, flat)
    wsum = torch.zeros(length, dtype=torch.float64)
    wsum.index_add_(0, idx, wflat)
    rec = out / wsum.clamp_min(1e-12).view(1, length, 1)
    diff = float((rec - x).abs().max())
    return {**cov, "max_abs_diff": diff, "ok": bool(cov["ok"] and diff < 1e-9)}


if HAS_TORCH:
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _patchify_torch(x, patch_len: int, stride: int):
        """`x (B,L,C)` → `patches (B,N,P,C)` + spans。"""
        if x.dim() != 3:
            raise ValueError(f"x 必须为 (B,L,C)，got {tuple(x.shape)}")
        length = int(x.shape[1])
        spans = patch_spans(length, patch_len, stride)
        out = torch.stack([x[:, s:e, :] for s, e in spans], dim=1)
        return out, spans

    def _unpatchify_torch(pred, w, length: int, spans) -> Any:
        """逐 **token** 向量 → 逐行向量：`(B,N,·)` + 权重 `(N,P)` → `(B,L,·)`。

        每个 token 的向量按权重广播到它覆盖的 P 行，再做**归一化** overlap-add；
        归一化保证任意长度（含奇数、尾 patch 右对齐）都精确覆盖，无未定义行。
        """
        B, N = pred.shape[0], pred.shape[1]
        rest = pred.shape[2:]
        if N != w.shape[0]:
            raise ValueError(f"token 数 {N} 与权重行数 {w.shape[0]} 不一致")
        P = w.shape[1]
        idx = torch.cat([torch.arange(s, e, device=pred.device) for s, e in spans]).long()
        wflat = w.reshape(-1).to(pred.dtype)
        flat = (pred[:, :, None] * w.view((1, N, P) + (1,) * len(rest))).reshape(
            B, N * P, *rest)
        out = pred.new_zeros((B, length, *rest))
        out.index_add_(1, idx, flat)
        wsum = pred.new_zeros((length,))
        wsum.index_add_(0, idx, wflat)
        return out / wsum.clamp_min(1e-12).view((1, length) + (1,) * len(rest))

    class RelativePositionBias(nn.Module):
        """可学习相对位置偏置（加性 attention bias，供 `scaled_dot_product_attention` 使用）。"""

        def __init__(self, n_heads: int, max_tokens: int = 512):
            super().__init__()
            self.max_tokens = int(max_tokens)
            self.bias = nn.Parameter(torch.zeros(n_heads, 2 * self.max_tokens - 1))
            nn.init.normal_(self.bias, std=0.02)

        def forward(self, n_tokens: int):
            idx = torch.arange(n_tokens, device=self.bias.device)
            rel = (idx[None, :] - idx[:, None]).clamp(-(self.max_tokens - 1),
                                                       self.max_tokens - 1)
            return self.bias[:, rel + self.max_tokens - 1]      # (H,N,N)

    class PreLNBlock(nn.Module):
        """Pre-LN Transformer block（自研 MHA 以支持加性位置偏置；用 SDPA，无编译扩展）。"""

        attn_api = "sdpa"

        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                     ffn_mult: int = 4):
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError(f"d_model({d_model}) 必须被 n_heads({n_heads}) 整除")
            self.n_heads = int(n_heads)
            self.d_head = int(d_model) // int(n_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model))
            self.drop = nn.Dropout(dropout)
            nn.init.zeros_(self.proj.bias)
            nn.init.zeros_(self.ffn[-1].bias)

        def _attn(self, h, bias=None):
            B, N, _ = h.shape
            qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            attn_mask = None if bias is None else bias[None, ...].to(q.dtype)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                               dropout_p=0.0, is_causal=False)
            return o.transpose(1, 2).reshape(B, N, self.n_heads * self.d_head)

        def forward(self, x, bias=None):
            h = self.norm1(x)
            x = x + self.drop(self.proj(self._attn(h, bias)))
            return x + self.drop(self.ffn(self.norm2(x)))

    class PatchTF(nn.Module):
        """逐行输出键与 `SeqHead` 一致的 Patch Transformer（深度维 patch，CI 可选）。"""

        def __init__(self, d_in: int, patch_len: int = 32, stride: int = 16,
                     d_model: int = 128, n_layers: int = 4, n_heads: int = 8,
                     dropout: float = 0.1, channel_independent: bool = True,
                     rel_pos: bool = True, weight_kind: str = "triangular",
                     head_hidden: int = 128, max_tokens: int = 512,
                     ffn_mult: int = 4, init_stats: dict | None = None):
            super().__init__()
            from .heads import SeqHead, init_head_from_stats

            if weight_kind not in WEIGHT_KINDS:
                raise ValueError(f"weight_kind ∈ {WEIGHT_KINDS}，got {weight_kind}")
            if d_model % n_heads != 0:
                raise ValueError(f"d_model({d_model}) 必须被 n_heads({n_heads}) 整除")
            self.d_in = int(d_in)
            self.patch_len = int(patch_len)
            self.stride = int(stride)
            self.d_model = int(d_model)
            self.channel_independent = bool(channel_independent)
            self.rel_pos = bool(rel_pos)
            self.weight_kind = weight_kind
            self.max_tokens = int(max_tokens)

            # 因子化（CI）或联合（非 CI）patch 嵌入
            self.patch_embed_ci = nn.Linear(self.patch_len, d_model)
            self.patch_mix = nn.Linear(self.d_in * d_model, d_model)
            self.patch_embed_joint = nn.Linear(self.patch_len * self.d_in, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, d_model))
            nn.init.normal_(self.pos_embed, std=0.02)
            self.rel_bias = RelativePositionBias(n_heads, self.max_tokens)
            self.blocks = nn.ModuleList([PreLNBlock(d_model, n_heads, dropout, ffn_mult)
                                         for _ in range(int(n_layers))])
            self.norm = nn.LayerNorm(d_model)
            self.head = SeqHead(d_model, hidden=head_hidden, dropout=dropout)
            init_head_from_stats(self.head, init_stats)

        # -------------------------------------------------------- patch 嵌入
        def _embed(self, patches):
            B, N, P, C = patches.shape
            if self.channel_independent:
                # 共享 Linear(P→d) 逐通道作用（因子化），再按 patch 混合通道
                h = self.patch_embed_ci(patches.permute(0, 1, 3, 2))     # (B,N,C,d)
                h = self.patch_mix(h.reshape(B, N, C * self.d_model))
            else:
                h = self.patch_embed_joint(patches.reshape(B, N, P * C))
            return h

        def tokens(self, x):
            """`x (·,L,C)` → patch 隐状态 `h (B,N,d_model)` + spans（分块推理用）。"""
            require("torch")
            if x.dim() == 2:
                x = x[None, ...]
            if int(x.shape[1]) < self.patch_len:
                raise ValueError(f"序列长度 {int(x.shape[1])} < patch_len "
                                 f"{self.patch_len}：请增大 chunk 或减小 patch_len")
            patches, spans = _patchify_torch(x, self.patch_len, self.stride)
            B, N = patches.shape[0], patches.shape[1]
            if N > self.max_tokens:
                raise ValueError(f"token 数 {N} 超过 max_tokens={self.max_tokens}；"
                                 f"请增大 --chunk 或 max_tokens")
            h = self._embed(patches)
            h = h + self.pos_embed[:, :N, :]
            bias = self.rel_bias(N) if self.rel_pos else None
            for blk in self.blocks:
                h = blk(h, bias)
            return self.norm(h), spans

        def forward(self, x, return_states: bool = False):
            """`x (B,L,C)` → 同键逐行输出 +（可选）逐行隐状态。

            逐行分辨率：token 隐状态先按归一化重叠权重拼回 `(B,L,d_model)`（= 逐行特征），
            再由 `SeqHead` 逐行输出——因此**每行都有独立预测**（分辨率不受 stride 限制），
            且长度严格等于输入长度。
            """
            require("torch")
            squeeze = x.dim() == 2
            if squeeze:
                x = x[None, ...]
            h, spans = self.tokens(x)
            length = int(x.shape[1])
            w = torch.as_tensor(overlap_weights(h.shape[1], self.patch_len, self.weight_kind),
                                dtype=h.dtype, device=h.device)
            states = _unpatchify_torch(h, w, length, spans)
            out = self.head(states)
            if return_states:
                out = {**out, "states": states}
            if squeeze:
                out = {k: v[0] for k, v in out.items()}
            return out

        @torch.no_grad()
        def forward_states(self, x):
            """只取逐行隐状态 `(B,L,d_model)`（E4/P1 融合 / E5 冻结骨干用）。"""
            return self.forward(x, return_states=True)["states"]

        def summary(self, length: int = 1024) -> dict[str, Any]:
            n_tok = len(patch_spans(length, self.patch_len, self.stride))
            return {"d_in": self.d_in, "patch_len": self.patch_len, "stride": self.stride,
                    "d_model": self.d_model, "n_layers": len(self.blocks),
                    "n_heads": self.blocks[0].n_heads,
                    "channel_independent": self.channel_independent,
                    "rel_pos": self.rel_pos, "attn_api": PreLNBlock.attn_api,
                    "tokens_at_length": n_tok,
                    "receptive_field_rows": self.patch_len + (n_tok - 1) * self.stride,
                    "n_params": int(sum(p.numel() for p in self.parameters()
                                        if p.requires_grad))}


def build_patchtf(n_features: int, init_stats: dict | None = None, **kwargs):
    """按 E4/P0 的参数表构造 PatchTF（`d_model % n_heads == 0` 由构造器断言）。"""
    require("torch")
    return PatchTF(int(n_features), init_stats=init_stats, **kwargs)


def model_summary(model, length: int = 1024) -> dict:
    return model.summary(length=length)
