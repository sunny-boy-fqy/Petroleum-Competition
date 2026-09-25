"""非因果 TCN 序列主干（E3/P1 §5 步 2）：**全段 seq2seq**、逐行同长输出。

关键取舍
--------
- **非因果（centered）**：本任务是**离线整井推理**，没有实时性约束；因果化会白丢未来
  上下文（E3 §8 明令禁止）。实现方式：卷积前两侧各 pad `(k-1)·d/2`，输出居中。
- `dilation = 2^i`（i=0..n_blocks-1，上限 `dilation_max`，默认 512 ≈ 50 m 感受野）；
  块结构 = `x + dropout(conv2(gelu(conv1(pad(x)))))` + 残差（`资料库/08` §6）。
- `weight_norm` 给卷积核加参数化（TCN 惯例），`chomp` 不需要（居中 pad 已保长）。

⚠️ 计划口径更正（必须记录）
--------------------------
E3/P1 §7 与 E3/P2 §5 写的是"确认非因果（把输入尾部置零不影响头部输出）"——
**这句话描述的其实是因果性的判据**：因果模型里 `out[t]` 只依赖 `x[≤t]`，所以尾部置零
天然不影响头部。对**非因果**模型恰恰相反：尾部（未来）在感受野内时，尾部置零**必须**
改变头部输出。E3 §8 又明确要求"TCN 做成因果卷积"是禁止项。

因此本模块按**真正的非因果判据**实现并测试（`tests/test_models_seq.py`）：
  1. 尾部置零 **改变**头部输出（证明用到了未来上下文）；
  2. 头部置零 **改变**尾部输出（证明用到了过去上下文）。
两条同时成立才是"居中/双向"卷积。若把这条照抄成"尾部置零不影响头部"，
验收反而会强迫实现退化成因果卷积 —— 与 E3 §8 直接冲突。
"""
from __future__ import annotations

from typing import Any

from ..portability import HAS_TORCH, require
from .heads import SeqHead, count_parameters, init_head_from_stats
from .padding import ascend_safe_dilation, replicate_pad1d

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # PyTorch ≥2.1 的新式参数化 API（`torch.nn.utils.weight_norm` 已弃用并会告警）
    from torch.nn.utils.parametrizations import weight_norm


if HAS_TORCH:

    class TCNBlock(nn.Module):
        """残差块：`x + dropout(conv2(gelu(conv1(centered_pad(x)))))`。"""

        # Ascend 反传输入卷积的 pad 随 (k-1)*dilation 增长；k=3、dilation=255
        # 时 pad 量级 510，仍会在 Conv2DBackpropInput 报 "backprop pad value invalid"。
        # 因此按 kernel 反推安全 dilation。
        def __init__(self, ch: int, k: int = 3, dilation: int = 1, dropout: float = 0.1,
                     use_weight_norm: bool = True):
            super().__init__()
            self.requested_dilation = int(dilation)
            self.dilation = ascend_safe_dilation(k, dilation)
            self.pad = (k - 1) * self.dilation // 2   # 居中 -> 非因果
            c1 = nn.Conv1d(ch, ch, k, dilation=self.dilation)
            c2 = nn.Conv1d(ch, ch, k, dilation=self.dilation)
            self.conv1 = weight_norm(c1) if use_weight_norm else c1
            self.conv2 = weight_norm(c2) if use_weight_norm else c2
            self.drop = nn.Dropout(dropout)

        def forward(self, x):                          # (B,C,L)
            h = replicate_pad1d(x, self.pad)
            h = F.gelu(self.conv1(h))
            h = replicate_pad1d(h, self.pad)
            h = self.drop(self.conv2(h))
            return x + h                               # 残差（保长）


    class TCN(nn.Module):
        def __init__(self, n_features: int, channels: int = 128, n_blocks: int = 9,
                     k: int = 3, dilation_base: int = 2, dilation_max: int = 512,
                     dropout: float = 0.1, head_hidden: int = 128,
                     init_stats: dict | None = None, use_weight_norm: bool = True,
                     **head_kwargs) -> None:
            super().__init__()
            self.n_features = int(n_features)
            self.n_blocks = int(n_blocks)
            self.k = int(k)
            self.dilation_base = int(dilation_base)
            self.dilation_max = int(dilation_max)
            self.stem = nn.Conv1d(self.n_features, channels, 1)
            blocks = []
            for i in range(self.n_blocks):
                d = min(int(dilation_base) ** i, int(dilation_max))
                blocks.append(TCNBlock(channels, k=k, dilation=d, dropout=dropout,
                                       use_weight_norm=use_weight_norm))
            self.blocks = nn.ModuleList(blocks)
            self.head = init_head_from_stats(
                SeqHead(channels, hidden=head_hidden, dropout=dropout, **head_kwargs),
                init_stats)

        def dilations(self) -> list[int]:
            """实际生效的逐块 dilation（Ascend 反传约束可能已截断）。"""
            return [int(b.dilation) for b in self.blocks]

        def requested_dilations(self) -> list[int]:
            """构造时按 `dilation_max` 请求的逐块 dilation（未截断）。"""
            return [min(int(self.dilation_base) ** i, int(self.dilation_max))
                    for i in range(self.n_blocks)]

        def receptive_field(self) -> int:
            """理论感受野（点）：1 + Σ 2·pad（居中卷积两侧各 pad）。"""
            return int(1 + sum(2 * b.pad for b in self.blocks))

        def forward(self, x):                          # (B,L,F)
            require("torch")
            h = self.stem(x.transpose(1, 2))           # (B,C,L)
            for b in self.blocks:
                h = b(h)
            return self.head(h.transpose(1, 2))        # (B,L,·) -> 同键输出


def build_tcn(n_features: int, init_stats: dict | None = None, **kw) -> Any:
    require("torch")
    return TCN(n_features, init_stats=init_stats, **kw)


def model_summary(model) -> dict[str, Any]:
    actual = model.dilations() if hasattr(model, "dilations") else None
    requested = (model.requested_dilations()
                 if hasattr(model, "requested_dilations") else None)
    return {"arch": type(model).__name__, "n_params": count_parameters(model),
            "n_blocks": getattr(model, "n_blocks", None),
            "dilation_max": getattr(model, "dilation_max", None),
            "dilations": actual, "requested_dilations": requested,
            "dilation_truncated": bool(actual is not None and requested is not None
                                       and actual != requested),
            "receptive_field_points": (model.receptive_field()
                                       if hasattr(model, "receptive_field") else None),
            "non_causal": True}
