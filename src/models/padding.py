"""NPU/bf16 友好的 1D padding / 上采样助手。

背景：Ascend aclnnReplicationPad1d 在 torch_npu 2.8 上不支持 bfloat16，
直接调用 ``F.pad(..., mode="replicate")`` 会在 NPU 上报
``Tensor self not implemented for DT_BFLOAT16``。本模块用 ``expand + cat``
手工实现 replicate padding，避免该算子；nearest 上采样在 bf16 下先转 fp32
再转回，规避同类算子 dtype 限制。
"""
from __future__ import annotations

from typing import Any

from ..portability import require


def replicate_pad1d(x: Any, pad: int):
    """等价 ``F.pad(x, (pad, pad), mode="replicate")``，但支持 NPU + bf16。

    x 形状 ``(..., L)``；pad 为两侧各补齐长度。pad=0 时原样返回。
    """
    require("torch")
    import torch
    pad = int(pad)
    if pad <= 0:
        return x
    if x.shape[-1] == 0:
        raise ValueError("replicate_pad1d: empty sequence")
    left = x[..., :1].expand(*x.shape[:-1], pad)
    right = x[..., -1:].expand(*x.shape[:-1], pad)
    return torch.cat([left, x, right], dim=-1)


def nearest_upsample1d(x: Any, size: int):
    """``F.interpolate(x, size=size, mode="nearest")`` 的 bf16-safe 版本。"""
    require("torch")
    import torch
    import torch.nn.functional as F
    size = int(size)
    if int(x.shape[-1]) == size:
        return x
    if x.dtype == torch.bfloat16:
        return F.interpolate(x.float(), size=size, mode="nearest").to(dtype=x.dtype)
    return F.interpolate(x, size=size, mode="nearest")
