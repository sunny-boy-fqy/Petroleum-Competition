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


# Ascend aclnn Conv2DBackpropInput（Conv1d 在 NPU 上通常落到 Conv2D 反传）对
# 反传输入卷积的 pad 有上限。对 kernel k、dilation d、forward padding=0 的卷积，
# 其反传输入卷积的有效 pad 量级为 (k-1)*d；实测 d=255/k=5 时该值 1020，
# 会触发 `backprop pad value invalid[conv2d_backprop_input.cc]`。
# 因此不能只把 dilation clamp 到 255，还必须保证 (k-1)*dilation <= 255。
ASCEND_BACKPROP_PAD_MAX = 255


def ascend_safe_dilation(kernel_size: int, dilation: int) -> int:
    """返回 Ascend Conv2DBackpropInput 反传 pad 约束下的安全 dilation。

    约束：``(kernel_size - 1) * dilation <= ASCEND_BACKPROP_PAD_MAX``。
    例：k=5 -> 最大 63；k=3 -> 最大 127。k=1 时不受该约束。
    """
    span = int(kernel_size) - 1
    if span <= 0:
        return max(int(dilation), 1)
    cap = max(int(ASCEND_BACKPROP_PAD_MAX) // span, 1)
    return min(max(int(dilation), 1), cap)
