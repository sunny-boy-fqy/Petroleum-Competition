"""WP7：自监督预训练公共件（masked curve modeling）。

只使用输入曲线（可以包含测试井输入，**无标签**）：
  * ``mask_curve_batch``：对 ``(B,L,F)`` 随机掩码整段曲线；
  * ``masked_reconstruction_loss``：只对掩码位置算 SmoothL1；
  * ``span_mask``：连续片段掩码（比逐点更难，迫使模型学地层上下文）。

预训练只作为 E3/E4 的初始化；必须用 paired CI 证明对下游有正增益，否则丢弃。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..portability import HAS_TORCH, require


def span_mask(length: int, rng: np.random.Generator, span: int = 10,
              n_spans: int = 3) -> np.ndarray:
    """在 ``[0,length)`` 上随机选 ``n_spans`` 个连续片段，返回 bool 掩码。"""
    m = np.zeros(int(length), dtype=bool)
    if length <= 0:
        return m
    span = int(np.clip(span, 1, length))
    for _ in range(int(n_spans)):
        s = int(rng.integers(0, max(length - span + 1, 1)))
        m[s:s + span] = True
    return m


def mask_curve_batch(x: Any, missing: Any | None = None,
                     rng: np.random.Generator | None = None,
                     mask_ratio: float = 0.15, span: int = 10,
                     n_spans: int = 3) -> dict[str, Any]:
    """对 ``(B,L,F)`` 输入做随机曲线掩码。

    返回 ``{x_masked, mask, x_original}``；``mask`` 形状同 x（1=被掩码）。
    已缺失的位置不会被重复计算（``missing`` 为 1 的位置保持 0 损失）。
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype="float32")
    if x.ndim != 3:
        raise ValueError(f"x 必须为 (B,L,F)，got {x.shape}")
    b, length, f = x.shape
    mask = np.zeros((b, length, f), dtype=bool)
    for i in range(b):
        for j in range(f):
            if rng.random() < float(mask_ratio):
                mask[i, :, j] = span_mask(length, rng, span=span, n_spans=n_spans)
    x_masked = x.copy()
    x_masked[mask] = 0.0
    if missing is not None:
        miss = np.asarray(missing, dtype=bool)
        mask = mask & (~miss)
    return {"x_masked": x_masked, "mask": mask, "x_original": x}


def masked_reconstruction_loss(pred: Any, target: Any, mask: Any,
                               beta: float = 1.0) -> Any:
    """SmoothL1，仅对 mask=1 位置求均值（torch 门控）。"""
    require("torch")
    import torch
    import torch.nn.functional as F
    p = torch.as_tensor(pred)
    t = torch.as_tensor(target, dtype=p.dtype, device=p.device)
    m = torch.as_tensor(mask, dtype=p.dtype, device=p.device)
    loss = F.smooth_l1_loss(p, t, beta=float(beta), reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def ssl_epoch_loss(model, x_masked: Any, x_original: Any, mask: Any,
                   device: str = "auto") -> float:
    """跑一次 SSL forward/backward 之外的损失计算（调用方负责 backward）。"""
    require("torch")
    from . import loop as L
    dev = L.resolve_device(L.TrainConfig(device=device))
    import torch
    model.train().to(dev)
    xb = torch.from_numpy(np.asarray(x_masked, dtype="float32")).to(dev)
    out = model(xb)
    pred = out["recon"] if isinstance(out, dict) and "recon" in out else out
    return float(masked_reconstruction_loss(pred, x_original, mask).detach().cpu())
