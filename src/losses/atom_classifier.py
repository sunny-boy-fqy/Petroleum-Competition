"""WP2：序列原子分类的专用损失（纯 torch 门控）。

与 ``score_aligned.atom_bce`` 的区别：
  * Focal BCE（γ 可调）+ 逐目标正类权重；
  * **非联合原子行**额外加权（q_joint 漏掉的那 31k SW 原子行）；
  * **边界难负例**加权（真实连续值贴近原子值的行）。
默认参数保持向后兼容：``gamma=0`` 时退化为普通 BCE。
"""
from __future__ import annotations

from typing import Any, Sequence

from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


def _as_target(t, dtype):
    return torch.as_tensor(t, dtype=dtype)


def focal_bce_with_logits(logits: Any, targets: Any, gamma: float = 0.0,
                          pos_weight: Any | None = None, mask: Any | None = None,
                          row_weight: Any | None = None) -> Any:
    """逐元素 Focal BCE（返回 (...,3) 的未归一化损失）。"""
    require("torch")
    z = torch.as_tensor(logits)
    y = _as_target(targets, z.dtype)
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=z.dtype, device=z.device).reshape(-1)
        if pw.numel() == 1:
            pw = pw.expand(3)
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=pw)
    if gamma and float(gamma) > 0:
        p = torch.sigmoid(z)
        pt = torch.where(y > 0.5, p, 1.0 - p)
        bce = bce * torch.pow(torch.clamp(1.0 - pt, min=1e-6), float(gamma))
    if mask is not None:
        bce = bce * torch.as_tensor(mask, dtype=z.dtype, device=z.device)
    if row_weight is not None:
        bce = bce * torch.as_tensor(row_weight, dtype=z.dtype, device=z.device)
    return bce


def atom_classifier_loss(logits: Any, y_atom: Any, mask: Any,
                         y_joint: Any | None = None, gamma: float = 1.5,
                         pos_weight: Any | None = None,
                         alpha_nonjoint: float = 1.0,
                         boundary_weight: Any | None = None,
                         return_parts: bool = False) -> Any:
    """逐目标 focal BCE 的标量损失（对 3 个目标取平均）。

    ``boundary_weight`` 可传 (...,3) 的逐元素额外权重（边界难负例）。
    """
    require("torch")
    z = torch.as_tensor(logits)
    y = torch.as_tensor(y_atom, dtype=z.dtype, device=z.device)
    m = torch.as_tensor(mask, dtype=z.dtype, device=z.device)
    if y_joint is not None:
        yj = torch.as_tensor(y_joint, dtype=z.dtype, device=z.device).unsqueeze(-1)
        nonjoint = (1.0 - yj).expand_as(y)
    else:
        nonjoint = torch.ones_like(y)
    w = 1.0 + float(alpha_nonjoint) * y * nonjoint
    if boundary_weight is not None:
        w = w * torch.as_tensor(boundary_weight, dtype=z.dtype, device=z.device)
    elem = focal_bce_with_logits(z, y, gamma=gamma, pos_weight=pos_weight,
                                 mask=m, row_weight=w)
    parts = {}
    for j, name in enumerate(("POR", "PERM", "SW")):
        denom = (m[..., j] * w[..., j]).sum().clamp_min(1.0)
        parts[name] = (elem[..., j]).sum() / denom
    mean = (parts["POR"] + parts["PERM"] + parts["SW"]) / 3.0
    if return_parts:
        parts["atom"] = mean
        return parts
    return mean


def boundary_hard_negative_weight(cont: Any, y: Any, atom_values: Sequence[float],
                                  delta: Sequence[float], max_weight: float = 3.0
                                  ) -> Any:
    """边界难负例权重（WP2）。

    对**非原子行**加权，权重由两部分组成：

    * ``r_lab``：真实标签 y 到原子值的相对距离（越近越难，例如 SW 99.0 vs 99.9）；
    * ``r_pred``：连续预测 cont 到原子值的相对距离（假原子风险，例如 POR cont≈0.1 但 y 非原子）。

    ``w = 1 + (max_weight-1)·[exp(-r_lab²/2) + 0.5·exp(-r_pred²/2)]``，并夹到
    ``[1, max_weight]``；原子行权重恒为 1。PERM 用 log10 空间，容差固定为 1 个数量级。
    """
    require("torch")
    c = torch.as_tensor(cont, dtype=torch.float32)
    t = torch.as_tensor(y, dtype=torch.float32)
    av = torch.as_tensor(atom_values, dtype=torch.float32, device=t.device)[None, :]
    is_atom = (torch.abs(t - av) <= 1e-9).to(t.dtype)

    def _r(v):
        rs = []
        for j, d in enumerate(delta):
            if d is None:
                vj = torch.clamp(v[..., j], min=1e-12)
                aj = torch.clamp(av[..., j], min=1e-12)
                rs.append(torch.abs(torch.log10(vj) - torch.log10(aj)))
            else:
                denom = (float(d) * (torch.abs(t[..., j]) + 1e-3)).clamp_min(1e-9)
                rs.append(torch.abs(v[..., j] - av[..., j]) / denom)
        return torch.stack(rs, dim=-1)

    r_lab = _r(t)
    r_pred = _r(c)
    bump = (torch.exp(-0.5 * r_lab * r_lab)
            + 0.5 * torch.exp(-0.5 * r_pred * r_pred))
    w = 1.0 + (float(max_weight) - 1.0) * bump
    w = torch.clamp(w, 1.0, float(max_weight))
    return 1.0 + (w - 1.0) * (1.0 - is_atom)
