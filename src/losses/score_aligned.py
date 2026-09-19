"""评分对齐损失（与 `rules.md` §7.3 同构，可微）。

依据 `资料库/12` §2.2–2.4：
  1. Charbonnier 平滑绝对值：`|x| ≈ sqrt(x²+α²) − α`（消除 |·| 在 0 处的尖峰，梯度有界于 [-1,1]）；
  2. softplus 平滑截断：`min(x,1) ≈ x − softplus(x−1; β)`；
  于是 `max(0, 1−ℓ) ≈ 1 − ℓ + softplus(ℓ−1)`，β→∞ 时逐点等于官方得分，且处处可微；
  3. 三段式：`L = L_align + λ1·L_aux + λ2·L_ph`（`资料库/12` §2.3），
     λ1 从 1.0 退火到 0.1，λ2 固定 0.2。

**重要纪律**（`资料库/12` §2.4 提示、§2.3 末）：
  - 损失只用于反向传播；**模型选择/早停一律用真实 `score.py` 分数**；
  - PERM 一律在 log10 空间，网络输出 z，`PERM = 10^z`；用 tanh 夹到 [-6,6]，**不用 ReLU**；
  - `eps` 取 1e-3（不是 1e-6），显著稳定梯度。

掩码语义：`mask=0` 的行（缺测）不参与任何损失；**占位行 mask=1 且照常参与回归监督**
（占位常量本身就是要预测的正确目标）。
"""
from __future__ import annotations

from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


def smooth_abs(x, alpha: float = 1e-3):
    """Charbonnier 平滑绝对值，α 控制平滑半径（对标准化量纲取 1e-3~1e-2）。"""
    require("torch")
    return torch.sqrt(x * x + alpha * alpha) - alpha


def soft_min1(x, beta: float = 20.0):
    """平滑 min(x, 1)：x − softplus(x−1; β)。β→+∞ 时等于 min(x,1)。"""
    require("torch")
    return x - F.softplus(x - 1.0, beta=beta)


def align_score_relative(y, yhat, delta: float, eps: float = 1e-3,
                         alpha: float = 1e-3, beta: float = 20.0):
    """POR / SW 的对齐**得分**（越大越好）：s ≈ max(0, 1 − |ŷ−y|/(δ(|y|+ε)))。"""
    require("torch")
    denom = delta * (y.abs() + eps)
    ell = smooth_abs(yhat - y, alpha) / denom
    return 1.0 - ell + F.softplus(ell - 1.0, beta=beta)


def align_score_log(z, zhat, alpha: float = 1e-3, beta: float = 20.0,
                    eps: float = 1e-3):
    """PERM 的对齐**得分**（log10 空间，与官方严格同构）。

    官方：`s = max(0, 1 − |log10(max(ŷ/y, ε))|)`
      - 当 `ŷ/y ≥ ε` 时，`log10(max(ŷ/y,ε)) = ẑ − z`；
      - 当 `ŷ/y < ε`（严重低估）时，官方把比值**截断在 ε**，
        误差恒为 `log10(1/ε)`（ε=1e-3 → 3.0），不再随低估程度增长。
    **R2-H3 修复**：此前直接用 `|ẑ − z|`，在 `ŷ/y < ε` 区域比官方惩罚更重
    （例如 `ẑ−z=−5` 时官方误差 3.0、旧实现 5.0），梯度方向与官方评分不一致。
    这里对 **log 空间的差值**做同样的下截断：`d = max(ẑ − z, log10(ε))`。
    """
    require("torch")
    import math as _math
    d = zhat - z
    d = torch.maximum(d, torch.full_like(d, _math.log10(eps)))
    ell = smooth_abs(d, alpha)
    return 1.0 - ell + F.softplus(ell - 1.0, beta=beta)


def masked_mean(x, mask=None):
    """掩码均值。**R2-H2 修复**：先把被屏蔽位置的 NaN/Inf 清零再乘掩码，
    否则 `NaN * 0 = NaN` 会让整个 batch 的 loss 变成 NaN。"""
    require("torch")
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return (x_safe * m).sum() / m.sum().clamp_min(1.0)


def aligned_loss(y_por, p_por, z_perm, zhat_perm, y_sw, p_sw, mask=None,
                 w_por: float = 0.30, w_perm: float = 0.35, w_sw: float = 0.35,
                 eps: float = 1e-3, alpha: float = 1e-3, beta: float = 20.0):
    """三目标加权对齐损失（返回标量，越小越好）。

    mask : (B, 3) float，1=该目标参与监督（缺测为 0）
    """
    require("torch")
    m_por = None if mask is None else mask[:, 0]
    m_perm = None if mask is None else mask[:, 1]
    m_sw = None if mask is None else mask[:, 2]

    s_por = align_score_relative(y_por, p_por, 0.08, eps, alpha, beta)
    s_perm = align_score_log(z_perm, zhat_perm, alpha, beta)
    s_sw = align_score_relative(y_sw, p_sw, 0.05, eps, alpha, beta)

    return -(
        w_por * masked_mean(s_por, m_por)
        + w_perm * masked_mean(s_perm, m_perm)
        + w_sw * masked_mean(s_sw, m_sw)
    )


def aux_loss(y_por, p_por, z_perm, zhat_perm, y_sw, p_sw, mask=None,
             huber_beta: float = 1.0, eps: float = 1e-3):
    """变换空间稠密损失（早期梯度来源）。

    - PERM : log10 空间 Smooth L1（与官方"对数空间评分"同构）
    - POR/SW: 直接 Smooth L1（相对误差形式在占位 99.9 上会把权重压得过小，
      因此这里用绝对形式，相对性由 align 项负责）
    """
    require("torch")

    def sl1(a, b, m):
        l = F.smooth_l1_loss(a, b, beta=huber_beta, reduction="none")
        return masked_mean(l, m)

    m_por = None if mask is None else mask[:, 0]
    m_perm = None if mask is None else mask[:, 1]
    m_sw = None if mask is None else mask[:, 2]
    return (
        0.30 * sl1(p_por, y_por, m_por)
        + 0.35 * sl1(zhat_perm, z_perm, m_perm)
        + 0.35 * sl1(p_sw, y_sw, m_sw)
    )


def placeholder_bce(y_ph, q_logit, mask=None):
    """联合常量占位状态的 BCE（mask 取"任一行非缺测"即可）。"""
    require("torch")
    l = F.binary_cross_entropy_with_logits(q_logit, y_ph, reduction="none")
    return masked_mean(l, mask)


def total_loss(out: dict, batch: dict, lam1: float = 1.0, lam2: float = 0.2,
               use_align: bool = True, use_aux: bool = True, use_ph: bool = True,
               **kw) -> tuple:
    """组合损失。

    out  : 模型输出 {'por','perm_z','sw','ph_logit'}
    batch: {'por','perm_z','sw','mask','y_ph'}
    返回 (total, parts_dict)
    """
    require("torch")
    mask = batch["mask"]
    parts: dict[str, "torch.Tensor"] = {}
    total = None

    if use_align:
        parts["align"] = aligned_loss(
            batch["por"], out["por"], batch["perm_z"], out["perm_z"],
            batch["sw"], out["sw"], mask, **kw)
        total = parts["align"] if total is None else total + parts["align"]
    if use_aux:
        parts["aux"] = aux_loss(
            batch["por"], out["por"], batch["perm_z"], out["perm_z"],
            batch["sw"], out["sw"], mask)
        total = lam1 * parts["aux"] if total is None else total + lam1 * parts["aux"]
    if use_ph:
        # 占位行必然非缺测；用"任一目标非缺测"作为 BCE 掩码
        ph_mask = (mask.sum(dim=1) > 0).to(mask.dtype)
        parts["ph"] = placeholder_bce(batch["y_ph"], out["ph_logit"], ph_mask)
        total = lam2 * parts["ph"] if total is None else total + lam2 * parts["ph"]

    if total is None:
        raise ValueError("at least one loss term must be enabled")
    return total, {k: float(v.detach()) for k, v in parts.items()}
