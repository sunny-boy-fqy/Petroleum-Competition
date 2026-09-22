"""可选物理软约束 ``L_phys``（E7/P0 exp7）。

本模块只实现**可解释、可在现有数据上计算**的一类软约束：

* 用训练输入中的 AC / DEN / CNL 曲线，按 Wyllie 声波孔隙度、密度孔隙度、中子孔隙度
  计算一个“测井孔隙度先验” ``phi_log``；
* 对**非缺测、非原子占位**的 POR 行，惩罚连续头预测 ``por`` 与该先验的 Huber 偏差。

严格说，这只是一个简化的岩石物理先验，不是完整 Archie / Kozeny–Carman 反演：
- 常数取 `features.physics` 的中国陆相砂岩常用近似值；
- 不引入 Rw、m、n、比表面等无法从当前数据可靠标定的参数；
- 默认 ``lam_phys=0``，只有 E7 exp7 显式打开时才进入总损失；
- 必须通过消融验证后才能进入正式臂。

输入约定
--------
`batch["x"]` 是**标准化后**的特征矩阵（row: (B,F) 或 seq: (B,L,F)）；
`row_scaler` 提供 mean/std 与 feature_names。物理模块只使用 F1 的原始曲线列
（F2 的前 32 列是 F1，因此同样适用）。
"""
from __future__ import annotations

from typing import Any, Sequence

from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

# 与 features.physics 保持同源近似常数
ACMA_US_M = 55.5 * 3.28084
ACF_US_M = 189.0 * 3.28084
RHO_MA = 2.65
RHO_F = 1.0


def _find(names: Sequence[str], want: str) -> int | None:
    w = str(want).strip().lower()
    for i, n in enumerate(names):
        if str(n).strip().lower() == w:
            return i
    return None


def _inverse_scaled(x, row_scaler):
    """把标准化特征还原到训练时的近似原始尺度。"""
    mean = torch.as_tensor(row_scaler.mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(row_scaler.std, dtype=x.dtype, device=x.device)
    return x * std + mean


def physics_porosity_loss(pred_por, batch: dict, row_scaler: Any = None,
                          feature_names: Sequence[str] | None = None,
                          huber_beta: float = 1.0,
                          por_scale: float = 100.0):
    """返回标量 ``L_phys``；没有可用物理列时返回 0。"""
    require("torch")
    # 硬门（审查 H2/M4）：开启 L_phys 时缺关键上下文必须显式报错。
    # 旧实现静默返回 0，导致 E7 exp7 的 lam_phys 消融看似跑了、实际完全没生效。
    missing: list[str] = []
    if row_scaler is None:
        missing.append("row_scaler")
    if not feature_names:
        missing.append("feature_names")
    x = batch.get("x")
    if x is None:
        missing.append("batch['x']")
    if missing:
        raise ValueError(
            "physics_porosity_loss: 已启用物理损失，但缺少 " + ", ".join(missing)
            + "。请由训练循环把标准化特征写入 batch['x']，并传入训练折的 "
              "RowScaler/feature_names（禁止静默退化为 0）。")
    names = list(feature_names)
    ac_i, den_i, cnl_i = _find(names, "AC"), _find(names, "DEN"), _find(names, "CNL")
    if ac_i is None or den_i is None or cnl_i is None:
        raise ValueError(
            "physics_porosity_loss: feature_names 缺少 AC/DEN/CNL 中的至少一列，"
            f"无法计算测井孔隙度先验；got AC={ac_i}, DEN={den_i}, CNL={cnl_i}")

    x_raw = _inverse_scaled(x, row_scaler)
    ac = x_raw[..., ac_i]
    den = x_raw[..., den_i]
    cnl = x_raw[..., cnl_i]

    # 缺失指示位：F1 中列名为 `<CURVE>_miss`；标准化后反变换到 0/1。
    miss = {}
    for curve, idx in (("AC", ac_i), ("DEN", den_i), ("CNL", cnl_i)):
        mi = _find(names, f"{curve}_miss")
        if mi is not None:
            m = _inverse_scaled(x, row_scaler)[..., mi]
            miss[curve] = (m > 0.5)
        else:
            miss[curve] = torch.zeros_like(ac, dtype=torch.bool)

    finite = torch.isfinite(ac) & torch.isfinite(den) & torch.isfinite(cnl)
    valid_ac = finite & (~miss["AC"])
    valid_den = finite & (~miss["DEN"])
    valid_cnl = finite & (~miss["CNL"])

    phi_s = (ac - ACMA_US_M) / (ACF_US_M - ACMA_US_M)
    phi_d = (RHO_MA - den) / (RHO_MA - RHO_F)
    phi_n = cnl / 100.0

    cnt = (valid_ac.to(phi_s.dtype) + valid_den.to(phi_s.dtype)
           + valid_cnl.to(phi_s.dtype))
    num = (phi_s * valid_ac.to(phi_s.dtype)
           + phi_d * valid_den.to(phi_d.dtype)
           + phi_n * valid_cnl.to(phi_n.dtype))
    phi = torch.where(cnt > 0, num / cnt.clamp_min(1.0), torch.zeros_like(num))
    phys_valid = cnt > 0

    mask = batch.get("mask")
    if mask is None:
        obs = torch.ones_like(pred_por, dtype=torch.bool)
    else:
        obs = mask[..., 0] > 0.5
    y_atom = batch.get("y_atom")
    if y_atom is not None:
        obs = obs & (y_atom[..., 0] <= 0.5)
    use = obs & phys_valid & torch.isfinite(pred_por)
    if not bool(use.any()):
        return torch.zeros((), dtype=pred_por.dtype, device=pred_por.device)

    target = phi * float(por_scale)
    err = F.smooth_l1_loss(pred_por / float(por_scale), target / float(por_scale),
                           beta=float(huber_beta), reduction="none")
    return (err * use.to(err.dtype)).sum() / use.to(err.dtype).sum().clamp_min(1.0)
