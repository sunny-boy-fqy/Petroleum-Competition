"""行级多任务 MLP（E1 基线，**无任何序列上下文**）+ 原子门多头（E6-R3）。

结构（`资料库/08` §1.4 的硬参数共享 + 多头；R3 新增原子头）：

    x (B, F)
      └─ MLP trunk: Linear(F→h) → [BN → GELU → Dropout] × L → (B, h)
           ├─ q_joint  : Linear(h→1) logit   # 联合占位（辅助项 + 可选高置信守卫）
           ├─ q_por    : Linear(h→1) logit   # P(POR  == 0.1)   原子保护（主）
           ├─ q_perm   : Linear(h→1) logit   # P(PERM == 0.01)  原子保护（主）
           ├─ q_sw     : Linear(h→1) logit   # P(SW   == 99.9)  原子保护（主）
           ├─ cont_por : Linear(h→1)         # POR，标签尺度
           ├─ cont_perm: Linear(h→1)         # log10(PERM)
           └─ cont_sw  : Linear(h→1)         # SW，**归一化尺度**

连续头参数化（R3 冻结，`constants.py` 为唯一事实源）
------------------------------------------------------
- POR : `por = por_max · sigmoid(g)`，`por_max` 是 **buffer（不可学习）**，
        默认 `1.2 × 33.177 ≈ 39.8`。输出天然非负、可精确趋近 0（数据中存在真实 0.0，
        以及 576 行 < 1 的有效 POR）。**禁止 `0.1 + softplus(g)`**（会把下界锁死在 0.1）。
- PERM: `perm_z = 6 · tanh(g)` = log10(PERM)，夹到 [-6, 6]，保证 PERM > 0。
- SW  : `sw = sw_mu + sw_sigma · g`，`sw_mu`/`sw_sigma` 是 **buffer**，
        由训练折有效 SW 统计写入。SW 头 bias 初始化为 **0**，使初始输出 ≈ `sw_mu`（≈82.8），
        而**不是** 0；SW 是单一标签尺度（百分数），绝不做 [0,1] 归一化输出。

`forward` 返回 dict：
    `por`(B,) `perm_z`(B,) `sw`(B,) `q_atom`(B,3，列序 POR/PERM/SW) `q_joint`(B,)
    `ph_logit`(B,) —— = `q_joint` 的别名，保留以兼容旧调用点。

设计取舍
--------
- **BN 而非 LN 作默认**：BN 的批统计跨井起正则作用；E3 会做 BN/LN/GN 消融。
- **不把连续头初始化到占位常量**：占位行由原子头/联合头负责命中，连续头保持"有效值先验"。
- **参数量小**（默认 2×256 ≈ 8 万参数）：E1 的目的是给出"纯净的行级上限"。
- 解码逻辑在 `features/basic.py::decode_predictions` 与 `inference/atomic_gate.py`。
"""
from __future__ import annotations

from .. import constants as C
from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn as nn


if HAS_TORCH:

    class RowMLP(nn.Module):
        def __init__(self, n_features: int, hidden: int = 256, layers: int = 2,
                     dropout: float = 0.1, perm_log_abs: float = 6.0,
                     por_max: float | None = None, sw_mu: float | None = None,
                     sw_sigma: float | None = None):
            super().__init__()
            self.perm_log_abs = float(perm_log_abs)
            # R3 连续头参数化的"标尺"全部是 buffer（不可学习、随 .to(device) 迁移）。
            self.register_buffer(
                "por_max",
                torch.tensor(float(por_max if por_max is not None
                                   else C.POR_MAX_BUFFER * C.POR_VALID_MAX)),
            )
            self.register_buffer(
                "sw_mu",
                torch.tensor(float(sw_mu if sw_mu is not None else C.SW_VALID_MEDIAN)),
            )
            self.register_buffer(
                "sw_sigma",
                torch.tensor(float(sw_sigma if sw_sigma is not None else 20.0)),
            )
            blocks: list[nn.Module] = []
            d = n_features
            for _ in range(layers):
                blocks += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                           nn.GELU(), nn.Dropout(dropout)]
                d = hidden
            self.trunk = nn.Sequential(*blocks)

            # 原子/联合头（logit 输出）
            self.q_joint = nn.Linear(d, 1)
            self.q_por = nn.Linear(d, 1)
            self.q_perm = nn.Linear(d, 1)
            self.q_sw = nn.Linear(d, 1)
            # 连续头
            self.cont_por = nn.Linear(d, 1)
            self.cont_perm = nn.Linear(d, 1)
            self.cont_sw = nn.Linear(d, 1)

            # 所有头小权重初始化（std 0.01）+ bias 0；
            # bias 的"先验"由 init_from_stats 写入（SW 连续头 bias 保持 0 -> 输出 = sw_mu）。
            for h in (self.q_joint, self.q_por, self.q_perm, self.q_sw,
                      self.cont_por, self.cont_perm, self.cont_sw):
                nn.init.zeros_(h.bias)
                nn.init.normal_(h.weight, std=0.01)

        def forward(self, x):
            require("torch")
            h = self.trunk(x)
            # POR：下界 0、上界 por_max，可表示真实 0.0（禁止 0.1+softplus）
            por = self.por_max * torch.sigmoid(self.cont_por(h).squeeze(-1))
            perm_z = self.perm_log_abs * torch.tanh(self.cont_perm(h).squeeze(-1))
            # SW：仿射反归一化到标签尺度；bias=0 时初值即 sw_mu
            sw = self.sw_mu + self.sw_sigma * self.cont_sw(h).squeeze(-1)
            q_joint = self.q_joint(h).squeeze(-1)
            q_atom = torch.stack(
                [self.q_por(h).squeeze(-1),
                 self.q_perm(h).squeeze(-1),
                 self.q_sw(h).squeeze(-1)],
                dim=1,
            )
            return {
                "por": por,
                "perm_z": perm_z,
                "sw": sw,
                "q_atom": q_atom,
                "q_joint": q_joint,
                "ph_logit": q_joint,   # 旧键别名（兼容既有调用点）
            }

        @torch.no_grad()
        def init_from_stats(self, por_median: float = 11.34, por_max: float = 39.8,
                            sw_mu: float = 82.805, sw_sigma: float = 20.0,
                            perm_z_median: float = -0.08, joint_atom_rate: float = 0.667,
                            atom_rates: tuple = (0.6674, 0.6773, 0.7097)) -> None:
            """用**训练折**统计初始化输出头标尺与先验（R3）。

            - POR : `sigmoid(b) = por_median / por_max` → `b = logit(p)`，
                    使初始连续输出 ≈ `por_median`（有效 POR 中位数，而非 0.1）。
            - PERM: `tanh(b) = perm_z_median / 6` → `b = atanh(...)`，初始 `perm_z` ≈ 中位数。
            - SW  : bias 保持 0 → 初始输出 = `sw_mu`（≈82.8，标签尺度）。
            - 原子/联合头：bias = logit(先验发生率)（`atom_rates` 列序 POR/PERM/SW）。
            """
            import math as _m

            self.por_max.fill_(float(por_max))
            self.sw_mu.fill_(float(sw_mu))
            self.sw_sigma.fill_(max(float(sw_sigma), 1e-6))

            def _logit(p: float) -> float:
                q = min(max(float(p), 1e-6), 1.0 - 1e-6)
                return _m.log(q / (1.0 - q))

            self.cont_por.bias.fill_(_logit(float(por_median) / max(float(por_max), 1e-9)))

            z_ratio = float(perm_z_median) / max(self.perm_log_abs, 1e-9)
            z_ratio = min(max(z_ratio, -0.999999), 0.999999)
            self.cont_perm.bias.fill_(_m.atanh(z_ratio))

            self.cont_sw.bias.fill_(0.0)

            self.q_joint.bias.fill_(_logit(joint_atom_rate))
            for head, rate in zip((self.q_por, self.q_perm, self.q_sw), tuple(atom_rates)):
                head.bias.fill_(_logit(rate))


def build_model(n_features: int, hidden: int = 256, layers: int = 2,
                dropout: float = 0.1, seed: int | None = None,
                init_stats: dict | None = None):
    """工厂函数（也供 predict.py / manifest 引用）。

    `init_stats`：可选 dict，非空时转发给 `RowMLP.init_from_stats(**init_stats)`
    （例如 `{"por_max": sc["por_max"], "sw_mu": sc["sw_mu"], "sw_sigma": sc["sw_sigma"]}`）。
    """
    require("torch")
    if seed is not None:
        torch.manual_seed(seed)
    model = RowMLP(n_features, hidden=hidden, layers=layers, dropout=dropout)
    if init_stats:
        model.init_from_stats(**init_stats)
    return model


def count_parameters(model) -> int:
    require("torch")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
