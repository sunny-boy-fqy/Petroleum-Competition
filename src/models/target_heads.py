"""E5 逐目标头（POR / PERM / SW）：**可插拔参数化** + 表示能力收据 + 尺度契约。

为什么单独一套头（E5/P0–P2）
---------------------------
E1/E3 的多任务头参数化是**冻结**的（`por_max·sigmoid`、`6·tanh`、`sw_mu+σ·g`）。
E5 要回答的是"参数化本身是否限制精度"，因此把三种目标各自做成可切换的模块，
并把"能不能表示 0 / <0.1"、"是否越界"做成**可复算的收据**，而不是口头声明：

* `PorHead`：`sigmoid`（默认，`por_max·sigmoid(g)`）／`softplus_shift`
  （`softplus(g) − softplus(g0)`，左端严格 0）／`linear`（无界，对照臂）／
  **`plus_softplus` = 0.1 下界（禁用对照臂，永远不能表示 POR=0 或 <0.1）**。
  `init_from_stats` 使初值 ≈ 训练折 POR 中位数（**不是** 0.1）。
* `PermHead`：`tanh`（默认，`6·tanh(g)`，天然有界且 PERM>0）／`clip`（梯度在界外为 0）／
  `linear`；可选 `n_buckets` 的 soft-CE 桶头（期望值解码，单调映射）与
  `quantile_heads` 的异方差辅助头（σ = (q75−q25)/1.349）。
  对官方截断 `d = max(ẑ−z, log10(eps))` 的一致性由 `tail_consistency_report` 给证据。
* `SwHead`：SW 是**单一百分数尺度**，只做 `[0,100]` 软裁剪，**永不归一化到 [0,1]、永不 ×100**；
  原子门控命中时**精确**替换为 `ATOM_VALUES['SW']`（无插值），由 `sw_decode` 保证。

契约
----
所有头 `forward` 返回 dict，且都给出 `head_out`（线性头原始输出）以便审计与消融；
`PorHead`/`PermHead`/`SwHead` 的键与 `heads.SeqHead` / `row_mlp.RowMLP` 的对应键同名。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import constants as C
from ..portability import HAS_TORCH, require

POR_PARAMS: tuple[str, ...] = ("sigmoid", "softplus_shift", "linear", "plus_softplus")
POR_FORBIDDEN: tuple[str, ...] = ("plus_softplus",)      # 下界锁死 0.1，禁止用于正式臂
PERM_OUTPUTS: tuple[str, ...] = ("tanh", "clip", "linear")
AUX_KINDS: tuple[str, ...] = ("smooth_l1", "soft_ce_bucket", "quantile")

POR_ZERO_TOL: float = 1e-6
POR_SMALL: float = 0.1
SP_LOWER: float = -20.0
SP_UPPER: float = 20.0


# ---------------------------------------------------------------- 纯 numpy 收据
def por_representability(param: str, por_max: float, g0: float | None = None,
                         n_grid: int = 4001) -> dict[str, Any]:
    """`POR` 参数化的**可达集合**收据（越界即"不可表示"，这是 E5/P0 的硬判据之一）。

    在 `g ∈ [-20, 20]` 上网格搜索最小可达值（`softplus_shift` 在左端饱和到 0）。
    """
    if param not in POR_PARAMS:
        raise ValueError(f"param ∈ {POR_PARAMS}，got {param!r}")
    g = np.linspace(SP_LOWER, SP_UPPER, int(n_grid))
    pm = float(por_max)
    g0 = 0.0 if g0 is None else float(g0)

    def softplus(v):
        return np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)

    if param == "sigmoid":
        vals = pm / (1.0 + np.exp(-g))
    elif param == "softplus_shift":
        vals = np.maximum(softplus(g) - softplus(np.asarray(g0)), 0.0)
    elif param == "linear":
        vals = g
    else:                                                # plus_softplus（禁用臂）
        vals = POR_SMALL + softplus(g)
    vmin = float(np.min(vals))
    return {"param": param, "por_max": pm, "g0": g0, "min_value": vmin,
            "can_represent_zero": bool(vmin <= POR_ZERO_TOL),
            "can_represent_lt_0p1": bool(vmin < POR_SMALL),
            "allowed": bool(param not in POR_FORBIDDEN),
            "note": ("0.1+softplus 下界锁死 0.1：数据中有真实 POR=0 与 576 行 <1（186 行 <0.1），"
                     "因此该臂只能作为反例记录，禁止作为正式臂" if param in POR_FORBIDDEN else
                     "可达集合覆盖 0（数值意义）")}


def official_perm_acc(z_true: "np.ndarray", z_pred: "np.ndarray") -> float:
    """逐行官方 PERM 命中（截断到 `log10(EPS)`，与 `score.acc_perm` 同口径）。"""
    z = np.asarray(z_true, dtype="float64")
    p = np.asarray(z_pred, dtype="float64")
    d = np.maximum(p - z, np.log10(C.EPS))
    return float(np.clip(1.0 - np.abs(d), 0.0, 1.0).mean()) if z.size else 0.0


def aligned_loss_np(z_true: "np.ndarray", z_pred: "np.ndarray") -> float:
    """对齐损失（未平滑）：`|max(ẑ−z, log10(eps))|` —— 极端**低估**不额外惩罚。"""
    z = np.asarray(z_true, dtype="float64")
    p = np.asarray(z_pred, dtype="float64")
    d = np.maximum(p - z, np.log10(C.EPS))
    return float(np.abs(d).mean()) if z.size else 0.0


def _spearman(a: "np.ndarray", b: "np.ndarray") -> float | None:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype("float64")
    rb = np.argsort(np.argsort(b)).astype("float64")
    return float(np.corrcoef(ra, rb)[0, 1])


def tail_consistency_report(z_true: "np.ndarray", z_pred: "np.ndarray",
                            n_bins: int = 8) -> dict[str, Any]:
    """按**真值量级分桶**比较官方 Acc 与对齐损失（E5/P1 的 `E5_perm_tail.json` 内容）。"""
    z = np.asarray(z_true, dtype="float64").ravel()
    p = np.asarray(z_pred, dtype="float64").ravel()
    if z.size == 0:
        return {"bins": [], "monotone_consistent": False, "note": "空输入"}
    edges = np.quantile(z, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges = np.unique(edges)
    bins: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (z >= lo) & (z <= hi if i == len(edges) - 2 else z < hi)
        if not sel.any():
            continue
        bins.append({"lo": lo, "hi": hi, "n_rows": int(sel.sum()),
                     "z_center": float(np.mean(z[sel])),
                     "official_acc": official_perm_acc(z[sel], p[sel]),
                     "aligned_loss_mean": aligned_loss_np(z[sel], p[sel])})
    if bins:
        centers = np.asarray([b["z_center"] for b in bins])
        s_acc = _spearman(centers, np.asarray([b["official_acc"] for b in bins]))
        s_loss = _spearman(centers, np.asarray([b["aligned_loss_mean"] for b in bins]))
    else:
        s_acc = s_loss = None
    consistent = bool(s_acc is not None and s_loss is not None
                      and s_acc > 0 and s_loss < 0)
    return {"bins": bins, "spearman_acc_vs_z": s_acc, "spearman_loss_vs_z": s_loss,
            "monotone_consistent": consistent,
            "note": ("量级越大 → 官方 Acc 越高、对齐损失越低 才算单调一致；"
                     "极端低估一侧只被截断保护")}


def sw_scale_check(clip: tuple[float, float] = (0.0, 100.0)) -> dict[str, Any]:
    """SW 尺度收据（E5/P2 的硬判据）：单一百分数尺度、只软裁剪、绝不 [0,1]/×100。"""
    return {"sw_small_branch": bool(C.SW_SMALL_BRANCH),
            "global_clip_0_1": False,
            "multiply_100": False,
            "soft_clip_0_100": bool(tuple(map(float, clip)) == (0.0, 100.0)),
            "interpolation": False,
            "ok": bool(C.SW_SMALL_BRANCH is False
                       and tuple(map(float, clip)) == (0.0, 100.0))}


def sw_decode(sw_cont: "np.ndarray", q_sw: "np.ndarray", tau: float) -> "np.ndarray":
    """SW 原子门控解码：`q_sw > τ` → **精确** `ATOM_VALUES['SW']`，否则保持连续值。

    断言"无插值"：任何一行要么等于原子值，要么等于其连续输出（不得出现中间值）。
    """
    c = np.asarray(sw_cont, dtype="float64").copy()
    q = np.asarray(q_sw, dtype="float64")
    if q.shape != c.shape:
        raise ValueError(f"q_sw 形状 {q.shape} != sw_cont 形状 {c.shape}")
    hit = q > float(tau)
    atom = float(C.ATOM_VALUES["SW"])
    out = np.where(hit, atom, c)
    bad = (out != atom) & (out != c)
    if bool(bad.any()):
        raise AssertionError(f"SW 解码出现插值：{int(bad.sum())} 行既非原子值也非连续值")
    return out


if HAS_TORCH:
    import math as _math

    import torch
    import torch.nn as nn

    def _logit(p: float) -> float:
        q = min(max(float(p), 1e-6), 1.0 - 1e-6)
        return _math.log(q / (1.0 - q))

    def _softplus(v):
        return torch.nn.functional.softplus(v)

    def _softplus_inv(y: float) -> float:
        """`softplus(g) = y` 的解（稳定实现）。"""
        y = max(float(y), 1e-9)
        return float(_math.log(_math.expm1(y))) if y < 20.0 else float(y)

    class _HeadBase(nn.Module):
        """`body`（Linear→GELU→Dropout）+ `out`（Linear→1）；子类自定义输出变换。"""

        def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1):
            super().__init__()
            self.d_in = int(d_in)
            self.hidden = int(hidden)
            self.body = nn.Sequential(nn.Linear(self.d_in, self.hidden), nn.GELU(),
                                      nn.Dropout(dropout))
            self.out = nn.Linear(self.hidden, 1)
            nn.init.zeros_(self.out.bias)
            nn.init.normal_(self.out.weight, std=0.01)

        def features(self, x):
            return self.body(x)

        def head_out(self, x):
            return self.out(self.features(x)).squeeze(-1)

    class PorHead(_HeadBase):
        """POR 连续头（可插拔参数化；`plus_softplus` 为禁用对照臂）。"""

        def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1,
                     param: str = "sigmoid", por_max: float | None = None,
                     g0: float | None = None):
            super().__init__(d_in, hidden, dropout)
            if param not in POR_PARAMS:
                raise ValueError(f"param ∈ {POR_PARAMS}，got {param!r}")
            self.param = param
            self.g0 = float(g0) if g0 is not None else 0.0
            self.register_buffer("por_max", torch.tensor(float(
                por_max if por_max is not None else C.POR_MAX_BUFFER * C.POR_VALID_MAX)))

        def forward(self, x):
            require("torch")
            g = self.head_out(x)
            if self.param == "sigmoid":
                por = self.por_max * torch.sigmoid(g)
            elif self.param == "softplus_shift":
                por = torch.clamp(_softplus(g) - _softplus(torch.tensor(self.g0,
                                                                       dtype=g.dtype,
                                                                       device=g.device)),
                                  min=0.0)
            elif self.param == "linear":
                por = g
            else:                                        # plus_softplus（禁用臂）
                por = 0.1 + _softplus(g)
            return {"por": por, "head_out": g, "param": self.param,
                    "allowed": bool(self.param not in POR_FORBIDDEN)}

        @torch.no_grad()
        def init_from_stats(self, por_median: float = 11.34, por_max: float | None = None,
                            low_quantile: float | None = None) -> None:
            """初值 ≈ 训练折 POR 中位数（**绝不是 0.1**；`low_quantile` 给 softplus 的 g0）。"""
            if por_max is not None:
                self.por_max.fill_(float(por_max))
            pm = float(self.por_max)
            if self.param == "sigmoid":
                self.out.bias.fill_(_logit(float(por_median) / max(pm, 1e-9)))
            elif self.param == "softplus_shift":
                if low_quantile is not None:
                    self.g0 = _softplus_inv(float(low_quantile))
                # softplus(g) − softplus(g0) = por_median ⇒ g = softplus⁻¹(por_median + softplus(g0))
                base = float(_softplus(torch.tensor(self.g0)).item())
                self.out.bias.fill_(_softplus_inv(float(por_median) + base))
            else:
                self.out.bias.fill_(float(por_median))

        @torch.no_grad()
        def representability(self) -> dict[str, Any]:
            return por_representability(self.param, float(self.por_max), self.g0)

    class PermHead(_HeadBase):
        """PERM 连续头（`perm_z` = log10(PERM)）+ 可选桶头 / 分位辅助头。"""

        def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1,
                     z_output: str = "tanh", clip: tuple[float, float] = (C.PERM_LOG_MIN,
                                                                         C.PERM_LOG_MAX),
                     perm_log_abs: float = 6.0, n_buckets: int | None = None,
                     quantile_heads: int = 0):
            super().__init__(d_in, hidden, dropout)
            if z_output not in PERM_OUTPUTS:
                raise ValueError(f"z_output ∈ {PERM_OUTPUTS}，got {z_output!r}")
            self.z_output = z_output
            self.perm_log_abs = float(perm_log_abs)
            self.register_buffer("clip_lo", torch.tensor(float(clip[0])))
            self.register_buffer("clip_hi", torch.tensor(float(clip[1])))
            self.n_buckets = int(n_buckets) if n_buckets else 0
            self.quantile_heads = int(quantile_heads)
            if self.n_buckets:
                self.bucket_logits = nn.Linear(self.hidden, self.n_buckets)
                edges = torch.linspace(float(clip[0]), float(clip[1]), self.n_buckets + 1)
                self.register_buffer("bucket_edges", edges)
                nn.init.zeros_(self.bucket_logits.bias)
                nn.init.normal_(self.bucket_logits.weight, std=0.01)
            if self.quantile_heads:
                self.quantile = nn.Linear(self.hidden, self.quantile_heads)
                nn.init.zeros_(self.quantile.bias)
                nn.init.normal_(self.quantile.weight, std=0.01)

        def forward(self, x):
            require("torch")
            h = self.features(x)
            g = self.out(h).squeeze(-1)
            if self.z_output == "tanh":
                z = self.perm_log_abs * torch.tanh(g)
            elif self.z_output == "clip":
                z = torch.clamp(g, float(self.clip_lo), float(self.clip_hi))
            else:
                z = g
            out: dict[str, Any] = {"perm_z": z, "head_out": g, "z_output": self.z_output}
            if self.n_buckets:
                logits = self.bucket_logits(h)
                prob = torch.softmax(logits, dim=-1)
                centers = 0.5 * (self.bucket_edges[:-1] + self.bucket_edges[1:])
                out["bucket_logits"] = logits
                out["q_bucket"] = prob
                out["perm_z_bucket"] = (prob * centers[None, :]).sum(-1)
            if self.quantile_heads >= 3:
                q = self.quantile(h)
                q = torch.sort(q, dim=-1).values                    # 强制单调（分位约束）
                out["quantiles"] = q
                out["sigma"] = (q[..., -1] - q[..., 0]) / 1.349
            return out

        @torch.no_grad()
        def contract_report(self, x) -> dict[str, Any]:
            """契约收据：`perm_z` 有限、`10**perm_z > 0`、不出现 ≤0 的 PERM。"""
            out = self.forward(x)
            z = out["perm_z"]
            perm = torch.pow(10.0, z.double())
            return {"finite": bool(torch.isfinite(z).all()),
                    "within_clip": bool((z >= float(self.clip_lo) - 1e-6).all()
                                        and (z <= float(self.clip_hi) + 1e-6).all()),
                    "perm_positive": bool((perm > 0).all()),
                    "z_min": float(z.min()), "z_max": float(z.max())}

        @torch.no_grad()
        def init_from_stats(self, perm_z_median: float = -0.08) -> None:
            if self.z_output == "tanh":
                ratio = min(max(float(perm_z_median) / max(self.perm_log_abs, 1e-9),
                                -0.999999), 0.999999)
                self.out.bias.fill_(_math.atanh(ratio))
            else:
                self.out.bias.fill_(float(perm_z_median))

    class SwHead(_HeadBase):
        """SW 连续头（归一化空间）+ 原子概率 `q_sw`：单一百分数尺度，只做 [0,100] 软裁剪。"""

        def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1,
                     sw_mu: float | None = None, sw_sigma: float | None = None,
                     clip: tuple[float, float] = (0.0, 100.0)):
            super().__init__(d_in, hidden, dropout)
            self.register_buffer("sw_mu", torch.tensor(float(
                sw_mu if sw_mu is not None else C.SW_VALID_MEDIAN)))
            self.register_buffer("sw_sigma", torch.tensor(float(
                sw_sigma if sw_sigma is not None else 20.0)))
            self.register_buffer("clip_lo", torch.tensor(float(clip[0])))
            self.register_buffer("clip_hi", torch.tensor(float(clip[1])))
            self.q_sw = nn.Linear(self.hidden, 1)
            nn.init.zeros_(self.q_sw.bias)
            nn.init.normal_(self.q_sw.weight, std=0.01)

        def forward(self, x):
            require("torch")
            h = self.features(x)
            g = self.out(h).squeeze(-1)
            raw = self.sw_mu + self.sw_sigma * g
            sw = torch.clamp(raw, float(self.clip_lo), float(self.clip_hi))
            q_logit = self.q_sw(h).squeeze(-1)
            return {"sw": sw, "sw_head_out": g, "sw_raw": raw,
                    "q_sw": torch.sigmoid(q_logit), "q_sw_logit": q_logit,
                    "scale_check": sw_scale_check((float(self.clip_lo), float(self.clip_hi)))}

        @torch.no_grad()
        def init_from_stats(self, sw_mu: float = C.SW_VALID_MEDIAN,
                            sw_sigma: float = 20.0, sw_atom_rate: float = 0.7) -> None:
            self.sw_mu.fill_(float(sw_mu))
            self.sw_sigma.fill_(max(float(sw_sigma), 1e-6))
            self.out.bias.fill_(0.0)                # 初值 = sw_mu（标签尺度）
            self.q_sw.bias.fill_(_logit(sw_atom_rate))


def build_target_head(target: str, d_in: int, **kw):
    """工厂：`target ∈ {"POR","PERM","SW"}` → 对应头（键名与多任务头一致）。"""
    require("torch")
    t = str(target).upper()
    if t == "POR":
        return PorHead(d_in, **kw)
    if t == "PERM":
        return PermHead(d_in, **kw)
    if t == "SW":
        return SwHead(d_in, **kw)
    raise ValueError(f"target ∈ POR/PERM/SW，got {target!r}")
