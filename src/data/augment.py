"""数据增强（E2/P2）：**只扰动特征，绝不移动标签**。

四类增强（E2/P2 §5）
--------------------
1. `channel_outage`：随机把某条**曲线通道**在连续一段深度上置缺（模拟仪器失效）——
   在**曲线层**实现（`augment_curves`），因此物理/窗口特征会随之变化，语义正确；
2. `depth_jitter`：深度轴 ±k 点抖动 —— 实现为"**特征**按 ±k 行错位取值，标签留在原行"
   （`X_aug[i] = X[i+δ_i]`），这正是 E2/P2 §5 步 2 要求的"对同一行做特征扰动"；
3. `segment_resample`：井内随机段做"隔点抽取 + 线性回插"（模拟不同采样率）；
4. `gauss_noise`：按**训练折**逐列稳健尺度缩放的高斯噪声（σ = frac × 该列尺度）。

两条路径与取舍
--------------
- `augment_curves(inputs, missing, ...)`：**理论上正确**（特征由扰动后的曲线重新派生，
  窗口/物理特征自洽）；代价是每个 epoch 要重算特征（E2/P2 用吞吐报告标定这笔开销）。
- `augment_matrix(X, ...)`：训练循环里的**便宜默认**（直接作用在已缓存的 F2 矩阵上）。
  代价：通道置缺退化为"把对应的 F1/phys/win/well 列置 NaN"，窗口统计不会重算 —— 因此
  `augment_matrix` 只做 `depth_jitter` / `gauss_noise` / **列级** dropout，不在文档里
  冒充"仪器失效"。两者都以"标签逐行不变"为前提。

纪律：增强**只在训练折**打开（`split == "train"`），验证/测试折永远不增强；
噪声尺度只能来自训练折（`column_scales`）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .. import constants as C

_CURVE_INDEX = {name: i for i, name in enumerate(C.INPUT_COLUMNS)}


@dataclass
class AugmentConfig:
    """默认值来自 E2/P2 §6（概率与幅度都是**折内选择**的起点，不是结论）。"""
    enabled: bool = False
    channel_mask_p: float = 0.05          # 每条曲线被置缺的概率
    channel_outage_len: int = 51          # 置缺段长度（点，≈5.1 m）
    depth_jitter: int = 2                 # ±k 点（0 = 关闭）
    gauss_sigma_frac: float = 0.01        # σ = frac × 该列训练折稳健尺度（0 = 关闭）
    segment_resample_p: float = 0.10      # 井内触发"隔点抽取+回插"的概率
    segment_len: int = 101
    max_shift_frac: float = 0.0           # 预留：全井对齐偏移（默认关闭）

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AugmentConfig":
        known = {f for f in AugmentConfig().as_dict()}
        return AugmentConfig(**{k: v for k, v in (d or {}).items() if k in known})


def column_scales(X: "np.ndarray", mask: "np.ndarray | None" = None) -> "np.ndarray":
    """逐列稳健尺度（IQR/1.349，退化用 std）——**只能由训练折调用**。"""
    x = np.asarray(X, dtype="float64")
    out = np.ones(x.shape[1], dtype="float64")
    m = None if mask is None else np.asarray(mask, dtype=bool)
    for j in range(x.shape[1]):
        col = x[:, j]
        if m is not None:
            col = np.where(m, col, np.nan)
        v = col[np.isfinite(col)]
        if v.size < 2:
            out[j] = 0.0
            continue
        q1, q3 = np.percentile(v, [25.0, 75.0])
        iqr = float(q3 - q1)
        out[j] = (iqr / 1.349) if iqr > 0 else float(v.std())
    return out


def _roll_rows(X: "np.ndarray", shift: int) -> "np.ndarray":
    """按行整体错位 `shift`（edge 保持，不引入跨井数据）。"""
    if shift == 0:
        return X
    out = np.empty_like(X)
    n = X.shape[0]
    if shift > 0:
        out[:shift] = X[0]
        out[shift:] = X[:-shift]
    else:
        k = -shift
        out[-k:] = X[-1]
        out[:-k] = X[k:]
    return out


def augment_curves(inputs: "np.ndarray", missing: "np.ndarray | None",
                   cfg: AugmentConfig, rng: "np.random.Generator"
                   ) -> tuple["np.ndarray", "np.ndarray"]:
    """曲线层增强：返回 `(inputs_aug, missing_aug)`；**标签不受影响**。

    - 通道置缺：在随机起点把某条曲线连续 `channel_outage_len` 点置 NaN；
    - 深度抖动：把整条曲线按 ±`depth_jitter` 行错位（标签仍在原行）；
    - 段重采样：随机段"隔点抽取 + 线性回插"；
    - 高斯噪声：按列稳健尺度加噪。
    """
    if not cfg.enabled:
        m = (~np.isfinite(inputs)).astype("int8") if missing is None \
            else np.asarray(missing, dtype="int8").copy()
        return np.asarray(inputs, dtype="float32").copy(), m
    x = np.asarray(inputs, dtype="float64").copy()
    n, k = x.shape
    if missing is None:
        miss = (~np.isfinite(x)).astype("int8")
    else:
        miss = np.asarray(missing, dtype="int8").copy()

    if cfg.depth_jitter:
        for j in range(k):
            d = int(rng.integers(-cfg.depth_jitter, cfg.depth_jitter + 1))
            if d:
                x[:, j] = _roll_rows(x[:, j][:, None], d)[:, 0]

    if cfg.segment_resample_p > 0 and n > cfg.segment_len + 2:
        for j in range(k):
            if rng.random() >= cfg.segment_resample_p:
                continue
            start = int(rng.integers(0, n - cfg.segment_len))
            end = start + cfg.segment_len
            seg = x[start:end, j]
            idx = np.arange(seg.size)
            keep = idx[::2]
            if keep.size < 2:
                continue
            x[start:end, j] = np.interp(idx, keep, seg[keep],
                                        left=np.nan, right=np.nan)

    if cfg.channel_mask_p > 0:
        for j in range(k):
            if rng.random() < cfg.channel_mask_p:
                L = min(int(cfg.channel_outage_len), n)
                s = int(rng.integers(0, max(n - L, 1)))
                x[s:s + L, j] = np.nan
                miss[s:s + L, j] = 1

    if cfg.gauss_sigma_frac > 0:
        sc = column_scales(x)
        x = x + rng.normal(0.0, 1.0, size=x.shape) * (cfg.gauss_sigma_frac * sc)[None, :]

    miss = np.where(np.isfinite(x), miss, 1).astype("int8")
    return x.astype("float32"), miss


def augment_matrix(X: "np.ndarray", cfg: AugmentConfig, rng: "np.random.Generator",
                   scales: "np.ndarray | None" = None,
                   column_dropout_p: float | None = None) -> "np.ndarray":
    """特征矩阵层增强（训练循环的便宜路径，**不重算窗口/物理特征**）。

    只做：行错位（depth jitter）、按列稳健尺度加噪、可选**整列** dropout。
    通道置缺的语义正确版本请用 `augment_curves`（见模块 docstring）。
    """
    x = np.asarray(X, dtype="float32")
    if not cfg.enabled:
        return x.copy()
    out = x.astype("float64", copy=True)
    n, k = out.shape
    if cfg.depth_jitter:
        for j in range(k):
            d = int(rng.integers(-cfg.depth_jitter, cfg.depth_jitter + 1))
            if d:
                out[:, j] = _roll_rows(out[:, j][:, None], d)[:, 0]
    if cfg.gauss_sigma_frac > 0:
        sc = column_scales(out) if scales is None else np.asarray(scales, dtype="float64")
        out = out + rng.normal(0.0, 1.0, size=out.shape) * (cfg.gauss_sigma_frac * sc)[None, :]
    p = cfg.channel_mask_p if column_dropout_p is None else float(column_dropout_p)
    if p > 0:
        drop = rng.random(k) < p
        if drop.any():
            out[:, drop] = np.nan
    return out.astype("float32")


def label_invariance(X_before: "np.ndarray", X_after: "np.ndarray",
                     y: "np.ndarray | None", mask: "np.ndarray | None") -> dict[str, Any]:
    """增强的**契约检查**（E2/P2 §7）：行数不变、特征确实变了、标签逐行相等。

    注意本函数只做断言用的取证；调用方负责持有"增强前的标签"（增强本身从不改标签）。
    """
    a = np.asarray(X_before)
    b = np.asarray(X_after)
    if a.shape != b.shape:
        raise AssertionError(f"增强改变了形状：{a.shape} -> {b.shape}")
    changed = bool(np.any((a != b) | (~np.isfinite(a) & np.isfinite(b))
                          | (np.isfinite(a) & ~np.isfinite(b))))
    return {"rows_preserved": True, "shape": list(a.shape), "features_changed": changed,
            "y_rows": None if y is None else int(np.asarray(y).shape[0]),
            "mask_rows": None if mask is None else int(np.asarray(mask).shape[0]),
            "note": "增强只作用于特征侧；标签与 mask 逐行不变（E2/P2 §7）"}


def augmentation_off(cfg: AugmentConfig) -> AugmentConfig:
    """消融用：返回关闭增强的副本。"""
    d = cfg.as_dict()
    d["enabled"] = False
    return AugmentConfig.from_dict(d)
