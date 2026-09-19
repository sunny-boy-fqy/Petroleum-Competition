"""F1 行级特征：`F_raw(14) + F_miss(14+1) + F_depth(3)` = 32 维。

设计依据
--------
- `资料库/07` §5：标准化参数**只在训练折 fit**（本模块只负责"产出原始+缺失指示"，
  标准化由 `RowScaler` 在折内完成）；
- `资料库/08` §0.1-4：工程曲线（CAL/DEVI/AZIM/BIT/CASE）在单井内近常数，
  井级信息留到 E2 的 `F_well`；E1 只做行级，保持"纯行级基线"的可解释性；
- `资料库/12` §3.4：占位状态需要显式监督 → 每行同时输出 `is_placeholder` 标签（在标签侧，不在特征里）。

特征列定义（顺序固定，写入 `FEATURE_NAMES`）
------------------------------------------------
0-13   : 14 条原始曲线（保留 NaN 语义）
14-27  : 14 条逐曲线缺失指示（0/1）
28     : 该行输入缺失比例（0..1）
29     : 井内相对深度 (depth - depth_min) / (depth_max - depth_min)，缺省 0
30     : 相邻采样间隔（m，0.1 量级），首行用中位间隔
31     : 井内累计深度序号归一化 index/(n-1)

**必须用中心窗口/井级统计时属于 E2**；E1 只允许逐行可计算的量，
其中 29–31 是"行内可知"的深度编码（不跨行泄漏，因为只用该井自身的深度序列，
而测试井的深度序列本来就是可观测输入）。
"""
from __future__ import annotations

import numpy as np

from .. import constants as C
from ..portability import HAS_NUMPY

N_RAW_CURVES = len(C.INPUT_COLUMNS)      # 13（GR..CASE，E0-R2 修正：DEPTH 不计入曲线）
N_RAW = N_RAW_CURVES + 1                 # 14 = 13 条曲线 + DEPTH 原始值
N_MISS = N_RAW_CURVES + 1                # 14 = 逐曲线缺失位 + DEPTH 缺失位
N_EXTRA = 4                              # missing_ratio, rel_depth, depth_step, depth_index_norm
N_FEATURES = N_RAW + N_MISS + N_EXTRA    # 32

FEATURE_NAMES: tuple[str, ...] = tuple(C.INPUT_COLUMNS) + ("DEPTH_RAW",) + tuple(
    f"{c}_miss" for c in C.INPUT_COLUMNS
) + ("DEPTH_miss",) + (
    "miss_ratio", "rel_depth", "depth_step", "depth_index_norm",
)

if HAS_NUMPY:
    import numpy as np  # noqa: F811


def build_row_features(inputs: np.ndarray, missing: np.ndarray,
                       depth: np.ndarray) -> np.ndarray:
    """由分片数据构造 (n, 32) float32 行级特征。

    inputs : (n, 14) float32，已把哨兵转成 NaN
    missing: (n, 14) int8，1 表示该曲线在此行缺测
    depth  : (n,) float32
    """
    n = inputs.shape[0]
    if inputs.shape[1] != N_RAW_CURVES:
        raise AssertionError(
            f"build_row_features expects {N_RAW_CURVES} curves, got {inputs.shape[1]}"
        )
    if missing.shape[1] != N_RAW_CURVES:
        raise AssertionError(
            f"missing mask must have {N_RAW_CURVES} columns, got {missing.shape[1]}"
        )
    X = np.empty((n, N_FEATURES), dtype="float32")
    # 前 13 列：曲线原始值；第 14 列：DEPTH 原始值（深度是有物理含义的通道）
    X[:, 0:N_RAW_CURVES] = inputs
    X[:, N_RAW_CURVES] = depth
    # 缺失位：13 条曲线 + DEPTH
    depth_miss = (~np.isfinite(depth)).astype("int8")
    X[:, N_RAW:N_RAW + N_RAW_CURVES] = missing.astype("float32")
    X[:, N_RAW + N_RAW_CURVES] = depth_miss.astype("float32")

    # 缺失比例：13 条曲线 + depth
    miss_all = np.concatenate([missing, depth_miss[:, None]], axis=1)
    X[:, N_RAW + N_MISS] = miss_all.mean(axis=1).astype("float32")

    # 深度编码（只用该井自身可观测的深度序列）
    d = np.asarray(depth, dtype="float64")
    finite = np.isfinite(d)
    if finite.any():
        dmin, dmax = np.nanmin(d), np.nanmax(d)
        span = max(dmax - dmin, 1e-6)
        rel = (d - dmin) / span
        d_sorted = np.sort(d[finite])
        if d_sorted.size > 1:
            diffs = np.diff(d_sorted)
            step_med = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 0.1
        else:
            step_med = 0.1
        step = np.empty(n, dtype="float64")
        step[1:] = np.diff(d)
        step[0] = step_med
        step = np.where(np.isfinite(step) & (step > 0), step, step_med)
    else:
        rel = np.zeros(n)
        step = np.full(n, 0.1)
    idx_norm = np.arange(n, dtype="float64") / max(n - 1, 1)

    X[:, N_RAW + N_MISS + 1] = np.nan_to_num(rel, nan=0.0).astype("float32")
    X[:, N_RAW + N_MISS + 2] = np.clip(step, 0.0, 5.0).astype("float32")
    X[:, N_RAW + N_MISS + 3] = idx_norm.astype("float32")
    return X


def build_labels(targets: np.ndarray, target_missing: np.ndarray,
                 placeholder: np.ndarray) -> dict[str, np.ndarray]:
    """构造训练标签（含 PERM 的 log10 变换与各目标掩码）。

    返回:
        por      : (n,) float32  原始尺度
        perm_z   : (n,) float32  log10(PERM)，缺测处填 0（由 mask 屏蔽）
        sw       : (n,) float32  标签尺度（占位 99.9；有效实测 8.305–99.9）
        mask     : (n, 3) float32 1=参与监督
        y_ph     : (n,) float32  联合占位标签
    """
    por = np.asarray(targets[:, 0], dtype="float32")
    perm = np.asarray(targets[:, 1], dtype="float64")
    sw = np.asarray(targets[:, 2], dtype="float32")
    tmiss = np.asarray(target_missing, dtype=bool)  # (n,3)

    perm_pos = perm > 0
    perm_safe = np.where(perm_pos, perm, 1e-12)
    perm_z = np.log10(perm_safe).astype("float32")
    perm_z = np.where(perm_pos, perm_z, 0.0).astype("float32")

    mask = (~tmiss).astype("float32")
    y_ph = np.asarray(placeholder, dtype="float32")
    return {
        "por": por,
        "perm_z": perm_z,
        "sw": sw,
        "mask": mask,
        "y_ph": y_ph,
    }


def decode_predictions(por: np.ndarray, perm_z: np.ndarray, sw: np.ndarray,
                       q_ph: np.ndarray | None = None, tau: float | None = None) -> np.ndarray:
    """把网络输出解码成 (n, 3) 的**标签尺度**预测 [POR, PERM, SW]。

    - PERM = 10**clip(z)
    - q_ph/tau 给出时执行**硬切换**：q>=tau 的行直接输出占位常量（E6 才会用到；
      E1 只做连续输出，但保留接口以便同一套代码复用到 E6）。
    """
    from .. import constants as C

    p = np.asarray(por, dtype="float64")
    z = np.clip(np.asarray(perm_z, dtype="float64"), C.PERM_LOG_MIN, C.PERM_LOG_MAX)
    s = np.asarray(sw, dtype="float64")
    perm = np.power(10.0, z)
    out = np.stack([p, perm, s], axis=1)
    if q_ph is not None and tau is not None:
        hit = np.asarray(q_ph, dtype="float64") >= tau
        for j, t in enumerate(C.TARGET_COLUMNS):
            out[hit, j] = C.PLACEHOLDER[t]
    return out.astype("float64")
