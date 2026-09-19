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


def _atom_tolerances() -> np.ndarray:
    """每个原子值的比较容差：max(1e-9, 1 ULP of float32)。

    标签在 `parse.py` 里是 float64，但 `build_labels` 会把 POR/SW 转成 float32；
    直接用 `== 0.1` 在 float32 上会失败（float32(0.1) ≠ 0.1），所以用
    "半个 float32 ULP" 级别的极小容差，既吸收 float32 舍入，又不会把真实连续值误判为原子。
    """
    vals = np.asarray([C.ATOM_VALUES[t] for t in C.TARGETS], dtype="float64")
    return np.maximum(C.PLACEHOLDER_ABS_TOL,
                      np.abs(np.spacing(vals.astype("float32")).astype("float64")))


def build_labels(targets: np.ndarray, target_missing: np.ndarray,
                 placeholder: np.ndarray) -> dict[str, np.ndarray]:
    """构造训练标签（含 PERM 的 log10 变换、各目标掩码与**原子/联合**标签）。

    返回:
        por      : (n,) float32  原始尺度
        perm_z   : (n,) float32  log10(PERM)，缺测处填 0（由 mask 屏蔽）
        sw       : (n,) float32  标签尺度（占位 99.9；有效实测 8.305–99.9）
        mask     : (n, 3) float32 1=参与监督
        y_ph     : (n,) float32  联合占位标签（= y_joint 的旧名，保留兼容）
        y_atom   : (n, 3) float32 逐目标原子标签（POR/PERM/SW 列序）
        y_joint  : (n,) float32  三目标同时为原子值 且 三目标均非缺测

    **原子标签要求该目标被观测到**（NOT missing）：
        `y_atom[:,t] = (y[:,t] == ATOM_VALUES[t]) & (~missing[:,t])`
    """
    raw = np.asarray(targets, dtype="float64")
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"targets must be (n,3), got {tuple(raw.shape)}")
    por = np.asarray(raw[:, 0], dtype="float32")
    perm = np.asarray(raw[:, 1], dtype="float64")
    sw = np.asarray(raw[:, 2], dtype="float32")
    tmiss = np.asarray(target_missing, dtype=bool)  # (n,3)
    if tmiss.shape != raw.shape:
        raise ValueError(f"target_missing must be {raw.shape}, got {tuple(tmiss.shape)}")

    perm_pos = perm > 0
    perm_safe = np.where(perm_pos, perm, 1e-12)
    perm_z = np.log10(perm_safe).astype("float32")
    perm_z = np.where(perm_pos, perm_z, 0.0).astype("float32")

    mask = (~tmiss).astype("float32")
    y_ph = np.asarray(placeholder, dtype="float32")

    # 原子标签（观测到 且 等于原子哨兵值）
    atom_vals = np.asarray([C.ATOM_VALUES[t] for t in C.TARGETS], dtype="float64")
    tol = _atom_tolerances()[None, :]
    observed = ~tmiss
    y_atom_hit = (np.abs(raw - atom_vals[None, :]) <= tol) & observed
    y_atom = y_atom_hit.astype("float32")
    y_joint = (y_atom_hit.all(axis=1) & (~tmiss).any(axis=1)).astype("float32")

    return {
        "por": por,
        "perm_z": perm_z,
        "sw": sw,
        "mask": mask,
        "y_ph": y_ph,
        "y_atom": y_atom,
        "y_joint": y_joint,
    }


# ---------------------------------------------------------------- 训练折目标尺度
def _robust_scale(v: np.ndarray) -> float:
    """稳健尺度 `IQR/1.349`；IQR 为 0 时退化为标准差。"""
    if v.size < 2:
        return float(np.std(v)) if v.size else 0.0
    q1, q3 = np.percentile(v, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        return float(np.std(v))
    return iqr / 1.349


def fit_target_scalers(y_por: np.ndarray, y_sw: np.ndarray, mask: np.ndarray,
                       z_perm: np.ndarray | None = None) -> dict:
    """**只用训练折**观测样本拟合目标尺度（纯 numpy，返回可 JSON 序列化的 dict）。

    返回键（全部为 python float）::

        por_median, por_max (=1.2·max(valid POR)), sw_mu (=median(valid SW)),
        sw_sigma (=IQR(valid SW)/1.349), perm_z_median, s_por, s_sw

    `s_por`/`s_sw` 是 `aux_loss` 的稳健归一化尺度（IQR/1.349，退化时用 std），
    必须与 `sw_mu`/`sw_sigma` 一起写入 checkpoint manifest / scaler JSON。
    `z_perm` 可选：给出时用有效行的中位数填 `perm_z_median`，否则用默认 −0.08。

    无任何有效行时抛 `ValueError`（绝不允许在空切片上静默产出 NaN 尺度）。
    """
    por = np.asarray(y_por, dtype="float64").reshape(-1)
    sw = np.asarray(y_sw, dtype="float64").reshape(-1)
    m = np.asarray(mask, dtype=bool)
    if m.shape != (por.size, 3):
        raise ValueError(f"mask must be ({por.size},3) float/bool, got {tuple(m.shape)}")

    vp = por[m[:, 0] & np.isfinite(por)]
    vs = sw[m[:, 2] & np.isfinite(sw)]
    if vp.size == 0:
        raise ValueError("fit_target_scalers: no valid (observed, finite) POR rows in this fold")
    if vs.size == 0:
        raise ValueError("fit_target_scalers: no valid (observed, finite) SW rows in this fold")

    s_por = _robust_scale(vp)
    s_sw = _robust_scale(vs)
    scaler = {
        "por_median": float(np.median(vp)),
        "por_max": float(C.POR_MAX_BUFFER * np.max(vp)),
        "sw_mu": float(np.median(vs)),
        "sw_sigma": float(s_sw),
        "perm_z_median": -0.08,
        "s_por": float(s_por),
        "s_sw": float(s_sw),
    }
    if z_perm is not None:
        z = np.asarray(z_perm, dtype="float64").reshape(-1)
        if z.size != por.size:
            raise ValueError("z_perm must have the same length as y_por")
        vz = z[m[:, 1] & np.isfinite(z)]
        if vz.size:
            scaler["perm_z_median"] = float(np.median(vz))
    return scaler


def apply_sw_scaler(sw: np.ndarray, scaler: dict) -> np.ndarray:
    """SW 标签尺度 → 归一化尺度：`z = (sw − sw_mu) / sw_sigma`。"""
    sigma = float(scaler["sw_sigma"])
    if sigma == 0.0:
        raise ZeroDivisionError("apply_sw_scaler: sw_sigma is 0")
    return (np.asarray(sw, dtype="float64") - float(scaler["sw_mu"])) / sigma


def invert_sw_scaler(z: np.ndarray, scaler: dict) -> np.ndarray:
    """`apply_sw_scaler` 的逆：`sw = sw_mu + sw_sigma · z`。"""
    return np.asarray(z, dtype="float64") * float(scaler["sw_sigma"]) + float(scaler["sw_mu"])


def decode_predictions(por, perm_z=None, sw=None,
                       q_ph: np.ndarray | None = None, tau: float | None = None,
                       q_atom: np.ndarray | None = None,
                       tau_atom=None, q_joint: np.ndarray | None = None,
                       tau_high: float | None = None,
                       atom_values: dict | None = None) -> np.ndarray:
    """把网络输出解码成 (n, 3) 的**标签尺度**预测 [POR, PERM, SW]。

    兼容两种调用：
      1. 旧式位置参数 `decode_predictions(por, perm_z, sw, q_ph, tau)`；
      2. 新式 dict：`decode_predictions(out)`，`out` 含 `por/perm_z/sw/q_atom/q_joint`
         （`ph_logit` 作为 `q_joint` 的别名）。

    解码规则：
      - POR 直出标签尺度；`PERM = 10**clip(z, -6, 6)`（严格 > 0）；
      - **SW 绝不裁剪到 [0,1]**；仅做 [0,100] 的软保护裁剪（把外推越界值压回标签域，
        实测有效 8.305–99.9，占位 99.9）；
      - `q_atom`/`tau_atom` 给出时执行**逐目标硬切换**（`inference.atomic_gate`），
        **无任何插值**；否则若给出旧式 `q_ph`/`tau`，则按联合占位硬切换（保留旧行为）；
      - `q_joint`/`tau_high` 给出时执行联合守卫（默认由调用方决定是否开启）。
    """
    from ..inference.atomic_gate import joint_guard, per_target_hard_switch

    if isinstance(por, dict):
        out = por
        por = out.get("por")
        perm_z = out.get("perm_z")
        sw = out.get("sw")
        q_atom = out.get("q_atom", q_atom)
        q_joint = out.get("q_joint", out.get("ph_logit", q_joint))
    if por is None or perm_z is None or sw is None:
        raise ValueError("decode_predictions needs por, perm_z and sw")

    p = np.asarray(por, dtype="float64")
    z = np.clip(np.asarray(perm_z, dtype="float64"), C.PERM_LOG_MIN, C.PERM_LOG_MAX)
    s = np.clip(np.asarray(sw, dtype="float64"), 0.0, 100.0)   # 软保护裁剪（非 [0,1]！）
    out = np.stack([p, np.power(10.0, z), s], axis=1)

    if q_atom is not None and tau_atom is not None:
        out = per_target_hard_switch(out, np.asarray(q_atom, dtype="float64"),
                                     tau_atom, atom_values)
    elif q_ph is not None and tau is not None:
        hit = np.asarray(q_ph, dtype="float64") >= float(tau)
        for j, t in enumerate(C.TARGETS):
            out[hit, j] = C.ATOM_VALUES[t]
    if q_joint is not None and tau_high is not None:
        out = joint_guard(out, np.asarray(q_joint, dtype="float64"), tau_high, atom_values)
    return out.astype("float64")
