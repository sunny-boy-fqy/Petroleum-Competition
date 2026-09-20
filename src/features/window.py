"""F_win：**居中**多尺度窗口统计（E2/P1）。

契约（E2/P1 §5/§6/§7）
----------------------
- 窗长 `{11, 51, 201}` 点（0.1 m 采样 → 1.1 / 5.1 / 20.1 m），**居中对齐**；
- 逐曲线统计 `mean / std / min / max / trend / coverage`：
  `trend` = 窗内对"行序号"做最小二乘的斜率（每点变化量），`coverage` = 窗内有效值占比；
- **缺失不填 0**：窗口统计只用窗内有效值；全窗缺测 → NaN（`coverage` 仍给出 0）；
- 只用**同一口井**的邻域：不跨井、不跨折 → 无井级泄漏。
- 本任务是离线任务，没有因果约束，因此允许居中窗口（与 v1 的 C1W 一致）；
  `trend` 因此也不含"未来信息泄漏"的问题 —— 但**测试井的深度序列本身就可见**，
  所以推理期同样可算（E2/P1 §7）。

实现要点：用 `numpy.lib.stride_tricks.sliding_window_view` 一次拿到所有窗口，
再按 `nan*` 归约，避免 Python 循环；窗长超过井长时自动收缩到井长（并在报告里记 `shrunk`）。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .. import constants as C

DEFAULT_WINDOWS: tuple[int, ...] = (11, 51, 201)
DEFAULT_STATS: tuple[str, ...] = ("mean", "std", "min", "max", "trend", "coverage")

_CURVE_INDEX = {name: i for i, name in enumerate(C.INPUT_COLUMNS)}


def window_names(curves: Sequence[str] = C.INPUT_COLUMNS,
                 windows: Sequence[int] = DEFAULT_WINDOWS,
                 stats: Sequence[str] = DEFAULT_STATS) -> list[str]:
    return [f"{c}_w{w}_{s}" for w in windows for c in curves for s in stats]


def _padded(x: np.ndarray, half: int) -> np.ndarray:
    """按边值（edge）填充 `half` 个点；NaN 边值则补 NaN（不引入假数据）。"""
    pad = [(half, half)] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad, mode="edge")


def _windows(x: np.ndarray, win: int) -> np.ndarray:
    """(n, k) → (n, win, k)：第 i 行是 `[i-half, i+half]` 的居中窗口。"""
    half = win // 2
    xp = _padded(x, half)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(xp, win, axis=0)          # (n+1, k, win)
    return np.transpose(w[: x.shape[0]], (0, 2, 1))   # (n, win, k)


def _trend_slope(win_vals: np.ndarray) -> np.ndarray:
    """窗内最小二乘斜率（对窗内位置 0..win-1 回归），缺失点剔除。

    `win_vals`: (n, win, k) → (n, k)。全窗有效点 < 2 → NaN。
    """
    n, win, k = win_vals.shape
    t = np.arange(win, dtype="float64")
    valid = np.isfinite(win_vals)                       # (n, win, k)
    cnt = valid.sum(axis=1)                             # (n, k)
    v = np.where(valid, win_vals, 0.0)
    t_b = t[None, :, None]
    t_sum = (valid * t_b).sum(axis=1)                   # (n, k)
    y_sum = v.sum(axis=1)                               # (n, k)
    ty_sum = (valid * t_b * v).sum(axis=1)              # (n, k)
    tt_sum = (valid * t_b * t_b).sum(axis=1)            # (n, k)
    denom = cnt * tt_sum - t_sum * t_sum
    num = cnt * ty_sum - t_sum * y_sum
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = num / denom
    slope = np.where((cnt >= 2) & (np.abs(denom) > 1e-12), slope, np.nan)
    return slope


def _stats_for_window(win_vals: np.ndarray, stats: Iterable[str]) -> "np.ndarray":
    """(n, win, k) → (n, k, len(stats))，缺失用 `nan*` 归约（全缺 → NaN）。

    全缺窗口会触发 numpy 的 "All-NaN slice" RuntimeWarning —— 这是**预期**语义
    （输出 NaN），因此这里显式抑制该告警，而不是让它污染训练日志。
    """
    import warnings
    out = []
    stats = list(stats)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for s in stats:
            if s == "mean":
                out.append(np.nanmean(win_vals, axis=1))
            elif s == "std":
                out.append(np.nanstd(win_vals, axis=1))
            elif s == "min":
                out.append(np.nanmin(win_vals, axis=1))
            elif s == "max":
                out.append(np.nanmax(win_vals, axis=1))
            elif s == "trend":
                out.append(_trend_slope(win_vals.astype("float64", copy=False)))
            elif s == "coverage":
                out.append(np.isfinite(win_vals).mean(axis=1))
            else:
                raise ValueError(f"unknown window stat: {s!r}")
    arr = np.stack(out, axis=2)                          # (n, k, n_stats)
    return np.where(np.isfinite(arr), arr, np.nan)


def build_window_features(inputs: "np.ndarray",
                          windows: Sequence[int] = DEFAULT_WINDOWS,
                          stats: Sequence[str] = DEFAULT_STATS,
                          min_window: int = 3) -> tuple["np.ndarray", list[str], dict[str, Any]]:
    """由单井 (n,13) 曲线矩阵构造 F_win。

    返回 `(X, names, meta)`；`meta` 记录实际使用的窗长（过长的窗会被收缩到井长，
    这是**必须上报**的事实，否则小井上的列语义会悄悄变化）。
    """
    # float32 存储：w=201 时 float64 的窗口视图是 ~188 MB/井，而 float32 减半；
    # 统计量对 float32 完全够用（trend 内部再升 float64）。
    x = np.asarray(inputs, dtype="float32")
    if x.ndim != 2 or x.shape[1] != len(C.INPUT_COLUMNS):
        raise ValueError(f"build_window_features expects (n,{len(C.INPUT_COLUMNS)}), "
                         f"got {tuple(x.shape)}")
    n = x.shape[0]
    names: list[str] = []
    blocks: list[np.ndarray] = []
    used: dict[str, int] = {}
    shrunk: list[int] = []
    for w in windows:
        w_eff = int(w)
        if n < w_eff:
            w_eff = max(int(n) if n % 2 == 1 else max(n - 1, 1), min_window)
            shrunk.append(int(w))
        if w_eff % 2 == 0:                     # 居中窗口必须是奇数（half 对称）
            w_eff += 1
        used[str(w)] = w_eff
        names += [f"{c}_w{w}_{s}" for c in C.INPUT_COLUMNS for s in stats]
        wv = _windows(x, w_eff)
        st = _stats_for_window(wv, stats)
        blocks.append(st.reshape(n, -1))
    if not blocks:
        raise ValueError("build_window_features: windows must be non-empty")
    X = np.concatenate(blocks, axis=1).astype("float32")
    meta = {"windows_requested": [int(w) for w in windows],
            "windows_used": used, "shrunk": shrunk, "stats": list(stats),
            "n_rows": int(n), "n_features": int(X.shape[1])}
    return X, names, meta


def label_independence_audit() -> dict[str, Any]:
    """静态审计：窗口特征只能吃曲线矩阵（不得出现目标名）。"""
    import inspect
    banned = {t.lower() for t in C.TARGETS} | {"y_por", "y_perm", "y_sw", "targets"}
    offenders: list[str] = []
    for name, fn in sorted(globals().items()):
        if not inspect.isfunction(fn) or fn.__module__ != __name__:
            continue
        for pname in inspect.signature(fn).parameters:
            if pname.lower() in banned:
                offenders.append(f"{name}({pname})")
    return {"ok": not offenders, "offenders": offenders}
