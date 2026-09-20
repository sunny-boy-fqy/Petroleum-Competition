"""F_well：井级聚合特征（E2/P1），**逐行广播**到该井的每一行。

契约（E2/P1 §5/§6）
------------------
- 每口井 13 条曲线各取 `mean / std / p10 / p50 / p90`（缺失用 `nan*` 归约）；
- 加上与曲线无关的井级量：行数、深度跨度、平均采样间隔、井斜（DEVI）均值、
  曲线整体缺失比例（14 位口径与 F1 一致）；
- **只用该井自身的行**：`.npz` 里的曲线本来就是"同一口井"，因此不存在跨井信息；
  这些统计**不依赖标签**，所以在验证折与测试井上同样可算（推理期一致）；
- 与"跨井标准化"的区别：后者是**折内 fit 的标量**（`RowScaler`），前者是"该井的形状描述"，
  两者不冲突；但**不得**用验证折/测试集的**标签**统计任何东西。

为什么需要它：`资料库/08` §0.1-4 指出 CAL/DEVI/AZIM/BIT/CASE 在单井内近常数，
其"井级水平"才是井间区分度；把井均值显式喂给模型，比让网络从近常数通道里自己推更省样本。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import constants as C

WELL_STATS: tuple[str, ...] = ("mean", "std", "p10", "p50", "p90")
WELL_SCALARS: tuple[str, ...] = ("n_rows", "depth_span", "depth_step_mean", "devi_mean",
                                 "miss_frac_mean")


def well_names(curves: Sequence[str] = C.INPUT_COLUMNS,
               stats: Sequence[str] = WELL_STATS,
               scalars: Sequence[str] = WELL_SCALARS) -> list[str]:
    return [f"{c}_well_{s}" for c in curves for s in stats] + [f"well_{s}" for s in scalars]


def _agg(col: "np.ndarray", stat: str, quantiles: dict[str, float]) -> float:
    v = col[np.isfinite(col)]
    if v.size == 0:
        return float("nan")
    if stat == "mean":
        return float(v.mean())
    if stat == "std":
        return float(v.std(ddof=0))
    q = quantiles.get(stat)
    if q is None:
        raise ValueError(f"unknown well stat: {stat!r}")
    return float(np.percentile(v, q * 100.0))


def build_well_features(inputs: "np.ndarray", depth: "np.ndarray",
                        missing: "np.ndarray | None" = None,
                        stats: Sequence[str] = WELL_STATS,
                        scalars: Sequence[str] = WELL_SCALARS
                        ) -> tuple["np.ndarray", list[str]]:
    """由**单口井**的曲线/深度构造 F_well，并广播到 (n, n_well_cols)。

    `inputs`: (n,13) 曲线（NaN 表示缺测）；`depth`: (n,)；`missing`: (n,13) 0/1。
    返回 `(X, names)`；全部统计只由该井自身的行得到。
    """
    x = np.asarray(inputs, dtype="float64")
    d = np.asarray(depth, dtype="float64").reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(C.INPUT_COLUMNS):
        raise ValueError(f"build_well_features expects (n,{len(C.INPUT_COLUMNS)}), "
                         f"got {tuple(x.shape)}")
    n = x.shape[0]
    quantiles = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

    names: list[str] = []
    vals: list[float] = []
    curve_vals: list[float] = []
    for j, c in enumerate(C.INPUT_COLUMNS):
        for s in stats:
            names.append(f"{c}_well_{s}")
            v = _agg(x[:, j], s, quantiles)
            vals.append(v)
            curve_vals.append(v)
    if not np.isfinite(np.asarray(curve_vals, dtype="float64")).any():
        # 一条曲线都没有有效值 -> 这是数据错误，不允许静默产出一行全 NaN 的特征
        raise ValueError("build_well_features: 该井所有曲线都无有效值（数据异常）")

    if missing is None:
        m = (~np.isfinite(x)).astype("float64")
    else:
        m = np.asarray(missing, dtype="float64")
    d_fin = d[np.isfinite(d)]
    if d_fin.size > 1:
        diffs = np.diff(d_fin)
        step = float(np.mean(np.abs(diffs))) if diffs.size else 0.0
        span = float(d_fin.max() - d_fin.min())
    else:
        step, span = 0.0, 0.0
    scalar_values = {
        "n_rows": float(n),
        "depth_span": span,
        "depth_step_mean": step,
        "devi_mean": _agg(x[:, C.INPUT_COLUMNS.index("DEVI")], "mean", quantiles),
        "miss_frac_mean": float(m.mean()) if m.size else float("nan"),
    }
    for s in scalars:
        if s not in scalar_values:
            raise ValueError(f"unknown well scalar: {s!r}")
        names.append(f"well_{s}")
        vals.append(scalar_values[s])

    row = np.asarray(vals, dtype="float32")
    if not np.isfinite(row).any():
        raise ValueError("build_well_features: 该井所有统计都是 NaN（数据异常）")
    X = np.broadcast_to(row[None, :], (n, row.size)).copy()
    return X, names


def label_independence_audit() -> dict[str, Any]:
    """静态审计：井级特征只能吃曲线/深度/缺测位（不得出现目标名）。"""
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
