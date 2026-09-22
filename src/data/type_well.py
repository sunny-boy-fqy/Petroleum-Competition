"""WP8：类型井选择（KL 散度 / DTW 距离）——参考 SPWLA 2021 冠军方案。

用法：对每口测试井，在训练井中找“最相似”的 top-k 类型井，用于：
  * 折内 scaler / 分布匹配的参考；
  * 只在该类型井上训练一个“适配成员”；
  * E8 集成中的井级多样性来源。

只使用**输入曲线**，不接触任何标签；纯 numpy，scipy 可选。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_CURVES: tuple[str, ...] = ("GR", "AC", "DEN", "CNL", "RT", "RXO")


def _shard_inputs(shard: Mapping[str, Any], curves: Sequence[str] = DEFAULT_CURVES) -> np.ndarray:
    """从 raw shard 取曲线子集；优先 inputs 列名映射，缺失列用 NaN。"""
    inputs = np.asarray(shard.get("inputs"), dtype="float64")
    if inputs.ndim != 2:
        raise ValueError(f"shard['inputs'] 必须 (n,13)，got {inputs.shape}")
    from .. import constants as C
    idx = {name: i for i, name in enumerate(C.INPUT_COLUMNS)}
    cols = []
    for name in curves:
        if name in idx and idx[name] < inputs.shape[1]:
            cols.append(inputs[:, idx[name]])
        else:
            cols.append(np.full(inputs.shape[0], np.nan, dtype="float64"))
    return np.stack(cols, axis=1)


def _finite_matrix(X: np.ndarray, max_rows: int | None = 2000,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """去掉含 NaN 的行；最多下采样到 max_rows（KL/DTW 用）。"""
    X = np.asarray(X, dtype="float64")
    keep = np.isfinite(X).all(axis=1)
    X = X[keep]
    if max_rows is not None and X.shape[0] > int(max_rows):
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(X.shape[0], size=int(max_rows), replace=False)
        X = X[np.sort(sel)]
    return X


def standardize_features(X: np.ndarray, mu: np.ndarray | None = None,
                         sd: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype="float64")
    mu = X.mean(axis=0) if mu is None else np.asarray(mu, dtype="float64")
    sd = X.std(axis=0) if sd is None else np.asarray(sd, dtype="float64")
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _pairwise_sqdist(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty((a.shape[0], b.shape[0]), dtype="float64")
    for i in range(0, a.shape[0], chunk):
        aa = a[i:i + chunk]
        d = ((aa[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        out[i:i + chunk] = d
    return out


def kl_divergence(x: Any, y: Any, jitter: float = 1e-9,
                  seed: int = 0) -> float:
    """Pérez-Cruz 的 kNN KL 估计 ``D(x || y)``（对称化由调用方做）。

    为保证纯 numpy 可用：最多各取 2000 行，brute-force 最近邻（分块）。
    """
    rng = np.random.default_rng(seed)
    x = _finite_matrix(x, max_rows=2000, rng=rng)
    y = _finite_matrix(y, max_rows=2000, rng=rng)
    if x.shape[0] < 3 or y.shape[0] < 3:
        return float("nan")
    d = x.shape[1]
    # 加极小 jitter 避免测井曲线取整导致的并列
    x = x + float(jitter) * rng.normal(size=x.shape)
    y = y + float(jitter) * rng.normal(size=y.shape)
    dxx = _pairwise_sqdist(x, x)
    dxy = _pairwise_sqdist(x, y)
    np.fill_diagonal(dxx, np.inf)
    r = np.sqrt(np.partition(dxx, 1, axis=1)[:, 1])       # x 到最近邻 x
    s = np.sqrt(dxy.min(axis=1))                           # x 到最近邻 y
    r = np.maximum(r, 1e-12)
    s = np.maximum(s, 1e-12)
    n, m = x.shape[0], y.shape[0]
    return float(-np.log(r / s).sum() * d / n + np.log(m / (n - 1.0)))


def symmetric_kl(x: Any, y: Any, seed: int = 0) -> float:
    a = kl_divergence(x, y, seed=seed)
    b = kl_divergence(y, x, seed=seed + 1)
    return float(a + b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def dtw_distance(a: Any, b: Any, max_points: int = 400,
                 band: float | None = 0.2) -> float:
    """归一化 DTW 距离（下采样到 max_points，Sakoe-Chiba 带约束）。

    仅使用第一条曲线或一维序列；多维时先做标准化后取欧氏距离。
    """
    a = _finite_matrix(np.asarray(a, dtype="float64").reshape(-1, 1),
                       max_rows=max_points)
    b = _finite_matrix(np.asarray(b, dtype="float64").reshape(-1, 1),
                       max_rows=max_points)
    n, m = a.shape[0], b.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    a = (a - a.mean()) / max(a.std(), 1e-12)
    b = (b - b.mean()) / max(b.std(), 1e-12)
    D = np.abs(a.reshape(-1) [:, None] - b.reshape(-1)[None, :])
    w = int(max(1, float(band) * max(n, m))) if band else max(n, m)
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - w)
        hi = min(m, i + w)
        for j in range(lo, hi + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    return float(C[n, m] / max(n + m, 1))


def well_signature(shard: Mapping[str, Any], curves: Sequence[str] = DEFAULT_CURVES,
                   max_rows: int = 2000) -> dict[str, Any]:
    """井签名：曲线均值/标准差，用于快速相似度/代表采样。"""
    X = _finite_matrix(_shard_inputs(shard, curves), max_rows=max_rows)
    if X.shape[0] == 0:
        return {"mean": np.full(len(curves), np.nan), "std": np.full(len(curves), np.nan)}
    return {"mean": X.mean(axis=0), "std": X.std(axis=0), "n": int(X.shape[0]),
            "curves": tuple(curves)}


def select_type_wells(test_shard: Mapping[str, Any],
                      train_shards: Mapping[str, Mapping[str, Any]],
                      topk: int = 3, method: str = "kl",
                      curves: Sequence[str] = DEFAULT_CURVES,
                      standardize: bool = True) -> dict[str, Any]:
    """为一口测试井选 top-k 训练类型井。

    返回 ``{test, method, topk, candidates:[{well, distance}]}``；distance 越小越相似。
    """
    method = str(method).lower()
    Xt = _finite_matrix(_shard_inputs(test_shard, curves), max_rows=2000)
    if Xt.shape[0] < 3:
        return {"test": str(test_shard.get("well_id", "?")), "method": method,
                "topk": int(topk), "candidates": []}
    rows = []
    for wid, sh in train_shards.items():
        Xs = _finite_matrix(_shard_inputs(sh, curves), max_rows=2000)
        if Xs.shape[0] < 3:
            continue
        if method == "kl":
            a, b = Xt, Xs
            if standardize:
                star = np.vstack([a, b])
                mu, sd = star.mean(axis=0), star.std(axis=0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                a, b = (a - mu) / sd, (b - mu) / sd
            dist = symmetric_kl(a, b)
        elif method == "dtw":
            dist = dtw_distance(Xt[:, 0], Xs[:, 0])
        elif method == "signature":
            st, ss = well_signature(test_shard, curves), well_signature(sh, curves)
            dist = float(np.linalg.norm(np.nan_to_num(st["mean"] - ss["mean"])))
        else:
            raise ValueError(f"unknown method {method!r}; expected kl/dtw/signature")
        rows.append({"well": str(wid), "distance": float(dist)})
    rows.sort(key=lambda r: (np.inf if not np.isfinite(r["distance"]) else r["distance"]))
    return {"test": str(test_shard.get("well_id", "?")), "method": method,
            "topk": int(topk), "candidates": rows[:max(int(topk), 1)]}


def select_type_wells_batch(test_shards: Mapping[str, Mapping[str, Any]],
                            train_shards: Mapping[str, Mapping[str, Any]],
                            topk: int = 3, method: str = "kl",
                            curves: Sequence[str] = DEFAULT_CURVES) -> dict[str, Any]:
    return {str(tid): select_type_wells(sh, train_shards, topk=topk, method=method,
                                        curves=curves)
            for tid, sh in test_shards.items()}
