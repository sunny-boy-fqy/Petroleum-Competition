"""WP9：Kennard-Stone 代表采样（参考 SPWLA 2021 MoLPhy）。

用途：inner-train/inner-val 划分或选择代表性子集，使验证集覆盖特征空间极端点。
outer 折仍必须按井互斥且冻结；本模块只改 inner split。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _dist_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype="float64")
    sq = (X * X).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def kennard_stone(X: Any, n_select: int, seed: Sequence[int] | None = None) -> list[int]:
    """返回 KS 选中的行索引（按选择顺序）。"""
    X = np.asarray(X, dtype="float64")
    n = X.shape[0]
    if n <= 0:
        return []
    n_select = int(np.clip(n_select, 1, n))
    D = _dist_matrix(X)
    if seed is None or len(seed) == 0:
        i, j = np.unravel_index(np.argmax(D), D.shape)
        selected = [int(i), int(j)] if i != j else [int(i)]
    else:
        selected = [int(s) for s in seed]
    selected = list(dict.fromkeys(selected))[:n_select]
    min_dist = np.min(D[selected], axis=0)
    for s in selected:
        min_dist[s] = -np.inf
    while len(selected) < n_select:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, D[nxt])
        min_dist[nxt] = -np.inf
    return selected


def representative_inner_split(well_ids: Sequence[str],
                               signatures: Mapping[str, Any],
                               n_val: int, seed: int = 42) -> tuple[list[str], list[str]]:
    """按井签名做 KS：选 ``n_val`` 口代表井作 inner-val，其余 inner-train。

    ``signatures[w]`` 可为 1D 向量（如 mean/std 拼接）。
    """
    wells = [str(w) for w in well_ids]
    if len(wells) <= int(n_val) + 1:
        return list(wells), list(wells)
    X = np.stack([np.asarray(signatures[w], dtype="float64").reshape(-1)
                  for w in wells], axis=0)
    X = np.nan_to_num(X)
    # KS 先从全局极端点开始；这里 seed 用最远的两口井
    idx = kennard_stone(X, n_select=int(n_val), seed=None)
    val = [wells[i] for i in idx]
    tr = [w for w in wells if w not in set(val)]
    return tr, val
