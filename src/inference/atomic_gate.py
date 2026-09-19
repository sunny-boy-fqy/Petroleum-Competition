"""原子门：逐目标硬切换 + σ 阈值选择 + 误判代价分解（E6-R3，**纯 numpy，不 import torch**）。

设计契约
--------
官方 `Total = 0.30·Acc_POR + 0.35·Acc_PERM + 0.35·Acc_SW`，而训练标签里
POR=0.1 / PERM=0.01 / SW=99.9 的"原子哨兵行"占了 2/3 以上（联合占位 66.719%）。
对这些行，只要把预测**精确**写成哨兵值就是满分；写错则因为容差
（POR ±0.08·|y|、SW ±0.05·|y|、PERM 1 个数量级）几乎必然丢分。

因此推理期采用**硬切换**：

    pred[:, t] = ATOM_VALUES[t]   if q_atom[:, t] > tau[t]   else   cont[:, t]

**绝不插值**——插值会把 `0.1 + ε` 这种"两头不讨好"的值送进官方评分（POR 容差只有 8%，
占位行插值几乎必错），所以本模块只做精确替换，并用断言锁死这一点。

阈值 τ 的纪律
-------------
`select_tau_per_target` 在**内层 OOF** 上、**逐目标独立**搜索 τ，目标函数就是官方
加权总分贡献（**不是** F1、也不是原子分类准确率）。为避免把 τ 过拟合到单点尖峰，
采用**平台中点规则**：把落在"最优值 `tol` 相对邻域"内的 τ 视为平台，取最长连续平台的
**中点**。这是对"内层 OOF τ 过拟合"的显式防护（写入实验报告的固定口径）。
"""
from __future__ import annotations

from .. import constants as C
from ..portability import HAS_NUMPY, require

if HAS_NUMPY:
    import numpy as np


# ---------------------------------------------------------------- 内部工具
def _atom_vector(atom_values: dict | None = None) -> np.ndarray:
    """原子值向量（列序固定 POR, PERM, SW）。"""
    src = C.ATOM_VALUES if atom_values is None else atom_values
    return np.asarray([float(src[t]) for t in C.TARGETS], dtype="float64")


def _tau_vector(tau) -> np.ndarray:
    """把标量 / 长度 3 的 tau 规范成 (3,) float64。"""
    arr = np.asarray(tau, dtype="float64").reshape(-1)
    if arr.size == 1:
        return np.repeat(arr, 3)
    if arr.size != 3:
        raise ValueError(f"tau must be a scalar or length-3, got {arr.size}")
    return arr


def _resolve_score_fn(score_fn, t: int):
    """解析逐目标分数函数。

    `score_fn` 可以是：
      - 单个 callable `f(y_t, pred_t, mask_t)`（三目标同口径时使用）；
      - 长度 3 的 tuple/list，或 {target_name: callable} 的 dict（POR/PERM/SW 口径不同时使用）。
    """
    if isinstance(score_fn, dict):
        return score_fn[C.TARGETS[t]]
    if isinstance(score_fn, (tuple, list)):
        if len(score_fn) != 3:
            raise ValueError(f"score_fn sequence must have 3 entries, got {len(score_fn)}")
        return score_fn[t]
    return score_fn


def _assert_no_interpolation(cont: np.ndarray, out: np.ndarray, hit_t: np.ndarray,
                             atom_value: float, col: int) -> None:
    """锁死"硬切换、无插值"：命中列必须精确等于原子值，未命中列必须与连续值逐位相同。"""
    if not np.array_equal(out[hit_t, col], np.full(int(hit_t.sum()), atom_value)):
        raise AssertionError(
            f"per_target_hard_switch: column {col} atom branch is not exactly the atom value")
    miss = ~hit_t
    if not np.array_equal(out[miss, col], cont[miss, col]):
        raise AssertionError(
            f"per_target_hard_switch: column {col} continuous branch was modified (interpolation?)")


# ---------------------------------------------------------------- 硬切换
def per_target_hard_switch(cont, q_atom, tau, atom_values: dict | None = None) -> np.ndarray:
    """**逐目标独立**硬切换（无插值）。

    cont : (N,3) 标签尺度连续预测 [POR, PERM(线性), SW]
    q_atom: (N,3) 原子概率（列序 POR/PERM/SW）
    tau  : 标量或长度 3；命中条件为**严格大于** `q_atom[:,t] > tau[t]`
    返回 : (N,3) float64，列 t 为 `ATOM_VALUES[t]`（命中）或 `cont[:,t]`（未命中）
    """
    require("numpy")
    av = _atom_vector(atom_values)
    c = np.asarray(cont, dtype="float64")
    q = np.asarray(q_atom, dtype="float64")
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"cont must be (N,3), got {tuple(c.shape)}")
    if q.shape != c.shape:
        raise ValueError(f"q_atom must have the same shape as cont {tuple(c.shape)}, got {tuple(q.shape)}")
    t = _tau_vector(tau)

    hit = q > t[None, :]
    out = c.copy()
    for j in range(3):
        out[hit[:, j], j] = av[j]

    # 硬切换断言：既不是插值，也不做跨列串扰
    for j in range(3):
        _assert_no_interpolation(c, out, hit[:, j], float(av[j]), j)
    return out


def joint_guard(out, q_joint, tau_high, atom_values: dict | None = None) -> np.ndarray:
    """联合守卫：`q_joint > tau_high` 的行**三列同时**改写为原子值。

    这是**高置信度保险**，默认关闭（由调用方显式决定是否使用）。
    """
    require("numpy")
    av = _atom_vector(atom_values)
    o = np.asarray(out, dtype="float64").copy()
    if o.ndim != 2 or o.shape[1] != 3:
        raise ValueError(f"out must be (N,3), got {tuple(o.shape)}")
    q = np.asarray(q_joint, dtype="float64").reshape(-1)
    if q.size != o.shape[0]:
        raise ValueError(f"q_joint length {q.size} != N {o.shape[0]}")
    hit = q > float(tau_high)
    for j in range(3):
        o[hit, j] = av[j]
    return o


# ---------------------------------------------------------------- τ 选择
def _longest_plateau_midpoint(taus: np.ndarray, objs: np.ndarray, tol: float) -> tuple:
    """在目标曲线 `objs` 上取"最优值 tol 相对邻域"的最长连续平台，返回 (midpoint, (lo,hi))。

    平台判定：`objs >= best − tol·max(|best|, 1e-12)`。
    并列最长时取"平台内最大值更大"的那段（含全局 argmax 者优先）。
    """
    best = float(np.max(objs))
    thr = best - tol * max(abs(best), 1e-12)
    good = objs >= thr
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(objs)
    while i < n:
        if good[i]:
            j = i
            while j + 1 < n and good[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:                                  # 理论上不会发生（best 自身必满足）
        k = int(np.argmax(objs))
        return float(taus[k]), (float(taus[k]), float(taus[k]))
    best_run = max(runs, key=lambda r: (r[1] - r[0], float(np.max(objs[r[0]:r[1] + 1]))))
    lo = float(taus[best_run[0]])
    hi = float(taus[best_run[1]])
    return 0.5 * (lo + hi), (lo, hi)


def select_tau_per_target(score_fn, cont, q_atom, y, mask, grid=None,
                          tol: float = 1e-4) -> dict:
    """逐目标独立搜索硬切换阈值 τ（目标 = 官方加权总分贡献）。

    score_fn(y_t, pred_t, mask_t) -> float
        由调用方提供的**官方逐目标准确率**包装（[0,1]）；本模块不依赖 `score.py`。
        可传单个 callable，或长度 3 的 tuple/list、`{target: callable}` dict
        （POR/PERM/SW 口径不同，见 `_resolve_score_fn`）。

    tol : 平台判定的**相对**容差（默认 1e-4），见模块 docstring 的"平台中点规则"。

    返回::

        {
          "tau": (3,) float64,                 # 选中阈值（平台中点）
          "grid": (G,) 使用的候选网格,
          "curve": {target: [(tau, acc), ...]},
          "objective": float,                  # Σ w_t · acc_t(选中 τ)
          "plateau": {target: [lo, hi]},
        }
    """
    require("numpy")
    c = np.asarray(cont, dtype="float64")
    q = np.asarray(q_atom, dtype="float64")
    yy = np.asarray(y, dtype="float64")
    m = np.asarray(mask, dtype="float64")
    if c.shape != q.shape or yy.shape != c.shape or m.shape != c.shape:
        raise ValueError(
            f"cont/q_atom/y/mask must all be (N,3): {tuple(c.shape)} {tuple(q.shape)} "
            f"{tuple(yy.shape)} {tuple(m.shape)}")
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    grid = np.asarray(grid, dtype="float64").reshape(-1)
    if grid.size == 0:
        raise ValueError("grid must be non-empty")
    av = _atom_vector(None)

    curve: dict[str, list] = {}
    plateau: dict[str, list] = {}
    taus = np.empty(3, dtype="float64")
    accs = np.empty(3, dtype="float64")

    for t, name in enumerate(C.TARGETS):
        obs = m[:, t] > 0
        fn = _resolve_score_fn(score_fn, t)
        acc_curve = np.empty(grid.size, dtype="float64")
        for gi, tau in enumerate(grid):
            pred = np.where(q[:, t] > tau, av[t], c[:, t])
            if not obs.any():
                acc_curve[gi] = 0.0
                continue
            acc_curve[gi] = float(fn(yy[obs, t], pred[obs], m[obs, t]))
        obj_curve = float(C.TARGET_WEIGHTS[t]) * acc_curve
        mid, (lo, hi) = _longest_plateau_midpoint(grid, obj_curve, float(tol))
        taus[t] = mid
        # 选中的 τ 必须能通过格点重建（平台中点可能不是格点，但必须在平台内）：
        # 采用平台内与中点最接近的实际格点，保证可复现且落在 plateau 内。
        k = int(np.argmin(np.abs(grid - mid)))
        if not (lo - 1e-12 <= grid[k] <= hi + 1e-12):
            k = int(np.argmax(obj_curve))
        taus[t] = float(grid[k])
        accs[t] = float(acc_curve[k])
        curve[name] = [(float(g), float(a)) for g, a in zip(grid, acc_curve)]
        plateau[name] = [float(lo), float(hi)]

    return {
        "tau": taus,
        "grid": grid,
        "curve": curve,
        "objective": float(np.sum(np.asarray(C.TARGET_WEIGHTS, dtype="float64") * accs)),
        "plateau": plateau,
    }


# ---------------------------------------------------------------- 误判代价分解
def misclassification_cost_report(score_fn, cont, q_atom, y, mask, tau,
                                  atom_values: dict | None = None) -> dict:
    """误判代价分解（"误判代价分解"报告口径）。

    逐目标报告：
      (a) `n_false_atom`  : 观测到的**非原子行**被硬切换成原子值的行数；
      (b) `n_missed_atom` : **原子行**被留给连续头的行数；
      以及连续口径 / 切换口径的官方准确率与**分数增量** `delta = acc_switch − acc_cont`。

    `score_fn(y_t, pred_t, mask_t)` 语义同 `select_tau_per_target`。
    返回 `{target: {...}, "total": {..., "delta": 加权总分增量}}`。
    """
    require("numpy")
    av = _atom_vector(atom_values)
    c = np.asarray(cont, dtype="float64")
    q = np.asarray(q_atom, dtype="float64")
    yy = np.asarray(y, dtype="float64")
    m = np.asarray(mask, dtype="float64")
    if c.shape != q.shape or yy.shape != c.shape or m.shape != c.shape:
        raise ValueError("cont/q_atom/y/mask must all be (N,3) with identical shapes")
    t = _tau_vector(tau)

    report: dict = {}
    total_cont = total_switch = 0.0
    for j, name in enumerate(C.TARGETS):
        obs = m[:, j] > 0
        if not obs.any():
            report[name] = {"n_observed": 0, "n_false_atom": 0, "n_missed_atom": 0,
                            "acc_cont": 0.0, "acc_switch": 0.0, "delta": 0.0}
            continue
        fn = _resolve_score_fn(score_fn, j)
        is_atom = np.abs(yy[:, j] - av[j]) <= C.PLACEHOLDER_ABS_TOL
        hit = q[:, j] > t[j]
        n_false_atom = int(np.sum(obs & (~is_atom) & hit))
        n_missed_atom = int(np.sum(obs & is_atom & (~hit)))
        pred_switch = np.where(hit, av[j], c[:, j])
        acc_cont = float(fn(yy[obs, j], c[obs, j], m[obs, j]))
        acc_switch = float(fn(yy[obs, j], pred_switch[obs], m[obs, j]))
        w = float(C.TARGET_WEIGHTS[j])
        total_cont += w * acc_cont
        total_switch += w * acc_switch
        report[name] = {
            "n_observed": int(obs.sum()),
            "n_atom": int((obs & is_atom).sum()),
            "n_false_atom": n_false_atom,
            "n_missed_atom": n_missed_atom,
            "acc_cont": acc_cont,
            "acc_switch": acc_switch,
            "delta": acc_switch - acc_cont,
        }
    report["total"] = {
        "acc_cont": total_cont,
        "acc_switch": total_switch,
        "delta": total_switch - total_cont,
    }
    return report
