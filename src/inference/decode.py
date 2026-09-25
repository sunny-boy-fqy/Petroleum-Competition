"""E7/P1 解码后处理（**纯 numpy**）：逐目标 bias / 收缩 / 期望值解码 / 分位收缩 / 敏感性。

为什么单独一层
-------------
E7 要回答"解码本身还能不能榨出分数"，而这件事**必须可冻结、可复算**：
搜索出来的 `decode_v1.json` 要能被 `predict.py` 直接读回。因此把全部算子放在这里，
每个算子都满足三条纪律：

1. **不碰原子门控的优先级**：原子切换永远在连续头后处理**之前/之上**做优先级判定，
   后处理只作用在"没被切换到原子值"的行上（`assert_atom_priority` 给机械证据）；
2. **SW 只做 [0,100] 软裁剪**（绝不 [0,1]、绝不 ×100）——`sw_only_soft_clip` 复用同一收据；
3. **所有参数只在 inner-OOF 上选**（`E7/code/decode_search.py::search_decode`），
   本模块只提供算子与敏感性报告，不做任何"看全折"的选择。

算子（逐目标独立）
----------------
* `apply_bias`        : `ŷ ← ŷ + b_t`（修正系统性偏移）；
* `apply_shrink`      : `ŷ ← μ_t + α_t·(ŷ − μ_t)`（α<1 收缩方差，α>1 放大；μ 由调用方给训练折统计）；
* `quantile_shrink`   : `ŷ ← (1−a)·ŷ + a·Q̂(ŷ)`（Q̂ 为目标列在 OOF 上的经验分位映射，单调）；
* `expected_value_decode`: 按 `q_atom` 分箱，逐箱比较"原子值 vs 连续值"的官方得分并给出动作表，
   以及由动作表导出的**单调 τ**（高置信箱才切原子）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import constants as C
from ..portability import HAS_NUMPY

# 官方容差（用于敏感性判定与得分函数）
DELTA = {"POR": C.DELTA_POR, "PERM": None, "SW": C.DELTA_SW}
DEFAULT_SENSITIVITY_TOL: float = 0.005
DEFAULT_BIAS_GRID = {"POR": (-0.005, 0.005, 11), "PERM": (-0.05, 0.05, 11),
                     "SW": (-0.02, 0.02, 11)}


def _require_numpy() -> None:
    if not HAS_NUMPY:
        raise RuntimeError("decode 需要 numpy")


# ---------------------------------------------------------------- 配置读写
@dataclass
class DecodeConfig:
    """可冻结的解码配置（`versions/configs/decode_v1.json`）。"""
    schema_version: int = 1
    version: str = "decode_v1"
    bias: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in C.TARGETS})
    shrink: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in C.TARGETS})
    shrink_centers: dict[str, float] = field(default_factory=dict)
    quantile_shrink: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in C.TARGETS})
    expected_value: dict[str, bool] = field(default_factory=lambda: {t: False for t in C.TARGETS})
    tau: dict[str, float] = field(default_factory=dict)
    # WP1：原子概率校准参数（temperature / isotonic），只允许 inner-OOF 拟合
    atom_calibration: dict[str, Any] = field(default_factory=dict)
    # WP1：期望分数动作表（每目标 bin_edges/bin_actions/bin_delta）
    action_table: dict[str, Any] = field(default_factory=dict)
    sw_clip: tuple[float, float] = (0.0, 100.0)
    inner_only: bool = True
    selected_on: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_decode_config(pred: Any, config: DecodeConfig | None = None,
                        q_atom: Any | None = None,
                        reference: Any | None = None) -> "np.ndarray":
    """把冻结的 ``decode_v1`` 配置作用到标签尺度连续预测上。

    严格顺序：**连续后处理 → 原子硬切换**。本函数只做连续后处理，原子硬切换仍由调用方
    使用返回后的连续值执行，以保证命中原子行不会被后处理挪走。

    当前支持：逐目标 bias、shrink（需 ``shrink_centers``）、quantile_shrink（需
    ``reference`` 分位数）。``expected_value`` 通过调用方传入 config.tau 作为硬切换
    阈值来体现。
    """
    _require_numpy()
    out = np.asarray(pred, dtype="float64").copy()
    if config is None:
        return sw_only_soft_clip(out, (0.0, 100.0))
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"pred 必须为 (N,3)，got {out.shape}")
    # 1) 逐目标 bias
    if config.bias:
        out = apply_bias(out, config.bias)
    # 2) 逐目标收缩；center 必须来自冻结配置，禁止在测试集上重新估计
    if config.shrink:
        if config.shrink_centers:
            out = apply_shrink(out, config.shrink, centers=config.shrink_centers)
        elif all(abs(float(v) - 1.0) <= 1e-12 for v in config.shrink.values()):
            pass
        else:
            # 没有冻结 center 时不做 shrink，避免用测试集中位数引入分布漂移。
            pass
    # 3) 分位收缩（可选，只在内折上拟合参考分布时使用）
    if reference is not None and config.quantile_shrink:
        out = quantile_shrink(out, config.quantile_shrink, reference)
    return sw_only_soft_clip(out, config.sw_clip)


def calibrate_atom_probs(q_atom: Any, calibration: dict[str, Any] | None) -> "np.ndarray":
    """按冻结配置校准 q_atom；没有配置时原样返回。"""
    _require_numpy()
    q = np.asarray(q_atom, dtype="float64")
    if not calibration:
        return q
    out = q.copy()
    # 1) 全局温度（也支持逐目标 list/dict）
    temp = calibration.get("temperature")
    if temp is not None:
        if isinstance(temp, dict):
            temps = [float(temp.get(t, 1.0)) for t in C.TARGETS]
        elif isinstance(temp, (list, tuple)):
            temps = [float(v) for v in temp]
        else:
            temps = [float(temp)] * out.shape[1]
        from . import calibration as CAL
        for j in range(out.shape[1]):
            T = max(float(temps[j]), 1e-6)
            out[:, j] = CAL.apply_temperature(CAL.logit(out[:, j]), T)
    # 2) 等渗校准（逐目标优先，其次全局）
    iso = calibration.get("isotonic")
    if iso:
        from . import calibration as CAL
        if isinstance(iso, dict) and "bin_edges" in iso:
            out = CAL.apply_isotonic_binned(out, iso)
        elif isinstance(iso, dict):
            for j, t in enumerate(C.TARGETS):
                fit = iso.get(t)
                if isinstance(fit, dict):
                    out[:, j] = CAL.apply_isotonic_binned(out[:, j], fit)
    return np.clip(out, 0.0, 1.0)


def apply_atom_decision(cont: Any, q_atom: Any,
                        config: DecodeConfig | None = None) -> tuple:
    """应用 WP1 的校准 + 期望分数决策。

    返回 ``(pred, used_decision)``：``used_decision=False`` 表示配置里没有 action_table，
    调用方应回退到 ``per_target_hard_switch``/``tau``。
    """
    _require_numpy()
    if config is None or not getattr(config, "action_table", None):
        return np.asarray(cont, dtype="float64"), False
    from . import atom_decision as AD
    q = calibrate_atom_probs(q_atom, getattr(config, "atom_calibration", None))
    pred, _actions = AD.apply_decision_table(cont, q, config.action_table)
    return sw_only_soft_clip(pred, config.sw_clip), True


def save_decode_config(cfg: DecodeConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def load_decode_config(path: str | Path) -> DecodeConfig:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return DecodeConfig(
        schema_version=int(d.get("schema_version", 1)),
        version=str(d.get("version", "decode_v1")),
        bias={k: float(v) for k, v in (d.get("bias") or {}).items()},
        shrink={k: float(v) for k, v in (d.get("shrink") or {}).items()},
        shrink_centers={k: float(v) for k, v in (d.get("shrink_centers") or {}).items()},
        quantile_shrink={k: float(v) for k, v in (d.get("quantile_shrink") or {}).items()},
        expected_value={k: bool(v) for k, v in (d.get("expected_value") or {}).items()},
        tau={k: float(v) for k, v in (d.get("tau") or {}).items()},
        atom_calibration=dict(d.get("atom_calibration") or {}),
        action_table=dict(d.get("action_table") or {}),
        sw_clip=tuple(d.get("sw_clip") or (0.0, 100.0)),
        inner_only=bool(d.get("inner_only", True)),
        selected_on=d.get("selected_on"), notes=str(d.get("notes", "")))


# ---------------------------------------------------------------- 基本算子
def apply_bias(pred: Any, bias: Mapping[str, float] | Sequence[float] | None,
               targets: Sequence[str] = C.TARGETS) -> "np.ndarray":
    """逐目标偏移：**POR/SW 加性、PERM 在 log10 空间加性**（即乘性）。

    为什么 PERM 必须走 log10：PERM 跨 6 个数量级、官方口径按 `log10` 比值计分，
    在标签尺度上做加法等于"大 PERM 修得多、小 PERM 几乎不动"，与官方口径不一致。
    """
    _require_numpy()
    p = np.asarray(pred, dtype="float64").copy()
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pred 必须为 (N,3)，got {p.shape}")
    if bias is None:
        return p
    if isinstance(bias, Mapping):
        vals = [float(bias.get(t, 0.0)) for t in targets]
    else:
        vals = [float(v) for v in bias]
    if len(vals) != p.shape[1]:
        raise ValueError(f"bias 长度 {len(vals)} != 目标数 {p.shape[1]}")
    p[:, 0] = p[:, 0] + vals[0]
    tiny = np.finfo("float64").tiny
    p[:, 1] = np.power(10.0, np.log10(np.clip(p[:, 1], tiny, None)) + vals[1])
    p[:, 2] = p[:, 2] + vals[2]
    return sw_only_soft_clip(p)


def apply_shrink(pred: Any, shrink: Mapping[str, float] | Sequence[float] | None,
                 centers: Mapping[str, float] | Sequence[float] | None = None,
                 targets: Sequence[str] = C.TARGETS) -> "np.ndarray":
    _require_numpy()
    p = np.asarray(pred, dtype="float64").copy()
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pred 必须为 (N,3)，got {p.shape}")
    if shrink is None:
        return p
    a = ([float(shrink.get(t, 1.0)) for t in targets] if isinstance(shrink, Mapping)
         else [float(v) for v in shrink])
    if centers is None:
        mu = np.median(p, axis=0)
    else:
        mu = np.asarray([float(centers.get(t, np.median(p[:, i])))
                         for i, t in enumerate(targets)] if isinstance(centers, Mapping)
                        else [float(v) for v in centers], dtype="float64")
    if len(a) != p.shape[1]:
        raise ValueError(f"shrink 长度 {len(a)} != 目标数 {p.shape[1]}")
    out = mu[None, :] + np.asarray(a, dtype="float64")[None, :] * (p - mu[None, :])
    return sw_only_soft_clip(out)


def quantile_shrink(pred: Any, alpha: Mapping[str, float] | Sequence[float] | None,
                    reference: Any, targets: Sequence[str] = C.TARGETS) -> "np.ndarray":
    """按 `reference`（OOF 上的目标列）的经验分位映射做单调收缩。

    `α=0` 恒等；`α=1` 变成"预测值的秩 → 参考分布的分位"的单调映射（保序）。
    """
    _require_numpy()
    p = np.asarray(pred, dtype="float64").copy()
    r = np.asarray(reference, dtype="float64")
    if r.ndim == 1:
        r = r[:, None]
    if p.shape[1] != r.shape[1]:
        raise ValueError(f"pred 列数 {p.shape[1]} != reference 列数 {r.shape[1]}")
    if alpha is None:
        return p
    a = ([float(alpha.get(t, 0.0)) for t in targets] if isinstance(alpha, Mapping)
         else [float(v) for v in alpha])
    out = p.copy()
    for i in range(p.shape[1]):
        ai = min(max(float(a[i]), 0.0), 1.0)
        if ai <= 0.0 or r[:, i].size == 0:
            continue
        q = np.quantile(r[:, i], np.linspace(0.0, 1.0, 101))
        u = np.clip((np.argsort(np.argsort(p[:, i], kind="mergesort")) + 0.5)
                    / max(p.shape[0], 1), 0.0, 1.0)
        mapped = np.interp(u, np.linspace(0.0, 1.0, 101), q)
        out[:, i] = (1.0 - ai) * p[:, i] + ai * mapped
    return sw_only_soft_clip(out)


def sw_only_soft_clip(pred: Any, clip: tuple[float, float] = (0.0, 100.0)) -> "np.ndarray":
    """只对 SW 做 [0,100] 软裁剪；POR 保持非负、PERM 严格为正（对数空间反变换保证）。"""
    _require_numpy()
    p = np.asarray(pred, dtype="float64").copy()
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pred 必须为 (N,3)，got {p.shape}")
    p[:, 2] = np.clip(p[:, 2], float(clip[0]), float(clip[1]))
    p[:, 0] = np.clip(p[:, 0], 0.0, None)
    p[:, 1] = np.clip(p[:, 1], np.finfo("float64").tiny, None)
    return p


def sw_scale_receipt(clip: tuple[float, float] = (0.0, 100.0)) -> dict[str, Any]:
    """SW 尺度收据（与 E5/P2 同一条纪律）。"""
    return {"global_clip_0_1": False, "multiply_100": False,
            "soft_clip_0_100": bool(tuple(map(float, clip)) == (0.0, 100.0)),
            "interpolation": False,
            "ok": bool(tuple(map(float, clip)) == (0.0, 100.0))}


# ---------------------------------------------------------------- 得分 / 分箱动作
def official_acc_columns(y: Any, pred: Any, mask: Any, targets: Sequence[str] = C.TARGETS
                         ) -> dict[str, float]:
    """逐目标官方命中率（缺测行排除；PERM 用截断口径）。"""
    _require_numpy()
    from ..score import acc_perm, acc_relative
    yy = np.asarray(y, dtype="float64")
    pp = np.asarray(pred, dtype="float64")
    mm = np.asarray(mask, dtype=bool)
    out: dict[str, float] = {}
    for i, t in enumerate(targets):
        keep = mm[:, i]
        if not keep.any():
            out[t] = float("nan")
        elif t == "PERM":
            out[t] = float(acc_perm(yy[keep, i], pp[keep, i]))
        else:
            out[t] = float(acc_relative(yy[keep, i], pp[keep, i], float(DELTA[t])))
    return out


def expected_value_table(y: Any, cont: Any, q_atom: Any, mask: Any, target_index: int,
                         n_bins: int = 10) -> dict[str, Any]:
    """按 `q_atom` 分箱比较"切原子 vs 保连续"的官方得分 → 逐箱动作表（期望值解码）。"""
    _require_numpy()
    t = C.TARGETS[int(target_index)]
    yy = np.asarray(y, dtype="float64")[:, int(target_index)]
    cc = np.asarray(cont, dtype="float64")[:, int(target_index)]
    qq = np.asarray(q_atom, dtype="float64")[:, int(target_index)]
    mm = np.asarray(mask, dtype=bool)[:, int(target_index)]
    atom = float(C.ATOM_VALUES[t])
    atom_pred = np.full_like(cc, atom)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, Any]] = []
    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        sel = mm & (qq >= lo) & (qq < hi if b < int(n_bins) - 1 else qq <= hi)
        if not sel.any():
            rows.append({"bin": b, "lo": lo, "hi": hi, "n_rows": 0, "action": "continuous",
                         "score_atom": None, "score_cont": None, "gain_atom": None})
            continue
        s_atom = official_acc_columns(y[sel][:, [int(target_index)]],
                                      atom_pred[sel][:, None], mm[sel][:, None],
                                      (t,))[t]
        s_cont = official_acc_columns(y[sel][:, [int(target_index)]],
                                      cc[sel][:, None], mm[sel][:, None], (t,))[t]
        action = "atom" if (s_atom is not None and s_cont is not None and s_atom > s_cont) \
            else "continuous"
        rows.append({"bin": b, "lo": lo, "hi": hi, "n_rows": int(sel.sum()),
                     "action": action, "score_atom": float(s_atom),
                     "score_cont": float(s_cont), "gain_atom": float(s_atom - s_cont)})
    # 单调化：可部署的 τ 规则要求原子动作是**高 q 侧的后缀**，且不能从 0 号箱开始
    # （否则 tau=0.0，sigmoid 输出 q>0 恒真，会把整个目标强制写成原子值）。
    atoms = [r["bin"] for r in rows if r["action"] == "atom"]
    n_b = len(rows)
    suffix_ok = bool(atoms) and atoms == list(range(atoms[0], n_b)) and atoms[0] > 0
    monotone = bool(not atoms or suffix_ok)
    tau = None
    if suffix_ok:
        tau = float(rows[atoms[0]]["lo"])
    return {"target": t, "n_bins": int(n_bins), "bins": rows, "monotone": monotone,
            "suffix_start": (int(atoms[0]) if atoms else None),
            "tau_from_table": tau,
            "note": ("动作表必须单调（高置信才切原子）且原子箱是高 q 侧后缀、起点 >0，"
                     "才能压成一个 τ；否则只作报告，部署仍用 τ 网格搜索/期望动作表。")}


def sensitivity_report(y: Any, pred: Any, mask: Any, target_index: int,
                       grid: tuple[float, float, int] | None = None,
                       tol: float = DEFAULT_SENSITIVITY_TOL) -> dict[str, Any]:
    """一维敏感性：扫描该目标的整体偏移，找"分数变化 ≤ tol"的**平坦区中点**。"""
    _require_numpy()
    t = C.TARGETS[int(target_index)]
    lo, hi, n = grid or DEFAULT_BIAS_GRID[t]
    vals = np.linspace(float(lo), float(hi), int(n))
    base = official_acc_columns(y, pred, mask)[t]
    rows = []
    for b in vals:
        shifted = pred.copy()
        shifted[:, int(target_index)] = shifted[:, int(target_index)] + float(b)
        shifted = sw_only_soft_clip(shifted)
        rows.append({"bias": float(b), "acc": official_acc_columns(y, shifted, mask)[t]})
    ok = [r for r in rows if abs(r["acc"] - base) <= float(tol)]
    mid = (float(np.mean([r["bias"] for r in ok])) if ok else 0.0)
    best = max(rows, key=lambda r: r["acc"])
    return {"target": t, "base_acc": float(base), "tol": float(tol), "curve": rows,
            "flat_region": ([min(r["bias"] for r in ok), max(r["bias"] for r in ok)]
                            if ok else None),
            "flat_midpoint": float(mid), "flat_hit": bool(ok),
            "argmax_bias": float(best["bias"]), "argmax_acc": float(best["acc"]),
            "argmax_gain": float(best["acc"] - base),
            "recommended": (float(mid) if ok and (abs(best["bias"])
                                                  <= max(abs(min(r["bias"] for r in ok)),
                                                         abs(max(r["bias"] for r in ok)))
                                                  + 1e-12) else float(best["bias"])),
            "note": ("推荐值 = 平坦区中点（若最优值落在平坦区内），否则用最优格点；"
                     "整段都平坦说明该目标的偏移对分数无影响，宁可不调")}


def assert_atom_priority(cont_after: Any, cont_before: Any, q_atom: Any, tau,
                         out_actual: Any | None = None) -> dict[str, Any]:
    """原子优先级证据：**先连续后处理、后原子硬切换**时，原子行必须逐位等于原子值。

    审计修复（P2）：
    - 传入 ``out_actual``（实际管线输出）时，断言：
      1. 命中原子行（q > tau）的实际输出必须**逐位等于原子值**；
      2. 未命中行的实际输出必须**逐位等于后处理后的连续值 cont_after**；
      两者任一不满足 -> ok=False，能真正发现“先切原子再后处理”的顺序错误。
    - 未传 ``out_actual`` 时保留旧的诊断路径（只比较 cont_before/cont_after 的硬切换），
      但新调用方都应传 ``out_actual``。
    """
    _require_numpy()
    from ..inference.atomic_gate import per_target_hard_switch
    q = np.asarray(q_atom, dtype="float64")
    before = per_target_hard_switch(np.asarray(cont_before, dtype="float64"), q, tau)
    after = per_target_hard_switch(np.asarray(cont_after, dtype="float64"), q, tau)
    taus = np.asarray(tau, dtype="float64").reshape(-1)
    if taus.size == 1:
        taus = np.repeat(taus, 3)
    else:
        taus = np.resize(taus, 3)
    hit = q > taus[None, :]
    changed = int(np.sum((np.asarray(after) != np.asarray(before)) & hit))
    out = {"n_atom_rows": int(hit.sum()), "n_changed_on_atom_rows": changed,
           "ok": bool(changed == 0), "tau": [float(v) for v in taus],
           "note": "正确顺序：连续后处理 → 再做原子硬切换；命中行必须逐位等于原子值"}
    if out_actual is not None:
        actual = np.asarray(out_actual, dtype="float64")
        if actual.shape != q.shape:
            return {**out, "ok": False, "error": f"out_actual 形状 {actual.shape} != {q.shape}"}
        atom_vals = np.asarray([C.ATOM_VALUES[t] for t in C.TARGETS], dtype="float64")
        wrong_atom = int(np.sum((~np.isclose(actual, atom_vals[None, :], rtol=1e-9,
                                              atol=1e-9)) & hit))
        wrong_cont = int(np.sum((~np.isclose(actual, np.asarray(cont_after, dtype="float64"),
                                              rtol=1e-9, atol=1e-9)) & (~hit)))
        out.update({"n_wrong_on_atom_rows": wrong_atom, "n_wrong_on_cont_rows": wrong_cont,
                    "ok": bool(changed == 0 and wrong_atom == 0 and wrong_cont == 0),
                    "note": ("实际输出 on atom rows 必须等于原子值，on continuous rows 必须"
                             "等于 cont_after；否则顺序写反或后处理污染了原子行。")})
    return out
