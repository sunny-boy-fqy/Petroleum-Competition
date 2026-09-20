"""F_phys：岩石物理与交会特征（E2/P0）。

设计纪律（E2/P0 §7/§8）
-----------------------
1. **纯函数**：每个派生列一个函数，公式与出处写在 docstring 里；不 import torch。
2. **不碰标签**：只允许用 13 条曲线 + DEPTH 派生，任何函数都不得接收 POR/PERM/SW。
3. **缺失传播**：输入缺测 → 输出 NaN（**绝不填 0**），并同时给出 0/1 指示位。
4. **数值保护**：分母夹取、log 前夹正、结果做**物理区间检查**并把越界比例写进报告
   （E2/P0 §9 的"单位换算错误"风险信号）。
5. **折内参数**：`GRmin/GRmax`、`SPmin/SPmax` 这类分位数参数**只能由训练折拟合**
   （`fit_physics_params`），推理期从 manifest 读回 —— 与 `RowScaler`/目标尺度同一纪律。

单位（必须与数据一致）
----------------------
数据里 `AC` 是 **μs/m**（实测 ~240），而文献常数常给 μs/ft：
    ACma = 55.5 μs/ft × 3.28084 ≈ **182.1 μs/m**（砂岩骨架）
    ACf  = 189  μs/ft × 3.28084 ≈ **620.1 μs/m**（流体）
`DEN` 是 g/cm³（ρma=2.65 石英、ρf=1.0 水）；`CNL` 是**百分数**（÷100 得小数孔隙度）。
E2/P0 §9 明确警告单位换算错误，因此每个派生列都做区间检查。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .. import constants as C

# ---------------------------------------------------------------- 常数
FT_TO_M = 3.28084
ACMA_US_M = 55.5 * FT_TO_M          # ≈182.09 μs/m
ACF_US_M = 189.0 * FT_TO_M          # ≈620.08 μs/m
RHO_MA = 2.65                        # g/cm³ 石英骨架
RHO_F = 1.0                          # g/cm³ 水
PE_QUARTZ = 1.81                     # 石英光石电吸收指数
GR_MIN_Q = 5.0                       # 折内分位数（下）
GR_MAX_Q = 95.0                      # 折内分位数（上）

# 每列的物理合理区间（越界比例会被记录，用于抓单位错误）
SANE_RANGE: dict[str, tuple[float, float]] = {
    "phi_sonic": (-0.5, 1.5),
    "phi_den": (-0.5, 1.5),
    "phi_neu": (0.0, 1.5),
    "phi_den_neu_avg": (-0.5, 1.5),
    "phi_eff": (-0.5, 1.5),
    "igr": (0.0, 1.0),
    "vsh_gr": (0.0, 1.0),
    "sp_index": (0.0, 1.0),
    "rt_rxo_ratio": (0.0, 1e4),
    "log10_rt": (-3.0, 5.0),
    "log10_rxo": (-3.0, 5.0),
    "log10_rt_rxo": (-5.0, 5.0),
    "den_cnl_cross": (-1.0, 1.0),
    "ac_den_cross": (-5.0, 5.0),
    "pe_quartz_diff": (-5.0, 20.0),
    "shale_corrected_por": (-0.5, 1.5),
}

# 13 条曲线在 F1 特征里的列号（`features.basic` 的布局：0..12 = GR..CASE）
_CURVE_INDEX = {name: i for i, name in enumerate(C.INPUT_COLUMNS)}


@dataclass
class PhysicsParams:
    """**训练折**拟合的物理特征参数（分位数基线）。"""
    gr_min: float = 20.0
    gr_max: float = 120.0
    sp_min: float = 20.0
    sp_max: float = 100.0
    fitted_on: str = "train_fold_only"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PhysicsParams":
        return PhysicsParams(gr_min=float(d["gr_min"]), gr_max=float(d["gr_max"]),
                             sp_min=float(d["sp_min"]), sp_max=float(d["sp_max"]),
                             fitted_on=str(d.get("fitted_on", "train_fold_only")))

    def to_vector(self) -> "np.ndarray":
        return np.asarray([self.gr_min, self.gr_max, self.sp_min, self.sp_max],
                          dtype="float64")


PHYSICS_FEATURES: tuple[str, ...] = (
    "phi_sonic", "phi_den", "phi_neu", "phi_den_neu_avg", "phi_eff",
    "igr", "vsh_gr", "sp_index", "rt_rxo_ratio", "log10_rt", "log10_rxo",
    "log10_rt_rxo", "den_cnl_cross", "ac_den_cross", "pe_quartz_diff",
    "shale_corrected_por",
)

# 逐列（列名 → 公式, 依据）—— 写入 `E2_feature_provenance.csv`（E2/P0 §7 要求全覆盖）
PHYSICS_PROVENANCE: dict[str, tuple[str, str]] = {
    "phi_sonic": ("Wyllie: (AC − ACma)/(ACf − ACma)，ACma=55.5、ACf=189 μs/ft 换算为 μs/m",
                  "资料库/01 §2（声波孔隙度）"),
    "phi_den": ("(ρma − DEN)/(ρma − ρf)，ρma=2.65、ρf=1.0 g/cm³",
                "资料库/01 §3（密度孔隙度）"),
    "phi_neu": ("CNL/100（CNL 为百分数）", "资料库/01 §3（中子孔隙度）"),
    "phi_den_neu_avg": ("(φD + φN)/2", "资料库/01 §3（密度-中子平均）"),
    "phi_eff": ("φN < φD 时 sqrt((φN²+φD²)/2)，否则 (φN+φD)/2（气层校正）",
                "资料库/01 §3；资料库/16（交会图版）"),
    "igr": ("clip((GR − GRmin)/(GRmax − GRmin), 0, 1)；GRmin/GRmax = 训练折 P5/P95",
            "资料库/01 §4（GR 指数）；折内 fit 纪律见 E2/P0 §6"),
    "vsh_gr": ("Vsh = IGR（线性 GR 法）", "资料库/01 §4（泥质含量）"),
    "sp_index": ("clip((SP − SPmin)/(SPmax − SPmin), 0, 1)；分位数只在训练折拟合",
                 "资料库/01 §4（SP 指数）"),
    "rt_rxo_ratio": ("clip(RXO/RT, 0, 1e4)（冲洗带/原状地层）", "资料库/02（电阻率交会）"),
    "log10_rt": ("log10(max(RT, 1e-3))", "资料库/02（对数电阻率）"),
    "log10_rxo": ("log10(max(RXO, 1e-3))", "资料库/02（对数电阻率）"),
    "log10_rt_rxo": ("log10(RT) − log10(RXO)", "资料库/02（电阻率比值）"),
    "den_cnl_cross": ("φN − φD（密度-中子交会差）", "资料库/16（交会图版）"),
    "ac_den_cross": ("(AC − 240)/100 − (DEN − 2.35)", "资料库/16（AC-DEN 交会）"),
    "pe_quartz_diff": ("PE − 1.81（相对石英骨架的偏离）", "资料库/01 §5（骨架矿物）"),
    "shale_corrected_por": ("φD·(1 − Vsh)", "资料库/01 §4（泥质校正孔隙度）"),
}


def _f(x) -> "np.ndarray":
    return np.asarray(x, dtype="float64")


def _safe_div(num, den, lo: float = 1e-9):
    """分母夹到 `>= lo`（保持符号语义：这里所有分母在物理上都是正的）。"""
    d = np.where(np.abs(den) < lo, np.nan, den)
    return num / d


def phi_sonic(ac, acma: float = ACMA_US_M, acf: float = ACF_US_M):
    """Wyllie 声波孔隙度 `φS = (AC − ACma)/(ACf − ACma)`（`资料库/01` §2）。"""
    return _safe_div(_f(ac) - acma, acf - acma)


def phi_den(den, rho_ma: float = RHO_MA, rho_f: float = RHO_F):
    """密度孔隙度 `φD = (ρma − DEN)/(ρma − ρf)`（`资料库/01` §3）。"""
    return _safe_div(rho_ma - _f(den), rho_ma - rho_f)


def phi_neu(cnl):
    """中子孔隙度：`CNL` 是百分数 → 小数（`资料库/01` §3）。"""
    return _f(cnl) / 100.0


def phi_den_neu_avg(pd_, pn):
    """密度-中子平均孔隙度（气层/泥质页岩的常用折中）。"""
    return 0.5 * (_f(pd_) + _f(pn))


def phi_eff(pd_, pn):
    """气层校正有效孔隙度（`资料库/01` §3）：φN < φD 时用 RMS 组合，否则取平均。

    `φe = sqrt((φN² + φD²)/2)`（含气，中子被低估） vs `(φN+φD)/2`（一般情形）。
    与 `phi_den_neu_avg` **不是同一列**（后者恒为算术平均），因此两者可以独立消融。
    """
    a = _f(pd_)
    b = _f(pn)
    rms = np.sqrt((a * a + b * b) / 2.0)
    return np.where(b < a, rms, 0.5 * (a + b))


def igr(gr, gr_min: float, gr_max: float):
    """GR 指数 `IGR = (GR − GRmin)/(GRmax − GRmin)`，分位数**只在训练折**拟合。"""
    return np.clip(_safe_div(_f(gr) - gr_min, max(gr_max - gr_min, 1e-9)), 0.0, 1.0)


def vsh_gr(igr_):
    """泥质含量（线性 GR 法）`Vsh = IGR`（`资料库/01` §4）。"""
    return np.clip(_f(igr_), 0.0, 1.0)


def sp_index(sp, sp_min: float, sp_max: float):
    """SP 归一化指数 `(SP − SPmin)/(SPmax − SPmin)`（折内分位数）。"""
    return np.clip(_safe_div(_f(sp) - sp_min, max(sp_max - sp_min, 1e-9)), 0.0, 1.0)


def rt_rxo_ratio(rt, rxo):
    """冲洗带/原状地层电阻率比 `RXO/RT`（流体可动性指示，`资料库/02`）。"""
    return np.clip(_safe_div(_f(rxo), np.maximum(_f(rt), 1e-6)), 0.0, 1e4)


def log10_rt(rt):
    return np.log10(np.maximum(_f(rt), 1e-3))


def log10_rxo(rxo):
    return np.log10(np.maximum(_f(rxo), 1e-3))


def log10_rt_rxo(rt, rxo):
    """`log10(RT/RXO)`：与 `log10_rt − log10_rxo` 同值，单独成列便于线性模型使用。"""
    return log10_rt(rt) - log10_rxo(rxo)


def den_cnl_cross(pd_, pn):
    """密度-中子交会差 `φN − φD`（正值=气/轻质，负值=泥质/重矿物）。"""
    return _f(pn) - _f(pd_)


def ac_den_cross(ac, den, ac_ref: float = 240.0, den_ref: float = 2.35):
    """AC-DEN 交会偏离量（相对参考点的标准化差，无量纲）。"""
    return _safe_div(_f(ac) - ac_ref, 100.0) - (_f(den) - den_ref)


def pe_quartz_diff(pe, pe_q: float = PE_QUARTZ):
    """光石电吸收相对石英的偏离 `PE − PE_quartz`（骨架矿物指示）。"""
    return _f(pe) - pe_q


def shale_corrected_por(pd_, vsh_):
    """泥质校正孔隙度 `φe ≈ φD·(1 − Vsh)`（`资料库/01` §4）。"""
    return _f(pd_) * (1.0 - np.clip(_f(vsh_), 0.0, 1.0))


def fit_physics_params(inputs: "np.ndarray", gr_q: float = GR_MIN_Q,
                       gr_max_q: float = GR_MAX_Q) -> PhysicsParams:
    """**只用训练折行**拟合分位数基线（缺测行不参与）。

    `inputs` 是 (n,13) 曲线矩阵（`C.INPUT_COLUMNS` 顺序）。
    """
    x = np.asarray(inputs, dtype="float64")
    gi = _CURVE_INDEX["GR"]
    si = _CURVE_INDEX["SP"]
    gr = x[:, gi]
    sp = x[:, si]
    gr = gr[np.isfinite(gr)]
    sp = sp[np.isfinite(sp)]
    if gr.size == 0 or sp.size == 0:
        raise ValueError("fit_physics_params: 训练折里没有有效的 GR/SP 行")
    return PhysicsParams(
        gr_min=float(np.percentile(gr, gr_q)),
        gr_max=float(np.percentile(gr, gr_max_q)),
        sp_min=float(np.percentile(sp, gr_q)),
        sp_max=float(np.percentile(sp, gr_max_q)),
    )


def build_physics_features(inputs: "np.ndarray", params: PhysicsParams,
                           with_indicators: bool = True) -> tuple["np.ndarray", list[str]]:
    """由 (n,13) 曲线矩阵构造 F_phys。

    返回 `(X, names)`；`with_indicators=True` 时每个派生列附带 `<col>_ok` 指示位
    （E2/P0 §5 步 3 要求"缺失输入传播为缺失输出，并同步生成缺失指示位"）。
    """
    x = np.asarray(inputs, dtype="float64")
    if x.ndim != 2 or x.shape[1] != len(C.INPUT_COLUMNS):
        raise ValueError(f"build_physics_features expects (n,{len(C.INPUT_COLUMNS)}), "
                         f"got {tuple(x.shape)}")

    def col(name: str) -> "np.ndarray":
        return x[:, _CURVE_INDEX[name]]

    ps = phi_sonic(col("AC"))
    pd_ = phi_den(col("DEN"))
    pn = phi_neu(col("CNL"))
    avg = phi_den_neu_avg(pd_, pn)
    pe_ = phi_eff(pd_, pn)
    ig = igr(col("GR"), params.gr_min, params.gr_max)
    vsh = vsh_gr(ig)
    spi = sp_index(col("SP"), params.sp_min, params.sp_max)
    ratio = rt_rxo_ratio(col("RT"), col("RXO"))
    lrt = log10_rt(col("RT"))
    lrxo = log10_rxo(col("RXO"))
    lrr = log10_rt_rxo(col("RT"), col("RXO"))
    dc = den_cnl_cross(pd_, pn)
    adc = ac_den_cross(col("AC"), col("DEN"))
    peq = pe_quartz_diff(col("PE"))
    scp = shale_corrected_por(pd_, vsh)

    values = {"phi_sonic": ps, "phi_den": pd_, "phi_neu": pn, "phi_den_neu_avg": avg,
              "phi_eff": pe_, "igr": ig, "vsh_gr": vsh, "sp_index": spi,
              "rt_rxo_ratio": ratio, "log10_rt": lrt, "log10_rxo": lrxo,
              "log10_rt_rxo": lrr, "den_cnl_cross": dc, "ac_den_cross": adc,
              "pe_quartz_diff": peq, "shale_corrected_por": scp}

    names = list(PHYSICS_FEATURES)
    cols = [values[n] for n in PHYSICS_FEATURES]
    if with_indicators:
        names += [f"{n}_ok" for n in PHYSICS_FEATURES]
        cols += [np.isfinite(values[n]).astype("float64") for n in PHYSICS_FEATURES]
    X = np.stack([np.asarray(c, dtype="float64").reshape(-1) for c in cols], axis=1)
    return X.astype("float32"), names


def out_of_range_report(X: "np.ndarray", names: list[str]) -> dict[str, Any]:
    """逐列物理区间越界比例（E2/P0 §9 的单位错误探测器）。"""
    x = np.asarray(X, dtype="float64")
    rep: dict[str, Any] = {}
    for j, name in enumerate(names):
        base = name[:-3] if name.endswith("_ok") else name
        lo, hi = SANE_RANGE.get(base, (float("-inf"), float("inf")))
        col = x[:, j]
        fin = np.isfinite(col)
        n = int(fin.sum())
        if n == 0:
            rep[name] = {"n_valid": 0, "frac_out_of_range": None}
            continue
        bad = int((~((col[fin] >= lo) & (col[fin] <= hi))).sum())
        rep[name] = {"n_valid": n, "frac_out_of_range": bad / n,
                     "range": [lo, hi], "nan_frac": float(1.0 - n / col.size)}
    return rep


def label_independence_audit() -> dict[str, Any]:
    """静态审计：本模块的函数签名里不得出现目标名（E2/P0 §8 的硬禁令）。"""
    import inspect
    banned = {t.lower() for t in C.TARGETS} | {"y_por", "y_perm", "y_sw", "targets"}
    offenders: list[str] = []
    for name, fn in sorted(globals().items()):
        if not inspect.isfunction(fn) or fn.__module__ != __name__:
            continue
        for pname in inspect.signature(fn).parameters:
            if pname.lower() in banned:
                offenders.append(f"{name}({pname})")
    return {"ok": not offenders, "offenders": offenders,
            "rule": "F_phys 只允许用 13 条曲线派生，禁止任何目标值参与"}
