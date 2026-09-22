"""WP10：扩展岩石物理特征（参考 SPWLA 2021 第 4 名 Atwah 的特征工程）。

包含：
  * 多骨架密度孔隙度（砂岩/灰岩/白云岩）；
  * Vsh：GR 线性、Larionov(old/tertiary)、Steiber、Clavier；
  * Sw：Archie、Simandoux、Indonesia；
  * Klogh（按地层系数 ``10^(a+b·φ+c·Vsh)``）；
  * 中子-密度组合、声波-密度交会等。

所有特征只使用输入曲线；参数（sand/shale 基线、Rw/Rsh）必须由训练折拟合
（``fit_petro_params``），推理期从 manifest 读回。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np

from .. import constants as C

# 骨架/流体密度（g/cm³）
RHO_MATRIX = {"sandstone": 2.65, "limestone": 2.71, "dolomite": 2.87, "shale": 2.45}
RHO_FLUID = 1.0
DEFAULT_KLOGH = {  # 参考 Atwah 的三套经验系数（a, b, c）
    "hugin": (2.0, 8.0, -9.0),
    "sleipner": (-3.0, 32.0, -2.0),
    "skagerak": (-1.85, 17.4, -3.0),
}

PETRO_FEATURES: tuple[str, ...] = (
    "phi_den_ss", "phi_den_ls", "phi_den_dol", "phi_den_neu_ss",
    "vsh_gr_ext", "vsh_larionov_old", "vsh_larionov_tert",
    "vsh_steiber", "vsh_clavier",
    "sw_archie", "sw_simandoux", "sw_indonesia",
    "klogh_hugin", "klogh_sleipner", "klogh_skagerak",
    "log10_rt_ext", "log10_rxo_ext", "rt_rxo_ratio_ext",
)


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def _clip01(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0.0, 1.0)


def phi_density(den, rho_ma: float = 2.65, rho_f: float = 1.0) -> np.ndarray:
    return _clip01((float(rho_ma) - _f(den)) / (float(rho_ma) - float(rho_f)))


def phi_density_matrix(den, matrix: str = "sandstone") -> np.ndarray:
    return phi_density(den, RHO_MATRIX.get(str(matrix), 2.65), RHO_FLUID)


def vsh_gr(gr, sand: float, shale: float) -> np.ndarray:
    denom = float(shale) - float(sand)
    if abs(denom) < 1e-12:
        return np.zeros_like(_f(gr))
    return _clip01((_f(gr) - float(sand)) / denom)


def vsh_larionov_old(gr, sand: float, shale: float) -> np.ndarray:
    igr = vsh_gr(gr, sand, shale)
    return _clip01(0.33 * (2.0 ** (2.0 * igr) - 1.0))


def vsh_larionov_tert(gr, sand: float, shale: float) -> np.ndarray:
    igr = vsh_gr(gr, sand, shale)
    return _clip01(0.083 * (2.0 ** (3.7 * igr) - 1.0))


def vsh_steiber(gr, sand: float, shale: float) -> np.ndarray:
    igr = vsh_gr(gr, sand, shale)
    return _clip01(igr / np.maximum(3.0 - 2.0 * igr, 1e-9))


def vsh_clavier(gr, sand: float, shale: float) -> np.ndarray:
    igr = vsh_gr(gr, sand, shale)
    inner = 3.38 - (igr + 0.7) ** 2.0
    out = 1.7 - np.sqrt(np.clip(inner, 0.0, None))
    out = np.where(np.isfinite(out), out, 1.0)
    return _clip01(out)


def sw_archie(rt, phi, rw: float = 0.05, a: float = 1.0, m: float = 2.0,
              n: float = 2.0) -> np.ndarray:
    phi = np.clip(_f(phi), 1e-4, None)
    rt = np.clip(_f(rt), 1e-6, None)
    sw = (float(a) * float(rw) / (phi ** float(m) * rt)) ** (1.0 / float(n))
    return _clip01(sw)


def sw_simandoux(rt, phi, vsh, rw: float = 0.05, rsh: float = 5.0,
                 a: float = 1.0, m: float = 2.0) -> np.ndarray:
    phi = np.clip(_f(phi), 1e-4, None)
    rt = np.clip(_f(rt), 1e-6, None)
    vsh = _clip01(_f(vsh))
    A = phi ** float(m) / (float(a) * float(rw))
    B = vsh / max(float(rsh), 1e-6)
    C = -1.0 / rt
    disc = np.clip(B * B - 4.0 * A * C, 0.0, None)
    sw = (-B + np.sqrt(disc)) / (2.0 * A)
    return _clip01(np.where(np.isfinite(sw), sw, 1.0))


def sw_indonesia(rt, phi, vsh, rw: float = 0.05, rsh: float = 5.0,
                 a: float = 1.0, m: float = 2.0) -> np.ndarray:
    phi = np.clip(_f(phi), 1e-4, None)
    rt = np.clip(_f(rt), 1e-6, None)
    vsh = _clip01(_f(vsh))
    A = (1.0 / rt) ** 0.5
    B = (vsh ** (1.0 - 0.5 * vsh)) / max(float(rsh), 1e-6) ** 0.5
    C = ((phi ** float(m)) / (float(a) * float(rw))) ** 0.5
    sw = A / np.maximum(B + C, 1e-9)
    return _clip01(np.where(np.isfinite(sw), sw, 1.0))


def klogh(phi, vsh, coeff: tuple[float, float, float]) -> np.ndarray:
    a, b, c = [float(v) for v in coeff]
    return 10.0 ** (a + b * _f(phi) + c * _f(vsh))


def _curve_index(name: str) -> int:
    return C.INPUT_COLUMNS.index(name)


@dataclass
class PetroParams:
    sand_line: float = 20.0
    shale_line: float = 120.0
    rw: float = 0.05
    rsh: float = 5.0
    fitted_on: str = "train_fold_only"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PetroParams":
        return PetroParams(sand_line=float(d.get("sand_line", 20.0)),
                           shale_line=float(d.get("shale_line", 120.0)),
                           rw=float(d.get("rw", 0.05)),
                           rsh=float(d.get("rsh", 5.0)),
                           fitted_on=str(d.get("fitted_on", "train_fold_only")))

    def to_vector(self) -> np.ndarray:
        return np.asarray([self.sand_line, self.shale_line, self.rw, self.rsh],
                          dtype="float64")


def fit_petro_params(inputs: Any, missing: Any | None = None) -> PetroParams:
    """只用训练折输入拟合 sand/shale 基线与 Rw/Rsh 的稳健默认值。"""
    X = np.asarray(inputs, dtype="float64")
    m = None if missing is None else np.asarray(missing, dtype=bool)
    gr = X[:, _curve_index("GR")]
    rt = X[:, _curve_index("RT")]
    if m is not None:
        gr = gr[~m[:, _curve_index("GR")]]
        rt = rt[~m[:, _curve_index("RT")]]
    gr = gr[np.isfinite(gr)]
    rt = rt[np.isfinite(rt) & (rt > 0)]
    sand = float(np.percentile(gr, 5)) if gr.size else 20.0
    shale = float(np.percentile(gr, 95)) if gr.size else 120.0
    rw = float(np.percentile(rt, 10)) if rt.size else 0.05
    rsh = float(np.percentile(rt, 90)) if rt.size else 5.0
    return PetroParams(sand_line=sand, shale_line=shale,
                       rw=max(rw, 1e-4), rsh=max(rsh, 1e-4))


def build_petro_features(inputs: Any, missing: Any | None = None,
                         params: PetroParams | None = None
                         ) -> tuple[np.ndarray, list[str]]:
    """构造扩展岩石物理特征矩阵；缺测处输出 NaN（由缺失位/插补处理）。"""
    X = np.asarray(inputs, dtype="float64")
    if X.ndim != 2 or X.shape[1] != len(C.INPUT_COLUMNS):
        raise ValueError(f"inputs 必须 (n,{len(C.INPUT_COLUMNS)})，got {X.shape}")
    p = params or fit_petro_params(X, missing)
    gr = X[:, _curve_index("GR")]
    den = X[:, _curve_index("DEN")]
    cnl = X[:, _curve_index("CNL")]
    ac = X[:, _curve_index("AC")]
    rt = X[:, _curve_index("RT")]
    rxo = X[:, _curve_index("RXO")]

    phi_ss = phi_density_matrix(den, "sandstone")
    phi_ls = phi_density_matrix(den, "limestone")
    phi_dol = phi_density_matrix(den, "dolomite")
    phi_neu = np.clip(cnl / 100.0, 0.0, 1.0)
    phi_den_neu_ss = np.clip(0.5 * (phi_ss + phi_neu), 0.0, 1.5)
    vsh_lin = vsh_gr(gr, p.sand_line, p.shale_line)
    vsh_lo = vsh_larionov_old(gr, p.sand_line, p.shale_line)
    vsh_lt = vsh_larionov_tert(gr, p.sand_line, p.shale_line)
    vsh_st = vsh_steiber(gr, p.sand_line, p.shale_line)
    vsh_cl = vsh_clavier(gr, p.sand_line, p.shale_line)
    sw_a = sw_archie(rt, np.where(np.isfinite(phi_ss), phi_ss, np.nan), p.rw)
    sw_s = sw_simandoux(rt, np.where(np.isfinite(phi_ss), phi_ss, np.nan), vsh_lin, p.rw, p.rsh)
    sw_i = sw_indonesia(rt, np.where(np.isfinite(phi_ss), phi_ss, np.nan), vsh_lin, p.rw, p.rsh)
    k_h = klogh(np.where(np.isfinite(phi_ss), phi_ss, np.nan), vsh_lin, DEFAULT_KLOGH["hugin"])
    k_s = klogh(np.where(np.isfinite(phi_ss), phi_ss, np.nan), vsh_lin, DEFAULT_KLOGH["sleipner"])
    k_k = klogh(np.where(np.isfinite(phi_ss), phi_ss, np.nan), vsh_lin, DEFAULT_KLOGH["skagerak"])
    log_rt = np.log10(np.clip(rt, 1e-6, None))
    log_rxo = np.log10(np.clip(rxo, 1e-6, None))
    ratio = np.clip(rxo / np.clip(rt, 1e-6, None), 0.0, 1e4)

    cols = [phi_ss, phi_ls, phi_dol, phi_den_neu_ss,
            vsh_lin, vsh_lo, vsh_lt, vsh_st, vsh_cl,
            sw_a, sw_s, sw_i, k_h, k_s, k_k,
            log_rt, log_rxo, ratio]
    out = np.stack(cols, axis=1).astype("float32")
    # 输入缺测 -> 输出 NaN（保持缺失传播纪律）
    if missing is not None:
        m = np.asarray(missing, dtype=bool)
        if m.shape == X.shape:
            # 该行只要有任一根底曲线缺失，对应派生值置 NaN（保守）
            key_ok = ~(m[:, _curve_index("GR")] | m[:, _curve_index("DEN")]
                       | m[:, _curve_index("CNL")] | m[:, _curve_index("RT")])
            out[~key_ok] = np.nan
    return out, list(PETRO_FEATURES)
