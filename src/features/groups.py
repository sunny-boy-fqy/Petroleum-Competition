"""特征组注册表：把 F1 + F_phys + F_win + F_well 拼成**可独立消融**的特征矩阵（E2）。

为什么需要它
------------
E2 的 Gate 要求"三组特征各自在行级 MLP 上有消融结果"（E2 §7），E3 又要在同一套数据上换主干。
因此"特征 = 哪些组的并集"必须是**一个显式、可序列化、可复现**的对象，而不是散在脚本里的
列切片 —— 否则无法回答"这次 OOF 用的是哪 364 列"。

布局（列序固定，任何新增组只能追加）
------------------------------------
    F1    : 32 列（13 曲线 + DEPTH + 14 缺失位 + 4 深度编码；`features.basic`）
    phys  : 16 派生 + 16 指示 = 32 列（`features.physics`）
    win   : 13 曲线 × 窗长{11,51,201} × 6 统计 = 234 列（`features.window`）
    well  : 13 曲线 × 5 统计 + 5 标量 = 70 列（`features.well`）

缓存布局：`$V4_CACHE_ROOT/feat/<spec_key>/<split>/<well>.npz`（原子写）。
`spec_key` 由组集合与窗口/统计配置共同决定，**换配置就是换目录**，不会读到旧列。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .. import constants as C
from . import basic as F
from . import physics as PH
from . import well as WL
from . import window as WN

GROUP_ORDER: tuple[str, ...] = ("F1", "phys", "win", "well")
DEFAULT_GROUPS: tuple[str, ...] = ("F1",)
F2_GROUPS: tuple[str, ...] = GROUP_ORDER

# 每组列数（用于报告与断言）
GROUP_FEATURE_NAMES: dict[str, tuple[str, ...]] = {}


def _f1_names() -> tuple[str, ...]:
    return tuple(F.FEATURE_NAMES)


def _phys_names(with_indicators: bool = True) -> tuple[str, ...]:
    names = list(PH.PHYSICS_FEATURES)
    if with_indicators:
        names += [f"{n}_ok" for n in PH.PHYSICS_FEATURES]
    return tuple(names)


def _win_names(windows: Sequence[int] = WN.DEFAULT_WINDOWS,
               stats: Sequence[str] = WN.DEFAULT_STATS) -> tuple[str, ...]:
    return tuple(WN.window_names(windows=windows, stats=stats))


def _well_names(stats: Sequence[str] = WL.WELL_STATS,
                scalars: Sequence[str] = WL.WELL_SCALARS) -> tuple[str, ...]:
    return tuple(WL.well_names(stats=stats, scalars=scalars))


@dataclass(frozen=True)
class FeatureSpec:
    """一次训练/推理用的特征组配置（**可 JSON 化**，写进 manifest 与报告）。"""
    groups: tuple[str, ...] = DEFAULT_GROUPS
    windows: tuple[int, ...] = WN.DEFAULT_WINDOWS
    win_stats: tuple[str, ...] = WN.DEFAULT_STATS
    well_stats: tuple[str, ...] = WL.WELL_STATS
    well_scalars: tuple[str, ...] = WL.WELL_SCALARS
    phys_indicators: bool = True

    def __post_init__(self) -> None:
        unknown = [g for g in self.groups if g not in GROUP_ORDER]
        if unknown:
            raise ValueError(f"unknown feature groups: {unknown}; known={list(GROUP_ORDER)}")
        # 组顺序必须与 GROUP_ORDER 一致（保证列序可复现）
        ordered = tuple(g for g in GROUP_ORDER if g in self.groups)
        if ordered != tuple(self.groups):
            object.__setattr__(self, "groups", ordered)

    # ------------------------------------------------------------ 元信息
    @property
    def key(self) -> str:
        """缓存目录名 / manifest 指纹（配置变了就是另一个 key）。"""
        if self.groups == ("F1",):
            return "F1"
        parts = ["F2" if set(self.groups) == set(GROUP_ORDER) else "FX"]
        for g in self.groups:
            if g == "F1":
                continue
            parts.append(g)
        if "win" in self.groups:
            parts.append("w" + "-".join(str(w) for w in self.windows))
            parts.append("s" + "".join(s[0] for s in self.win_stats))
        if "well" in self.groups:
            parts.append("ws" + "".join(s[0] for s in self.well_stats))
        if "phys" in self.groups and not self.phys_indicators:
            parts.append("noind")
        return "_".join(parts)

    @property
    def is_f1_only(self) -> bool:
        return self.groups == ("F1",)

    def names(self) -> list[str]:
        out: list[str] = []
        for g in self.groups:
            if g == "F1":
                out += list(_f1_names())
            elif g == "phys":
                out += list(_phys_names(self.phys_indicators))
            elif g == "win":
                out += list(_win_names(self.windows, self.win_stats))
            elif g == "well":
                out += list(_well_names(self.well_stats, self.well_scalars))
        return out

    def group_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for g in self.groups:
            if g == "F1":
                sizes[g] = len(_f1_names())
            elif g == "phys":
                sizes[g] = len(_phys_names(self.phys_indicators))
            elif g == "win":
                sizes[g] = len(_win_names(self.windows, self.win_stats))
            elif g == "well":
                sizes[g] = len(_well_names(self.well_stats, self.well_scalars))
        return sizes

    def n_features(self) -> int:
        return sum(self.group_sizes().values())

    def as_dict(self) -> dict[str, Any]:
        return {"groups": list(self.groups), "key": self.key,
                "windows": [int(w) for w in self.windows],
                "win_stats": list(self.win_stats), "well_stats": list(self.well_stats),
                "well_scalars": list(self.well_scalars),
                "phys_indicators": bool(self.phys_indicators),
                "n_features": self.n_features(), "group_sizes": self.group_sizes()}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FeatureSpec":
        return FeatureSpec(
            groups=tuple(d.get("groups", DEFAULT_GROUPS)),
            windows=tuple(int(w) for w in d.get("windows", WN.DEFAULT_WINDOWS)),
            win_stats=tuple(d.get("win_stats", WN.DEFAULT_STATS)),
            well_stats=tuple(d.get("well_stats", WL.WELL_STATS)),
            well_scalars=tuple(d.get("well_scalars", WL.WELL_SCALARS)),
            phys_indicators=bool(d.get("phys_indicators", True)),
        )


def spec_from_name(name: str) -> FeatureSpec:
    """`F1` / `F2` / `F1+phys` / `F2-win-well` 这类简写到 `FeatureSpec`。"""
    raw = str(name or "").strip()
    if raw.upper() == "F1":
        return FeatureSpec(groups=("F1",))
    if raw.upper() == "F2":
        return FeatureSpec(groups=F2_GROUPS)
    toks = [t for t in raw.replace("+", "-").split("-") if t]
    groups = ["F1"]
    for t in toks:
        tl = t.lower()
        if tl in ("f1", "f2"):
            continue
        if tl not in GROUP_ORDER:
            raise ValueError(f"unknown feature group token {t!r} in {name!r}")
        if tl not in groups:
            groups.append(tl)
    return FeatureSpec(groups=tuple(groups))


# ---------------------------------------------------------------- 逐井构造
def build_matrix(shard: dict[str, Any], spec: FeatureSpec,
                 phys_params: PH.PhysicsParams | None = None,
                 with_meta: bool = False):
    """由单井 raw 分片构造 `spec` 指定的特征矩阵。

    `shard` 需要 `inputs`(n,13) / `missing`(n,13) / `depth`(n)。
    `phys_params is None` 且 spec 含 phys 组时抛错（**禁止**用全量分位数兜底）。
    """
    inputs = np.asarray(shard["inputs"], dtype="float32")
    missing = np.asarray(shard["missing"], dtype="int8")
    depth = np.asarray(shard["depth"], dtype="float32")
    blocks: list[np.ndarray] = []
    names: list[str] = []
    meta: dict[str, Any] = {"groups": {}, "spec": spec.as_dict()}

    for g in spec.groups:
        if g == "F1":
            x = F.build_row_features(inputs, missing, depth)
            blocks.append(x)
            names += list(_f1_names())
            meta["groups"]["F1"] = {"n_features": int(x.shape[1])}
        elif g == "phys":
            if phys_params is None:
                raise ValueError("build_matrix: phys 组需要训练折拟合的 PhysicsParams")
            x, nm = PH.build_physics_features(inputs, phys_params,
                                              with_indicators=spec.phys_indicators)
            blocks.append(x)
            names += nm
            meta["groups"]["phys"] = {"n_features": int(x.shape[1]),
                                      "params": phys_params.as_dict()}
        elif g == "win":
            x, nm, wmeta = WN.build_window_features(inputs, spec.windows, spec.win_stats)
            blocks.append(x)
            names += nm
            meta["groups"]["win"] = wmeta
        elif g == "well":
            x, nm = WL.build_well_features(inputs, depth, missing,
                                           spec.well_stats, spec.well_scalars)
            blocks.append(x)
            names += nm
            meta["groups"]["well"] = {"n_features": int(x.shape[1])}
    if not blocks:
        raise ValueError("build_matrix: spec.groups 为空")
    X = np.concatenate(blocks, axis=1).astype("float32")
    if X.shape[1] != len(names):
        raise AssertionError(f"列数与列名不一致：{X.shape[1]} vs {len(names)}")
    return (X, names, meta) if with_meta else (X, names)


def fit_physics_params(shards: Iterable[dict[str, Any]]) -> PH.PhysicsParams:
    """只用**给定井**的曲线拟合 GR/SP 分位数基线（折内 fit 的唯一入口）。"""
    chunks = [np.asarray(s["inputs"], dtype="float64") for s in shards]
    if not chunks:
        raise ValueError("fit_physics_params: 没有井")
    return PH.fit_physics_params(np.concatenate(chunks, axis=0))


# ---------------------------------------------------------------- 缓存
def feature_dir(cache_root: str | Path, spec: FeatureSpec, split: str = "train") -> Path:
    return Path(cache_root) / "feat" / spec.key / split


def feature_path(cache_root: str | Path, spec: FeatureSpec, well_id: str,
                 split: str = "train") -> Path:
    return feature_dir(cache_root, spec, split) / f"{well_id}.npz"


def write_feature_cache(cache_root: str | Path, spec: FeatureSpec, well_id: str,
                        X: "np.ndarray", names: Sequence[str], split: str = "train",
                        extra: dict[str, Any] | None = None) -> Path:
    """原子写（tmp → rename）；`names` 与 `spec` 一并落盘以便自检。"""
    p = feature_path(cache_root, spec, well_id, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"X": np.asarray(X, dtype="float32"),
               "names": np.array(list(names), dtype=object),
               "spec_key": np.array([spec.key], dtype=object)}
    for k, v in (extra or {}).items():
        payload[k] = np.asarray(v)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(p)
    return p


def read_feature_cache(cache_root: str | Path, spec: FeatureSpec, well_id: str,
                       split: str = "train") -> tuple["np.ndarray", list[str]]:
    p = feature_path(cache_root, spec, well_id, split)
    if not p.is_file():
        raise FileNotFoundError(f"feature cache missing: {p}（先跑 E2/code/build_features.py）")
    with np.load(p, allow_pickle=True) as z:
        X = z["X"].astype("float32")
        names = [str(x) for x in list(z["names"])]
        key = str(z["spec_key"][0]) if "spec_key" in z.files else None
    if key and key != spec.key:
        raise ValueError(f"{p} 是 {key} 的缓存，与请求的 {spec.key} 不符（换配置要换目录）")
    if X.shape[1] != len(names):
        raise ValueError(f"{p}: X 列数 {X.shape[1]} 与 names {len(names)} 不一致")
    return X, names


def build_feature_cache(cache_root: str | Path, spec: FeatureSpec, wells: Sequence[str],
                        phys_params: PH.PhysicsParams, split: str = "train",
                        from_raw: Any = None, verbose: bool = False) -> dict[str, Any]:
    """为一组井构建特征缓存（幂等：已存在且 spec 一致则跳过）。

    `from_raw(well)` 需返回 raw 分片 dict（由调用方注入 `row_dataset`/`dataset` 的读取器，
    避免本模块依赖具体缓存实现）。
    """
    import time
    t0 = time.time()
    built, skipped, rows = 0, 0, 0
    for w in wells:
        p = feature_path(cache_root, spec, w, split)
        if p.is_file():
            skipped += 1
            continue
        shard = from_raw(w)
        X, names, _ = build_matrix(shard, spec, phys_params, with_meta=True)
        write_feature_cache(cache_root, spec, w, X, names, split=split)
        rows += int(X.shape[0])
        built += 1
        if verbose and built % 20 == 0:
            print(f"  [feat] {spec.key} {split}: {built} built, {rows} rows", flush=True)
    d = feature_dir(cache_root, spec, split)
    nbytes = sum(f.stat().st_size for f in d.glob("*.npz")) if d.is_dir() else 0
    return {"spec_key": spec.key, "split": split, "wells": len(wells), "built": built,
            "skipped": skipped, "rows": rows, "bytes": nbytes,
            "seconds": round(time.time() - t0, 3)}


# ---------------------------------------------------------------- 溯源
F1_PROVENANCE: dict[str, tuple[str, str]] = {
    "DEPTH_RAW": ("DEPTH 原始值（m）", "原始数据列"),
    "miss_ratio": ("14 位（13 曲线 + DEPTH）缺测比例", "features/basic.py"),
    "rel_depth": ("(depth − depth_min)/(depth_max − depth_min)", "features/basic.py"),
    "depth_step": ("相邻采样间隔（m，首行用中位间隔）", "features/basic.py"),
    "depth_index_norm": ("井内行序号 /(n−1)", "features/basic.py"),
}


def _column_group(name: str, spec: FeatureSpec) -> str:
    """**唯一**的列 → 组归属判据（供溯源与审计共用，避免两套逻辑漂移）。

    顺序敏感：先 F1 的固定列名，再 phys（含 `_ok` 指示位），再 win（`_w<数字>_`），
    最后 well（`_well_` 与 `well_*` 标量）。
    """
    if name in C.INPUT_COLUMNS or name in F1_PROVENANCE or name == "DEPTH_miss" \
            or name.endswith("_miss"):
        return "F1"
    if name.endswith("_ok") or name in PH.PHYSICS_PROVENANCE:
        return "phys"
    if "_w" in name:
        tail = name.rsplit("_w", 1)[1]
        if tail.split("_", 1)[0].isdigit():
            return "win"
    if "_well_" in name or name.startswith("well_"):
        return "well"
    return "unknown"


def provenance_rows(spec: FeatureSpec) -> list[dict[str, str]]:
    """(列名, 组, 公式, 依据) —— 写入 `E2_feature_provenance.csv`。"""
    rows: list[dict[str, str]] = []
    unknown: list[str] = []
    for name in spec.names():
        grp = _column_group(name, spec)
        if name in F1_PROVENANCE:
            formula, src = F1_PROVENANCE[name]
        elif name in C.INPUT_COLUMNS:
            formula, src = f"{name} 原始曲线", "原始数据列"
        elif name == "DEPTH_miss":
            formula, src = "DEPTH 缺测指示位（哨兵/NaN）", "features/basic.py"
        elif name.endswith("_miss"):
            formula, src = f"{name[:-5]} 的缺测指示位（哨兵/NaN）", "features/basic.py"
        elif name.endswith("_ok"):
            formula, src = (f"{name[:-3]} 是否为有限值（缺失传播指示位）",
                            "features/physics.py")
        elif name in PH.PHYSICS_PROVENANCE:
            formula, src = PH.PHYSICS_PROVENANCE[name]
        elif grp == "win":
            head, tail = name.rsplit("_w", 1)
            w, stat = tail.split("_", 1)
            formula = (f"{head} 的居中窗口(w={w} 点) {stat}（仅同井邻域；缺失不填 0；"
                       f"trend = 窗内最小二乘斜率）")
            src = "资料库/07 §6（窗口统计）"
        elif grp == "well" and "_well_" in name:
            head, stat = name.rsplit("_well_", 1)
            formula = f"{head} 在该井内的 {stat}（逐行广播；只用该井自身的行）"
            src = "资料库/08 §0.1-4（井级信息）"
        elif name.startswith("well_"):
            formula, src = {
                "well_n_rows": ("该井行数", "features/well.py"),
                "well_depth_span": ("深度跨度 max−min", "features/well.py"),
                "well_depth_step_mean": ("相邻深度差的平均绝对值", "features/well.py"),
                "well_devi_mean": ("井斜 DEVI 的井内均值", "features/well.py"),
                "well_miss_frac_mean": ("逐行 14 位缺测比例的井内均值", "features/well.py"),
            }.get(name, ("井级标量", "features/well.py"))
        else:
            formula, src = "（未登记）", "（未登记）"
            unknown.append(name)
        rows.append({"column": name, "group": grp, "formula": formula, "source": src})
    if unknown:
        raise AssertionError(f"provenance 未覆盖这些列：{unknown[:10]}"
                             f"（共 {len(unknown)} 列）—— E2/P0 §7 要求全覆盖")
    return rows


def provenance_group_counts(spec: FeatureSpec) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in provenance_rows(spec):
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    return counts


def audit_no_target_derivation(spec: FeatureSpec) -> dict[str, Any]:
    """审计：任何列名都不得暗示用了目标值（E2 §8）。"""
    bad_tokens = ("por_", "perm_", "sw_")
    offenders = []
    for n in spec.names():
        low = n.lower()
        # F1/phys 里合法的名字（phi_* 是孔隙度派生，不是标签；por_max 之类不在列名里）
        if low.startswith(("phi_", "den_cnl", "ac_den")):
            continue
        if low.startswith(bad_tokens) or low in ("por", "perm", "sw"):
            offenders.append(n)
    return {"ok": not offenders, "offenders": offenders,
            "note": "派生特征不得消费 POR/PERM/SW 标签（E2 §8）"}
