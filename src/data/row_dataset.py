"""E1/P0 行级数据装配层：F1 分片缓存 + **折内**标准化 + 目标尺度参数。

本模块是「特征/标签 → 训练张量」的唯一通道，只依赖 numpy（**不需要 torch**），
因此本机（无 GPU）就能跑全部口径层单测。

三条硬纪律（违反即数据泄漏，E1/P0 §7/§8）
------------------------------------------
1. **标准化参数只在训练折 fit**：`RowScaler.fit` 只接受训练折的行；
   `assemble(...)` 永远要求调用方显式传入已 fit 的 scaler（或传入训练折自行 fit 后
   再 transform 验证折），**不存在** "在整表上 fit" 的入口。
2. **目标尺度参数（`s_por`/`s_sw`/`sw_mu`/`sw_sigma`/`por_max`）只在训练折 fit**：
   由 `features.basic.fit_target_scalers` 计算，本模块只负责把它随折一起落盘/读回
   （`scalers/E1_fold{k}.json`），推理期用它做反变换。
3. **标签与特征逐行对齐**：`X.shape[0] == y.shape[0] == mask.shape[0]`，并按井记录
   `well_index`/`offset`，使得"按井汇总"、"逐井非退化比例"、"井级 cluster bootstrap"
   都不需要字符串列表常驻内存。

数据布局
--------
输入（E0/P1 的分片）::

    $V4_CACHE_ROOT/raw/<split>/<well>.npz    depth (n,), inputs (n,13), missing (n,13)
    $V4_CACHE_ROOT/labels/<well>.npz         targets (n,3), target_missing (n,3), placeholder (n,)

输出（本模块的 F1 缓存，可选但推荐）::

    $V4_CACHE_ROOT/row/<split>/<well>.npz    X_raw (n,32) f32, depth (n,), + 标签列
    $V4_REPORTS_DIR/E1_row_features.json     维度/缺失率/内存/耗时/逐折尺度
    $V4_DATA_ROOT/v4/scalers/E1_fold{k}.json 标准化 + 目标尺度参数（**仅训练折**）

内存预算（16 GiB 机器上的实测口径）
-----------------------------------
80 井 × 730,268 行 × 32 列 float32 ≈ 93 MB；加标签/mask/井索引约 130 MB。
因此 E1 允许"按井预分配 + 逐井填充"的整折装配（不做全量 concat 的多次拷贝），
本模块的 `assemble` 正是这样实现的：先算总行数，`np.empty` 一次，再逐井写入。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import constants as C
from ..features import basic as F
from ..portability import HAS_NUMPY

if HAS_NUMPY:
    import numpy as np

from . import dataset as D


# ---------------------------------------------------------------- F1 缓存路径
def row_dir(cache_root: str | Path, split: str = "train") -> Path:
    return Path(cache_root) / "row" / split


def row_path(cache_root: str | Path, well_id: str, split: str = "train") -> Path:
    return row_dir(cache_root, split) / f"{well_id}.npz"


LABEL_KEYS: tuple[str, ...] = ("por", "perm_z", "sw", "mask", "y_atom", "y_joint")


def build_well_row_arrays(cache_root: str | Path, well_id: str,
                          split: str = "train") -> dict[str, Any]:
    """由 E0 分片构造单井的 (X_raw, depth, 标签...)。

    X_raw 是**未标准化**的 F1 特征（标准化必须等折划分之后）；
    `split == "test"` 时没有标签（返回 None）。
    """
    if not HAS_NUMPY:
        raise RuntimeError("build_well_row_arrays requires numpy")
    sh = D.read_well_shard(cache_root, well_id, split)
    X = F.build_row_features(sh["inputs"], sh["missing"], sh["depth"])
    out: dict[str, Any] = {"X_raw": X, "depth": sh["depth"].astype("float32"),
                           "n_rows": int(X.shape[0])}
    if sh["targets"] is not None:
        lab = F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
        out.update(lab)
    return out


def write_row_shard(cache_root: str | Path, well_id: str, split: str,
                    rec: dict[str, Any]) -> Path:
    """原子写 F1 分片（tmp → rename）。"""
    if not HAS_NUMPY:
        raise RuntimeError("write_row_shard requires numpy")
    p = row_path(cache_root, well_id, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"X_raw": rec["X_raw"].astype("float32"),
                               "depth": rec["depth"].astype("float32")}
    for k in LABEL_KEYS:
        if rec.get(k) is not None:
            payload[k] = np.asarray(rec[k])
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(p)
    return p


def read_row_shard(cache_root: str | Path, well_id: str, split: str = "train") -> dict[str, Any]:
    if not HAS_NUMPY:
        raise RuntimeError("read_row_shard requires numpy")
    p = row_path(cache_root, well_id, split)
    if not p.is_file():
        return build_well_row_arrays(cache_root, well_id, split)
    out: dict[str, Any] = {}
    with np.load(p, allow_pickle=True) as z:
        for k in z.files:
            out[k] = z[k]
    out["X_raw"] = out["X_raw"].astype("float32")
    out["depth"] = out["depth"].astype("float32")
    return out


def build_row_cache(cache_root: str | Path, wells: Sequence[str], split: str = "train",
                    verbose: bool = True) -> dict[str, Any]:
    """把 F1 特征落到 `cache/row/<split>/`（幂等；已存在则跳过）。"""
    t0 = time.time()
    rows, built = 0, 0
    for w in wells:
        p = row_path(cache_root, w, split)
        if p.is_file():
            with np.load(p, allow_pickle=True) as z:
                rows += int(z["X_raw"].shape[0])
            continue
        rec = build_well_row_arrays(cache_root, w, split)
        write_row_shard(cache_root, w, split, rec)
        rows += rec["n_rows"]
        built += 1
        if verbose and built % 20 == 0:
            print(f"  [row] {built} wells, {rows} rows", flush=True)
    return {"split": split, "wells": len(wells), "rows": rows, "built": built,
            "seconds": round(time.time() - t0, 3),
            "bytes": sum(f.stat().st_size for f in row_dir(cache_root, split).glob("*.npz"))
            if row_dir(cache_root, split).is_dir() else 0}


# ---------------------------------------------------------------- 折内标准化
@dataclass
class RowScaler:
    """中位数填补 + 零均值单位方差标准化（**参数只允许来自训练折**）。

    变换顺序固定为：`x ← (x − mean) / std`，其中缺失值先被 `median` 填补。
    对"缺失指示位"这类恒为 0/1 的列同样适用（其 std 通常 > 0）。

    `names` 记录列语义（F1 32 列 / F2 368 列 / 消融子集都可能），
    这样 scaler JSON 与 checkpoint manifest 能自解释，而不是靠"宽度猜特征版本"。
    """
    median: "np.ndarray"
    mean: "np.ndarray"
    std: "np.ndarray"
    n_fit_rows: int = 0
    fit_wells: tuple[str, ...] = ()
    names: tuple[str, ...] = ()

    @staticmethod
    def fit(X: "np.ndarray", wells: Sequence[str] = (),
            names: Sequence[str] | None = None) -> "RowScaler":
        X = np.asarray(X, dtype="float64")
        if X.ndim != 2:
            raise ValueError(f"RowScaler.fit expects 2-D, got {X.shape}")
        finite = np.isfinite(X)
        if not finite.any():
            raise ValueError("RowScaler.fit: all values are NaN/Inf")
        # 逐列中位数（忽略 NaN）；整列全缺时退化为 0
        med = np.zeros(X.shape[1], dtype="float64")
        for j in range(X.shape[1]):
            col = X[finite[:, j], j]
            med[j] = float(np.median(col)) if col.size else 0.0
        filled = np.where(finite, X, med[None, :])
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        if names is None:
            names = tuple(F.FEATURE_NAMES) if X.shape[1] == F.N_FEATURES else \
                tuple(f"f{j}" for j in range(X.shape[1]))
        if len(names) != X.shape[1]:
            raise ValueError(f"RowScaler.fit: names {len(names)} != 列数 {X.shape[1]}")
        return RowScaler(median=med, mean=mean, std=std,
                         n_fit_rows=int(X.shape[0]), fit_wells=tuple(wells),
                         names=tuple(names))

    def transform(self, X: "np.ndarray") -> "np.ndarray":
        X = np.asarray(X, dtype="float64")
        if X.shape[1] != self.median.shape[0]:
            raise ValueError(f"RowScaler.transform: expected {self.median.shape[0]} cols, "
                             f"got {X.shape[1]}")
        filled = np.where(np.isfinite(X), X, self.median[None, :])
        return ((filled - self.mean[None, :]) / self.std[None, :]).astype("float32")

    def transform_blocked(self, X: "np.ndarray", block_rows: int = 65536
                          ) -> "np.ndarray":
        """分块调用 ``transform``，峰值内存从 O(float64 整表) 降到 O(block)。

        逐元素计算与整表 ``transform`` 一致，因此不改变数值结果；
        只是避免 E3 在 584k×266 输入上一次性分配多份 float64。
        """
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"RowScaler.transform_blocked expects 2-D, got {arr.shape}")
        if arr.shape[1] != self.median.shape[0]:
            raise ValueError(f"RowScaler.transform_blocked: expected "
                             f"{self.median.shape[0]} cols, got {arr.shape[1]}")
        out = np.empty((arr.shape[0], arr.shape[1]), dtype="float32")
        step = max(int(block_rows), 1)
        for start in range(0, arr.shape[0], step):
            stop = min(start + step, arr.shape[0])
            out[start:stop] = self.transform(arr[start:stop])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "median_mean_std",
            "median": [float(v) for v in self.median],
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "n_features": int(self.median.shape[0]),
            "feature_names": list(self.names) if self.names
            else list(F.FEATURE_NAMES),
            "n_fit_rows": int(self.n_fit_rows),
            "fit_wells": list(self.fit_wells),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RowScaler":
        return RowScaler(median=np.asarray(d["median"], dtype="float64"),
                         mean=np.asarray(d["mean"], dtype="float64"),
                         std=np.asarray(d["std"], dtype="float64"),
                         n_fit_rows=int(d.get("n_fit_rows", 0)),
                         fit_wells=tuple(d.get("fit_wells", ())),
                         names=tuple(d.get("feature_names", ())))


# ---------------------------------------------------------------- 折装配
def fold_wells(folds: dict[str, Any], k: int) -> tuple[list[str], list[str]]:
    """outer 折 k 的 (训练井, 验证井)。**井维度互斥**，不存在井级泄漏。"""
    n = int(folds["n_folds"])
    if not (0 <= int(k) < n):
        raise ValueError(f"fold {k} out of range [0, {n})")
    train = [w for w in folds["well_list"] if int(folds["fold_of_well"][w]) != int(k)]
    val = [w for w in folds["well_list"] if int(folds["fold_of_well"][w]) == int(k)]
    if set(train) & set(val):
        raise AssertionError(f"fold {k}: train/val wells overlap")
    if len(train) + len(val) != len(folds["well_list"]):
        raise AssertionError(f"fold {k}: wells lost (train={len(train)} val={len(val)})")
    return train, val


@dataclass
class FoldTensors:
    """整折张量（行对齐；按井记录 offset 以便逐井汇总）。"""
    X: "np.ndarray"                 # (n, 32) float32，已标准化
    depth: "np.ndarray"             # (n,)
    well_index: "np.ndarray"        # (n,) int32
    well_ids: tuple[str, ...]
    offsets: "np.ndarray"           # (n_wells+1,) int64
    y_por: "np.ndarray | None" = None
    y_perm_z: "np.ndarray | None" = None
    y_sw: "np.ndarray | None" = None
    mask: "np.ndarray | None" = None
    y_atom: "np.ndarray | None" = None
    y_joint: "np.ndarray | None" = None

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_wells(self) -> int:
        return len(self.well_ids)

    def batch_dict(self) -> dict[str, Any]:
        """训练用 batch 字典（torch 侧自行切片；键名与 `total_loss` 对齐）。"""
        d = {"por": self.y_por, "perm_z": self.y_perm_z, "sw": self.y_sw,
             "mask": self.mask, "y_atom": self.y_atom, "y_joint": self.y_joint}
        return {k: v for k, v in d.items() if v is not None}


def _well_feature_matrix(cache_root: str | Path, well: str, split: str, spec,
                         phys_params=None, allow_onfly: bool = True):
    """取单井特征矩阵：F1 走分片缓存；其它 spec 走 `cache/feat/<key>/`（缺失时现场构造）。

    现场构造是**兜底**（E2 缓存没建时会慢，但不会静默失败）；`allow_onfly=False`
    时直接抛错，便于 Gate 强制"必须用冻结的缓存"。
    """
    from ..features import groups as GRP   # 延迟导入：避免 data<->features 循环
    if spec is None or spec.is_f1_only:
        return read_row_shard(cache_root, well, split)["X_raw"], list(F.FEATURE_NAMES)
    # H2 审查修复：只要 spec 含 phys，就必须使用调用方传入的**折内** phys_params，
    # 不得回退到 `cache/feat/<spec>/` —— 否则缓存的全局分位数会静默造成分布泄漏。
    if "phys" in tuple(spec.groups):
        if phys_params is None:
            raise ValueError(
                "_well_feature_matrix: spec 含 phys 组但未提供 folded PhysicsParams；"
                "禁止读取全局 feat 缓存（会造成训练/验证分布泄漏）。")
        shard = D.read_well_shard(cache_root, well, split)
        X, names = GRP.build_matrix(shard, spec, phys_params)
        return X, names
    try:
        return GRP.read_feature_cache(cache_root, spec, well, split)
    except FileNotFoundError:
        if not allow_onfly:
            raise
        shard = D.read_well_shard(cache_root, well, split)
        X, names = GRP.build_matrix(shard, spec, phys_params)
        return X, names


def assemble(wells: Sequence[str], cache_root: str | Path, scaler: RowScaler | None = None,
             split: str = "train", with_targets: bool = True, spec=None,
             phys_params=None, allow_onfly: bool = True) -> FoldTensors:
    """按井装配整折张量（一次 `np.empty` 预分配 + 逐井写入，避免多次全量拷贝）。

    `spec` 为 `None` 或 F1 时行为与历史版本一致（32 列）；
    给出 `FeatureSpec`（如 F2=368 列）时按 `cache/feat/<spec.key>/` 读取。

    `scaler is None` 时**不做标准化**（返回原始特征），仅用于"先看数据"的诊断路径；
    训练路径必须显式传入**训练折 fit 出来**的 scaler。
    """
    if not HAS_NUMPY:
        raise RuntimeError("assemble requires numpy")
    wells = list(wells)
    pairs = [_well_feature_matrix(cache_root, w, split, spec, phys_params, allow_onfly)
             for w in wells]
    mats = [p_[0] for p_ in pairs]
    names = list(pairs[0][1]) if pairs else list(F.FEATURE_NAMES)
    recs = [read_row_shard(cache_root, w, split) for w in wells]
    sizes = [int(m.shape[0]) for m in mats]
    n = int(sum(sizes))
    n_feat = int(mats[0].shape[1]) if mats else int(F.N_FEATURES)
    if len(names) != n_feat:
        raise ValueError(f"特征列名 {len(names)} 与列数 {n_feat} 不一致（spec={spec}）")
    offsets = np.zeros(len(wells) + 1, dtype="int64")
    offsets[1:] = np.cumsum(sizes)

    X = np.empty((n, n_feat), dtype="float32")
    depth = np.empty(n, dtype="float32")
    well_index = np.empty(n, dtype="int32")
    for i, (mat, r) in enumerate(zip(mats, recs)):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if mat.shape[1] != n_feat:
            raise ValueError(f"well {wells[i]}: {mat.shape[1]} 列 != 首井的 {n_feat} 列"
                             "（特征版本不一致？）")
        X[a:b] = scaler.transform(mat) if scaler is not None else mat
        depth[a:b] = r["depth"]
        well_index[a:b] = i

    out = FoldTensors(X=X, depth=depth, well_index=well_index,
                      well_ids=tuple(wells), offsets=offsets)
    if with_targets:
        if any(r.get("por") is None for r in recs):
            raise ValueError("assemble(with_targets=True) needs labels for every well "
                             "(test split has none)")
        out.y_por = np.empty(n, dtype="float32")
        out.y_perm_z = np.empty(n, dtype="float32")
        out.y_sw = np.empty(n, dtype="float32")
        out.mask = np.empty((n, 3), dtype="float32")
        out.y_atom = np.empty((n, 3), dtype="float32")
        out.y_joint = np.empty(n, dtype="float32")
        for i, r in enumerate(recs):
            a, b = int(offsets[i]), int(offsets[i + 1])
            out.y_por[a:b] = r["por"]
            out.y_perm_z[a:b] = r["perm_z"]
            out.y_sw[a:b] = r["sw"]
            out.mask[a:b] = r["mask"]
            out.y_atom[a:b] = r["y_atom"]
            out.y_joint[a:b] = r["y_joint"]
    return out


def fit_scalers_from_wells(train_wells: Sequence[str], cache_root: str | Path,
                           spec=None, return_tensors: bool = False) -> dict[str, Any]:
    """由**显式给定的训练井**拟合标准化与目标尺度（折内 fit 的唯一入口）。

    ``return_tensors=True`` 时额外返回 ``out["tensors"]``，即用于拟合的
    ``FoldTensors``（X 尚未标准化）。E3 序列路径可复用它，避免训练时逐 chunk
    重新读盘；默认 False 保持旧调用方行为不变。


    `fit_fold` 是"按 outer 折号"的语法糖；本函数供两阶段协议里的
    "inner-train 井"与"全部 outer-train 井"分别调用，语义完全一致：
    **只喂训练井，验证井绝不参与任何参数估计**。
    """
    from ..features import groups as GRP
    train_wells = list(train_wells)
    phys = None
    if spec is not None and "phys" in tuple(spec.groups):
        # GRmin/GRmax/SPmin/SPmax 等分位数基线**只能**由训练井的曲线拟合（E2/P0 §6）
        phys = GRP.fit_physics_params(D.read_well_shard(cache_root, w, "train")
                                      for w in train_wells)
    tr = assemble(train_wells, cache_root, scaler=None, with_targets=True, spec=spec,
                  phys_params=phys)
    names = tuple(GRP.FeatureSpec.from_dict(spec.as_dict()).names()) if spec is not None else None
    scaler = RowScaler.fit(tr.X, wells=train_wells, names=names)
    target = F.fit_target_scalers(tr.y_por, tr.y_sw, tr.mask, z_perm=tr.y_perm_z, y_perm=None)
    out = {"scaler": scaler, "target": target, "n_train_rows": tr.n_rows,
           "train_wells": train_wells}
    if phys is not None:
        out["phys_params"] = phys
    if return_tensors:
        out["tensors"] = tr
    return out


def fit_fold(folds: dict[str, Any], k: int, cache_root: str | Path,
             spec=None) -> dict[str, Any]:
    """折 k 的**训练折** fit：标准化 + 目标尺度（+ F2 的物理分位数），全部只来自训练井。"""
    train_wells, val_wells = fold_wells(folds, k)
    fit = fit_scalers_from_wells(train_wells, cache_root, spec=spec)
    fit.update({"fold": int(k), "val_wells": val_wells})
    return fit


def scaler_payload(fold_fit: dict[str, Any]) -> dict[str, Any]:
    """落盘用的一份 JSON（标准化 + 目标尺度 + 折井清单）。"""
    out = {
        "fold": int(fold_fit["fold"]),
        "row_scaler": fold_fit["scaler"].to_dict(),
        "target_scalers": dict(fold_fit["target"]),
        "train_wells": list(fold_fit["train_wells"]),
        "val_wells": list(fold_fit["val_wells"]),
        "n_train_rows": int(fold_fit["n_train_rows"]),
        "note": "所有尺度参数只由**训练折**有效标签/行拟合；推理期用同一份做反变换",
    }
    if fold_fit.get("phys_params") is not None:
        out["physics_params"] = fold_fit["phys_params"].as_dict()
    return out


def save_scaler_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def load_scaler_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def memory_report(t: FoldTensors) -> dict[str, Any]:
    """整折张量的常驻内存估算（nbytes 累加，不实际分配）。"""
    total = int(t.X.nbytes + t.depth.nbytes + t.well_index.nbytes + t.offsets.nbytes)
    for k in ("y_por", "y_perm_z", "y_sw", "mask", "y_atom", "y_joint"):
        v = getattr(t, k)
        if v is not None:
            total += int(v.nbytes)
    return {"n_rows": t.n_rows, "n_wells": t.n_wells,
            "bytes": total, "gib": round(total / (1024 ** 3), 4)}


def feature_report(t: FoldTensors, scaler: RowScaler | None = None) -> dict[str, Any]:
    """特征维度 / 缺失率 / 内存（写入 `E1_row_features.json`）。"""
    if not HAS_NUMPY:
        raise RuntimeError("feature_report requires numpy")
    X_raw = np.asarray(t.X, dtype="float32")
    n_feat = int(X_raw.shape[1])
    # M7：F2 有 368 列，不能固定用 F1 的 32 个列名（否则 IndexError/名字错配）。
    if scaler is not None and scaler.names and len(scaler.names) == n_feat:
        names = list(scaler.names)
    elif n_feat == len(F.FEATURE_NAMES):
        names = list(F.FEATURE_NAMES)
    else:
        names = [f"f{j}" for j in range(n_feat)]
    rep: dict[str, Any] = {
        "n_features": n_feat,
        "feature_names": names,
        "n_rows": t.n_rows,
        "n_wells": t.n_wells,
        "nan_rate_overall": float(np.mean(~np.isfinite(X_raw))),
        "nan_rate_per_feature": [float(v) for v in np.mean(~np.isfinite(X_raw), axis=0)],
        "memory": memory_report(t),
    }
    if scaler is not None:
        rep["row_scaler"] = {"n_fit_rows": scaler.n_fit_rows,
                             "fit_wells": list(scaler.fit_wells),
                             "zero_std_features": [names[j]
                                                   for j in range(min(len(names),
                                                                      scaler.std.shape[0]))
                                                   if scaler.std[j] == 1.0]}
    return rep


def assert_alignment(t: FoldTensors) -> None:
    """逐行对齐断言（E1/P0 §7 第一条）。"""
    n = t.n_rows
    assert t.depth.shape[0] == n, "depth rows mismatch"
    assert t.well_index.shape[0] == n, "well_index rows mismatch"
    for name in ("y_por", "y_perm_z", "y_sw", "y_joint"):
        v = getattr(t, name)
        if v is not None:
            assert v.shape[0] == n, f"{name} rows mismatch"
    if t.mask is not None:
        assert t.mask.shape == (n, 3), f"mask shape {t.mask.shape} != {(n, 3)}"
    if t.y_atom is not None:
        assert t.y_atom.shape == (n, 3), f"y_atom shape {t.y_atom.shape} != {(n, 3)}"
    assert int(t.offsets[-1]) == n, "offsets do not sum to n_rows"
    assert t.well_index.min() >= 0 and t.well_index.max() < t.n_wells, "well_index out of range"


def subset(t: FoldTensors, wells: Iterable[str]) -> FoldTensors:
    """按井名取子集（inner 折切分用；保持行序与偏移）。"""
    want = set(wells)
    idx = [i for i, w in enumerate(t.well_ids) if w in want]
    if not idx:
        raise ValueError("subset: no well matched")
    sl = np.concatenate([np.arange(t.offsets[i], t.offsets[i + 1]) for i in idx])
    out = FoldTensors(
        X=t.X[sl], depth=t.depth[sl],
        well_index=np.zeros(sl.size, dtype="int32"),
        well_ids=tuple(t.well_ids[i] for i in idx),
        offsets=np.zeros(len(idx) + 1, dtype="int64"),
    )
    sizes = [int(t.offsets[i + 1] - t.offsets[i]) for i in idx]
    out.offsets[1:] = np.cumsum(sizes)
    for i, _ in enumerate(idx):
        a, b = int(out.offsets[i]), int(out.offsets[i + 1])
        out.well_index[a:b] = i
    for name in ("y_por", "y_perm_z", "y_sw", "y_joint"):
        v = getattr(t, name)
        if v is not None:
            setattr(out, name, v[sl])
    for name in ("mask", "y_atom"):
        v = getattr(t, name)
        if v is not None:
            setattr(out, name, v[sl])
    return out


def atom_slice_report(y_true: Any, y_pred: Any, mask: Any) -> dict[str, Any]:
    """`POR=0` / `POR<0.1` 切片的逐目标 Acc（E1/P0 §7 与 §5 步 7）。

    构造方式与训练标签同源：原子行由 `ATOM_VALUES` 判定，切片只看**POR 目标非缺测**的行。
    """
    if not HAS_NUMPY:
        raise RuntimeError("atom_slice_report requires numpy")
    from ..score import score_arrays

    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    m = np.asarray(mask, dtype=bool)
    obs = m[:, 0]
    por = yt[:, 0]
    out: dict[str, Any] = {}
    for name, sel in (("por_eq_0", obs & (por == 0.0)),
                      ("por_lt_0p1", obs & (por < 0.1)),
                      ("por_eq_atom", obs & F.is_atom_value(por, "POR"))):
        n = int(sel.sum())
        entry: dict[str, Any] = {"n_rows": n}
        if n:
            s = score_arrays(yt[sel], yp[sel], missing=~m[sel])
            entry.update({k: float(v) for k, v in s.items()
                          if k not in ("missing_mode", "n_rows")})
        out[name] = entry
    return out
