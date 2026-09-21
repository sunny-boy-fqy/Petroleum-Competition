"""序列数据集与分块（E3/P0）：chunk 规划、接缝加权拼接、按需读分片、内存自检。

三条硬契约
----------
1. **覆盖完整无丢点**：`chunks_for(n)` 覆盖 `[0, n)` 的每一个点（末尾 chunk 允许右对齐
   回退，而不是丢弃不足一窗的尾巴）；
2. **接缝无跳变**：重叠区用**加权**（三角/汉宁）融合并**除以权重和**归一化，
   `stitch_chunks(整段推理, 分块加权拼接)` 的逐点差必须 < 容差；
3. **不缓存分片**：`WellShardReader` 每次取数都"打开 → 取 → 关闭"，绝不把整井放进全局字典
   （16 GiB 内存是真正的瓶颈）；worker 常驻内存有显式自检（E3/P0 §7 要求 < 300 MB）。
"""
from __future__ import annotations

import resource
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from .. import constants as C
from ..portability import HAS_NUMPY, HAS_TORCH
from . import dataset as D
from . import row_dataset as RD

DEFAULT_CHUNK = 1024
DEFAULT_OVERLAP = 128
WORKER_MEM_LIMIT_MB = 300.0


# ---------------------------------------------------------------- chunk 规划
def chunks_for(n: int, chunk: int = DEFAULT_CHUNK, overlap: int = DEFAULT_OVERLAP
               ) -> list[tuple[int, int]]:
    """返回 `[(start, length), ...]`，覆盖 `[0, n)` 的每个点。

    末尾不足一窗时**右对齐回退**（`start = max(n - chunk, 0)`），因此：
    - 所有点都被覆盖；
    - 每个 chunk 长度 = `min(chunk, n)`（最后一块也不例外），便于定长训练。
    """
    if n <= 0:
        return []
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if not (0 <= overlap < chunk):
        raise ValueError(f"overlap must be in [0, chunk)，got {overlap} / {chunk}")
    if n <= chunk:
        return [(0, int(n))]
    step = chunk - overlap
    out: list[tuple[int, int]] = []
    start = 0
    while start + chunk < n:
        out.append((int(start), int(chunk)))
        start += step
    out.append((int(max(n - chunk, 0)), int(chunk)))
    return out


def coverage_report(n: int, chunks: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """覆盖率与重叠统计（E3/P0 §7 第一条）。"""
    seen = np.zeros(int(n), dtype="int32")
    for s, L in chunks:
        seen[s:s + L] += 1
    covered = int((seen > 0).sum())
    return {"n_rows": int(n), "n_chunks": len(chunks),
            "covered": covered, "uncovered": int(n - covered),
            "max_overlap_count": int(seen.max()) if n else 0,
            "seam_rows": int((seen > 1).sum()),
            "complete": bool(covered == n)}


# ---------------------------------------------------------------- 拼接权重
def chunk_weights(length: int, kind: str = "triangular") -> "np.ndarray":
    """单个 chunk 的融合权重（长度 `length`，两端小、中间大）。"""
    if kind == "equal":
        return np.ones(int(length), dtype="float64")
    x = np.linspace(-1.0, 1.0, int(length)) if length > 1 else np.zeros(1)
    if kind == "triangular":
        return (1.0 - np.abs(x)).astype("float64")
    if kind == "hann":
        return (0.5 * (1.0 + np.cos(np.pi * x))).astype("float64")
    raise ValueError(f"unknown chunk weight kind: {kind!r}")


def stitch_chunks(chunks: Sequence[tuple[int, int]], preds: Sequence["np.ndarray"],
                  n: int, kind: str = "triangular", min_weight: float = 1e-6
                  ) -> "np.ndarray":
    """重叠-相加再归一化（overlap-add + weighted average）。

    `preds[i]` 形状 `(length_i, ...)`，与 `chunks[i]` 一一对应；
    返回 `(n, ...)`。**两端权重为 0 的窗口会让边缘点拿不到权重**，因此权重下限用
    `min_weight` 兜底（三角窗两端恰好为 0，见下）。
    """
    if not HAS_NUMPY:
        raise RuntimeError("stitch_chunks requires numpy")
    if len(preds) != len(chunks):
        raise ValueError(f"chunks/preds 数量不一致：{len(chunks)} vs {len(preds)}")
    if not preds and n > 0:
        raise ValueError("stitch_chunks: 有行但没有 chunk")
    shape = (int(n),) + tuple(np.asarray(preds[0]).shape[1:])
    acc = np.zeros(shape, dtype="float64")
    wsum = np.zeros(int(n), dtype="float64")
    for (s, L), p in zip(chunks, preds):
        p = np.asarray(p, dtype="float64")
        if p.shape[0] != L:
            raise ValueError(f"chunk 长度 {p.shape[0]} 与规划 {L} 不一致")
        w = np.maximum(chunk_weights(L, kind), min_weight)
        acc[s:s + L] += p * w.reshape((-1,) + (1,) * (p.ndim - 1))
        wsum[s:s + L] += w
    if (wsum <= 0).any():
        raise AssertionError("stitch_chunks: 存在未被任何 chunk 覆盖的行（规划有 bug）")
    return (acc / wsum.reshape((-1,) + (1,) * (acc.ndim - 1))).astype("float64")


def seam_report(chunks: Sequence[tuple[int, int]], stitched: "np.ndarray",
                whole: "np.ndarray", tol: float = 1e-6) -> dict[str, Any]:
    """整段推理 vs 分块拼接的逐点差（E3/P1 §7「无缝跳变」判据）。"""
    a = np.asarray(stitched, dtype="float64")
    b = np.asarray(whole, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致：{a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    seams = [s for s, _ in chunks[1:]]
    seam_vals = [float(np.abs(diff[min(s, len(diff) - 1)]).max()) for s in seams
                 if len(diff)]
    seam_max = float(max(seam_vals)) if seam_vals else 0.0
    return {"max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "seam_max_abs_diff": seam_max, "n_seams": len(seams),
            "within_tolerance": bool((diff.max() if diff.size else 0.0) <= tol),
            "tolerance": tol}


# ---------------------------------------------------------------- 分片读取
@dataclass
class WellShardReader:
    """按需读井分片：**每次调用都重新打开文件**（不缓存），用完即释放。

    特点：
      * `feature_matrix(well)` 返回 `(n, F)` float32（F1 或 `FeatureSpec` 缓存）；
      * `chunk(well, start, length)` 只返回该 chunk 的 `(X, y, mask, y_atom, y_joint)`；
      * `rss_mb()` 上报当前进程常驻内存，`assert_worker_budget()` 做 E3/P0 §7 的自检。
    """
    cache_root: Any
    split: str = "train"
    spec: Any = None
    phys_params: Any = None

    def feature_matrix(self, well: str) -> "np.ndarray":
        X, _names = RD._well_feature_matrix(self.cache_root, well, self.split, self.spec,
                                            self.phys_params)
        return np.asarray(X, dtype="float32")

    def labels(self, well: str) -> dict[str, "np.ndarray"]:
        sh = D.read_well_shard(self.cache_root, well, self.split)
        if sh.get("targets") is None:
            raise ValueError(f"{well}: 该 split 没有标签（推理路径请用 feature_matrix）")
        from ..features import basic as F
        return F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])

    def chunk(self, well: str, start: int, length: int) -> dict[str, Any]:
        X = self.feature_matrix(well)[start:start + length]
        out: dict[str, Any] = {"X": X, "start": int(start), "length": int(X.shape[0])}
        if self.split == "train":
            lab = self.labels(well)
            for k in ("por", "perm_z", "sw", "mask", "y_atom", "y_joint"):
                out[k] = np.asarray(lab[k])[start:start + length]
        return out

    # ------------------------------------------------------------ 内存纪律
    @staticmethod
    def rss_mb() -> float:
        """当前进程常驻内存（MiB；Linux ru_maxrss 单位为 KiB）。"""
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    @classmethod
    def assert_worker_budget(cls, limit_mb: float = WORKER_MEM_LIMIT_MB,
                             baseline_mb: float = 0.0) -> float:
        """E3/P0 §7：**每个 worker 的增量常驻**必须 < 300 MB；超限报错而不是静默增长。

        `baseline_mb`：进入数据加载循环**之前**的 RSS（父进程里包含 torch 与框架本身，
        通常 300–600 MB，直接拿绝对 RSS 比 300 MB 会把框架开销误判成"缓存了分片"）。
        E3 的脚本在训练开始时测一次基线并传入；DataLoader worker 内部可传 0。
        """
        rss = cls.rss_mb()
        delta = rss - float(baseline_mb)
        if delta > float(limit_mb):
            raise MemoryError(
                f"worker RSS 增量 {delta:.1f} MiB（当前 {rss} / 基线 {baseline_mb}）"
                f"超过上限 {limit_mb} MiB —— 分片被判为被缓存了？"
                "检查是否把整井矩阵放进全局字典或 Dataset 属性")
        return rss


if HAS_TORCH:
    import torch
    from torch.utils.data import Dataset

    class SeqChunkDataset(Dataset):
        """按 chunk 采样的 Dataset：`__getitem__` 打开分片 → 取 chunk → 关闭。

        采样顺序由 `(seed, epoch)` 决定，**固定 seed 下可复现**（E3/P0 §7）。
        """

        def __init__(self, cache_root, wells: Sequence[str], chunk: int = DEFAULT_CHUNK,
                     overlap: int = DEFAULT_OVERLAP, split: str = "train",
                     spec=None, phys_params=None, seed: int = 42, epoch: int = 0,
                     with_targets: bool | None = None, scaler=None):
            self.reader = WellShardReader(cache_root, split=split, spec=spec,
                                          phys_params=phys_params)
            # 折内 scaler（可选）：在 `__getitem__` 里就地标准化，**不修改类**。
            # 早期版本在外部 patch `ds.__class__.__getitem__`，那是全局副作用
            # （会污染所有实例并可能重复包装），已废弃。
            self.scaler = scaler
            self.wells = list(wells)
            self.chunk = int(chunk)
            self.overlap = int(overlap)
            self.seed = int(seed)
            self.epoch = int(epoch)
            self.with_targets = (split == "train") if with_targets is None else with_targets
            self.index: list[tuple[int, int, int]] = []      # (well_idx, start, length)
            self.well_lengths: dict[str, int] = {}
            for wi, w in enumerate(self.wells):
                n = int(self.reader.feature_matrix(w).shape[0])
                self.well_lengths[w] = n
                for s, L in chunks_for(n, self.chunk, self.overlap):
                    self.index.append((wi, s, L))
            rng = np.random.default_rng((self.seed, self.epoch))
            self.order = rng.permutation(len(self.index)).tolist()

        def set_epoch(self, epoch: int) -> None:
            """每个 epoch 开始时重算采样顺序；`--resume` 后可复现同一 epoch 的 order。"""
            self.epoch = int(epoch)
            rng = np.random.default_rng((self.seed, self.epoch))
            self.order = rng.permutation(len(self.index)).tolist()

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, i: int) -> dict[str, Any]:
            wi, s, L = self.index[self.order[i]]
            w = self.wells[wi]
            rec = self.reader.chunk(w, s, L)
            X = rec["X"]
            if self.scaler is not None:
                X = self.scaler.transform(X)
            item = {"X": torch.from_numpy(np.ascontiguousarray(X)),
                    "well_idx": torch.tensor(wi, dtype=torch.long),
                    "start": torch.tensor(rec["start"], dtype=torch.long)}
            if self.with_targets:
                for k in ("por", "perm_z", "sw", "mask", "y_atom", "y_joint"):
                    v = rec.get(k)
                    if v is not None:
                        item[k] = torch.from_numpy(np.ascontiguousarray(v))
            return item

        def coverage(self) -> dict[str, Any]:
            total = sum(len(chunks_for(n, self.chunk, self.overlap))
                        for n in self.well_lengths.values())
            ok = all(coverage_report(n, chunks_for(n, self.chunk, self.overlap))["complete"]
                     for n in self.well_lengths.values())
            return {"n_wells": len(self.wells), "n_chunks": total, "complete": bool(ok),
                    "chunk": self.chunk, "overlap": self.overlap,
                    "total_rows": int(sum(self.well_lengths.values()))}

        def order_digest(self) -> str:
            """采样顺序指纹（固定 seed/epoch 下必须稳定）。"""
            import hashlib
            h = hashlib.sha256()
            for i in self.order:
                wi, s, L = self.index[i]
                h.update(f"{self.wells[wi]}:{s}:{L};".encode())
            return h.hexdigest()[:16]

    def collate_chunks(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """把同长度的 chunk 拼成 batch（chunk 定长，因此可直接 stack）。"""
        out: dict[str, Any] = {}
        for k in items[0]:
            if k in ("start",):
                out[k] = torch.stack([it[k] for it in items])
            else:
                out[k] = torch.stack([it[k] for it in items])
        return out
