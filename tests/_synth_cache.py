"""合成井数据 + 缓存构造（测试专用；文件名不以 `test_` 开头，不会被 unittest 收集）。

用途：让训练层测试**完全不依赖真实数据**也能端到端跑通（本机 CPU、云端 GPU 都一样），
同时构造出 E1 明确要求的边界切片：`POR == 0`、`POR < 0.1`、占位行、全缺测行、
输入曲线缺测（哨兵 -99999）。
"""
from __future__ import annotations

import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402


def well_csv(n_rows: int = 40, seed: int = 0, depth0: float = 1000.0,
             missing_input_frac: float = 0.1, atom_frac: float = 0.45,
             all_missing_frac: float = 0.05) -> str:
    """生成一口井的 CSV 文本（表头 + 单位行 + 数据行，与真实数据同构）。"""
    rng = np.random.default_rng(seed)
    lines = [",".join(C.COLUMNS), ",".join(["m"] + ["none"] * 16)]
    for i in range(n_rows):
        depth = depth0 + 0.1 * i
        gr = 30.0 + 0.5 * i + rng.normal(0, 0.5)
        pe = 2.0 + 0.01 * i
        sp = 70.0 + 0.2 * i
        cal = 20.0 + 0.05 * i
        ac = 240.0 - 0.3 * i
        den = 2.3 + 0.001 * i
        cnl = 15.0 + 0.1 * i
        rxo, rt = 50.0, 90.0
        devi, azim = 0.2, 45.0
        bit, case = 21.59, -99999.0
        curves = [gr, pe, sp, cal, ac, den, cnl, rxo, rt, devi, azim, bit, case]
        # 输入缺测：哨兵 -99999（parse 会转成 NaN 并置缺失位）
        for j in range(len(curves)):
            if rng.random() < missing_input_frac and j not in (0,):
                curves[j] = -99999.0
        if i % 13 == 5:                       # POR 恰好为 0 的行（E1 切片判据）
            por, perm, sw = 0.0, 0.05, 20.0
        elif i % 13 == 7:                     # POR < 0.1 的行
            por, perm, sw = 0.05, 0.2, 12.0
        elif rng.random() < all_missing_frac:  # 三目标全缺
            por, perm, sw = -99999.0, -99999.0, -99999.0
        elif rng.random() < atom_frac:        # 联合占位
            por, perm, sw = C.PLACEHOLDER["POR"], C.PLACEHOLDER["PERM"], C.PLACEHOLDER["SW"]
        else:                                  # 有效连续值（与 GR 相关，可学习）
            por = float(np.clip(5.0 + 0.02 * gr, 0.2, 30.0))
            perm = float(np.clip(10 ** (-1.0 + 0.05 * gr), 1e-3, 1e3))
            sw = float(np.clip(40.0 + 0.5 * gr, C.SW_VALID_MIN, 99.0))
        vals = [depth] + curves + [por, perm, sw]
        lines.append(",".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals))
    return "\n".join(lines) + "\n"


def write_dataset(root: str | Path, wells, n_rows: int = 40, seed: int = 0,
                  split: str = "train") -> Path:
    """把合成井写进 `<root>/<split>/<well>.txt`，返回该目录。"""
    d = Path(root) / split
    d.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(wells):
        (d / f"{w}.txt").write_text(well_csv(n_rows=n_rows, seed=seed + i * 977),
                                    encoding="utf-8")
    return d


def build_cache(cache_root: str | Path, wells, n_rows: int = 40, seed: int = 0):
    """写数据集 → 建 raw/labels 分片 → 建 row 分片；返回 (train_dir, test_dir, cache_root)。"""
    from src.data import dataset as D
    from src.data import row_dataset as RD

    root = Path(cache_root).parent / (Path(cache_root).name + "_src")
    tr = write_dataset(root, wells, n_rows=n_rows, seed=seed, split="train")
    te = write_dataset(root, wells, n_rows=max(n_rows // 2, 5), seed=seed, split="test")
    D.build_cache(tr, te, cache_root, verbose=False)
    RD.build_row_cache(cache_root, list(wells), "train", verbose=False)
    return tr, te, Path(cache_root)


def fold0_wells(n_val: int = 8, n_train: int = 8):
    """取真实折文件里 fold0 的前若干验证井/训练井（与 `--max-wells` 的切片语义一致）。"""
    from src.validation import folds as FOLDS

    f = FOLDS.load_folds()
    tr = [w for w in f["well_list"] if f["fold_of_well"][w] != 0]
    va = [w for w in f["well_list"] if f["fold_of_well"][w] == 0]
    return tr[:n_train], va[:n_val]
