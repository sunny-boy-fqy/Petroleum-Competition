"""Well parsing + npz cache. Frozen data contract for v5."""
from __future__ import annotations

import glob
import os

import numpy as np

DATA_ROOT = os.environ.get("V5_DATA", "/root/v4rt/v4/data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
CACHE_DIR = os.environ.get("V5_CACHE", "/data/v5/cache")

COLUMNS = ("DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT",
           "DEVI", "AZIM", "BIT", "CASE", "POR", "PERM", "SW")
INPUTS = COLUMNS[1:14]          # 13 curves
TARGETS = ("POR", "PERM", "SW")
N_IN = len(INPUTS)
PLACEHOLDER = np.array([0.1, 0.01, 99.9], dtype=np.float64)
MISSING_LT = -1000.0
TOL = 1e-9
EPS = 1e-3
W_POR, W_PERM, W_SW = 0.30, 0.35, 0.35
DELTA_POR, DELTA_SW = 0.08, 0.05


def read_well(path: str):
    """Return (well_id, header, values (n, len(header)) float64, depths)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    body = [ln for ln in lines[2:] if ln.strip()]
    ncol = len(header)
    arr = np.empty((len(body), ncol), dtype=np.float64)
    for i, ln in enumerate(body):
        parts = ln.split(",")
        for j in range(ncol):
            v = parts[j].strip()
            arr[i, j] = np.nan if v == "" else float(v)
    return os.path.basename(path)[:-4], header, arr


def canonicalise(header, arr):
    """Map arbitrary header order to the canonical 17 columns; extras dropped."""
    idx = {c: i for i, c in enumerate(header)}
    missing = [c for c in COLUMNS if c not in idx]
    if missing:
        raise ValueError(f"missing columns {missing}")
    cols = [arr[:, idx[c]] for c in COLUMNS]
    return np.stack(cols, axis=1).astype(np.float64)


def build_cache(force: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "wells.npz")
    if os.path.exists(path) and not force:
        return path
    store = {}
    for split, d in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        files = sorted(glob.glob(os.path.join(d, "*.txt")))
        ids, mats = [], []
        for p in files:
            wid, hdr, arr = read_well(p)
            mats.append(canonicalise(hdr, arr))
            ids.append(wid)
        store[f"{split}_ids"] = np.array(ids)
        store[f"{split}_lens"] = np.array([m.shape[0] for m in mats], dtype=np.int64)
        store[f"{split}_data"] = np.concatenate(mats, axis=0)
    np.savez(path, **store)
    return path


def load_cache():
    path = build_cache()
    z = np.load(path, allow_pickle=False)
    out = {}
    for split in ("train", "test"):
        lens = z[f"{split}_lens"]
        data = z[f"{split}_data"]
        ids = z[f"{split}_ids"]
        off = np.concatenate([[0], np.cumsum(lens)])
        out[split] = {
            "ids": [str(x) for x in ids],
            "lens": lens,
            "data": data,
            "off": off,
            "wells": [data[off[i]:off[i + 1]] for i in range(len(lens))],
        }
    return out


def targets_of(mat):
    """mat (n,17) -> y (n,3), missing (n,3) bool, is_ph (n,3) bool."""
    y = mat[:, 14:17].copy()
    missing = y < MISSING_LT
    is_ph = (~missing) & (np.abs(y - PLACEHOLDER) < TOL)
    return y, missing, is_ph


if __name__ == "__main__":
    p = build_cache(force=True)
    d = load_cache()
    print("cache:", p, "train wells", len(d["train"]["ids"]), "rows", d["train"]["data"].shape,
          "test wells", len(d["test"]["ids"]), "rows", d["test"]["data"].shape)
