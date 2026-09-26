"""Quick data contract exploration (E0 equivalent)."""
from __future__ import annotations
import glob, os, sys
import numpy as np

DATA = "/root/v4rt/v4/data"
TRAIN = os.path.join(DATA, "train")
TEST = os.path.join(DATA, "test")
COLUMNS = ("DEPTH","GR","PE","SP","CAL","AC","DEN","CNL","RXO","RT","DEVI","AZIM","BIT","CASE","POR","PERM","SW")
PLACEHOLDER = {"POR":0.1,"PERM":0.01,"SW":99.9}


def read_well(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    # unit row may have ragged spacing -> split by comma anyway
    body = [ln for ln in lines[2:] if ln.strip()]
    arr = np.empty((len(body), len(header)), dtype=np.float64)
    for i, ln in enumerate(body):
        parts = ln.split(",")
        for j in range(len(header)):
            v = parts[j].strip()
            arr[i, j] = np.nan if v == "" else float(v)
    return header, arr


def main():
    tw = sorted(glob.glob(os.path.join(TRAIN, "*.txt")))
    ew = sorted(glob.glob(os.path.join(TEST, "*.txt")))
    print(f"train wells={len(tw)} test wells={len(ew)}")
    schemas = {}
    rows = 0
    ph_joint = 0
    ph_each = np.zeros(3, dtype=int)
    valid_each = np.zeros(3, dtype=int)
    missing_each = np.zeros(3, dtype=int)
    sentinel_counts = np.zeros(17, dtype=int)
    per_well = []
    stats = {t: [] for t in ("POR","PERM","SW")}
    for p in tw:
        hdr, arr = read_well(p)
        schemas[tuple(hdr)] = schemas.get(tuple(hdr), 0) + 1
        idx = {c: i for i, c in enumerate(hdr)}
        rows += arr.shape[0]
        y = np.stack([arr[:, idx[t]] for t in ("POR","PERM","SW")], axis=1)
        miss = y < -1000
        missing_each += miss.sum(axis=0)
        valid = ~miss
        ph = np.zeros_like(valid)
        for j, t in enumerate(("POR","PERM","SW")):
            ph[:, j] = valid[:, j] & (np.abs(y[:, j] - PLACEHOLDER[t]) < 1e-9)
        ph_joint += int((ph.all(axis=1)).sum())
        for j in range(3):
            ph_each[j] += int(ph[:, j].sum())
            v = valid[:, j] & ~ph[:, j]
            valid_each[j] += int(v.sum())
            if v.any():
                stats[("POR","PERM","SW")[j]].append(y[v, j])
        sc = (arr < -1000).sum(axis=0)
        for j, c in enumerate(hdr):
            sentinel_counts[COLUMNS.index(c) if c in COLUMNS else 16] += int(sc[j])
        per_well.append((os.path.basename(p)[:-4], arr.shape[0], int((ph.all(axis=1)).sum()),
                         float(np.isnan(arr).sum())))
    print("schemas:", {len(k): v for k, v in schemas.items()})
    for k, v in schemas.items():
        if len(k) != 17:
            print("  nonstandard:", v, k)
    print("total train rows", rows)
    print("joint placeholder rows", ph_joint, f"{ph_joint/rows:.4f}")
    print("per-target placeholder", ph_each, "valid", valid_each, "missing", missing_each)
    # is placeholder always joint?
    for t in ("POR","PERM","SW"):
        v = np.concatenate(stats[t])
        v = v[np.isfinite(v)]
        qs = np.percentile(v, [0, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 100])
        print(f"valid {t}: n={len(v)} min={v.min():.4f} q={np.round(qs,4)}")
    print("sentinel(<-1000) per column:", dict(zip(COLUMNS, sentinel_counts.tolist())))
    print("test:")
    trows = 0
    for p in ew:
        hdr, arr = read_well(p)
        print("  ", os.path.basename(p)[:-4], arr.shape, len(hdr), "nan", int(np.isnan(arr).sum()))
        trows += arr.shape[0]
    print("test rows", trows)


if __name__ == "__main__":
    main()
