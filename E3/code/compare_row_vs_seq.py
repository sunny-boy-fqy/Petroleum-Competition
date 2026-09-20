#!/usr/bin/env python3
"""E3/P2：**行级 vs 序列**的受控对照（同折同头同超参 + 配对井级 cluster bootstrap）。

输入是两份 OOF：序列的 `$RUN/E3/oof_<tag>.npz` 与行级的 `$RUN/E1/oof.npz`
（E1 用同一套头与损失，只是主干换成 MLP）。判据（E3/P2 §6）：加权配对 bootstrap
95% CI **下界 > 0** 才算"序列显著优于行级"。

用法::

    python3 E3/code/compare_row_vs_seq.py --seq-oof /tmp/run/E3/oof_tcn.npz \\
        --row-oof /tmp/run/E1/oof.npz --reports-dir /tmp/reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.score import score_arrays  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _well_totals(oof: dict) -> tuple["np.ndarray", "np.ndarray", list[str]]:
    """逐井 Total（按**井名**对齐，因此两份 OOF 必须覆盖同样的井）。"""
    yt, yp = oof["y_true"], oof["y_pred"]
    m = oof["mask"] >= 0.5
    ids = [str(x) for x in list(oof["well_ids"])]
    wi = oof["well_index"]
    tot = np.full(len(ids), np.nan)
    rows = np.zeros(len(ids), dtype="int64")
    for i in range(len(ids)):
        sel = wi == i
        rows[i] = int(sel.sum())
        if rows[i]:
            tot[i] = score_arrays(yt[sel], yp[sel], missing=~m[sel],
                                  missing_mode=C.SCORE_MISSING_MODE)["total"]
    return tot, rows, ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 E3 row-vs-seq controlled comparison")
    ap.add_argument("--seq-oof", required=True)
    ap.add_argument("--row-oof", required=True)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args(argv)

    seq, row = _load(Path(args.seq_oof)), _load(Path(args.row_oof))
    s_tot, s_rows, s_ids = _well_totals(seq)
    r_tot, r_rows, r_ids = _well_totals(row)
    common = sorted(set(s_ids) & set(r_ids))
    if not common:
        print("[cmp] FATAL: 两份 OOF 没有共同井（折协议不一致？）", file=sys.stderr)
        return 4
    si = {w: i for i, w in enumerate(s_ids)}
    ri = {w: i for i, w in enumerate(r_ids)}
    d = np.array([s_tot[si[w]] - r_tot[ri[w]] for w in common], dtype="float64")
    w = np.array([s_rows[si[w]] for w in common], dtype="float64")
    boot = FOLDS.bootstrap_ci(d, iters=int(args.iters), weights=w, seed=42)
    s_all = M.score_of(seq["y_true"], seq["y_pred"], seq["mask"])
    r_all = M.score_of(row["y_true"], row["y_pred"], row["mask"])
    out = {
        "stage": "E3", "kind": "row_vs_seq",
        "n_common_wells": len(common),
        "seq_oof_total": float(s_all["total"]), "row_oof_total": float(r_all["total"]),
        "delta": float(s_all["total"] - r_all["total"]),
        "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
        "point_delta": float(boot["point"]),
        "significant": bool(boot["ci_low"] > 0),
        "decision": ("序列显著优于同头行级（CI 下界 > 0）" if boot["ci_low"] > 0
                     else "未达显著：保留行级为基线，序列仅作为候选（不得宣称增益）"),
        "per_well_delta": {w_: float(s_tot[si[w_]] - r_tot[ri[w_]]) for w_ in common},
        "note": "受控对照：同一份头/损失/折协议，只换主干（MLP vs U-Net/TCN）",
    }
    if args.reports_dir:
        p = Path(args.reports_dir) / "E3_row_vs_seq.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(M.jsonable(out), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_well_delta"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
