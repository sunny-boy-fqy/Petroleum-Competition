"""WP8 回归：类型井选择（纯 numpy）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.data import type_well as TW  # noqa: E402


def _shard(n=500, center=0.0, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    # 13 列：只在若干位置放有效曲线
    X = np.full((n, 13), np.nan, dtype="float64")
    cols = {"GR": 0, "AC": 4, "DEN": 5, "CNL": 6, "RT": 8, "RXO": 7}
    for j, c in enumerate(cols.values()):
        X[:, c] = center + scale * rng.normal(size=n)
    return {"well_id": "w", "inputs": X,
            "missing": (~np.isfinite(X)).astype("int8"),
            "depth": np.arange(n, dtype="float64")}


class TestTypeWell(unittest.TestCase):
    def test_selects_most_similar_well(self):
        test = _shard(center=0.0, scale=1.0, seed=0)
        train = {
            "near": _shard(center=0.05, scale=1.0, seed=1),
            "far": _shard(center=5.0, scale=1.0, seed=2),
        }
        res = TW.select_type_wells(test, train, topk=2, method="kl")
        self.assertEqual(res["candidates"][0]["well"], "near")

    def test_signature_method(self):
        a = TW.well_signature(_shard(center=0.0, seed=3))
        b = TW.well_signature(_shard(center=0.1, seed=4))
        self.assertEqual(a["mean"].shape, b["mean"].shape)

    def test_dtw_distance_identity_zero(self):
        x = np.linspace(0, 1, 100)
        self.assertLess(TW.dtw_distance(x, x), 1e-6)

    def test_batch(self):
        test = {"t1": _shard(seed=5)}
        train = {"a": _shard(seed=6), "b": _shard(center=3, seed=7)}
        out = TW.select_type_wells_batch(test, train, topk=1)
        self.assertIn("t1", out)
        self.assertEqual(len(out["t1"]["candidates"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
