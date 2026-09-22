"""WP8 回归：井间分布匹配（纯 numpy）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.data import well_adapt as WA  # noqa: E402


class TestWellAdapt(unittest.TestCase):
    def test_linear_match_makes_stats_close(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(10, 2, (500, 3))
        src = rng.normal(-5, 0.5, (400, 3))
        out = WA.linear_match(src, ref, alpha=1.0)
        for j in range(3):
            self.assertAlmostEqual(float(out[:, j].mean()), float(ref[:, j].mean()), places=1)
            self.assertAlmostEqual(float(out[:, j].std()), float(ref[:, j].std()), places=1)

    def test_quantile_match_preserves_order(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, (500, 2))
        src = np.sort(rng.normal(5, 3, (300, 2)), axis=0)
        out = WA.quantile_match(src, ref, alpha=1.0)
        for j in range(2):
            self.assertTrue(np.all(np.diff(out[:, j]) >= -1e-9))

    def test_alpha_zero_returns_source(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(size=(100, 2))
        src = rng.normal(size=(100, 2))
        out = WA.match_matrix(src, ref, method="linear", alpha=0.0)
        self.assertTrue(np.allclose(out, src))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            WA.match_matrix(np.zeros((3, 2)), np.zeros((3, 2)), method="nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
