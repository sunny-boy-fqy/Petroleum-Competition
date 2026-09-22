"""WP9 回归：异常权重 + KS 代表采样（纯 numpy）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.data import outliers as OUT  # noqa: E402
from src.validation import representative as REP  # noqa: E402


class TestOutliers(unittest.TestCase):
    def test_iqr_flags_extreme(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 4))
        X[0, 0] = 100.0
        bad = OUT.iqr_outlier_mask(X, factor=3.0)
        self.assertTrue(bad[0, 0])

    def test_sample_weights_downweight(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 3))
        X[0] = 50.0
        w, rep = OUT.sample_weights_from_outliers(X, method="iforest",
                                                  contamination=0.02, seed=0)
        self.assertEqual(w.shape, (300,))
        self.assertLess(float(w.min()), 1.0)
        self.assertGreaterEqual(float(w.min()), 0.2)
        self.assertIn("n_flagged", rep)


class TestRepresentative(unittest.TestCase):
    def test_ks_selects_extremes(self):
        X = np.array([[0.0], [1.0], [0.5], [0.51], [10.0]])
        idx = REP.kennard_stone(X, n_select=3)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[1], 4)

    def test_inner_split_disjoint(self):
        sig = {"a": [0.0, 0.0], "b": [1.0, 1.0], "c": [0.5, 0.5],
               "d": [10.0, 10.0], "e": [5.0, 5.0]}
        tr, va = REP.representative_inner_split(list(sig), sig, n_val=2)
        self.assertEqual(set(tr) & set(va), set())
        self.assertEqual(len(tr) + len(va), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
