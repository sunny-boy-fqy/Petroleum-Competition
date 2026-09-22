"""WP10 回归：扩展岩石物理特征（纯 numpy）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.features import physics_ext as PE  # noqa: E402


def _inputs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.full((n, len(C.INPUT_COLUMNS)), np.nan)
    X[:, C.INPUT_COLUMNS.index("GR")] = rng.normal(60, 20, n)
    X[:, C.INPUT_COLUMNS.index("DEN")] = rng.normal(2.4, 0.15, n)
    X[:, C.INPUT_COLUMNS.index("CNL")] = rng.uniform(5, 45, n)
    X[:, C.INPUT_COLUMNS.index("AC")] = rng.normal(80, 15, n)
    X[:, C.INPUT_COLUMNS.index("RT")] = 10 ** rng.normal(1.0, 0.7, n)
    X[:, C.INPUT_COLUMNS.index("RXO")] = 10 ** rng.normal(0.8, 0.7, n)
    return X


class TestPhysicsExt(unittest.TestCase):
    def test_feature_shapes_and_ranges(self):
        X = _inputs()
        F, names = PE.build_petro_features(X)
        self.assertEqual(F.shape, (X.shape[0], len(PE.PETRO_FEATURES)))
        self.assertEqual(names, list(PE.PETRO_FEATURES))
        self.assertTrue(np.isfinite(F).all())
        for j, name in enumerate(names):
            if name.startswith(("vsh_", "phi_", "sw_")):
                self.assertGreaterEqual(float(np.nanmin(F[:, j])), -1e-6)
                self.assertLessEqual(float(np.nanmax(F[:, j])), 1.5)

    def test_vsh_monotone_in_gr(self):
        gr = np.linspace(10, 130, 50)
        v = PE.vsh_gr(gr, 20.0, 120.0)
        self.assertTrue(np.all(np.diff(v) >= -1e-12))
        self.assertAlmostEqual(float(v[0]), 0.0, places=6)
        self.assertAlmostEqual(float(v[-1]), 1.0, places=6)

    def test_fit_params_sane(self):
        X = _inputs()
        p = PE.fit_petro_params(X)
        self.assertLess(p.sand_line, p.shale_line)
        self.assertGreater(p.rw, 0.0)
        self.assertGreater(p.rsh, 0.0)

    def test_missing_propagates(self):
        X = _inputs(50)
        m = np.zeros_like(X, dtype=bool)
        m[0, C.INPUT_COLUMNS.index("GR")] = True
        F, _ = PE.build_petro_features(X, m)
        self.assertTrue(np.isnan(F[0]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
