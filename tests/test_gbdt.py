"""WP11 回归：GBDT 成员（sklearn 可选）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.models import gbdt as GB  # noqa: E402


class TestGBDT(unittest.TestCase):
    def test_available_kinds_returns_dict(self):
        kinds = GB.available_kinds()
        for k in ("histgb", "lgbm", "xgboost", "catboost"):
            self.assertIn(k, kinds)

    @unittest.skipUnless(GB.available_kinds().get("histgb"), "sklearn not installed")
    def test_histgb_multi_target(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        Y = np.stack([X[:, 0] * 2 + 1, X[:, 1] - X[:, 2], X[:, 3] * 0.5], axis=1)
        m = np.ones_like(Y, dtype=bool)
        model = GB.fit_multi_target(X[:220], Y[:220], mask=m[:220],
                                    kind="histgb", params={"max_iter": 20})
        pred = model.predict(X[220:])
        self.assertEqual(pred.shape, (80, 3))
        self.assertTrue(np.isfinite(pred).all())

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            GB.make_gbdt("nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
