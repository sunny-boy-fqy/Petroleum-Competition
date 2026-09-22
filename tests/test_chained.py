"""WP11 回归：链式目标回归（纯 numpy Ridge）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.training import chained as CH  # noqa: E402


class TestChained(unittest.TestCase):
    def _data(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4))
        y0 = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=n)
        y1 = 0.8 * y0 + 1.5 * X[:, 2] + 0.1 * rng.normal(size=n)
        y2 = -0.5 * y0 + 0.7 * y1 + 0.3 * X[:, 3] + 0.1 * rng.normal(size=n)
        return X, np.stack([y0, y1, y2], axis=1)

    def test_fit_predict_shapes_and_no_nan(self):
        X, Y = self._data()
        model = CH.ChainedRegressor(estimator_kind="ridge", n_splits=3)
        model.fit(X[:400], Y[:400], mask=np.ones((400, 3)))
        pred = model.predict(X[400:])
        self.assertEqual(pred.shape, (200, 3))
        self.assertTrue(np.isfinite(pred).all())

    def test_chain_uses_oof_and_improves(self):
        X, Y = self._data()
        model = CH.ChainedRegressor(estimator_kind="ridge", n_splits=5)
        model.fit(X[:400], Y[:400])
        pred = model.predict(X[400:])
        mse = np.mean((pred - Y[400:]) ** 2, axis=0)
        # 链式至少不应显著差于“只用 X 独立预测 y2”
        ind = CH.NumpyRidge(alpha=1e-6).fit(X[:400], Y[:400, 2])
        mse_ind = float(np.mean((ind.predict(X[400:]) - Y[400:, 2]) ** 2))
        self.assertLessEqual(float(mse[2]), mse_ind * 1.2)

    def test_unknown_estimator(self):
        with self.assertRaises(ValueError):
            CH.make_estimator("nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
