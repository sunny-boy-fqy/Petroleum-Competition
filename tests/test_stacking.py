"""WP6 回归：二阶 stacking（纯 numpy）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.ensemble import stacking as ST  # noqa: E402


class TestStacking(unittest.TestCase):
    def test_ridge_stacker_learns_linear(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 3))
        y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5
        m = ST.RidgeStacker(l2=1e-6, nonneg=False).fit(X, y)
        pred = m.predict(X)
        self.assertLess(float(np.mean((pred - y) ** 2)), 1e-3)

    def test_ridge_nonneg_clips(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 2))
        y = -3.0 * X[:, 0] + 2.0 * X[:, 1]
        m = ST.RidgeStacker(l2=1e-6, nonneg=True).fit(X, y)
        self.assertTrue(np.all(m.coef_ >= 0))

    def test_logistic_stacker_separable(self):
        rng = np.random.default_rng(2)
        X = np.concatenate([rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))])
        y = np.concatenate([np.zeros(100), np.ones(100)])
        m = ST.LogisticStacker(l2=0.0, lr=0.5, iters=500).fit(X, y)
        auc = np.mean(m.predict_proba(X[y == 1]) > np.mean(m.predict_proba(X[y == 0])))
        self.assertEqual(auc, 1.0)

    def test_build_features_and_cv(self):
        rng = np.random.default_rng(3)
        members = {
            "a": {"por": rng.normal(size=200), "q_atom": rng.random((200, 3))},
            "b": {"por": rng.normal(size=200), "q_atom": rng.random((200, 3))},
        }
        names, X = ST.build_stacking_features(members)
        self.assertEqual(X.shape[0], 200)
        self.assertEqual(X.shape[1], len(names))
        y = X[:, 0] * 2 + 1
        folds = np.repeat(np.arange(4), 50)
        out = ST.cv_predict_stacker(X, y, folds, kind="ridge", l2=1e-6)
        self.assertTrue(np.isfinite(out["oof"]).all())
        self.assertEqual(len(out["folds"]), 4)

    def test_make_stacker_unknown(self):
        with self.assertRaises(ValueError):
            ST.make_stacker("nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
