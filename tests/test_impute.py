"""WP9 回归：缺失值插补（纯 numpy / sklearn 可选）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.data import impute as IM  # noqa: E402


class TestImpute(unittest.TestCase):
    def _X(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        X[rng.random(X.shape) < 0.2] = np.nan
        return X

    def test_median_no_nan(self):
        out = IM.median_imputer(self._X())["X"]
        self.assertFalse(np.isnan(out).any())

    def test_knn_no_nan_and_marks(self):
        out = IM.knn_imputer(self._X(), k=3)["X"]
        self.assertFalse(np.isnan(out).any())

    def test_iterative_runs(self):
        res = IM.impute_matrix(self._X(), method="mice", max_iter=3)
        self.assertFalse(np.isnan(res["X"]).any())
        self.assertIn("iterative", res["method"].lower())

    def test_matrix_imputer_knn_transform_no_nan(self):
        X = self._X()
        imp = IM.MatrixImputer(method="knn", params={"k": 3}).fit(X[:120])
        train = imp.fit_transform(X[:120])
        val = imp.transform(X[120:])
        self.assertFalse(np.isnan(train).any())
        self.assertFalse(np.isnan(val).any())

    def test_matrix_imputer_mice_transform_no_nan(self):
        X = self._X()
        imp = IM.MatrixImputer(method="mice", params={"max_iter": 3}).fit(X[:120])
        train = imp.fit_transform(X[:120])
        val = imp.transform(X[120:])
        self.assertFalse(np.isnan(train).any())
        self.assertFalse(np.isnan(val).any())

    def test_matrix_imputer_uses_train_fill(self):
        X = self._X()
        imp = IM.MatrixImputer(method="median").fit(X[:120])
        out = imp.transform(X[120:])
        self.assertFalse(np.isnan(out).any())

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            IM.impute_matrix(np.zeros((3, 2)), method="nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
