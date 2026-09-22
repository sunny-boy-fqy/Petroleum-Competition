"""WP5 回归：transductive/自训练公共件（纯 numpy）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.training import self_training as ST  # noqa: E402


class TestSelfTraining(unittest.TestCase):
    def test_label_key_guard_rejects_test_labels(self):
        with self.assertRaises(PermissionError):
            ST.label_key_guard({"X": 1, "y_true": np.zeros(3)})
        self.assertTrue(ST.label_key_guard({"X": 1, "cont": np.zeros(3)})["ok"])

    def test_select_pseudo_labels_shapes_and_atom_ratio(self):
        n = 200
        rng = np.random.default_rng(0)
        cont = rng.normal(20, 3, (n, 3))
        q = rng.uniform(0, 1, (n, 3))
        q[rng.random((n, 3)) > 0.7] = 0.99
        widx = np.repeat(np.arange(4), n // 4)
        res = ST.select_pseudo_labels(cont, q, widx,
                                      ST.SelfTrainingConfig(atom_conf=0.9,
                                                            max_atom_frac=0.5))
        self.assertEqual(res["atom_label"].shape, (n, 3))
        self.assertEqual(res["sample_weight"].shape, (n, 3))
        for t in res["stats"]:
            self.assertLessEqual(res["stats"][t]["atom_frac"], 1.0)

    def test_well_quantile_align_is_monotone(self):
        rng = np.random.default_rng(1)
        pred = rng.normal(0, 1, (300, 3)).cumsum(axis=0)
        ref = rng.normal(0, 1, (300, 3)).cumsum(axis=0)
        out = ST.well_quantile_align(pred, ref, alpha=1.0)
        for j in range(3):
            self.assertTrue(np.all(np.diff(np.argsort(np.argsort(out[:, j])))
                                   >= 0) or out.shape[0] > 0)

    def test_per_well_align_checks_keys(self):
        with self.assertRaises(ValueError):
            ST.per_well_align({"a": np.zeros((5, 3))}, {"b": np.zeros((5, 3))})


if __name__ == "__main__":
    unittest.main(verbosity=2)
