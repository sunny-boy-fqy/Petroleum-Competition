"""原子头 AUC/AP 测试（**口径层，无 torch**）：E6/P0 要求**逐目标**上报，不是只报总分。

判据来源（E6/P0 §7）：`min_auc ≥ 0.97`、`min_atom_acc ≥ 0.99`、`min_atom_recall ≥ 0.98`。
AUC 用平均秩实现（不依赖 sklearn/scipy），**并列取均值**；单类标签返回 `None`
而不是 0.5——否则"没有正样本"会被伪装成"随机水平"。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.training.metrics import (_average_ranks, auc_report,  # noqa: E402
                                  average_precision, binary_auc)


class TestBinaryAuc(unittest.TestCase):
    def test_perfect_and_inverted(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(binary_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0, places=9)
        self.assertAlmostEqual(binary_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0, places=9)

    def test_all_ties_is_half(self):
        y = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(binary_auc(y, np.zeros(4)), 0.5, places=9)

    def test_partial_ties_average_rank(self):
        y = np.array([0, 1, 0, 1])
        s = np.array([1.0, 1.0, 2.0, 3.0])
        # 4 个正负对：并列 1 对计 0.5，正确 2 对计 1，反向 1 对计 0 → 2.5/4
        expect = (0.5 + 0.0 + 1.0 + 1.0) / 4.0
        self.assertAlmostEqual(binary_auc(y, s), expect, places=9)

    def test_single_class_returns_none(self):
        self.assertIsNone(binary_auc(np.ones(5, dtype=bool), np.arange(5.0)))
        self.assertIsNone(binary_auc(np.zeros(5, dtype=bool), np.arange(5.0)))

    def test_empty_or_shape_mismatch_returns_none(self):
        self.assertIsNone(binary_auc(np.zeros(0, dtype=bool), np.zeros(0)))
        self.assertIsNone(binary_auc(np.zeros(3, dtype=bool), np.zeros(2)))

    def test_average_ranks_ties(self):
        r = _average_ranks(np.array([5.0, 5.0, 1.0]))
        self.assertTrue(np.allclose(r, [2.5, 2.5, 1.0]))


class TestAveragePrecision(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        y = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(average_precision(y, np.array([0.9, 0.8, 0.2, 0.1])), 1.0,
                               places=9)

    def test_no_positives_returns_none(self):
        self.assertIsNone(average_precision(np.zeros(4, dtype=bool), np.arange(4.0)))

    def test_ap_between_auc_like_bounds(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        s = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        ap = average_precision(y, s)
        self.assertGreater(ap, 0.0)
        self.assertLessEqual(ap, 1.0)


class TestAucReport(unittest.TestCase):
    def test_per_target_and_joint_keys(self):
        rng = np.random.RandomState(0)
        n = 200
        y = np.zeros((n, 3), dtype=bool)
        y[:80, :] = True                     # 前 80 行三目标同时原子
        q = 0.15 + 0.70 * y.astype("float64") + 0.05 * rng.rand(n, 3)
        q = np.clip(q, 0.0, 1.0)
        rep = auc_report(y, q)
        self.assertEqual(set(rep["per_target"]), {"POR", "PERM", "SW", "joint"})
        for name in ("POR", "PERM", "SW", "joint"):
            self.assertGreater(rep["per_target"][name]["auc"], 0.99)
            self.assertIsNotNone(rep["per_target"][name]["average_precision"])
        self.assertGreater(rep["min_auc"], 0.99)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            auc_report(np.zeros((4, 3), dtype=bool), np.zeros((3, 3)))

    def test_min_auc_none_when_single_class(self):
        rep = auc_report(np.ones((5, 3), dtype=bool), np.random.rand(5, 3))
        self.assertIsNone(rep["min_auc"])


if __name__ == "__main__":
    unittest.main()
