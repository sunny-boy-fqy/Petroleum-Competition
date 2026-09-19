"""评分器测试（官方公式边界 + 两种分母口径 + 总分恒等式）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C           # noqa: E402
from src.score import (acc_perm, acc_relative, constant_prediction,  # noqa: E402
                       score_arrays)


class TestRelativeAcc(unittest.TestCase):
    def test_perfect_prediction(self):
        y = np.array([0.1, 0.2, 10.0])
        self.assertAlmostEqual(acc_relative(y, y, C.DELTA_POR), 1.0, places=12)

    def test_exact_tolerance_boundary(self):
        # |ŷ-y| = δ(|y|+eps) -> 得分应 ≈ 0（带 eps 修正，略大于 0）
        y = np.array([0.2])
        yhat = np.array([0.2 + C.DELTA_POR * (0.2 + C.EPS)])
        s = acc_relative(y, yhat, C.DELTA_POR)
        self.assertLess(s, 1e-6)
        self.assertGreaterEqual(s, 0.0)

    def test_clipped_at_zero(self):
        y = np.array([0.2])
        yhat = np.array([5.0])
        self.assertEqual(acc_relative(y, yhat, C.DELTA_POR), 0.0)

    def test_half_tolerance_scores_half(self):
        y = np.array([1.0])
        yhat = np.array([1.0 + 0.5 * C.DELTA_POR * (1.0 + C.EPS)])
        self.assertAlmostEqual(acc_relative(y, yhat, C.DELTA_POR), 0.5, places=5)


class TestPermAcc(unittest.TestCase):
    def test_exact(self):
        y = np.array([1.0, 0.01, 100.0])
        self.assertAlmostEqual(acc_perm(y, y), 1.0, places=12)

    def test_ten_times_off_scores_zero(self):
        y = np.array([1.0])
        self.assertEqual(acc_perm(y, np.array([10.0])), 0.0)
        self.assertEqual(acc_perm(y, np.array([0.1])), 0.0)

    def test_log_symmetric(self):
        # 高估与低估同倍数应得同分
        y = np.array([1.0])
        up = acc_perm(y, np.array([3.0]))
        dn = acc_perm(y, np.array([1.0 / 3.0]))
        self.assertAlmostEqual(up, dn, places=10)


class TestMissingModes(unittest.TestCase):
    def test_drop_vs_mask(self):
        y = np.array([[0.1, 0.01, 99.9],
                      [0.1, 0.01, 99.9],
                      [-99999.0, -99999.0, -99999.0]])
        m = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=bool)
        p = constant_prediction(3)
        drop = score_arrays(y, p, m, missing_mode="drop")
        mask = score_arrays(y, p, m, missing_mode="mask")
        self.assertAlmostEqual(drop["total"], 100.0, places=6)   # 有效行全中
        self.assertAlmostEqual(mask["total"], 100.0 * 2 / 3, places=6)

    def test_frozen_mode_is_drop(self):
        self.assertEqual(C.SCORE_MISSING_MODE, "drop")

    def test_total_identity(self):
        """总分必须等于 100*Σw·acc（候选注册表的一致性不变式）。"""
        rng = np.random.default_rng(0)
        y = rng.uniform(0.05, 30, size=(500, 3))
        y[:, 2] = rng.uniform(20, 100, size=500)
        p = y + rng.normal(0, 0.5, size=(500, 3))
        r = score_arrays(y, p)
        want = 100.0 * (C.SCORE_WEIGHTS["POR"] * r["acc_por"]
                        + C.SCORE_WEIGHTS["PERM"] * r["acc_perm"]
                        + C.SCORE_WEIGHTS["SW"] * r["acc_sw"])
        self.assertAlmostEqual(r["total"], want, places=10)


class TestConstantBaselineAnchor(unittest.TestCase):
    """用真实数据复算常数基线，必须命中 70.4907（本机可跑，不需 torch）。"""

    def test_anchor(self):
        try:
            import sys as _s
            from src.data import labels as _L
            from src.data import parse as _P
            train = sorted((V4.parent / "data" / "train").glob("*.txt"))
            if not train:
                self.skipTest("dataset not found")
            recs = [_P.parse_well(f, with_targets=True) for f in train]
            y = np.concatenate([r.targets for r in recs], axis=0).astype("float64")
            m = _L.missing_masks(y)
            r = score_arrays(y, constant_prediction(y.shape[0]), m, missing_mode="drop")
            self.assertAlmostEqual(r["total"], C.CONSTANT_BASELINE_OOF, delta=1e-4)
            # mask 口径应显著更低（约 0.65 分）
            rm = score_arrays(y, constant_prediction(y.shape[0]), m, missing_mode="mask")
            self.assertLess(rm["total"], r["total"] - 0.5)
        except ImportError:
            self.skipTest("numpy unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
