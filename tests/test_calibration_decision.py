"""WP1 回归：原子概率校准 + 期望分数决策（纯 numpy，不需要 torch）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.inference import atom_decision as AD  # noqa: E402
from src.inference import calibration as CAL  # noqa: E402


class TestCalibration(unittest.TestCase):
    def test_temperature_reduces_nll_on_overconfident_logits(self):
        rng = np.random.default_rng(0)
        n = 4000
        p = rng.uniform(0.05, 0.95, n)
        y = (rng.random(n) < p).astype("float64")
        logits = CAL.logit(p) * 4.0          # 过度自信
        raw = CAL.bce_with_logits(logits, y)
        fit = CAL.fit_temperature(logits, y)
        self.assertGreater(fit["temperature"], 1.0)
        self.assertLess(fit["nll"], raw)

    def test_isotonic_binned_is_monotone_and_applies(self):
        rng = np.random.default_rng(1)
        n = 2000
        p = rng.uniform(0.0, 1.0, n)
        y = (rng.random(n) < p).astype("float64")
        fit = CAL.fit_isotonic_binned(p, y, n_bins=10)
        vals = np.asarray(fit["bin_values"])
        self.assertTrue(np.all(np.diff(vals) >= -1e-12))
        out = CAL.apply_isotonic_binned([0.1, 0.5, 0.9], fit)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(np.diff(out) >= -1e-12))

    def test_reliability_report(self):
        rng = np.random.default_rng(2)
        y = (rng.random(1000) < 0.3).astype(float)
        q = np.where(y > 0, 0.9, 0.1)
        rep = CAL.reliability_report(q, y, n_bins=5)
        self.assertLess(rep["ece"], 0.11)
        self.assertEqual(rep["n"], 1000)


def _decision_dataset(n: int = 1000, seed: int = 3):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.0, 1.0, (n, 3))
    atom = q > 0.8
    y = np.empty((n, 3), dtype="float64")
    cont = np.empty((n, 3), dtype="float64")
    valid = np.asarray([20.0, 10.0, 50.0])
    atoms = np.asarray([0.1, 0.01, 99.9])
    for j in range(3):
        y[:, j] = np.where(atom[:, j], atoms[j], valid[j])
        # 连续值：非原子行精确命中；原子行给一个会丢分的连续预测
        cont[:, j] = np.where(atom[:, j], valid[j], valid[j])
    mask = np.ones((n, 3), dtype="float64")
    return cont, q, y, mask


class TestExpectedScoreDecision(unittest.TestCase):
    def test_high_q_bins_choose_atom(self):
        cont, q, y, mask = _decision_dataset()
        table = AD.fit_decision_table(cont, q, y, mask, n_bins=10, min_bin=10,
                                      min_gain=0.0)
        pred, actions = AD.apply_decision_table(cont, q, table)
        high = q > 0.9
        low = q < 0.7
        # 高 q 桶应切 atom（至少 3 个目标里大部分目标）
        self.assertGreater(actions[high].mean(), 0.5)
        # 低 q 桶应保持连续
        self.assertLess(actions[low].mean(), 0.1)
        # 切 atom 的列必须精确等于原子值
        for j, v in enumerate([0.1, 0.01, 99.9]):
            sel = actions[:, j]
            if sel.any():
                self.assertTrue(np.allclose(pred[sel, j], v, atol=1e-12))

    def test_no_table_falls_back_to_continuous(self):
        cont, q, _, _ = _decision_dataset()
        pred, actions = AD.apply_decision_table(cont, q, {})
        self.assertTrue(np.array_equal(pred, cont))
        self.assertFalse(actions.any())

    def test_min_gain_blocks_zero_gain_switch(self):
        # delta 恒为 0：min_gain>0 时不得切 atom
        n = 200
        rng = np.random.default_rng(5)
        q = rng.uniform(0, 1, (n, 3))
        y = np.tile([0.1, 0.01, 99.9], (n, 1))
        cont = y.copy()
        mask = np.ones((n, 3))
        table = AD.fit_decision_table(cont, q, y, mask, n_bins=4, min_bin=1,
                                      min_gain=1e-6)
        _, actions = AD.apply_decision_table(cont, q, table)
        self.assertFalse(actions.any())

    def test_object_roundtrip(self):
        cont, q, y, mask = _decision_dataset(n=400)
        obj = AD.ExpectedScoreDecision().fit(cont, q, y, mask, n_bins=5, min_bin=5)
        d = obj.to_dict()
        obj2 = AD.ExpectedScoreDecision.from_dict(d)
        p1, a1 = obj.predict(cont, q)
        p2, a2 = obj2.predict(cont, q)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertTrue(np.array_equal(a1, a2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
