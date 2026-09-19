"""原子门测试（**纯 numpy，无 torch，任何环境都应运行**）。

覆盖：逐目标硬切换（无插值、精确原子值、逐目标独立）、联合守卫开关、
`select_tau_per_target` 的官方加权目标 + 平台中点规则、极端原子概率下 τ 复现原子值、
以及误判代价分解。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.inference.atomic_gate import (  # noqa: E402
    _longest_plateau_midpoint,
    joint_guard,
    misclassification_cost_report,
    per_target_hard_switch,
    select_tau_per_target,
)
from src.portability import HAS_NUMPY  # noqa: E402

if HAS_NUMPY:
    import numpy as np


def _official_acc_por(y, pred, mask):
    """POR 官方逐目标准确率：|pred − y| < 0.08·(|y| + 1e-3)。"""
    m = mask > 0
    y, pred = y[m], pred[m]
    if y.size == 0:
        return 0.0
    err = np.abs(pred - y) / (0.08 * (np.abs(y) + 1e-3))
    return float(np.mean(err < 1.0))


def _official_acc_perm(y, pred, mask):
    """PERM 官方逐目标准确率：|log10(max(pred/y, 1e-3))| < 1。"""
    m = mask > 0
    y, pred = y[m], pred[m]
    if y.size == 0:
        return 0.0
    ratio = np.maximum(pred / np.maximum(y, 1e-12), 1e-3)
    return float(np.mean(np.abs(np.log10(ratio)) < 1.0))


def _official_acc_sw(y, pred, mask):
    """SW 官方逐目标准确率：|pred − y| < 0.05·(|y| + 1e-3)。"""
    m = mask > 0
    y, pred = y[m], pred[m]
    if y.size == 0:
        return 0.0
    err = np.abs(pred - y) / (0.05 * (np.abs(y) + 1e-3))
    return float(np.mean(err < 1.0))


# 逐目标官方口径（POR/SW 相对误差、PERM 对数误差）——单元测试传给 select_tau 的三元组
_SCORES = (_official_acc_por, _official_acc_perm, _official_acc_sw)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestHardSwitch(unittest.TestCase):
    def test_exact_atom_values_and_per_target_independence(self):
        cont = np.array([[5.0, 0.5, 60.0],
                         [30.0, 1e-4, 90.0],
                         [1.0, 2.0, 12.0]])
        q = np.array([[0.9, 0.1, 0.1],
                      [0.1, 0.95, 0.1],
                      [0.1, 0.1, 0.99]])
        out = per_target_hard_switch(cont, q, tau=0.5)
        # 只有对角命中，逐目标独立
        self.assertAlmostEqual(out[0, 0], C.ATOM_VALUES["POR"])
        self.assertAlmostEqual(out[1, 1], C.ATOM_VALUES["PERM"])
        self.assertAlmostEqual(out[2, 2], C.ATOM_VALUES["SW"])
        # 未命中列逐位等于连续值
        np.testing.assert_array_equal(out[1, 0], cont[1, 0])
        np.testing.assert_array_equal(out[0, 1], cont[0, 1])
        np.testing.assert_array_equal(out[0, 2], cont[0, 2])

    def test_no_interpolation_anywhere(self):
        rng = np.random.default_rng(0)
        cont = rng.normal(size=(200, 3)) * 10.0 + 50.0
        q = rng.random((200, 3))
        tau = np.array([0.3, 0.5, 0.7])
        out = per_target_hard_switch(cont, q, tau)
        hit = q > tau[None, :]
        av = np.array([C.ATOM_VALUES[t] for t in C.TARGETS])
        for j in range(3):
            np.testing.assert_array_equal(out[hit[:, j], j],
                                          np.full(int(hit[:, j].sum()), av[j]))
            miss = ~hit[:, j]
            np.testing.assert_array_equal(out[miss, j], cont[miss, j])
        # 断言不可能出现"介于两者之间"的插值
        mixed = (out != cont) & (out != av[None, :])
        self.assertFalse(bool(mixed.any()))

    def test_tau_validation(self):
        cont = np.zeros((2, 3)); q = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            per_target_hard_switch(cont, q, tau=[0.1, 0.2])
        with self.assertRaises(ValueError):
            per_target_hard_switch(cont, np.zeros((3, 2)), tau=0.5)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestJointGuard(unittest.TestCase):
    def setUp(self):
        self.cont = np.array([[5.0, 0.5, 60.0], [8.0, 0.8, 70.0]])
        self.q = np.array([0.99, 0.10])

    def test_off(self):
        out = joint_guard(self.cont, self.q, tau_high=0.999)
        np.testing.assert_array_equal(out, self.cont)

    def test_on_overwrites_all_three(self):
        out = joint_guard(self.cont, self.q, tau_high=0.5)
        np.testing.assert_array_equal(out[0], [C.ATOM_VALUES[t] for t in C.TARGETS])
        np.testing.assert_array_equal(out[1], self.cont[1])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestSelectTau(unittest.TestCase):
    def _por_dataset(self):
        """一半原子行（q=0.9）、一半非原子行（q=0.2）——天然形成 [0.2, 0.9) 平台。"""
        n = 200
        cont = np.empty((n, 3))
        y = np.empty((n, 3))
        q = np.full((n, 3), 0.5)
        mask = np.ones((n, 3))
        y[: n // 2, 0] = C.ATOM_VALUES["POR"]
        cont[: n // 2, 0] = 11.0            # 连续头在原子行上偏离
        y[n // 2:, 0] = 20.0
        cont[n // 2:, 0] = 20.0
        q[: n // 2, 0] = 0.9
        q[n // 2:, 0] = 0.2
        # PERM/SW 设为"连续头已完美"，原子概率低
        for j, (atom, good) in enumerate(
                ((C.ATOM_VALUES["PERM"], 1.0), (C.ATOM_VALUES["SW"], 80.0)), start=1):
            y[:, j] = good
            cont[:, j] = good
            q[:, j] = 0.01
        return cont, q, y, mask

    def test_plateau_midpoint_and_official_objective(self):
        cont, q, y, mask = self._por_dataset()
        res = select_tau_per_target(_SCORES, cont, q, y, mask)
        # POR 平台覆盖 tau ∈ [0.2, 0.9)，最长平台中点应落在内部（既不是下沿也不是上沿）
        lo, hi = res["plateau"]["POR"]
        self.assertGreaterEqual(lo, 0.15)
        self.assertLessEqual(hi, 0.9)
        tau_por = float(res["tau"][0])
        self.assertGreater(tau_por, lo)
        self.assertLess(tau_por, hi)
        # 目标函数 = 官方加权贡献 ~ 1.0（POR 全对 + PERM/SW 全对）
        self.assertAlmostEqual(res["objective"], 1.0, places=6)
        # 曲线长度 = 网格长度
        self.assertEqual(len(res["curve"]["POR"]), 19)
        # 官方权重确实被使用（POR 权重 0.30）
        acc_at_tau = dict(res["curve"]["POR"])[tau_por]
        self.assertAlmostEqual(acc_at_tau, 1.0, places=6)

    def test_longest_plateau_not_argmax_spike(self):
        """人为构造"长平台 + 远处等分尖峰"，必须选长平台中点而非尖峰。"""
        grid = np.linspace(0.0, 1.0, 11)
        objs = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
        mid, (lo, hi) = _longest_plateau_midpoint(grid, objs, tol=1e-6)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.9)
        self.assertAlmostEqual(mid, 0.7)      # 平台 [0.5,0.9] 的中点，而非 tau=0.0 的尖峰

    def test_public_select_prefers_plateau_interior(self):
        """公开接口：Σ 平台长度最大 -> 取内部中点，不取平台下沿/上沿。"""
        grid = np.linspace(0.05, 0.95, 19)
        n = 100
        cont = np.ones((n, 3))
        y = np.zeros((n, 3))
        q = np.full((n, 3), 0.5)
        mask = np.ones((n, 3))

        def score_all_hit(y_t, pred_t, mask_t):
            return float(np.mean(pred_t == C.ATOM_VALUES["POR"]))

        res = select_tau_per_target(score_all_hit, cont, q, y, mask, grid=grid, tol=1e-6)
        # tau < 0.5 时 q=0.5 命中 -> 全为原子值（满分）；平台从 0.05 覆盖到 ≈0.5
        lo, hi = res["plateau"]["POR"]
        self.assertAlmostEqual(lo, 0.05, delta=0.011)
        self.assertGreater(hi, 0.40)
        self.assertLess(hi, 0.56)
        self.assertGreater(float(res["tau"][0]), lo)
        self.assertLess(float(res["tau"][0]), hi)

    def test_extreme_probability_reproduces_atom(self):
        n = 50
        cont = np.full((n, 3), 5.0)
        y = np.zeros((n, 3))
        y[:, 0] = C.ATOM_VALUES["POR"]
        y[:, 1] = C.ATOM_VALUES["PERM"]
        y[:, 2] = C.ATOM_VALUES["SW"]
        q = np.full((n, 3), 0.999)
        mask = np.ones((n, 3))
        res = select_tau_per_target(_SCORES, cont, q, y, mask)
        out = per_target_hard_switch(cont, q, res["tau"])
        for j, t in enumerate(C.TARGETS):
            np.testing.assert_array_equal(out[:, j], np.full(n, C.ATOM_VALUES[t]))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestMisclassificationCost(unittest.TestCase):
    def test_counts_and_delta(self):
        n = 40
        cont = np.empty((n, 3))
        y = np.empty((n, 3))
        q = np.full((n, 3), 0.5)
        mask = np.ones((n, 3))
        # POR: 前 20 行原子（q=0.9 命中 -> 正确），第 20-25 行非原子但 q=0.9（误判为原子），
        #      第 25-30 行原子但 q=0.1（漏判），其余非原子未命中（正确）
        cont[:, 0] = 20.0
        y[:, 0] = 20.0
        y[:20, 0] = C.ATOM_VALUES["POR"]
        cont[:20, 0] = 11.0
        q[:25, 0] = 0.9
        q[20:25, 0] = 0.9
        q[25:30, 0] = 0.1
        y[25:30, 0] = C.ATOM_VALUES["POR"]   # 原子但被漏判
        cont[25:30, 0] = 11.0
        # PERM/SW 连续头完美
        for j, good in ((1, 1.0), (2, 80.0)):
            y[:, j] = good
            cont[:, j] = good
            q[:, j] = 0.01
        rep = misclassification_cost_report(
            _SCORES, cont, q, y, mask, tau=[0.5, 0.5, 0.5])
        por = rep["POR"]
        self.assertEqual(por["n_false_atom"], 5)    # 20..24
        self.assertEqual(por["n_missed_atom"], 5)   # 25..29
        self.assertEqual(por["n_atom"], 25)
        self.assertEqual(por["n_observed"], 40)
        self.assertAlmostEqual(por["delta"], por["acc_switch"] - por["acc_cont"])
        self.assertIn("total", rep)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestTargetScalersAndDecode(unittest.TestCase):
    """训练折目标尺度（训练/推理共用）与解码路径——纯 numpy，两个环境都跑。"""

    def _data(self):
        por = np.array([0.1, 0.0, 10.0, 20.0, 30.0])
        sw = np.array([99.9, 10.0, 50.0, 80.0, 95.0])
        z = np.log10(np.array([0.01, 0.1, 1.0, 10.0, 100.0]))
        mask = np.ones((5, 3))
        return por, sw, z, mask

    def test_fit_returns_json_scalars_and_expected_values(self):
        import json

        from src.features.basic import fit_target_scalers
        por, sw, z, mask = self._data()
        sc = fit_target_scalers(por, sw, mask, z)
        self.assertEqual(set(sc), {"por_median", "por_max", "sw_mu", "sw_sigma",
                                   "perm_z_median", "s_por", "s_sw"})
        self.assertTrue(all(isinstance(v, float) for v in sc.values()))
        json.dumps(sc)                                    # 可 JSON 序列化
        self.assertAlmostEqual(sc["por_max"], C.POR_MAX_BUFFER * 30.0, places=9)
        self.assertAlmostEqual(sc["sw_mu"], 80.0, places=9)
        self.assertAlmostEqual(sc["por_median"], 10.0, places=9)

    def test_sw_scaler_roundtrip(self):
        from src.features.basic import (apply_sw_scaler, fit_target_scalers,
                                        invert_sw_scaler)
        por, sw, z, mask = self._data()
        sc = fit_target_scalers(por, sw, mask, z)
        zz = apply_sw_scaler(sw, sc)
        np.testing.assert_allclose(invert_sw_scaler(zz, sc), sw, atol=1e-9)

    def test_no_valid_rows_raises(self):
        from src.features.basic import fit_target_scalers
        y = np.zeros(4)
        mask = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            fit_target_scalers(y, y, mask)

    def test_atom_labels_require_observation(self):
        from src.features.basic import build_labels
        targets = np.array([[0.1, 0.01, 99.9],
                            [-9999.0, 0.01, 99.9],
                            [5.0, 0.02, 50.0]])
        tmiss = np.array([[False, False, False],
                          [True, False, False],
                          [False, False, False]])
        lab = build_labels(targets, tmiss, np.zeros(3, dtype="float32"))
        self.assertEqual(lab["y_atom"][0].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(lab["y_joint"][0], 1.0)
        # 缺测行即使数值等于原子值也不算原子标签
        self.assertEqual(lab["y_atom"][1].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(lab["y_joint"][1], 0.0)

    def test_decode_new_dict_api_and_no_sw_01_clip(self):
        from src.features.basic import decode_predictions
        out = {"por": np.array([5.0, 30.0]),
               "perm_z": np.array([-1.0, 2.0]),
               "sw": np.array([0.8, 120.0]),
               "q_atom": np.array([[0.99, 0.0, 0.0], [0.0, 0.0, 0.0]])}
        dec = decode_predictions(out, tau_atom=[0.5, 0.5, 0.5])
        # SW=0.8 保持 0.8 —— 绝不压到 [0,1] 之外再缩放
        self.assertAlmostEqual(dec[0, 2], 0.8)
        # SW=120 软裁剪回 100
        self.assertAlmostEqual(dec[1, 2], 100.0)
        # q_atom 命中 -> POR 精确为 0.1
        self.assertAlmostEqual(dec[0, 0], C.ATOM_VALUES["POR"])
        self.assertAlmostEqual(dec[1, 0], 30.0)

    def test_decode_legacy_positional_api(self):
        from src.features.basic import decode_predictions
        dec = decode_predictions(np.array([5.0]), np.array([0.0]), np.array([80.0]),
                                 q_ph=np.array([0.9]), tau=0.5)
        np.testing.assert_allclose(dec[0], [C.ATOM_VALUES[t] for t in C.TARGETS])


if __name__ == "__main__":
    unittest.main(verbosity=2)
