"""原子门测试（**纯 numpy，无 torch，任何环境都应运行**）。

覆盖：逐目标硬切换（无插值、精确原子值、逐目标独立）、联合守卫开关、
`select_tau_per_target` 的官方加权目标 + 平台中点规则、极端原子概率下 τ 复现原子值、
误判代价分解、`official_score_fns()` 与 `src/score.py` 的**同源等价**（R4-M4）、
以及 decoder 的 logit→概率契约（R4-B2）与 SW 契约反例（R4-B3）。

口径说明（R4-M4）
-----------------
下面 `_binary_acc_*` 三个 helper 是 **0/1 容差准确率**，只用来构造"平台形状清晰"的
合成数据；它们**不是**官方目标函数。真正的官方目标由 `official_score_fns()` 提供
（直接包装 `src.score`），`TestOfficialScoreFn` 断言两者在需要时确实不同，
从而防止生产路径照抄测试 helper 优化错目标。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.inference.atomic_gate import (  # noqa: E402
    DEFAULT_PLATEAU_TOL,
    DEFAULT_TAU_GRID_STEP,
    _longest_plateau_midpoint,
    default_tau_grid,
    joint_guard,
    make_official_score_fn,
    misclassification_cost_report,
    official_score_fns,
    per_target_hard_switch,
    select_tau_per_target,
)
from src.portability import HAS_NUMPY  # noqa: E402

if HAS_NUMPY:
    import numpy as np


def _binary_acc_por(y, pred, mask):
    """**0/1 容差准确率**（非官方目标）：|pred − y| < 0.08·(|y| + 1e-3)。"""
    m = mask > 0
    y, pred = y[m], pred[m]
    if y.size == 0:
        return 0.0
    err = np.abs(pred - y) / (0.08 * (np.abs(y) + 1e-3))
    return float(np.mean(err < 1.0))


def _binary_acc_perm(y, pred, mask):
    """**0/1 容差准确率**（非官方目标）：|log10(max(pred/y, 1e-3))| < 1。"""
    m = mask > 0
    y, pred = y[m], pred[m]
    if y.size == 0:
        return 0.0
    ratio = np.maximum(pred / np.maximum(y, 1e-12), 1e-3)
    return float(np.mean(np.abs(np.log10(ratio)) < 1.0))


def _binary_acc_sw(y, pred, mask):
    """**0/1 容差准确率**（非官方目标）：|pred − y| < 0.05·(|y| + 1e-3)。"""
    m = mask > 0
    y, pred = y[m], pred[m]
    if y.size == 0:
        return 0.0
    err = np.abs(pred - y) / (0.05 * (np.abs(y) + 1e-3))
    return float(np.mean(err < 1.0))


# 0/1 容差口径（仅供合成数据的平台形状测试）
_BINARY_SCORES = (_binary_acc_por, _binary_acc_perm, _binary_acc_sw)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestOfficialScoreFn(unittest.TestCase):
    """R4-M4：`official_score_fns()` 必须与 `src/score.py` 逐值同源。"""

    def test_matches_score_py_exactly(self):
        from src.score import acc_perm, acc_relative
        rng = np.random.default_rng(7)
        y = rng.uniform(9.0, 99.0, 200)
        pred = y + rng.normal(0.0, 6.0, 200)
        m = np.ones(200)
        m[:20] = 0.0                     # 含缺测行，验证 drop 口径
        fns = official_score_fns()
        self.assertAlmostEqual(
            fns["POR"](y, pred, m),
            float(acc_relative(y, pred, C.DELTA_POR,
                               missing_mask=(m <= 0), missing_mode="drop")), places=12)
        self.assertAlmostEqual(
            fns["SW"](y, pred, m),
            float(acc_relative(y, pred, C.DELTA_SW,
                               missing_mask=(m <= 0), missing_mode="drop")), places=12)
        yp = rng.uniform(0.01, 100.0, 200)
        pp = yp * np.exp(rng.normal(0.0, 0.5, 200))
        self.assertAlmostEqual(
            fns["PERM"](yp, pp, m),
            float(acc_perm(yp, pp, missing_mask=(m <= 0), missing_mode="drop")), places=12)

    def test_soft_score_and_binary_accuracy_pick_different_tau(self):
        """官方 soft score 与 0/1 容差准确率在**同一组数据**上选出不同的 τ（R4-M4）。

        构造（只看 SW 列，全部 `q=0.9` -> `τ=0.90` 起走连续分支）：
          - 60 行 "原子行"：`y=99.9`（原子值本身满分），连续头 `99.0`（容差内小误差）；
          - 60 行 "陷阱行"：`y=95.5`，连续头恰好 `95.5`（满分），而原子值 `99.9`
            虽然**仍在 5% 容差内**（0/1 口径算对），soft 得分只剩 0.0785。

        两种口径的差别：
          - **0/1 准确率**：原子分支与连续分支都是 1.0 → 整个网格一块平台 → τ≈0.50；
          - **官方 soft score**：原子分支 0.539、连续分支 0.911 → 只有连续分支是平台 → τ≈0.92。
        生产若误用 0/1 口径，τ 会从 0.92 漂到 0.50，硬切换行为完全不同。
        """
        n = 120
        cont = np.full((n, 3), 80.0)
        y = np.zeros((n, 3))
        q = np.full((n, 3), 0.9)
        mask = np.ones((n, 3))
        y[:60, 2] = C.ATOM_VALUES["SW"]       # 99.9
        cont[:60, 2] = 99.0
        y[60:, 2] = 95.5
        cont[60:, 2] = 95.5
        ones = lambda a, b, c: 1.0                       # noqa: E731

        res_soft = select_tau_per_target(
            {"POR": ones, "PERM": ones, "SW": make_official_score_fn("SW")},
            cont, q, y, mask)
        res_bin = select_tau_per_target(
            {"POR": ones, "PERM": ones, "SW": _binary_acc_sw}, cont, q, y, mask)

        # 口径标记必须能区分（防止生产静默串用）
        self.assertEqual(res_soft["score_fn"], "caller")
        # 0/1 口径下两个分支都是"全对"，soft 口径下原子分支明显更低
        self.assertAlmostEqual(res_bin["curve"]["SW"][0][1], 1.0, places=9)
        self.assertAlmostEqual(res_bin["curve"]["SW"][-1][1], 1.0, places=9)
        self.assertLess(res_soft["curve"]["SW"][0][1], 0.60)
        self.assertGreater(res_soft["curve"]["SW"][-1][1], 0.90)
        self.assertAlmostEqual(
            res_soft["curve"]["SW"][-1][1],
            make_official_score_fn("SW")(y[:, 2], cont[:, 2], mask[:, 2]), places=12)
        # 关键结论：两种口径选出的 τ 不同（0/1 -> ≈0.5 的"伪平台"；soft -> ≈0.92）
        self.assertAlmostEqual(float(res_bin["tau"][2]), 0.5, delta=0.06)
        self.assertGreater(float(res_soft["tau"][2]), 0.88)
        self.assertNotAlmostEqual(float(res_soft["tau"][2]),
                                  float(res_bin["tau"][2]), places=2)

    def test_default_grid_and_tol_match_plan(self):
        """R4-M2：默认口径必须与 `E6/P1/PLAN.md` 声明一致（91 点 / 步长 0.01 / tol 1e-3）。"""
        g = default_tau_grid()
        self.assertEqual(g.size, 91)
        self.assertAlmostEqual(float(g[0]), 0.05, places=12)
        self.assertAlmostEqual(float(g[-1]), 0.95, places=12)
        self.assertAlmostEqual(float(g[1] - g[0]), DEFAULT_TAU_GRID_STEP, places=12)
        self.assertAlmostEqual(DEFAULT_PLATEAU_TOL, 1e-3, places=12)

    def test_default_score_fn_is_official(self):
        """不传 score_fn 时必须自动使用官方口径（杜绝生产抄错目标函数）。"""
        n = 10
        cont = np.full((n, 3), 50.0)
        y = np.full((n, 3), 50.0)
        q = np.full((n, 3), 0.1)
        res = select_tau_per_target(None, cont, q, y, np.ones((n, 3)))
        self.assertEqual(res["score_fn"], "official(src.score)")


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

    def test_nonfinite_cont_fails_fast(self):
        """B-2：NaN/Inf 必须显式报错，不能伪装成 interpolation。"""
        cont = np.array([[np.nan, 1.0, 2.0]], dtype="float64")
        q = np.ones((1, 3), dtype="float64")
        with self.assertRaises(FloatingPointError):
            per_target_hard_switch(cont, q, [0.5, 0.5, 0.5])

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

    def test_plateau_midpoint_and_weighted_objective(self):
        cont, q, y, mask = self._por_dataset()
        res = select_tau_per_target(_BINARY_SCORES, cont, q, y, mask)
        # POR 平台覆盖 tau ∈ [0.2, 0.9)，最长平台中点应落在内部（既不是下沿也不是上沿）
        lo, hi = res["plateau"]["POR"]
        self.assertGreaterEqual(lo, 0.15)
        self.assertLessEqual(hi, 0.9)
        tau_por = float(res["tau"][0])
        self.assertGreater(tau_por, lo)
        self.assertLess(tau_por, hi)
        # 目标函数 = 官方加权贡献 ~ 1.0（POR 全对 + PERM/SW 全对）
        self.assertAlmostEqual(res["objective"], 1.0, places=6)
        # 曲线长度 = 默认网格长度（R4-M2：91 点，与 E6/P1/PLAN.md 一致）
        self.assertEqual(len(res["curve"]["POR"]), 91)
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
        grid = default_tau_grid()
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
        self.assertAlmostEqual(lo, 0.05, delta=0.011)  # 91 点网格下沿仍是 0.05
        self.assertGreater(hi, 0.40)
        self.assertLess(hi, 0.56)
        self.assertGreater(float(res["tau"][0]), lo)
        self.assertLess(float(res["tau"][0]), hi)

    def test_placeholder_constraint_selects_feasible_tau(self):
        """P0 审计回归：给定 y_atom + min_placeholder_acc 时，必须选出满足占位 Acc 的 τ。"""
        n = 200
        cont = np.empty((n, 3))
        y = np.empty((n, 3))
        q = np.full((n, 3), 0.5)
        mask = np.ones((n, 3))
        y_atom = np.zeros((n, 3))
        # POR：前 100 行是占位原子行（q=0.5），后 100 行是连续有效行（q=0.95）
        y[: n // 2, 0] = C.ATOM_VALUES["POR"]
        cont[: n // 2, 0] = 11.0
        y_atom[: n // 2, 0] = 1.0
        y[n // 2:, 0] = 20.0
        cont[n // 2:, 0] = 20.0
        q[n // 2:, 0] = 0.95
        # 其它目标保持“连续头完美、无占位”
        for j, good in enumerate((1.0, 80.0), start=1):
            y[:, j] = good
            cont[:, j] = good
            q[:, j] = 0.01
        res = select_tau_per_target(_BINARY_SCORES, cont, q, y, mask,
                                    y_atom=y_atom, min_placeholder_acc=0.99)
        self.assertTrue(res["constraint_feasible"]["POR"])
        self.assertTrue(res["constrained"]["POR"])
        self.assertGreaterEqual(res["placeholder_acc"]["POR"], 0.99)
        tau_por = float(res["tau"][0])
        self.assertLess(tau_por, 0.5)   # 必须把 q=0.5 的占位行切到原子值

    def test_nonfinite_select_tau_raises(self):
        """B-2：τ 搜索拿到 NaN 连续预测时必须拒绝，而不是静默塌到 grid 边界。"""
        cont = np.array([[np.nan, 1.0, 2.0]], dtype="float64")
        q = np.ones((1, 3), dtype="float64")
        y = np.ones((1, 3), dtype="float64")
        mask = np.ones((1, 3), dtype="float64")
        with self.assertRaises(FloatingPointError):
            select_tau_per_target(cont=cont, q_atom=q, y=y, mask=mask)

    def test_extreme_probability_reproduces_atom(self):
        n = 50
        cont = np.full((n, 3), 5.0)
        y = np.zeros((n, 3))
        y[:, 0] = C.ATOM_VALUES["POR"]
        y[:, 1] = C.ATOM_VALUES["PERM"]
        y[:, 2] = C.ATOM_VALUES["SW"]
        q = np.full((n, 3), 0.999)
        mask = np.ones((n, 3))
        res = select_tau_per_target(_BINARY_SCORES, cont, q, y, mask)
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
            _BINARY_SCORES, cont, q, y, mask, tau=[0.5, 0.5, 0.5])
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

        from src.features.basic import (DEFAULT_ATOM_RATES, DEFAULT_JOINT_ATOM_RATE,
                                        fit_target_scalers)
        por, sw, z, mask = self._data()
        sc = fit_target_scalers(por, sw, mask, z)
        # R4-M1：返回键是 build_model(init_stats=...) 的超集（多出 s_por/s_sw）
        self.assertEqual(set(sc), {"por_median", "por_max", "sw_mu", "sw_sigma",
                                   "perm_z_median", "s_por", "s_sw",
                                   "atom_rates", "joint_atom_rate"})
        self.assertTrue(all(isinstance(sc[k], float) for k in
                            ("por_median", "por_max", "sw_mu", "sw_sigma",
                             "perm_z_median", "s_por", "s_sw", "joint_atom_rate")))
        self.assertIsInstance(sc["atom_rates"], tuple)
        json.dumps(sc)                                    # 可 JSON 序列化
        self.assertAlmostEqual(sc["por_max"], C.POR_MAX_BUFFER * 30.0, places=9)
        # 尺度只由非原子有效行拟合：SW 去掉 99.9 -> median([10,50,80,95])=65；
        # POR 去掉 0.1 -> median([0,10,20,30])=15。
        self.assertAlmostEqual(sc["sw_mu"], 65.0, places=9)
        self.assertAlmostEqual(sc["por_median"], 15.0, places=9)
        # M2：只给 z_perm（log10(PERM)）时也要还原原始尺度统计折内原子率先验，
        # 不再无条件回落到 E0 默认值。本数据里 POR/PERM/SW 各只有第 0 行是原子。
        self.assertEqual(tuple(sc["atom_rates"]), (0.2, 0.2, 0.2))
        self.assertAlmostEqual(sc["joint_atom_rate"], 0.2, places=9)

    def test_fit_target_scalers_defaults_only_when_perm_absent(self):
        from src.features.basic import (DEFAULT_ATOM_RATES, DEFAULT_JOINT_ATOM_RATE,
                                        fit_target_scalers)
        por, sw, _z, mask = self._data()
        sc = fit_target_scalers(por, sw, mask, None)
        self.assertEqual(tuple(sc["atom_rates"]), tuple(DEFAULT_ATOM_RATES))
        self.assertAlmostEqual(sc["joint_atom_rate"], DEFAULT_JOINT_ATOM_RATE, places=9)

    def test_fit_target_scalers_computes_atom_rates_from_fold(self):
        """R4-M1：给了 y_perm 就必须用**折内**统计覆盖默认先验。"""
        from src.features.basic import fit_target_scalers
        n = 10
        por = np.array([0.1] * 4 + [10.0] * 6)
        perm = np.array([0.01] * 7 + [1.0] * 3)
        sw = np.array([99.9] * 5 + [80.0] * 5)
        mask = np.ones((n, 3))
        sc = fit_target_scalers(por, sw, mask, np.log10(perm), y_perm=perm)
        self.assertAlmostEqual(sc["atom_rates"][0], 0.4, places=9)   # POR 4/10
        self.assertAlmostEqual(sc["atom_rates"][1], 0.7, places=9)   # PERM 7/10
        self.assertAlmostEqual(sc["atom_rates"][2], 0.5, places=9)   # SW 5/10
        # 三目标同时原子：POR 前 4 且 PERM 前 7 且 SW 前 5 -> 前 4 行
        self.assertAlmostEqual(sc["joint_atom_rate"], 0.4, places=9)

    def test_decode_logits_are_sigmoided_not_used_raw(self):
        """R4-B2：只给 `q_atom_logit` 时必须先 sigmoid 再与概率阈值比较。

        logit=0 -> 概率 0.5：tau=0.4 应命中、tau=0.6 不应命中。
        若把 logit 当成概率直接比，`0.0 > 0.4` 为假 → 两种情况都不命中，测试即失败。
        """
        from src.features.basic import decode_predictions
        out = {"por": np.array([5.0]), "perm_z": np.array([0.0]), "sw": np.array([80.0]),
               "q_atom_logit": np.zeros((1, 3)), "q_joint_logit": np.zeros(1)}
        hit = decode_predictions(out, tau_atom=[0.4, 0.4, 0.4])
        np.testing.assert_allclose(hit[0], [C.ATOM_VALUES[t] for t in C.TARGETS])
        miss = decode_predictions(out, tau_atom=[0.6, 0.6, 0.6])
        np.testing.assert_allclose(miss[0], [5.0, 1.0, 80.0])
        # 概率键存在时**不再**做第二次 sigmoid（0.5 > 0.4 命中；若误做 sigmoid 会变 0.62 仍命中，
        # 因此再用 tau=0.55 区分：真概率 0.5 不命中，二次 sigmoid 0.62 会命中）
        out_prob = dict(out)
        out_prob["q_atom"] = np.full((1, 3), 0.5)
        hit2 = decode_predictions(out_prob, tau_atom=[0.4, 0.4, 0.4])
        np.testing.assert_allclose(hit2[0], [C.ATOM_VALUES[t] for t in C.TARGETS])
        miss2 = decode_predictions(out_prob, tau_atom=[0.55, 0.55, 0.55])
        np.testing.assert_allclose(miss2[0], [5.0, 1.0, 80.0])

    def test_sw_contract_rejects_atom_majority_hiding_normalized_branch(self):
        """R4-B3：7/10 原子行把中位数拉到 99.9 时，契约仍必须拒绝被归一化的连续分支。"""
        from src.inference.contract import validate_payload
        preds = ([{"depth": 1.0 + 0.1 * i, "POR": 0.1, "PERM": 0.01, "SW": 99.9}
                  for i in range(7)]
                 + [{"depth": 2.0 + 0.1 * i, "POR": 12.0, "PERM": 1.5, "SW": 0.8}
                    for i in range(3)])
        payload = {"modelId": "", "modelName": "t", "version": "1.0",
                   "resultData": [{"logId": "w1", "predictions": preds}]}
        r = validate_payload(payload, expected_rows=10, strict_keys=True)
        self.assertFalse(r.ok, "原子行占多数时契约不能靠中位数放行")
        self.assertAlmostEqual(r.stats["sw_median"], 99.9, places=6)   # 中位数守卫确实被绕过
        self.assertEqual(r.stats["sw_n_lt_guard"], 3)
        self.assertGreater(r.stats["sw_lt_guard_frac"], 1e-3)
        self.assertTrue(any("低值计数" in e for e in r.errors))
        # 低分位守卫也独立生效
        self.assertLess(r.stats["sw_non_atom_p05"], C.SW_VALID_MIN)

    def test_sw_contract_accepts_atom_majority_with_healthy_continuous_branch(self):
        """正例：原子行占多数但连续分支在真实尺度（≥ 8.305）时必须通过。"""
        from src.inference.contract import validate_payload
        preds = ([{"depth": 1.0 + 0.1 * i, "POR": 0.1, "PERM": 0.01, "SW": 99.9}
                  for i in range(7)]
                 + [{"depth": 2.0 + 0.1 * i, "POR": 12.0, "PERM": 1.5, "SW": 42.0}
                    for i in range(3)])
        payload = {"modelId": "", "modelName": "t", "version": "1.0",
                   "resultData": [{"logId": "w1", "predictions": preds}]}
        r = validate_payload(payload, expected_rows=10, strict_keys=True)
        self.assertTrue(r.ok, f"不应拒绝健康尺度：{r.errors}")
        self.assertEqual(r.stats["sw_n_lt_guard"], 0)

    def test_percentile_sorted_matches_numpy(self):
        from src.inference.contract import _percentile_sorted
        vals = sorted([0.1, 8.4, 12.0, 42.0, 80.0, 99.9])
        for q in (0.0, 5.0, 25.0, 50.0, 75.0, 100.0):
            self.assertAlmostEqual(_percentile_sorted(vals, q),
                                   float(np.percentile(vals, q)), places=9)

    def test_p05_guard_skips_tiny_payloads(self):
        """p05 守卫在非原子行太少时跳过（避免把小样例的单点外推误判成量纲错误），
        但"低于有效最小值占比"守卫仍然独立生效。"""
        from src.inference.contract import validate_payload
        n = 100
        preds = ([{"depth": 1.0 + 0.01 * i, "POR": 0.1, "PERM": 0.01, "SW": 99.9}
                  for i in range(n - 5)]
                 + [{"depth": 5.0 + 0.01 * i, "POR": 12.0, "PERM": 1.5, "SW": 5.0}
                    for i in range(5)])
        payload = {"modelId": "", "modelName": "t", "version": "1.0",
                   "resultData": [{"logId": "w1", "predictions": preds}]}
        r = validate_payload(payload, expected_rows=n, strict_keys=True)
        self.assertEqual(r.stats["sw_non_atom_n"], 5)          # < MIN_NONATOM(20)
        self.assertLess(r.stats["sw_non_atom_p05"], C.SW_VALID_MIN)
        # 5/100 = 5% > 1% -> "低于有效最小值"守卫必须触发（p05 守卫被跳过也不放行）
        self.assertFalse(r.ok)
        self.assertEqual(r.stats["sw_n_lt_guard"], 0)          # 都不 < 1.0
        self.assertEqual(r.stats["sw_n_lt_valid_min"], 5)      # 但都 < 8.305
        self.assertTrue(any("低于有效最小值" in e for e in r.errors))
        self.assertFalse(any("p05" in e for e in r.errors))

    def test_small_boundary_extrapolation_is_tolerated(self):
        """真值最小 8.305，模型在外推时偶有 <8.305 的行是可接受的（占比 <1% 即放行）。"""
        from src.inference.contract import validate_payload
        n = 1000
        preds = ([{"depth": 1.0 + 0.001 * i, "POR": 0.1, "PERM": 0.01, "SW": 99.9}
                  for i in range(n - 5)]
                 + [{"depth": 5.0 + 0.001 * i, "POR": 12.0, "PERM": 1.5,
                     "SW": 8.0 + 0.01 * i} for i in range(5)])
        payload = {"modelId": "", "modelName": "t", "version": "1.0",
                   "resultData": [{"logId": "w1", "predictions": preds}]}
        r = validate_payload(payload, expected_rows=n, strict_keys=True)
        self.assertEqual(r.stats["sw_n_lt_valid_min"], 5)      # 8.00/8.01/…/8.04
        self.assertTrue(r.ok, f"5/1000=0.5% 的边界外推不应拒绝提交：{r.errors}")

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
