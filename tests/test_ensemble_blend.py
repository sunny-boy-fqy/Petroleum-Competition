"""E8/P2 集成融合测试（**纯 numpy，口径层**）：权重选择 / 同源性 / 显著性 / EMA·SWA。

只测"纪律"与算术，不测模型：
  * 权重**只在 inner-OOF 选**：`select_weights_inner` 只看传入矩阵，网格候选数可复现，
    非负且和为 1；
  * **同源平均不算增益**：恒等成员 → `same_source=true` 且 `delta==0` → `no_go`；
  * **增益必须过显著性**：互补成员（误差反相关）→ 融合 100 分、CI 下界 > 0 → `adopted`；
  * `mask` 语义为 **True=有效**（与 `score_arrays(missing=...)` 相反），τ 只在融合之后做硬切换。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C                                # noqa: E402
from src.ensemble.blend import (EnsembleReport, blend_predictions,  # noqa: E402
                                ema_update, homology_report, judge_ensemble,
                                normalize_weights, score_prediction,
                                select_weights_inner, swa_average, well_totals)

N_WELLS = 12
ROWS_PER_WELL = 6
N_ROWS = N_WELLS * ROWS_PER_WELL

# 每口井一个 SW 偏差（同一口井内所有行相同 → 井级 cluster 才有方差）
D = np.array([1.0 + 0.1 * w for w in range(N_WELLS)], dtype="float64")
WELL_INDEX = np.repeat(np.arange(N_WELLS), ROWS_PER_WELL)

POR_T, PERM_T, SW_T = 0.2, 100.0, 50.0


def _truth():
    y = np.zeros((N_ROWS, 3), dtype="float64")
    y[:, 0] = POR_T
    y[:, 1] = PERM_T
    y[:, 2] = SW_T
    return y


def _pred(por=None, perm_z=None, sw=None, q_atom=None):
    p = {"por": np.full(N_ROWS, POR_T if por is None else por, dtype="float64"),
         "perm_z": np.full(N_ROWS, np.log10(PERM_T) if perm_z is None else perm_z,
                           dtype="float64"),
         "sw": np.full(N_ROWS, SW_T if sw is None else sw, dtype="float64")}
    if q_atom is not None:
        p["q_atom"] = np.asarray(q_atom, dtype="float64")
    return p


def _mask_all():
    return np.ones((N_ROWS, 3), dtype=bool)


def _complementary():
    """A 偏 +d、B 偏 −d（误差反相关）→ 等权平均恰好等于真值。"""
    base = np.repeat(D, ROWS_PER_WELL)
    a, b = _pred(), _pred()
    a["sw"] = SW_T + base
    b["sw"] = SW_T - base
    return {"A": a, "B": b}


class TestWeights(unittest.TestCase):
    def test_normalize_sums_to_one(self):
        w = normalize_weights({"a": 2.0, "b": 6.0})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)
        self.assertAlmostEqual(w["a"], 0.25, places=12)

    def test_normalize_clips_negative(self):
        w = normalize_weights({"a": -1.0, "b": 3.0, "c": 1.0})
        self.assertEqual(w["a"], 0.0)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)

    def test_all_zero_raises(self):
        with self.assertRaises(ValueError):
            normalize_weights({"a": 0.0, "b": 0.0})


class TestBlendPredictions(unittest.TestCase):
    def test_weighted_average_exact(self):
        p = {"A": _pred(sw=SW_T + 2.0), "B": _pred(sw=SW_T - 2.0)}
        out = blend_predictions(p, {"A": 0.25, "B": 0.75})
        # (0.25*(52) + 0.75*(48)) = 49
        self.assertTrue(np.allclose(out["sw"], SW_T - 1.0))
        self.assertTrue(np.allclose(out["por"], POR_T))

    def test_row_mismatch_raises(self):
        p = {"A": _pred(), "B": _pred()}
        p["B"]["sw"] = p["B"]["sw"][:5]
        with self.assertRaises(ValueError):
            blend_predictions(p, {"A": 0.5, "B": 0.5})

    def test_unknown_members_raise(self):
        with self.assertRaises(ValueError):
            blend_predictions({"A": _pred()}, {"Z": 1.0})

    def test_prob_key_requires_all_members(self):
        q = np.full((N_ROWS, 3), 0.5)
        p = {"A": _pred(q_atom=q), "B": _pred()}
        self.assertNotIn("q_atom", blend_predictions(p, {"A": 0.5, "B": 0.5}))
        p["B"]["q_atom"] = q
        self.assertTrue(np.allclose(blend_predictions(p, {"A": 0.5, "B": 0.5})["q_atom"], 0.5))

    def test_prob_shape_mismatch_raises(self):
        p = {"A": _pred(q_atom=np.full((N_ROWS, 3), 0.5)),
             "B": _pred(q_atom=np.full((N_ROWS, 3), 0.5))}
        p["B"]["q_atom"] = p["B"]["q_atom"][:, :2]
        with self.assertRaises(ValueError):
            blend_predictions(p, {"A": 0.5, "B": 0.5})


class TestScorePrediction(unittest.TestCase):
    def test_perfect_scores_100(self):
        sc = score_prediction(_pred(), _truth(), _mask_all())
        self.assertAlmostEqual(sc["total"], 100.0, places=9)

    def test_invalid_rows_dropped(self):
        mask = _mask_all()
        mask[:12, :] = False                      # 丢两整口井的坏预测
        p = _pred()
        p["sw"][:12] = SW_T + 100.0
        self.assertAlmostEqual(score_prediction(p, _truth(), mask)["total"], 100.0, places=9)
        self.assertAlmostEqual(
            score_prediction(p, _truth(), _mask_all())["total"] < 100.0, True)

    def test_tau_switches_to_atom_after_blend(self):
        y = _truth()
        y[:, 0] = C.ATOM_VALUES["POR"]
        y[:, 1] = C.ATOM_VALUES["PERM"]
        y[:, 2] = C.ATOM_VALUES["SW"]
        q = np.ones((N_ROWS, 3), dtype="float64")
        raw = _pred(q_atom=q)
        raw["perm_z"] = np.full(N_ROWS, np.log10(C.ATOM_VALUES["PERM"]))
        without = score_prediction(raw, y, _mask_all())            # 未给 τ → 不切换
        with_tau = score_prediction(raw, y, _mask_all(), tau=[0.5, 0.5, 0.5])
        self.assertLess(without["total"], 100.0)
        self.assertAlmostEqual(with_tau["total"], 100.0, places=9)
        # 未命中（q_atom=0）时必须保持连续值 → 无插值
        quiet = _pred(q_atom=np.zeros((N_ROWS, 3)), sw=SW_T)
        self.assertAlmostEqual(
            score_prediction(quiet, _truth(), _mask_all(), tau=[0.5] * 3)["total"], 100.0,
            places=9)


class TestSelectWeightsInner(unittest.TestCase):
    def test_complementary_picks_midpoint(self):
        r = select_weights_inner(_complementary(), _truth(), _mask_all(), grid_step=0.1)
        self.assertEqual(set(r["weights"]), {"A", "B"})
        self.assertAlmostEqual(r["weights"]["A"], 0.5, places=9)
        self.assertAlmostEqual(r["inner_total"], 100.0, places=6)

    def test_grid_is_nonnegative_simplex(self):
        r = select_weights_inner(_complementary(), _truth(), _mask_all())
        self.assertEqual(r["candidates"], 11)             # 2 成员 × step 0.1 → C(10+1,1)=11
        self.assertTrue(all(v >= 0.0 for v in r["weights"].values()))
        self.assertAlmostEqual(sum(r["weights"].values()), 1.0, places=9)

    def test_single_member(self):
        p = {"only": _pred()}
        r = select_weights_inner(p, _truth(), _mask_all())
        self.assertEqual(r["weights"], {"only": 1.0})
        self.assertEqual(r["candidates"], 1)
        self.assertAlmostEqual(r["inner_total"], 100.0, places=9)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_weights_inner({}, _truth(), _mask_all())

    def test_uses_only_given_matrix(self):
        """两成员互换名字时权重随之互换：选择完全由传入的 inner-OOF 决定。"""
        p = _complementary()
        r1 = select_weights_inner(p, _truth(), _mask_all())
        swapped = {"A": p["B"], "B": p["A"]}
        r2 = select_weights_inner(swapped, _truth(), _mask_all())
        self.assertAlmostEqual(r1["weights"]["A"], r2["weights"]["B"], places=9)


class TestHomology(unittest.TestCase):
    def test_identical_members_flagged(self):
        p = _complementary()
        p["B"] = {"por": p["A"]["por"].copy(), "perm_z": p["A"]["perm_z"].copy(),
                  "sw": p["A"]["sw"].copy()}
        rep = homology_report(p)
        self.assertEqual(rep["same_source_pairs"], [["A", "B"]])
        self.assertTrue(rep["pairs"]["A|B"]["same_source"])
        self.assertAlmostEqual(rep["pairs"]["A|B"]["mean_corr"], 1.0, places=9)
        self.assertIn("不得", rep["warning"])

    def test_anticorrelated_not_flagged(self):
        rep = homology_report(_complementary())
        self.assertEqual(rep["same_source_pairs"], [])
        self.assertAlmostEqual(rep["pairs"]["A|B"]["per_target"]["sw"], -1.0, places=9)
        self.assertFalse(rep["pairs"]["A|B"]["same_source"])

    def test_constant_head_reports_none(self):
        rep = homology_report({"A": _pred(), "B": _pred()})
        self.assertIsNone(rep["pairs"]["A|B"]["per_target"]["por"])
        self.assertFalse(rep["pairs"]["A|B"]["same_source"])


class TestJudgeEnsemble(unittest.TestCase):
    def test_same_source_average_is_no_go(self):
        base = _complementary()["A"]
        dup = {"por": base["por"].copy(), "perm_z": base["perm_z"].copy(),
               "sw": base["sw"].copy()}
        members = {"A": base, "B": dup}
        tau = None
        fused = blend_predictions(members, {"A": 0.5, "B": 0.5})
        rep = judge_ensemble(fused, members, _truth(), _mask_all(), WELL_INDEX, N_WELLS,
                             tau=tau, iters=400)
        self.assertEqual(rep.decision, "no_go")           # delta==0，且同源
        self.assertAlmostEqual(rep.delta, 0.0, places=9)
        self.assertLessEqual(rep.ci[0], 0.0)
        self.assertEqual(rep.homology["same_source_pairs"], [["A", "B"]])

    def test_complementary_average_adopted(self):
        members = _complementary()
        fused = blend_predictions(members, {"A": 0.5, "B": 0.5})
        rep = judge_ensemble(fused, members, _truth(), _mask_all(), WELL_INDEX, N_WELLS,
                             iters=400, seed=7)
        self.assertEqual(rep.decision, "adopted")
        self.assertAlmostEqual(rep.fused_total, 100.0, places=6)
        self.assertGreater(rep.delta, 0.0)
        self.assertGreater(rep.ci[0], 0.0)
        self.assertIn(rep.best_member, ("A", "B"))

    def test_well_totals_clusters(self):
        sc = well_totals(_pred(), _truth(), _mask_all(), WELL_INDEX, N_WELLS)
        self.assertEqual(sc.shape, (N_WELLS,))
        self.assertTrue(np.allclose(sc, 100.0, atol=1e-9))

    def test_report_json_serializable(self):
        rep = judge_ensemble(blend_predictions(_complementary(), {"A": 0.5, "B": 0.5}),
                            _complementary(), _truth(), _mask_all(), WELL_INDEX, N_WELLS,
                            iters=100)
        self.assertIsInstance(rep, EnsembleReport)
        blob = json.dumps(rep.as_dict(), ensure_ascii=False)
        self.assertIn("decision", json.loads(blob))


class TestEmaSwa(unittest.TestCase):
    def test_ema_math(self):
        shadow = {"w": np.array([1.0, 1.0])}
        cur = {"w": np.array([2.0, 0.0])}
        out = ema_update(shadow, cur, decay=0.9)
        self.assertTrue(np.allclose(out["w"], [1.1, 0.9]))

    def test_ema_decay_one_rejected(self):
        with self.assertRaises(ValueError):
            ema_update({"w": np.zeros(1)}, {"w": np.ones(1)}, decay=1.0)

    def test_swa_average(self):
        s = [{"w": np.zeros(2)}, {"w": np.ones(2)}, {"w": np.full(2, 2.0)}]
        self.assertTrue(np.allclose(swa_average(s)["w"], np.full(2, 1.0)))

    def test_swa_empty_raises(self):
        with self.assertRaises(ValueError):
            swa_average([])


if __name__ == "__main__":
    unittest.main()
