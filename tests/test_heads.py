"""行级 MLP 头测试（**torch 门控**：无 torch 时全部 skip）。

覆盖：forward 形状/dtype、`ph_logit` 别名、原子头/连续头结构、
`init_from_stats` 的初值、**POR 可表示 < 0.1（下界为 0，非 0.1）**、
SW 输出在标签尺度（≈80 而非 ≈0.8）、以及 SW 归一化往返。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

if HAS_TORCH:
    import torch

    from src.models.row_mlp import RowMLP, build_model, count_parameters  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestForward(unittest.TestCase):
    def test_shapes_and_dtype(self):
        m = build_model(32, hidden=32, layers=2, seed=0).eval()
        x = torch.randn(8, 32)
        out = m(x)
        self.assertEqual(out["por"].shape, (8,))
        self.assertEqual(out["perm_z"].shape, (8,))
        self.assertEqual(out["sw"].shape, (8,))
        self.assertEqual(out["q_atom"].shape, (8, 3))
        self.assertEqual(out["q_joint"].shape, (8,))
        self.assertEqual(out["q_atom_logit"].shape, (8, 3))
        self.assertEqual(out["q_joint_logit"].shape, (8,))
        self.assertEqual(out["ph_logit"].shape, (8,))
        self.assertEqual(out["por"].dtype, torch.float32)
        for k in ("por", "perm_z", "sw", "q_atom", "q_joint",
                  "q_atom_logit", "q_joint_logit"):
            self.assertTrue(torch.isfinite(out[k]).all())

    def test_gating_keys_are_probabilities_loss_keys_are_logits(self):
        """R4-B2：`q_*` 必须是 sigmoid(logit)（门控用），不是 logit 本身。"""
        m = build_model(16, hidden=16, layers=1, seed=11).eval()
        out = m(torch.randn(32, 16))
        self.assertTrue(torch.allclose(out["q_atom"], torch.sigmoid(out["q_atom_logit"]),
                                       atol=0, rtol=0))
        self.assertTrue(torch.allclose(out["q_joint"], torch.sigmoid(out["q_joint_logit"]),
                                       atol=0, rtol=0))
        self.assertTrue(bool(((out["q_atom"] >= 0) & (out["q_atom"] <= 1)).all()))

    def test_ph_logit_is_alias_of_q_joint_logit(self):
        m = build_model(16, hidden=16, layers=1, seed=1).eval()
        out = m(torch.randn(4, 16))
        self.assertTrue(torch.equal(out["ph_logit"], out["q_joint_logit"]))

    def test_buffers_not_parameters(self):
        m = RowMLP(16, hidden=8, layers=1)
        params = dict(m.named_parameters())
        buffers = dict(m.named_buffers())
        for name in ("por_max", "sw_mu", "sw_sigma"):
            self.assertNotIn(name, params)
            self.assertIn(name, buffers)
        self.assertGreater(count_parameters(m), 0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestInitFromStats(unittest.TestCase):
    def test_initial_predictions_match_stats(self):
        m = build_model(32, hidden=32, layers=1, seed=0).eval()
        m.init_from_stats()
        out = m(torch.randn(64, 32))
        self.assertAlmostEqual(float(out["por"].mean()), 11.34, delta=0.6)
        self.assertAlmostEqual(float(out["perm_z"].mean()), -0.08, delta=0.5)
        self.assertAlmostEqual(float(out["sw"].mean()), 82.805, delta=2.0)
        # SW 必须是标签尺度（≈80），绝不是归一化值（≈0.8）
        self.assertGreater(float(out["sw"].mean()), 50.0)

    def test_custom_stats(self):
        m = build_model(16, hidden=16, layers=1, seed=0).eval()
        m.init_from_stats(por_median=20.0, por_max=40.0, sw_mu=70.0, sw_sigma=10.0,
                          perm_z_median=1.0)
        out = m(torch.randn(64, 16))
        self.assertAlmostEqual(float(out["por"].mean()), 20.0, delta=1.5)
        self.assertAlmostEqual(float(out["perm_z"].mean()), 1.0, delta=0.5)
        self.assertAlmostEqual(float(out["sw"].mean()), 70.0, delta=2.0)

    def test_atom_priors_written_into_biases(self):
        m = build_model(16, hidden=16, layers=1, seed=0)
        m.init_from_stats(joint_atom_rate=0.667, atom_rates=(0.6674, 0.6773, 0.7097))
        m.eval()
        out = m(torch.randn(1024, 16))
        # R4-B2：`q_joint`/`q_atom` 已经是概率，**不需要**再 sigmoid
        self.assertAlmostEqual(float(out["q_joint"].mean()), 0.667, delta=0.05)
        self.assertAlmostEqual(float(out["q_atom"][:, 0].mean()), 0.6674, delta=0.05)
        # logit 与概率必须自洽
        self.assertAlmostEqual(
            float(torch.sigmoid(out["q_atom_logit"][:, 2]).mean()), 0.7097, delta=0.05)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPorRepresentability(unittest.TestCase):
    def test_very_negative_logit_gives_zero(self):
        """POR 下界是 0，不是 0.1：极端负 logit 必须能给出 < 0.1（float32 下精确 0）。"""
        m = RowMLP(32, hidden=8, layers=1)
        m.eval()
        with torch.no_grad():
            m.cont_por.weight.zero_()
            m.cont_por.bias.fill_(-200.0)
        out = m(torch.randn(4, 32))
        self.assertTrue(bool((out["por"] < 0.1).all()))
        self.assertEqual(float(out["por"].max()), 0.0)

    def test_mid_logit_can_be_below_atom_band(self):
        """构造使初始 por ≈ 0.05 的 logit，验证可低于占位带 0.1。"""
        m = RowMLP(32, hidden=8, layers=1)
        m.eval()
        with torch.no_grad():
            m.cont_por.weight.zero_()
            pmax = float(m.por_max)
            p = 0.05 / pmax
            m.cont_por.bias.fill_(float(torch.log(torch.tensor(p / (1.0 - p)))))
        out = m(torch.randn(4, 32))
        self.assertAlmostEqual(float(out["por"].mean()), 0.05, delta=1e-3)

    def test_gradient_step_with_zero_target_stays_finite(self):
        from src.losses.score_aligned import aux_loss
        m = build_model(16, hidden=16, layers=1, seed=3)
        m.init_from_stats()
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=1e-3)
        batch = {
            "por": torch.tensor([0.0, 0.0, 5.0, 12.0]),
            "perm_z": torch.zeros(4),
            "sw": torch.tensor([80.0, 99.9, 60.0, 82.0]),
            "mask": torch.ones(4, 3),
        }
        out = m(torch.randn(4, 16))
        loss = aux_loss(batch["por"], out["por"], batch["perm_z"], out["perm_z"],
                        batch["sw"], out["sw"], batch["mask"], s_por=11.34, s_sw=20.0)
        loss.backward()
        opt.step()
        out2 = m(torch.randn(4, 16))
        self.assertTrue(torch.isfinite(out2["por"]).all())
        self.assertTrue(bool((out2["por"] >= 0.0).all()))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSwScale(unittest.TestCase):
    def test_sw_to_norm_roundtrip(self):
        for sw in (8.305, 50.0, 82.805, 99.9):
            self.assertAlmostEqual(C.sw_from_norm(C.sw_to_norm(sw)), sw, places=9)
        for sw in (10.0, 70.0):
            self.assertAlmostEqual(
                C.sw_from_norm(C.sw_to_norm(sw, mu=50.0, sigma=12.0), mu=50.0, sigma=12.0),
                sw, places=9)

    def test_sw_output_is_label_scale(self):
        m = build_model(32, hidden=32, layers=1, seed=0).eval()
        m.init_from_stats(sw_mu=82.805, sw_sigma=20.0)
        out = m(torch.randn(256, 32))
        self.assertGreater(float(out["sw"].mean()), 70.0)
        self.assertLess(float(out["sw"].mean()), 95.0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestRealForwardDecodePipeline(unittest.TestCase):
    """**真实 `RowMLP.forward` → `decode_predictions` 链路**（四审 R4-B2 要求的集成测试）。

    四审指出：此前所有原子门测试都用**手工构造的概率矩阵**，所以"模型输出 logits、
    门控按概率解释"这个接口错位完全没被覆盖。这里用真实模型输出跑完整链路，
    并断言硬切换只在 `q_atom > τ` 的行发生、命中值精确等于原子值。
    """

    def setUp(self):
        from src.features.basic import decode_predictions
        self.decode = decode_predictions
        self.model = build_model(16, hidden=16, layers=1, seed=5).eval()
        self.model.init_from_stats()
        torch.manual_seed(0)
        self.x = torch.randn(64, 16)
        with torch.no_grad():
            self.out = self.model(self.x)

    def _numpy_out(self):
        return {k: v.detach().cpu().numpy() for k, v in self.out.items()}

    def test_decode_uses_probabilities_and_switches_exactly(self):
        import numpy as np

        from src import constants as C
        o = self._numpy_out()
        tau = [0.5, 0.5, 0.5]
        dec = self.decode(o, tau_atom=tau)
        q = o["q_atom"]
        hit = q > np.asarray(tau)[None, :]
        av = np.asarray([C.ATOM_VALUES[t] for t in C.TARGETS])
        for j, t in enumerate(C.TARGETS):
            # 命中行精确等于原子值
            np.testing.assert_array_equal(dec[hit[:, j], j],
                                          np.full(int(hit[:, j].sum()), av[j]))
            # 未命中行：POR 直出、PERM = 10**z、SW 软裁剪 —— 都不等于原子值
            miss = ~hit[:, j]
            if miss.any():
                self.assertFalse(bool(np.all(dec[miss, j] == av[j])))
        # SW 绝不落到 [0,1] 的小数区间
        self.assertGreater(float(np.min(dec[:, 2])), 8.0)

    def test_logit_only_dict_matches_probability_dict(self):
        """只给 logits 的解码结果必须与给概率的结果**逐位一致**（不会二次 sigmoid）。"""
        import numpy as np
        o = self._numpy_out()
        only_logit = {"por": o["por"], "perm_z": o["perm_z"], "sw": o["sw"],
                      "q_atom_logit": o["q_atom_logit"],
                      "q_joint_logit": o["q_joint_logit"]}
        only_prob = {"por": o["por"], "perm_z": o["perm_z"], "sw": o["sw"],
                     "q_atom": o["q_atom"], "q_joint": o["q_joint"]}
        tau = [0.4, 0.55, 0.7]
        a = self.decode(only_logit, tau_atom=tau)
        b = self.decode(only_prob, tau_atom=tau)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_joint_guard_via_real_output(self):
        import numpy as np

        from src import constants as C
        o = self._numpy_out()
        tau_high = 0.99
        dec = self.decode(o, tau_atom=[0.999, 0.999, 0.999],
                          q_joint=o["q_joint"], tau_high=tau_high)
        hit = o["q_joint"] > tau_high
        if hit.any():
            av = [C.ATOM_VALUES[t] for t in C.TARGETS]
            np.testing.assert_allclose(dec[hit], np.tile(av, (int(hit.sum()), 1)))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTotalLossLogitContract(unittest.TestCase):
    """R4-B2：`total_loss` 只接受 logits；把概率当 logits 用必须显式报错。"""

    def _batch(self, n=8):
        return {
            "por": torch.rand(n) * 30.0,
            "perm_z": torch.randn(n),
            "sw": torch.rand(n) * 90.0 + 8.0,
            "mask": torch.ones(n, 3),
            "y_atom": (torch.rand(n, 3) > 0.6).float(),
            "y_joint": (torch.rand(n) > 0.5).float(),
        }

    def test_probability_only_raises_clear_error(self):
        from src.losses.score_aligned import total_loss
        out = {"por": torch.rand(8) * 30, "perm_z": torch.randn(8),
               "sw": torch.rand(8) * 90 + 8,
               "q_atom": torch.rand(8, 3), "q_joint": torch.rand(8)}
        with self.assertRaises(ValueError) as ctx:
            total_loss(out, self._batch())
        self.assertIn("q_joint_logit", str(ctx.exception))

    def test_real_model_output_is_accepted_and_finite(self):
        from src.losses.score_aligned import total_loss
        m = build_model(16, hidden=16, layers=1, seed=2).eval()
        out = m(torch.randn(8, 16))
        total, parts = total_loss(out, self._batch())
        self.assertTrue(torch.isfinite(total))
        for k in ("align", "aux", "joint", "atom"):
            self.assertIn(k, parts)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestBuildModelInitStats(unittest.TestCase):
    """R4-M1：`fit_target_scalers` 的输出必须能直接喂给 `build_model`。"""

    def test_fit_target_scalers_feeds_build_model(self):
        import numpy as np

        from src.features.basic import fit_target_scalers
        n = 200
        rng = np.random.default_rng(0)
        por = np.concatenate([np.full(50, 0.1), rng.uniform(5.0, 30.0, n - 50)])
        perm = np.concatenate([np.full(60, 0.01), rng.uniform(0.05, 50.0, n - 60)])
        sw = np.concatenate([np.full(70, 99.9), rng.uniform(20.0, 95.0, n - 70)])
        mask = np.ones((n, 3))
        sc = fit_target_scalers(por, sw, mask, np.log10(perm), y_perm=perm)
        # 之前这里会 TypeError: init_from_stats() got an unexpected keyword 's_por'
        m = build_model(32, hidden=16, layers=1, seed=0, init_stats=sc)
        self.assertEqual(set(m.init_stats_ignored), {"s_por", "s_sw"})   # 不被静默丢弃
        m.eval()
        out = m(torch.randn(64, 32))
        # 折内尺度的效果：初始连续输出接近折内中位数
        self.assertAlmostEqual(float(out["por"].mean()), float(sc["por_median"]), delta=2.0)
        self.assertAlmostEqual(float(out["sw"].mean()), float(sc["sw_mu"]), delta=3.0)
        # 原子先验来自折内统计（约 0.25 / 0.30 / 0.35）
        self.assertAlmostEqual(float(out["q_atom"][:, 2].mean()),
                               float(sc["atom_rates"][2]), delta=0.06)

    def test_unknown_key_is_recorded_not_swallowed_silently(self):
        m = build_model(16, hidden=8, layers=1, seed=0,
                        init_stats={"por_median": 11.0, "typo_key": 1.0})
        self.assertEqual(set(m.init_stats_ignored), {"typo_key"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
