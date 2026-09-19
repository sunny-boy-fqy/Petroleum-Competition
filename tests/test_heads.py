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
        self.assertEqual(out["ph_logit"].shape, (8,))
        self.assertEqual(out["por"].dtype, torch.float32)
        for k in ("por", "perm_z", "sw", "q_atom", "q_joint"):
            self.assertTrue(torch.isfinite(out[k]).all())

    def test_ph_logit_is_alias_of_q_joint(self):
        m = build_model(16, hidden=16, layers=1, seed=1).eval()
        out = m(torch.randn(4, 16))
        self.assertTrue(torch.equal(out["ph_logit"], out["q_joint"]))

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
        # 初始原子概率应接近先验（仅 bias 决定，权重贡献很小）
        self.assertAlmostEqual(float(torch.sigmoid(out["q_joint"]).mean()), 0.667, delta=0.05)
        self.assertAlmostEqual(float(torch.sigmoid(out["q_atom"][:, 0]).mean()), 0.6674,
                               delta=0.05)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
