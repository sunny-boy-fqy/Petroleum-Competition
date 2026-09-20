"""E5 逐目标头测试（**torch 门控**）：POR 表示能力 / PERM 截断与契约 / SW 单尺度。

这些是 E5 的**硬判据**（P0 §7、P1 §7、P2 §7），必须在本地就能复算：
  * POR：`0.1+softplus` 是禁用臂（下界锁死 0.1，无法表示 POR=0 与 <0.1），
    `sigmoid`/`softplus_shift` 的可达集合覆盖 0；初值 ≈ 训练折中位数而非 0.1；
  * PERM：`perm_z` 有界/有限、`10**perm_z > 0`；桶头概率和为 1 且期望落在界内；
    官方截断口径下"误差随量级单调"（`tail_consistency_report`）；
  * SW：单一百分数尺度——只做 [0,100] 软裁剪，**绝不** [0,1]、**绝不** ×100；
    原子命中精确等于 99.9（无插值）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C                                  # noqa: E402
from src.models.target_heads import (aligned_loss_np, official_perm_acc,  # noqa: E402
                                     por_representability, sw_decode, sw_scale_check,
                                     tail_consistency_report)
from src.portability import HAS_TORCH                           # noqa: E402


class TestPorRepresentabilityNumpy(unittest.TestCase):
    """纯 numpy 收据（无 torch 也要能跑）：这是"禁用臂"结论的事实来源。"""

    def test_sigmoid_covers_zero_and_small(self):
        r = por_representability("sigmoid", 39.8)
        self.assertTrue(r["can_represent_zero"])
        self.assertTrue(r["can_represent_lt_0p1"])
        self.assertTrue(r["allowed"])

    def test_softplus_shift_covers_zero(self):
        r = por_representability("softplus_shift", 39.8, g0=0.5)
        self.assertEqual(r["min_value"], 0.0)
        self.assertTrue(r["can_represent_lt_0p1"])

    def test_plus_softplus_locked_at_point_one(self):
        r = por_representability("plus_softplus", 39.8)
        self.assertFalse(r["can_represent_zero"])
        self.assertFalse(r["can_represent_lt_0p1"])
        self.assertFalse(r["allowed"])
        self.assertAlmostEqual(r["min_value"], 0.1, places=6)

    def test_unknown_param_raises(self):
        with self.assertRaises(ValueError):
            por_representability("relu", 39.8)


class TestPermTailNumpy(unittest.TestCase):
    def test_official_acc_truncation(self):
        z = np.array([0.0, 0.0, 0.0])
        # 低估 1000 倍：d = max(-3, -3) = -3 → acc 被截到 0（不是负分）
        self.assertAlmostEqual(official_perm_acc(z, z - 10.0), 0.0, places=9)
        self.assertAlmostEqual(official_perm_acc(z, z), 1.0, places=9)

    def test_tail_report_monotone_consistent(self):
        z = np.repeat(np.linspace(-2.0, 2.0, 4), 25)
        offs = np.repeat(np.array([2.0, 1.0, 0.5, 0.0]), 25)
        rep = tail_consistency_report(z, z - offs, n_bins=4)
        self.assertEqual(len(rep["bins"]), 4)
        self.assertTrue(rep["monotone_consistent"], rep)
        self.assertGreater(rep["spearman_acc_vs_z"], 0)
        self.assertLess(rep["spearman_loss_vs_z"], 0)

    def test_tail_report_detects_inconsistent(self):
        z = np.repeat(np.linspace(-2.0, 2.0, 4), 25)
        offs = np.repeat(np.array([0.0, 0.5, 1.0, 2.0]), 25)   # 越大量级越差
        rep = tail_consistency_report(z, z - offs, n_bins=4)
        self.assertFalse(rep["monotone_consistent"], rep)

    def test_empty_input(self):
        rep = tail_consistency_report(np.zeros(0), np.zeros(0))
        self.assertFalse(rep["monotone_consistent"])


class TestSwScaleNumpy(unittest.TestCase):
    def test_scale_check_flags(self):
        r = sw_scale_check()
        self.assertTrue(r["ok"])
        self.assertFalse(r["sw_small_branch"])
        self.assertFalse(r["global_clip_0_1"])
        self.assertFalse(r["multiply_100"])
        self.assertTrue(r["soft_clip_0_100"])
        self.assertFalse(r["interpolation"])
        self.assertIs(C.SW_SMALL_BRANCH, False)

    def test_sw_decode_exact_atom_or_continuous(self):
        cont = np.array([10.0, 50.0, 99.0])
        q = np.array([0.9, 0.1, 0.95])
        out = sw_decode(cont, q, tau=0.5)
        self.assertEqual(out[0], C.ATOM_VALUES["SW"])
        self.assertEqual(out[1], 50.0)
        self.assertEqual(out[2], C.ATOM_VALUES["SW"])
        self.assertTrue(np.all((out == C.ATOM_VALUES["SW"]) | (out == cont)))

    def test_sw_decode_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            sw_decode(np.zeros(3), np.zeros(2), tau=0.5)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPorHeadTorch(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(0)
        from src.models.target_heads import PorHead
        self.PorHead = PorHead
        self.x = torch.randn(6, 8)

    def test_all_params_forward(self):
        for p in ("sigmoid", "softplus_shift", "linear", "plus_softplus"):
            h = self.PorHead(8, hidden=16, param=p)
            out = h(self.x)
            self.assertEqual(out["por"].shape, (6,))
            self.assertEqual(out["allowed"], p != "plus_softplus")

    def test_head_biased_init_hits_median(self):
        for p in ("sigmoid", "softplus_shift", "linear"):
            h = self.PorHead(8, hidden=16, param=p)
            with self.torch.no_grad():
                h.body[0].weight.zero_()            # 去掉随机项 → 只剩输出层 bias
                h.body[0].bias.zero_()
            h.init_from_stats(por_median=11.34, por_max=39.8, low_quantile=0.05)
            with self.torch.no_grad():
                y = float(h(self.x)["por"][0])
            self.assertAlmostEqual(y, 11.34, delta=1e-3, msg=p)

    def test_plus_softplus_cannot_reach_zero(self):
        h = self.PorHead(8, hidden=16, param="plus_softplus")
        with self.torch.no_grad():
            h.out.bias.fill_(-50.0)
        self.assertGreater(float(h(self.x)["por"].min()), 0.0999)

    def test_representability_receipt_matches_head(self):
        h = self.PorHead(8, hidden=16, param="sigmoid", por_max=39.8)
        r = h.representability()
        self.assertTrue(r["can_represent_lt_0p1"])
        self.assertEqual(r["param"], "sigmoid")

    def test_bad_param_raises(self):
        with self.assertRaises(ValueError):
            self.PorHead(8, param="relu")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPermHeadTorch(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(0)
        from src.models.target_heads import PermHead
        self.PermHead = PermHead
        self.x = torch.randn(6, 8)

    def test_tanh_is_bounded(self):
        h = self.PermHead(8, hidden=16, z_output="tanh")
        with self.torch.no_grad():
            h.out.weight.mul_(100.0)
        out = h(self.x)
        self.assertLessEqual(float(out["perm_z"].abs().max()), 6.0 + 1e-6)
        rep = h.contract_report(self.x)
        self.assertTrue(rep["finite"] and rep["perm_positive"] and rep["within_clip"])

    def test_clip_arm_and_linear_arm(self):
        hc = self.PermHead(8, hidden=16, z_output="clip")
        hl = self.PermHead(8, hidden=16, z_output="linear")
        with self.torch.no_grad():
            hc.out.bias.fill_(20.0)                 # 远超界：clip 臂饱和，linear 臂越界
            hl.out.bias.fill_(20.0)
        zc = hc(self.x)["perm_z"]
        zl = hl(self.x)["perm_z"]
        self.assertAlmostEqual(float(zc.abs().max()), 6.0, places=5)
        self.assertGreater(float(zl.abs().max()), 6.0)          # 对照臂确实无界
        self.assertTrue(bool(self.torch.pow(10.0, zl.double()).gt(0).all()))
        rep = hl.contract_report(self.x)
        self.assertTrue(rep["finite"] and rep["perm_positive"])
        self.assertFalse(rep["within_clip"])

    def test_bucket_head_probabilities_sum_to_one(self):
        h = self.PermHead(8, hidden=16, n_buckets=6)
        out = h(self.x)
        self.assertAlmostEqual(float(out["q_bucket"].sum(-1).min()), 1.0, places=5)
        self.assertAlmostEqual(float(out["q_bucket"].sum(-1).max()), 1.0, places=5)
        self.assertGreaterEqual(float(out["perm_z_bucket"].min()), C.PERM_LOG_MIN - 1e-6)
        self.assertLessEqual(float(out["perm_z_bucket"].max()), C.PERM_LOG_MAX + 1e-6)

    def test_quantile_heads_sigma_positive_and_sorted(self):
        h = self.PermHead(8, hidden=16, quantile_heads=3)
        out = h(self.x)
        self.assertGreaterEqual(float(out["sigma"].min()), 0.0)
        q = out["quantiles"]
        self.assertTrue(bool((q[..., 1:] >= q[..., :-1] - 1e-6).all()))

    def test_init_bias_matches_z_median(self):
        h = self.PermHead(8, hidden=16, z_output="tanh")
        with self.torch.no_grad():
            h.body[0].weight.zero_()
            h.body[0].bias.zero_()
        h.init_from_stats(perm_z_median=-0.08)
        with self.torch.no_grad():
            z = float(h(self.x)["perm_z"][0])
        self.assertAlmostEqual(z, -0.08, delta=1e-4)

    def test_bad_z_output_raises(self):
        with self.assertRaises(ValueError):
            self.PermHead(8, z_output="relu")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSwHeadTorch(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(0)
        from src.models.target_heads import SwHead
        self.SwHead = SwHead
        self.x = torch.randn(6, 8)

    def test_label_scale_not_normalized(self):
        h = self.SwHead(8, hidden=16, sw_mu=50.0, sw_sigma=20.0)
        with self.torch.no_grad():
            h.body[0].weight.zero_()
            h.body[0].bias.zero_()
            h.out.bias.fill_(0.0)
        out = h(self.x)
        self.assertTrue(bool(self.torch.allclose(out["sw"],
                                                 self.torch.full((6,), 50.0), atol=1e-5)))
        self.assertLessEqual(float(out["sw"].max()), 100.0)
        self.assertGreaterEqual(float(out["sw"].min()), 0.0)

    def test_soft_clip_at_upper_bound(self):
        h = self.SwHead(8, hidden=16, sw_mu=50.0, sw_sigma=20.0)
        with self.torch.no_grad():
            h.body[0].weight.zero_()
            h.body[0].bias.zero_()
            h.out.bias.fill_(10.0)
        self.assertTrue(bool(self.torch.allclose(h(self.x)["sw"],
                                                 self.torch.full((6,), 100.0), atol=1e-5)))

    def test_q_sw_is_probability_and_scale_check_ok(self):
        h = self.SwHead(8, hidden=16)
        out = h(self.x)
        self.assertTrue(bool(((out["q_sw"] > 0) & (out["q_sw"] < 1)).all()))
        self.assertTrue(out["scale_check"]["ok"])
        self.assertFalse(out["scale_check"]["multiply_100"])

    def test_init_from_stats_initial_output_is_sw_mu(self):
        h = self.SwHead(8, hidden=16)
        with self.torch.no_grad():
            h.body[0].weight.zero_()
            h.body[0].bias.zero_()
        h.init_from_stats(sw_mu=82.805, sw_sigma=20.0, sw_atom_rate=0.7)
        with self.torch.no_grad():
            y = float(h(self.x)["sw"][0])
        self.assertAlmostEqual(y, 82.805, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
