"""E8 库层测试（**torch 门控**）：MMoE 同键契约 / 井级分支容量与恒等 / EMA 影子。

只测"结构与纪律"，不测精度（精度属云端 Gate）：
  * MMoE 输出键与 `SeqHead` **完全一致**（否则损失/解码/τ 全得改）；
  * 硬共享 vs MMoE 的参数量对等（`moe_pair`）→ 结构对照才有意义；
  * 门控熵可诊断塌陷；负载均衡损失可反向传播；
  * 井级分支容量 ≤ 主干 1/8（超限抛错）、λ=0 精确恒等、偏置有界、无井身份键；
  * EMA：逐 step 数学正确、`context_ema` 精确还原、shadow 可存取。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.portability import HAS_TORCH                     # noqa: E402

D_IN = 64


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestMMoE(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        from src.models.mmoe import MMoE
        from src.models.heads import SeqHead
        self.MMoE, self.SeqHead = MMoE, SeqHead
        torch.manual_seed(0)
        self.model = MMoE(D_IN, hidden=128, n_experts=4)

    def test_same_keys_as_seq_head(self):
        x = self.torch.randn(5, D_IN)
        out = self.model(x)
        ref = self.SeqHead(D_IN, hidden=128)(x)
        self.assertEqual(set(out), set(ref))
        self.assertEqual(out["q_atom"].shape, (5, 3))
        self.assertEqual(out["por"].shape, (5,))

    def test_sequence_shapes_and_scale_contract(self):
        x = self.torch.randn(2, 7, D_IN)
        out = self.model(x)
        self.assertEqual(out["por"].shape, (2, 7))
        self.assertEqual(out["q_atom"].shape, (2, 7, 3))
        self.assertTrue(bool((out["por"] >= 0).all()))
        self.assertLessEqual(float(out["perm_z"].abs().max()), 6.0 + 1e-6)

    def test_bad_input_rank_raises(self):
        with self.assertRaises(ValueError):
            self.model(self.torch.randn(3, 4, 5, D_IN))

    def test_param_parity_with_hard_share(self):
        from src.models.mmoe import moe_pair, param_comparison
        hard, mmoe = moe_pair(D_IN, hidden=128, n_experts=4)
        rep = param_comparison(hard, mmoe)
        self.assertTrue(rep["ok"], rep)
        self.assertLess(abs(rep["ratio"] - 1.0), 0.15)

    def test_gate_report_shapes_and_collapse_flag(self):
        x = self.torch.randn(32, D_IN)
        rep = self.model.gate_report(x)
        self.assertTrue(rep["ok"])
        for b in rep["branches"]:
            self.assertGreater(rep["branches"][b]["entropy_norm"], 0.0)
            self.assertLessEqual(rep["branches"][b]["entropy_norm"], 1.0 + 1e-9)
            self.assertAlmostEqual(sum(rep["branches"][b]["usage"]), 1.0, places=6)
        # 门槛设到 1.1 → 必然报塌陷（阈值语义可复现）
        strict = self.model.gate_report(x, collapse_entropy=1.1)
        self.assertFalse(strict["ok"])
        self.assertEqual(len(strict["collapsed_branches"]), 5)

    def test_load_balance_loss_backward(self):
        x = self.torch.randn(16, D_IN, requires_grad=True)
        loss = self.model.load_balance_loss(x)
        self.assertEqual(loss.dim(), 0)
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_gate_temp_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.MMoE(D_IN, hidden=32, n_experts=2, gate_temp=0.0)

    def test_expert_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.MMoE(D_IN, hidden=32, n_experts=0)

    def test_init_from_stats_sets_scales(self):
        head = self.MMoE(D_IN, hidden=64, n_experts=2)
        head.init_from_stats(por_median=10.0, por_max=40.0, sw_mu=80.0, sw_sigma=15.0)
        self.assertAlmostEqual(float(head.por_max), 40.0, places=5)
        self.assertAlmostEqual(float(head.sw_mu), 80.0, places=5)
        self.assertAlmostEqual(float(head.sw_sigma), 15.0, places=5)

    def test_gradient_cosines_bounded(self):
        from src.models.mmoe import task_gradient_cosines
        x = self.torch.randn(8, D_IN)
        out = self.model(x)
        losses = {"por": out["por"].sum(), "sw": out["sw"].sum(), "joint": out["q_joint"].sum()}
        shared = [self.model.experts[0][0].weight, self.model.experts[1][0].weight]
        rep = task_gradient_cosines(losses, shared)
        self.assertEqual(rep["n_tasks"], 3)
        for v in rep["cosines"].values():
            if v is not None:
                self.assertGreaterEqual(v, -1.0 - 1e-6)
                self.assertLessEqual(v, 1.0 + 1e-6)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestWellHead(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        from src.models.well_head import WellAttentionHead
        self.Head = WellAttentionHead
        torch.manual_seed(0)

    def test_capacity_rule(self):
        from src.models.well_head import capacity_receipt
        self.assertEqual(capacity_receipt(128, None)["hidden"], 16)
        self.assertTrue(capacity_receipt(128, 16)["ok"])
        self.assertFalse(capacity_receipt(128, 17)["ok"])
        with self.assertRaises(ValueError):
            self.Head(128, hidden=17)

    def test_lambda_zero_is_exact_identity(self):
        head = self.Head(D_IN)
        x = self.torch.randn(4, 9, D_IN)
        cont = self.torch.randn(4, 3)
        out = head(x, cont=cont, lam=0.0)
        self.assertTrue(bool(self.torch.equal(out["cont"], cont)))
        out2 = head(x, cont=cont, lam=0.5)
        self.assertFalse(bool(self.torch.allclose(out2["cont"], cont)))

    def test_delta_is_bounded(self):
        head = self.Head(D_IN, max_delta=2.5)
        with self.torch.no_grad():
            head.mlp[-1].bias.fill_(100.0)
        delta = head(self.torch.randn(3, 5, D_IN))["delta"]
        self.assertLessEqual(float(delta.abs().max()), 2.5 + 1e-5)

    def test_attention_pool_normalized(self):
        head = self.Head(D_IN, pool="attention")
        out = head(self.torch.randn(3, 11, D_IN))
        self.assertEqual(tuple(out["attn"].shape), (3, 11))
        self.assertTrue(bool(self.torch.allclose(out["attn"].sum(1),
                                                 self.torch.ones(3), atol=1e-5)))
        ent = head.attention_entropy(self.torch.randn(3, 11, D_IN))
        self.assertGreater(ent, 0.0)
        self.assertLessEqual(ent, 1.0 + 1e-9)

    def test_mean_pool_has_no_attention(self):
        head = self.Head(D_IN, pool="mean")
        out = head(self.torch.randn(3, 11, D_IN))
        self.assertIsNone(out["attn"])
        self.assertIsNone(head.attention_entropy(self.torch.randn(3, 11, D_IN)))

    def test_bad_pool_and_rank_raise(self):
        with self.assertRaises(ValueError):
            self.Head(D_IN, pool="median")
        with self.assertRaises(ValueError):
            self.Head(D_IN)(self.torch.randn(3, D_IN))

    def test_no_well_identity_keys(self):
        from src.models.well_head import assert_no_well_identity
        head = self.Head(D_IN)
        self.assertEqual(assert_no_well_identity(list(head.state_dict())), [])
        self.assertEqual(assert_no_well_identity(["enc.logId_emb.weight"]),
                         ["enc.logId_emb.weight"])
        self.assertEqual(assert_no_well_identity(["well_id_bias"]), ["well_id_bias"])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestEmaShadow(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 3))

    def test_step_math(self):
        from src.training.ema import ema_step, init_shadow
        for p in self.model.parameters():
            p.data.fill_(2.0)
        sh = init_shadow(self.model)                    # 影子初值 = 2.0
        for p in self.model.parameters():
            p.data.fill_(5.0)
        ema_step(sh, self.model, decay=0.9)
        # 0.9*2 + 0.1*5 = 2.3
        for k, v in sh.items():
            if v.is_floating_point():
                self.assertTrue(bool(self.torch.allclose(v, self.torch.full_like(v, 2.3))), k)

    def test_context_restores_exactly(self):
        from src.training.ema import context_ema, ema_step, init_shadow
        sh = init_shadow(self.model)
        for p in self.model.parameters():
            p.data.fill_(5.0)
        ema_step(sh, self.model, decay=0.0)          # decay=0 → 影子 == 当前权重
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with context_ema(self.model, sh) as m:
            self.assertTrue(bool(self.torch.allclose(
                next(m.parameters()), self.torch.full_like(next(m.parameters()), 5.0))))
            for p in m.parameters():
                p.data.fill_(-1.0)                   # 内部改动也必须被还原
        for k, v in self.model.state_dict().items():
            self.assertTrue(bool(self.torch.equal(v, before[k])), k)

    def test_shadow_roundtrip_and_key_mismatch(self):
        from src.training.ema import init_shadow, load_shadow, shadow_state_dict
        sh = init_shadow(self.model)
        payload = shadow_state_dict(sh, decay=0.999)
        sh2, d = load_shadow(payload, self.model)
        self.assertAlmostEqual(d, 0.999)
        self.assertEqual(set(sh), set(sh2))
        bad = {"decay": 0.9, "shadow": {"nope": self.torch.zeros(1)}}
        with self.assertRaises(KeyError):
            load_shadow(bad, self.model)

    def test_decay_validation(self):
        from src.training.ema import check_decays
        self.assertEqual(check_decays([0.99, 0.999, 0.9995])[0], 0.99)
        with self.assertRaises(ValueError):
            check_decays([1.0])
        with self.assertRaises(ValueError):
            check_decays([])

    def test_decay_report(self):
        from src.training.ema import decay_report, ema_step, init_shadow
        sh = init_shadow(self.model)
        for p in self.model.parameters():
            p.data.fill_(3.0)
        ema_step(sh, self.model, decay=0.999)
        rep = decay_report(self.model, {0.999: sh})
        self.assertEqual(len(rep), 1)
        self.assertGreater(rep[0]["rel_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
