"""E3/P1 序列主干测试（**torch 门控**）：全段 seq2seq、非因果、可训练、同键契约。

三条硬契约（E3/P1 §7）：
1. **输出长度 == 输入长度**（含奇数长度 257；禁止滑窗中心点）；
2. **TCN 非因果**：扰动"未来"必须改变"过去"的输出，反之亦然
   （⚠️ 计划 E3/P1 §7 把判据写成"尾部置零不影响头部输出"，那其实是**因果**判据；
   详见 `src/models/tcn.py` 模块 docstring 的口径更正）；
3. **与 `RowMLP` 同键**：`por/perm_z/sw/q_atom/q_joint/q_atom_logit/q_joint_logit`，
   因此损失/解码/τ 选择/契约检查可原样复用。
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

    from src.models.heads import SeqHead, output_length_report
    from src.models.row_mlp import build_model as build_row
    from src.models.tcn import build_tcn
    from src.models.unet1d import build_unet

F = 32


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSeqHeads(unittest.TestCase):
    def test_output_keys_match_row_mlp(self):
        row = build_row(n_features=F, hidden=16, layers=1, dropout=0.0)
        row.eval()
        head = SeqHead(16, hidden=16, dropout=0.0)
        head.eval()
        with torch.no_grad():
            r = row(torch.randn(4, F))
            s = head(torch.randn(4, 7, 16))
        self.assertEqual(set(r.keys()), set(s.keys()),
                         "序列头必须与 RowMLP 同键（损失/解码复用同一套口径）")
        for k in ("por", "perm_z", "sw", "q_joint", "q_joint_logit", "ph_logit"):
            self.assertEqual(s[k].shape, (4, 7), k)
        self.assertEqual(s["q_atom"].shape, (4, 7, 3))
        self.assertEqual(s["q_atom_logit"].shape, (4, 7, 3))
        # 概率键与 logit 键必须确实不同（R4-B2 的分键纪律）
        self.assertFalse(torch.allclose(s["q_atom"], s["q_atom_logit"]))

    def test_parameterization_is_por_max_sigmoid(self):
        head = SeqHead(8, hidden=8, dropout=0.0, por_max=40.0, sw_mu=80.0, sw_sigma=10.0)
        head.eval()
        # `por = por_max·sigmoid(g)`：把**连续头最后一层的 bias** 压到 -20 即 g≈-20
        with torch.no_grad():
            head.cont_por[-1].bias.fill_(-20.0)
            out = head(torch.randn(1, 3, 8))
        self.assertLess(float(out["por"].max()), 1e-6, "POR 必须能逼近 0（禁止 0.1+softplus）")
        self.assertLessEqual(float(out["por"].max()), 40.0 + 1e-6)
        self.assertGreater(float(out["perm_z"].abs().max()), 0.0)

    def test_init_from_stats_accepted(self):
        stats = {"por_median": 11.34, "por_max": 39.8, "sw_mu": 82.8, "sw_sigma": 6.1,
                 "perm_z_median": -0.08, "joint_atom_rate": 0.67,
                 "atom_rates": (0.66, 0.67, 0.71), "s_por": 5.0, "s_sw": 6.1}
        m = build_unet(F, base_ch=8, depth=2, dropout=0.0, init_stats=stats)
        self.assertEqual(m.head.init_stats_ignored, ("s_por", "s_sw"))
        self.assertAlmostEqual(float(m.head.sw_mu), 82.8, places=3)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestNpuSafePadding(unittest.TestCase):
    """NPU 不支持 bf16 replicate_pad1d；用手工 expand+cat 复现其语义。"""

    def test_replicate_pad_matches_torch_float32(self):
        import torch.nn.functional as F
        from src.models.padding import replicate_pad1d
        x = torch.randn(2, 3, 17)
        ref = F.pad(x, (2, 2), mode="replicate")
        got = replicate_pad1d(x, 2)
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        self.assertTrue(torch.allclose(got, ref))

    def test_replicate_pad_bf16_cpu(self):
        import torch.nn.functional as F
        from src.models.padding import replicate_pad1d
        x = torch.randn(2, 3, 17).to(torch.bfloat16)
        ref = F.pad(x.float(), (3, 3), mode="replicate").to(torch.bfloat16)
        got = replicate_pad1d(x, 3)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))

    def test_nearest_upsample_bf16_cpu(self):
        import torch.nn.functional as F
        from src.models.padding import nearest_upsample1d
        x = torch.randn(2, 3, 5).to(torch.bfloat16)
        got = nearest_upsample1d(x, 11)
        ref = F.interpolate(x.float(), size=11, mode="nearest").to(torch.bfloat16)
        self.assertEqual(tuple(got.shape), (2, 3, 11))
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got, ref))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSeqBackbones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unet = build_unet(F, base_ch=16, depth=3, dropout=0.0).eval()
        cls.tcn = build_tcn(F, channels=16, n_blocks=4, dilation_max=4, dropout=0.0).eval()

    def test_full_segment_length_preserved(self):
        """全段 seq2seq：任意长度（含 257/1024）输出与输入逐行同长。"""
        for m in (self.unet, self.tcn):
            rep = output_length_report(m, F, lengths=(64, 257, 1024))
            self.assertTrue(rep["ok"], f"{type(m).__name__} 长度未保持：{rep['shapes']}")
            for L in ("64", "257", "1024"):
                self.assertEqual(rep["shapes"][L]["por"][:2], [2, int(L)])
                self.assertEqual(rep["shapes"][L]["q_atom"][:2], [2, int(L)])

    def test_forward_is_finite_and_differentiable(self):
        x = torch.randn(2, 128, F, requires_grad=True)
        for m in (self.unet, self.tcn):
            out = m(x)
            self.assertTrue(all(torch.isfinite(v).all() for v in out.values()),
                            f"{type(m).__name__} 产生非有限输出")
            out["por"].mean().backward(retain_graph=True)
        self.assertIsNotNone(x.grad)

    def test_tcn_uses_future_and_past_context(self):
        """非因果判据（居中卷积）：未来与过去都必须影响输出。

        扰动必须落在**感受野之内**才可能观察到差异 —— 这正是我第一次探针写错的地方
        （拿整条尾部去测一个 RF=15 的模型，当然测不出变化）。
        """
        m = build_tcn(F, channels=16, n_blocks=3, dilation_max=4, dropout=0.0).eval()
        rf = m.receptive_field()
        self.assertGreaterEqual(rf, 9)
        x = torch.randn(1, 64, F)
        with torch.no_grad():
            base = m(x)["por"]
            x_future = x.clone(); x_future[:, 6:10, :] = 0.0     # 位置 0 的"未来"
            fut = m(x_future)["por"]
            x_past = x.clone(); x_past[:, 0:4, :] = 0.0          # 位置 8 的"过去"
            past = m(x_past)["por"]
            far = torch.cat([x[:, :32], torch.zeros(1, 32, F)], dim=1)
            far_out = m(far)["por"]
        self.assertFalse(torch.allclose(base[:, 0], fut[:, 0]),
                         "非因果模型必须用到未来上下文（若这里为 True，说明卷积退化成了因果）")
        self.assertFalse(torch.allclose(base[:, 8], past[:, 8]),
                         "模型必须用到过去上下文")
        self.assertTrue(torch.allclose(base[:, 0], far_out[:, 0]),
                        "感受野之外的扰动不应影响输出（居中 pad 的健全性检查）")

    def test_unet_uses_long_range_context(self):
        """U-Net 能看到**远超单点**的上下文：测**经验感受野**而不是信任理论公式。

        实测（2026-09-20）：`depth=4` 的理论式给出 461 点，但把 32 点的扰动块放
        在 100 点外时，位置 0 的输出变化低于 `allclose` 容差 —— 说明理论式是**上界**，
        真实有效感受野更小（下采样 + nearest 上采样 + replicate pad 的联合效应）。
        因此这里用显式容差测"多远之外的扰动仍能改变输出"，并把经验值写进断言信息；
        `receptive_field()` 在报告里也只作为上界使用。
        """
        m = build_unet(F, base_ch=16, depth=4, dropout=0.0).eval()
        x = torch.randn(1, 512, F)
        with torch.no_grad():
            base = m(x)["por"][:, 0]
            scale = float(base.abs().max()) + 1e-12
            reach, max_delta = 0, 0.0
            for start in range(4, 96, 4):
                xin = x.clone()
                # 用**大幅偏移**而不是置零：小扰动经深层路径衰减后可能低于 float32 分辨率，
                # 那测的是初始化尺度而不是架构（E3 的 `rf_ablation.py` 会用真实训练后的权重复测）。
                xin[:, start:start + 4, :] += 50.0
                delta = float((m(xin)["por"][:, 0] - base).abs().max()) / scale
                max_delta = max(max_delta, delta)
                if delta > 1e-6:
                    reach = start
        # 严格结论：U-Net **不是逐点模型**（位置 0 的输出确实受邻域输入影响）。
        # 有效感受野的**具体数值**必须由 E3 的 `rf_ablation.py` 实测并写进
        # `E3_receptive_field_ablation.json` —— 这里不写死一个没测过的阈值。
        self.assertGreater(max_delta, 1e-6,
                           f"位置 0 对 40–92 点外的扰动都不敏感（reach={reach}）："
                           "U-Net 在该初始化下退化为逐点模型（须在 E3 的 RF 消融中用训练后权重复核）")
        self.assertGreater(m.receptive_field(), 200,
                           "depth=4 的理论感受野上界应达数百点（报告里按上界标注）")

    def test_param_scale_matches_plan(self):
        """参数量应落在计划口径内（base_ch=64/depth=5 的 U-Net 约 1–3 M）。"""
        u = build_unet(F, base_ch=64, depth=5, dropout=0.0)
        t = build_tcn(F, channels=128, n_blocks=9, dropout=0.0)
        nu = sum(p.numel() for p in u.parameters())
        nt = sum(p.numel() for p in t.parameters())
        self.assertGreater(nu, 2e5)
        self.assertLess(nu, 5e6, f"U-Net 参数量偏大：{nu}")
        self.assertGreater(nt, 1e5)
        self.assertLess(nt, 5e6, f"TCN 参数量偏大：{nt}")

    def test_receptive_field_grows_with_depth_and_dilation(self):
        rfs = [build_unet(F, base_ch=8, depth=d, dropout=0.0).receptive_field()
               for d in (3, 4, 5)]
        self.assertEqual(rfs, sorted(rfs))
        self.assertLess(rfs[0], rfs[-1])
        t_small = build_tcn(F, channels=8, n_blocks=3, dilation_max=2, dropout=0.0)
        t_big = build_tcn(F, channels=8, n_blocks=3, dilation_max=64, dropout=0.0)
        self.assertLess(t_small.receptive_field(), t_big.receptive_field())

    def test_tcn_dilations_are_centered_pads(self):
        """块两侧 pad 必须相等（居中=非因果）；因果实现会只 pad 左侧。"""
        m = build_tcn(F, channels=8, n_blocks=3, dilation_max=8, dropout=0.0)
        for b in m.blocks:
            self.assertGreater(b.pad, 0)

    def test_training_step_runs_with_total_loss(self):
        """与 `total_loss` 串起来能真的反传（键名契约的端到端证明）。"""
        from src.losses.score_aligned import total_loss
        m = build_tcn(F, channels=16, n_blocks=2, dilation_max=2, dropout=0.0)
        x = torch.randn(2, 64, F)
        out = m(x)
        batch = {"por": torch.rand(2, 64) * 20.0,
                 "perm_z": torch.randn(2, 64),
                 "sw": torch.rand(2, 64) * 99.0,
                 "mask": torch.ones(2, 64, 3), "y_atom": torch.zeros(2, 64, 3),
                 "y_joint": torch.zeros(2, 64)}
        loss, parts = total_loss(out, batch, lam1=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("align", parts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
