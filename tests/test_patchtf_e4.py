"""E4/P0 PatchTF 测试（**torch 门控**）：patch 几何 / 精确还原 / 同键契约 / CI 与位置消融开关。

硬契约（E4/P0 §7）：
  * 输出长度 == 输入长度（含**奇数**长度与右对齐尾 patch）；
  * `patchify → unpatchify` 逐点精确（重叠权重归一化，不留未覆盖行）；
  * 输出键与 `SeqHead` 完全一致；
  * 只用 torch 2.4 的 SDPA（`attn_api == "sdpa"`），不依赖编译扩展；
  * channel-independent 与 rel_pos 两个开关都能前向（消融可执行）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.portability import HAS_TORCH                        # noqa: E402


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPatchGeometry(unittest.TestCase):
    def test_spans_cover_tail_right_aligned(self):
        from src.models.patchtf import patch_spans
        spans = patch_spans(1024, 32, 16)
        self.assertEqual(spans[0], (0, 32))
        self.assertEqual(spans[-1][1], 1024)
        prev_end = 0
        for s, e in spans:
            self.assertLessEqual(s, prev_end + 16)       # 重叠且不留缝隙
            prev_end = max(prev_end, e)
        self.assertEqual(prev_end, 1024)

    def test_short_length_single_patch(self):
        from src.models.patchtf import patch_spans
        self.assertEqual(patch_spans(10, 32, 16), [(0, 10)])

    def test_bad_params_raise(self):
        from src.models.patchtf import patch_spans
        with self.assertRaises(ValueError):
            patch_spans(100, 0, 16)
        with self.assertRaises(ValueError):
            patch_spans(100, 32, 0)

    def test_coverage_every_row_weighted(self):
        from src.models.patchtf import coverage_report
        for length in (257, 1024, 4096):
            rep = coverage_report(length, 32, 16)
            self.assertTrue(rep["ok"], (length, rep))
            self.assertEqual(rep["uncovered"], 0)
            self.assertGreater(rep["min_weight"], 0.0)

    def test_reconstruction_exact_all_kinds(self):
        from src.models.patchtf import reconstruction_check
        for kind in ("triangular", "hann", "equal"):
            rep = reconstruction_check(257, 32, 16, kind=kind)
            self.assertTrue(rep["ok"], (kind, rep))
            self.assertLess(rep["max_abs_diff"], 1e-9)

    def test_bad_weight_kind_raises(self):
        from src.models.patchtf import overlap_weights
        with self.assertRaises(ValueError):
            overlap_weights(4, 8, "gauss")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPatchTF(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(0)
        from src.models.patchtf import build_patchtf
        self.build = build_patchtf
        self.d_in = 16
        self.model = build_patchtf(self.d_in, patch_len=32, stride=16, d_model=32,
                                   n_layers=2, n_heads=4, max_tokens=128)

    def test_same_keys_as_seq_head(self):
        from src.models.heads import SeqHead
        x = self.torch.randn(2, 96, self.d_in)
        out = self.model(x)
        ref = SeqHead(self.d_in, hidden=32)(x)
        self.assertEqual(set(out), set(ref))
        self.assertEqual(out["q_atom"].shape, (2, 96, 3))

    def test_odd_length_contract(self):
        for L in (97, 257, 511):
            out = self.model(self.torch.randn(2, L, self.d_in))
            self.assertEqual(out["por"].shape, (2, L))
            self.assertEqual(out["q_joint"].shape, (2, L))
            self.assertEqual(out["sw"].shape, (2, L))

    def test_2d_input_squeezed(self):
        out = self.model(self.torch.randn(96, self.d_in))
        self.assertEqual(out["por"].shape, (96,))

    def test_channels_independent_switch(self):
        joint = self.build(self.d_in, patch_len=32, stride=16, d_model=32, n_layers=1,
                           n_heads=4, channel_independent=False, max_tokens=64)
        x = self.torch.randn(2, 96, self.d_in)
        self.assertEqual(joint(x)["por"].shape, (2, 96))
        self.assertNotEqual(joint.summary()["n_params"], self.model.summary()["n_params"])

    def test_rel_pos_switch(self):
        abs_pos = self.build(self.d_in, patch_len=32, stride=16, d_model=32, n_layers=1,
                             n_heads=4, rel_pos=False, max_tokens=64)
        self.assertFalse(abs_pos.rel_pos)
        self.assertEqual(abs_pos(self.torch.randn(2, 96, self.d_in))["por"].shape, (2, 96))

    def test_d_model_heads_divisibility(self):
        with self.assertRaises(ValueError):
            self.build(self.d_in, d_model=30, n_heads=8)

    def test_summary_fields(self):
        s = self.model.summary(length=1024)
        self.assertEqual(s["attn_api"], "sdpa")
        self.assertEqual(s["tokens_at_length"], 63)
        self.assertEqual(s["receptive_field_rows"], 32 + 62 * 16)
        self.assertGreater(s["n_params"], 0)
        self.assertTrue(s["channel_independent"])

    def test_states_shape_and_finite(self):
        st = self.model.forward_states(self.torch.randn(2, 257, self.d_in))
        self.assertEqual(st.shape[:2], (2, 257))
        self.assertEqual(st.shape[2], self.model.d_model)
        self.assertTrue(bool(self.torch.isfinite(st).all()))

    def test_eval_determinism(self):
        self.model.eval()
        x = self.torch.randn(1, 257, self.d_in)
        with self.torch.no_grad():
            a = self.model(x)["por"]
            b = self.model(x)["por"]
        self.assertTrue(bool(self.torch.equal(a, b)))

    def test_too_many_tokens_raise(self):
        small = self.build(self.d_in, patch_len=32, stride=16, d_model=32, n_layers=1,
                           n_heads=4, max_tokens=8)
        with self.assertRaises(ValueError):
            small(self.torch.randn(1, 1024, self.d_in))

    def test_length_shorter_than_patch_raise(self):
        with self.assertRaises(ValueError):
            self.model(self.torch.randn(1, 8, self.d_in))


if __name__ == "__main__":
    unittest.main()
