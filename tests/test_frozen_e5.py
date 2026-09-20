"""E5 冻结骨干测试（**torch 门控**）：权重读回冻结 / 分块取隐状态 / 钩子兜底 / 内存收据。

要点：取特征必须与**训练同一条分块路径**（`chunks_for` + 加权拼接），否则特征口径与
训练时的输入分布不一致，E5 的对照就不可信。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.portability import HAS_TORCH                        # noqa: E402
from src.training.frozen import pooled_from_states, states_supported  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestFrozenBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.manual_seed(0)
        from src.models.patchtf import build_patchtf
        from src.training import checkpoint as CK
        from src.training import loop as L
        from src.training import seq_loop as SL

        cls.F, cls.L_, cls.H = 6, 96, 16
        cls.model = build_patchtf(cls.F, patch_len=16, stride=8, d_model=cls.H, n_layers=1,
                                  n_heads=4, max_tokens=64)
        cls.td = tempfile.TemporaryDirectory()
        cls.ckpt = Path(cls.td.name) / "best.pt"
        CK.save_checkpoint(cls.ckpt, cls.model, bf16=False)
        cls.cfg = L.TrainConfig(device="cpu", amp_dtype="fp32", batch_size=8)
        cls.opt_all = SL.SeqOptions(spec=None, chunk=cls.L_, overlap=0, batch_chunks=1,
                                    weight_kind="triangular", arch="patchtf")
        cls.opt_half = SL.SeqOptions(spec=None, chunk=48, overlap=16, batch_chunks=1,
                                     weight_kind="triangular", arch="patchtf")
        cls.X = np.random.RandomState(0).randn(cls.L_, cls.F).astype("float32")

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_states_supported_flag(self):
        self.assertEqual(states_supported(self.model), "native")

    def test_load_frozen_is_frozen_and_equivalent(self):
        from src.training.frozen import load_frozen_model
        m = load_frozen_model(self.ckpt, "patchtf", self.F, arch_kwargs={
            "patch_len": 16, "stride": 8, "d_model": self.H, "n_layers": 1, "n_heads": 4,
            "max_tokens": 64})
        self.assertFalse(m.training)
        self.assertTrue(all(not p.requires_grad for p in m.parameters()))
        x = self.torch.from_numpy(self.X[None, ...])
        self.model.eval()                       # 关掉 dropout 才能逐位比较
        with self.torch.no_grad():
            a = m(x)["por"]
            b = self.model(x)["por"]
        self.assertTrue(bool(self.torch.allclose(a, b, atol=1e-6)))

    def test_native_states_shape_and_match_full_forward(self):
        from src.training.frozen import well_states
        st = well_states(self.model, self.X, self.cfg, self.opt_all, "cpu")
        self.assertEqual(st.shape, (self.L_, self.H))
        with self.torch.no_grad():
            ref = self.model.forward_states(self.torch.from_numpy(self.X[None, ...]))[0]
        self.assertTrue(np.allclose(st, ref.numpy(), atol=1e-5))

    def test_chunked_states_stay_close_to_whole_well(self):
        """分块（chunk=48/overlap=16）与整井的隐状态差异必须有界（无拼缝跳变）。"""
        from src.training.frozen import well_states
        a = well_states(self.model, self.X, self.cfg, self.opt_all, "cpu")
        b = well_states(self.model, self.X, self.cfg, self.opt_half, "cpu")
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        # 未训练权重的隐状态尺度任意，因此判据是**相对**的：拼缝差异不得超过状态幅度
        scale = float(np.abs(a).max()) + 1e-6
        self.assertLess(float(np.abs(a - b).max()) / scale, 1.0)

    def test_hook_fallback(self):
        import torch.nn as nn
        from src.training.frozen import well_states

        class Dummy(nn.Module):
            def __init__(self, f, h):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(f, h), nn.GELU())
                self.blocks = nn.ModuleList([self.body])

            def forward(self, x):
                h = self.body(x)
                return {"por": h.mean(-1)}

        d = Dummy(self.F, 5)
        self.assertEqual(states_supported(d), "hook")
        X = self.X[:, :self.F]
        st = well_states(d, X, self.cfg, self.opt_all, "cpu", hook_module=d.blocks[-1])
        self.assertEqual(st.shape, (self.L_, 5))
        with self.torch.no_grad():
            ref = d.body(self.torch.from_numpy(X[None, ...]))[0].numpy()
        self.assertTrue(np.allclose(st, ref, atol=1e-5))

    def test_hook_required_error(self):
        import torch.nn as nn
        from src.training.frozen import well_states

        class Plain(nn.Module):
            def forward(self, x):
                return {"por": x.mean(-1)}

        with self.assertRaises(ValueError):
            well_states(Plain(), self.X, self.cfg, self.opt_all, "cpu")

    def test_state_report_memory_math(self):
        from src.training.frozen import state_report
        st = {"a": np.zeros((100, 8), dtype="float32"), "b": np.zeros((50, 8), dtype="float32")}
        rep = state_report(st)
        self.assertEqual(rep["n_rows"], 150)
        self.assertEqual(rep["d_model"], 8)
        self.assertLess(abs(rep["float32_mb"] - 150 * 8 * 4 / 1024 ** 2), 1e-3)
        self.assertTrue(rep["ok"])

    def test_bad_input_rank_raises(self):
        from src.training.frozen import well_states
        with self.assertRaises(ValueError):
            well_states(self.model, self.X[None, ...], self.cfg, self.opt_all, "cpu")


class TestPooledFromStates(unittest.TestCase):
    def test_mean_and_max(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.allclose(pooled_from_states(x, "mean"), [2.0, 3.0]))
        self.assertTrue(np.allclose(pooled_from_states(x, "max"), [3.0, 4.0]))

    def test_bad_how_and_rank(self):
        with self.assertRaises(ValueError):
            pooled_from_states(np.zeros((2, 2)), "median")
        with self.assertRaises(ValueError):
            pooled_from_states(np.zeros(2), "mean")


if __name__ == "__main__":
    unittest.main()
