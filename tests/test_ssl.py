"""WP7 回归：自监督掩码公共件。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.portability import HAS_TORCH  # noqa: E402
from src.training import ssl as SSL  # noqa: E402


class TestSSL(unittest.TestCase):
    def test_span_mask_shape_and_coverage(self):
        rng = np.random.default_rng(0)
        m = SSL.span_mask(100, rng, span=10, n_spans=3)
        self.assertEqual(m.shape, (100,))
        self.assertGreater(m.sum(), 0)
        self.assertLessEqual(m.sum(), 30)

    def test_mask_curve_batch(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 50, 4)).astype("float32")
        out = SSL.mask_curve_batch(x, rng=rng, mask_ratio=1.0, span=10, n_spans=2)
        self.assertEqual(out["x_masked"].shape, x.shape)
        self.assertEqual(out["mask"].shape, x.shape)
        self.assertTrue((out["x_masked"][out["mask"]] == 0).all())

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_masked_reconstruction_loss(self):
        import torch
        p = torch.zeros(2, 5, 3)
        t = torch.ones(2, 5, 3)
        m = torch.zeros(2, 5, 3)
        m[:, 0, :] = 1
        loss = SSL.masked_reconstruction_loss(p, t, m)
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
