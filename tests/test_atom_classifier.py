"""WP2 回归：原子分类 focal 损失 / 边界权重 / 专用头 / 训练器（torch 门控）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.portability import HAS_TORCH  # noqa: E402

if HAS_TORCH:
    import torch


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestAtomClassifier(unittest.TestCase):
    def test_focal_loss_finite_and_shapes(self):
        from src.losses.atom_classifier import atom_classifier_loss
        z = torch.randn(8, 3)
        y = (torch.rand(8, 3) > 0.6).float()
        m = torch.ones(8, 3)
        j = (torch.rand(8) > 0.5).float()
        loss = atom_classifier_loss(z, y, m, j, gamma=2.0, alpha_nonjoint=1.0)
        self.assertTrue(torch.isfinite(loss))
        parts = atom_classifier_loss(z, y, m, j, gamma=2.0, return_parts=True)
        for k in ("POR", "PERM", "SW", "atom"):
            self.assertIn(k, parts)

    def test_boundary_weight_higher_near_atom(self):
        from src.losses.atom_classifier import boundary_hard_negative_weight
        # 非原子行：near 的真实/预测都靠近原子值，far 远离
        y_near = torch.tensor([[0.13, 0.012, 99.0]])
        y_far = torch.tensor([[0.5, 5.0, 50.0]])
        c_near = torch.tensor([[0.1, 0.01, 99.9]])
        c_far = torch.tensor([[0.45, 50.0, 20.0]])
        atom = [0.1, 0.01, 99.9]
        delta = [0.08, None, 0.05]
        w_near = boundary_hard_negative_weight(c_near, y_near, atom, delta)
        w_far = boundary_hard_negative_weight(c_far, y_far, atom, delta)
        for j in range(3):
            self.assertGreater(float(w_near[0, j]), float(w_far[0, j]))

    def test_atom_head_shapes(self):
        from src.models.atom_head import build_atom_head
        m = build_atom_head(8, hidden=16, layers=2)
        self.assertEqual(tuple(m(torch.randn(4, 8))["q_atom"].shape), (4, 3))
        self.assertEqual(tuple(m(torch.randn(2, 5, 8))["q_atom"].shape), (2, 5, 3))

    def test_atom_trainer_runs_and_calibrates(self):
        from src.training.atom_train import train_atom_head, predict_atom_logits
        rng = np.random.default_rng(0)
        n, d = 240, 6
        X = rng.normal(size=(n, d)).astype("float32")
        q = rng.random((n, 3))
        y = (q > 0.7).astype("float32")
        m = np.ones((n, 3), dtype="float32")
        j = (y.all(axis=1)).astype("float32")
        cont = (y * np.asarray([0.1, 0.01, 99.9]) +
                (1 - y) * np.asarray([20.0, 10.0, 50.0])).astype("float32")
        res = train_atom_head(X[:180], y[:180], m[:180], j[:180],
                              X[180:], y[180:], m[180:], j[180:],
                              cont_tr=cont[:180], cont_va=cont[180:],
                              hidden=16, epochs=2, batch_size=64, device="cpu",
                              patience=5)
        self.assertIn("model", res)
        self.assertTrue(np.isfinite(res["best_metric"]))
        self.assertEqual(res["q_atom_va"].shape, (60, 3))
        self.assertIn("atom_calibration", res)
        self.assertIn("action_table", res)
        out = predict_atom_logits(res["model"], X[:5], device="cpu")
        self.assertEqual(out.shape, (5, 3))
        self.assertTrue(((out >= 0) & (out <= 1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
