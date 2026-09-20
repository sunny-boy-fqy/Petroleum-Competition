"""E6/P0 两阶段训练件测试（**torch 门控**）：切片权重 / 阶段 2 参数分组 / 负对照 / 泄漏审计。

这些是 E6 的 mandatory 判据（E6/P0 §7）与"不许静默错训"的机械保证：
  * 切片权重按 joint / 非联合原子 / 有效三档，**永不为 0**（连续头是 fallback）；
  * 阶段 2 必须能**冻结**原子头：冻结后 `q_*` 参数的 `requires_grad` 为 False，
    且真的不被更新（权重逐位不变），而连续头照常更新；
  * 标签打乱负对照**保持逐目标/联合边际分布**但破坏配对（行一致率 ≈ 1/N）；
  * 输入列审计必须命中目标派生列（含 `target/label/atom/logId`），F1 正常列通过；
  * 原子标记必须严格等于"标签 == 原子值"（口径错误即报不通过）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training.state_train import (atom_label_provenance,  # noqa: E402
                                      input_no_label_leak_full, label_shuffle_control,
                                      slice_weight_matrix, slice_weight_report)


class TestSliceWeights(unittest.TestCase):
    def _toy(self):
        y_atom = np.zeros((5, 3), dtype=bool)
        y_atom[0] = True                      # 联合原子行
        y_atom[1, 0] = True                   # 非联合原子（仅 POR）
        y_joint = np.array([True, False, False, False, False])
        mask = np.ones((5, 3), dtype=bool)
        return y_atom, y_joint, mask

    def test_three_buckets(self):
        y, j, m = self._toy()
        w = slice_weight_matrix(y, j, m, w_joint=0.2, w_nonjoint_atom=0.3, w_valid=1.0)
        self.assertEqual(w.shape, (5, 3))
        self.assertTrue(np.allclose(w[0], 0.2))          # 联合原子行
        self.assertAlmostEqual(w[1, 0], 0.3)             # 非联合原子
        self.assertAlmostEqual(w[1, 1], 1.0)             # 非原子
        self.assertTrue(np.allclose(w[2:], 1.0))         # 有效行

    def test_never_zero(self):
        y, j, m = self._toy()
        w = slice_weight_matrix(y, j, m, w_joint=0.0, w_nonjoint_atom=0.0, w_valid=0.0)
        rep = slice_weight_report(w)
        self.assertTrue(rep["never_zero"])
        self.assertGreaterEqual(rep["min"], 1e-3)
        self.assertEqual(rep["n_zero"], 0)

    def test_mask_keeps_valid_weight(self):
        y, j, m = self._toy()
        m[0] = False
        w = slice_weight_matrix(y, j, m, w_joint=0.2, w_nonjoint_atom=0.3)
        self.assertTrue(np.allclose(w[0], 1.0), "缺测行权重回落到 w_valid，参与与否由 mask 决定")

    def test_shape_errors(self):
        y, j, m = self._toy()
        with self.assertRaises(ValueError):
            slice_weight_matrix(y[:, :2], j, m)
        with self.assertRaises(ValueError):
            slice_weight_matrix(y, j, m[:, :2])
        with self.assertRaises(ValueError):
            slice_weight_matrix(y, j[:3], m)


class TestLeakAudit(unittest.TestCase):
    def test_banned_names_are_hit(self):
        rep = input_no_label_leak_full(["GR", "RHOB", "target_por", "label_sw", "logId"])
        self.assertFalse(rep["ok"])
        self.assertIn("por", rep["hits"])
        self.assertIn("label", rep["hits"])
        self.assertIn("logid", rep["hits"])

    def test_f1_like_names_pass(self):
        names = ["GR", "RHOB", "NPHI", "DT", "RT", "PE", "DEPT", "sp", "CALI",
                 "DRHO", "PEF", "DEPTH", "miss_GR", "depth_sin", "depth_cos"]
        rep = input_no_label_leak_full(names)
        self.assertTrue(rep["ok"], rep["hits"])

    def test_atom_provenance_matches_labels(self):
        y_por = np.array([0.1, 5.0, 0.1])
        y_perm = np.array([0.01, 0.01, 100.0])
        y_sw = np.array([99.9, 50.0, 99.9])
        y_atom = np.column_stack([y_por == C.ATOM_VALUES["POR"],
                                  y_perm == C.ATOM_VALUES["PERM"],
                                  y_sw == C.ATOM_VALUES["SW"]])
        mask = np.ones((3, 3), dtype=bool)
        rep = atom_label_provenance(y_atom, y_por, y_perm, y_sw, mask)
        self.assertTrue(rep["ok"], rep)
        bad = y_atom.copy()
        bad[0, 0] = False
        self.assertFalse(atom_label_provenance(bad, y_por, y_perm, y_sw, mask)["ok"])


class TestLabelShuffle(unittest.TestCase):
    def test_marginals_preserved_pairing_broken(self):
        rng = np.random.RandomState(0)
        n = 200
        y_atom = rng.rand(n, 3) < np.array([0.6, 0.7, 0.65])
        y_joint = y_atom.all(axis=1)
        rep = label_shuffle_control(y_atom, y_joint, seed=7)
        self.assertTrue(rep["marginals_preserved"])
        for t in C.TARGETS:
            a, b = rep["per_target_rates"][t]
            self.assertAlmostEqual(a, b, places=12)
        self.assertLess(rep["row_identity_rate"], 0.35)

    def test_deterministic(self):
        rng = np.random.RandomState(1)
        y = rng.rand(50, 3) < 0.5
        j = y.all(axis=1)
        a = label_shuffle_control(y, j, seed=3)["y_atom"]
        b = label_shuffle_control(y, j, seed=3)["y_atom"]
        self.assertTrue(np.array_equal(a, b))

    def test_shape_error(self):
        with self.assertRaises(ValueError):
            label_shuffle_control(np.zeros((4, 2), dtype=bool), np.zeros(4, dtype=bool))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTwoStageTorch(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(0)
        from src.models.row_mlp import RowMLP
        from src.training import loop as L
        self.model = RowMLP(8, hidden=16, layers=1)
        self.cfg = L.TrainConfig(lr=1e-2, weight_decay=0.0, batch_size=32, device="cpu",
                                 amp_dtype="fp32", epochs=2)
        rng = np.random.RandomState(0)
        n = 64
        X = rng.randn(n, 8).astype("float32")
        y_atom = rng.rand(n, 3) < 0.5
        self.arrays = {
            "X": X,
            "batch": {"por": (rng.rand(n) * 10).astype("float32"),
                      "perm_z": rng.randn(n).astype("float32"),
                      "sw": (50 + 20 * rng.rand(n)).astype("float32"),
                      "mask": np.ones((n, 3), dtype="float32"),
                      "y_atom": y_atom.astype("float32"),
                      "y_joint": y_atom.all(axis=1).astype("float32")},
        }

    def test_param_groups_freeze(self):
        from src.training.state_train import make_param_groups
        groups, rep = make_param_groups(self.model, q_head_lr_mult=0.0, base_lr=1e-3)
        self.assertTrue(rep["frozen_q_heads"])
        q_params = [p for n, p in self.model.named_parameters()
                    if n in set(rep["q_head_params"])]
        self.assertTrue(q_params)
        self.assertTrue(all(not p.requires_grad for p in q_params))
        self.assertEqual(len(groups), 1, "冻结时不应有 q-head 参数组")
        cont = [p for n, p in self.model.named_parameters()
                if n in set(rep["cont_or_other_params"])]
        self.assertTrue(all(p.requires_grad for p in cont))

    def test_param_groups_lr_mult(self):
        from src.training.state_train import make_param_groups
        groups, rep = make_param_groups(self.model, q_head_lr_mult=0.05, base_lr=1e-3)
        self.assertFalse(rep["frozen_q_heads"])
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[1]["lr"], 5e-5, places=12)
        self.assertAlmostEqual(rep["q_head_lr"], 5e-5, places=12)

    def test_two_stage_runs_and_decreases_loss(self):
        from src.training.state_train import make_param_groups, two_stage_epochs
        h1 = two_stage_epochs(self.model, self.arrays, self.cfg, stage=1, epochs=3)
        self.assertEqual(len(h1["epochs"]), 3)
        self.assertLess(h1["epochs"][-1]["loss"], h1["epochs"][0]["loss"])
        self.assertIn("atom", h1["epochs"][-1]["parts"])
        # 阶段 2：冻结原子头 → 其权重必须逐位不变，而连续头必须变化
        groups, rep = make_param_groups(self.model, q_head_lr_mult=0.0,
                                        base_lr=self.cfg.lr)
        before_q = {n: p.detach().clone() for n, p in self.model.named_parameters()
                    if n in set(rep["q_head_params"])}
        before_c = {n: p.detach().clone() for n, p in self.model.named_parameters()
                    if n in set(rep["cont_or_other_params"])}
        opt = self.torch.optim.AdamW(groups, lr=self.cfg.lr)
        h2 = two_stage_epochs(self.model, self.arrays, self.cfg, stage=2, epochs=2,
                              optimizer=opt,
                              loss_kw={"slice_weight": slice_weight_matrix(
                                  self.arrays["batch"]["y_atom"].astype(bool),
                                  self.arrays["batch"]["y_joint"].astype(bool),
                                  self.arrays["batch"]["mask"].astype(bool))})
        self.assertEqual(h2["stage"], 2)
        for n, p in self.model.named_parameters():
            if n in before_q:
                self.assertTrue(bool(self.torch.equal(p.detach(), before_q[n])),
                                f"阶段 2 冻结失效：{n} 被更新")
        self.assertTrue(any(not bool(self.torch.equal(p.detach(), before_c[n]))
                            for n, p in self.model.named_parameters() if n in before_c),
                        "阶段 2 必须更新连续头")

    def test_stage2_slice_weight_accepts_matrix(self):
        from src.training.state_train import two_stage_epochs
        y = self.arrays["batch"]["y_atom"].astype(bool)
        j = self.arrays["batch"]["y_joint"].astype(bool)
        m = self.arrays["batch"]["mask"].astype(bool)
        w = slice_weight_matrix(y, j, m)
        h = two_stage_epochs(self.model, self.arrays, self.cfg, stage=2, epochs=1,
                             loss_kw={"slice_weight": w})
        self.assertIsNotNone(h["final_loss"])


if __name__ == "__main__":
    unittest.main()
