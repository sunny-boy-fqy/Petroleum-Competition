"""E7/P0 库层测试（**torch 门控**）：λ1 退火曲线与损失消融开关。

这些开关是 E7 七组消融臂的**机械前提**，必须先在库层钉死：
  * `lam1_schedule`：`constant` 恒定；`linear_to_0.1`/`cosine` 单调降到终值，且
    `lam1_at` 旧调用点行为不变（向后兼容，E1/E3/E6 的既有测试同时在校验）；
  * `aux_loss(normalize=False)`：绝对值臂必须与归一化臂**数值不同**，且归一化臂
    在"标签单位整体放大 k 倍"下**尺度不变**（否则 E7 的结论无法归因）；
  * `align_score_log(clamp=False)`：关掉官方截断后，极端低估的得分更低（惩罚更重）；
  * `total_loss` 必须把 `aux_normalize` / `perm_clamp` / `boundary_kappa` / `boundary_sigma`
    透传到对应项，而不是静默丢弃。
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.portability import HAS_TORCH  # noqa: E402


class TestLam1Schedule(unittest.TestCase):
    def setUp(self):
        from src.training import loop as L
        self.L = L
        self.cfg = L.TrainConfig(epochs=10, lam1_start=1.0, lam1_end=0.1, lam1_frac=0.6)

    def test_constant(self):
        vals = [self.L.lam1_schedule("constant", e, 10, self.cfg) for e in range(10)]
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in vals))

    def test_linear_monotone_to_end(self):
        vals = [self.L.lam1_schedule("linear_to_0.1", e, 10, self.cfg) for e in range(10)]
        self.assertAlmostEqual(vals[0], 1.0, places=9)
        self.assertAlmostEqual(vals[-1], 0.1, places=9)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])))

    def test_cosine_monotone_and_ends(self):
        vals = [self.L.lam1_schedule("cosine", e, 10, self.cfg) for e in range(10)]
        self.assertAlmostEqual(vals[0], 1.0, places=9)
        self.assertAlmostEqual(vals[-1], 0.1, places=9)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])))
        # 余弦在两端更平、中段更快：第 2 个点应高于线性
        lin = self.L.lam1_schedule("linear_to_0.1", 2, 10, self.cfg)
        self.assertGreater(vals[2], lin - 1e-12)

    def test_unknown_schedule_raises(self):
        with self.assertRaises(ValueError):
            self.L.lam1_schedule("exp", 0, 10, self.cfg)

    def test_lam1_at_backwards_compatible(self):
        cfg = self.L.TrainConfig(epochs=10, lam1_start=1.0, lam1_end=0.2, lam1_frac=0.5)
        self.assertAlmostEqual(self.L.lam1_at(0, 10, cfg), 1.0, places=9)
        self.assertAlmostEqual(self.L.lam1_at(9, 10, cfg), 0.2, places=9)
        # span = round(10*0.5) = 5 → r=0.4 → 1.0 + (0.2−1.0)*0.4 = 0.68
        self.assertAlmostEqual(self.L.lam1_at(2, 10, cfg), 0.68, places=9)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestLossKnobs(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        from src.losses import score_aligned as SAL
        self.SAL = SAL
        torch.manual_seed(0)
        n = 64
        self.y_por = 10.0 + 5.0 * torch.rand(n)
        self.y_perm = torch.randn(n)
        self.y_sw = 50.0 + 30.0 * torch.rand(n)
        self.mask = torch.ones(n, 3)
        self.p_por = self.y_por + 0.6          # 有误差，避免退化成 0 损失
        self.p_perm = self.y_perm + 0.4
        self.p_sw = self.y_sw + 8.0

    def _aux(self, normalize=True, scale=1.0):
        return self.SAL.aux_loss(self.y_por * scale, self.p_por * scale,
                                 self.y_perm, self.p_perm,
                                 self.y_sw * scale, self.p_sw * scale, self.mask,
                                 s_por=11.34 * scale, s_sw=20.0 * scale,
                                 normalize=normalize)

    def test_absolute_differs_from_normalized(self):
        a = float(self._aux(normalize=True))
        b = float(self._aux(normalize=False))
        self.assertGreater(a, 0.0)
        self.assertGreater(b, 0.0)
        self.assertNotAlmostEqual(a, b, places=6)

    def test_normalized_is_scale_invariant(self):
        base = float(self._aux(normalize=True, scale=1.0))
        for k in (10.0, 100.0):
            self.assertAlmostEqual(float(self._aux(normalize=True, scale=k)), base,
                                   places=5)

    def test_absolute_is_not_scale_invariant(self):
        base = float(self._aux(normalize=False, scale=1.0))
        self.assertNotAlmostEqual(float(self._aux(normalize=False, scale=100.0)), base,
                                  places=6)

    def test_perm_clamp_penalizes_extreme_underestimate(self):
        z = self.torch.zeros(16)
        zhat = self.torch.full((16,), -5.0)          # 低估 5 个数量级
        # beta 取 1.0：β 很大时 softplus 退化成 max(·,0)，两者都会贴到 0 而看不出差异
        clamped = float(self.SAL.align_score_log(z, zhat, beta=1.0, clamp=True).mean())
        unclamped = float(self.SAL.align_score_log(z, zhat, beta=1.0, clamp=False).mean())
        self.assertGreater(clamped, unclamped, "关掉截断后得分应更低（惩罚更重）")

    def test_boundary_focus_changes_loss(self):
        out = {"por": self.p_por, "perm_z": self.p_perm, "sw": self.p_sw}
        batch = {"por": self.y_por, "perm_z": self.y_perm, "sw": self.y_sw,
                 "mask": self.mask}
        base = self.SAL.aligned_loss(batch["por"], out["por"], batch["perm_z"],
                                     out["perm_z"], batch["sw"], out["sw"], self.mask)
        focus = self.SAL.aligned_loss(batch["por"], out["por"], batch["perm_z"],
                                      out["perm_z"], batch["sw"], out["sw"], self.mask,
                                      boundary_kappa=2.0, boundary_sigma=0.25)
        self.assertNotAlmostEqual(float(base), float(focus), places=8)

    def test_total_loss_forwards_new_kwargs(self):
        n = 64
        out = {"por": self.p_por, "perm_z": self.p_perm, "sw": self.p_sw,
               "q_atom_logit": self.torch.randn(n, 3),
               "q_joint_logit": self.torch.randn(n)}
        batch = {"por": self.y_por, "perm_z": self.y_perm, "sw": self.y_sw,
                 "mask": self.mask, "y_atom": (self.torch.rand(n, 3) > 0.5).float(),
                 "y_joint": (self.torch.rand(n) > 0.5).float()}
        total, parts = self.SAL.total_loss(out, batch, lam1=0.5, lam_atom=0.5,
                                           lam_joint=0.2, aux_normalize=False,
                                           perm_clamp=False, boundary_kappa=1.0,
                                           boundary_sigma=0.3)
        self.assertGreater(float(total), 0.0)
        for key in ("align", "aux", "joint", "atom"):
            self.assertIn(key, parts)
        # 开关真的改变结果（不是被静默丢弃）
        total2, _ = self.SAL.total_loss(out, batch, lam1=0.5, lam_atom=0.5, lam_joint=0.2,
                                        aux_normalize=True, perm_clamp=True,
                                        boundary_kappa=0.0)
        self.assertNotAlmostEqual(float(total), float(total2), places=8)


if __name__ == "__main__":
    unittest.main()
