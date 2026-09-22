"""损失层测试（**torch 门控**：无 torch 时全部 skip）。

覆盖：`masked_mean` 的 NaN 免疫、`align_score_log` 的下截断尾部、`boundary_focus_weight`
的开关与峰值、`aux_loss` 的尺度归一化、`atom_bce` 的非联合原子加权、`total_loss` 四段式组合。
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

if HAS_TORCH:
    import torch

    from src.losses.physics import (ACF_US_M, ACMA_US_M,  # noqa: E402
                                    physics_porosity_loss)
    from src.losses.score_aligned import (  # noqa: E402
        align_score_log,
        atom_bce,
        aux_loss,
        boundary_focus_weight,
        joint_bce,
        masked_mean,
        total_loss,
    )


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestMaskedMean(unittest.TestCase):
    def test_nan_does_not_pollute(self):
        x = torch.tensor([float("nan"), 2.0])
        m = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(float(masked_mean(x, m)), 2.0, places=6)

    def test_all_masked(self):
        x = torch.tensor([float("nan"), float("inf")])
        m = torch.zeros(2)
        self.assertEqual(float(masked_mean(x, m)), 0.0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestAlignScoreLog(unittest.TestCase):
    @staticmethod
    def _official(z, zhat, eps=1e-3):
        """官方闭式：max(0, 1 − |max(ẑ − z, log10 ε)|)。"""
        d = max(zhat - z, math.log10(eps))
        return max(0.0, 1.0 - abs(d))

    def test_underestimation_tail_matches_official(self):
        z = torch.zeros(1)
        for dz in (-5.0, -4.0, -3.0, -0.5, 0.0, 0.5):
            s = float(align_score_log(z, torch.full((1,), dz)))
            self.assertAlmostEqual(s, self._official(0.0, dz), delta=2e-3,
                                   msg=f"dz={dz}")

    def test_tail_is_clamped_not_unbounded(self):
        """zhou-y=1e-5 与 1e-9 的官方误差都封顶在 3.0，得分都应为 0。"""
        z = torch.zeros(1)
        for dz in (-5.0, -9.0):
            self.assertAlmostEqual(float(align_score_log(z, torch.full((1,), dz))), 0.0, places=3)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestBoundaryFocus(unittest.TestCase):
    def test_kappa_zero_is_all_ones(self):
        r = torch.linspace(0.0, 2.0, 101)
        w = boundary_focus_weight(r, kappa=0.0)
        self.assertTrue(torch.equal(w, torch.ones_like(r)))

    def test_peaks_at_r_one(self):
        r = torch.linspace(0.0, 2.0, 201)
        w = boundary_focus_weight(r, kappa=2.0, sigma=0.25)
        self.assertTrue(torch.all(w >= 1.0))
        k = int(torch.argmax(w))
        self.assertAlmostEqual(float(r[k]), 1.0, delta=0.02)
        self.assertAlmostEqual(float(w[k]), 3.0, delta=1e-3)

    def test_wired_into_aligned_loss_and_atom_exclusion(self):
        from src.losses.score_aligned import aligned_loss
        y = torch.tensor([10.0, 10.0])
        p = torch.tensor([10.0, 11.0])
        z = torch.zeros(2)
        sw = torch.tensor([80.0, 80.0])
        mask = torch.ones(2, 3)
        base = float(aligned_loss(y, p, z, z, sw, sw, mask))
        foc = float(aligned_loss(y, p, z, z, sw, sw, mask, boundary_kappa=2.0))
        self.assertNotAlmostEqual(base, foc, places=4)
        atom = torch.ones(2, 3)
        excl = float(aligned_loss(y, p, z, z, sw, sw, mask, boundary_kappa=2.0,
                                  y_atom=atom))
        self.assertAlmostEqual(base, excl, places=6)
        ignored = float(aligned_loss(y, p, z, z, sw, sw, mask, boundary_kappa=2.0,
                                     y_atom=atom, include_atom_mask=False))
        self.assertAlmostEqual(foc, ignored, places=6)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestAuxLossScale(unittest.TestCase):
    def test_sw_term_is_scale_invariant(self):
        y_por = torch.tensor([11.0, 0.1])
        p_por = torch.tensor([11.5, 3.0])
        z_perm = torch.zeros(2)
        zhat_perm = torch.tensor([0.2, -0.3])
        y_sw = torch.tensor([80.0, 99.9])
        sw_hat = torch.tensor([78.0, 95.0])
        mask = torch.ones(2, 3)
        a = float(aux_loss(y_por, p_por, z_perm, zhat_perm, y_sw, sw_hat, mask,
                           s_por=11.34, s_sw=20.0))
        b = float(aux_loss(y_por, p_por, z_perm, zhat_perm, y_sw * 100.0, sw_hat * 100.0,
                           mask, s_por=11.34, s_sw=2000.0))
        self.assertAlmostEqual(a, b, places=5)

    def test_slice_weight_never_zero(self):
        y = torch.tensor([11.0, 0.1])
        p = torch.tensor([11.0, 0.1])
        z = torch.zeros(2)
        mask = torch.ones(2, 3)
        w = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]])
        loss = aux_loss(y, p, z, z, y, p, mask, slice_weight=w)
        self.assertTrue(torch.isfinite(loss))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestAtomBce(unittest.TestCase):
    def _setup(self):
        q = torch.zeros(2, 3)
        q[0, 0] = 2.0    # 联合原子行，较易
        q[1, 0] = 0.0    # 非联合原子行，较难
        y_atom = torch.zeros(2, 3)
        y_atom[0, 0] = 1.0
        y_atom[1, 0] = 1.0
        y_joint = torch.tensor([1.0, 0.0])
        mask = torch.ones(2, 3)
        return q, y_atom, y_joint, mask

    def test_nonjoint_atom_upweighted(self):
        q, y_atom, y_joint, mask = self._setup()
        weighted = float(atom_bce(q, y_atom, mask, alpha_nonjoint=1.0, y_joint=y_joint))
        plain = float(atom_bce(q, y_atom, mask, alpha_nonjoint=0.0, y_joint=y_joint))
        # 非联合的"难"原子行权重 ×2，拉高平均值
        self.assertGreater(weighted, plain)

    def test_return_parts(self):
        q, y_atom, y_joint, mask = self._setup()
        parts = atom_bce(q, y_atom, mask, alpha_nonjoint=1.0, y_joint=y_joint,
                         return_parts=True)
        for k in ("POR", "PERM", "SW", "atom"):
            self.assertIn(k, parts)
            self.assertTrue(torch.isfinite(parts[k]))

    def test_pos_weight_accepted(self):
        q, y_atom, y_joint, mask = self._setup()
        a = atom_bce(q, y_atom, mask, pos_weight=torch.tensor([1.0, 2.0, 3.0]), y_joint=y_joint)
        b = atom_bce(q, y_atom, mask, pos_weight=[1.0, 2.0, 3.0], y_joint=y_joint)
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_joint_bce(self):
        q = torch.zeros(4)
        y = torch.tensor([1.0, 1.0, 0.0, 0.0])
        m = torch.tensor([1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(joint_bce(q, y, m)), math.log(2.0), places=5)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTotalLoss(unittest.TestCase):
    def _batch(self, n=8):
        return {
            "por": torch.rand(n) * 30.0,
            "perm_z": torch.randn(n),
            "sw": torch.rand(n) * 90.0 + 8.0,
            "mask": torch.ones(n, 3),
            "y_atom": (torch.rand(n, 3) > 0.6).float(),
            "y_joint": (torch.rand(n) > 0.5).float(),
        }

    def test_four_terms(self):
        from src.models.row_mlp import build_model
        m = build_model(32, hidden=32, layers=1, seed=0).eval()
        out = m(torch.randn(8, 32))
        batch = self._batch()
        total, parts = total_loss(out, batch)
        self.assertTrue(torch.isfinite(total))
        for k in ("align", "aux", "joint", "atom"):
            self.assertIn(k, parts)
        self.assertIn("ph", parts)          # 旧键别名
        self.assertTrue(all(isinstance(v, float) for v in parts.values()))

    def test_legacy_batch_without_atom(self):
        from src.models.row_mlp import build_model
        m = build_model(32, hidden=32, layers=1, seed=0).eval()
        out = m(torch.randn(8, 32))
        batch = self._batch()
        batch.pop("y_atom")
        batch.pop("y_joint")
        batch["y_ph"] = (torch.rand(8) > 0.5).float()
        total, parts = total_loss(out, batch)
        self.assertTrue(torch.isfinite(total))
        self.assertNotIn("atom", parts)

    def test_explicit_atom_missing_raises(self):
        from src.models.row_mlp import build_model
        m = build_model(32, hidden=32, layers=1, seed=0).eval()
        out = m(torch.randn(8, 32))
        batch = self._batch()
        batch.pop("y_atom")
        with self.assertRaises(ValueError):
            total_loss(out, batch, use_atom=True)

    def test_atom_kwargs_not_forwarded_to_align(self):
        """pos_weight/alpha_nonjoint 是原子项专属，不能被透传给 aligned_loss。"""
        from src.models.row_mlp import build_model
        m = build_model(32, hidden=32, layers=1, seed=0).eval()
        out = m(torch.randn(8, 32))
        total, parts = total_loss(out, self._batch(), pos_weight=[1.0, 2.0, 3.0],
                                  alpha_nonjoint=2.0)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("atom", parts)

    def test_boundary_kappa_passthrough(self):
        from src.models.row_mlp import build_model
        m = build_model(32, hidden=32, layers=1, seed=0).eval()
        out = m(torch.randn(8, 32))
        total, parts = total_loss(out, self._batch(), boundary_kappa=1.0)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("align", parts)

    def test_default_mask_supports_sequence_shapes(self):
        """mask 缺省时，total_loss 必须兼容 (B,L,3) 输出，而不是只支持行级 (B,3)。"""
        B, L = 2, 3
        out = {
            "por": torch.rand(B, L),
            "perm_z": torch.rand(B, L),
            "sw": torch.rand(B, L),
            "q_atom_logit": torch.rand(B, L, 3),
            "q_joint_logit": torch.rand(B, L),
        }
        batch = {
            "por": torch.rand(B, L),
            "perm_z": torch.rand(B, L),
            "sw": torch.rand(B, L),
            "y_atom": (torch.rand(B, L, 3) > 0.5).float(),
            "y_joint": (torch.rand(B, L) > 0.5).float(),
        }
        total, parts = total_loss(out, batch, use_align=True, use_aux=True,
                                  use_atom=False, use_joint=False)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("align", parts)


class TestPermAsymmetry(unittest.TestCase):
    def test_overprediction_is_penalized_more(self):
        z = torch.zeros(4)
        zhat = torch.tensor([-2.0, -0.5, 0.5, 1.5])
        sym = align_score_log(z, zhat)
        asym = align_score_log(z, zhat, over_weight=3.0, under_weight=1.0)
        # d<0 不变，d>0 得分更低
        self.assertAlmostEqual(float(sym[0]), float(asym[0]), places=6)
        self.assertAlmostEqual(float(sym[1]), float(asym[1]), places=6)
        self.assertLess(float(asym[2]), float(sym[2]))
        self.assertLess(float(asym[3]), float(sym[3]))


class TestPhysicsLoss(unittest.TestCase):
    class _Scaler:
        def __init__(self):
            self.names = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
                          "RXO", "RT", "DEVI", "AZIM", "BIT", "CASE"]
            self.mean = [0.0] * len(self.names)
            self.std = [1.0] * len(self.names)

    def test_porosity_prior_zero_on_exact_match(self):
        s = self._Scaler()
        ac = ACMA_US_M + 0.2 * (ACF_US_M - ACMA_US_M)
        den = 2.65 - 0.2 * (2.65 - 1.0)
        x = torch.zeros(2, len(s.names))
        x[:, 4] = ac
        x[:, 5] = den
        x[:, 6] = 20.0
        batch = {"x": x, "mask": torch.ones(2, 3), "y_atom": torch.zeros(2, 3)}
        loss = physics_porosity_loss(torch.full((2,), 20.0), batch,
                                     row_scaler=s, feature_names=s.names)
        self.assertLess(float(loss), 1e-6)

    def test_physics_loss_requires_context(self):
        """审查 H2/M4：lam_phys>0 时缺 x/row_scaler 必须显式报错，不得静默返回 0。"""
        out = {"por": torch.full((4,), 20.0), "perm_z": torch.zeros(4),
               "sw": torch.full((4,), 80.0),
               "q_atom_logit": torch.zeros(4, 3), "q_joint_logit": torch.zeros(4)}
        batch = {"por": torch.full((4,), 20.0), "perm_z": torch.zeros(4),
                 "sw": torch.full((4,), 80.0),
                 "mask": torch.ones(4, 3), "y_atom": torch.zeros(4, 3),
                 "y_joint": torch.zeros(4)}
        with self.assertRaises(ValueError):
            total_loss(out, batch, use_align=False, use_aux=False,
                       use_atom=False, use_joint=False, lam_phys=0.05)

    def test_total_loss_reports_phys_part(self):
        s = self._Scaler()
        out = {"por": torch.full((4,), 20.0), "perm_z": torch.zeros(4),
               "sw": torch.full((4,), 80.0),
               "q_atom_logit": torch.zeros(4, 3), "q_joint_logit": torch.zeros(4)}
        batch = {"por": torch.full((4,), 20.0), "perm_z": torch.zeros(4),
                 "sw": torch.full((4,), 80.0),
                 "mask": torch.ones(4, 3), "y_atom": torch.zeros(4, 3),
                 "y_joint": torch.zeros(4)}
        x = torch.zeros(4, len(s.names))
        x[:, 4] = ACMA_US_M + 0.2 * (ACF_US_M - ACMA_US_M)
        x[:, 5] = 2.65 - 0.2 * (2.65 - 1.0)
        x[:, 6] = 20.0
        batch["x"] = x
        total, parts = total_loss(out, batch, use_align=False, use_aux=False,
                                  use_atom=False, use_joint=False,
                                  lam_phys=0.05, row_scaler=s,
                                  feature_names=s.names)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("phys", parts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
