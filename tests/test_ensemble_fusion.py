"""WP3 回归：先融合再硬切换 / 同源剔除（纯 numpy）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.ensemble import blend as BL  # noqa: E402


def _members(n: int = 6):
    a = {"por": np.full(n, 20.0), "perm_z": np.full(n, 1.0),
         "sw": np.full(n, 50.0), "q_atom": np.tile([0.9, 0.1, 0.1], (n, 1))}
    b = {"por": np.full(n, 0.1), "perm_z": np.full(n, 1.0),
         "sw": np.full(n, 50.0), "q_atom": np.tile([0.1, 0.1, 0.1], (n, 1))}
    return {"a": a, "b": b}


class TestFuseThenSwitch(unittest.TestCase):
    def test_fused_switch_does_not_average_gated_outputs(self):
        preds = _members(4)
        # 融合后 q_por=0.5，τ=0.6 -> 不切；若先各自切换再平均会得到 (0.1+20)/2=10.05
        q = {"a": preds["a"], "b": preds["b"]}
        # 用等权融合：q_por=0.5 < 0.6 -> 全部连续；por 融合=10.05
        out = BL.fuse_and_decode(q, {"a": 0.5, "b": 0.5}, tau=[0.6, 0.5, 0.5])
        self.assertTrue(np.allclose(out[:, 0], 10.05))
        # τ=0.4 -> q=0.5>0.4 全部切 0.1（不是混合）
        out2 = BL.fuse_and_decode(q, {"a": 0.5, "b": 0.5}, tau=[0.4, 0.5, 0.5])
        self.assertTrue(np.allclose(out2[:, 0], 0.1))

    def test_prune_correlated_removes_duplicate(self):
        a = {"por": np.linspace(0, 1, 100)}
        b = {"por": a["por"] * 2.0}
        c = {"por": np.sin(np.linspace(0, 6, 100))}
        res = BL.prune_correlated({"a": a, "b": b, "c": c}, threshold=0.999)
        self.assertEqual(res["kept"], ["a", "c"])
        self.assertEqual(res["dropped"][0]["member"], "b")

    def test_fuse_and_decode_with_action_table(self):
        preds = _members(3)
        # 无 action table 时回退 τ
        out = BL.fuse_and_decode(preds, {"a": 1.0, "b": 0.0}, tau=None)
        self.assertEqual(out.shape, (3, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
