"""E7/P1 测试（**口径层，无 torch**）：解码算子 + 解码搜索端到端。

算子层（`src/inference/decode.py`）：
  * bias/收缩/分位收缩是**逐目标独立**的，且 SW 始终只做 [0,100] 软裁剪（尺度收据全绿）；
  * 期望值动作表给出逐箱"原子 vs 连续"的官方得分与单调 τ；
  * 敏感性报告给出平坦区中点（不在平坦区就返回 0，宁可不调）；
  * **原子优先级**：正确顺序（先连续后处理、后原子硬切换）下命中行逐位相同。

搜索层（`E7/code/decode_search.py`）：
  * 只在给定（内折）OOF 上搜索，采纳需"总分上升 + 配对 CI 下界 > 0"；
  * 合成一份**有系统偏移**的 OOF → bias 被选中且总分上升；
  * `--smoke` 不写仓库 `versions/configs/decode_v1.json`；缺 OOF → 退出码 4。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.inference import decode as DEC  # noqa: E402

SCRIPT = V4 / "E7" / "code" / "decode_search.py"
N_WELLS, ROWS_PER_WELL = 6, 20
N_ROWS = N_WELLS * ROWS_PER_WELL


def _toy(n: int = 100, seed: int = 0):
    rng = np.random.RandomState(seed)
    y = np.column_stack([10 + 4 * rng.rand(n), 40 + 20 * rng.rand(n),
                         60 + 20 * rng.rand(n)])
    mask = np.ones((n, 3), dtype=bool)
    return y, mask


class TestDecodeOps(unittest.TestCase):
    def test_bias_and_soft_clip(self):
        y, mask = _toy()
        cont = y + np.array([1.0, 0.0, 30.0])
        fixed = DEC.apply_bias(cont, {"POR": -1.0, "PERM": 0.0, "SW": -30.0})
        self.assertLess(abs(fixed[:, 0] - y[:, 0]).max(), 1e-9)
        self.assertLess(abs(fixed[:, 2] - y[:, 2]).max(), 1e-9)
        # 偏移后 SW 仍被软裁剪在 [0,100]
        huge = DEC.apply_bias(cont, {"POR": 0.0, "PERM": 0.0, "SW": 100.0})
        self.assertLessEqual(float(huge[:, 2].max()), 100.0)

    def test_shrink_keeps_center(self):
        y, _ = _toy()
        mu = {"POR": float(np.median(y[:, 0])), "PERM": float(np.median(y[:, 1])),
              "SW": float(np.median(y[:, 2]))}
        out = DEC.apply_shrink(y, {"POR": 0.5, "PERM": 0.5, "SW": 0.5}, centers=mu)
        for i, t in enumerate(C.TARGETS):
            self.assertAlmostEqual(float(np.median(out[:, i])), mu[t], places=6)

    def test_quantile_shrink_is_monotone_and_identity_at_zero(self):
        y, _ = _toy()
        ident = DEC.quantile_shrink(y, {t: 0.0 for t in C.TARGETS}, y)
        self.assertTrue(np.allclose(ident, y))
        out = DEC.quantile_shrink(y, {"POR": 1.0, "PERM": 0.0, "SW": 0.0}, y)
        order_in = np.argsort(y[:, 0], kind="mergesort")
        order_out = np.argsort(out[:, 0], kind="mergesort")
        self.assertTrue(np.array_equal(order_in, order_out), "分位映射必须保序（单调）")

    def test_sw_scale_receipt(self):
        r = DEC.sw_scale_receipt()
        self.assertTrue(r["ok"])
        self.assertFalse(r["global_clip_0_1"])
        self.assertFalse(r["multiply_100"])
        self.assertTrue(r["soft_clip_0_100"])

    def test_expected_value_table_monotone_tau(self):
        rng = np.random.RandomState(1)
        n = 400
        y = np.column_stack([np.where(rng.rand(n) < 0.5, 0.1, 12 + rng.rand(n)),
                             50 + rng.rand(n), 60 + rng.rand(n)])
        q1 = np.clip(0.4 + 0.5 * (y[:, 0] == 0.1) + 0.05 * rng.rand(n), 0, 1)
        q = np.column_stack([q1, rng.rand(n), rng.rand(n)])
        cont = y.copy()
        cont[:, 0] = 12 + rng.rand(n)              # 连续头在原子行上不准
        mask = np.ones((n, 3), dtype=bool)
        rep = DEC.expected_value_table(y, cont, q, mask, 0, n_bins=5)
        self.assertEqual(rep["target"], "POR")
        self.assertEqual(len(rep["bins"]), 5)
        self.assertIsInstance(rep["monotone"], bool)
        if rep["monotone"] and rep["tau_from_table"] is not None:
            self.assertGreaterEqual(rep["tau_from_table"], 0.0)
            self.assertLessEqual(rep["tau_from_table"], 1.0)

    def test_sensitivity_flat_midpoint(self):
        y, mask = _toy(200)
        cont = y.copy()
        rep = DEC.sensitivity_report(y, cont, mask, 0, tol=0.01)
        self.assertTrue(rep["flat_hit"])
        self.assertIsNotNone(rep["flat_region"])
        self.assertAlmostEqual(rep["flat_midpoint"], 0.0, places=6)

    def test_atom_priority_holds_for_correct_order(self):
        y, mask = _toy(200)
        q = np.full((200, 3), 0.9)
        tau = np.full(3, 0.5)
        after = DEC.apply_bias(y, {"POR": 1.0, "PERM": 1.0, "SW": 1.0})
        rep = DEC.assert_atom_priority(after, y, q, tau)
        self.assertTrue(rep["ok"])
        self.assertGreater(rep["n_atom_rows"], 0)


def _make_oof(path: Path, *, biased: bool = True, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    y = np.column_stack([10 + 4 * rng.rand(N_ROWS), 40 + 20 * rng.rand(N_ROWS),
                         60 + 20 * rng.rand(N_ROWS)])
    cont = y.copy()
    if biased:                                     # POR/SW 加性偏移；PERM 乘性（log10）
        cont[:, 0] += 0.9
        cont[:, 1] *= 10.0 ** 0.3
        cont[:, 2] += 4.0
    np.savez_compressed(path, cont=cont, q_atom=rng.rand(N_ROWS, 3),
                        q_joint=rng.rand(N_ROWS), y_true=y,
                        mask=np.ones((N_ROWS, 3)),
                        well_index=np.repeat(np.arange(N_WELLS), ROWS_PER_WELL),
                        fold_of_row=np.zeros(N_ROWS, dtype="int64"),
                        tau_row=np.full((N_ROWS, 3), 0.5),
                        well_ids=np.asarray([f"w{i}" for i in range(N_WELLS)],
                                            dtype=object))



    def test_apply_frozen_decode_config(self):
        from src.inference import decode as DEC
        pred = np.tile(np.array([[10.0, 1.0, 50.0]]), (4, 1))
        cfg = DEC.DecodeConfig(
            bias={"POR": 0.5, "PERM": 0.0, "SW": -1.0},
            shrink={"POR": 1.0, "PERM": 1.0, "SW": 1.0},
            shrink_centers={"POR": 10.0, "PERM": 1.0, "SW": 50.0})
        out = DEC.apply_decode_config(pred, cfg)
        np.testing.assert_allclose(out[0, 0], 10.5)
        np.testing.assert_allclose(out[0, 1], 1.0)
        np.testing.assert_allclose(out[0, 2], 49.0)


class TestDecodeSearch(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        (self.reports / "training_time_log.json").write_text(json.dumps(
            {"folds": [{"fold": 0, "seconds": 3.0}]}), encoding="utf-8")
        self.oof = self.root / "inner_oof.npz"
        _make_oof(self.oof, biased=True)
        self.cfg = self.root / "decode_v1.json"

    def tearDown(self):
        self._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(SCRIPT), "--oof", str(self.oof),
               "--reports-dir", str(self.reports), "--run-root", str(self.root / "runs"),
               "--out-config", str(self.cfg), "--candidates", str(self.root / "cands.json"),
               *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1200:],
                                              proc.stderr[-1200:])
        rep_path = self.reports / "E7_decode_search.json"
        return json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.is_file() else {}

    def test_biased_oof_gains_from_bias_correction(self):
        # 默认（E7 规格的紧网格）不采纳是可以接受的结果：宁可不调
        tight = self._run("--smoke")
        self.assertFalse(tight["bias"]["accepted"])
        # 放宽搜索范围后（规格外，但机制必须可用）应能修掉系统偏移
        rep = self._run("--smoke", "--bias-por-range", "1.0", "--bias-perm-range", "0.5",
                        "--bias-sw-range", "8.0", "--bias-steps", "21")
        self.assertTrue(rep["inner_only"])
        self.assertGreater(rep["final"]["total"], rep["base"]["total"])
        self.assertTrue(rep["bias"]["accepted"], rep["bias"])
        for t in C.TARGETS:
            self.assertIn(t, rep["bias"]["selected"])
            self.assertIn(t, rep["bias"]["sensitivity"])
        self.assertTrue((self.cfg).is_file())
        cfg = json.loads(self.cfg.read_text(encoding="utf-8"))
        self.assertTrue(cfg["inner_only"])
        self.assertIn("selected_on", cfg)
        gate = json.loads((self.reports / "E7_P1_gate.json").read_text(encoding="utf-8"))
        for key in ("inner_only_selection", "atom_priority_preserved", "sw_single_label_scale",
                    "sensitivity_reported", "contract_ok"):
            self.assertTrue(gate["checks"][key], key)
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")

    def test_smoke_does_not_write_repo_config(self):
        target = V4 / "versions" / "configs" / "decode_v1.json"
        before = target.read_text(encoding="utf-8") if target.is_file() else None
        self._run("--smoke")
        after = target.read_text(encoding="utf-8") if target.is_file() else None
        self.assertEqual(before, after, "smoke 不得写仓库 decode_v1.json")
        # 显式给了 --out-config 时写该路径（测试用 tmp），仓库文件保持不变
        self.assertTrue(self.cfg.is_file())
        cfg = json.loads(self.cfg.read_text(encoding="utf-8"))
        self.assertTrue(cfg["inner_only"])

    def test_expected_decode_off_records_disabled(self):
        rep = self._run("--smoke", "--expected-decode", "off")
        self.assertIsNone(rep["expected_value"])
        self.assertTrue(all(v is False for v in rep["selected_config"]["expected_value"].values()))

    def test_missing_oof_fails_with_4(self):
        self.oof.unlink()
        self._run("--smoke", expect_rc=4)


class TestDecodeConfigRoundTrip(unittest.TestCase):
    def test_save_load_roundtrip(self):
        cfg = DEC.DecodeConfig(bias={"POR": 0.001, "PERM": -0.02, "SW": 0.0},
                               shrink={"POR": 0.95, "PERM": 1.0, "SW": 1.05},
                               expected_value={"POR": True, "PERM": False, "SW": False},
                               tau={"POR": 0.6, "PERM": 0.5, "SW": 0.5},
                               selected_on="inner_oof.npz")
        with tempfile.TemporaryDirectory() as td:
            p = DEC.save_decode_config(cfg, Path(td) / "decode_v1.json")
            back = DEC.load_decode_config(p)
        self.assertEqual(back.bias["POR"], 0.001)
        self.assertEqual(back.shrink["SW"], 1.05)
        self.assertTrue(back.expected_value["POR"])
        self.assertTrue(back.inner_only)

    def test_script_help(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--oof", "--out-config", "--expected-decode", "--quantile-shrink",
                     "--n-bins", "--sensitivity-tol", "--min-gain", "--baseline-oof"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
