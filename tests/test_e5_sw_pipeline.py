"""E5/P2 端到端（**torch 门控**）：SW 头的单尺度契约、有效/占位分项与禁止尺度反例。

断言要点（E5/P2 §7）：
  * `scale_check` 全绿：只做 [0,100] 软裁剪，**不** [0,1]、**不** ×100、`SW_SMALL_BRANCH` 关闭；
  * 预测范围落在 [0,100] 且**明显超出 [0,1]**（证明没有把百分数压成 0–1）；
  * 有效行与占位行**分项**上报；无占位行时 `sw_placeholder_acc is None`（不伪造 0）；
  * `sw_mu/sw_sigma` 每折留痕且来源为训练折；
  * 禁止尺度臂（`normalized_0_1` / `times_100`）被标记 `forbidden` 且预测范围暴露其尺度错误；
  * 随机初始化骨干时 `passed is None` / `baseline_unavailable is True`。
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

ARCH_KWARGS = {"patch_len": 16, "stride": 8, "d_model": 16, "n_layers": 1, "n_heads": 4,
               "max_tokens": 64}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(V4 / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE5SwPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr_wells + va_wells, n_rows=40, seed=11)
        sys.path.insert(0, str(V4 / "E5" / "code"))
        cls.mod = _load("v4_e5_head_sw", "E5/code/head_sw.py")
        cls.reports = root / "reports"
        cls.runs = root / "runs"
        cls.scalers = root / "scalers"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str) -> dict:
        argv = ["--smoke", "--folds", "0", "--max-wells", "6", "--epochs", "2",
                "--arch", "patchtf", "--arch-kwargs", json.dumps(ARCH_KWARGS),
                "--chunk", "96", "--overlap", "32",
                "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
                "--run-root", str(self.runs), "--scalers-dir", str(self.scalers), *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.mod.main(argv)
        self.assertEqual(rc, 0, buf.getvalue()[-2000:])
        return json.loads((self.reports / "E5_P2_gate.json").read_text(encoding="utf-8"))

    def test_smoke_reports_and_scale_contract(self):
        gate = self._run("--ablation")
        for name in ("E5_sw.json", "E5_sw_scale_ablation.json", "E5_P2_gate.json",
                     "training_time_log.json"):
            self.assertTrue((self.reports / name).is_file(), name)
        m = json.loads((self.reports / "E5_sw.json").read_text(encoding="utf-8"))
        self.assertEqual(m["target"], "SW")
        sc = m["scale_check"]
        self.assertTrue(sc["ok"])
        self.assertFalse(sc["sw_small_branch"])
        self.assertFalse(sc["global_clip_0_1"])
        self.assertFalse(sc["multiply_100"])
        self.assertTrue(sc["soft_clip_0_100"])
        self.assertFalse(sc["interpolation"])
        self.assertIs(C.SW_SMALL_BRANCH, False)
        self.assertTrue(m["contract"]["only_soft_clip_0_100"])
        self.assertTrue(m["contract"]["no_interpolation"])
        lo, hi = m["sw_range"]
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 100.0)
        self.assertGreater(hi, 1.0, "预测必须在百分数尺度上（不是 [0,1] 归一化）")
        self.assertEqual(len(m["paired_ci"]), 2)
        self.assertEqual(len(m["per_well_delta"]), m["n_wells"])
        self.assertTrue(m["no_label_leak"])
        self.assertTrue(m["selection_score_only"])
        self.assertTrue(gate["baseline_unavailable"])
        self.assertIsNone(gate["passed"])
        self.assertTrue(gate["checks"]["sw_scale_unit_test"])
        self.assertTrue(gate["checks"]["sw_small_branch_off"])
        self.assertTrue(gate["checks"]["mu_sigma_train_fold_only"])
        self.assertFalse(gate["nogo"])

    def test_placeholder_and_valid_rows_reported_separately(self):
        self._run()
        m = json.loads((self.reports / "E5_sw.json").read_text(encoding="utf-8"))
        self.assertIn("n_valid_rows", m)
        self.assertIn("n_placeholder_rows", m)
        self.assertLessEqual(m["n_valid_rows"] + m["n_placeholder_rows"], m["n_rows"])
        if m["n_placeholder_rows"] == 0:
            self.assertIsNone(m["sw_placeholder_acc"], "无占位行时必须给 None 而不是 0")
        else:
            self.assertIsNotNone(m["sw_placeholder_acc"])
        for fold in m["folds_detail"]:
            self.assertGreater(fold["sw_sigma"], 0.0)
            self.assertIn("sw_valid_acc", fold)

    def test_forbidden_scale_arms_expose_cost(self):
        self._run("--ablation")
        a = json.loads((self.reports / "E5_sw_scale_ablation.json")
                       .read_text(encoding="utf-8"))
        arms = {r["arm"]: r for r in a["rows"]}
        self.assertTrue({"label", "normalized_0_1", "times_100"}.issubset(arms))
        self.assertFalse(arms["label"]["forbidden"])
        self.assertTrue(arms["normalized_0_1"]["forbidden"])
        self.assertTrue(arms["times_100"]["forbidden"])
        n_lo, n_hi = arms["normalized_0_1"]["sw_range"]
        self.assertLessEqual(n_hi, 1.0, "[0,1] 归一化臂的预测范围应被压到 0–1")
        t_lo, t_hi = arms["times_100"]["sw_range"]
        self.assertGreaterEqual(t_lo, 1.0)
        self.assertEqual(a["recommended"], "label")

    def test_sw_mu_sigma_default_from_train_fold(self):
        self._run()
        m = json.loads((self.reports / "E5_sw.json").read_text(encoding="utf-8"))
        per_fold = m["sw_mu_sigma_per_fold"]
        self.assertTrue(per_fold)
        for v in per_fold.values():
            self.assertEqual(v["source"], "train_fold")
            self.assertGreater(v["sw_sigma"], 0.0)

    def test_cli_override_is_recorded(self):
        self._run("--sw-mu", "80.0", "--sw-sigma", "12.0")
        m = json.loads((self.reports / "E5_sw.json").read_text(encoding="utf-8"))
        for v in m["sw_mu_sigma_per_fold"].values():
            self.assertEqual(v["source"], "cli")
            self.assertAlmostEqual(v["sw_mu"], 80.0, places=6)


class TestE5SwRouting(unittest.TestCase):
    def test_run_train_routes_sw(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn("E5/code/head_sw.py", src)

    def test_help_lists_sw_knobs(self):
        import subprocess
        out = subprocess.run([sys.executable, str(V4 / "E5" / "code" / "head_sw.py"),
                              "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        for flag in ("--sw-mu", "--sw-sigma", "--clip-lo", "--clip-soft-hi", "--tau-sw",
                     "--tau-source", "--valid-lo", "--valid-hi"):
            self.assertIn(flag, out.stdout, flag)

    def test_apply_scale_arm_numpy_path(self):
        import numpy as np
        lo = self
        sys.path.insert(0, str(V4 / "E5" / "code"))
        mod = _load("v4_e5_head_sw_arm", "E5/code/head_sw.py")
        x = np.array([50.0, 99.9])
        self.assertTrue(np.allclose(mod.apply_scale_arm(x, "label"), x))
        self.assertTrue(np.allclose(mod.apply_scale_arm(x, "normalized_0_1"), [0.5, 0.999]))
        self.assertTrue(np.allclose(mod.apply_scale_arm(x, "times_100"), [100.0, 100.0]))
        with self.assertRaises(ValueError):
            mod.apply_scale_arm(x, "log")
        del lo


if __name__ == "__main__":
    unittest.main()
