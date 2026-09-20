"""E5/P1 端到端（**torch 门控**）：PERM 头入口脚本的契约、尾部报告与消融。

断言要点（E5/P1 §7）：
  * `E5_perm.json` / `E5_perm_tail.json` / `E5_P1_gate.json` 落盘，且 tail 报告有分桶结论；
  * 契约：`perm_z` 有限、`10**perm_z > 0`；`z_init` 用训练折中位数（非 0）；
  * 逐井配对 CI 与 `n_wells` 一致；连续切片 Acc 与基线可比较；
  * 随机初始化骨干时 `passed is None` / `baseline_unavailable is True`（不判 PASS）；
  * `run_train.sh --stage E5 --target perm` 路由到 `head_perm.py`。
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
from src.portability import HAS_TORCH  # noqa: E402

ARCH_KWARGS = {"patch_len": 16, "stride": 8, "d_model": 16, "n_layers": 1, "n_heads": 4,
               "max_tokens": 64}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(V4 / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE5PermPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr_wells + va_wells, n_rows=40, seed=11)
        # `e5_common` 与脚本同目录，需要先把它所在目录放进 sys.path
        sys.path.insert(0, str(V4 / "E5" / "code"))
        cls.mod = _load("v4_e5_head_perm", "E5/code/head_perm.py")
        cls.reports = root / "reports"
        cls.runs = root / "runs"
        cls.scalers = root / "scalers"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str) -> dict:
        argv = ["--smoke", "--folds", "0", "--max-wells", "6", "--epochs", "2",
                "--arch", "patchtf", "--arch-kwargs", json.dumps(ARCH_KWARGS),
                "--chunk", "96", "--overlap", "32", "--tail-bins", "4",
                "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
                "--run-root", str(self.runs), "--scalers-dir", str(self.scalers), *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.mod.main(argv)
        self.assertEqual(rc, 0, buf.getvalue()[-2000:])
        return json.loads((self.reports / "E5_P1_gate.json").read_text(encoding="utf-8"))

    def test_smoke_reports_and_contracts(self):
        gate = self._run("--ablation")
        for name in ("E5_perm.json", "E5_perm_tail.json", "E5_perm_param_ablation.json",
                     "E5_P1_gate.json", "training_time_log.json"):
            self.assertTrue((self.reports / name).is_file(), name)
        m = json.loads((self.reports / "E5_perm.json").read_text(encoding="utf-8"))
        self.assertEqual(m["target"], "PERM")
        self.assertTrue(m["contract"]["finite"])
        self.assertTrue(m["contract"]["perm_positive"])
        self.assertNotAlmostEqual(float(m["z_init"]), 0.0, places=6)
        self.assertEqual(len(m["paired_ci"]), 2)
        self.assertEqual(len(m["per_well_delta"]), m["n_wells"])
        self.assertIn("frac_abs_dz_lt_1", m)
        self.assertTrue(m["no_label_leak"])
        self.assertTrue(m["selection_score_only"])
        tail = json.loads((self.reports / "E5_perm_tail.json").read_text(encoding="utf-8"))
        self.assertTrue(tail["bins"])
        self.assertIsInstance(tail["monotone_consistent"], bool)
        for b in tail["bins"]:
            self.assertIn("official_acc", b)
            self.assertIn("aligned_loss_mean", b)
        self.assertTrue(gate["baseline_unavailable"])
        self.assertIsNone(gate["passed"])
        self.assertTrue(gate["checks"]["perm_positive_ok"])
        self.assertTrue(gate["checks"]["perm_finite_ok"])
        self.assertTrue(gate["checks"]["tail_report_written"])
        self.assertTrue(gate["checks"]["ablation_table_complete"])
        self.assertFalse(gate["nogo"], "预检不得记为 NO-GO")

    def test_ablation_arms_include_bucket(self):
        self._run("--ablation")
        a = json.loads((self.reports / "E5_perm_param_ablation.json")
                       .read_text(encoding="utf-8"))
        arms = {r["arm"] for r in a["rows"]}
        self.assertTrue({"tanh", "clip", "linear"}.issubset(arms))
        self.assertIn("tanh+bucket", arms)
        self.assertGreaterEqual(len(a["rows"]), 3)

    def test_bucket_head_reports_match_contract(self):
        gate = self._run("--n-buckets", "6")
        m = json.loads((self.reports / "E5_perm.json").read_text(encoding="utf-8"))
        self.assertEqual(m["n_buckets"], 6)
        self.assertTrue(m["contract"]["perm_positive"])
        self.assertIsNone(gate["passed"])

    def test_linear_arm_runs(self):
        self._run("--z-output", "linear")
        m = json.loads((self.reports / "E5_perm.json").read_text(encoding="utf-8"))
        self.assertEqual(m["z_output"], "linear")
        self.assertTrue(m["contract"]["finite"])


class TestE5Routing(unittest.TestCase):
    def test_run_train_routes_target(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn("E5/code/head_perm.py", src)
        self.assertIn("E5/code/head_sw.py", src)
        self.assertIn('"--target"', src)

    def test_help_lists_perm_knobs(self):
        import subprocess
        out = subprocess.run([sys.executable, str(V4 / "E5" / "code" / "head_perm.py"),
                              "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        for flag in ("--z-output", "--clip-min", "--clip-max", "--init-z-median", "--aux",
                     "--n-buckets", "--quantile-heads", "--tail-bins", "--tau-perm"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
