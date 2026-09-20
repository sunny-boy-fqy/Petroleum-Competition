"""E5/P0 端到端（**torch 门控**）：POR 头入口脚本真能跑出可复算的分项与消融。

用**合成井** + 随机初始化冻结骨干（不训练骨架）跑完整链路：折内标尺 → 逐行隐状态 →
两阶段 POR 头训练（内折早停）→ 分项指标 → 配对 CI → 参数化消融 → 报告与 Gate。

断言的硬契约（E5/P0 §7）：
  * 主判据是**连续切片** Acc，且占位行单独报（`placeholder_rows.hit_rate`）；
  * 参数化消融 ≥3 臂且含禁用反例 `plus_softplus`（`can_represent_zero=False`、`allowed=False`）；
  * 随机初始化骨干时**不得判 PASS**：`passed is None`、`baseline_unavailable is True`；
  * 仓库 `versions/candidates.json` 不被写。
"""
from __future__ import annotations

import contextlib
import hashlib
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

REPO_CANDIDATES = V4 / "versions" / "candidates.json"
ARCH_KWARGS = {"patch_len": 16, "stride": 8, "d_model": 16, "n_layers": 1, "n_heads": 4,
               "max_tokens": 64}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(V4 / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE5PorPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr_wells + va_wells, n_rows=40, seed=11)
        cls.mod = _load("v4_e5_head_por", "E5/code/head_por.py")
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
        return json.loads((self.reports / "E5_P0_gate.json").read_text(encoding="utf-8"))

    def test_smoke_reports_and_contracts(self):
        before = _sha(REPO_CANDIDATES) if REPO_CANDIDATES.is_file() else None
        gate = self._run("--ablation")
        for name in ("E5_por.json", "E5_por_param_ablation.json", "E5_P0_gate.json",
                     "training_time_log.json", "E5_P0_gate_prereg.json"):
            self.assertTrue((self.reports / name).is_file(), name)
        self.assertTrue((self.scalers / "E5_por_fold0.json").is_file())
        m = json.loads((self.reports / "E5_por.json").read_text(encoding="utf-8"))
        self.assertEqual(m["target"], "POR")
        self.assertTrue(m["selection_score_only"])
        self.assertTrue(m["backbone_random_init"])
        self.assertIn("placeholder_rows", m)
        self.assertIsInstance(m["paired_ci"][0], float)
        self.assertEqual(len(m["paired_ci"]), 2)
        self.assertGreater(m["n_rows"], 0)
        self.assertLessEqual(m["cont_slice_rows"], m["n_rows"])
        self.assertGreater(m["n_wells"], 0)
        self.assertIn("por_lt_0p1_n", m)
        self.assertTrue(gate["baseline_unavailable"])
        self.assertIsNone(gate["passed"])
        self.assertFalse(gate["exploratory"] is None)
        self.assertEqual(gate["gate_id"], "E5_P0_gate")
        self.assertTrue(gate["checks"]["contract_ok"])
        self.assertTrue(gate["checks"]["no_label_leak"])
        self.assertTrue(gate["checks"]["ablation_table_complete"])
        self.assertFalse(gate["nogo"], "预检（passed=None）不应记为 NO-GO")
        # 冻结骨干未被改动：scaler 只由训练折写、且报告留痕
        sc = json.loads((self.scalers / "E5_por_fold0.json").read_text(encoding="utf-8"))
        self.assertEqual(sc["fold"], 0)
        self.assertTrue(sc["train_wells"])
        self.assertTrue(set(sc["train_wells"]).isdisjoint(set(sc["val_wells"])))
        if before is not None:
            self.assertEqual(_sha(REPO_CANDIDATES), before)

    def test_ablation_table_has_forbidden_arm(self):
        self._run("--ablation")
        a = json.loads((self.reports / "E5_por_param_ablation.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(a["rows"]), 3)
        arms = {r["param"]: r for r in a["rows"]}
        self.assertEqual(set(arms), {"sigmoid", "softplus_shift", "linear", "plus_softplus"})
        self.assertTrue(arms["sigmoid"]["can_represent_zero"])
        self.assertTrue(arms["softplus_shift"]["can_represent_lt_0p1"])
        bad = arms["plus_softplus"]
        self.assertFalse(bad["can_represent_zero"])
        self.assertFalse(bad["can_represent_lt_0p1"])
        self.assertFalse(bad["allowed"])
        self.assertEqual(a["forbidden_arm"], "plus_softplus")

    def test_init_mode_flag_is_recorded(self):
        """`--init-mode zero_point_one` 是**反面对照**（应记录在报告里，可用于对照实验）。"""
        gate = self._run("--init-mode", "zero_point_one")
        m = json.loads((self.reports / "E5_por.json").read_text(encoding="utf-8"))
        self.assertEqual(m["init_mode"], "zero_point_one")
        self.assertIsNone(gate["passed"])

    def test_joint_mode_refused(self):
        argv = ["--smoke", "--training-mode", "joint", "--cache-root", str(self.cache),
                "--reports-dir", str(self.reports), "--run-root", str(self.runs)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = self.mod.main(argv)
        self.assertEqual(rc, 6, buf.getvalue()[-500:])


class TestE5ScriptContract(unittest.TestCase):
    def test_run_train_dispatches_e5(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn("E5)", src)
        self.assertIn("E5/code/head_por.py", src)

    def test_help_lists_e5_knobs(self):
        import subprocess
        out = subprocess.run([sys.executable, str(V4 / "E5" / "code" / "head_por.py"),
                              "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        for flag in ("--backbone-ckpt", "--arch-kwargs", "--param", "--por-max", "--init-mode",
                     "--boundary-weight", "--training-mode", "--tau-por", "--ablation",
                     "--ablation-folds"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
