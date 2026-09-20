"""E9/P1-confirm 测试（**torch 门控**）：16 井确认折复验的纪律与产物。

断言要点：
  * **不重训**：只用给到的注册权重推理（多折时按连续头平均）；
  * **breakdown 判据**：`-delta > threshold` 才判 breakdown，并在 `breakdown_candidates` 里点名；
  * **负对照**：标签打乱后必须给出分数（不静默跳过）；
  * **诚实标注**：`v1_exposed=true` 与 `not_independent_confirmation=true` 必须为真，
    且报告里写明"这不是独立确认"；
  * 缺确认折文件 → 显式 `status="missing_confirm_folds"`（**不假装做过**），
    smoke 下 rc 0、正式运行 rc 4。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

SCRIPT = V4 / "E9" / "code" / "confirm_check.py"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(V4 / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestE9ConfirmMissingFolds(unittest.TestCase):
    def test_missing_folds_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = subprocess.run([sys.executable, str(SCRIPT), "--folds-confirm",
                                  str(root / "nope.json"), "--reports-dir", str(root),
                                  "--smoke"], capture_output=True, text=True, timeout=300)
            self.assertEqual(out.returncode, 0, out.stderr[-600:])
            rep = json.loads((root / "E9_confirm.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["status"], "missing_confirm_folds")
            self.assertEqual(rep["decision"], "not_run")
            self.assertTrue(rep["v1_exposed"])
            self.assertTrue(rep["not_independent_confirmation"])
            gate = json.loads((root / "E9_confirm_gate.json").read_text(encoding="utf-8"))
            self.assertFalse(gate["checks"]["confirm_no_breakdown"])
            # 非 smoke 时必须以 4 明确失败（而不是静默成功）
            out2 = subprocess.run([sys.executable, str(SCRIPT), "--folds-confirm",
                                   str(root / "nope.json"), "--reports-dir", str(root)],
                                  capture_output=True, text=True, timeout=300)
            self.assertEqual(out2.returncode, 4)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE9ConfirmRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=4, n_train=4)
        cls.wells = list(dict.fromkeys(tr_wells + va_wells))[:4]
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, cls.wells, n_rows=40, seed=11)
        from src.data import row_dataset as RD
        fit = RD.fit_scalers_from_wells(cls.wells, cls.cache, spec=None)
        import torch
        from src.models.row_mlp import build_model
        from src.training import checkpoint as CK
        torch.manual_seed(0)
        model = build_model(32, hidden=16, layers=1)
        cls.ckpt = root / "pd1_fold0.pt"
        CK.save_checkpoint(cls.ckpt, model, meta={
            "row_scaler": fit["scaler"].to_dict(),
            "target_scalers": dict(fit["target"]),
            "model": {"arch": "RowMLP", "n_features": 32, "hidden": 16, "layers": 1,
                      "dropout": 0.0},
            "tau_atom": [0.5, 0.5, 0.5], "scalers_fitted_on": "train_fold_only"}, bf16=False)
        cls.root = root
        cls.reports = root / "reports"
        cls.reports.mkdir(parents=True, exist_ok=True)
        (cls.reports / "training_time_log.json").write_text("{}", encoding="utf-8")
        (root / "confirm_folds.json").write_text(json.dumps(
            {"wells": cls.wells, "source": "v2/E0（16 井确认折的本地子集）"}),
            encoding="utf-8")
        (cls.reports / "E9_validation_report.json").write_text(json.dumps(
            {"candidates": [{"candidate_id": "PD1", "oof_total": 95.0}]}), encoding="utf-8")
        (cls.reports / "E9_leakage_audit.json").write_text("{}", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(SCRIPT), "--folds-confirm",
               str(self.root / "confirm_folds.json"),
               "--checkpoints", f"PD1={self.ckpt}",
               "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
               "--smoke", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        return json.loads((self.reports / "E9_confirm.json").read_text(encoding="utf-8"))

    def test_confirm_run_reports_and_flags(self):
        rep = self._run()
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["n_wells"], len(self.wells))
        r = rep["results"][0]
        self.assertEqual(r["candidate_id"], "PD1")
        self.assertEqual(r["status"], "ok")
        for key in ("total", "por", "perm", "sw"):
            self.assertIn(key, r["confirm"])
        self.assertIsNotNone(r["oof_total"])
        self.assertIsNotNone(r["delta_vs_oof"])
        self.assertIn(r["breakdown"], (True, False))
        self.assertIn("label_shuffle", r, "标签打乱负对照必须给出")
        self.assertIsNotNone(r["label_shuffle"]["total"])
        self.assertTrue(rep["v1_exposed"])
        self.assertTrue(rep["not_independent_confirmation"])
        self.assertIn("不构成独立确认", rep["statement"])
        self.assertIsNotNone(rep["const_baseline_on_confirm"])
        gate = json.loads((self.reports / "E9_confirm_gate.json").read_text(encoding="utf-8"))
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        self.assertTrue(gate["checks"]["label_shuffle_control_ran"])
        self.assertTrue(gate["checks"]["not_independent_confirmation_flagged"])
        self.assertTrue(gate["checks"]["v1_exposed_flagged"])
        self.assertTrue(gate["checks"]["leakage_audit_complete"])

    def _set_oof(self, total: float) -> None:
        (self.reports / "E9_validation_report.json").write_text(json.dumps(
            {"candidates": [{"candidate_id": "PD1", "oof_total": total}]}), encoding="utf-8")

    def test_breakdown_threshold_is_applied(self):
        """基线压得很低 → 确认分高于基线，不判 breakdown；基线压得很高 → 判 breakdown。"""
        self._set_oof(10.0)
        low = self._run("--breakdown-threshold", "1.5")
        self.assertFalse(low["results"][0]["breakdown"],
                         low["results"][0]["delta_vs_oof"])
        self.assertGreater(low["results"][0]["delta_vs_oof"], 0.0)
        self._set_oof(500.0)
        high = self._run("--breakdown-threshold", "0.0")
        self.assertTrue(high["results"][0]["breakdown"])
        self.assertIn("PD1", high["breakdown_candidates"])
        self.assertEqual(high["decision"], "rejected")
        self._set_oof(95.0)                     # 还原，避免影响其它用例

    def test_missing_checkpoint_recorded(self):
        rep = self._run("--checkpoints", f"GHOST={self.root / 'nope.pt'}",
                        expect_rc=0)
        self.assertEqual(rep["results"][0]["status"], "checkpoint_missing")

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--folds-confirm", "--checkpoints", "--breakdown-threshold",
                     "--label-shuffle", "--confirm-split", "--validation-report"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
