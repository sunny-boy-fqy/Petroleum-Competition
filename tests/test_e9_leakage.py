"""E9/P1 泄漏审计测试（**口径层，无 torch**）：四类必查项与 `residual_risk` 语义。

纪律：
  * 折维度用**冻结折文件**验证：每口井恰属一个折（同井跨折=硬泄漏）；
  * 标尺必须只在训练折拟合；`train_wells ∩ val_wells ≠ ∅` → **fail**（硬泄漏，rc 3）；
  * 溯源表列名含目标派生列 → fail；
  * transductive 报告声明 `uses_test_labels=true` → fail；
  * **缺证据 → `residual_risk`**（既不算通过也不算失败）且必须出现在
    `residual_risks` 列表里（不许静默通过）；整体 verdict 为 `pass_with_residual_risk`。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

SCRIPT = V4 / "E9" / "code" / "leakage_audit.py"


def _run(script_args: list[str], expect_rc: int = 0) -> dict:
    proc = subprocess.run([sys.executable, str(SCRIPT), *script_args],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1200:],
                                          proc.stderr[-1200:])
    for a in script_args:
        pass
    rep = None
    for i, a in enumerate(script_args):
        if a == "--reports-dir":
            rep = Path(script_args[i + 1]) / "E9_leakage_audit.json"
    return json.loads(rep.read_text(encoding="utf-8")) if rep and rep.is_file() else {}


class TestLeakageAudit(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        (self.reports / "training_time_log.json").write_text("{}", encoding="utf-8")
        self.scalers = self.root / "scalers"
        self.scalers.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._td.cleanup()

    def _args(self, *extra: str) -> list[str]:
        return ["--reports-dir", str(self.reports), "--scalers-dir", str(self.scalers),
                *extra]

    def _scaler(self, name: str, train, val, fitted="train_fold_only") -> None:
        (self.scalers / name).write_text(json.dumps(
            {"fold": 0, "train_wells": list(train), "val_wells": list(val),
             "fitted_on": fitted, "row_scaler": {}, "target_scalers": {}}),
            encoding="utf-8")

    def test_all_residual_risk_when_no_evidence(self):
        rep = _run(self._args("--smoke"))
        self.assertEqual(rep["verdict"], "pass_with_residual_risk")
        self.assertFalse(rep["high_risk_leak"])
        self.assertEqual(rep["n_residual_risk"], 3)
        self.assertEqual(len(rep["residual_risks"]), 3)
        self.assertEqual({a["id"] for a in rep["audits"]},
                         {"fold_dimension", "input_columns",
                          "scalers_fitted_on_train_fold", "pseudo_label_source"})
        self.assertEqual(next(a for a in rep["audits"]
                              if a["id"] == "fold_dimension")["verdict"], "pass",
                         "冻结折文件应通过折维度审计")

    def test_valid_scalers_pass(self):
        self._scaler("E6_state_fold0.json", ["a", "b"], ["c", "d"])
        rep = _run(self._args("--smoke"))
        audit = next(a for a in rep["audits"] if a["id"] == "scalers_fitted_on_train_fold")
        self.assertEqual(audit["verdict"], "pass")
        self.assertTrue(all(f["ok"] for f in audit["evidence"]["files"]))

    def test_overlapping_scalers_is_hard_leak(self):
        self._scaler("bad.json", ["a", "b"], ["b", "c"])
        _run(self._args(), expect_rc=3)
        rep = _run(self._args("--smoke"))
        audit = next(a for a in rep["audits"] if a["id"] == "scalers_fitted_on_train_fold")
        self.assertEqual(audit["verdict"], "fail")
        self.assertTrue(rep["high_risk_leak"])
        self.assertIn("硬泄漏", audit["residual_risk"])

    def test_provenance_csv_with_target_column_fails(self):
        csv = self.root / "provenance.csv"
        csv.write_text("name,group,target_por\nx,F1,0\n", encoding="utf-8")
        rep = _run(self._args("--smoke", "--provenance-csv", str(csv)))
        audit = next(a for a in rep["audits"] if a["id"] == "input_columns")
        self.assertEqual(audit["verdict"], "fail")
        self.assertIn("por", audit["evidence"]["provenance_csv"]["hits"])
        self.assertTrue(rep["high_risk_leak"])

    def test_transductive_with_test_labels_fails(self):
        (self.reports / "E8_transductive.json").write_text(json.dumps({
            "method": "well_mean_align",
            "legality": {"uses_test_labels": True},
            "test_side": {"guard": {"ok": True}}}), encoding="utf-8")
        rep = _run(self._args("--smoke"))
        audit = next(a for a in rep["audits"] if a["id"] == "pseudo_label_source")
        self.assertEqual(audit["verdict"], "fail")
        self.assertTrue(rep["high_risk_leak"])

    def test_transductive_legal_path_passes(self):
        (self.reports / "E8_transductive.json").write_text(json.dumps({
            "method": "well_mean_align",
            "legality": {"uses_test_labels": False},
            "test_side": {"guard": {"ok": True}},
            "eval_mode": "held_out_train_wells"}), encoding="utf-8")
        rep = _run(self._args("--smoke"))
        audit = next(a for a in rep["audits"] if a["id"] == "pseudo_label_source")
        self.assertEqual(audit["verdict"], "pass")

    def test_clean_run_passes_gate_without_residual_risk(self):
        self._scaler("E6_state_fold0.json", ["a"], ["b"])
        csv = self.root / "prov.csv"
        csv.write_text("name,group\nGR,F1\n", encoding="utf-8")
        (self.reports / "E8_transductive.json").write_text(json.dumps({
            "method": "none", "legality": {"uses_test_labels": False},
            "test_side": {"guard": {"ok": True}}}), encoding="utf-8")
        rep = _run(self._args("--provenance-csv", str(csv)))
        self.assertEqual(rep["verdict"], "pass")
        self.assertEqual(rep["n_residual_risk"], 0)
        gate = json.loads((self.reports / "E9_leakage_gate.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["passed"], gate["aggregate"])
        self.assertTrue(gate["checks"]["leakage_audit_complete"])
        self.assertTrue(gate["checks"]["no_high_risk_leak"])

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--folds", "--folds-report", "--provenance-csv", "--scalers-dir",
                     "--transductive-report", "--json"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
