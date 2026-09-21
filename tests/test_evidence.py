"""H5 回归：Gate 证据读取必须"拿不到 = False"，不允许硬编码/恒真。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.validation import evidence as EVID  # noqa: E402


class TestEvidenceReaders(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.reports = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_missing_evidence_is_false(self):
        ok, ev = EVID.leakage_audit_ok(self.reports)
        self.assertFalse(ok)
        self.assertFalse(ev["present"])
        self.assertFalse(EVID.atomic_precision_reported(self.reports)[0])
        self.assertFalse(EVID.checkpoint_resumable(self.reports)[0])
        self.assertFalse(EVID.training_time_log_valid(self.reports)[0])

    def test_real_leakage_audit_evidence_controls_no_label_leak(self):
        (self.reports / "E9_leakage_audit.json").write_text(json.dumps({
            "verdict": "pass", "high_risk_leak": False,
            "n_fail": 0, "n_residual_risk": 0}), encoding="utf-8")
        ok, _ = EVID.leakage_audit_ok(self.reports)
        self.assertTrue(ok)
        (self.reports / "E9_leakage_audit.json").write_text(json.dumps({
            "verdict": "fail", "high_risk_leak": True, "n_fail": 1}),
            encoding="utf-8")
        self.assertFalse(EVID.leakage_audit_ok(self.reports)[0])

    def test_atomic_and_resumable_require_real_report(self):
        (self.reports / "E6_atomic_report.json").write_text(json.dumps({
            "atom_metrics_tau_half": {"POR": {}, "PERM": {}, "SW": {}},
            "folds_detail": [{"fold": 0, "resumable": {"ok": True}}]}),
            encoding="utf-8")
        self.assertTrue(EVID.atomic_precision_reported(self.reports)[0])
        self.assertTrue(EVID.checkpoint_resumable(self.reports)[0])
        (self.reports / "E6_atomic_report.json").write_text(json.dumps({
            "atom_metrics_tau_half": {}, "folds_detail": []}), encoding="utf-8")
        self.assertFalse(EVID.atomic_precision_reported(self.reports)[0])
        self.assertFalse(EVID.checkpoint_resumable(self.reports)[0])
        self.assertFalse(EVID.disk_budget_ok("cleanup"))
        self.assertTrue(EVID.disk_budget_ok("ok"))


if __name__ == "__main__":
    unittest.main()
