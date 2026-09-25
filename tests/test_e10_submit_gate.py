"""E10/P2 提交 Gate 证据接线回归（C2 修复）。

锁定：`atomic_precision_reported` / `checkpoint_resumable` / `no_label_leak` /
`training_time_log_valid` / `no_retune_from_feedback` 必须来自证据文件，不能硬编码 True。
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

SCRIPT = V4 / "E10" / "code" / "submit.py"


def _run(reports: Path, *extra: str) -> dict:
    zipf = reports / "result.zip"
    zipf.write_bytes(b"zip")
    cmd = [sys.executable, str(SCRIPT), "--zip", str(zipf), "--code-zip", str(zipf),
           "--reports-dir", str(reports), "--dry-run", "--smoke", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-1200:], proc.stderr[-1200:])
    p = reports / "E10_P2_gate.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


class TestE10SubmitGateEvidence(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.reports = Path(self._td.name) / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._td.cleanup()

    def test_missing_evidence_fails_mandatory_checks(self):
        gate = _run(self.reports)
        checks = gate["checks"]
        for key in ("atomic_precision_reported", "checkpoint_resumable",
                    "no_label_leak", "training_time_log_valid"):
            self.assertFalse(checks[key], key)
        self.assertTrue(checks["no_retune_from_feedback"])

    def test_real_evidence_makes_checks_pass(self):
        atomic = {
            "atom_metrics_tau_half": {t: {"precision": 1.0, "recall": 1.0, "f1": 1.0}
                                      for t in ("POR", "PERM", "SW")},
            "folds_detail": [{"fold": 0, "resumable": {"ok": True}}],
        }
        (self.reports / "E6_atomic_report.json").write_text(
            json.dumps(atomic), encoding="utf-8")
        (self.reports / "E9_leakage_audit.json").write_text(
            json.dumps({"verdict": "pass", "high_risk_leak": False}), encoding="utf-8")
        (self.reports / "training_time_log.json").write_text(
            json.dumps({"valid": True, "folds": [{"fold": 0, "seconds": 1.0}]}),
            encoding="utf-8")
        gate = _run(self.reports)
        checks = gate["checks"]
        for key in ("atomic_precision_reported", "checkpoint_resumable",
                    "no_label_leak", "training_time_log_valid",
                    "no_retune_from_feedback"):
            self.assertTrue(checks[key], key)
        self.assertIn("evidence", gate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
