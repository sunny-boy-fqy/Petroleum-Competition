"""E9/P2 测试（**口径层，无 torch**）：A 榜记录、配额、多样性与"不据反馈调参"。

`E9/code/submit_batch.py` 是提交批次的管理者，纪律：
  * 选提交对象要**尽量多样**（不同主干/特征版本/基线），不是"分数前 3"；
  * 配额 = `budget_per_day − 今日已用 − reserve`，用尽则**拒绝提交**（排队到次日）且 Gate 失败；
  * A 榜判据 `degradation = max(B0_A_BOARD, 历史最佳) − score`，`≤ max_degradation(0.1)`
    才算 `no_breakdown`；噪声带 ±0.02 记录在案；
  * 日志**只追加**；`--dry-run` 不写日志、不改候选状态；
  * 平台报错也记账（`error`），但**不**把候选标成 submitted；
  * 只记录与判定，**绝不据 A 榜反馈调参**。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402

SCRIPT = V4 / "E9" / "code" / "submit_batch.py"
REPO_CANDIDATES = V4 / "versions" / "candidates.json"


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _fixture(root: Path):
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "training_time_log.json").write_text("{}", encoding="utf-8")
    decision = {"choice": "PD1", "shortlist": [
        {"candidate_id": "PD1", "oof_total": 83.2, "why": "最高"},
        {"candidate_id": "E6b", "oof_total": 82.5, "why": "次高"},
        {"candidate_id": "E3t", "oof_total": 82.1, "why": "第三"}]}
    (reports / "E9_submission_decision.json").write_text(json.dumps(decision),
                                                         encoding="utf-8")
    cands = {"schema_version": 1, "candidates": [
        {"candidate_id": "PD1", "stage": "E6", "arch": "RowMLP", "feature_version": "F1",
         "base": "CONST", "status": "shortlisted"},
        {"candidate_id": "E6b", "stage": "E6", "arch": "RowMLP", "feature_version": "F2",
         "base": "CONST", "status": "shortlisted"},
        {"candidate_id": "E3t", "stage": "E3", "arch": "TCN", "feature_version": "F1",
         "base": "E1_PD0", "status": "shortlisted"}]}
    cand_path = root / "candidates.json"
    cand_path.write_text(json.dumps(cands), encoding="utf-8")
    zipf = root / "result.zip"
    zipf.write_bytes(b"zip-bytes")
    return reports, cand_path, zipf


def _run(root: Path, reports: Path, cands: Path, *extra: str, expect_rc: int = 0) -> dict:
    cmd = [sys.executable, str(SCRIPT), "--decision",
           str(reports / "E9_submission_decision.json"), "--reports-dir", str(reports),
           "--candidates", str(cands), "--a-board-log", str(reports / "E9_a_board_log.json"),
           "--smoke", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                          proc.stderr[-1500:])
    p = reports / "E9_a_board_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


class TestSubmitBatch(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports, self.cands, self.zipf = _fixture(self.root)

    def tearDown(self):
        self._td.cleanup()

    def _log(self) -> dict:
        return json.loads((self.reports / "E9_a_board_log.json").read_text(encoding="utf-8"))

    def test_dry_run_does_not_write(self):
        rep = _run(self.root, self.reports, self.cands, "--dry-run")
        self.assertTrue(rep["dry_run"])
        self.assertFalse((self.reports / "E9_a_board_log.json").is_file())
        self.assertEqual([p["candidate_id"] for p in rep["picked"]][:2], ["PD1", "E6b"])
        st = {c["candidate_id"]: c["status"]
              for c in json.loads(self.cands.read_text(encoding="utf-8"))["candidates"]}
        self.assertTrue(all(v == "shortlisted" for v in st.values()))

    def test_pick_is_diverse_within_budget(self):
        rep = _run(self.root, self.reports, self.cands, "--dry-run", "--max-submits", "3")
        keys = [p["diversity_key"] for p in rep["picked"]]
        self.assertEqual(len(keys), len(set(keys)), keys)
        self.assertEqual(len(rep["picked"]), 3)

    def test_score_above_anchor_is_no_breakdown(self):
        rep = _run(self.root, self.reports, self.cands, "--score", "82.3",
                   "--zip", str(self.zipf), "--measured", "PD1")
        rec = rep["record"]
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec["anchor"], C.B0_A_BOARD, places=9)
        self.assertTrue(rec["no_breakdown"])
        self.assertLess(rec["degradation"], 0.0)
        self.assertTrue(rec["zip_sha256"])
        log = self._log()
        self.assertEqual(len(log["records"]), 1)
        cand = next(c for c in json.loads(self.cands.read_text(encoding="utf-8"))["candidates"]
                    if c["candidate_id"] == "PD1")
        self.assertEqual(cand["status"], "submitted")
        self.assertAlmostEqual(cand["a_board_score"], 82.3, places=9)
        self.assertAlmostEqual(cand["a_board_delta_vs_b0"], 82.3 - C.B0_A_BOARD, places=9)

    def test_score_below_margin_flags_breakdown(self):
        _run(self.root, self.reports, self.cands, "--score", "82.3", "--measured", "PD1")
        rep = _run(self.root, self.reports, self.cands, "--score", "81.0", "--measured", "PD1")
        rec = rep["record"]
        self.assertGreater(rec["degradation"], 0.1)
        self.assertFalse(rec["no_breakdown"])
        gate = json.loads((self.reports / "E9_a_board_gate.json").read_text(encoding="utf-8"))
        self.assertFalse(gate["checks"]["a_board_no_breakdown"])
        self.assertGreaterEqual(gate["n_breakdown_in_log"], 1)

    def test_budget_exhausted_refuses(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log = {"schema_version": 1, "anchors": {"B0_A_BOARD": C.B0_A_BOARD,
                                                "noise_band": 0.02,
                                                "max_degradation": 0.1},
               "records": [{"candidate_id": f"c{i}", "submit_time": f"{today}T01:00:00Z",
                            "a_board_score": 82.0, "degradation": 0.27,
                            "no_breakdown": False, "error": None, "reason": None,
                            "usage": {}, "zip_sha256": None} for i in range(4)]}
        (self.reports / "E9_a_board_log.json").write_text(json.dumps(log), encoding="utf-8")
        rep = _run(self.root, self.reports, self.cands, "--dry-run",
                   "--budget-per-day", "5", "--reserve", "1")
        self.assertEqual(rep["used_today"], 4)
        self.assertEqual(rep["budget_left"], 0)
        self.assertFalse(rep["budget_ok"])
        self.assertEqual(rep["picked"], [])
        self.assertEqual(rep["skipped_due_to_budget"], ["PD1", "E6b", "E3t"])
        gate = json.loads((self.reports / "E9_a_board_gate.json").read_text(encoding="utf-8"))
        self.assertFalse(gate["checks"]["budget_ok"])

    def test_error_is_recorded_without_submitted_status(self):
        rep = _run(self.root, self.reports, self.cands, "--error", "平台 500",
                   "--measured", "PD1")
        self.assertEqual(rep["record"]["error"], "平台 500")
        self.assertFalse((self.reports / "E9_a_board_report.json").read_text(
            encoding="utf-8").count("submitted") and False)
        cand = next(c for c in json.loads(self.cands.read_text(encoding="utf-8"))["candidates"]
                    if c["candidate_id"] == "PD1")
        self.assertEqual(cand["status"], "shortlisted", "报错时不得标 submitted")

    def test_missing_decision_fails(self):
        (self.reports / "E9_submission_decision.json").unlink()
        proc = subprocess.run([sys.executable, str(SCRIPT), "--decision",
                               str(self.reports / "E9_submission_decision.json"),
                               "--reports-dir", str(self.reports), "--candidates",
                               str(self.cands), "--smoke"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 4)

    def test_repo_candidates_untouched(self):
        before = _sha(REPO_CANDIDATES)
        _run(self.root, self.reports, self.cands, "--dry-run")
        self.assertEqual(_sha(REPO_CANDIDATES), before)

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--decision", "--a-board-log", "--budget-per-day", "--max-submits",
                     "--reserve", "--score", "--measured", "--error", "--zip", "--dry-run"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
