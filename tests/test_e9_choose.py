"""E9/P0 决策测试（**口径层，无 torch**）：短名单、拒绝标记与不可覆盖纪律。

`E9/code/choose_submission.py` 是候选 `status` 的**唯一写入点**（`aggregate_oof.py` 只读），
因此这些纪律必须在这里被钉死：

  * `--dry-run` **不写**注册表；正式运行时通过护栏者 → `shortlisted`、不通过者 → `rejected`
    且**仍保留在候选表**（证据不删）；
  * `submitted` / `frozen_best` 候选**不可改**，并出现在 `skipped_immutable`；
  * 无人通过护栏 → `choice == B0_FALLBACK`（B0 兜底）且原因写明"回退"；
  * 决策 JSON 必须含 `choice/reason/candidate_oof/guardrail_floor/reference_oof/protocol_matched`；
  * 短名单 ≤ `--max-shortlist`。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

SCRIPT = V4 / "E9" / "code" / "choose_submission.py"
REPO_CANDIDATES = V4 / "versions" / "candidates.json"
REPO_DECISION = V4 / "reports" / "E9_submission_decision.json"


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _fixture(root: Path, *, passing=(("PD1", 83.2), ("E6b", 82.0)),
             failing=(("E1c", 70.0),), immutable=("OLD",)):
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "training_time_log.json").write_text("{}", encoding="utf-8")
    rows = [{"candidate_id": cid, "guardrail_pass": True, "oof_total": tot,
             "oof_path": f"{cid}.npz", "guardrail_reason": f"OOF {tot} ≥ floor"}
            for cid, tot in passing]
    rows += [{"candidate_id": cid, "guardrail_pass": False, "oof_total": tot,
              "guardrail_reason": f"OOF {tot} < floor"} for cid, tot in failing]
    rows += [{"candidate_id": cid, "guardrail_pass": False, "oof_total": 60.0,
              "guardrail_reason": "冻结候选"} for cid in immutable]
    (reports / "E9_validation_report.json").write_text(json.dumps({
        "guardrail": {"floor": 81.7757,
                      "anchors": {"B0_LOCAL_OOF": 80.382479, "B0_A_BOARD": 82.2757,
                                  "PASS_LINE": 75.0}},
        "protocol_note": "B0 协议不完全一致（protocol_matched=false）",
        "missing_mode": "drop", "candidates": rows}), encoding="utf-8")
    cands = {"schema_version": 1, "candidates": [
        {"candidate_id": cid, "stage": "E6", "status": "local_only"}
        for cid, _ in passing] + [
        {"candidate_id": cid, "stage": "E1", "status": "local_only"} for cid, _ in failing] + [
        {"candidate_id": cid, "stage": "E1", "status": "submitted"} for cid in immutable]}
    cand_path = root / "candidates.json"
    cand_path.write_text(json.dumps(cands), encoding="utf-8")
    return reports, cand_path


def _run(root: Path, reports: Path, cands: Path, *extra: str, expect_rc: int = 0) -> dict:
    cmd = [sys.executable, str(SCRIPT), "--validation-report", str(reports),
           "--reports-dir", str(reports), "--candidates", str(cands), "--smoke", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1200:],
                                          proc.stderr[-1200:])
    p = reports / "E9_submission_decision.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


class TestChooseSubmission(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports, self.cands = _fixture(self.root)

    def tearDown(self):
        self._td.cleanup()

    def _statuses(self) -> dict:
        return {c["candidate_id"]: c["status"]
                for c in json.loads(self.cands.read_text(encoding="utf-8"))["candidates"]}

    def test_dry_run_does_not_write(self):
        d = _run(self.root, self.reports, self.cands, "--dry-run")
        self.assertTrue(d["dry_run"])
        self.assertEqual(self._statuses(),
                         {"PD1": "local_only", "E6b": "local_only",
                          "E1c": "local_only", "OLD": "submitted"})
        self.assertEqual(d["choice"], "PD1")
        self.assertEqual(d["shortlist"][0]["candidate_id"], "PD1")
        self.assertTrue(all(entry.get("why") for entry in d["shortlist"]),
                        "每个入选候选都必须写清入选原因")

    def test_writes_shortlist_and_rejected_keeping_all(self):
        d = _run(self.root, self.reports, self.cands)
        st = self._statuses()
        self.assertEqual(st["PD1"], "shortlisted")
        self.assertEqual(st["E6b"], "shortlisted")
        self.assertEqual(st["E1c"], "rejected")
        self.assertEqual(st["OLD"], "submitted", "submitted 不可改")
        self.assertIn("OLD", d["skipped_immutable"])
        self.assertIn("E1c", d["rejected"])
        self.assertTrue(d["rejected_kept_in_registry"])

    def test_decision_keys_and_floor(self):
        d = _run(self.root, self.reports, self.cands, "--dry-run")
        for key in ("choice", "reason", "candidate_oof", "guardrail_floor", "reference_oof",
                    "protocol_matched"):
            self.assertIn(key, d)
        self.assertAlmostEqual(d["guardrail_floor"], 81.7757, places=4)
        self.assertFalse(d["protocol_matched"])
        self.assertIn("protocol_matched=false", d["protocol_note"])
        self.assertAlmostEqual(d["candidate_oof"], 83.2, places=6)

    def test_no_passing_candidate_falls_back(self):
        reports, cands = _fixture(self.root / "b", passing=(), failing=(("E1c", 70.0),),
                                  immutable=())
        d = _run(self.root / "b", reports, cands, "--dry-run")
        self.assertEqual(d["choice"], "B0_FALLBACK")
        self.assertIn("回退", d["reason"])
        self.assertEqual(d["shortlist"], [])
        self.assertIsNone(d["candidate_oof"])

    def test_shortlist_budget(self):
        d = _run(self.root, self.reports, self.cands, "--dry-run", "--max-shortlist", "1")
        self.assertEqual(len(d["shortlist"]), 1)
        self.assertEqual(d["choice"], "PD1")

    def test_missing_report_fails(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "--validation-report",
                               str(self.root / "nope"), "--reports-dir",
                               str(self.reports), "--candidates", str(self.cands),
                               "--smoke", "--dry-run"],
                              capture_output=True, text=True, timeout=300)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("缺少验证报告", proc.stderr + proc.stdout)

    def test_repo_files_untouched(self):
        before_c = _sha(REPO_CANDIDATES)
        before_d = _sha(REPO_DECISION)
        _run(self.root, self.reports, self.cands, "--dry-run")
        self.assertEqual(_sha(REPO_CANDIDATES), before_c)
        self.assertEqual(_sha(REPO_DECISION), before_d)

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--validation-report", "--candidates", "--max-shortlist",
                     "--a-board-log", "--dry-run"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
