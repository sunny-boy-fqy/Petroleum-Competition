"""E10 复现与提交测试：`verify_inference.py`（干净目录复现）+ `submit.py`（记录/配额/冻结）。

复现（口径层）：
  * 干净目录里**只放**代码包 + `data/`（不带仓库/缓存），照样跑出 `result.json`；
  * 两次运行 sha256 必须一致；与参考结果逐点差 ≤ 容差；超容差必须判失败；
  * 缺代码包 / 缺数据 → 明确失败（rc 4）。

提交（口径层）：
  * 日志**只追加**，记录 zip/code-zip 指纹与时间；
  * 配额用尽 → 拒绝（`rejected` 写明原因）且 Gate 失败；
  * `--dry-run` 不写日志、不改候选状态；
  * `--freeze` 且成绩有效 → 候选标 `submitted` + `frozen_best` + 记录 `frozen_sha256`；
  * 报错提交（`--error`）**不冻结**；`choice=b0_fallback` 如实记录。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402

VERIFY = V4 / "E10" / "code" / "verify_inference.py"
SUBMIT = V4 / "E10" / "code" / "submit.py"

STUB = '''import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", "--data-dir", dest="data_dir", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--expected-wells", type=int, default=None)
ap.add_argument("--expected-rows", type=int, default=None)
a = ap.parse_args()
MODE = "{mode}"
base = Path(a.data_dir)
if not list(base.glob("*.txt")) and (base / "test").is_dir():
    base = base / "test"          # 与真实 predict.py 的 --data_dir 解析一致
rows = []
for f in sorted(base.glob("*.txt")):
    n = 3
    preds = []
    for i in range(n):
        jitter = 0.0 if MODE == "clean" else 1e-3
        preds.append({{"depth": 1000.0 + i, "POR": 0.1 + jitter, "PERM": 0.01,
                       "SW": 99.9}})
    rows.append({{"logId": f.stem, "predictions": preds}})
Path(a.output).write_text(json.dumps({{"modelId": "", "modelName": "stub",
                                       "version": "1.0", "resultData": rows}}))
print("stub", MODE, len(rows))
'''


class TestVerifyInference(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.data = self.root / "data" / "test"
        self.data.mkdir(parents=True)
        for i in range(2):
            (self.data / f"w{i}.txt").write_text("a,b\n1,2\n", encoding="utf-8")
        self.reports = self.root / "reports"

    def tearDown(self):
        self._td.cleanup()

    def _zip(self, mode: str) -> Path:
        p = self.root / f"code_{mode}.zip"
        stub = self.root / "predict.py"
        stub.write_text(STUB.format(mode=mode), encoding="utf-8")
        with zipfile.ZipFile(p, "w") as zf:
            zf.write(stub, "predict.py")
        return p

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(VERIFY), "--data-dir", str(self.data),
               "--reports-dir", str(self.reports), "--expected-rows", "6",
               "--expected-wells", "2", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        p = self.reports / "E10_reproduce_report.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def test_clean_dir_reproduction_passes(self):
        z = self._zip("clean")
        rep = self._run("--zip", str(z), "--clean-dir", str(self.root / "clean"),
                        "--smoke")
        self.assertTrue(rep["inference_verified"], rep)
        self.assertEqual(rep["max_point_diff"], 0.0)
        self.assertTrue(rep["two_runs_identical"])
        self.assertIn("predict.py --data_dir ./data", rep["command"])
        self.assertLess(rep["minutes"], 30.0)
        gate = json.loads((self.reports / "E10_reproduce_gate.json").read_text(
            encoding="utf-8"))
        self.assertTrue(gate["checks"]["clean_dir_reproduce"])
        self.assertTrue(gate["checks"]["deterministic_two_runs"])
        self.assertTrue(gate["checks"]["point_diff_ok"])
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")

    def test_reference_mismatch_fails(self):
        good = self._zip("clean")
        ref = self.root / "reference.json"
        proc = subprocess.run([sys.executable, str(VERIFY), "--zip", str(good),
                               "--data-dir", str(self.data),
                               "--reports-dir", str(self.reports),
                               "--clean-dir", str(self.root / "clean1"),
                               "--output", "ref.json", "--expected-rows", "6",
                               "--expected-wells", "2", "--smoke"],
                              capture_output=True, text=True, timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stdout[-800:])
        shutil_src = self.root / "clean1" / "ref.json"
        ref.write_text(shutil_src.read_text(encoding="utf-8"), encoding="utf-8")
        bad = self._zip("jitter")
        rep = self._run("--zip", str(bad), "--clean-dir", str(self.root / "clean2"),
                        "--reference", str(ref), "--point-tol", "1e-6", "--smoke")
        self.assertGreater(rep["max_point_diff"], 1e-6)
        self.assertFalse(rep["inference_verified"])
        self.assertGreater(rep["n_over_tol"], 0)
        rc = subprocess.run([sys.executable, str(VERIFY), "--zip", str(bad),
                             "--data-dir", str(self.data), "--reports-dir",
                             str(self.reports), "--clean-dir", str(self.root / "clean3"),
                             "--reference", str(ref), "--expected-rows", "6",
                             "--expected-wells", "2"], capture_output=True, text=True,
                            timeout=900)
        self.assertEqual(rc.returncode, 3)

    def test_missing_inputs_fail(self):
        out = subprocess.run([sys.executable, str(VERIFY), "--zip",
                              str(self.root / "nope.zip"), "--data-dir", str(self.data),
                              "--reports-dir", str(self.reports), "--smoke"],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 4)
        out2 = subprocess.run([sys.executable, str(VERIFY), "--zip", str(self._zip("clean")),
                               "--data-dir", str(self.root / "nope"),
                               "--reports-dir", str(self.reports), "--smoke"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(out2.returncode, 4)


class TestSubmitRecorder(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        (self.reports / "training_time_log.json").write_text("{}", encoding="utf-8")
        self.zip = self.root / "result.zip"
        self.zip.write_bytes(b"zip-bytes")
        self.code_zip = self.root / "code.zip"
        self.code_zip.write_bytes(b"code-bytes")
        self.cands = self.root / "candidates.json"
        self.cands.write_text(json.dumps({"schema_version": 1, "candidates": [
            {"candidate_id": "PD1", "stage": "E6", "status": "shortlisted"}]}),
            encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(SUBMIT), "--zip", str(self.zip),
               "--code-zip", str(self.code_zip), "--candidates", str(self.cands),
               "--reports-dir", str(self.reports), "--log",
               str(self.reports / "E10_submission_log.json"),
               "--candidate-id", "PD1", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        p = self.reports / "E10_submission_report.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def _status(self, cid: str = "PD1") -> str:
        doc = json.loads(self.cands.read_text(encoding="utf-8"))
        return next(c["status"] for c in doc["candidates"] if c["candidate_id"] == cid)

    def test_dry_run_writes_nothing(self):
        rep = self._run("--dry-run", "--score", "82.0")
        self.assertTrue(rep["dry_run"])
        self.assertFalse((self.reports / "E10_submission_log.json").is_file())
        self.assertEqual(self._status(), "shortlisted")

    def test_successful_submission_freezes(self):
        rep = self._run("--score", "82.4", "--freeze", "--smoke")
        rec = rep["record"]
        self.assertEqual(rec["zip_sha256"], __import__("hashlib").sha256(
            b"zip-bytes").hexdigest())
        self.assertEqual(rec["code_zip_sha256"], __import__("hashlib").sha256(
            b"code-bytes").hexdigest())
        self.assertTrue(rec["frozen"])
        log = json.loads((self.reports / "E10_submission_log.json").read_text(
            encoding="utf-8"))
        self.assertEqual(len(log["records"]), 1)
        doc = json.loads(self.cands.read_text(encoding="utf-8"))
        cand = next(c for c in doc["candidates"] if c["candidate_id"] == "PD1")
        self.assertEqual(cand["frozen_sha256"], rec["zip_sha256"])
        self.assertIn(cand["status"], ("submitted", "frozen_best"))

    def test_error_submission_is_not_frozen(self):
        rep = self._run("--error", "平台 500", "--score", "0.0", "--freeze", "--smoke")
        self.assertEqual(rep["record"]["error"], "平台 500")
        self.assertFalse(rep["record"]["frozen"])
        self.assertEqual(self._status(), "shortlisted")

    def test_budget_exhausted_rejects(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log = {"schema_version": 1, "records": [
            {"submit_time": f"{today}T01:00:00Z", "zip_sha256": "x"} for _ in range(2)]}
        (self.reports / "E10_submission_log.json").write_text(json.dumps(log),
                                                              encoding="utf-8")
        rep = self._run("--score", "82.0", "--budget-per-day", "2", "--smoke")
        self.assertFalse(rep["budget_ok"])
        self.assertIsNotNone(rep["rejected"])
        self.assertIn("budget", rep["rejected"]["why"])
        self.assertEqual(len(json.loads((self.reports / "E10_submission_log.json")
                                        .read_text(encoding="utf-8"))["records"]), 2)

    def test_b0_fallback_choice_recorded(self):
        rep = self._run("--choice", "b0_fallback", "--reason", "v4 未通过确认复验", "--smoke")
        self.assertEqual(rep["choice"], "b0_fallback")
        self.assertIn("b0_fallback", json.dumps(rep["record"], ensure_ascii=False))

    def test_help(self):
        out = subprocess.run([sys.executable, str(SUBMIT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        for flag in ("--zip", "--code-zip", "--score", "--choice", "--freeze",
                     "--budget-per-day", "--candidate-id"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
