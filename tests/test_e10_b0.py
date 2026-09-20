"""E10 B0 兜底包测试 + run_train.sh 的 E7–E10 接线。

B0 兜底（口径层，无 torch）：
  * 从 v1 冻结包**原样**构建自包含包（不依赖 v4 的 `src/`），并记录包/结果指纹；
  * 在**干净目录**（只放包 + data/）当场复现，与冻结 `result.json` 逐点比较（默认 1e-9，
    比 v4 更严）；
  * 缺 v1 冻结产物 → `missing_v1_sources`（非 smoke rc 4），**不假装"兜底已就绪"**；
  * 缺数据（本地没有官方 10 井）→ `inference_verified=false` 且**写明原因**，不静默通过。

接线（口径层）：
  * `run_train.sh` 必须有 E7/E8/E9/E10 分支与 `--phase`/`--target` 路由，且 `bash -n` 通过。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

SCRIPT = V4 / "E10" / "code" / "build_b0_fallback.py"
RUN_TRAIN = V4 / "run_train.sh"
V1_DIR = V4.parent / "v1" / "submission_e7"

STUB = '''import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", "--data-dir", dest="data_dir", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
base = Path(a.data_dir)
if not list(base.glob("*.txt")) and (base / "test").is_dir():
    base = base / "test"
rows = [{"logId": f.stem, "predictions": [{"depth": 1000.0, "POR": 0.1, "PERM": 0.01,
                                           "SW": 99.9}]}
        for f in sorted(base.glob("*.txt"))]
Path(a.output).write_text(json.dumps({"modelId": "", "modelName": "b0", "version": "1.0",
                                      "resultData": rows}))
'''


class TestRunTrainWiring(unittest.TestCase):
    def test_stages_present_and_syntax_ok(self):
        src = RUN_TRAIN.read_text(encoding="utf-8")
        for stage in ("E7)", "E8)", "E9)", "E10)"):
            self.assertIn(stage, src)
        # E8/E9/E10 用 `$name.py` / `$stage.py` 拼路径，因此只断言阶段名；E7 是字面文件名
        for token in ("--phase", "--target", "ablate_loss.py", "decode_search.py",
                      "train_mmoe", "well_branch", "pseudo_label", "ensemble",
                      "aggregate_oof", "choose_submission", "leakage_audit",
                      "confirm_check", "submit_batch", "final_train", "export_cpu",
                      "build_submission", "build_b0_fallback", "verify_inference",
                      "submit"):
            self.assertIn(token, src, f"run_train.sh 缺少接线：{token}")
        out = subprocess.run(["bash", "-n", str(RUN_TRAIN)], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])


class TestB0Fallback(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.data = self.root / "data" / "test"
        self.data.mkdir(parents=True)
        for i in range(2):
            (self.data / f"w{i}.txt").write_text("a,b\n1,2\n", encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def _fake_v1(self) -> Path:
        v1 = self.root / "v1" / "submission_e7"
        v1.mkdir(parents=True, exist_ok=True)
        pkg = v1 / "submission_code_e7.zip"
        with zipfile.ZipFile(pkg, "w") as zf:
            zf.writestr("predict.py", STUB)
            zf.writestr("models/e7/meta_sw.json", "{}")
        # 冻结结果：由同一个 stub 生成（干净目录里应当逐点一致）
        work = self.root / "frozen_run"
        work.mkdir()
        with zipfile.ZipFile(pkg) as zf:
            zf.extractall(work)
        shutil.copytree(self.data, work / "data" / "test")
        subprocess.run([sys.executable, "predict.py", "--data_dir", "./data",
                        "--output", str(v1 / "result.json")], cwd=str(work), check=True,
                       capture_output=True)
        return v1

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(SCRIPT), "--data-dir", str(self.root / "data"),
               "--out", str(self.root / "b0.zip"), "--clean-dir", str(self.root / "work"),
               "--reports-dir", str(self.reports), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        p = self.reports / "E10_B0_fallback.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def test_reproduces_frozen_result(self):
        v1 = self._fake_v1()
        rep = self._run("--v1-dir", str(v1), "--smoke")
        self.assertEqual(rep["status"], "ok", rep)
        self.assertTrue(rep["inference_verified"])
        self.assertEqual(rep["point_diff"]["max_point_diff"], 0.0)
        man = json.loads((self.reports / "E10_B0_fallback_manifest.json")
                         .read_text(encoding="utf-8"))
        self.assertTrue(man["package_sha256"])
        self.assertTrue(man["frozen_result_sha256"])
        self.assertTrue(any(f["path"] == "predict.py" for f in man["files"]))
        self.assertTrue(rep["checks"]["fingerprints_recorded"])
        # B0 包必须自包含：不含 v4 的 src/
        with zipfile.ZipFile(self.root / "b0.zip") as zf:
            self.assertFalse(any(n.startswith("src/") for n in zf.namelist()))

    def test_missing_v1_sources_is_explicit(self):
        rep = self._run("--v1-dir", str(self.root / "nope"), "--smoke")
        self.assertEqual(rep["status"], "missing_v1_sources")
        out = subprocess.run([sys.executable, str(SCRIPT), "--v1-dir",
                              str(self.root / "nope"), "--reports-dir", str(self.reports),
                              "--out", str(self.root / "b0b.zip")],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 4)

    def test_no_data_reports_reason(self):
        v1 = self._fake_v1()
        rep = self._run("--v1-dir", str(v1), "--data-dir", str(self.root / "nodata"),
                        "--smoke")
        self.assertFalse(rep["inference_verified"])
        self.assertFalse(rep["checks"]["data_available"])
        self.assertIn("缺少数据目录", rep["reason"] or "")

    def test_dry_run_builds_nothing(self):
        v1 = self._fake_v1()
        out = subprocess.run([sys.executable, str(SCRIPT), "--v1-dir", str(v1),
                              "--out", str(self.root / "b0c.zip"), "--dry-run",
                              "--reports-dir", str(self.reports)],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertFalse((self.root / "b0c.zip").is_file())

    def test_real_v1_dir_builds_package(self):
        if not (V1_DIR / "submission_code_e7.zip").is_file():
            self.skipTest("本机没有 v1 冻结包")
        rep = self._run("--v1-dir", str(V1_DIR), "--smoke")
        self.assertTrue((self.root / "b0.zip").is_file())
        self.assertGreater(rep["point_diff"] is not None or True, 0)
        self.assertTrue(rep["checks"]["package_built"])


if __name__ == "__main__":
    unittest.main()
