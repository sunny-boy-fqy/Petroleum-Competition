"""可复用大成果存储测试：cache / 折 pkl / 提交包 / 候选状态。

小状态层（checkpoint/OOF/report/scaler）由 `tools/sync_state.py` 负责，
这里只测 `tools/artifact_store.py` 的补充职责。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
TOOL = V4 / "tools" / "artifact_store.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(V4),
    )


class TestArtifactStore(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)
        self.local = self.td / "local"
        self.remote = self.td / "remote"
        (self.local / "v4" / "cache" / "raw").mkdir(parents=True)
        (self.local / "v4" / "cache" / "feat").mkdir(parents=True)
        (self.local / "v4" / "runs" / "E3" / "fold_results").mkdir(parents=True)
        (self.local / "v4" / "state").mkdir(parents=True)
        (self.local / "v4" / "reports").mkdir(parents=True)
        (self.local / "v4" / "cache" / "raw" / "a.npz").write_bytes(b"raw")
        (self.local / "v4" / "cache" / "feat" / "b.npz").write_bytes(b"feat")
        (self.local / "v4" / "runs" / "E3" / "fold_results" / "f0.pkl").write_bytes(b"pkl")
        (self.local / "v4" / "runs" / "E3" / "model.pt").write_bytes(b"pt")
        (self.local / "v4" / "runs" / "E10_submission").mkdir(parents=True)
        (self.local / "v4" / "runs" / "E10_submission" / "result.zip").write_bytes(b"zip")
        (self.local / "v4" / "state" / "candidates.json").write_text("{}", encoding="utf-8")
        (self.local / "v4" / "state" / "all_pipeline_progress.json").write_text(
            "{}", encoding="utf-8")
        (self.local / "v4" / "reports" / "E3_metrics.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def test_publish_only_selected_reusable_artifacts(self):
        r = _run("publish", "--local-root", str(self.local),
                 "--remote-root", str(self.remote), "--quiet")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        art = self.remote / "v4" / "artifacts"
        self.assertTrue((art / "cache" / "raw" / "a.npz").is_file())
        self.assertTrue((art / "cache" / "feat" / "b.npz").is_file())
        self.assertTrue((art / "run_root" / "E3" / "fold_results" / "f0.pkl").is_file())
        self.assertTrue((art / "run_root" / "E10_submission" / "result.zip").is_file())
        self.assertTrue((art / "state" / "candidates.json").is_file())
        # 小状态层负责的内容不应在这里重复大拷贝。
        self.assertFalse((art / "run_root" / "E3" / "model.pt").exists())
        self.assertFalse((art / "state" / "all_pipeline_progress.json").exists())
        self.assertFalse((art / "reports").exists())

    def test_restore_roundtrip(self):
        r = _run("publish", "--local-root", str(self.local),
                 "--remote-root", str(self.remote), "--quiet")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # 清掉大成果，模拟新机器本地 runtime 为空。
        for rel in ("v4/cache", "v4/runs/E3/fold_results", "v4/runs/E10_submission",
                    "v4/state/candidates.json"):
            p = self.local / rel
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        r = _run("restore", "--local-root", str(self.local),
                 "--remote-root", str(self.remote), "--quiet")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.local / "v4/cache/raw/a.npz").read_bytes(), b"raw")
        self.assertEqual((self.local / "v4/cache/feat/b.npz").read_bytes(), b"feat")
        self.assertTrue((self.local / "v4/runs/E3/fold_results/f0.pkl").is_file())
        self.assertTrue((self.local / "v4/runs/E10_submission/result.zip").is_file())
        self.assertTrue((self.local / "v4/state/candidates.json").is_file())

    def test_budget_guard(self):
        r = _run("publish", "--local-root", str(self.local),
                 "--remote-root", str(self.remote), "--max-gb", "1e-9")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("预算超限", r.stderr)

    def test_status_reads_manifest(self):
        r = _run("publish", "--local-root", str(self.local),
                 "--remote-root", str(self.remote), "--quiet")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = _run("status", "--local-root", str(self.local),
                 "--remote-root", str(self.remote))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("run_extras", r.stdout)
        self.assertIn("cache", r.stdout)


class TestArtifactStoreWiring(unittest.TestCase):
    def test_run_train_registers_artifact_store(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        for token in (
            "tools/artifact_store.py",
            "restore_artifacts_from_network",
            "publish_artifacts_to_network",
            "/data/v4/artifacts",
        ):
            self.assertIn(token, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
