"""E10/P0 + 训练入口测试：`final_train.py`（折集成/全量重训）与 `train.py`（阶段转发）。

断言要点：
  * **折集成不重训**：只复制已注册的逐折权重并导出 fp32 副本；折间 manifest 不一致 → 拒做；
  * **全量重训固定 epoch、不早停**：`--max-wells` 只用于预检且必须在 manifest 里记录
    `all_train_wells_used=false`（不许把预检结果当成正式全量）；
  * `--dry-run` 只打印计划、不落盘；
  * `train.py` 能按 `--stage` 转发（`--dry-run` 打印命令）并把 `configs/v4.yaml` 的默认值
    翻成 CLI（键名映射错了会被这条测试逮住）。
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

FINAL = V4 / "E10" / "code" / "final_train.py"
TRAIN = V4 / "train.py"


class TestTrainEntry(unittest.TestCase):
    def test_list_stages(self):
        out = subprocess.run([sys.executable, str(TRAIN), "--list-stages"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        for stage in ("E1", "E3", "E4", "E5", "E6", "E7", "E8", "E10"):
            self.assertIn(stage, out.stdout)

    def test_dry_run_maps_config_to_cli(self):
        out = subprocess.run([sys.executable, str(TRAIN), "--stage", "E6",
                              "--config", str(V4 / "configs" / "v4.yaml"),
                              "--set", "training.epochs=7", "--set", "model.hidden=64",
                              "--dry-run"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        cmd = out.stdout.strip()
        self.assertIn("E6/code/train_state.py", cmd)
        self.assertIn("--epochs 7", cmd)
        self.assertIn("--hidden 64", cmd)
        self.assertIn("--spec F1", cmd)

    def test_bad_set_fails(self):
        out = subprocess.run([sys.executable, str(TRAIN), "--set", "wrong",
                              "--dry-run"], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("KEY=VALUE", out.stderr + out.stdout)

    def test_help(self):
        out = subprocess.run([sys.executable, str(TRAIN), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("--stage", out.stdout)


class TestFinalTrainEnsemble(unittest.TestCase):
    """折集成路径**不需要 torch**（只复制权重 + 校验 manifest）。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.out = self.root / "final"
        self.ckpt = self.root / "fold0.pt"
        self.ckpt.write_bytes(b"weights")
        man = {"row_scaler": {"median": [0.0] * 32}, "feature_names": ["GR"],
               "model": {"arch": "RowMLP", "n_features": 32, "hidden": 16, "layers": 1},
               "tau_atom": [0.5, 0.5, 0.5], "dtype": "float32"}
        self.ckpt.with_suffix(".manifest.json").write_text(json.dumps(man),
                                                           encoding="utf-8")
        (self.root / "cands.json").write_text(json.dumps({"schema_version": 1, "candidates": [
            {"candidate_id": "PD1", "stage": "E6", "status": "shortlisted", "oof_total": 83.0,
             "checkpoints": [str(self.ckpt)], "checkpoint": str(self.ckpt)}]}),
            encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(FINAL), "--aggregate", "fold_ensemble",
               "--candidates", str(self.root / "cands.json"), "--out-dir", str(self.out),
               "--reports-dir", str(self.reports), "--min-free-gb", "0", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        p = self.reports / "E10_final_train.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def test_ensemble_exports_fp32_copy(self):
        rep = self._run("--smoke")
        self.assertEqual(rep["status"], "ok")
        self.assertTrue((self.out / "final_fold0.pt").is_file())
        man = json.loads((self.out / "final_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["aggregate"], "fold_ensemble")
        self.assertEqual(man["candidate_id"], "PD1")
        self.assertEqual(len(man["weights"]), 1)
        self.assertEqual(man["folds"], 1)

    def test_dry_run_writes_nothing(self):
        rep = self._run("--dry-run")
        self.assertEqual(rep["status"], "dry_run")
        self.assertFalse((self.out / "final_manifest.json").is_file())

    def test_inconsistent_folds_refused(self):
        second = self.root / "fold1.pt"
        second.write_bytes(b"weights2")
        man = json.loads(self.ckpt.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        man["model"] = {**man["model"], "hidden": 64}
        second.with_suffix(".manifest.json").write_text(json.dumps(man), encoding="utf-8")
        (self.root / "cands.json").write_text(json.dumps({"schema_version": 1, "candidates": [
            {"candidate_id": "PD1", "stage": "E6", "status": "shortlisted", "oof_total": 83.0,
             "checkpoints": [str(self.ckpt), str(second)]}]}), encoding="utf-8")
        rep = self._run("--smoke")
        self.assertEqual(rep["status"], "inconsistent_folds")
        self.assertTrue(rep["result"]["problems"])

    def test_no_candidate_is_reported(self):
        (self.root / "empty.json").write_text(json.dumps({"schema_version": 1,
                                                          "candidates": []}),
                                              encoding="utf-8")
        cmd = [sys.executable, str(FINAL), "--aggregate", "fold_ensemble",
               "--candidates", str(self.root / "empty.json"), "--out-dir", str(self.out),
               "--reports-dir", str(self.reports), "--min-free-gb", "0", "--smoke"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0)
        rep = json.loads((self.reports / "E10_final_train.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["status"], "no_candidate")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestFinalTrainFullRetrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr_wells + va_wells, n_rows=40, seed=11)
        cls.root = root
        cls.reports = root / "reports"
        cls.out = root / "final"
        cls.scalers = root / "scalers"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_full_retrain_precheck_records_partial_wells(self):
        cmd = [sys.executable, str(FINAL), "--aggregate", "full_retrain", "--epochs", "2",
               "--hidden", "32", "--layers", "1", "--max-wells", "8",
               "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
               "--out-dir", str(self.out), "--scalers-dir", str(self.scalers),
               "--min-free-gb", "0", "--smoke"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        self.assertEqual(proc.returncode, 0, proc.stdout[-1500:] + proc.stderr[-1500:])
        rep = json.loads((self.reports / "E10_final_train.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["status"], "ok")
        man = json.loads((self.out / "final_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["aggregate"], "full_retrain")
        self.assertEqual(man["epochs"], 2)
        self.assertEqual(man["n_train_wells"], 8)
        self.assertFalse(man["all_train_wells_used"], "预检必须标记未用全量井")
        self.assertTrue(rep["result"]["resumable"]["ok"])
        gate = json.loads((self.reports / "E10_P0_gate.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["checks"]["final_weights_written"])
        self.assertTrue(gate["checks"]["disk_budget_ok"])
        self.assertTrue(gate["checks"]["no_early_stop"])


if __name__ == "__main__":
    unittest.main()
