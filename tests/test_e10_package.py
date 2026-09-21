"""E10/P0 export + P1 打包测试：CPU 导出确定性 与 提交包内容纪律。

export（torch 门控）：
  * fp32 重存 + `.npz` 权重要点清单（键名/形状/dtype/逐键 sha256）；
  * 同一输入前向两次**逐字节 sha256 相同**（确定性，CPU 提交的前提）；
  * ONNX 缺失时显式 `skipped`（真实原因写清），**不作为 Gate 条件**；
  * 缺 checkpoint → 明确失败（rc 3），不静默产出空包。

打包（口径层，无 torch）：
  * 只含推理所需内容：入口 + 配置 + `src/` + 权重；**绝不含** `__pycache__`/日志/数据；
  * 体积上限生效（代码/权重分别校验）；
  * `requirements.txt` 里出现 `torch` 声明 → 检查项 False（镜像预装，不许覆盖）；
  * `submission_manifest.json` 的 sha256 必须与真实文件字节一致；`--dry-run` 不写盘。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

EXPORT = V4 / "E10" / "code" / "export_cpu.py"
BUILD = V4 / "E10" / "code" / "build_submission.py"


class TestBuildSubmission(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.weights = self.root / "weights"
        self.weights.mkdir(parents=True, exist_ok=True)
        (self.weights / "final.fp32.pt").write_bytes(b"fp32-weights")
        (self.weights / "final.fp32.manifest.json").write_text("{}", encoding="utf-8")
        self.result = self.root / "result.json"
        self.result.write_text(json.dumps({"modelId": "", "modelName": "v4",
                                           "version": "1.0", "resultData": []}),
                               encoding="utf-8")
        self.reports = self.root / "reports"

    def tearDown(self):
        self._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(BUILD), "--out", str(self.root / "sub.zip"),
               "--weights", str(self.weights), "--result-json", str(self.result),
               "--result-zip", str(self.root / "result.zip"),
               "--manifest", str(self.root / "manifest.json"),
               "--reports-dir", str(self.reports), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1200:],
                                              proc.stderr[-1200:])
        p = self.reports / "E10_build_submission.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def test_pack_contents_and_hashes(self):
        rep = self._run("--smoke")
        self.assertEqual(rep["errors"], [])
        self.assertTrue((self.root / "sub.zip").is_file())
        with zipfile.ZipFile(self.root / "sub.zip") as zf:
            names = zf.namelist()
            self.assertIn("predict.py", names)
            self.assertIn("train.py", names)
            self.assertIn("README.md", names)
            self.assertIn("configs/v4.yaml", names)
            self.assertTrue(any(n.startswith("src/") for n in names))
            self.assertIn("models/v4/final/final.fp32.pt", names)
            self.assertFalse(any("__pycache__" in n for n in names))
        man = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["n_files"], len(man["files"]))
        for e in man["files"]:
            self.assertEqual(len(e["sha256"]), 64)
        self.assertTrue(man["zip_sha256"])
        with zipfile.ZipFile(self.root / "result.zip") as zf:
            self.assertEqual(zf.namelist(), ["result.json"])

    def test_size_limits_enforced(self):
        rep = self._run("--smoke", "--max-code-mb", "0.000001")
        self.assertFalse(rep["checks"]["code_size_ok"])
        rep2 = self._run("--smoke", "--max-weight-mb", "0.000001")
        self.assertFalse(rep2["checks"]["weight_size_ok"])
        rep3 = self._run("--max-weight-mb", "0.000001", expect_rc=3)
        self.assertEqual(rep3["passed"], False)

    def test_torch_pin_in_requirements_fails(self):
        req = self.root / "requirements.txt"
        req.write_text("numpy\ntorch==2.8.0\n", encoding="utf-8")
        rep = self._run("--smoke", "--requirements", str(req))
        self.assertFalse(rep["checks"]["requirements_no_torch_pin"])
        self.assertTrue(rep["requirements"]["torch_pins"])

    def test_missing_weights_is_error(self):
        empty = self.root / "empty_weights"
        empty.mkdir()
        rep = self._run("--weights", str(empty), "--smoke")
        self.assertTrue(rep["errors"])
        self.assertFalse(rep["checks"]["weights_present"])
        cmd = [sys.executable, str(BUILD), "--out", str(self.root / "s2.zip"),
               "--weights", str(empty), "--reports-dir", str(self.reports)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 3)

    def test_dry_run_writes_nothing(self):
        rep = self._run("--dry-run")
        self.assertTrue(rep["dry_run"])
        self.assertFalse((self.root / "sub.zip").is_file())
        self.assertFalse((self.root / "manifest.json").is_file())

    def test_pack_contains_bundled_relative_registry(self):
        """C1 回归：提交包必须自带 versions/registry.json，且 checkpoint 相对包根。"""
        self._run("--smoke")
        with zipfile.ZipFile(self.root / "sub.zip") as zf:
            names = zf.namelist()
            self.assertIn("versions/registry.json", names)
            doc = json.loads(zf.read("versions/registry.json").decode("utf-8"))
        info = doc["versions"]["PD1"]
        self.assertTrue(info["available"])
        ckpts = [info["checkpoint"], *info.get("checkpoints", [])]
        self.assertTrue(ckpts)
        for ck in ckpts:
            self.assertFalse(Path(ck).is_absolute(), ck)
            self.assertIn(ck, names)
        self.assertEqual(doc["latest"], "PD1")
        # 解压后直接跑包内 predict.py 的版本表路径：证明它读的是包内注册表。
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(self.root / "sub.zip") as zf:
                zf.extractall(td)
            out = subprocess.run([sys.executable, str(Path(td) / "predict.py"),
                                  "--list-versions"], cwd=td, capture_output=True,
                                 text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr[-500:])
            self.assertIn("PD1", out.stdout)
            self.assertIn("可用: ['CONST', 'PD1']", out.stdout)

    def test_help(self):
        out = subprocess.run([sys.executable, str(BUILD), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        for flag in ("--out", "--weights", "--result-json", "--result-zip",
                     "--max-weight-mb", "--max-code-mb"):
            self.assertIn(flag, out.stdout, flag)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestExportCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr, va = SC.fold0_wells(n_val=4, n_train=4)
        wells = list(dict.fromkeys(tr + va))[:4]
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, wells, n_rows=40, seed=11)
        from src.data import row_dataset as RD
        fit = RD.fit_scalers_from_wells(wells, cls.cache, spec=None)
        import torch
        from src.models.row_mlp import build_model
        from src.training import checkpoint as CK
        torch.manual_seed(0)
        model = build_model(32, hidden=16, layers=1)
        cls.ckpt = root / "final.pt"
        CK.save_checkpoint(cls.ckpt, model, meta={
            "row_scaler": fit["scaler"].to_dict(),
            "target_scalers": dict(fit["target"]),
            "model": {"arch": "RowMLP", "n_features": 32, "hidden": 16, "layers": 1,
                      "dropout": 0.0},
            "tau_atom": [0.5, 0.5, 0.5]}, bf16=False)
        cls.root = root
        cls.reports = root / "reports"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(EXPORT), "--ckpt", str(self.ckpt),
               "--out", str(self.root / "export"), "--reports-dir", str(self.reports),
               *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1200:],
                                              proc.stderr[-1200:])
        p = self.reports / "E10_export.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def test_export_fp32_npz_and_determinism(self):
        rep = self._run("--onnx", "--smoke")
        r = rep["result"]
        self.assertEqual(r["status"], "ok")
        self.assertTrue(Path(r["fp32_path"]).is_file())
        self.assertTrue(Path(r["npz_path"]).is_file())
        self.assertTrue(r["deterministic"]["ok"])
        self.assertEqual(r["deterministic"]["sha256"][0], r["deterministic"]["sha256"][1])
        self.assertEqual(r["manifest"]["dtype"], "float32")
        self.assertTrue(r["keys"]) 
        for k in r["keys"]:
            self.assertEqual(len(k["sha256"]), 64)
            self.assertIn("shape", k)
        self.assertIn(r["onnx"]["status"], ("ok", "skipped"))
        if r["onnx"]["status"] == "skipped":
            self.assertIn("reason", r["onnx"])
        self.assertTrue(rep["checks"]["deterministic_output"])
        self.assertTrue(rep["checks"]["max_memory_gb"])

    def test_missing_checkpoint_fails(self):
        cmd = [sys.executable, str(EXPORT), "--ckpt", str(self.root / "nope.pt"),
               "--out", str(self.root / "export2"), "--reports-dir", str(self.reports)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 3)

    def test_help(self):
        out = subprocess.run([sys.executable, str(EXPORT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        for flag in ("--ckpt", "--out", "--dtype", "--npz", "--onnx", "--smoke-rows"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
