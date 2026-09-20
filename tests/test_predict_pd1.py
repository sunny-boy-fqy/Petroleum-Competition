"""PD1 推理接线测试：版本表写回（口径层）+ `predict.py --use-version PD1` 端到端（torch 门控）。

为什么这条链路必须测
------------------
E6/P2 与 E10 都要靠 `predict.py --use-version PD1` 产出提交载荷，而此前 `predict.py`
只注册了 `CONST`：权重路径、manifest（RowScaler/τ）、缺失矩阵导出任何一环接错，
提交就会静默产出无意义结果。因此这里既测**注册纪律**（不许注册没有权重的版本、
不许把不可用版本设为 latest），也测**真实推理端到端**（合成数据 → 权重 → 提交契约）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.versioning import registry as REG  # noqa: E402


class TestRegistryVersionWrites(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.reg = Path(self._td.name) / "registry.json"
        self.ckpt = Path(self._td.name) / "pd1.pt"
        self.ckpt.write_bytes(b"fake-weights")          # 只校验"文件存在"

    def tearDown(self):
        self._td.cleanup()

    def test_register_requires_existing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            REG.register_pipeline("PD1", Path(self._td.name) / "missing.pt", path=self.reg)

    def test_register_and_latest_rule(self):
        REG.register_pipeline("PD1", self.ckpt, oof_total=81.2, make_latest=True,
                              path=self.reg)
        vs = REG.versions(self.reg)
        self.assertTrue(vs["PD1"]["available"])
        self.assertTrue(vs["PD1"]["completed"])
        self.assertAlmostEqual(vs["PD1"]["oof_total"], 81.2, places=9)
        self.assertEqual(REG.latest(self.reg), "PD1")
        # 不可用版本不得设为 latest
        REG.register_pipeline("PD2", self.ckpt, available=False, completed=False,
                              path=self.reg)
        with self.assertRaises(ValueError):
            REG.set_latest("PD2", path=self.reg)

    def test_set_version_updates_fields(self):
        REG.register_pipeline("PD1", self.ckpt, path=self.reg)
        REG.set_version("PD1", available=False, oof_total=80.0, note="hidden", path=self.reg)
        v = REG.versions(self.reg)["PD1"]
        self.assertFalse(v["available"])
        self.assertAlmostEqual(v["oof_total"], 80.0, places=9)
        self.assertEqual(v["note"], "hidden")
        with self.assertRaises(KeyError):
            REG.set_version("nope", available=True, path=self.reg)

    def test_default_registry_is_repo_file(self):
        self.assertEqual(REG.REGISTRY, V4 / "versions" / "registry.json")
        self.assertTrue(REG.REGISTRY.is_file())


class TestPredictListVersions(unittest.TestCase):
    def test_list_versions_runs_without_torch(self):
        out = subprocess.run([sys.executable, str(V4 / "predict.py"), "--list-versions"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        self.assertIn("CONST", out.stdout)
        self.assertIn("PD1", out.stdout)
        self.assertIn("latest:", out.stdout)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPredictPD1EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=4, n_train=4)
        cls.wells = list(dict.fromkeys(tr_wells + va_wells))[:4]
        cls.tr_dir, cls.te_dir, cls.cache = SC.build_cache(root / "cache", cls.wells,
                                                           n_rows=40, seed=11)
        cls.root = root
        # 训练折标尺（manifest 必须带它，否则反变换整体偏移）
        from src.data import row_dataset as RD
        fit = RD.fit_scalers_from_wells(cls.wells, cls.cache, spec=None)
        import torch
        from src.models.row_mlp import build_model
        from src.training import checkpoint as CK
        torch.manual_seed(0)
        model = build_model(32, hidden=16, layers=1)
        cls.ckpt = root / "pd1_fold0.pt"
        CK.save_checkpoint(cls.ckpt, model, meta={
            "row_scaler": fit["scaler"].to_dict(),
            "target_scalers": dict(fit["target"]),
            "model": {"n_features": 32, "hidden": 16, "layers": 1, "dropout": 0.0},
            "tau_atom": [0.5, 0.5, 0.5],
            "scalers_fitted_on": "train_fold_only",
        }, bf16=False)
        cls.reg = root / "registry.json"
        REG.register_pipeline("PD1", cls.ckpt, oof_total=80.0, make_latest=True,
                              path=cls.reg)
        cls.out = root / "result.json"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str):
        argv = [sys.executable, str(V4 / "predict.py"),
                "--registry", str(self.reg), "--use-version", "PD1",
                "--data_dir", str(self.root / "cache_src"),
                "--output", str(self.out),
                "--expected-wells", str(len(self.wells)), "--expected-rows", "80", *extra]
        return subprocess.run(argv, capture_output=True, text=True, timeout=1800)

    def test_pd1_produces_valid_submission(self):
        out = self._run()
        self.assertEqual(out.returncode, 0, out.stderr[-2000:] + out.stdout[-1500:])
        payload = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["resultData"]), len(self.wells))
        n_rows = sum(len(w["predictions"]) for w in payload["resultData"])
        self.assertEqual(n_rows, 80)
        for w in payload["resultData"]:
            for row in w["predictions"][:5]:
                self.assertGreater(row["PERM"], 0.0)
                self.assertGreaterEqual(row["SW"], 0.0)
                self.assertLessEqual(row["SW"], 100.0)
                self.assertLessEqual(row["POR"], 40.0)

    def test_pd1_is_deterministic_and_sw_stays_percentage_scale(self):
        self._run()
        first = self.out.read_bytes()
        self._run()
        self.assertEqual(first, self.out.read_bytes(), "同一权重两次推理必须逐字节一致")
        payload = json.loads(first.decode("utf-8"))
        sw = [r["SW"] for w in payload["resultData"] for r in w["predictions"]]
        self.assertTrue(any(v > 1.0 for v in sw),
                        "SW 必须保持百分数尺度（不得整体压到 [0,1]）")

    def test_missing_checkpoint_fails_loudly(self):
        bad = self.root / "bad_registry.json"
        REG.register_pipeline("PD1", self.ckpt, path=bad)
        REG.set_version("PD1", available=True, checkpoint=str(self.root / "nope.pt"),
                        path=bad)
        out = subprocess.run([sys.executable, str(V4 / "predict.py"),
                              "--registry", str(bad), "--use-version", "PD1",
                              "--data_dir", str(self.root / "cache_src"),
                              "--output", str(self.out)],
                             capture_output=True, text=True, timeout=600)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("checkpoint", (out.stderr + out.stdout))

    def test_registered_but_unwired_version_fails_loudly(self):
        reg = self.root / "reg_unwired.json"
        REG.register_pipeline("PD9", self.ckpt, path=reg)
        out = subprocess.run([sys.executable, str(V4 / "predict.py"),
                              "--registry", str(reg), "--use-version", "PD9",
                              "--data_dir", str(self.root / "cache_src"),
                              "--output", str(self.out)],
                             capture_output=True, text=True, timeout=600)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("未接线", out.stderr + out.stdout)

    def test_unavailable_version_refuses(self):
        reg = self.root / "reg_unavail.json"
        REG.register_pipeline("PD3", self.ckpt, available=False, completed=False, path=reg)
        out = subprocess.run([sys.executable, str(V4 / "predict.py"),
                              "--registry", str(reg), "--use-version", "PD3",
                              "--data_dir", str(self.root / "cache_src"),
                              "--output", str(self.out)],
                             capture_output=True, text=True, timeout=600)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("NOT trained", out.stderr + out.stdout)


if __name__ == "__main__":
    unittest.main()
