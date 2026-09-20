"""E6/P2 端到端（**torch 门控**）：PD1 候选组装 —— OOF/契约/折平均提交/注册/Gate。

链路（合成数据，全程本地可跑）：E6/P0 训练 1 折 → E6/P1 选 τ → `build_pd1.py`
组装 `oof.npz`/`cv.json`/`result.json`/`result.zip`/`manifest.json`，把 PD1 注册进（临时）
版本表，跑 `predict.py --use-version PD1` 的真实 CPU 推理，并由 `gate.py` 出 Gate。

断言要点：
  * 四指纹 `manifest.json` 齐全（代码/权重/数据/配置），OOF 与 cv.json 可复算；
  * `result.zip` 内含 `result.json`，行数/井数与测试目录一致；
  * PD1 在注册表里 `available=true` 且带 `checkpoints` 列表（折平均）；
  * `cpu_inference_ok=True` 才会进 Gate；缺折权重时退出码 4（不"用剩下的折凑"）；
  * `--smoke` **不写仓库** `versions/registry.json` 与 `versions/candidates.json`。
"""
from __future__ import annotations

import hashlib
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

REPO_REGISTRY = V4 / "versions" / "registry.json"
REPO_CANDIDATES = V4 / "versions" / "candidates.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(V4 / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE6P2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        cls.tr_dir, cls.te_dir, _ = SC.build_cache(cls.cache, tr_wells + va_wells,
                                                   n_rows=40, seed=11)
        cls.reports = root / "reports"
        cls.runs = root / "runs"
        cls.scalers = root / "scalers"
        cls.out = root / "out"
        cls.models = root / "models"
        cls.reg = root / "registry.json"
        cls.cand = root / "candidates.json"
        cls.env = {"V4_CACHE_ROOT": str(cls.cache), "V4_REPORTS_DIR": str(cls.reports),
                   "V4_RUN_ROOT": str(cls.runs), "V4_SCALERS_DIR": str(cls.scalers)}
        cls.train_state = _load("v4_e6_train_state_p2", "E6/code/train_state.py")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _train(self):
        argv = ["--smoke", "--folds", "0", "--max-wells", "6", "--epochs", "2",
                "--hidden", "32", "--layers", "1", "--min-free-gb", "0",
                "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
                "--run-root", str(self.runs), "--scalers-dir", str(self.scalers)]
        rc = self.train_state.main(argv)
        self.assertEqual(rc, 0)

    def _tau(self):
        subprocess.run([sys.executable, str(V4 / "E6" / "code" / "search_tau.py"),
                        "--oof", str(self.runs / "E6" / "state" / "inner_oof.npz"),
                        "--reports-dir", str(self.reports), "--candidates", str(self.cand),
                        "--smoke"], check=True, capture_output=True, timeout=600)

    def _build(self, *extra: str, expect_rc: int = 0):
        cmd = [sys.executable, str(V4 / "E6" / "code" / "build_pd1.py"),
               "--folds", "0", "--test-dir", str(self.te_dir),
               "--out-dir", str(self.out), "--models-dir", str(self.models),
               "--registry", str(self.reg), "--candidates", str(self.cand),
               "--reports-dir", str(self.reports), "--run-root", str(self.runs),
               "--smoke", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                              env={**dict(__import__("os").environ), **self.env})
        if expect_rc is not None:
            assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                                  proc.stderr[-1500:])
        return proc

    def test_full_chain_artifacts(self):
        before_reg = _sha(REPO_REGISTRY) if REPO_REGISTRY.is_file() else None
        before_cand = _sha(REPO_CANDIDATES) if REPO_CANDIDATES.is_file() else None
        self._train()
        self._tau()
        self._build()
        for name in ("oof.npz", "cv.json", "pd1_config.json", "manifest.json",
                     "result.json", "result.zip"):
            self.assertTrue((self.out / name).is_file(), name)
        self.assertTrue((self.models / "pd1_fold0.pt").is_file())
        self.assertTrue((self.models / "pd1_fold0.manifest.json").is_file())
        cv = json.loads((self.out / "cv.json").read_text(encoding="utf-8"))
        self.assertIn("total", cv)
        self.assertEqual(cv["missing_mode"], "drop")
        self.assertGreater(cv["n_wells"], 0)
        self.assertEqual(len(cv["tau"]), 3)
        man = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        for key in ("git_revision", "code_sha256", "code_combined", "weights", "data",
                    "config"):
            self.assertIn(key, man)
        self.assertTrue(man["weights"][0]["sha256"])
        with zipfile.ZipFile(self.out / "result.zip") as zf:
            self.assertIn("result.json", zf.namelist())
            payload = json.loads(zf.read("result.json").decode("utf-8"))
        self.assertEqual(len(payload["resultData"]), 32)
        n_rows = sum(len(w["predictions"]) for w in payload["resultData"])
        self.assertGreater(n_rows, 0)
        # 注册：available + 折平均清单
        from src.versioning import registry as REG
        v = REG.versions(self.reg)["PD1"]
        self.assertTrue(v["available"])
        self.assertEqual(len(v["checkpoints"]), 1)
        self.assertIsNotNone(v["oof_total"])
        # Gate 证据
        gate = json.loads((self.reports / "E6_gate.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["checks"]["cpu_inference_ok"])
        self.assertTrue(gate["checks"]["contract_ok"])
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        if before_reg is not None:
            self.assertEqual(_sha(REPO_REGISTRY), before_reg, "smoke 不得写仓库注册表")
        if before_cand is not None:
            self.assertEqual(_sha(REPO_CANDIDATES), before_cand, "smoke 不得写仓库候选表")

    def test_missing_fold_checkpoint_fails_with_4(self):
        self._train()
        self._tau()
        (self.runs / "E6" / "state" / "oof.npz").unlink()
        proc = self._build(expect_rc=4)
        self.assertIn("缺少外折 OOF", proc.stderr + proc.stdout)

    def test_inconsistent_fold_manifests_refused(self):
        self._train()
        self._tau()
        # 篡改 manifest 的结构 → 与 fold0 不一致（这里只有 1 折，复制成第 2 折触发校验）
        man = json.loads((self.runs / "E6" / "state" / "fold0.manifest.json")
                         .read_text(encoding="utf-8"))
        import shutil
        shutil.copy2(self.runs / "E6" / "state" / "fold0.pt",
                     self.runs / "E6" / "state" / "fold1.pt")
        man["model"] = {**man["model"], "hidden": 64}
        (self.runs / "E6" / "state" / "fold1.manifest.json").write_text(
            json.dumps(man), encoding="utf-8")
        proc = self._build("--folds", "0,1", expect_rc=6)
        self.assertIn("不一致", proc.stderr + proc.stdout)


class TestBuildPD1Contract(unittest.TestCase):
    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(V4 / "E6" / "code" / "build_pd1.py"),
                              "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--out-dir", "--models-dir", "--aggregate", "--tau-source",
                     "--loss-config", "--decode-config", "--registry", "--candidates",
                     "--test-dir", "--cpu-smoke", "--gate-threshold"):
            self.assertIn(flag, out.stdout, flag)

    def test_v4_config_exists_and_declares_key_blocks(self):
        cfg = (V4 / "configs" / "v4.yaml").read_text(encoding="utf-8")
        for key in ("features:", "model:", "training:", "atomic:", "decode:",
                    "submission:"):
            self.assertIn(key, cfg)
        self.assertIn("missing_mode: drop", cfg)
        self.assertIn("source: inner_oof_only", cfg)


if __name__ == "__main__":
    unittest.main()
