"""E8/P1 端到端（**torch 门控**）：井级分支开/关消融、λ 内折搜索与容量/身份收据。

断言要点（E8/P1 §7）：
  * **开/关消融必须完成**：每折都有完整的 λ 曲线，且 λ=0 的结果与"纯主干解码"**逐分相同**
    （不是拿 λ=0 跟自己做比较）；
  * λ 只在内折上选（记录 `lambda_star` 与内折曲线），外折只评估一次；
  * **容量硬上限**：井级分支隐藏宽 ≤ 主干/8（`capacity_within_cap`）；
  * **无井身份特征**：`assert_no_well_identity` 审计通过；
  * **逐井非退化比例**与配对 CI 同时达标才 `adopted`，否则 `no_go` 并写明原因；
  * 随机初始化骨干 + 2 epoch 的预检不应判 PASS（`passed is None`）。
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

SCRIPT = V4 / "E8" / "code" / "well_branch.py"
ARCH_KWARGS = {"patch_len": 16, "stride": 8, "d_model": 16, "n_layers": 1, "n_heads": 4,
               "max_tokens": 64}


class TestE8WellBranchContract(unittest.TestCase):
    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--pool", "--lambda-max", "--lambda-steps", "--lambda-train",
                     "--regression-margin", "--nondegrade-floor", "--pool-ablation"):
            self.assertIn(flag, out.stdout, flag)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE8WellBranchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr_wells + va_wells, n_rows=40, seed=11)
        cls.reports = root / "reports"
        cls.scalers = root / "scalers"
        cls.runs = root / "runs"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(SCRIPT), "--smoke", "--folds", "0", "--max-wells", "6",
               "--epochs", "2", "--arch", "patchtf", "--arch-kwargs", json.dumps(ARCH_KWARGS),
               "--chunk", "96", "--overlap", "32", "--lambda-steps", "5",
               "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
               "--run-root", str(self.runs), "--scalers-dir", str(self.scalers), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        rep = self.reports / "E8_well_branch.json"
        return json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}

    def test_ablation_and_receipts(self):
        rep = self._run()
        self.assertEqual(rep["stage"], "E8")
        self.assertEqual(rep["pool"], "attention")
        self.assertTrue(rep["selection_score_only"])
        self.assertEqual(len(rep["folds_detail"]), 1)
        fold = rep["folds_detail"][0]
        self.assertEqual(len(fold["lambda_curve"]), 5)
        self.assertTrue(fold["lambda0_identity_ok"], "λ=0 必须与纯主干解码逐分相同")
        self.assertAlmostEqual(fold["outer"]["lambda0"], fold["raw_backbone_total"], places=9)
        self.assertTrue(fold["capacity"]["ok"])
        self.assertLessEqual(fold["capacity"]["hidden"], fold["capacity"]["cap"])
        self.assertFalse(fold["well_identity_violations"])
        self.assertTrue(rep["well_identity_audit"]["ok"])
        for p in fold["per_well"]:
            self.assertIn("degraded", p)
            self.assertIn("delta", p)
        self.assertGreaterEqual(fold["non_degradation_ratio"], 0.0)
        self.assertLessEqual(fold["non_degradation_ratio"], 1.0)
        self.assertIn(rep["decision"], ("adopted", "no_go"))
        self.assertTrue(rep["reason"])

    def test_gate_checks(self):
        self._run()
        gate = json.loads((self.reports / "E8_P1_gate.json").read_text(encoding="utf-8"))
        for key in ("on_off_ablation_completed", "lambda_inner_only", "capacity_within_cap",
                    "no_well_identity_features", "per_well_non_degradation_reported",
                    "identity_at_lambda_zero", "contract_ok", "no_label_leak"):
            self.assertTrue(gate["checks"][key], key)
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        self.assertEqual(gate["decision"], json.loads(
            (self.reports / "E8_well_branch.json").read_text(encoding="utf-8"))["decision"])

    def test_mean_pool_arm(self):
        rep = self._run("--pool", "mean")
        self.assertEqual(rep["pool"], "mean")
        self.assertTrue(rep["folds_detail"][0]["capacity"]["ok"])

    def test_lambda_zero_disables_branch(self):
        """`--lambda-max 0` 时 λ 网格只有 0 → 不可能采纳（等于"关掉分支"的对照）。"""
        rep = self._run("--lambda-max", "0")
        fold = rep["folds_detail"][0]
        self.assertTrue(all(abs(c["lam"]) < 1e-12 for c in fold["lambda_curve"]))
        self.assertEqual(rep["decision"], "no_go")
        self.assertAlmostEqual(fold["outer"]["delta"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
