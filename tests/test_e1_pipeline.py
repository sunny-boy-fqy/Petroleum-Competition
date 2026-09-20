"""E1 端到端集成测试（**torch 门控**）：`E1/code/train_row.py` 真的能跑出 Gate。

用**合成井**（`tests/_synth_cache.py`）跑完整链路，不依赖真实数据、不需要 NPU：
缓存构建 → 折内 fit → 两阶段训练（inner 选 epoch / 选 τ）→ OOF → metrics/gate/prereg/
loss_curve/candidates → 契约与 mandatory checks。

两条路径都测：
  * `--smoke`：不判 Gate、不登记候选（只验证链路与产物）；
  * 非 smoke（极小规模）：真的走 τ 选择与候选登记，并断言**仓库的 candidates.json 不被写**。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

REPO_CANDIDATES = V4 / "versions" / "candidates.json"


def _load_train_row():
    spec = importlib.util.spec_from_file_location(
        "v4_e1_train_row", str(V4 / "E1" / "code" / "train_row.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE1Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=8, n_train=8)
        cls.wells = tr_wells + va_wells
        cls.root = root
        cls.cache = root / "cache"
        cls.tr_dir, cls.te_dir, _ = SC.build_cache(cls.cache, cls.wells, n_rows=40, seed=11)
        cls.mod = _load_train_row()

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _args(self, tag: str, *extra: str) -> list[str]:
        return ["--train-dir", str(self.tr_dir), "--test-dir", str(self.te_dir),
                "--cache-root", str(self.cache),
                "--out-dir", str(self.root / tag / "runs"),
                "--reports-dir", str(self.root / tag / "reports"),
                "--scalers-dir", str(self.root / tag / "scalers"),
                "--candidates", str(self.root / tag / "candidates.json"),
                "--device", "cpu", *extra]

    # ---------------------------------------------------------------- smoke
    def test_smoke_runs_end_to_end_without_gate(self):
        repo_before = _sha(REPO_CANDIDATES)
        rc = self.mod.main(self._args("smoke", "--smoke"))
        self.assertEqual(rc, 0, "smoke 必须返回 0（不判 Gate）")
        rep = self.root / "smoke" / "reports"
        for name in ("E1_gate.json", "E1_metrics.json", "E1_P1_gate_prereg.json",
                     "E1_loss_curve.csv", "E1_row_features.json", "training_time_log.json",
                     "E1_const_baseline.json"):
            self.assertTrue((rep / name).is_file(), f"缺少产物 {name}")
        gate = json.loads((rep / "E1_gate.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["smoke"])
        # 六项 mandatory check 在 smoke 下也应全部为真（链路本身必须是健康的）
        for name, ok in gate["checks"].items():
            self.assertTrue(ok, f"smoke 下 mandatory check {name} 应为真")
        self.assertIn("passed", gate)
        self.assertEqual(gate["contract"]["ok"], True)
        # smoke 不登记候选：仓库与 tmp 的 candidates 都不该出现 E1_PD0
        self.assertEqual(_sha(REPO_CANDIDATES), repo_before, "不得污染仓库的 candidates.json")
        tmp_cand = self.root / "smoke" / "candidates.json"
        if tmp_cand.is_file():
            self.assertEqual(json.loads(tmp_cand.read_text())["candidates"], [])
        # 权重与 manifest
        fold_dir = self.root / "smoke" / "runs" / "fold0"
        for name in ("best.pt", "best.manifest.json", "last.pt", "last.manifest.json"):
            self.assertTrue((fold_dir / name).is_file(), f"缺少 {name}")
        # OOF 可读且行/列自洽
        import numpy as np
        with np.load(self.root / "smoke" / "runs" / "oof.npz", allow_pickle=True) as z:
            n = z["y_true"].shape[0]
            self.assertEqual(z["y_true"].shape, (n, 3))
            self.assertEqual(z["y_pred"].shape, (n, 3))
            self.assertEqual(z["q_atom"].shape, (n, 3))
            self.assertEqual(int(z["fold_of_row"].max()), 0)
            self.assertTrue(np.isfinite(z["y_pred"]).all())

    # ------------------------------------------------------------ 非 smoke
    def test_non_smoke_selects_tau_and_registers_candidate(self):
        repo_before = _sha(REPO_CANDIDATES)
        rc = self.mod.main(self._args("full", "--folds", "0", "--max-wells", "6",
                                      "--epochs", "2", "--patience", "9"))
        self.assertIn(rc, (0, 3), f"Gate 可过可不过，但不得是其它错误码（rc={rc}）")
        rep = self.root / "full" / "reports"
        gate = json.loads((rep / "E1_gate.json").read_text(encoding="utf-8"))
        self.assertFalse(gate["smoke"])
        self.assertIsInstance(gate["passed"], bool)
        met = json.loads((rep / "E1_metrics.json").read_text(encoding="utf-8"))
        # τ 必须在 (0,1) 内且来自 inner-OOF 选择
        taus = [f["tau"] for f in gate["folds"]]
        self.assertTrue(taus and all(len(t) == 3 for t in taus))
        for row in taus:
            for t in row:
                self.assertGreater(t, 0.0)
                self.assertLess(t, 1.0)
        self.assertEqual(met["tau_selected_on"], "inner_oof_official_total")
        self.assertTrue(met["no_interpolation"])
        # 逐折 delta 与聚合口径自洽
        self.assertAlmostEqual(met["delta_vs_const"],
                               met["oof_total"] - C.CONSTANT_BASELINE_OOF, places=6)
        self.assertEqual(len(met["folds"]), 1)
        # 候选必须写进 tmp，且 cv 恒等式成立（tools/check_consistency.py 的规则）
        cand = json.loads((self.root / "full" / "candidates.json").read_text(encoding="utf-8"))
        entry = [c for c in cand["candidates"] if c["candidate_id"] == "E1_PD0"][0]
        cv = entry["cv"]
        want = 100.0 * (0.30 * cv["por"] + 0.35 * cv["perm"] + 0.35 * cv["sw"])
        self.assertAlmostEqual(cv["total"], want, places=5)
        self.assertEqual(cv["missing_mode"], "drop")
        self.assertTrue(cv["selection_score_only"])
        self.assertEqual(entry["scalers"]["fitted_on"], "train_fold_only")
        # 仓库文件不得被写
        self.assertEqual(_sha(REPO_CANDIDATES), repo_before,
                         "测试不得污染仓库的 versions/candidates.json")

    def test_scalers_are_written_per_fold_and_only_train_wells(self):
        scal = self.root / "full" / "scalers" / "E1_fold0.json"
        self.assertTrue(scal.is_file())
        payload = json.loads(scal.read_text(encoding="utf-8"))
        tr_wells, va_wells = SC.fold0_wells(n_val=8, n_train=8)
        self.assertEqual(set(payload["train_wells"]), set(tr_wells[:6]))
        self.assertEqual(set(payload["val_wells"]), set(va_wells[:6]))
        fit_wells = set(payload["row_scaler"]["fit_wells"])
        self.assertTrue(fit_wells.issubset(set(payload["train_wells"])),
                        "标准化参数只能来自训练井")
        self.assertTrue(fit_wells.isdisjoint(payload["val_wells"]),
                        "验证井不得参与任何尺度拟合")
        for k in ("por_max", "sw_mu", "sw_sigma", "s_por", "s_sw"):
            self.assertIn(k, payload["target_scalers"])

    def test_prereg_is_never_rewritten(self):
        """预注册一旦写入，重跑不得改阈值（只能新建修订号）。"""
        rep = self.root / "full" / "reports"
        prereg = rep / "E1_P1_gate_prereg.json"
        before = json.loads(prereg.read_text(encoding="utf-8"))
        before_sha = _sha(prereg)
        self.mod.main(self._args("full", "--folds", "0", "--max-wells", "6",
                                 "--epochs", "1", "--patience", "9"))
        self.assertEqual(_sha(prereg), before_sha, "预注册文件被改写了")
        self.assertEqual(json.loads(prereg.read_text(encoding="utf-8")), before)

    def test_prereg_passes_the_gate_validator(self):
        from src.validation import gates as G
        rep = self.root / "full" / "reports"
        prereg = json.loads((rep / "E1_P1_gate_prereg.json").read_text(encoding="utf-8"))
        self.assertEqual(G.validate_prereg(prereg), [])
        self.assertEqual(prereg["thresholds"]["oof_total_min"], 78.0)
        self.assertEqual(prereg["thresholds"]["min_delta"], 7.5)
        for c in ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                  "training_time_log_valid", "checkpoint_resumable", "no_label_leak"):
            self.assertIn(c, prereg["mandatory_checks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
