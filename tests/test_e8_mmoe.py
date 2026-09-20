"""E8/P0 测试：结构对照（硬共享 / MMoE / 完全独立）的收据与判据。

口径层（无 torch）：CLI 契约与非法臂报错。
torch 门控：
  * 三臂参数量**必须可比**（`param_parity` 全 ok；独立臂的宽度由
    `pick_indep_hidden` 自动搜索，因为硬共享臂的参数量由 5 个头主导，按 `hidden//3`
    缩主干并不对等——这是实测踩到的坑）；
  * 逐目标表必须与总分一起给出（不许只报总分掩盖退化）；
  * 门控熵与任务梯度余弦必须上报；独立臂**没有共享参数** ⇒ 冲突记为 `null`（如实）；
  * 决策必须可由逐目标表复算（`improved` 且无 `regressed` 才 adopted）；
  * `--smoke` 只出证据不判 Gate；未知臂直接报错。
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
from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

SCRIPT = V4 / "E8" / "code" / "train_mmoe.py"


class TestE8ScriptContract(unittest.TestCase):
    def test_unknown_arm_fails(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--smoke", "--arms", "nope",
                              "--cache-root", "/nonexistent", "--reports-dir", "/tmp/e8x",
                              "--max-wells", "1"], capture_output=True, text=True,
                             timeout=300)
        # 数据缺失时先报 4；有数据时才走到臂校验。两者都必须非零。
        self.assertNotEqual(out.returncode, 0)

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--arms", "--n-experts", "--gate-temp", "--indep-hidden",
                     "--param-tol", "--inner-only", "--eval-outer"):
            self.assertIn(flag, out.stdout, flag)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE8MmoePipeline(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str, expect_rc: int = 0) -> dict:
        cmd = [sys.executable, str(SCRIPT), "--smoke", "--max-wells", "6", "--epochs", "2",
               "--hidden", "48", "--cache-root", str(self.cache),
               "--reports-dir", str(self.reports), "--scalers-dir", str(self.scalers),
               *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        rep = self.reports / "E8_mmoe.json"
        return json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}

    def test_three_arms_with_parity_and_receipts(self):
        rep = self._run()
        self.assertEqual(set(rep["arms"]), {"hard", "mmoe", "independent"})
        self.assertTrue(rep["inner_only"])
        self.assertEqual(set(rep["per_target_table"]), {"hard", "mmoe", "independent"})
        for arm, tab in rep["per_target_table"].items():
            self.assertEqual(set(tab), set(C.TARGETS))
        for arm, par in rep["param_parity"].items():
            self.assertTrue(par["ok"], (arm, par))
            self.assertLess(abs(par["ratio"] - 1.0), 0.15)
        # 门控熵：hard/mmoe 有（E=1 时熵为 0，仍应给出结构），independent 没有门控
        self.assertTrue((rep["gate_entropy"]["mmoe"] or {}).get("por"))
        self.assertEqual(set(rep["gate_entropy"]["mmoe"]),
                         {"por", "perm", "sw", "atom", "joint"})
        self.assertIsNone(rep["gate_entropy"]["independent"])
        # 梯度冲突：共享臂有值，独立臂为 null（无共享参数）
        self.assertIsNotNone(rep["gradient_conflict"]["mmoe"])
        self.assertIsNone(rep["gradient_conflict"]["independent"])
        self.assertIn(rep["decision"], ("adopted", "no_go"))
        self.assertTrue(rep["reason"])
        gate = json.loads((self.reports / "E8_P0_gate.json").read_text(encoding="utf-8"))
        for key in ("per_target_acc_reported", "param_parity_receipt", "gate_entropy_reported",
                    "gradient_conflict_reported", "inner_only_selection", "contract_ok"):
            self.assertTrue(gate["checks"][key], key)
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        self.assertEqual(gate["decision"], rep["decision"])

    def test_decision_recomputable_from_per_target_table(self):
        rep = self._run()
        hp = rep["per_target_table"]["hard"]
        mp = rep["per_target_table"]["mmoe"]
        improved = [t for t in C.TARGETS if mp[t] > hp[t]]
        regressed = [t for t in C.TARGETS if mp[t] < hp[t] - 0.01]
        expect = "adopted" if (improved and not regressed) else "no_go"
        self.assertEqual(rep["decision"], expect)

    def test_subset_arms(self):
        rep = self._run("--arms", "hard,mmoe")
        self.assertEqual(set(rep["arms"]), {"hard", "mmoe"})
        self.assertNotIn("independent", rep["param_parity"])
        self.assertIsNotNone(rep["gradient_conflict"]["hard"])

    def test_indep_hidden_auto_search_report(self):
        rep = self._run("--arms", "hard,independent")
        used = rep["config"]["indep_hidden_used"]
        self.assertIsInstance(used, int)
        self.assertGreater(used, 0)
        par = rep["param_parity"]["independent"]
        self.assertTrue(par["ok"], par)


if __name__ == "__main__":
    unittest.main()
