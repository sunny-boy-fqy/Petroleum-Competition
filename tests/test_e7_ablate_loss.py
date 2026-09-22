"""E7/P0 端到端（**torch 门控**）：损失消融脚本的臂覆盖、内折纪律与产物。

断言要点：
  * exp1/exp6 子集可跑，报告含逐臂逐目标指标 + `perm_low_tail` + 联合原子 AUC；
  * `not_implemented` 必须显式列出 exp7（L_phys），绝不静默跳过；
  * `--smoke` 不写仓库 `versions/configs/loss_v1.json`（改写到 reports 下）；
  * 臂集合的定义与 E7/P0 §5 一致（可用 `build_arms` 在无训练的情况下核对数量）；
  * 未知臂 id 直接报错（不"跑 0 个臂然后说通过"）。
"""
from __future__ import annotations

import importlib.util
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

REPO_LOSS_CFG = V4 / "versions" / "configs" / "loss_v1.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(V4 / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestArmDefinitions(unittest.TestCase):
    """臂集合定义（不需要 torch）：与 E7/P0 §5 的数量逐组对齐。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load("v4_e7_ablate", "E7/code/ablate_loss.py")

    def test_group_counts(self):
        counts = {}
        for a in self.mod.build_arms("all", None):
            counts[a["group"]] = counts.get(a["group"], 0) + 1
        self.assertEqual(counts.get("exp1"), 3)
        self.assertEqual(counts.get("exp2"), 9)
        self.assertEqual(counts.get("exp3"), 3)
        self.assertEqual(counts.get("exp4"), 2)
        self.assertEqual(counts.get("exp6"), 2)
        self.assertEqual(counts.get("exp8"), 2)
        self.assertNotIn("exp5", counts, "exp5（边界聚焦）默认关，不进 all")

    def test_exp5_group_is_opt_in(self):
        arms = self.mod.build_arms("exp5", None)
        self.assertEqual(len(arms), 9)
        self.assertTrue(all(a["group"] == "exp5" for a in arms))

    def test_unknown_arm_raises(self):
        with self.assertRaises(SystemExit):
            self.mod.build_arms("exp1", "nope")

    def test_exp7_is_implemented_and_opt_in(self):
        self.assertFalse(any("exp7" in s for s in self.mod.NOT_IMPLEMENTED))
        arms = self.mod.build_arms("exp7", None)
        self.assertEqual(len(arms), 2)
        self.assertTrue(all(a["group"] == "exp7" for a in arms))
        self.assertTrue(all("lam3" in a["overrides"] for a in arms))

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(V4 / "E7" / "code" / "ablate_loss.py"),
                              "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--arm-set", "--arms", "--inner-only", "--eval-outer", "--lam1",
                     "--lam1-schedule", "--lam2", "--lam3", "--perm-over-weight",
                     "--aux-normalize", "--boundary-kappa", "--boundary-sigma",
                     "--perm-clamp", "--out-config"):
            self.assertIn(flag, out.stdout, flag)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestAblateLossPipeline(unittest.TestCase):
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
        cmd = [sys.executable, str(V4 / "E7" / "code" / "ablate_loss.py"),
               "--smoke", "--folds", "0", "--max-wells", "6", "--epochs", "2",
               "--hidden", "32", "--layers", "1",
               "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
               "--scalers-dir", str(self.scalers), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                              proc.stderr[-1500:])
        rep = self.reports / "E7_loss_ablation.json"
        return json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}

    def test_exp1_subset_reports_and_config(self):
        before = REPO_LOSS_CFG.read_text(encoding="utf-8") if REPO_LOSS_CFG.is_file() else None
        rep = self._run("--arm-set", "exp1")
        self.assertEqual(rep["n_arms"], 3)
        self.assertTrue(rep["inner_only"])
        self.assertIn(rep["best_arm"], [a["arm_id"] for a in rep["arms"]])
        for a in rep["arms"]:
            for key in ("total", "cont_total", "acc_por", "acc_perm", "acc_sw",
                        "joint_atom_auc", "perm_low_tail", "eval_mode"):
                self.assertIn(key, a)
            self.assertEqual(a["eval_mode"], "inner_only")
        self.assertEqual(rep["not_implemented"], [])
        self.assertIsNotNone(rep["recommended"])
        self.assertTrue((self.reports / "loss_v1_smoke.json").is_file())
        after = REPO_LOSS_CFG.read_text(encoding="utf-8") if REPO_LOSS_CFG.is_file() else None
        self.assertEqual(before, after, "smoke 不得写仓库 loss_v1.json")
        gate = json.loads((self.reports / "E7_P0_gate.json").read_text(encoding="utf-8"))
        for key in ("inner_only_selection", "no_label_leak", "perm_low_tail_reported",
                    "masked_mean_nan_unit_test", "contract_ok"):
            self.assertTrue(gate["checks"][key], key)
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")

    def test_exp6_clamp_arms_run_and_report_tail(self):
        """两条截断臂都必须跑出来并上报 PERM 尾部。

        注意：**合并成相同 loss 是合法结果**——合成标签里几乎不出现 ẑ−z < log10(ε)
        的极端低估，截断在该数据上不起作用。开关本身的有效性由
        `test_loss_ablation_lib.test_perm_clamp_penalizes_extreme_underestimate`
        在库层证明（那里构造了 5 个数量级的低估）。
        """
        rep = self._run("--arm-set", "exp6")
        ids = {a["arm_id"]: a for a in rep["arms"]}
        self.assertEqual(set(ids), {"exp6_clamp_on", "exp6_clamp_off"})
        for arm in ids.values():
            self.assertIsNotNone(arm["final_loss"])
            self.assertIn("frac_abs_dz_lt_1", arm["perm_low_tail"])
            self.assertIn("tail_low_rate", arm["perm_low_tail"])

    def test_custom_out_config_is_mirrored_to_reports(self):
        """审查 H4 回归：E7 写自定义配置路径时，也必须镜像到 --reports-dir。"""
        out_cfg = self.root / "custom_cfg" / "loss_v1.json"
        self._run("--arm-set", "exp1", "--out-config", str(out_cfg))
        self.assertTrue(out_cfg.is_file())
        self.assertTrue((self.reports / "loss_v1.json").is_file(),
                        "E7 配置必须镜像到 reports（随 mirror 持久化）")

    def test_exp7_lam3_arms_actually_apply_physics(self):
        """审查 H2 回归：exp7 的 lam3>0 必须真的产生非零 L_phys，而不是空操作。"""
        rep = self._run("--arm-set", "exp7")
        arms = {a["arm_id"]: a for a in rep["arms"]}
        self.assertEqual(set(arms), {"exp7_lam3_0.02", "exp7_lam3_0.05"})
        for arm in arms.values():
            self.assertGreater(arm["phys_loss"], 0.0,
                               f"{arm['arm_id']} 的 L_phys 不应为 0（空消融回归）")
        # 基线没有物理项 -> parts 里不应出现 phys
        base = self._run("--arm-set", "exp1")
        for arm in base["arms"]:
            self.assertEqual(arm["phys_loss"], 0.0)

    def test_arms_filter_and_unknown(self):
        rep = self._run("--arms", "exp1_align,exp1_aux")
        self.assertEqual(rep["n_arms"], 2)
        self._run("--arms", "does_not_exist", expect_rc=1)


if __name__ == "__main__":
    unittest.main()
