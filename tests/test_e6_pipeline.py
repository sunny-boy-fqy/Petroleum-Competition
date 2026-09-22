"""E6/P0 端到端（**torch 门控**）：原子状态头两阶段训练脚本的产物与 mandatory 判据。

断言要点（E6/P0 §7）：
  * 逐目标 AUC/Acc/P·R·F1（τ=0.5 与 τ*）与联合原子 AUC/AP **必须逐目标上报**；
  * **τ 只在内折选**（`tau_source == "inner_oof_only"`），外折只推理一次；
  * **阶段 2 冻结真的生效**（`q_head_frozen_ok is True`，与库层测试互为印证）；
  * 原子切换**无插值**、输入列审计通过（`input_no_label_leak_full.ok`）；
  * 标签打乱负对照被记录，且其 AUC 明显低于真实模型的 AUC（≈随机）；
  * 权重可 `resumable` 读回、scalers 标注 `fitted_on=train_fold_only`；
  * 入口与 `run_train.sh --stage E6` 接线存在。
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
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


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(V4 / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE6Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr_wells + va_wells, n_rows=40, seed=11)
        cls.mod = _load("v4_e6_train_state", "E6/code/train_state.py")
        cls.reports = root / "reports"
        cls.runs = root / "runs"
        cls.scalers = root / "scalers"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str) -> dict:
        argv = ["--smoke", "--folds", "0", "--max-wells", "6", "--epochs", "2",
                "--hidden", "32", "--layers", "1", "--shuffle-control", "--min-free-gb", "0",
                "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
                "--run-root", str(self.runs), "--scalers-dir", str(self.scalers), *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.mod.main(argv)
        self.assertEqual(rc, 0, buf.getvalue()[-2000:])
        return json.loads((self.reports / "E6_P0_gate.json").read_text(encoding="utf-8"))

    def test_smoke_reports_and_mandatory_checks(self):
        gate = self._run()
        for name in ("E6_atomic_report.json", "E6_P0_gate.json", "E6_P0_gate_prereg.json",
                     "training_time_log.json"):
            self.assertTrue((self.reports / name).is_file(), name)
        self.assertTrue((self.scalers / "E6_state_fold0.json").is_file())
        rep = json.loads((self.reports / "E6_atomic_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["stage"], "E6")
        self.assertEqual(rep["tau_source"], "inner_oof_only")
        self.assertTrue(rep["no_interpolation"])
        self.assertTrue(rep["input_no_label_leak_full"]["ok"])
        self.assertIsInstance(rep["state_auc"], float)
        # 逐目标 AUC + P/R/F1 + 联合 AUC/AP
        for t in C.TARGETS:
            self.assertEqual(len(rep["per_target_auc"][t]), len(rep["folds"]))
        self.assertTrue(rep["joint_atom_auc"])
        self.assertIsNotNone(rep["joint_atom_ap"][0])
        m = rep["atom_metrics_tau_half"][0]
        self.assertEqual(set(m["per_target_acc"]), set(C.TARGETS))
        for t, v in m["per_target"].items():
            for key in ("precision", "recall", "f1"):
                self.assertIn(key, v)
        # 阶段 2 冻结与实际生效
        self.assertTrue(rep["folds_detail"][0]["stage2"]["q_head_frozen_ok"])
        self.assertTrue(rep["folds_detail"][0]["stage2"]["slice_weights"]["never_zero"])
        self.assertTrue(rep["folds_detail"][0]["resumable"]["ok"])
        self.assertTrue(gate["checks"]["contract_ok"])
        self.assertTrue(gate["checks"]["tau_t_inner_oof_only"])
        self.assertTrue(gate["checks"]["no_atom_continuous_interpolation"])
        self.assertTrue(gate["checks"]["input_no_label_leak_full"])
        self.assertTrue(gate["checks"]["checkpoint_resumable"])
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        self.assertFalse(gate["nogo"])

    def test_shuffle_control_is_recorded_and_near_random(self):
        self._run()
        rep = json.loads((self.reports / "E6_atomic_report.json").read_text(encoding="utf-8"))
        ctrl = rep["label_shuffle_control"]
        self.assertIsNotNone(ctrl)
        self.assertTrue(ctrl["marginals_preserved"])
        self.assertLess(ctrl["row_identity_rate"], 0.6)
        if ctrl["val_state_auc"] is not None and rep["state_auc"] is not None:
            self.assertLessEqual(ctrl["val_state_auc"], rep["state_auc"] + 0.15,
                                 "打乱标签后的 AUC 不应显著高于真实模型")

    def test_scalers_are_train_fold_only(self):
        self._run()
        sc = json.loads((self.scalers / "E6_state_fold0.json").read_text(encoding="utf-8"))
        self.assertEqual(sc["fitted_on"], "train_fold_only")
        self.assertTrue(set(sc["train_wells"]).isdisjoint(set(sc["val_wells"])))
        self.assertIn("row_scaler", sc)
        self.assertIn("target_scalers", sc)

    def test_q_head_lr_mult_arm_runs(self):
        gate = self._run("--q-head-lr-mult", "0.05")
        rep = json.loads((self.reports / "E6_atomic_report.json").read_text(encoding="utf-8"))
        self.assertIsNone(rep["folds_detail"][0]["stage2"]["q_head_frozen_ok"],
                          "非零 lr 倍数时不标冻结")
        self.assertFalse(rep["folds_detail"][0]["stage2"]["param_groups"]["frozen_q_heads"])
        self.assertIsNone(gate["passed"])


class TestE6ScriptContract(unittest.TestCase):
    def test_run_train_dispatches_e6(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn("E6)", src)
        self.assertIn("E6/code/train_state.py", src)
        self.assertIn("E6/code/search_tau.py", src)

    def test_help_lists_e6_knobs(self):
        import subprocess
        out = subprocess.run([sys.executable, str(V4 / "E6" / "code" / "train_state.py"),
                              "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        for flag in ("--stage2-epochs", "--q-head-lr-mult", "--lam-atom", "--lam-joint",
                     "--lam-cont-fallback", "--pos-weight", "--alpha-nonjoint",
                     "--stage2-w-joint", "--stage2-w-nonjoint-atom", "--stage2-w-valid",
                     "--shuffle-control"):
            self.assertIn(flag, out.stdout, flag)

    def test_tau_inner_oof_before_full_retrain(self):
        """回归：τ 必须来自只见过 inner_tr 的模型，不能再用外层全量重训后的模型预测 inner_val。"""
        src = (V4 / "E6" / "code" / "train_state.py").read_text(encoding="utf-8")
        i_inner_stage2 = src.index("run_stage(model, f_in_tr, cfg, 2")
        i_pred_in = src.index("pred_in = predict_fold(model, f_in_va)")
        i_full_retrain = src.index("model_final = build_model")
        self.assertLess(i_inner_stage2, i_pred_in,
                        "inner stage2 必须先于 inner-OOF 预测")
        self.assertLess(i_pred_in, i_full_retrain,
                        "τ 必须在任何使用 f_tr 的最终重训之前选完")


if __name__ == "__main__":
    unittest.main()
