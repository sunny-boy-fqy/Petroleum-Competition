"""E6/P2 Gate 测试（**口径层，无 torch**）：证据齐全才可能过，缺证据一律 False。

`E6/code/gate.py` 只读证据文件，因此这里合成 P0/P1/P2 的产物把判据钉死：
  * 齐全 + `cpu_inference_ok` → `passed=True`（退出码 0）；
  * 缺任一 mandatory（含 `cpu_inference_ok`）→ `passed=False`（退出码 3），并在
    `aggregate.mandatory_failures` 里点名；
  * `oof_total < 82.0` / 联合 AUC < 0.9 / 原子 Acc·召回不达标 → 绝对门槛失败；
  * 缺 `E6_tau_search.json` 时 `tau_t_inner_oof_only=False`、`input_no_label_leak_full=False`
    （**不得**因为字段缺失而跳过）；
  * `--exploratory` 时 `passed=None`（只出证据，不判）。
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

from src import constants as C  # noqa: E402

SCRIPT = V4 / "E6" / "code" / "gate.py"


def _atomic(reports: Path, *, joint_auc=0.95, min_acc=0.995, min_recall=0.99,
            resumable=True, leak=False) -> None:
    inner, va = (["a"], ["b"]) if not leak else (["a"], ["a"])
    (reports / "E6_atomic_report.json").write_text(json.dumps({
        "stage": "E6", "joint_atom_auc": [joint_auc], "no_interpolation": True,
        "min_atom_acc": min_acc, "min_atom_recall": min_recall,
        "atom_metrics_tau_half": [{"per_target_acc": {t: min_acc for t in C.TARGETS},
                                   "per_target": {t: {"precision": 0.99, "recall": min_recall,
                                                      "f1": 0.99, "tau": 0.5}
                                                  for t in C.TARGETS}}],
        "folds_detail": [{"fold": 0, "inner_val": inner, "va_wells": va,
                          "resumable": {"ok": resumable}}],
    }), encoding="utf-8")


def _tau(reports: Path, *, no_interp=True, input_ok=True) -> None:
    (reports / "E6_tau_search.json").write_text(json.dumps({
        "stage": "E6", "tau_source": "inner_oof_only",
        "shared": {"no_interpolation": no_interp},
        "input_no_label_leak_full": {"ok": input_ok},
        "cont_slice": {"total": 80.0}, "gated_slice": {"total": 81.5},
    }), encoding="utf-8")


def _cv(path: Path, total: float = 82.5) -> None:
    path.write_text(json.dumps({"total": total, "por": 0.9, "perm": 0.6, "sw": 0.9}),
                    encoding="utf-8")


def _contract(path: Path, ok: bool = True, disk_ok: bool = True) -> None:
    path.write_text(json.dumps({"ok": ok, "disk": {"level": "ok" if disk_ok else "warn"}}),
                    encoding="utf-8")


def _oof(path: Path, *, good: bool = True) -> None:
    """合成外折 OOF：`good=True` 时 PD1 预测远好于 CONST 基线（delta≈+30）。"""
    import numpy as np
    n, n_wells = 60, 6
    rng = np.random.RandomState(0)
    y = np.zeros((n, 3), dtype="float64")
    y[:, 0] = 11.0 + rng.rand(n)            # 有效 POR（非原子）
    y[:, 1] = 50.0 + 10.0 * rng.rand(n)     # 有效 PERM
    y[:, 2] = 55.0 + 10.0 * rng.rand(n)     # 有效 SW（百分数尺度）
    # PD1 预测 = 真值（完美）；CONST 基线 = 原子常量 → delta 应显著为正
    gated = y.copy() if good else np.tile([C.ATOM_VALUES[t] for t in C.TARGETS], (n, 1))
    mask = np.ones((n, 3), dtype="float64")
    widx = np.repeat(np.arange(n_wells), n // n_wells)
    np.savez_compressed(path, y_true=y, gated=gated, mask=mask, well_index=widx,
                        well_ids=np.asarray([f"w{i}" for i in range(n_wells)], dtype=object),
                        cont=gated, q_atom=np.full((n, 3), 0.9), q_joint=np.full(n, 0.9),
                        fold_of_row=np.zeros(n, dtype="int64"), tau_row=np.full((n, 3), 0.5))


def _time_log(reports: Path) -> None:
    (reports / "training_time_log.json").write_text(json.dumps(
        {"folds": [{"fold": 0, "seconds": 12.0}]}), encoding="utf-8")


def _run(reports: Path, *extra: str) -> tuple[int, dict]:
    out = subprocess.run([sys.executable, str(SCRIPT), "--reports-dir", str(reports), *extra],
                         capture_output=True, text=True, timeout=600)
    gate = reports / "E6_gate.json"
    return out.returncode, (json.loads(gate.read_text(encoding="utf-8")) if gate.is_file()
                            else {})


class TestE6Gate(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.cv = self.root / "cv.json"
        self.contract = self.root / "contract.json"
        _atomic(self.reports)
        _tau(self.reports)
        _cv(self.cv)
        _contract(self.contract)
        _time_log(self.reports)
        self.oof = self.root / "oof.npz"
        _oof(self.oof)

    def tearDown(self):
        self._td.cleanup()

    def _args(self, *extra):
        return ["--cv", str(self.cv), "--contract-json", str(self.contract),
                "--oof", str(self.oof), *extra]

    def test_all_evidence_passes(self):
        rc, gate = _run(self.reports, *self._args("--cpu-inference-ok"))
        self.assertEqual(rc, 0, gate)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["oof_total"], 82.5)
        self.assertTrue(gate["checks"]["cpu_inference_ok"])
        self.assertEqual(gate["gate_id"], "E6_gate")
        self.assertFalse(gate["prereg_errors"])

    def test_missing_cpu_inference_fails(self):
        rc, gate = _run(self.reports, *self._args())
        self.assertEqual(rc, 3)
        self.assertFalse(gate["passed"])
        self.assertIn("cpu_inference_ok", gate["aggregate"]["mandatory_failures"])

    def test_low_oof_total_fails_absolute(self):
        _cv(self.cv, total=81.0)
        rc, gate = _run(self.reports, *self._args("--cpu-inference-ok"))
        self.assertEqual(rc, 3)
        self.assertFalse(gate["passed"])
        self.assertIn("oof_total_min", json.dumps(gate["aggregate"], ensure_ascii=False))

    def test_low_joint_auc_fails_absolute(self):
        _atomic(self.reports, joint_auc=0.5)
        rc, gate = _run(self.reports, *self._args("--cpu-inference-ok"))
        self.assertEqual(rc, 3)
        self.assertFalse(gate["passed"])
        self.assertIn("min_joint_atom_auc", json.dumps(gate["aggregate"], ensure_ascii=False))

    def test_missing_tau_report_is_not_skipped(self):
        (self.reports / "E6_tau_search.json").unlink()
        rc, gate = _run(self.reports, *self._args("--cpu-inference-ok"))
        self.assertEqual(rc, 3)
        self.assertFalse(gate["checks"]["tau_t_inner_oof_only"])
        self.assertFalse(gate["checks"]["input_no_label_leak_full"])

    def test_leak_and_nonresumable_detected(self):
        _atomic(self.reports, resumable=False, leak=True)
        rc, gate = _run(self.reports, *self._args("--cpu-inference-ok"))
        self.assertEqual(rc, 3)
        self.assertFalse(gate["checks"]["checkpoint_resumable"])
        self.assertFalse(gate["checks"]["no_label_leak"])

    def test_exploratory_does_not_judge(self):
        (self.reports / "E6_tau_search.json").unlink()
        rc, gate = _run(self.reports, *self._args("--exploratory"))
        self.assertEqual(rc, 0)
        self.assertIsNone(gate["passed"])

    def test_oof_sha256_and_delta_recorded(self):
        rc, gate = _run(self.reports, *self._args("--cpu-inference-ok"))
        self.assertEqual(rc, 0, gate)
        self.assertIsNotNone(gate["oof_sha256"])
        self.assertEqual(len(gate["oof_sha256"]), 64)
        self.assertGreater(gate["delta_vs_const"], 0.0)
        self.assertGreater(gate["paired_ci_vs_const"][0], 0.0)

    def test_missing_oof_means_no_delta(self):
        rc, gate = _run(self.reports, "--cv", str(self.cv), "--contract-json",
                        str(self.contract), "--cpu-inference-ok")
        self.assertEqual(rc, 3, "没有 OOF 就没有 delta/CI → 不得判过")
        self.assertIsNone(gate["delta_vs_const"])
        self.assertIsNone(gate["paired_ci_vs_const"][0])


if __name__ == "__main__":
    unittest.main()
