"""E6/P1 测试（**口径层，无 torch**）：τ 搜索的网格/平台规则、逐目标指标与候选登记。

`E6/code/search_tau.py` 只读内折 OOF，因此这里**合成** OOF 把每条判据钉住：
  * τ 必须落在官方网格 `[0.05, 0.95]` 上，曲线长度 = 91，且 `tau_source=inner_oof_only`；
  * 逐目标原子 Acc / P·R·F1（τ* 与 τ=0.5 两档）与连续切片 Acc 都要上报；
  * `joint_guard` 默认关；打开时给出的决策只能是 `adopted`/`no_go` 且带配对 CI；
  * `--smoke` **不登记候选**（绝不污染仓库 `versions/candidates.json`）；
    非 smoke 且 `--candidates` 指向 tmp 时应写入 `PD1.atomic`；
  * 缺 OOF 时以退出码 4 明确失败（不是静默 0 分）。
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

from src import constants as C  # noqa: E402
from src.inference import atomic_gate as AG  # noqa: E402

SCRIPT = V4 / "E6" / "code" / "search_tau.py"
N_WELLS, ROWS_PER_WELL = 6, 20
N_ROWS = N_WELLS * ROWS_PER_WELL


def _make_oof(path: Path, seed: int = 0, signal: float = 0.6) -> None:
    """合成内折 OOF：`q_atom` 与"标签是否原子"相关（signal 控制可分性）。"""
    rng = np.random.RandomState(seed)
    y_atom = rng.rand(N_ROWS, 3) < np.array([0.4, 0.45, 0.5])
    q = np.clip(signal * y_atom + (1.0 - signal) * rng.rand(N_ROWS, 3), 0.01, 0.99)
    cont = rng.rand(N_ROWS, 3).astype("float64")
    for i, t in enumerate(C.TARGETS):
        cont[y_atom[:, i], i] = C.ATOM_VALUES[t] * (1.0 + 0.1 * rng.rand(int(y_atom[:, i].sum())))
    y_true = cont.copy()
    mask = np.ones((N_ROWS, 3), dtype="float64")
    well_index = np.repeat(np.arange(N_WELLS), ROWS_PER_WELL)
    np.savez_compressed(path, cont=cont, q_atom=q, q_joint=rng.rand(N_ROWS),
                        y_true=y_true, mask=mask, well_index=well_index,
                        fold_of_row=np.zeros(N_ROWS, dtype="int64"),
                        well_ids=np.asarray([f"w{i}" for i in range(N_WELLS)], dtype=object))


def _atomic_report(path: Path) -> None:
    path.write_text(json.dumps({
        "stage": "E6", "joint_atom_auc": [0.91], "no_interpolation": True,
        "folds_detail": [{"fold": 0, "inner_val": ["a"], "val_wells": ["b"],
                          "resumable": {"ok": True}}],
    }), encoding="utf-8")


def _time_log(reports: Path) -> None:
    (reports / "training_time_log.json").write_text(json.dumps(
        {"stage": "E6", "folds": [{"fold": 0, "seconds": 1.0}]}), encoding="utf-8")


def _run(reports: Path, oof: Path, *extra: str, expect_rc: int | None = 0) -> dict:
    out = subprocess.run([sys.executable, str(SCRIPT), "--oof", str(oof),
                          "--reports-dir", str(reports), *extra],
                         capture_output=True, text=True, timeout=900)
    if expect_rc is not None:
        assert out.returncode == expect_rc, f"rc={out.returncode}\n{out.stdout[-1200:]}\n{out.stderr[-1200:]}"
    gate = reports / "E6_P1_gate.json"
    return (json.loads(gate.read_text(encoding="utf-8")) if gate.is_file() else {})


class TestE6TauSearch(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.oof = self.root / "inner_oof.npz"
        _make_oof(self.oof)
        _atomic_report(self.reports / "E6_atomic_report.json")
        _time_log(self.reports)
        self.cand = self.root / "candidates.json"

    def tearDown(self):
        self._td.cleanup()

    def _run(self, reports: Path, oof: Path, *extra: str, expect_rc: int = 0) -> dict:
        return _run(reports, oof, *extra, expect_rc=expect_rc)

    def test_smoke_reports_and_grid(self):
        gate = self._run(self.reports, self.oof, "--smoke", "--atomic-report",
                         str(self.reports / "E6_atomic_report.json"),
                         "--candidates", str(self.cand))
        rep = json.loads((self.reports / "E6_tau_search.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["tau_source"], "inner_oof_only")
        self.assertEqual(len(rep["tau"]), 3)
        grid = AG.default_tau_grid()
        for v in rep["tau"]:
            self.assertTrue(any(abs(v - float(g)) < 1e-9 for g in grid), v)
        self.assertEqual(rep["grid"]["n"], len(grid))
        self.assertEqual(set(rep["curve"]), set(C.TARGETS))
        for t in C.TARGETS:
            self.assertEqual(len(rep["curve"][t]), len(grid))
            self.assertEqual(len(rep["plateau"][t]), 2)
        for t in C.TARGETS:
            v = rep["per_target_atom_metrics_tau_star"][t]
            for key in ("precision", "recall", "f1"):
                self.assertIn(key, v)
            self.assertIn(t, rep["per_target_atom_acc_tau_star"])
            self.assertIn(t, rep["per_target_atom_acc_tau_half"])
        self.assertIsNotNone(rep["min_atom_acc"])
        self.assertIsNotNone(rep["min_atom_recall"])
        self.assertIn("total", rep["misclassification_cost"])
        self.assertEqual(rep["joint_guard"]["decision"], "off")
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        self.assertTrue(gate["checks"]["tau_t_inner_oof_only"])
        self.assertTrue(gate["checks"]["per_target_atom_precision_recall_f1_reported"])

    def test_smoke_does_not_register_candidate(self):
        self._run(self.reports, self.oof, "--smoke", "--candidates", str(self.cand))
        self.assertFalse(self.cand.exists(), "smoke 不得登记候选（哪怕路径在 tmp）")

    def test_non_smoke_registers_pd1_atomic(self):
        from src.versioning import registry as REG
        _run(self.reports, self.oof, "--exploratory", "--candidates", str(self.cand))
        e = REG.find_candidate("PD1", self.cand)
        self.assertIsNotNone(e)
        self.assertEqual(e["stage"], "E6")
        self.assertEqual(len(e["atomic"]["tau"]), 3)
        self.assertEqual(e["atomic"]["tau_source"], "inner_oof_only")
        self.assertEqual(e["status"], "local_only")
        self.assertIn("oof_sha256", e["atomic"])

    def test_joint_guard_decision_is_documented(self):
        _run(self.reports, self.oof, "--smoke", "--joint-guard", "--tau-joint-high", "0.5",
             "--iters", "200")
        rep = json.loads((self.reports / "E6_tau_search.json").read_text(encoding="utf-8"))
        g = rep["joint_guard"]
        self.assertTrue(g["enabled"])
        self.assertIn(g["decision"], ("adopted", "no_go"))
        self.assertEqual(len(g["paired_ci"]), 2)
        if g["decision"] == "no_go":
            self.assertIn("NO-GO", g["reason"])

    def test_missing_oof_fails_loudly(self):
        _run(self.reports, self.root / "nope.npz", "--smoke",
             "--candidates", str(self.cand), expect_rc=4)

    def test_atomic_report_absence_is_marked_not_faked(self):
        (self.reports / "E6_atomic_report.json").unlink()
        gate = self._run(self.reports, self.oof, "--smoke", "--candidates", str(self.cand))
        self.assertFalse(gate["checks"]["joint_atom_auc_reported"],
                         "缺 P0 报告时必须判 False（不得伪造 AUC）")
        self.assertFalse(gate["passed"] if gate["passed"] is not None else False)


class TestE6Routing(unittest.TestCase):
    def test_run_train_routes_phase(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn("E6/code/train_state.py", src)
        self.assertIn("E6/code/search_tau.py", src)
        self.assertIn("--phase", src)

    def test_help_lists_tau_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--oof", "--atomic-report", "--joint-guard", "--tau-joint-high",
                     "--tol", "--iters", "--candidates", "--baseline-oof"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
