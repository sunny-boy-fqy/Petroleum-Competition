"""E5 汇总测试（**口径层，无 torch**）：三目标联合判据 / 可复算 / 对齐 / swing 检查。

`E5/code/evaluate_targets.py` 只读 `oof.npz` 与三份报告，因此这里**合成**这些输入，
把每条判据单独钉住：
  * 报告数字必须能由 OOF 复算（不一致 → `recomputable=false` → Gate 失败）；
  * 三目标 OOF 的井序必须一致（不一致 → `wells_aligned=false`，不得给联合结论）；
  * 联合判据：有显著提升且无目标退步 > 0.01 → `accepted`；有退步 → `no_go`；
  * 缺件 → `incomplete`（**不**用两个目标的成绩冒充三目标结论）；
  * swing 检查：`ΔTotal_t = w_t·ΔAcc_t`，且联合增量等于各边际之和（可加性）。
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

SCRIPT = V4 / "E5" / "code" / "evaluate_targets.py"
N_WELLS = 6
ROWS_PER_WELL = 10
N_ROWS = N_WELLS * ROWS_PER_WELL
WELL_IDS = [f"w{i:02d}" for i in range(N_WELLS)]


def _base_arrays(seed: int = 0, base_por_err: float = 0.0, head_por_err: float = 0.0,
                 base_perm_err: float = 0.0, head_perm_err: float = 0.0,
                 base_sw_err: float = 0.0, head_sw_err: float = 0.0):
    """合成三目标 OOF：`base_*` 是冻结骨干的误差，`head_*` 是 E5 头的误差。"""
    rng = np.random.RandomState(seed)
    well_index = np.repeat(np.arange(N_WELLS), ROWS_PER_WELL)
    mask = np.ones(N_ROWS, dtype="float64")
    y_atom = np.zeros((N_ROWS, 3), dtype=bool)
    y_por = 10.0 + 5.0 * rng.rand(N_ROWS)
    z_true = -1.0 + 2.0 * rng.rand(N_ROWS)
    y_sw = 40.0 + 40.0 * rng.rand(N_ROWS)
    return {"well_index": well_index, "mask": mask, "y_atom": y_atom,
            "y_por": y_por, "por_cont": y_por + head_por_err,
            "base_cont": y_por + base_por_err,
            "z": z_true, "perm_z": z_true + head_perm_err,
            "base_z": z_true + base_perm_err,
            "sw": y_sw, "sw_pred": y_sw + head_sw_err, "base_sw": y_sw + base_sw_err}


def _write_case(root: Path, *, base_por_err=0.0, head_por_err=0.0, base_perm_err=0.0,
                head_perm_err=0.0, base_sw_err=0.0, head_sw_err=0.0,
                por_ci=(0.5, 0.9), perm_ci=(0.5, 0.9), sw_ci=(0.5, 0.9),
                skip=(), well_ids=None, mismatch_report=None):
    reports = root / "reports"
    run_root = root / "runs"
    (reports).mkdir(parents=True, exist_ok=True)
    a = _base_arrays(base_por_err=base_por_err, head_por_err=head_por_err,
                     base_perm_err=base_perm_err, head_perm_err=head_perm_err,
                     base_sw_err=base_sw_err, head_sw_err=head_sw_err)
    a["y"] = a["y_por"]                 # POR 目标的 OOF 真值键名（与 head_por.py 一致）
    ids = np.asarray(well_ids or WELL_IDS, dtype=object)
    # 每个目标的连续切片/有效行 Acc（供报告用；脚本会用 OOF 复算做一致性校验）
    specs = {
        "POR": ("por", "por_cont", "base_cont", "y_por", "por_cont_acc", por_ci),
        "PERM": ("perm", "perm_z", "base_z", "z", "perm_cont_acc", perm_ci),
        "SW": ("sw", "sw_pred", "base_sw", "sw", "sw_valid_acc", sw_ci),
    }
    for target, (sub, pred_k, base_k, y_k, acc_key, ci) in specs.items():
        d = run_root / "E5" / sub
        d.mkdir(parents=True, exist_ok=True)
        if target in skip:
            continue
        np.savez_compressed(
            d / "oof.npz", **{k: v for k, v in a.items()},
            well_ids=ids)
        # 报告：Acc 由 OOF 复算口径给出（POR/PERM 连续切片；SW 有效行）
        sel = np.ones(N_ROWS, dtype=bool)
        acc = _acc(target, a[y_k], a[pred_k], sel)
        base = _acc(target, a[y_k], a[base_k], sel)
        if mismatch_report == target:
            acc = acc + 0.02                      # 故意与 OOF 不一致
        (reports / {"POR": "E5_por.json", "PERM": "E5_perm.json",
                    "SW": "E5_sw.json"}[target]).write_text(json.dumps({
            "stage": "E5", "target": target, acc_key: acc,
            "paired_ci": list(ci), "delta_cont": acc - base,
            "placeholder_rows": {"hit_rate": {t: 1.0 for t in ("POR", "PERM", "SW")}},
            "folds_detail": [{"fold": 0, "inner_val": ["a"], "va_wells": ["b"],
                              "seconds": 1.0,
                              "delta_cont": acc - base, "delta_valid": acc - base,
                              "sw_valid_acc": acc},
                             {"fold": 1, "inner_val": ["c"], "va_wells": ["d"],
                              "seconds": 1.0,
                              "delta_cont": acc - base, "delta_valid": acc - base,
                              "sw_valid_acc": acc}],
        }), encoding="utf-8")
    (reports / "training_time_log.json").write_text(json.dumps(
        {"stage": "E5", "folds": [{"fold": 0, "seconds": 1.0},
                                  {"fold": 1, "seconds": 1.0}]}), encoding="utf-8")
    return reports, run_root


def _acc(target, y, pred, sel):
    from src.score import acc_relative
    if target == "PERM":
        d = np.maximum(np.asarray(pred) - np.asarray(y), np.log10(C.EPS))
        return float(np.clip(1.0 - np.abs(d), 0.0, 1.0).mean())
    delta = C.DELTA_POR if target == "POR" else C.DELTA_SW
    return float(acc_relative(np.asarray(y), np.asarray(pred), delta))


def _run(reports: Path, run_root: Path, *extra: str) -> dict:
    out = subprocess.run([sys.executable, str(SCRIPT), "--reports-dir", str(reports),
                          "--run-root", str(run_root), *extra],
                         capture_output=True, text=True, timeout=600)
    gate_path = reports / "E5_gate.json"
    if not gate_path.is_file():
        raise AssertionError(f"未产出 E5_gate.json；rc={out.returncode}\n{out.stdout[-1500:]}\n"
                             f"{out.stderr[-1500:]}")
    return {"rc": out.returncode,
            "gate": json.loads(gate_path.read_text(encoding="utf-8")),
            "per_target": json.loads((reports / "E5_per_target.json")
                                     .read_text(encoding="utf-8"))}


class TestE5Aggregate(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_accepted_when_significant_and_no_regression(self):
        reports, run_root = _write_case(self.root, base_por_err=3.0, head_por_err=0.0)
        r = _run(reports, run_root)
        self.assertEqual(r["gate"]["decision"], "accepted", r["gate"]["reason"])
        self.assertTrue(r["gate"]["passed"])
        self.assertEqual(r["rc"], 0)
        pt = r["per_target"]
        self.assertTrue(pt["targets"]["POR"]["significant"])
        self.assertTrue(pt["swing"]["additivity_ok"])
        self.assertAlmostEqual(pt["swing"]["combined_delta_total"],
                               pt["swing"]["per_target_marginal"]["POR"], places=9)

    def test_no_go_when_regression_beyond_margin(self):
        reports, run_root = _write_case(self.root, base_por_err=3.0, head_por_err=0.0,
                                        head_perm_err=2.0)
        r = _run(reports, run_root)
        self.assertEqual(r["gate"]["decision"], "no_go")
        self.assertFalse(r["gate"]["passed"])
        self.assertIn("PERM", r["gate"]["reason"])
        self.assertFalse(r["gate"]["checks"]["no_regression_beyond_margin"])
        self.assertEqual(r["rc"], 3, "Gate 失败必须以退出码 3 结束")

    def test_incomplete_on_missing_target(self):
        reports, run_root = _write_case(self.root, skip=("SW",))
        r = _run(reports, run_root)
        self.assertEqual(r["gate"]["decision"], "incomplete")
        self.assertFalse(r["gate"]["checks"]["sw_report_present"])
        self.assertEqual(r["per_target"]["targets"]["SW"]["status"], "pending")
        self.assertFalse(r["gate"]["passed"])

    def test_misaligned_wells_fail_gate(self):
        reports, run_root = _write_case(self.root)
        # 只把 SW 的井序打乱（其余两个保持参考顺序）
        shuffled = list(WELL_IDS)
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        a = _base_arrays()
        a["y"] = a["y_por"]
        np.savez_compressed(run_root / "E5" / "sw" / "oof.npz", **a,
                            well_ids=np.asarray(shuffled, dtype=object))
        r = _run(reports, run_root)
        self.assertFalse(r["gate"]["checks"]["wells_aligned"])
        self.assertEqual(r["gate"]["decision"], "incomplete")
        self.assertFalse(r["per_target"]["swing"]["aligned"])

    def test_recompute_mismatch_fails_gate(self):
        reports, run_root = _write_case(self.root, mismatch_report="PERM")
        r = _run(reports, run_root)
        self.assertFalse(r["gate"]["checks"]["recomputable"])
        self.assertFalse(r["per_target"]["targets"]["PERM"]["recomputable"])
        self.assertFalse(r["gate"]["passed"])

    def test_swing_uses_official_weights(self):
        reports, run_root = _write_case(self.root)
        r = _run(reports, run_root)
        w = dict(zip(("POR", "PERM", "SW"), C.TARGET_WEIGHTS))
        for t, marginal in r["per_target"]["swing"]["per_target_marginal"].items():
            delta = r["per_target"]["targets"][t]["delta"]
            self.assertAlmostEqual(marginal, w[t] * delta, places=9)

    def test_run_train_orchestrates_all_targets(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn("E5/code/evaluate_targets.py", src)
        self.assertIn("all)", src)

    def test_exploratory_does_not_judge(self):
        reports, run_root = _write_case(self.root)
        r = _run(reports, run_root, "--exploratory")
        self.assertIsNone(r["gate"]["passed"])
        self.assertEqual(r["rc"], 0)


if __name__ == "__main__":
    unittest.main()
