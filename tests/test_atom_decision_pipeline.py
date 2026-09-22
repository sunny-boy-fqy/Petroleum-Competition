"""WP1 端到端（纯 numpy）：E7/code/fit_atom_decision.py 的产物与纪律。"""
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

SCRIPT = V4 / "E7" / "code" / "fit_atom_decision.py"


def _make_oof(path: Path, n: int = 3000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.0, 1.0, (n, 3))
    atom = q > 0.8
    atoms = np.asarray([0.1, 0.01, 99.9])
    valid = np.asarray([20.0, 10.0, 50.0])
    y = np.empty((n, 3), dtype="float64")
    cont = np.empty((n, 3), dtype="float64")
    for j in range(3):
        y[:, j] = np.where(atom[:, j], atoms[j], valid[j])
        cont[:, j] = valid[j]
    mask = np.ones((n, 3), dtype="float64")
    well_index = (np.arange(n) // 150).astype("int64")
    fold_of_row = (well_index % 5).astype("int64")
    np.savez(path, cont=cont, q_atom=q, y_true=y, mask=mask,
             y_atom=atom.astype("float64"), well_index=well_index,
             fold_of_row=fold_of_row)


class TestFitAtomDecisionPipeline(unittest.TestCase):
    def test_report_and_no_config_when_not_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            oof = root / "oof.npz"
            _make_oof(oof)
            out_cfg = root / "decode_v1.json"
            reports = root / "reports"
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--oof", str(oof),
                 "--out-config", str(out_cfg), "--reports-dir", str(reports),
                 "--calibrate", "temperature", "--smoke"],
                capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
            rep_path = reports / "E7_atom_decision.json"
            self.assertTrue(rep_path.is_file())
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            for key in ("baseline", "crossfit_decision", "paired_ci", "adopted",
                        "atom_calibration", "action_table_summary"):
                self.assertIn(key, rep)
            self.assertIsInstance(rep["adopted"], bool)
            # smoke 不采纳时不得写坏 decode 配置
            if not rep["adopted"]:
                self.assertFalse(out_cfg.is_file())

    def test_missing_oof_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--oof", str(Path(td) / "nope.npz"),
                 "--reports-dir", str(Path(td) / "reports")],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(proc.returncode, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
