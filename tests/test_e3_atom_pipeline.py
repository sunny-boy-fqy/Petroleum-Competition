"""WP2 端到端：E3/code/train_atom.py（arrays 模式，torch 门控）。"""
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

from src.portability import HAS_TORCH  # noqa: E402

SCRIPT = V4 / "E3" / "code" / "train_atom.py"


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE3AtomPipeline(unittest.TestCase):
    def test_arrays_mode_runs_and_writes_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(0)
            n, d = 200, 5
            X = rng.normal(size=(n, d)).astype("float32")
            q = rng.random((n, 3))
            y = (q > 0.7).astype("float32")
            m = np.ones((n, 3), dtype="float32")
            j = (y.all(axis=1)).astype("float32")
            cont = (y * np.asarray([0.1, 0.01, 99.9]) +
                    (1 - y) * np.asarray([20.0, 10.0, 50.0])).astype("float32")
            np.savez(root / "train.npz", X=X[:140], y_atom=y[:140], mask=m[:140],
                     y_joint=j[:140], cont=cont[:140])
            np.savez(root / "val.npz", X=X[140:], y_atom=y[140:], mask=m[140:],
                     y_joint=j[140:], cont=cont[140:])
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--mode", "arrays",
                 "--train-npz", str(root / "train.npz"),
                 "--val-npz", str(root / "val.npz"),
                 "--reports-dir", str(root / "reports"),
                 "--run-root", str(root / "runs"),
                 "--epochs", "2", "--hidden", "8", "--batch-size", "32",
                 "--device", "cpu", "--smoke"],
                capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
            rep = json.loads((root / "reports" / "E3_atom_report.json").read_text(
                encoding="utf-8"))
            self.assertEqual(len(rep["rows"]), 1)
            row = rep["rows"][0]
            for key in ("fold", "checkpoint", "best_epoch", "metrics"):
                self.assertIn(key, row)
            ckpt = Path(row["checkpoint"])
            self.assertTrue(ckpt.is_file())
            self.assertTrue(ckpt.with_suffix(".manifest.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
