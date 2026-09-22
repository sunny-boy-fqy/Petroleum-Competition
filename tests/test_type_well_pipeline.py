"""WP8 端到端：E8/code/type_well_report.py（纯 numpy）。"""
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

SCRIPT = V4 / "E8" / "code" / "type_well_report.py"


class TestTypeWellReport(unittest.TestCase):
    def test_report_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wells = [f"w{i}" for i in range(6)]
            _, _, cache = SC.build_cache(root / "cache", wells, n_rows=60, seed=3)
            reports = root / "reports"
            proc = subprocess.run([sys.executable, str(SCRIPT),
                                   "--cache-root", str(cache),
                                   "--reports-dir", str(reports),
                                   "--topk", "2", "--method", "signature"],
                                  capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
            rep = json.loads((reports / "E8_type_well.json").read_text(encoding="utf-8"))
            self.assertIn("results", rep)
            self.assertGreaterEqual(len(rep["results"]), 1)
            for v in rep["results"].values():
                self.assertLessEqual(len(v["candidates"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
