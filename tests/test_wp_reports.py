"""WP9/WP10/WP11 CLI 端到端（合成 cache；纯 numpy / sklearn 可选）。"""
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
from src.validation import folds as FOLDS  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

SCRIPTS = {
    "quality": V4 / "E2" / "code" / "report_data_quality.py",
    "petro": V4 / "E2" / "code" / "report_petro.py",
    "tabular": V4 / "E8" / "code" / "tabular_member.py",
}


def _cache_with_first_wells(root: Path, n: int = 8):
    wells = list(FOLDS.load_folds()["well_list"])[:n]
    _, _, cache = SC.build_cache(root / "cache", wells, n_rows=50, seed=7)
    return wells, cache


class TestWPReports(unittest.TestCase):
    def test_data_quality_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wells, cache = _cache_with_first_wells(root, 8)
            reports = root / "reports"
            p = subprocess.run([sys.executable, str(SCRIPTS["quality"]),
                                "--cache-root", str(cache), "--reports-dir", str(reports),
                                "--max-wells", "8", "--smoke"],
                               capture_output=True, text=True, timeout=600)
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            rep = json.loads((reports / "E2_data_quality.json").read_text(encoding="utf-8"))
            self.assertIn("missing", rep)
            self.assertIn("impute", rep)
            self.assertIn("outliers", rep)

    def test_petro_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wells, cache = _cache_with_first_wells(root, 8)
            reports, runs = root / "reports", root / "runs"
            p = subprocess.run([sys.executable, str(SCRIPTS["petro"]),
                                "--cache-root", str(cache), "--reports-dir", str(reports),
                                "--run-root", str(runs), "--max-wells", "8", "--smoke"],
                               capture_output=True, text=True, timeout=600)
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            rep = json.loads((reports / "E2_petro_features.json").read_text(encoding="utf-8"))
            self.assertIn("stats", rep)
            self.assertEqual(len(rep["features"]), 18)
            self.assertTrue((runs / "E2" / "petro" / "petro_sample.npz").is_file())

    @unittest.skipUnless(HAS_SKLEARN, "sklearn not installed")
    def test_type_well_member_gbdt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tr, va = SC.fold0_wells(n_val=4, n_train=8)
            wells = tr[:8] + va[:4]
            _, _, cache = SC.build_cache(root / "cache", wells, n_rows=60, seed=13)
            reports, runs = root / "reports", root / "runs"
            p = subprocess.run([sys.executable, str(V4 / "E8" / "code" / "type_well_member.py"),
                                "--kind", "gbdt", "--gbdt-kind", "histgb",
                                "--folds", "0", "--max-wells", "4", "--topk", "2",
                                "--cache-root", str(cache), "--reports-dir", str(reports),
                                "--run-root", str(runs), "--spec", "F1", "--smoke"],
                               capture_output=True, text=True, timeout=900)
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            rep = json.loads((reports / "E8_type_well_member_gbdt.json").read_text(
                encoding="utf-8"))
            self.assertIn("oof_total", rep)
            self.assertTrue(Path(rep["oof_path"]).is_file())

    @unittest.skipUnless(HAS_SKLEARN, "sklearn not installed")
    def test_tabular_member_gbdt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tr, va = SC.fold0_wells(n_val=4, n_train=8)
            wells = tr[:8] + va[:4]
            _, _, cache = SC.build_cache(root / "cache", wells, n_rows=60, seed=9)
            reports, runs = root / "reports", root / "runs"
            p = subprocess.run([sys.executable, str(SCRIPTS["tabular"]),
                                "--kind", "gbdt", "--gbdt-kind", "histgb",
                                "--folds", "0", "--max-wells", "4",
                                "--cache-root", str(cache), "--reports-dir", str(reports),
                                "--run-root", str(runs), "--spec", "F1", "--tag", "smoke"],
                               capture_output=True, text=True, timeout=900)
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            rep = json.loads((reports / "E8_tabular_smoke.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["kind"], "gbdt")
            self.assertTrue(Path(rep["oof_path"]).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
