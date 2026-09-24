"""E2 -> 下游的特征版本自动选择与 run_train 接线测试。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
TOOL = V4 / "tools" / "select_feature_spec.py"


def _run(*args: str):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, cwd=str(V4))


class TestSelectFeatureSpec(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.reports = Path(self._td.name)
        (self.reports / "E2_ablation.json").write_text(json.dumps({
            "baseline": {"spec_key": "F1", "oof_total": 78.0},
            "groups": [
                {"spec_key": "FX_phys", "groups": ["F1", "phys"], "oof_total": 78.1,
                 "delta_vs_f1": 0.1, "paired_ci": [-0.1, 0.3], "decision": "no_go"},
                {"spec_key": "FX_win_w11-51-201_smsmmtc", "groups": ["F1", "win"],
                 "oof_total": 79.4, "delta_vs_f1": 1.4, "paired_ci": [0.8, 1.9],
                 "decision": "adopted"},
                {"spec_key": "FX_well_wsmsppp", "groups": ["F1", "well"],
                 "oof_total": 76.9, "delta_vs_f1": -1.1, "paired_ci": [-2.0, -0.2],
                 "decision": "no_go"},
            ],
        }, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def test_selects_best_adopted_spec(self):
        r = _run("--reports-dir", str(self.reports), "--print")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip(), "F1+win")
        doc = json.loads((self.reports / "E2_best_spec.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["selected_spec"], "F1+win")
        self.assertEqual(doc["selected_key"], "FX_win_w11-51-201_smsmmtc")

    def test_falls_back_to_f1_when_no_adopted(self):
        (self.reports / "E2_ablation.json").write_text(json.dumps({
            "baseline": {"spec_key": "F1", "oof_total": 78.0},
            "groups": [
                {"spec_key": "FX_phys", "groups": ["F1", "phys"], "oof_total": 78.1,
                 "delta_vs_f1": 0.1, "paired_ci": [-0.1, 0.3], "decision": "no_go"},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        r = _run("--reports-dir", str(self.reports), "--print")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip(), "F1")

    def test_print_key(self):
        r = _run("--reports-dir", str(self.reports), "--print-key")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip(), "FX_win_w11-51-201_smsmmtc")


class TestRunTrainAutoSpecWiring(unittest.TestCase):
    def test_run_train_has_auto_spec(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        for token in (
            "select_feature_spec.py",
            "resolve_feature_spec",
            "--spec \"$FEATURE_SPEC\"",
            "E2_work/oof_${e3_row_key}.npz",
        ):
            self.assertIn(token, src)

    def test_spec_is_stamped_in_fold_cache_and_checkpoint(self):
        e3 = (V4 / "E3" / "code" / "train_seq.py").read_text(encoding="utf-8")
        self.assertIn("_resume_stamp", e3)
        self.assertIn("feature_spec", e3)
        seq = (V4 / "src" / "training" / "seq_loop.py").read_text(encoding="utf-8")
        self.assertIn("man.get(\"feature_spec\") == want_spec", seq)


if __name__ == "__main__":
    unittest.main(verbosity=2)
