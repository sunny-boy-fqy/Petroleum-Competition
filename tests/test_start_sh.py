"""start.sh 云端入口的 dry-run 契约测试（不执行训练）。"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
START = V4 / "start.sh"


def run_start(*args):
    return subprocess.run(["bash", str(START), "--dry-run", *args],
                          capture_output=True, text=True, timeout=60)


class TestStartSh(unittest.TestCase):
    def test_help_and_list(self):
        for args in (["--help"], ["--list"]):
            p = subprocess.run(["bash", str(START), *args],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(p.stdout.strip())

    def test_to_e3_maps_to_task7(self):
        p = run_start("--to", "E3")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 7", p.stdout)

    def test_to_e3_main_maps_to_task6(self):
        p = run_start("--to", "E3-main")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 6", p.stdout)

    def test_single_stage_adds_prereq(self):
        p = run_start("--stage", "E3", "--phase", "main", "--folds", "0,1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 5", p.stdout)
        self.assertIn("--mode stage --stage E3 --phase main", p.stdout)
        self.assertIn("--folds 0\\,1", p.stdout)

    def test_e8_stage_default_target_all(self):
        p = run_start("--stage", "E8")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 11", p.stdout)
        self.assertIn("run_train.sh --mode stage --stage E8 --target all", p.stdout)

    def test_wp_atom_decision_prereq_e6(self):
        p = run_start("--wp", "atom-decision")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 10", p.stdout)
        self.assertIn("fit_atom_decision.py", p.stdout)

    def test_wp_gbdt_prereq_e2(self):
        p = run_start("--wp", "gbdt")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 5", p.stdout)
        self.assertIn("tabular_member.py --kind gbdt", p.stdout)

    def test_wp_type_well_adapt(self):
        p = run_start("--wp", "type-well-adapt")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 3", p.stdout)
        self.assertIn("type_well_member.py", p.stdout)

    def test_wp_data_quality_and_petro(self):
        for wp, script in (("data-quality", "report_data_quality.py"),
                           ("petro", "report_petro.py")):
            p = run_start("--wp", wp)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("run_train.sh --mode all --through 3", p.stdout)
            self.assertIn(script, p.stdout)

    def test_wp_self_training_and_ssl_module_checks(self):
        for wp, test in (("self-training", "test_self_training"), ("ssl", "test_ssl")):
            p = run_start("--wp", wp)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn(f"tests/run_all.py {test}", p.stdout)

    def test_stage_e5_prereq_e4(self):
        p = run_start("--stage", "E5")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("run_train.sh --mode all --through 8", p.stdout)
        self.assertIn("run_train.sh --mode stage --stage E5 --target all", p.stdout)

    def test_invalid_stage_fails(self):
        p = subprocess.run(["bash", str(START), "--stage", "NOPE"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2)
        self.assertIn("未知 --stage", p.stderr + p.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
