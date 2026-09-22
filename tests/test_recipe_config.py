"""E7 冻结配方读取/应用（审查 H4/M1 回归；不需要 torch）。

覆盖：
  * `$V4_REPORTS_DIR/loss_v1.json` 能被读到（多任务持久化路径）；
  * 显式路径不存在 / JSON 损坏 / 非 object -> 必须显式报错，不得静默回退；
  * 未知键必须能被告警列出（不静默吞掉）。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

V4 = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(V4))

from src.training import recipe as R  # noqa: E402


class TestRecipeConfigResolution(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.reports = self.root / "reports"
        self.reports.mkdir()

    def tearDown(self):
        self._td.cleanup()

    def test_reports_dir_is_preferred(self):
        cfg = {"loss": {"lam1": 0.7, "lam3": 0.02}}
        (self.reports / "loss_v1.json").write_text(json.dumps(cfg), encoding="utf-8")
        with mock.patch.dict(os.environ, {"V4_REPORTS_DIR": str(self.reports)}, clear=False):
            got = R.load_loss_recipe()
        self.assertEqual(got, cfg)

    def test_explicit_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            R.load_loss_recipe(self.root / "nope.json")

    def test_malformed_json_raises(self):
        bad = self.root / "loss_v1.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            R.load_loss_recipe(bad)

    def test_non_object_raises(self):
        bad = self.root / "loss_v1.json"
        bad.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(ValueError):
            R.load_loss_recipe(bad)

    def test_ignored_keys_are_reported(self):
        self.assertEqual(R.ignored_loss_keys({"loss": {"lam1": 1.0, "bogus": 2}}),
                         ["bogus"])
        self.assertEqual(R.ignored_loss_keys(None), [])

    def test_apply_loss_recipe_sets_cfg_and_warns(self):
        from src.training import loop as L
        cfg = L.TrainConfig()
        params = R.apply_loss_recipe(
            cfg, {"loss": {"lam1": 0.5, "lam1_schedule": "cosine", "lam3": 0.02,
                           "bogus": 9}})
        self.assertEqual(params["lam1"], 0.5)
        self.assertEqual(cfg.lam1_start, 0.5)
        self.assertEqual(cfg.lam1_schedule, "cosine")
        self.assertEqual(cfg.lam_phys, 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
