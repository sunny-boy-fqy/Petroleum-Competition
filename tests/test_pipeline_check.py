"""流水线完整性检查测试（**口径层，无 torch**）：清单缺失必须能被查出来。

`tools/check_pipeline.py` 的价值在于"文档说做过、仓库里必须有对应物"，因此这里：
  * 在真实仓库上跑 → 0 问题（E1–E10 全部脚本/接线/报告命名齐备）；
  * 在一个**按 checker 自己的清单生成**的最小仓库上跑 → 通过；
  * 删掉任一阶段脚本 → 报"缺少阶段脚本"；删掉 `run_train.sh` 的某个分支 → 报接线问题；
    删掉报告名 → 报命名不一致。三种失败都必须让退出码为 1。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

CHECKER = V4 / "tools" / "check_pipeline.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("v4_check_pipeline", str(CHECKER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPipelineCheckOnRepo(unittest.TestCase):
    def test_real_repo_is_complete(self):
        out = subprocess.run([sys.executable, str(CHECKER), "--json",
                              "/tmp/v4_pipeline_check.json"], capture_output=True,
                             text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stdout[-1500:])
        res = json.loads(Path("/tmp/v4_pipeline_check.json").read_text(encoding="utf-8"))
        self.assertTrue(res["ok"], res["problems"])
        self.assertEqual(res["n_problems"], 0)
        self.assertGreaterEqual(res["n_stages"], 11)
        self.assertGreaterEqual(res["n_scripts"], 30)

    def test_stage_wiring_in_run_train(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        for i in range(1, 11):
            self.assertIn(f"E{i})", src, f"run_train.sh 缺少 E{i} 分支")


class TestPipelineCheckDetectsGaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_checker()

    def _minimal_repo(self, root: Path) -> None:
        """按 checker 自己的清单生成一个最小但完整的仓库。"""
        for stage, scripts in self.mod.STAGE_SCRIPTS.items():
            d = root / stage
            (d / "PLAN.md").parent.mkdir(parents=True, exist_ok=True)
            (d / "PLAN.md").write_text("> " + "x" * 300, encoding="utf-8")
            for rel in scripts:
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                body = "import argparse  # 可运行入口\n"
                for tok in self.mod.REPORT_TOKENS.get(stage, ()):
                    body += f"# {tok}\n"
                p.write_text(body, encoding="utf-8")
        (root / "run_train.sh").write_text(
            "\n".join(f"    E{i}) true ;;" for i in range(1, 11)), encoding="utf-8")
        (root / "train.py").write_text('STAGES = ["E1", "E3", "E4", "E5", "E6", "E8", "E10"]\n',
                                       encoding="utf-8")
        (root / "predict.py").write_text("# PD1\n", encoding="utf-8")
        (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        (root / "configs").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "v4.yaml").write_text("pipeline: v4\n", encoding="utf-8")

    def test_minimal_repo_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._minimal_repo(root)
            res = self.mod.run(root)
            self.assertTrue(res["ok"], res["problems"])

    def test_missing_script_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._minimal_repo(root)
            (root / "E6" / "code" / "search_tau.py").unlink()
            res = self.mod.run(root)
            self.assertFalse(res["ok"])
            self.assertTrue(any("缺少阶段脚本" in p and "search_tau.py" in p
                                for p in res["problems"]), res["problems"])

    def test_missing_branch_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._minimal_repo(root)
            (root / "run_train.sh").write_text("E1) true ;;\n", encoding="utf-8")
            res = self.mod.run(root)
            self.assertFalse(res["ok"])
            self.assertTrue(any("E5 分支" in p for p in res["problems"]), res["problems"])

    def test_report_naming_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._minimal_repo(root)
            for f in sorted((root / "E9" / "code").glob("*.py")):
                f.write_text("import argparse  # 报告名被抹掉\n", encoding="utf-8")
            (root / "E9" / "PLAN.md").write_text("> " + "x" * 300, encoding="utf-8")
            res = self.mod.run(root)
            self.assertFalse(res["ok"])
            self.assertTrue(any("E9_validation_report.json" in p for p in res["problems"]),
                            res["problems"])

    def test_cli_exit_code_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._minimal_repo(root)
            (root / "predict.py").unlink()
            out = subprocess.run([sys.executable, str(CHECKER), "--repo", str(root)],
                                 capture_output=True, text=True, timeout=300)
            self.assertEqual(out.returncode, 1)
            self.assertIn("predict.py", out.stdout)


if __name__ == "__main__":
    unittest.main()
