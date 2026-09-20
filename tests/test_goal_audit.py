"""目标达成度审计测试（**口径层，无 torch**）：本地交付完整性必须可复算且不回退。

`tools/audit_goal.py` 把"目标是否达成"变成一条可重复执行的检查：
  * E0–E10 每个阶段都必须有：入口脚本、对应测试文件（且含 `def test_`）、报告命名、
    `run_train.sh` 接线、文档条目；
  * 横向模块（推理/训练入口、PD1 注册、集成融合、序列主干、冻结骨干、两阶段件、解码层、EMA）
    必须存在；
  * **云端待办必须显式列出**（不能被本地通过掩盖）；
  * 对缺测试/缺文档的仓库必须判 `local_complete=false` 且退出码 1。
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

AUDIT = V4 / "tools" / "audit_goal.py"


def _load():
    spec = importlib.util.spec_from_file_location("v4_audit_goal", str(AUDIT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGoalAudit(unittest.TestCase):
    def test_repo_is_locally_complete(self):
        out = subprocess.run([sys.executable, str(AUDIT), "--json", "/tmp/goal_audit_t.json"],
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:])
        res = json.loads(Path("/tmp/goal_audit_t.json").read_text(encoding="utf-8"))
        self.assertTrue(res["local_complete"], res["pipeline_check"]["problems"])
        for stage, info in res["stages"].items():
            self.assertTrue(info["ok"], f"{stage}: {info}")
            self.assertGreater(info["n_tests"], 0, stage)
            self.assertTrue(info["wiring"], stage)
            self.assertTrue(info["documented"], stage)
        for name, info in res["cross_cutting"].items():
            self.assertTrue(info["ok"], f"{name}: {info['missing']}")
        self.assertGreaterEqual(res["test_suite"]["n_test_methods"], 700)
        self.assertTrue(res["cloud_pending"], "云端待办必须显式列出")

    def test_cloud_pending_covers_every_stage(self):
        res = _load().audit(V4)
        pending = " ".join(res["cloud_pending"])
        for stage in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"):
            self.assertIn(stage, pending, f"云端待办缺少 {stage}")

    def test_empty_repo_fails_audit(self):
        with tempfile.TemporaryDirectory() as td:
            res = _load().audit(Path(td))
            self.assertFalse(res["local_complete"])
            self.assertTrue(any(not v["ok"] for v in res["stages"].values()))

    def test_missing_tests_detected(self):
        """把 tests 目录换成空的 → 阶段必须判 GAP（防止靠文档冒充测试）。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod = _load()
            # 复制最小仓库骨架（复用 pipeline checker 的清单生成器）
            cp = mod._load_pipeline_mod()
            for stage, scripts in cp.STAGE_SCRIPTS.items():
                (root / stage).mkdir(parents=True, exist_ok=True)
                (root / stage / "PLAN.md").write_text("> " + "x" * 300, encoding="utf-8")
                for rel in scripts:
                    p = root / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    body = "import argparse\n" + "".join(
                        f"# {t}\n" for t in cp.REPORT_TOKENS.get(stage, ()))
                    p.write_text(body, encoding="utf-8")
            (root / "run_train.sh").write_text(
                "\n".join(f"E{i}) true ;;" for i in range(1, 11)), encoding="utf-8")
            for rel, body in (("train.py", "E1 E3 E4 E5 E6 E8 E10"),
                              ("predict.py", "PD1"), ("requirements.txt", "numpy")):
                (root / rel).write_text(body, encoding="utf-8")
            (root / "configs").mkdir(exist_ok=True)
            (root / "configs" / "v4.yaml").write_text("pipeline: v4\n", encoding="utf-8")
            (root / "docs").mkdir(exist_ok=True)
            (root / "docs" / "PROJECT_FILES.md").write_text(
                " ".join(cp.STAGE_SCRIPTS), encoding="utf-8")
            res = mod.audit(root)
            self.assertFalse(res["local_complete"])
            self.assertIn("E1", res["stages"])
            self.assertFalse(res["stages"]["E1"]["ok"])


if __name__ == "__main__":
    unittest.main()
