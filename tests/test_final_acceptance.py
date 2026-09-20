"""最终验收批测试（**口径层，无 torch**）：验收脚本本身必须可用且 `--quick` 秒级通过。

`tools/final_acceptance.py` 是"本地证据总入口"，因此这里只做轻量守护：
  * `--quick` 必须通过（checker + 契约复算 + 参考数据 + 计划统计）；
  * 汇总 JSON 必须包含各 checker 的退出码与 `cloud_pending` 清单；
  * 解析函数能正确读懂 `run_all.py` 的总结行（避免"测试失败却被解析成通过"）。
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

SCRIPT = V4 / "tools" / "final_acceptance.py"


def _load():
    spec = importlib.util.spec_from_file_location("v4_final_acceptance", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFinalAcceptance(unittest.TestCase):
    def test_quick_mode_passes(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "acc.json"
            proc = subprocess.run([sys.executable, str(SCRIPT), "--quick", "--json", str(out)],
                                  capture_output=True, text=True, timeout=1800, cwd=str(V4))
            self.assertEqual(proc.returncode, 0, proc.stdout[-1500:] + proc.stderr[-800:])
            res = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(res["ok"])
        for key in ("check_consistency", "check_status", "check_pipeline", "audit_goal",
                    "contract_recompute", "verify_reference", "plan_stats"):
            self.assertIn(key, res["checks"])
            self.assertEqual(res["checks"][key]["returncode"], 0, key)
        self.assertTrue(res["cloud_pending"])
        self.assertIn("云端", res["note"])

    def test_suite_summary_parser(self):
        mod = _load()
        ok = mod.parse_suite("...\n运行 730 项，失败 0，错误 0，跳过 202\n跳过模块：[]")
        self.assertEqual(ok, {"parsed": True, "ran": 730, "failed": 0, "errors": 0,
                              "skipped": 202})
        bad = mod.parse_suite("运行 730 项，失败 2，错误 1，跳过 0")
        self.assertEqual(bad["failed"], 2)
        self.assertEqual(bad["errors"], 1)
        self.assertFalse(mod.parse_suite("no summary here")["parsed"])

    def test_interpreters_discovered(self):
        mod = _load()
        found = mod.interpreters()
        self.assertTrue(found)
        self.assertTrue(all(p.is_file() for _, p in found))
        self.assertGreaterEqual(len({str(p) for _, p in found}), 1)


if __name__ == "__main__":
    unittest.main()
