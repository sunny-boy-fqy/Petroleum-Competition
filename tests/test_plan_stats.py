#!/usr/bin/env python3
"""`tools/plan_stats.py` 的计划行数一致性测试（R3-C2 / R3-M3）。

三审发现：旧版 `--check` 只把文档里出现的**所有** `N 行` 数字收集起来，判断实测值是否在其中。
于是 `PLAN.md` 里「阶段 711 / P 4,982」是错的，却因为文档别处出现了正确的「合计 6,425 行」
而被完全漏检；`sync_plan_stats.py` 的正则也因为要求列尾紧接 `|` 而静默 `pattern not found`。

本测试锁定：
  * 当前仓库的所有行数声明（含**阶段/P 分项**）都与实测一致；
  * 一旦把某一个**分项**改错，检查器必须报错 —— 这是 R3-C2 的回归测试；
  * 校验用的 `_claims()` 与同步用的正则是同一份（防止再次漂移）。
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from tools import plan_stats as PS  # noqa: E402


class TestPlanStatsMeasurement(unittest.TestCase):
    def test_counts_are_structured(self):
        p = PS.measure()
        self.assertEqual(p["main"]["files"], 1)
        self.assertEqual(p["stage"]["files"], 12)
        self.assertEqual(p["p_level"]["files"], 33)
        self.assertEqual(p["total_plan_files"], 46)
        self.assertEqual(
            p["total_lines"],
            p["main"]["lines"] + p["stage"]["lines"] + p["p_level"]["lines"])

    def test_every_claim_pattern_matches_the_repo_docs(self):
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        missing = []
        for c in PS._claims():
            text = (V4 / c.path).read_text(encoding="utf-8")
            if not re.search(c.pattern, text):
                missing.append(f"{c.path}: {c.pattern[:60]}")
        self.assertEqual(missing, [], f"行数声明正则未命中（文档被改写？）：{missing}")

    def test_docs_are_consistent_with_measurements(self):
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        errs = PS.check_docs(p)
        self.assertEqual(errs, [], "文档行数声明与实测不一致（运行 tools/sync_plan_stats.py）")


class TestPlanStatsCatchesSubtotalDrift(unittest.TestCase):
    """R3-C2 回归：**分项**改错必须被抓到，而不能被总量掩盖。"""

    def test_stage_subtotal_drift_is_detected(self):
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        p["stage"]["lines"] += 13          # 模拟历史上的「阶段 711 vs 708」
        errs = PS.check_docs(p, ["PLAN.md"])
        self.assertTrue(any("PLAN.md" in e and "stage.lines" in e for e in errs), errs)

    def test_p_level_subtotal_drift_is_detected(self):
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        p["p_level"]["lines"] -= 6         # 模拟历史上的「P 4,982 vs 4,988」
        errs = PS.check_docs(p, ["PLAN.md"])
        self.assertTrue(any("p_level.lines" in e for e in errs), errs)

    def test_p_level_avg_drift_is_detected(self):
        """平均篇幅也是声明的一部分，必须一起校验。"""
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        p["p_level"]["avg"] += 1
        errs = PS.check_docs(p, ["PLAN.md", "docs/PROJECT_FILES.md"])
        self.assertTrue(any("p_level.avg" in e for e in errs), errs)

    def test_total_drift_is_detected(self):
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        p["total_lines"] += 1
        errs = PS.check_docs(p)
        self.assertTrue(any("total_lines" in e for e in errs), errs)

    def test_gate_count_drift_is_detected(self):
        """R3-C3：README 的 `N/M` Gate 计数必须与实物报告一致。"""
        p = PS.measure()
        p["gate"] = {"mandatory_passed": 10, "mandatory_total": 10}   # 旧口径
        errs = PS.check_docs(p, ["README.md"])
        self.assertTrue(any("gate.mandatory_total" in e for e in errs), errs)

    def test_status_summary_drift_is_detected(self):
        p = PS.measure(); p["total_lines"] += 1
        self.assertTrue(PS.check_status_summary(p))

    def test_missing_file_is_an_error_not_a_silent_pass(self):
        """请求校验一个不存在/未登记的文件时不得静默返回「通过」。"""
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        errs = PS.check_docs(p, ["docs/NOT_A_FILE.md"])
        self.assertTrue(errs, "不存在的文档不得被静默跳过")
        self.assertTrue(any("NOT_A_FILE" in e for e in errs), errs)
        # 已登记但文件真的不存在时，报「文件缺失」
        errs2 = PS.check_docs(p, ["PLAN.md"])
        self.assertFalse(errs2, errs2)


class TestSyncSharesTheCheckPatterns(unittest.TestCase):
    """同步与校验必须用**同一份**正则，否则会出现「同步静默不生效」。"""

    def test_sync_imports_claims_from_plan_stats(self):
        src = (V4 / "tools" / "sync_plan_stats.py").read_text(encoding="utf-8")
        self.assertIn("from tools.plan_stats import", src)
        self.assertIn("_claims", src)
        # 不允许同步工具自己再抄一份行数正则
        self.assertNotIn(r"合计 [\d,]+", src)

    def test_expected_claim_paths(self):
        paths = {c.path for c in PS._claims()}
        self.assertEqual(
            paths, {"PLAN.md", "README.md", "docs/PROJECT_FILES.md",
                    "E0/docs/data_card.md", "E0/PLAN.md"},
            "行数/Gate 计数声明只应出现在这些文档里；新增一处必须同步登记声明正则")


class TestPlanStatsEvidenceJson(unittest.TestCase):
    """R4-H2：`reports/E0_plan_stats.json` 是证据副本，必须与 `measure()` 逐字段相等。

    四审发现它停在旧值（730/708/4988/6426），而 PLAN/README/status 都已同步到 6781；
    两个检查器都不查它，于是它成了"已提交但过期"的手写副本。
    """

    def test_evidence_matches_measurement(self):
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        self.assertEqual(PS.check_evidence(p), [])

    def test_evidence_file_exists(self):
        self.assertTrue(V4.joinpath(*PS.EVIDENCE_JSON_RELPATH).is_file())

    def test_stale_evidence_is_detected(self):
        """过期副本必须被检出 —— 但**不得**覆写已提交的文件（review R7-7）。

        原实现把 `reports/E0_plan_stats.json` 就地改成过期值、再在 finally 里还原：
        崩溃/断电会污染仓库文件，并行跑其它检查器时还有瞬时不一致窗口。
        现在写进临时目录，用 `check_evidence(payload, path=...)` 校验。
        """
        import json
        import tempfile
        committed = V4.joinpath(*PS.EVIDENCE_JSON_RELPATH).read_text(encoding="utf-8")
        p = PS.measure(); p["gate"] = PS._local_gate_counts()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "E0_plan_stats.json"
            d = json.loads(committed)
            d["total_lines"] = 1                       # 模拟四审看到的过期副本
            d["stage"]["lines"] = 708
            tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            errs = PS.check_evidence(p, path=tmp)
            self.assertTrue(any("total_lines" in e for e in errs), errs)
            self.assertTrue(any("stage.lines" in e for e in errs), errs)
            # 同一 payload 对**真实**证据文件必须无错（且文件始终未被改动）
            self.assertEqual(PS.check_evidence(p), [])
            self.assertEqual(V4.joinpath(*PS.EVIDENCE_JSON_RELPATH).read_text(encoding="utf-8"),
                             committed, "本测试不得修改已提交的证据文件")
        self.assertEqual(PS.check_evidence(p), [])

    def test_sync_writes_the_evidence_json(self):
        """同步工具必须**同时**刷新证据 JSON，否则它会再次过期。"""
        src = (V4 / "tools" / "sync_plan_stats.py").read_text(encoding="utf-8")
        self.assertIn("E0_plan_stats.json", src)

    def test_check_status_validates_evidence(self):
        src = (V4 / "tools" / "check_status.py").read_text(encoding="utf-8")
        self.assertIn("check_evidence", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
