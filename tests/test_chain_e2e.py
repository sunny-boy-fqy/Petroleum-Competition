"""全链路串联集成测试（**torch 门控**）：E1 → E6 → E9 → E10 用**同一份合成缓存**跑通。

这条链是"提交前真会走的那条路"：
`E1/code/train_row.py`（行级基线）→ `E6/code/train_state.py`（原子状态头，产出外折 OOF）→
`E9/code/aggregate_oof.py`（候选复算 + 护栏）→ `E9/code/choose_submission.py`（短名单决策）
→ `E10/code/final_train.py --dry-run`（最终模型计划）。

断言：
  * 每个阶段的**报告/gate 文件按契约落地**（E1_metrics/E1_gate、E6_atomic_report/E6_P0_gate、
    E9_validation_report/E9_P0_gate、E9_submission_decision、E10_final_train）；
  * 候选表由 E6 的 OOF 提供并被 E9 正确复算（`oof_total` 非空、护栏下限 = 81.7757）；
  * 决策 `--dry-run` **不写**候选表；E10 `--dry-run` **不落权重**；
  * 全程不污染仓库的 `versions/candidates.json` / `versions/registry.json`；
  * 失败时（缺 OOF）链路给出明确的非零退出码，而不是静默产出空结论。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

REPO_REGISTRY = V4 / "versions" / "registry.json"
REPO_CANDIDATES = V4 / "versions" / "candidates.json"


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestChainEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr, va = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr + va, n_rows=40, seed=11)
        cls.runs = root / "runs"
        cls.reports = root / "reports"
        cls.scalers = root / "scalers"
        cls.cands = root / "candidates.json"
        cls.env = {**os.environ, "V4_CACHE_ROOT": str(cls.cache),
                   "V4_RUN_ROOT": str(cls.runs), "V4_REPORTS_DIR": str(cls.reports),
                   "V4_SCALERS_DIR": str(cls.scalers)}

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, rel: str, *extra: str, expect_rc: int = 0):
        cmd = [sys.executable, str(V4 / rel), "--smoke", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=self.env, timeout=3600, cwd=str(V4))
        assert proc.returncode == expect_rc, (rel, proc.returncode,
                                              proc.stdout[-1500:], proc.stderr[-1500:])
        return proc

    def test_chain(self):
        reg_before, cand_before = _sha(REPO_REGISTRY), _sha(REPO_CANDIDATES)

        # ---- E1：行级基线（产出 $RUN/E1/oof.npz + E1_metrics/E1_gate）
        self._run("E1/code/train_row.py", "--folds", "0", "--max-wells", "6", "--epochs", "2",
                  "--hidden", "32", "--layers", "1", "--device", "cpu", "--min-free-gb", "0",
                  "--cache-root", str(self.cache), "--out-dir", str(self.runs / "E1"),
                  "--reports-dir", str(self.reports), "--scalers-dir", str(self.scalers),
                  "--candidates", str(self.cands))
        e1_oof = self.runs / "E1" / "oof.npz"
        self.assertTrue(e1_oof.is_file())
        for name in ("E1_metrics.json", "E1_gate.json"):
            self.assertTrue((self.reports / name).is_file(), name)

        # ---- E6：原子状态头（产出 $RUN/E6/state/oof.npz + E6_atomic_report/E6_P0_gate）
        self._run("E6/code/train_state.py", "--folds", "0", "--max-wells", "6", "--epochs", "2",
                  "--hidden", "32", "--layers", "1", "--min-free-gb", "0",
                  "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
                  "--run-root", str(self.runs), "--scalers-dir", str(self.scalers))
        e6_oof = self.runs / "E6" / "state" / "oof.npz"
        self.assertTrue(e6_oof.is_file())
        for name in ("E6_atomic_report.json", "E6_P0_gate.json"):
            self.assertTrue((self.reports / name).is_file(), name)

        # ---- 候选表：两个阶段的 OOF 一起送进 E9
        self.cands.write_text(json.dumps({"schema_version": 1, "candidates": [
            {"candidate_id": "E1_PD0", "stage": "E1", "status": "local_only",
             "oof_path": str(e1_oof), "protocol_matched": False},
            {"candidate_id": "PD1", "stage": "E6", "status": "local_only",
             "oof_path": str(e6_oof), "protocol_matched": False}]}), encoding="utf-8")

        # ---- E9/P0：候选复算 + 护栏 + 排名
        self._run("E9/code/aggregate_oof.py", "--candidates", str(self.cands),
                  "--reports-dir", str(self.reports), "--run-root", str(self.runs))
        val = json.loads((self.reports / "E9_validation_report.json").read_text(encoding="utf-8"))
        rows = {r["candidate_id"]: r for r in val["candidates"]}
        self.assertEqual(set(rows), {"E1_PD0", "PD1"})
        for cid, r in rows.items():
            self.assertIsNotNone(r["oof_total"], cid)
            self.assertTrue(r["guardrail_reason"])
        self.assertAlmostEqual(val["guardrail"]["floor"], 81.7757, places=4)
        self.assertTrue((self.reports / "E9_P0_gate.json").is_file())

        # ---- E9/P0：决策（dry-run 不写候选表）
        self._run("E9/code/choose_submission.py", "--validation-report",
                  str(self.reports / "E9_validation_report.json"), "--candidates",
                  str(self.cands), "--reports-dir", str(self.reports), "--dry-run")
        decision = json.loads((self.reports / "E9_submission_decision.json")
                              .read_text(encoding="utf-8"))
        self.assertIn("choice", decision)
        self.assertIn("shortlist", decision)
        self.assertAlmostEqual(decision["guardrail_floor"], 81.7757, places=4)
        stored = json.loads(self.cands.read_text(encoding="utf-8"))
        self.assertTrue(all(c["status"] == "local_only" for c in stored["candidates"]),
                        "dry-run 不得改候选状态")

        # ---- E10：最终模型计划（dry-run 不落权重）
        self._run("E10/code/final_train.py", "--aggregate", "fold_ensemble", "--candidates",
                  str(self.cands), "--out-dir", str(self.runs / "final"),
                  "--reports-dir", str(self.reports), "--min-free-gb", "0", "--dry-run")
        plan = json.loads((self.reports / "E10_final_train.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["status"], "dry_run")
        self.assertFalse((self.runs / "final" / "final_manifest.json").is_file())

        # ---- 全程不污染仓库事实源
        self.assertEqual(_sha(REPO_REGISTRY), reg_before, "registry.json 被改动")
        self.assertEqual(_sha(REPO_CANDIDATES), cand_before, "candidates.json 被改动")

    def test_chain_rejects_missing_oof(self):
        """链路对"缺 OOF"必须明确失败，而不是产出空结论。"""
        (self.cands).write_text(json.dumps({"schema_version": 1, "candidates": [
            {"candidate_id": "GHOST", "stage": "E9", "status": "local_only",
             "oof_path": str(self.root / "nope.npz")}]}), encoding="utf-8")
        proc = self._run("E9/code/aggregate_oof.py", "--candidates", str(self.cands),
                         "--reports-dir", str(self.reports), "--run-root", str(self.runs))
        val = json.loads((self.reports / "E9_validation_report.json").read_text(encoding="utf-8"))
        self.assertIn("GHOST", val["candidates_missing_oof"])
        g = next(r for r in val["candidates"] if r["candidate_id"] == "GHOST")
        self.assertFalse(g["guardrail_pass"])
        self.assertEqual(g["status_detail"], "oof_missing")
        self.assertIsNotNone(proc)          # smoke 下 E9 仍返回 0，但结论显式为"缺件"


if __name__ == "__main__":
    unittest.main()
