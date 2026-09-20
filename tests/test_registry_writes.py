"""候选注册表写回测试（**口径层，无 torch**）：E9/E10 依赖的候选状态机。

规则来自 `versions/candidates.json::note`：
  * 未登记的候选不得提交；
  * `status ∈ local_only / shortlisted / submitted / frozen_best / rejected`；
  * **一旦 submitted 不得覆盖**，修改必须新建 candidate_id 并写 `parent`；
  * 写盘必须原子（tmp → replace），失败不得留下半个 JSON。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.versioning import registry as R  # noqa: E402

REPO_CANDIDATES = V4 / "versions" / "candidates.json"


class TestCandidateWrites(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "candidates.json"

    def tearDown(self):
        self._td.cleanup()

    def test_missing_file_gives_skeleton_without_writing(self):
        doc = R.load_candidates(self.path)
        self.assertEqual(doc["candidates"], [])
        self.assertFalse(self.path.exists(), "读操作不得创建文件")

    def test_upsert_then_update_preserves_unknown_fields(self):
        R.upsert_candidate({"candidate_id": "v4-e5-por", "stage": "E5",
                            "status": "local_only", "oof_total": 81.5}, self.path)
        R.upsert_candidate({"candidate_id": "v4-e5-por", "status": "shortlisted",
                            "note": "inner-OOF 选中"}, self.path)
        e = R.find_candidate("v4-e5-por", self.path)
        self.assertEqual(e["stage"], "E5")                 # 旧字段保留
        self.assertEqual(e["oof_total"], 81.5)
        self.assertEqual(e["status"], "shortlisted")
        self.assertEqual([h["status"] for h in e["status_history"]],
                         ["local_only", "shortlisted"])
        self.assertIn("registered_at", e)

    def test_upsert_requires_candidate_id(self):
        with self.assertRaises(ValueError):
            R.upsert_candidate({"stage": "E5"}, self.path)

    def test_illegal_status_rejected_on_write(self):
        with self.assertRaises(ValueError):
            R.save_candidates({"candidates": [{"candidate_id": "x", "status": "done"}]},
                              self.path)
        with self.assertRaises(ValueError):
            R.set_candidate_status("x", "done", self.path)

    def test_submitted_is_immutable(self):
        R.upsert_candidate({"candidate_id": "v4-e9-1", "status": "submitted",
                            "a_board_score": 82.3}, self.path)
        with self.assertRaises(PermissionError):
            R.upsert_candidate({"candidate_id": "v4-e9-1", "oof_total": 99.0}, self.path)
        # 显式允许时才可覆盖（仅供人工修复流程）
        R.upsert_candidate({"candidate_id": "v4-e9-1", "oof_total": 99.0}, self.path,
                           allow_submitted_overwrite=True)
        self.assertEqual(R.find_candidate("v4-e9-1", self.path)["oof_total"], 99.0)
        # 新建 candidate_id + parent 是正确路径
        R.upsert_candidate({"candidate_id": "v4-e9-2", "parent": "v4-e9-1",
                            "status": "local_only"}, self.path)
        self.assertEqual(R.find_candidate("v4-e9-2", self.path)["parent"], "v4-e9-1")

    def test_set_status_unknown_candidate_raises(self):
        with self.assertRaises(KeyError):
            R.set_candidate_status("nope", "rejected", self.path)

    def test_freeze_records_sha_and_time(self):
        R.upsert_candidate({"candidate_id": "v4-e10-final", "status": "shortlisted"},
                           self.path)
        R.freeze_candidate("v4-e10-final", self.path, sha256="abc123")
        e = R.find_candidate("v4-e10-final", self.path)
        self.assertEqual(e["status"], "frozen_best")
        self.assertEqual(e["frozen_sha256"], "abc123")
        self.assertIn("frozen_at", e)

    def test_write_is_atomic_and_json_valid(self):
        R.upsert_candidate({"candidate_id": "a", "status": "local_only"}, self.path)
        R.upsert_candidate({"candidate_id": "b", "status": "local_only"}, self.path)
        json.loads(self.path.read_text(encoding="utf-8"))
        self.assertFalse(self.path.with_suffix(".json.tmp").exists(), "残留 tmp 文件")
        self.assertEqual(len(R.candidates(self.path)), 2)

    def test_numpy_scalars_are_serialized(self):
        import numpy as np
        R.upsert_candidate({"candidate_id": "c", "status": "local_only",
                            "oof_total": np.float64(81.25),
                            "folds": [np.int64(0), np.int64(3)]}, self.path)
        e = R.find_candidate("c", self.path)
        self.assertEqual(e["oof_total"], 81.25)
        self.assertEqual(e["folds"], [0, 3])

    def test_repo_candidates_untouched_by_tests(self):
        """测试必须写 tmp；仓库候选注册表在任何写回路径下都不能被本文件的用例改动。"""
        before = REPO_CANDIDATES.read_text(encoding="utf-8") if REPO_CANDIDATES.is_file() else None
        R.upsert_candidate({"candidate_id": "tmp-only", "status": "local_only"}, self.path)
        after = REPO_CANDIDATES.read_text(encoding="utf-8") if REPO_CANDIDATES.is_file() else None
        self.assertEqual(before, after)


class TestRepoCandidatesSchema(unittest.TestCase):
    def test_repo_file_matches_helper_contract(self):
        if not REPO_CANDIDATES.is_file():
            self.skipTest("仓库暂无 candidates.json")
        doc = json.loads(REPO_CANDIDATES.read_text(encoding="utf-8"))
        self.assertIn("fields", doc)
        for e in doc.get("candidates", []):
            self.assertIn(e.get("status"), R.CANDIDATE_STATUSES)
            self.assertTrue(e.get("candidate_id"))
        self.assertEqual(R.candidates(), doc.get("candidates", []))


if __name__ == "__main__":
    unittest.main()
