"""E9/P0 测试（**口径层，无 torch**）：候选汇总、护栏下限与排名。

本地护栏下限必须是 **constants 里冻结锚点算出来的** `max(75.0, 80.382479−1.0)=79.382479`，
不允许在脚本里硬编码数字——测试直接比对 `constants.py` 的值；A 榜锚点单独校验。

断言要点：
  * `oof_total` 由 OOF **可复算**（同一份 OOF 重算结果一致），并按它排名；
  * 缺 OOF 的候选 → `guardrail_pass=False` + 记入 `missing_oof`，**不静默忽略**；
  * A 榜未知时只评估本地那一半并**如实标注**（不假装通过 A 榜）；
  * A 榜已知且低于 `82.2757−0.5` → 即使本地 OOF 达标也**不算通过**；
  * 短名单只含通过护栏的候选，且不超过 `--max-shortlist`；
  * 本脚本只读候选表（不写 status），`--smoke` 不动仓库文件。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402

SCRIPT = V4 / "E9" / "code" / "aggregate_oof.py"
REPO_CANDIDATES = V4 / "versions" / "candidates.json"
N_ROWS, N_WELLS = 60, 6


def _fixture(root: Path) -> tuple[Path, Path]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "training_time_log.json").write_text(
        json.dumps({"folds": [{"fold": 0, "seconds": 1.0}]}), encoding="utf-8")
    rng = np.random.RandomState(0)
    y = np.column_stack([10 + 4 * rng.rand(N_ROWS), 40 + 20 * rng.rand(N_ROWS),
                         60 + 20 * rng.rand(N_ROWS)])
    wi = np.repeat(np.arange(N_WELLS), N_ROWS // N_WELLS)
    fo = np.repeat(np.arange(N_WELLS) % 2, N_ROWS // N_WELLS)

    def save(name: str, pred: np.ndarray) -> None:
        np.savez_compressed(root / f"{name}.npz", y_true=y,
                            mask=np.ones((N_ROWS, 3)), gated=pred, cont=pred,
                            well_index=wi, fold_of_row=fo,
                            well_ids=np.asarray([f"w{i}" for i in range(N_WELLS)],
                                                dtype=object))
    save("good", y.copy())                       # total ≈ 100
    save("mid", y + 0.05)                        # 略差但远高于护栏（用于排名/短名单）
    save("const", np.tile([0.1, 0.01, 99.9], (N_ROWS, 1)))
    cands = {"schema_version": 1, "candidates": [
        {"candidate_id": "PD1", "stage": "E6", "status": "local_only",
         "oof_path": str(root / "good.npz"), "protocol_matched": False},
        {"candidate_id": "E6b", "stage": "E6", "status": "local_only",
         "oof_path": str(root / "mid.npz"), "a_board_score": 84.0},
        {"candidate_id": "E1c", "stage": "E1", "status": "local_only",
         "oof_path": str(root / "const.npz")},
        {"candidate_id": "GHOST", "stage": "E9", "status": "local_only",
         "oof_path": str(root / "nope.npz")},
    ]}
    cand_path = root / "candidates.json"
    cand_path.write_text(json.dumps(cands), encoding="utf-8")
    return cand_path, reports


def _run(root: Path, cand: Path, reports: Path, *extra: str, expect_rc: int = 0) -> dict:
    cmd = [sys.executable, str(SCRIPT), "--candidates", str(cand),
           "--reports-dir", str(reports), "--run-root", str(root / "runs"), "--smoke", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                          proc.stderr[-1500:])
    rep = reports / "E9_validation_report.json"
    return json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}


class TestE9Aggregate(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.cand, self.reports = _fixture(self.root)

    def tearDown(self):
        self._td.cleanup()

    def test_guardrail_floor_from_frozen_anchors(self):
        rep = _run(self.root, self.cand, self.reports)
        expect = max(C.PASS_LINE, C.B0_LOCAL_OOF - C.GUARDRAIL_TOLERANCE)
        self.assertAlmostEqual(rep["guardrail"]["floor"], expect, places=9)
        self.assertAlmostEqual(rep["guardrail"]["floor"], 79.382479, places=4)
        self.assertAlmostEqual(rep["guardrail"]["a_board_floor"],
                               C.B0_A_BOARD - C.GUARDRAIL_ABOARD_MARGIN, places=9)
        self.assertFalse(rep["protocol_matched"])
        self.assertIn("protocol_matched=false", rep["protocol_note"])

    def test_recomputable_and_ranked(self):
        rep = _run(self.root, self.cand, self.reports)
        rows = {r["candidate_id"]: r for r in rep["candidates"]}
        self.assertGreater(rows["PD1"]["oof_total"], 99.0)
        self.assertTrue(rows["PD1"]["guardrail_pass"])
        self.assertTrue(rows["E6b"]["guardrail_pass"], rows["E6b"])
        self.assertFalse(rows["E1c"]["guardrail_pass"])
        self.assertEqual(rep["ranked_by_oof"][0], "PD1")
        self.assertEqual(rep["shortlist"], ["PD1", "E6b"])
        for r in rep["candidates"]:
            if r["candidate_id"] != "GHOST":
                self.assertIsNotNone(r["oof_total"])
                self.assertTrue(r["guardrail_reason"])

    def test_missing_oof_recorded_not_ignored(self):
        rep = _run(self.root, self.cand, self.reports)
        self.assertIn("GHOST", rep["candidates_missing_oof"])
        g = next(r for r in rep["candidates"] if r["candidate_id"] == "GHOST")
        self.assertFalse(g["guardrail_pass"])
        self.assertEqual(g["status_detail"], "oof_missing")

    def test_a_board_unknown_is_labelled_not_faked(self):
        rep = _run(self.root, self.cand, self.reports)
        pd1 = next(r for r in rep["candidates"] if r["candidate_id"] == "PD1")
        self.assertIsNone(pd1["a_board_score"])
        self.assertIsNone(pd1["a_board_pass"])
        self.assertIn("A 榜未知", pd1["guardrail_reason"])

    def test_a_board_below_floor_blocks_even_if_local_passes(self):
        cand = json.loads(self.cand.read_text(encoding="utf-8"))
        cand["candidates"][0]["a_board_score"] = 80.0        # < A 榜护栏 81.7757
        p = self.root / "cand_low_a.json"
        p.write_text(json.dumps(cand), encoding="utf-8")
        rep = _run(self.root, p, self.reports)
        pd1 = next(r for r in rep["candidates"] if r["candidate_id"] == "PD1")
        self.assertFalse(pd1["a_board_pass"])
        self.assertFalse(pd1["guardrail_pass"])
        self.assertIn("A 榜 80.0000 <", pd1["guardrail_reason"])

    def test_shortlist_respects_max(self):
        rep = _run(self.root, self.cand, self.reports, "--max-shortlist", "1")
        self.assertEqual(len(rep["shortlist"]), 1)
        self.assertEqual(rep["shortlist"], ["PD1"])

    def test_smoke_does_not_touch_repo_candidates(self):
        before = (REPO_CANDIDATES.read_text(encoding="utf-8")
                  if REPO_CANDIDATES.is_file() else None)
        _run(self.root, self.cand, self.reports)
        after = (REPO_CANDIDATES.read_text(encoding="utf-8")
                 if REPO_CANDIDATES.is_file() else None)
        self.assertEqual(before, after)
        stored = json.loads(self.cand.read_text(encoding="utf-8"))
        self.assertTrue(all(c["status"] == "local_only" for c in stored["candidates"]),
                        "汇总脚本不得改候选状态（那是 choose_submission.py 的职责）")

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--candidates", "--bootstrap-iters", "--max-shortlist", "--json",
                     "--folds", "--seed"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
