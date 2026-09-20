"""E8/P2 集成驱动测试（**口径层，无 torch**）：同源性 / inner-OOF 选权 / 配对 CI / 采纳判定。

`E8/code/ensemble.py` 只读成员 OOF（`npz`），所以这里合成成员把每条纪律钉住：
  * **同源必须被标注**：两个完全相同的成员 → `same_source_pairs` 命中，且融合的 delta≈0
    时判 `no_go`（同源平均不得包装成增益）；
  * **互补成员才可能被采纳**：`A = y+e`、`B = y−e`（误差反相关）→ 等权平均 = 真值，
    delta 明显为正且 CI 下界 > 0 → `adopted`；
  * **权重只在 inner-OOF 选**：给了 `--inner-members` 才允许 `weight_source=inner_oof_simplex`；
    没给就退化为等权并写明 warning（绝不假装"选过"）；
  * **CI 含 0 一律 NO-GO**；`--smoke` 不写仓库候选表；
  * EMA/SWA/top-k 快照三类臂的到位情况必须显式记录（缺就写 `not_provided`）。
"""
from __future__ import annotations

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

SCRIPT = V4 / "E8" / "code" / "ensemble.py"
N_WELLS, ROWS_PER_WELL = 6, 20
N_ROWS = N_WELLS * ROWS_PER_WELL
REPO_CANDIDATES = V4 / "versions" / "candidates.json"


def _base(seed: int = 0):
    rng = np.random.RandomState(seed)
    y = np.column_stack([10 + 4 * rng.rand(N_ROWS), 40 + 20 * rng.rand(N_ROWS),
                         60 + 20 * rng.rand(N_ROWS)])
    return y, np.ones((N_ROWS, 3), dtype="float64"), \
        np.repeat(np.arange(N_WELLS), ROWS_PER_WELL)


def _save(path: Path, y, cont, mask, wi, tau=0.5):
    np.savez_compressed(path, cont=cont, q_atom=np.full((N_ROWS, 3), tau),
                        y_true=y, mask=mask, well_index=wi,
                        tau_row=np.full((N_ROWS, 3), tau),
                        well_ids=np.asarray([f"w{i}" for i in range(N_WELLS)], dtype=object))


def _complementary(root: Path):
    """A = y+e、B = y−e（反相关误差）→ 等权平均 = 真值。"""
    y, mask, wi = _base()
    rng = np.random.RandomState(1)
    e = np.zeros_like(y)
    e[:, 0] = 2.0 * rng.rand(N_ROWS)
    e[:, 1] = 20.0 * rng.rand(N_ROWS)
    e[:, 2] = 30.0 * rng.rand(N_ROWS)
    _save(root / "A.npz", y, y + e, mask, wi)
    _save(root / "B.npz", y, y - e, mask, wi)
    return y


def _run(root: Path, members: str, *extra: str, expect_rc: int = 0) -> dict:
    cmd = [sys.executable, str(SCRIPT), "--members", members,
           "--reports-dir", str(root / "reports"), "--out-dir", str(root / "out"),
           "--run-root", str(root / "runs"), "--candidates", str(root / "cand.json"),
           "--smoke", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                          proc.stderr[-1500:])
    rep = root / "reports" / "E8_ensemble_report.json"
    return json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}


class TestE8Ensemble(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        _complementary(self.root)

    def tearDown(self):
        self._td.cleanup()

    def _spec(self, *names: str) -> str:
        return ",".join(f"{n}={self.root}/{n}.npz" for n in names)

    def test_complementary_members_adopted(self):
        rep = _run(self.root, self._spec("A", "B"))
        self.assertEqual(rep["decision"], "adopted", rep["reason"])
        self.assertGreater(rep["fused_total"], 99.0)
        self.assertGreater(rep["delta"], 0.0)
        self.assertGreater(rep["paired_ci"][0], 0.0)
        self.assertFalse(rep["same_source_flagged"], rep["homology"]["same_source_pairs"])
        self.assertEqual(rep["fusion"]["weight_source"], "mean")
        self.assertIn("warning", rep["fusion"])
        gate = json.loads((self.root / "reports" / "E8_gate.json").read_text(encoding="utf-8"))
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        self.assertEqual(gate["decision"], "adopted")
        self.assertTrue(gate["checks"]["homology_reported"])

    def test_identical_member_is_flagged_and_not_adopted(self):
        import shutil
        shutil.copy2(self.root / "A.npz", self.root / "C.npz")
        rep = _run(self.root, self._spec("A", "C"))
        self.assertTrue(rep["same_source_flagged"])
        self.assertIn(["A", "C"], [sorted(p) for p in rep["homology"]["same_source_pairs"]])
        self.assertEqual(rep["decision"], "no_go")
        self.assertLessEqual(rep["paired_ci"][0], 0.0)
        self.assertIn("同源", rep["reason"])

    def test_weights_from_inner_only(self):
        # inner 里 B 极差 → 权重应偏向 A
        y, mask, wi = _base(seed=2)
        _save(self.root / "Ain.npz", y, y + 0.01, mask, wi)
        _save(self.root / "Bin.npz", y, y + 80.0, mask, wi)
        rep = _run(self.root, self._spec("A", "B"),
                   "--inner-members", f"A={self.root}/Ain.npz,B={self.root}/Bin.npz",
                   "--strategy", "weighted")
        self.assertEqual(rep["fusion"]["weight_source"], "inner_oof_simplex")
        self.assertGreater(rep["fusion"]["weights"]["A"], rep["fusion"]["weights"]["B"])
        self.assertGreater(rep["fusion"]["candidates"], 1)

    def test_stacking_uses_inner_ridge(self):
        y, mask, wi = _base(seed=3)
        _save(self.root / "Ain.npz", y, y + 0.01, mask, wi)
        _save(self.root / "Bin.npz", y, y + 40.0, mask, wi)
        rep = _run(self.root, self._spec("A", "B"),
                   "--inner-members", f"A={self.root}/Ain.npz,B={self.root}/Bin.npz",
                   "--strategy", "stacking", "--stacking-l2", "1.0")
        self.assertEqual(rep["fusion"]["weight_source"], "inner_oof_ridge_por")
        self.assertAlmostEqual(sum(rep["fusion"]["weights"].values()), 1.0, places=9)

    def test_arms_status_recorded(self):
        rep = _run(self.root, self._spec("A", "B"))
        self.assertEqual(set(rep["arms_status"]), {"ema", "swa", "topk_snapshot",
                                                   "multi_structure", "multi_seed"})
        self.assertTrue(all(v == "not_provided" for v in rep["arms_status"].values()))

    def test_smoke_does_not_write_repo_candidates(self):
        before = (REPO_CANDIDATES.read_text(encoding="utf-8")
                  if REPO_CANDIDATES.is_file() else None)
        _run(self.root, self._spec("A", "B"))
        after = (REPO_CANDIDATES.read_text(encoding="utf-8")
                 if REPO_CANDIDATES.is_file() else None)
        self.assertEqual(before, after)

    def test_bad_member_spec_fails(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "--members", "A",
                               "--reports-dir", str(self.root / "reports"), "--smoke"],
                              capture_output=True, text=True, timeout=300)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("name=path", proc.stderr + proc.stdout)

    def test_missing_member_file_fails(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "--members",
                               f"A={self.root}/nope.npz",
                               "--reports-dir", str(self.root / "reports"), "--smoke"],
                              capture_output=True, text=True, timeout=300)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("不存在", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
