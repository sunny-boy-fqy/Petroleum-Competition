"""E8/P1-transductive 测试（**口径层，无 torch**）：只用测试输入、绝不用测试标签。

`E8/code/pseudo_label.py` 的两条硬纪律在这里被钉死：
  * **反泄漏护栏**：`--test-preds` 里出现 `y_true`/`mask`/`labels` 等标签键 → 退出码 7，
    机制上杜绝"偷偷用测试标签"；
  * **合法性标注**：结论必须写明"增益（若有）来自推理期使用测试输入分布，不是训练时改进"；
  * 适配是**逐井**算子：必须保持井内排序（用多口井拼起来比较是错的，本条已在实现里纠正）；
  * 开/关消融必须完成；采纳需"总分上升 + 配对 CI 下界 > 0 + 逐井单调"。

伪测试井思路：把部分折当作"不知道标签的测试井"，适配后再打分。
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

SCRIPT = V4 / "E8" / "code" / "pseudo_label.py"
N_WELLS, ROWS_PER_WELL = 16, 20
N_ROWS = N_WELLS * ROWS_PER_WELL


def _well_offset_case(root: Path, offset: float = 3.0):
    """**分布漂移**场景：fold0 的井（伪测试井）POR 整体偏移 +offset，fold1 的井（参考）不偏。

    这正是 transductive 适配要处理的局面：参考分布来自"训练侧"井，被适配的井整体偏移，
    因此对齐中位数对**每一口**伪测试井都是同向改善 → 配对 CI 才能收敛到正区间。
    """
    rng = np.random.RandomState(0)
    y = np.column_stack([10 + 4 * rng.rand(N_ROWS), 40 + 20 * rng.rand(N_ROWS),
                         60 + 20 * rng.rand(N_ROWS)])
    wi = np.repeat(np.arange(N_WELLS), ROWS_PER_WELL)
    fold = np.repeat(np.arange(N_WELLS) % 2, ROWS_PER_WELL)
    cont = y.copy()
    cont[:, 0] += np.where(fold == 0, float(offset), 0.0)
    np.savez_compressed(root / "oof.npz", cont=cont, y_true=y,
                        mask=np.ones((N_ROWS, 3)), well_index=wi, fold_of_row=fold)
    return y, cont, wi


def _run(root: Path, *extra: str, expect_rc: int = 0) -> dict:
    cmd = [sys.executable, str(SCRIPT), "--oof", str(root / "oof.npz"),
           "--reports-dir", str(root / "reports"), "--run-root", str(root / "runs"),
           "--smoke", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == expect_rc, (proc.returncode, proc.stdout[-1500:],
                                          proc.stderr[-1500:])
    rep = root / "reports" / "E8_transductive.json"
    return json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else {}


class TestTransductiveGuards(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        _well_offset_case(self.root)

    def tearDown(self):
        self._td.cleanup()

    def test_label_keys_in_test_preds_are_refused(self):
        _, cont, wi = _well_offset_case(self.root)
        leak = self.root / "leak.npz"
        np.savez_compressed(leak, cont=cont[:20], y_true=np.zeros((20, 3)),
                            well_index=wi[:20])
        proc = subprocess.run([sys.executable, str(SCRIPT), "--oof",
                               str(self.root / "oof.npz"), "--test-preds", str(leak),
                               "--reports-dir", str(self.root / "reports"), "--smoke"],
                              capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 7)
        self.assertIn("拒绝运行", proc.stderr + proc.stdout)

    def test_test_side_adaptation_written_without_labels(self):
        _, cont, wi = _well_offset_case(self.root)
        tp = self.root / "test.npz"
        np.savez_compressed(tp, cont=cont[:40], well_index=np.repeat([0, 1], 20),
                            well_ids=np.asarray(["a", "b"], dtype=object))
        rep = _run(self.root, "--test-preds", str(tp), "--method", "well_mean_align",
                   "--adapt-targets", "POR", "--holdout-folds", "0")
        blk = rep["test_side"]
        self.assertIsNotNone(blk["path"])
        self.assertTrue(blk["guard"]["ok"])
        self.assertTrue((self.root / "runs" / "E8" / "transductive"
                         / "test_adapted.npz").is_file())
        arr = np.load(blk["path"], allow_pickle=True)
        self.assertIn("cont", arr.files)
        self.assertNotIn("y_true", arr.files)

    def test_missing_oof_fails(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "--oof",
                               str(self.root / "nope.npz"),
                               "--reports-dir", str(self.root / "reports"), "--smoke"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 4)


class TestTransductiveAblation(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        _well_offset_case(self.root)

    def tearDown(self):
        self._td.cleanup()

    def test_well_mean_align_fixes_well_level_bias(self):
        rep = _run(self.root, "--method", "well_mean_align", "--alpha", "1.0",
                   "--adapt-targets", "POR", "--holdout-folds", "0")
        self.assertEqual(rep["eval_mode"], "held_out_train_wells")
        self.assertGreater(rep["adapted_total"], rep["base_total"])
        self.assertGreater(rep["paired_ci"][0], 0.0)
        self.assertTrue(rep["monotone"]["ok"], "逐井排序必须保持")
        self.assertEqual(rep["decision"], "adopted", rep["reason"])
        self.assertIn("推理期", rep["legality"]["statement"])
        self.assertFalse(rep["legality"]["uses_test_labels"])
        gate = json.loads((self.root / "reports" / "E8_P1_transductive_gate.json")
                          .read_text(encoding="utf-8"))
        for key in ("ablation_on_off_completed", "antileak_guard_passed",
                    "legality_documented", "paired_ci_reported", "monotone_transform",
                    "contract_ok", "no_label_leak"):
            self.assertTrue(gate["checks"][key], key)
        self.assertIsNone(gate["passed"], "smoke 不判 Gate")
        self.assertEqual(gate["decision"], "adopted")

    def test_all_targets_over_correction_is_rejected(self):
        """对**已经校准**的目标（PERM/SW）也做井级平移只会伤分 → 必须如实判 no_go。"""
        rep = _run(self.root, "--method", "well_mean_align", "--adapt-targets", "all",
                   "--holdout-folds", "0")
        self.assertLess(rep["adapted_total"], rep["base_total"])
        self.assertEqual(rep["decision"], "no_go")
        self.assertEqual(rep["adapt_targets"], list(C.TARGETS))

    def test_none_method_is_identity(self):
        rep = _run(self.root, "--method", "none", "--holdout-folds", "0")
        self.assertAlmostEqual(rep["adapted_total"], rep["base_total"], places=9)
        self.assertAlmostEqual(rep["delta"], 0.0, places=9)
        self.assertEqual(rep["decision"], "no_go")

    def test_alpha_zero_is_identity_for_any_method(self):
        rep = _run(self.root, "--method", "quantile_map", "--alpha", "0",
                   "--holdout-folds", "0")
        self.assertAlmostEqual(rep["delta"], 0.0, places=9)

    def test_help_lists_knobs(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        for flag in ("--oof", "--test-preds", "--method", "--alpha", "--adapt-targets",
                     "--holdout-folds", "--iters"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
