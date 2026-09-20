"""数据解析与列布局测试（含 R2-B1/B5 回归）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C          # noqa: E402
from src.data import labels as L        # noqa: E402
from src.data import parse as P         # noqa: E402

DATA = V4.parent / "data"
ODD_WELLS = {
    "42f2870b6ea743518d4ff77acca0462a": (20, ()),
    "b7eb1274305446c499a7b03848fb5bd3": (21, ()),
    "c7611b0148bb4b878c00bc6d1367a136": (16, ("CASE",)),
}


class TestColumnLayout(unittest.TestCase):
    def test_constants_are_consistent(self):
        self.assertEqual(C.N_INPUT, 13)
        self.assertEqual(len(C.INPUT_COLUMNS), 13)
        self.assertEqual(C.INPUT_COLUMNS[0], "GR")
        self.assertEqual(C.INPUT_COLUMNS[-1], "CASE")
        self.assertEqual(len(C.COLUMNS), 17)
        self.assertEqual(C.TARGET_COLUMNS, ("POR", "PERM", "SW"))
        # 输入列与目标列不得相交（B1 的核心不变式）
        self.assertEqual(set(C.INPUT_COLUMNS) & set(C.TARGET_COLUMNS), set())
        self.assertEqual(C.DEPTH_COLUMN, "DEPTH")


class TestWellParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA.is_dir():
            raise unittest.SkipTest(f"dataset not found: {DATA}")
        cls.train_files = sorted((DATA / "train").glob("*.txt"))
        cls.test_files = sorted((DATA / "test").glob("*.txt"))

    def test_train_shape_and_no_leak(self):
        """每个训练井：13 列输入，且任何输入列与任何目标列不完全相等。"""
        import numpy as np
        self.assertTrue(self.train_files)
        for f in self.train_files[:10]:
            rec = P.parse_well(f, with_targets=True)
            self.assertEqual(rec.inputs.shape[1], C.N_INPUT, f.name)
            self.assertEqual(rec.targets.shape[1], 3, f.name)
            miss = L.missing_masks(rec.targets)
            for j in range(3):
                m = ~miss[:, j]
                if int(m.sum()) < 20:
                    continue
                for k in range(C.N_INPUT):
                    ratio = float(np.isclose(rec.inputs[m, k], rec.targets[m, j],
                                             atol=1e-9).mean())
                    self.assertLess(ratio, 0.999,
                                    f"{f.name}: inputs[:,{k}] == {C.TARGET_COLUMNS[j]}")

    def test_test_split_has_same_input_width(self):
        """测试井输入宽度必须与训练井一致（否则推理崩溃）。"""
        self.assertTrue(self.test_files)
        for f in self.test_files:
            rec = P.parse_well(f, with_targets=False)
            self.assertEqual(rec.inputs.shape[1], C.N_INPUT, f.name)
            self.assertIsNone(rec.targets, f.name)

    def test_test_split_reports_no_missing_columns(self):
        """R2-B5：测试集本就没有目标列，不应被报成 schema 非规范缺列。"""
        for f in self.test_files:
            rec = P.parse_well(f, with_targets=False)
            self.assertEqual(tuple(rec.missing_columns), (),
                             f"{f.name}: missing_columns={rec.missing_columns}")
            self.assertEqual(tuple(rec.extra_columns), (), f.name)

    def test_noncanonical_wells(self):
        """恰好 3 口非规范训练井：17,426 行多列 + 9,654 行缺 CASE。"""
        noncanon = {}
        for f in self.train_files:
            rec = P.parse_well(f, with_targets=True)
            if rec.missing_columns or rec.extra_columns:
                noncanon[rec.well_id] = (rec.n_header_cols, tuple(rec.missing_columns),
                                         tuple(rec.extra_columns), rec.n_rows)
        self.assertEqual(len(noncanon), 3, noncanon)
        for wid, (cols, miss) in ODD_WELLS.items():
            self.assertIn(wid, noncanon)
            got_cols, got_miss, _, _ = noncanon[wid]
            self.assertEqual(got_cols, cols, wid)
            self.assertEqual(got_miss, miss, wid)
        # 缺 CASE 的只有一口井
        lacking = [w for w, v in noncanon.items() if "CASE" in v[1]]
        self.assertEqual(len(lacking), 1)
        self.assertEqual(lacking[0], "c7611b0148bb4b878c00bc6d1367a136")

    def test_sentinel_handling(self):
        """哨兵必须变成 NaN —— 且必须是**由原始文本算出的**断言，不能同义反复。

        review R7-6：原断言 `isnan(inputs).any() or isfinite(inputs).all()` 恒为真
        （无 NaN 时第二项成立、有 NaN 时第一项成立），解析器即使把哨兵改成 `0.0`
        也照样通过。这里直接从原始 txt 的**输入列**里数出哨兵个数，再要求解析结果
        的 NaN 数不少于它 —— 这才真正锁住"哨兵→NaN"不变式。
        """
        import numpy as np
        from src.data import parse as _P
        n_raw_sentinel = 0
        rec = None
        for f in self.train_files:
            lines = Path(f).read_text(encoding="utf-8-sig").splitlines()
            header = [h.strip() for h in lines[0].split(",")]
            idx = [header.index(c) for c in C.INPUT_COLUMNS if c in header]
            n_here = 0
            for line in lines[2:]:                       # 跳过列名行与单位行
                toks = line.split(",")
                for i in idx:
                    if i >= len(toks):
                        continue
                    tok = toks[i].strip()
                    if not tok:
                        continue
                    try:
                        v = float(tok)
                    except ValueError:
                        continue
                    if v < C.MISSING_LT:                 # -99999 / -9999 / 任何 < -1000
                        n_here += 1
            if n_here > 0:
                rec, n_raw_sentinel = _P.parse_well(f, with_targets=True), n_here
                break
        self.assertIsNotNone(rec, "训练集里应至少有一口井的输入列含哨兵（否则本测试无效）")
        self.assertGreater(n_raw_sentinel, 0)
        n_nan = int(np.isnan(rec.inputs).sum())
        self.assertGreaterEqual(n_nan, n_raw_sentinel,
                                f"原始输入列有 {n_raw_sentinel} 个哨兵，解析后只有 {n_nan} 个 NaN")
        # 哨兵原值不得残留，也不得被换成任何哨兵常量
        self.assertFalse((rec.inputs < C.MISSING_LT).any())
        for s in C.SENTINELS:
            self.assertFalse((rec.inputs == s).any(), f"哨兵原值 {s} 残留在输入里")


class TestFeatureBuilder(unittest.TestCase):
    def test_shapes_and_depth_channel(self):
        import numpy as np
        from src.features import basic
        n = 12
        inputs = np.random.rand(n, C.N_INPUT).astype("float32")
        missing = np.zeros((n, C.N_INPUT), dtype="int8")
        depth = (np.arange(n) * 0.1 + 1800).astype("float32")
        X = basic.build_row_features(inputs, missing, depth)
        self.assertEqual(X.shape, (n, basic.N_FEATURES))
        self.assertEqual(basic.N_FEATURES, 32)
        self.assertEqual(len(basic.FEATURE_NAMES), 32)
        # 第 14 列是 DEPTH 原值
        self.assertTrue(np.allclose(X[:, 13], depth, atol=1e-4))
        # 相对深度首末为 0/1
        self.assertAlmostEqual(float(X[0, 29]), 0.0, places=5)
        self.assertAlmostEqual(float(X[-1, 29]), 1.0, places=5)

    def test_wrong_curve_count_raises(self):
        import numpy as np
        from src.features import basic
        with self.assertRaises(AssertionError):
            basic.build_row_features(np.zeros((3, 12), "float32"),
                                     np.zeros((3, 12), "int8"),
                                     np.zeros(3, "float32"))


class TestLabels(unittest.TestCase):
    def test_placeholder_flags_and_states(self):
        import numpy as np
        t = np.array([[0.1, 0.01, 99.9],
                      [0.2, 1.0, 50.0],
                      [-99999.0, -99999.0, -99999.0]])
        ph = L.placeholder_flags(t)
        self.assertTrue(bool(ph[0]))
        self.assertFalse(bool(ph[1]))
        st = L.state_labels(t)
        self.assertEqual(list(st), [1, 0, -1])

    def test_perm_log_roundtrip(self):
        import numpy as np
        z = np.array([-3.0, 0.0, 2.5])
        perm = L.perm_from_log10(z)
        self.assertTrue((perm > 0).all())
        self.assertTrue(np.allclose(L.perm_to_log10(perm), z, atol=1e-9))

    def test_sw_single_scale(self):
        """R2-B2：SW 是单一标签尺度，默认不做 ×100 换算。"""
        self.assertFalse(C.SW_SMALL_BRANCH)
        out = L.sw_decode([0.0], [80.0])
        self.assertAlmostEqual(float(out[0]), 80.0, places=6)   # q=0 -> 等于有效分支
        out2 = L.sw_decode([1.0], [80.0])
        self.assertAlmostEqual(float(out2[0]), 99.9, places=6)   # q=1 -> 占位常量


if __name__ == "__main__":
    unittest.main(verbosity=2)
