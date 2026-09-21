"""E1/P0 行级数据管线测试（**只依赖 numpy**，无 torch 也能跑）。

覆盖 E1/P0 §7 的完成判据：
  * 特征/标签逐行对齐、按井 offset 自洽；
  * `RowScaler` 只由给定（训练折）行拟合，可 JSON 往返；
  * 目标尺度参数只来自训练井（验证井标签不影响它）；
  * 折划分井维度互斥、无井级泄漏；
  * `POR=0` / `POR<0.1` 切片被正确构造并打分（连续头可表示 ~0 的判据）；
  * SW 仿射归一化与其逆严格互逆；
  * 缓存构建幂等、体积/内存可报告。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _synth_cache as SC  # noqa: E402
from src import constants as C  # noqa: E402
from src.data import dataset as D  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import basic as F  # noqa: E402

WELLS = [f"w{i:02d}" for i in range(6)]


class TestRowPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        cls.cache = root / "cache"
        cls.tr_dir, cls.te_dir, _ = SC.build_cache(cls.cache, WELLS, n_rows=48, seed=7)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    # ------------------------------------------------------------ 对齐
    def test_row_alignment_and_offsets(self):
        t = RD.assemble(WELLS, self.cache, scaler=None, with_targets=True)
        RD.assert_alignment(t)                       # 断言内部已做全部形状/offset 校验
        self.assertEqual(t.n_wells, len(WELLS))
        self.assertEqual(int(t.offsets[-1]), t.n_rows)
        self.assertEqual(t.X.shape[1], F.N_FEATURES)
        self.assertEqual(t.mask.shape, (t.n_rows, 3))
        self.assertEqual(t.y_atom.shape, (t.n_rows, 3))
        # 逐井 offset 必须与该井真实行数一致
        for i, w in enumerate(WELLS):
            n = int(RD.read_row_shard(self.cache, w, "train")["X_raw"].shape[0])
            self.assertEqual(int(t.offsets[i + 1] - t.offsets[i]), n, w)

    def test_subset_keeps_rows_of_selected_wells_only(self):
        t = RD.assemble(WELLS, self.cache, scaler=None, with_targets=True)
        sub = RD.subset(t, WELLS[:2])
        self.assertEqual(sub.well_ids, tuple(WELLS[:2]))
        RD.assert_alignment(sub)
        self.assertEqual(sub.n_rows, int(t.offsets[2]))
        for name in ("y_por", "y_sw", "mask", "y_atom", "y_joint"):
            np.testing.assert_allclose(getattr(sub, name), getattr(t, name)[:sub.n_rows])

    def test_features_contain_nan_for_missing_curves(self):
        """缺测曲线在特征里保持 NaN（标准化时才填补）——否则缺失信息被提前抹掉。"""
        t = RD.assemble(WELLS, self.cache, scaler=None, with_targets=False)
        nan_frac = float(np.mean(~np.isfinite(t.X)))
        self.assertGreater(nan_frac, 0.0)
        # 缺失指示位列（14..27）必须永远是有限值
        self.assertTrue(np.isfinite(t.X[:, 14:28]).all())

    # ------------------------------------------------------------ 折内 fit
    def test_target_scalers_exclude_atomic_rows(self):
        """sw_mu/por_median/perm_z_median 必须由非原子有效行拟合，不能被占位尖峰拉走。"""
        n, valid = 200, 60
        por = np.full(n, C.ATOM_VALUES["POR"], dtype="float64")
        perm = np.full(n, C.ATOM_VALUES["PERM"], dtype="float64")
        sw = np.full(n, C.ATOM_VALUES["SW"], dtype="float64")
        por[:valid] = np.linspace(5.0, 25.0, valid)
        perm[:valid] = np.logspace(-1.0, 1.0, valid)
        sw[:valid] = np.linspace(75.0, 95.0, valid)
        mask = np.ones((n, 3), dtype="float64")
        sc = F.fit_target_scalers(por, sw, mask, z_perm=np.log10(perm), y_perm=perm)
        self.assertAlmostEqual(sc["por_median"], float(np.median(por[:valid])), places=6)
        self.assertAlmostEqual(sc["sw_mu"], float(np.median(sw[:valid])), places=6)
        self.assertAlmostEqual(sc["perm_z_median"],
                              float(np.median(np.log10(perm[:valid]))), places=6)
        self.assertLess(sc["por_max"], 1.2 * 25.0 + 1e-9)
        self.assertGreater(sc["sw_sigma"], 1e-6)

    def test_row_scaler_fit_uses_only_given_rows(self):
        t = RD.assemble(WELLS, self.cache, scaler=None, with_targets=True)
        head = t.X[:100]
        a = RD.RowScaler.fit(head, wells=("w00",))
        self.assertEqual(a.fit_wells, ("w00",))
        self.assertEqual(a.n_fit_rows, 100)
        # 用**别的行**再 fit，参数必须不同（证明 fit 真的在看传入的行）
        b = RD.RowScaler.fit(t.X[100:400], wells=("w01",))
        self.assertFalse(np.allclose(a.mean, b.mean))
        # 相同输入 -> 相同参数（确定性）
        a2 = RD.RowScaler.fit(head, wells=("w00",))
        np.testing.assert_allclose(a.mean, a2.mean)
        np.testing.assert_allclose(a.std, a2.std)

    def test_row_scaler_json_roundtrip(self):
        t = RD.assemble(WELLS, self.cache, scaler=None, with_targets=True)
        s = RD.RowScaler.fit(t.X, wells=tuple(WELLS))
        back = RD.RowScaler.from_dict(json.loads(json.dumps(s.to_dict())))
        np.testing.assert_allclose(s.transform(t.X), back.transform(t.X))
        self.assertIn("feature_names", s.to_dict())

    def test_median_imputation_is_applied_on_transform(self):
        s = RD.RowScaler.fit(np.array([[1.0, 10.0], [3.0, 30.0]], dtype="float64"))
        out = s.transform(np.array([[np.nan, 30.0]], dtype="float64"))
        # NaN 被替换为该列 median(2.0) 后再标准化 -> 与显式填 2.0 的结果一致
        ref = s.transform(np.array([[2.0, 30.0]], dtype="float64"))
        np.testing.assert_allclose(out, ref)

    def test_target_scalers_only_use_train_wells(self):
        tr_wells, va_wells = WELLS[:4], WELLS[4:]
        fit = RD.fit_scalers_from_wells(tr_wells, self.cache)
        self.assertEqual(set(fit["scaler"].fit_wells), set(tr_wells))
        self.assertTrue(set(fit["scaler"].fit_wells).isdisjoint(va_wells))
        # 直接由训练井张量重算，必须逐键相等
        tr = RD.assemble(tr_wells, self.cache, scaler=None, with_targets=True)
        ref = F.fit_target_scalers(tr.y_por, tr.y_sw, tr.mask, z_perm=tr.y_perm_z)
        for k in ("por_max", "por_median", "sw_mu", "sw_sigma", "s_por", "s_sw"):
            self.assertAlmostEqual(float(fit["target"][k]), float(ref[k]), places=6, msg=k)

    def test_fold_wells_are_disjoint_and_complete(self):
        folds = {"n_folds": 3, "well_list": WELLS,
                 "fold_of_well": {w: i % 3 for i, w in enumerate(WELLS)}}
        seen = []
        for k in range(3):
            tr, va = RD.fold_wells(folds, k)
            self.assertEqual(set(tr) & set(va), set(), f"fold{k} 有重叠")
            self.assertEqual(set(tr) | set(va), set(WELLS))
            seen += va
        self.assertEqual(sorted(seen), sorted(WELLS))

    def test_fold_wells_rejects_bad_fold_index(self):
        folds = {"n_folds": 2, "well_list": WELLS, "fold_of_well": {w: 0 for w in WELLS}}
        with self.assertRaises(ValueError):
            RD.fold_wells(folds, 2)

    # ------------------------------------------------------------ 边界切片
    def test_por_zero_and_small_slices_exist_and_are_scored(self):
        t = RD.assemble(WELLS, self.cache, scaler=None, with_targets=True)
        y_true = np.stack([t.y_por, np.power(10.0, t.y_perm_z), t.y_sw], axis=1)
        rep = RD.atom_slice_report(y_true, y_true, t.mask >= 0.5)
        self.assertGreater(rep["por_eq_0"]["n_rows"], 0, "合成数据应含 POR=0 行")
        self.assertGreater(rep["por_lt_0p1"]["n_rows"], 0, "合成数据应含 POR<0.1 行")
        self.assertGreaterEqual(rep["por_lt_0p1"]["n_rows"], rep["por_eq_0"]["n_rows"])
        self.assertAlmostEqual(rep["por_eq_0"]["acc_por"], 1.0, places=6)
        self.assertIsInstance(rep["por_eq_0"]["n_rows"], int, "n_rows 必须是 int 而不是 float")

    def test_por_parameterization_can_represent_zero(self):
        """`por = por_max·sigmoid(g)`：g 负得多时必须能逼近 0（禁止 `0.1+softplus`）。"""
        por_max = 39.8
        for g in (-10.0, -20.0):
            val = por_max * (1.0 / (1.0 + np.exp(-g)))
            self.assertLess(val, 0.1)
        self.assertLess(por_max * (1.0 / (1.0 + np.exp(20.0))), 1e-6)
        # 反向：POR 中位数 11.34 必须可表示
        g = float(np.log(11.34 / (por_max - 11.34)))
        self.assertAlmostEqual(por_max * (1.0 / (1.0 + np.exp(-g))), 11.34, places=3)

    def test_sw_affine_roundtrip(self):
        scaler = {"sw_mu": 82.805, "sw_sigma": 6.5}
        sw = np.array([8.305, 45.0, 82.805, 99.9])
        z = F.apply_sw_scaler(sw, scaler)
        np.testing.assert_allclose(F.invert_sw_scaler(z, scaler), sw, rtol=1e-12)
        # 反变换后绝不落在 [0,1]（SW 是百分数尺度）
        self.assertTrue((F.invert_sw_scaler(z, scaler) > 1.0).all())

    def test_sw_sigma_zero_is_rejected(self):
        with self.assertRaises(ZeroDivisionError):
            F.apply_sw_scaler(np.array([50.0]), {"sw_mu": 50.0, "sw_sigma": 0.0})

    # ------------------------------------------------------------ 缓存与报告
    def test_build_cache_skips_existing_well_shards(self):
        before = {p.name: p.stat().st_mtime_ns
                  for p in sorted((self.cache / "raw" / "train").glob("*.npz"))}
        self.assertEqual(len(before), len(WELLS))
        before_man = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        man = D.build_cache(self.tr_dir, self.te_dir, self.cache, verbose=False)
        after = {p.name: p.stat().st_mtime_ns
                 for p in sorted((self.cache / "raw" / "train").glob("*.npz"))}
        self.assertEqual(before, after, "已存在且合法的井分片不得被重写")
        self.assertEqual(man["counts"], before_man["counts"])
        self.assertEqual(int(man["counts"]["train_wells"]), len(WELLS))
        self.assertEqual(int(man["counts"]["train_rows"]),
                         sum(int(np.load(p, allow_pickle=True)["n_rows"])
                             for p in sorted((self.cache / "raw" / "train").glob("*.npz"))))

    def test_row_cache_is_idempotent(self):
        info = RD.build_row_cache(self.cache, WELLS, "train", verbose=False)
        self.assertEqual(info["built"], 0, "第二次构建不应重写任何井")
        self.assertEqual(info["wells"], len(WELLS))
        self.assertGreater(info["bytes"], 0)

    def test_reports_are_jsonable_and_bounded(self):
        t = RD.assemble(WELLS, self.cache, scaler=None, with_targets=False)
        mem = RD.memory_report(t)
        self.assertEqual(mem["n_rows"], t.n_rows)
        self.assertLess(mem["gib"], 1.0, "合成小数据不应占 1 GiB 以上")
        rep = RD.feature_report(t)
        self.assertEqual(rep["n_features"], F.N_FEATURES)
        self.assertEqual(len(rep["feature_names"]), F.N_FEATURES)
        self.assertEqual(len(rep["nan_rate_per_feature"]), F.N_FEATURES)
        json.dumps(rep, ensure_ascii=False)          # 必须可 JSON 化

    def test_scaler_json_roundtrip_via_files(self):
        with tempfile.TemporaryDirectory() as td:
            p = RD.save_scaler_json(Path(td) / "s.json", {"fold": 0, "target_scalers": {}})
            self.assertTrue(p.is_file())
            self.assertEqual(RD.load_scaler_json(p)["fold"], 0)

    def test_constants_atom_values_are_the_labels_used(self):
        """切片判据用的原子值必须来自 constants（单一事实源）。"""
        self.assertEqual(C.ATOM_VALUES["POR"], 0.1)
        self.assertEqual(C.ATOM_VALUES["SW"], 99.9)
        self.assertFalse(C.SW_SMALL_BRANCH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
