"""E2 特征工程与增强测试（**只依赖 numpy**，无 torch 也能跑）。

覆盖 E2/P0 §7、E2/P1 §7、E2/P2 §7 的完成判据：
  * 物理派生列的公式正确、缺失传播为 NaN + 指示位、越界比例可报告、**不消费标签**；
  * GR/SP 分位数参数只由给定（训练）井拟合；
  * 窗口统计严格居中、coverage 正确、全缺窗口 → NaN、trend 在斜坡上 = 1、短井收缩上报；
  * 井级统计逐行广播、只用该井自身的行；
  * `FeatureSpec` 的列序/列数/缓存 key 自洽，provenance 覆盖全部列；
  * 增强只动特征、标签与 mask 逐行不变；同种子可复现；关闭时等价于拷贝。
"""
from __future__ import annotations

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
from src.data import augment as AUG  # noqa: E402
from src.data import dataset as D  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import basic as F  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.features import physics as PH  # noqa: E402
from src.features import well as WL  # noqa: E402
from src.features import window as WN  # noqa: E402


def _curves(n: int = 40, seed: int = 0, nan_at: tuple | None = None) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    x = np.zeros((n, len(C.INPUT_COLUMNS)), dtype="float64")
    for j in range(x.shape[1]):
        x[:, j] = 10.0 + j + np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
    # 给几条曲线真实量纲（AC μs/m ≈ 240、DEN ≈ 2.3、CNL ≈ 15、RT ≈ 90）
    x[:, C.INPUT_COLUMNS.index("AC")] = 240.0
    x[:, C.INPUT_COLUMNS.index("DEN")] = 2.3
    x[:, C.INPUT_COLUMNS.index("CNL")] = 15.0
    x[:, C.INPUT_COLUMNS.index("RT")] = 90.0
    x[:, C.INPUT_COLUMNS.index("RXO")] = 45.0
    x[:, C.INPUT_COLUMNS.index("PE")] = 2.5
    for t in (nan_at or ()):
        x[t[0], t[1]] = np.nan
    return x


class TestPhysicsFeatures(unittest.TestCase):
    def test_sonic_and_density_porosity_formulas(self):
        x = _curves(5)
        ps = PH.phi_sonic(x[:, C.INPUT_COLUMNS.index("AC")])
        np.testing.assert_allclose(ps, (240.0 - PH.ACMA_US_M) / (PH.ACF_US_M - PH.ACMA_US_M),
                                   rtol=1e-12)
        pd_ = PH.phi_den(x[:, C.INPUT_COLUMNS.index("DEN")])
        np.testing.assert_allclose(pd_, (2.65 - 2.3) / (2.65 - 1.0), rtol=1e-12)
        pn = PH.phi_neu(x[:, C.INPUT_COLUMNS.index("CNL")])
        np.testing.assert_allclose(pn, 0.15, rtol=1e-12)

    def test_acma_acf_units_are_metric(self):
        """AC 数据是 μs/m；常数必须是换算后的量级（E2/P0 §9 的单位风险）。"""
        self.assertGreater(PH.ACMA_US_M, 150.0)
        self.assertLess(PH.ACMA_US_M, 220.0)
        self.assertGreater(PH.ACF_US_M, 550.0)
        self.assertLess(PH.ACF_US_M, 700.0)

    def test_missing_input_propagates_to_nan_plus_indicator(self):
        x = _curves(10, nan_at=((3, C.INPUT_COLUMNS.index("AC")),))
        params = PH.PhysicsParams()
        X, names = PH.build_physics_features(x, params, with_indicators=True)
        j = names.index("phi_sonic")
        self.assertTrue(np.isnan(X[3, j]), "AC 缺测时 Wyllie 孔隙度必须为 NaN，不能填 0")
        self.assertEqual(float(X[3, names.index("phi_sonic_ok")]), 0.0)
        self.assertEqual(float(X[0, names.index("phi_sonic_ok")]), 1.0)
        # 未受影响的列在同一样本上仍然有效
        self.assertTrue(np.isfinite(X[3, names.index("phi_den")]))

    def test_igr_uses_fold_quantiles_and_clips(self):
        x = _curves(20)
        gi = C.INPUT_COLUMNS.index("GR")
        p = PH.PhysicsParams(gr_min=float(x[:, gi].min()), gr_max=float(x[:, gi].max()))
        ig = PH.igr(x[:, gi], p.gr_min, p.gr_max)
        self.assertGreaterEqual(float(ig.min()), 0.0)
        self.assertLessEqual(float(ig.max()), 1.0)
        self.assertAlmostEqual(float(ig.max()), 1.0, places=9)

    def test_fit_physics_params_uses_only_given_rows(self):
        a = _curves(30, seed=1)
        b = _curves(30, seed=2) + 50.0
        p_a = PH.fit_physics_params(a)
        p_b = PH.fit_physics_params(b)
        self.assertNotAlmostEqual(p_a.gr_min, p_b.gr_min)
        # 只用 a 拟合两次 -> 完全一致（确定性与"只看传入行"）
        p_a2 = PH.fit_physics_params(a)
        self.assertEqual(p_a.as_dict(), p_a2.as_dict())

    def test_out_of_range_report_flags_bad_units(self):
        """把 AC 当成 μs/ft（240 → 应报越界）会被区间检查抓到。"""
        x = _curves(10)
        x[:, C.INPUT_COLUMNS.index("AC")] = 24000.0        # 明显单位错误
        X, names = PH.build_physics_features(x, PH.PhysicsParams(), with_indicators=False)
        rep = PH.out_of_range_report(X, names)
        self.assertGreater(rep["phi_sonic"]["frac_out_of_range"], 0.9)

    def test_label_free_and_no_target_in_signature(self):
        self.assertTrue(PH.label_independence_audit()["ok"])
        self.assertTrue(G.audit_no_target_derivation(G.spec_from_name("F2"))["ok"])

    def test_provenance_covers_every_feature(self):
        for name in PH.PHYSICS_FEATURES:
            self.assertIn(name, PH.PHYSICS_PROVENANCE, f"{name} 缺少公式出处")
            formula, src = PH.PHYSICS_PROVENANCE[name]
            self.assertTrue(formula and src)


class TestWindowFeatures(unittest.TestCase):
    def test_window_is_centered_and_coverage_counts_valid(self):
        x = _curves(60, nan_at=((10, 0),))
        X, names, meta = WN.build_window_features(x, windows=(11,), stats=("mean", "coverage"))
        wv = WN._windows(x, 11)
        i = names.index("GR_w11_mean")
        np.testing.assert_allclose(X[30, i], np.nanmean(wv[30, :, 0]), rtol=1e-6)
        c = names.index("GR_w11_coverage")
        self.assertLess(float(X[10, c]), 1.0)
        self.assertAlmostEqual(float(X[30, c]), 1.0, places=6)
        self.assertEqual(meta["windows_used"]["11"], 11)

    def test_all_nan_window_is_nan_not_zero(self):
        x = np.full((20, len(C.INPUT_COLUMNS)), np.nan)
        x[:, 1] = 1.0
        X, names, _ = WN.build_window_features(x, windows=(11,), stats=("mean",))
        self.assertTrue(np.isnan(X[:, names.index("GR_w11_mean")]).all())
        self.assertTrue(np.isfinite(X[:, names.index("PE_w11_mean")]).all())

    def test_trend_equals_slope_on_a_ramp(self):
        x = np.zeros((41, len(C.INPUT_COLUMNS)))
        x[:, 0] = np.arange(41, dtype="float64")
        X, names, _ = WN.build_window_features(x, windows=(11,), stats=("trend",))
        np.testing.assert_allclose(X[20, names.index("GR_w11_trend")], 1.0, rtol=1e-6)

    def test_short_well_shrinks_and_reports(self):
        x = _curves(7)
        X, names, meta = WN.build_window_features(x, windows=(201,), stats=("mean",))
        self.assertEqual(meta["windows_used"]["201"], 7)
        self.assertEqual(meta["shrunk"], [201])
        self.assertTrue(np.isfinite(X).any())

    def test_column_count_matches_names(self):
        x = _curves(30)
        X, names, _ = WN.build_window_features(x)
        expected = len(C.INPUT_COLUMNS) * len(WN.DEFAULT_WINDOWS) * len(WN.DEFAULT_STATS)
        self.assertEqual(X.shape[1], expected)
        self.assertEqual(len(names), expected)

    def test_label_free(self):
        self.assertTrue(WN.label_independence_audit()["ok"])


class TestWellFeatures(unittest.TestCase):
    def test_stats_are_broadcast_and_correct(self):
        x = _curves(25)
        depth = 1000.0 + 0.1 * np.arange(25)
        X, names = WL.build_well_features(x, depth)
        self.assertEqual(X.shape[0], 25)
        i = names.index("AC_well_mean")
        for r in range(25):
            self.assertAlmostEqual(float(X[r, i]), float(np.nanmean(x[:, C.INPUT_COLUMNS.index("AC")])),
                                   places=6)
        self.assertAlmostEqual(float(X[0, names.index("well_depth_span")]), 2.4, places=6)
        self.assertAlmostEqual(float(X[0, names.index("well_n_rows")]), 25.0, places=6)

    def test_nan_curve_gives_nan_stat_but_well_still_builds(self):
        x = _curves(15)
        x[:, C.INPUT_COLUMNS.index("PE")] = np.nan
        depth = 1000.0 + 0.1 * np.arange(15)
        X, names = WL.build_well_features(x, depth)
        self.assertTrue(np.isnan(X[0, names.index("PE_well_mean")]))
        self.assertFalse(np.isnan(X[0, names.index("GR_well_mean")]))

    def test_all_nan_well_raises(self):
        x = np.full((5, len(C.INPUT_COLUMNS)), np.nan)
        depth = np.arange(5, dtype="float64")
        with self.assertRaises(ValueError):
            WL.build_well_features(x, depth)

    def test_label_free(self):
        self.assertTrue(WL.label_independence_audit()["ok"])


class TestFeatureSpec(unittest.TestCase):
    def test_group_sizes_and_total(self):
        spec = G.spec_from_name("F2")
        sizes = spec.group_sizes()
        self.assertEqual(sizes["F1"], F.N_FEATURES)
        self.assertEqual(sizes["phys"], 2 * len(PH.PHYSICS_FEATURES))
        self.assertEqual(sizes["win"],
                         len(C.INPUT_COLUMNS) * len(WN.DEFAULT_WINDOWS) * len(WN.DEFAULT_STATS))
        self.assertEqual(sizes["well"], len(WL.well_names()))
        self.assertEqual(spec.n_features(), sum(sizes.values()))
        self.assertEqual(len(spec.names()), spec.n_features())

    def test_group_order_is_canonical(self):
        """列序必须与 GROUP_ORDER 一致，否则同一 spec 会有两种列排列。"""
        spec = G.FeatureSpec(groups=("well", "F1", "win"))
        self.assertEqual(spec.groups, ("F1", "win", "well"))
        self.assertEqual(spec.names()[:F.N_FEATURES], list(F.FEATURE_NAMES))

    def test_spec_key_changes_with_config(self):
        a = G.spec_from_name("F1+win")
        b = G.FeatureSpec(groups=("F1", "win"), windows=(11, 51))
        self.assertNotEqual(a.key, b.key, "换窗口配置必须换缓存 key")

    def test_spec_json_roundtrip(self):
        spec = G.spec_from_name("F2")
        back = G.FeatureSpec.from_dict(spec.as_dict())
        self.assertEqual(back, spec)
        self.assertEqual(back.names(), spec.names())

    def test_provenance_full_coverage_and_groups(self):
        spec = G.spec_from_name("F2")
        rows = G.provenance_rows(spec)
        self.assertEqual(len(rows), spec.n_features())
        counts = G.provenance_group_counts(spec)
        self.assertEqual(counts, spec.group_sizes())

    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):
            G.FeatureSpec(groups=("F1", "nope"))
        with self.assertRaises(ValueError):
            G.spec_from_name("F1+bogus")


class TestAugment(unittest.TestCase):
    def _data(self, n=80):
        x = np.random.default_rng(0).normal(size=(n, 8)).astype("float32")
        y = np.random.default_rng(1).normal(size=(n, 3)).astype("float32")
        mask = np.ones((n, 3), dtype="float32")
        return x, y, mask

    def test_disabled_returns_identical_copy(self):
        x, _, _ = self._data()
        rng = np.random.default_rng(0)
        out = AUG.augment_matrix(x, AUG.AugmentConfig(enabled=False), rng)
        np.testing.assert_array_equal(out, x)
        self.assertIsNot(out, x)

    def test_labels_and_mask_are_untouched(self):
        x, y, mask = self._data()
        cfg = AUG.AugmentConfig(enabled=True, channel_mask_p=0.5, depth_jitter=2,
                                gauss_sigma_frac=0.05)
        out = AUG.augment_matrix(x, cfg, np.random.default_rng(0))
        # 增强只接受特征矩阵：形状不变，标签对象根本没被触碰
        self.assertEqual(out.shape, x.shape)
        rep = AUG.label_invariance(x, out, y, mask)
        self.assertTrue(rep["rows_preserved"])
        self.assertTrue(rep["features_changed"])
        self.assertEqual(rep["y_rows"], y.shape[0])
        self.assertEqual(rep["mask_rows"], mask.shape[0])

    def test_same_seed_is_reproducible(self):
        x, _, _ = self._data()
        cfg = AUG.AugmentConfig(enabled=True, gauss_sigma_frac=0.05, depth_jitter=2)
        a = AUG.augment_matrix(x, cfg, np.random.default_rng(7))
        b = AUG.augment_matrix(x, cfg, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_column_scales_are_robust(self):
        x = np.zeros((100, 3))
        x[:, 0] = np.arange(100, dtype="float64")
        x[:, 1] = 5.0
        sc = AUG.column_scales(x)
        self.assertGreater(sc[0], 0.0)
        self.assertEqual(sc[1], 0.0, "常数列为 0 尺度（不该造假噪声）")

    def test_curve_augmentation_sets_missing_bits(self):
        x = _curves(200)
        miss = np.zeros((200, len(C.INPUT_COLUMNS)), dtype="int8")
        cfg = AUG.AugmentConfig(enabled=True, channel_mask_p=1.0, channel_outage_len=51,
                                depth_jitter=0, gauss_sigma_frac=0.0, segment_resample_p=0.0)
        xa, ma = AUG.augment_curves(x, miss, cfg, np.random.default_rng(3))
        self.assertEqual(int(ma.sum()), 51 * len(C.INPUT_COLUMNS))
        self.assertEqual(int(np.isnan(xa).sum()), 51 * len(C.INPUT_COLUMNS))

    def test_disabled_curve_augmentation_is_identity(self):
        x = _curves(20)
        miss = np.zeros((20, len(C.INPUT_COLUMNS)), dtype="int8")
        xa, ma = AUG.augment_curves(x, miss, AUG.AugmentConfig(enabled=False),
                                    np.random.default_rng(0))
        np.testing.assert_allclose(xa, x, equal_nan=True)
        self.assertEqual(int(ma.sum()), 0)

    def test_augmentation_off_helper(self):
        cfg = AUG.AugmentConfig(enabled=True)
        self.assertFalse(AUG.augmentation_off(cfg).enabled)
        self.assertTrue(cfg.enabled)


class TestFeatureCacheRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.wells = [f"w{i}" for i in range(3)]
        cls.tr, cls.te, cls.cache = SC.build_cache(Path(cls._td.name) / "cache", cls.wells,
                                                   n_rows=40, seed=9)
        cls.phys = G.fit_physics_params(D.read_well_shard(cls.cache, w, "train")
                                        for w in cls.wells)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_cache_roundtrip_and_spec_mismatch(self):
        spec = G.spec_from_name("F1+phys")
        G.build_feature_cache(self.cache, spec, self.wells, self.phys, split="train",
                              from_raw=lambda w: D.read_well_shard(self.cache, w, "train"))
        X, names = G.read_feature_cache(self.cache, spec, self.wells[0], "train")
        self.assertEqual(X.shape[1], spec.n_features())
        self.assertEqual(names, spec.names())
        other = G.spec_from_name("F1+win")
        with self.assertRaises(FileNotFoundError):
            G.read_feature_cache(self.cache, other, self.wells[0], "train")

    def test_assemble_with_spec_and_scaler_names(self):
        spec = G.spec_from_name("F2")
        G.build_feature_cache(self.cache, spec, self.wells, self.phys, split="train",
                              from_raw=lambda w: D.read_well_shard(self.cache, w, "train"))
        fit = RD.fit_scalers_from_wells(self.wells, self.cache, spec=spec)
        self.assertIn("phys_params", fit)
        self.assertEqual(set(fit["phys_params"].as_dict()) >= {"gr_min", "gr_max"}, True)
        t = RD.assemble(self.wells, self.cache, scaler=fit["scaler"], spec=spec)
        RD.assert_alignment(t)
        self.assertEqual(t.X.shape[1], spec.n_features())
        self.assertEqual(list(fit["scaler"].names), spec.names())
        payload = RD.scaler_payload({**fit, "fold": 0, "val_wells": []})
        self.assertEqual(payload["row_scaler"]["feature_names"], spec.names())
        self.assertIn("physics_params", payload)

    def test_f1_path_unchanged(self):
        """F1（spec=None）必须与历史行为逐位一致：32 列且列名是 F1 的名字。"""
        t = RD.assemble(self.wells, self.cache, scaler=None)
        self.assertEqual(t.X.shape[1], F.N_FEATURES)
        self.assertEqual(list(F.FEATURE_NAMES)[:3], ["GR", "PE", "SP"])

    def test_fold_phys_overrides_cached_global_phys(self):
        """H2：调用方给了折内 phys_params 时，必须现场构造而不是读全局缓存。"""
        spec = G.spec_from_name("F1+phys")
        G.build_feature_cache(self.cache, spec, self.wells, self.phys, split="train",
                              from_raw=lambda w: D.read_well_shard(self.cache, w, "train"))
        X_cached, _ = G.read_feature_cache(self.cache, spec, self.wells[0], "train")
        other = PH.PhysicsParams(gr_min=0.0, gr_max=1000.0, sp_min=0.0, sp_max=1000.0)
        X_fold, names = RD._well_feature_matrix(self.cache, self.wells[0], "train", spec,
                                                phys_params=other)
        self.assertEqual(list(names), spec.names())
        self.assertFalse(np.allclose(X_cached, X_fold),
                         "缓存里的全局 phys 分位数不得覆盖调用方传入的折内参数")


if __name__ == "__main__":
    unittest.main(verbosity=2)
