"""E3/P0 序列数据集与拼接测试（numpy 部分无 torch 也能跑；torch 部分门控）。

覆盖 E3/P0 §7 的四条完成判据：
  * `chunk 数 × 长度 ≈ 井长`（覆盖完整、无丢点，含末尾不足一窗的井）；
  * 接缝：整段 vs 分块加权拼接的逐点差 < 容差，常数列拼接后仍是常数（无跳变）；
  * 固定 seed 下采样顺序可复现；不同 epoch 顺序不同；
  * chunk 边界处 (x, y, mask) 逐行对齐；worker 常驻内存 < 300 MB 自检生效。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _synth_cache as SC  # noqa: E402
from src.data import seq_dataset as SD  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

if HAS_TORCH:
    import torch

    from src.training import loop as L
    from src.training import seq_loop as SL


class TestChunkPlanning(unittest.TestCase):
    def test_coverage_complete_for_all_shapes(self):
        for n in (1, 7, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 4096):
            for chunk, ov in ((64, 0), (64, 8), (64, 32), (1024, 128)):
                rep = SD.coverage_report(n, SD.chunks_for(n, chunk, ov))
                self.assertTrue(rep["complete"], f"n={n} chunk={chunk} ov={ov} -> {rep}")
                self.assertEqual(rep["uncovered"], 0)

    def test_chunks_are_fixed_length(self):
        chunks = SD.chunks_for(3000, 1024, 128)
        self.assertTrue(all(L == 1024 for _s, L in chunks))
        self.assertGreater(chunks[-1][0] + chunks[-1][1], 3000 - 1)

    def test_short_well_single_chunk(self):
        self.assertEqual(SD.chunks_for(50, 1024, 128), [(0, 50)])
        self.assertEqual(SD.chunks_for(0), [])

    def test_bad_params_rejected(self):
        with self.assertRaises(ValueError):
            SD.chunks_for(100, 64, 64)
        with self.assertRaises(ValueError):
            SD.chunks_for(100, 0, 0)


class TestStitching(unittest.TestCase):
    def test_stitch_reproduces_whole_segment(self):
        n, chunk, ov = 1000, 256, 64
        whole = np.random.default_rng(0).normal(size=(n, 3))
        chunks = SD.chunks_for(n, chunk, ov)
        st = SD.stitch_chunks(chunks, [whole[s:s + L] for s, L in chunks], n)
        rep = SD.seam_report(chunks, st, whole, tol=1e-6)
        self.assertTrue(rep["within_tolerance"], rep)
        self.assertLess(rep["max_abs_diff"], 1e-9)

    def test_constant_is_preserved(self):
        n, chunk, ov = 500, 128, 32
        chunks = SD.chunks_for(n, chunk, ov)
        const = np.full((n, 2), 7.0)
        st = SD.stitch_chunks(chunks, [const[s:s + L] for s, L in chunks], n)
        np.testing.assert_allclose(st, 7.0, atol=1e-9)

    def test_weights_have_zero_ends_but_stitch_is_safe(self):
        w = SD.chunk_weights(5, "triangular")
        self.assertEqual(float(w[0]), 0.0)
        self.assertEqual(float(w[-1]), 0.0)
        self.assertGreater(float(w[2]), float(w[1]))
        # 权重为 0 的端点由 min_weight 兜底 -> 覆盖仍完整
        chunks = [(0, 5)]
        out = SD.stitch_chunks(chunks, [np.ones((5, 1))], 5)
        np.testing.assert_allclose(out, 1.0)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            SD.stitch_chunks([(0, 4)], [np.zeros((3, 1))], 4)
        with self.assertRaises(ValueError):
            SD.stitch_chunks([(0, 4)], [], 4)
        with self.assertRaises(ValueError):
            SD.chunk_weights(4, "bogus")

    def test_worker_memory_budget(self):
        rss = SD.WellShardReader.rss_mb()
        self.assertGreater(rss, 0.0)
        self.assertLess(rss, 4000.0)          # 测试进程本身不该异常膨胀
        with self.assertRaises(MemoryError):
            SD.WellShardReader.assert_worker_budget(limit_mb=0.001)
        # 基线机制：把当前 RSS 当基线时，增量 ≈ 0 -> 不报错
        SD.WellShardReader.assert_worker_budget(limit_mb=1.0,
                                                baseline_mb=SD.WellShardReader.rss_mb())


class TestChunkWells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.wells = [f"w{i}" for i in range(4)]
        cls.root = Path(cls._td.name)
        _tr, _te, cls.cache = SC.build_cache(cls.root / "cache", cls.wells, n_rows=300, seed=4)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_reader_chunk_alignment_with_labels(self):
        rd = SD.WellShardReader(self.cache, split="train")
        X = rd.feature_matrix(self.wells[0])
        lab = rd.labels(self.wells[0])
        for start, length in SD.chunks_for(X.shape[0], 64, 16):
            rec = rd.chunk(self.wells[0], start, length)
            self.assertEqual(rec["X"].shape[0], length)
            np.testing.assert_allclose(rec["X"], X[start:start + length])
            np.testing.assert_allclose(rec["por"], lab["por"][start:start + length])
            np.testing.assert_allclose(rec["mask"], lab["mask"][start:start + length])
            np.testing.assert_allclose(rec["y_atom"], lab["y_atom"][start:start + length])

    def test_reader_does_not_cache_shards(self):
        """每次取数都重新打开文件：reader 上除了显式字段不应留下整井矩阵。"""
        rd = SD.WellShardReader(self.cache, split="train")
        rd.chunk(self.wells[0], 0, 32)
        big = [k for k, v in vars(rd).items()
               if isinstance(v, np.ndarray) and v.size > 10_000]
        self.assertEqual(big, [], f"reader 里出现了大数组（疑似缓存）：{big}")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestChunkDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.wells = [f"w{i}" for i in range(3)]
        _tr, _te, cls.cache = SC.build_cache(Path(cls._td.name) / "cache", cls.wells,
                                             n_rows=200, seed=6)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_coverage_and_items(self):
        ds = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=1)
        cov = ds.coverage()
        self.assertTrue(cov["complete"])
        self.assertEqual(cov["total_rows"], 200 * len(self.wells))
        self.assertEqual(len(ds), cov["n_chunks"])
        item = ds[0]
        self.assertEqual(item["X"].shape[1], 32)
        for k in ("por", "perm_z", "sw", "mask", "y_atom", "y_joint"):
            self.assertIn(k, item)
        self.assertEqual(item["mask"].shape, (item["X"].shape[0], 3))
        # 300 MB 是**每 worker 的增量**预算；父进程含 torch 本身（~数百 MB），
        # 因此这里显式传基线，并另测"超限必须报错"。
        base = SD.WellShardReader.rss_mb()
        self.assertLessEqual(SD.WellShardReader.assert_worker_budget(baseline_mb=base), 4000.0)

    def test_order_reproducible_and_epoch_dependent(self):
        a = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=7, epoch=0)
        b = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=7, epoch=0)
        c = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=7, epoch=1)
        self.assertEqual(a.order_digest(), b.order_digest(), "同 seed/epoch 必须可复现")
        self.assertNotEqual(a.order_digest(), c.order_digest(), "不同 epoch 应重新洗牌")

    def test_set_epoch_rebuilds_order(self):
        ds = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=7, epoch=0)
        d0 = ds.order_digest()
        ds.set_epoch(1)
        self.assertNotEqual(ds.order_digest(), d0)
        ds.set_epoch(0)
        self.assertEqual(ds.order_digest(), d0, "回到同 epoch 必须与初始 order 一致")

    def test_scaler_is_applied_inside_dataset(self):
        from src.data import row_dataset as RD
        ds_plain = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=0)
        X0 = ds_plain[0]["X"].numpy()
        scaler = RD.RowScaler.fit(np.concatenate(
            [ds_plain.reader.feature_matrix(w) for w in self.wells]), wells=tuple(self.wells))
        ds_scaled = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=0,
                                       scaler=scaler)
        X1 = ds_scaled[0]["X"].numpy()
        self.assertFalse(np.allclose(X0, X1), "scaler 必须真的生效")
        # 多次构造不得互相污染（早期版本 patch 类，会重复包装）
        ds_again = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=0)
        np.testing.assert_allclose(ds_again[0]["X"].numpy(), X0, rtol=1e-6)

    def test_preloaded_dataset_matches_disk_dataset(self):
        """E3 优化：预加载整折后逐 chunk 切片，必须与旧磁盘 Dataset 逐值一致。"""
        from src.data import row_dataset as RD
        fit = RD.fit_scalers_from_wells(self.wells, self.cache, return_tensors=True)
        scaler = fit["scaler"]
        tensors = fit["tensors"]
        tensors.X = scaler.transform_blocked(tensors.X)
        pre = SD.PreloadedSeqDataset(tensors, self.wells, chunk=64, overlap=16, seed=0)
        disk = SD.SeqChunkDataset(self.cache, self.wells, chunk=64, overlap=16, seed=0,
                                  scaler=scaler)
        self.assertEqual(pre.order_digest(), disk.order_digest())
        self.assertEqual(pre.coverage(), disk.coverage())
        for i in range(len(pre)):
            a, b = pre[i], disk[i]
            np.testing.assert_allclose(a["X"].numpy(), b["X"].numpy(), rtol=0.0, atol=0.0)
            for key in ("por", "perm_z", "sw", "mask", "y_atom", "y_joint"):
                np.testing.assert_array_equal(a[key].numpy(), b[key].numpy())


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSeqFoldSmoke(unittest.TestCase):
    """E3 的两阶段序列单折（小规模合成井）：保长、可分块拼接、checkpoint 语义。"""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        tr, va = SC.fold0_wells(n_val=4, n_train=6)
        cls.wells = tr + va
        _tr, _te, cls.cache = SC.build_cache(Path(cls._td.name) / "cache", cls.wells,
                                             n_rows=200, seed=8)
        cls.folds = {"n_folds": 1, "well_list": cls.wells,
                     "fold_of_well": {**{w: 0 for w in va}, **{w: 1 for w in tr}}}

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, arch, tag, epochs=2, resume=False):
        cfg = L.TrainConfig(epochs=epochs, patience=9, seed=0, dropout=0.0, amp_dtype="fp32",
                            device="cpu", min_free_gb=0.0, batch_size=128)
        opt = SL.SeqOptions(chunk=64, overlap=16, batch_chunks=2, smoke=True, arch=arch,
                            save_checkpoints=True, select_tau=False, resume=resume,
                            run_dir=Path(self._td.name) / tag,
                            scalers_dir=Path(self._td.name) / tag / "scalers")
        kw = ({"base_ch": 8, "depth": 2} if arch == "unet"
              else {"channels": 12, "n_blocks": 2, "dilation_max": 4})
        return SL.run_two_phase_seq_fold(0, self.folds, self.cache, cfg, opt, arch_kwargs=kw)

    def test_unet_fold_produces_full_length_output(self):
        r = self._run("unet", "unet")
        for w in r.va_wells:
            n = int(r.pred[w]["por"].shape[0])
            self.assertEqual(n, 200, f"{w}: 逐行输出长度必须等于井长")
            self.assertEqual(r.pred[w]["q_atom"].shape, (200, 3))
        self.assertTrue(np.isfinite(r.pred[r.va_wells[0]]["por"]).all())
        self.assertTrue(r.resumable["ok"], "checkpoint 必须可读回")
        select = Path(self._td.name) / "unet" / "fold0" / "select"
        self.assertTrue((select / "last.pt").is_file(), "阶段 1 必须每 epoch 落 last.pt")
        self.assertTrue((select / "best.pt").is_file(), "阶段 1 必须落 best.pt")
        from src.training import checkpoint as CK
        man = CK.read_manifest(select / "last.pt")
        self.assertEqual(int(man.get("data_pipeline_rev", -1)), SL.SEQ_DATA_PIPELINE_REV)
        self.assertGreater(r.coverage["n_chunks"], 0)

    def test_tcn_fold_produces_full_length_output(self):
        r = self._run("tcn", "tcn")
        for w in r.va_wells:
            self.assertEqual(int(r.pred[w]["por"].shape[0]), 200)
        self.assertTrue(np.isfinite(r.pred[r.va_wells[0]]["sw"]).all())

    def test_pause_flag_stops_before_outer_outputs(self):
        flag = Path(self._td.name) / "pause.flag"
        flag.write_text("1", encoding="utf-8")
        old = os.environ.get("V4_PAUSE_FLAG")
        os.environ["V4_PAUSE_FLAG"] = str(flag)
        try:
            with self.assertRaises(L.TrainingPaused):
                self._run("unet", "paused")
        finally:
            if old is None:
                os.environ.pop("V4_PAUSE_FLAG", None)
            else:
                os.environ["V4_PAUSE_FLAG"] = old

    def test_stage1_resume_skips_completed_epochs(self):
        r1 = self._run("unet", "resume", epochs=2, resume=False)
        self.assertGreaterEqual(r1.hist1["n_epochs_run"], 1)
        r2 = self._run("unet", "resume", epochs=3, resume=True)
        self.assertEqual(r2.hist1["n_epochs_run"], 1,
                         "阶段 1 resume 后只应跑尚未完成的 epoch 2")

    def test_no_label_leak_in_seq_fold(self):
        r = self._run("tcn", "leak")
        self.assertEqual(set(r.tr_wells) & set(r.va_wells), set())
        self.assertTrue(set(r.fit_wells).issubset(set(r.tr_wells)))
        self.assertTrue(set(r.inner_val_wells).issubset(set(r.tr_wells)))
        self.assertEqual(set(r.inner_val_wells) & set(r.va_wells), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
