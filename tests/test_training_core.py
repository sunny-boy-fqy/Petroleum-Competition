"""训练核心层测试（**torch 门控**：无 torch 时整体 skip）。

覆盖：
  * `loop.lam1_at` 的线性退火；
  * `loop.run_training` 的"最优 epoch 权重写回"（`keep_best`）与真实评分早停、时间预算；
  * `losses.total_loss` 的 `s_por`/`s_sw` 通道（E1/P1 §5 步 4 的缺失修复）；
  * `checkpoint` 的 bf16 往返、manifest、滚动淘汰、可续训验证；
  * `metrics` 的原子门/占位行/连续切片口径；
  * `hardware.detect_accelerator` 驱动的设备解析（假 torch 桩）。
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import basic as F  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

if HAS_TORCH:
    import torch

    from src.models.row_mlp import build_model
    from src.training import checkpoint as CK
    from src.training import loop as L
    from src.training import metrics as M


def synth_fold(n_rows: int = 512, n_wells: int = 4, seed: int = 0) -> RD.FoldTensors:
    """构造与 `assemble` 同构的合成整折（可学习信号 + 占位行 + 边界 POR）。"""
    rng = np.random.default_rng(seed)
    per = n_rows // n_wells
    X = rng.normal(size=(per * n_wells, F.N_FEATURES)).astype("float32")
    # 让第 0 列成为强信号（模型能在少量 epoch 内学到）
    signal = X[:, 0]
    por = np.where(signal > 0, 8.0 + 4.0 * signal, 0.05).astype("float32")
    sw = np.clip(60.0 + 10.0 * signal, C.SW_VALID_MIN, 99.0).astype("float32")
    perm = np.power(10.0, -1.0 + 0.5 * signal).astype("float32")
    n = per * n_wells
    mask = np.ones((n, 3), dtype="float32")
    atom = rng.random(n) < 0.3
    por[atom] = C.ATOM_VALUES["POR"]
    perm[atom] = C.ATOM_VALUES["PERM"]
    sw[atom] = C.ATOM_VALUES["SW"]
    mask[rng.random(n) < 0.05] = 0.0
    y_atom = np.stack([np.isclose(por, C.ATOM_VALUES[t]) & (mask[:, i] > 0)
                       for i, t in enumerate(C.TARGETS)], axis=1).astype("float32")
    y_joint = (y_atom.all(axis=1)).astype("float32")
    well_index = np.repeat(np.arange(n_wells), per).astype("int32")
    offsets = np.arange(0, n + 1, per, dtype="int64")
    return RD.FoldTensors(
        X=X, depth=(1000.0 + 0.1 * np.arange(n)).astype("float32"),
        well_index=well_index, well_ids=tuple(f"w{i}" for i in range(n_wells)),
        offsets=offsets, y_por=por,
        y_perm_z=np.log10(np.maximum(perm, 1e-12)).astype("float32"), y_sw=sw,
        mask=mask, y_atom=y_atom, y_joint=y_joint)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestLam1Schedule(unittest.TestCase):
    def test_linear_decay_then_floor(self):
        cfg = L.TrainConfig(epochs=10, lam1_start=1.0, lam1_end=0.1, lam1_frac=0.6)
        self.assertAlmostEqual(L.lam1_at(0, 10, cfg), 1.0, places=6)
        self.assertAlmostEqual(L.lam1_at(6, 10, cfg), 0.1, places=6)
        self.assertAlmostEqual(L.lam1_at(9, 10, cfg), 0.1, places=6)
        mid = L.lam1_at(3, 10, cfg)
        self.assertTrue(0.1 < mid < 1.0)
        # 单调不增
        vals = [L.lam1_at(e, 10, cfg) for e in range(10)]
        self.assertEqual(vals, sorted(vals, reverse=True))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestRunTraining(unittest.TestCase):
    def _data(self, **kw):
        t = synth_fold(**kw)
        dev = torch.device("cpu")
        return t, L.TorchFold(t, dev)

    def test_keep_best_restores_best_epoch_weights(self):
        """eval_fn 给的分数在 epoch1 达峰 -> 训练结束时权重必须是 epoch1 的。"""
        t, data = self._data(n_rows=256)
        cfg = L.TrainConfig(epochs=4, batch_size=128, patience=99, seed=0, hidden=16,
                            layers=1, dropout=0.0, amp_dtype="fp32")
        model = build_model(n_features=F.N_FEATURES, hidden=16, layers=1, dropout=0.0,
                            seed=0)
        snapshots: dict[int, dict] = {}

        def eval_fn(m):
            return {"total": 10.0 if len(snapshots) == 1 else 1.0}

        def on_epoch(e, rec):
            snapshots[e] = {k: v.clone() for k, v in model.state_dict().items()}

        hist = L.run_training(model, data, cfg, eval_fn=eval_fn, on_epoch=on_epoch,
                              keep_best=True)
        self.assertEqual(hist["best_epoch"], 1)
        self.assertTrue(hist["best_restored"])
        after = model.state_dict()
        for k, v in snapshots[1].items():
            self.assertTrue(torch.allclose(after[k], v, atol=0, rtol=0),
                            f"参数 {k} 未回到 best epoch 的值")

    def test_no_eval_means_full_epochs_and_finite_loss(self):
        t, data = self._data(n_rows=256)
        cfg = L.TrainConfig(epochs=2, batch_size=64, patience=99, seed=0, hidden=16,
                            layers=1, dropout=0.0, amp_dtype="fp32")
        model = build_model(n_features=F.N_FEATURES, hidden=16, layers=1, dropout=0.0, seed=0)
        hist = L.run_training(model, data, cfg, eval_fn=None, keep_best=False)
        self.assertEqual(hist["n_epochs_run"], 2)
        self.assertEqual(hist["stopped_reason"], "completed")
        for rec in hist["epochs"]:
            self.assertTrue(np.isfinite(rec["total"]))
            self.assertGreaterEqual(rec["disk_free_gb"], 0.0)

    def test_time_budget_stops_gracefully(self):
        t, data = self._data(n_rows=256)
        cfg = L.TrainConfig(epochs=100, batch_size=64, patience=99, seed=0, hidden=16,
                            layers=1, dropout=0.0, amp_dtype="fp32", time_budget_h=0.0)
        model = build_model(n_features=F.N_FEATURES, hidden=16, layers=1, dropout=0.0, seed=0)
        hist = L.run_training(model, data, cfg, eval_fn=None, keep_best=False)
        self.assertEqual(hist["stopped_reason"], "time_budget")
        self.assertEqual(hist["n_epochs_run"], 0)

    def test_resume_epoch_and_save_hook_are_wired(self):
        """H3：resume_epoch 必须真的跳过已完成 epoch，save_hook 必须每 epoch 调用。"""
        t, data = self._data(n_rows=256)
        cfg = L.TrainConfig(epochs=5, batch_size=64, patience=99, seed=0, hidden=16,
                            layers=1, dropout=0.0, amp_dtype="fp32")
        model = build_model(n_features=F.N_FEATURES, hidden=16, layers=1, dropout=0.0, seed=0)
        calls: list[int] = []

        def hook(epoch, rec, model_, opt_, sched_):
            calls.append(int(epoch))

        hist = L.run_training(model, data, cfg, eval_fn=None, keep_best=False,
                              resume_epoch=2, save_hook=hook)
        self.assertEqual(calls, [3, 4])
        self.assertEqual([r["epoch"] for r in hist["epochs"]], [3, 4])
        self.assertEqual(hist["n_epochs_run"], 2)

    def test_early_stop_on_real_score(self):
        t, data = self._data(n_rows=256)
        cfg = L.TrainConfig(epochs=50, batch_size=64, patience=2, seed=0, hidden=16,
                            layers=1, dropout=0.0, amp_dtype="fp32")
        model = build_model(n_features=F.N_FEATURES, hidden=16, layers=1, dropout=0.0, seed=0)
        calls = {"n": 0}

        def eval_fn(m):
            calls["n"] += 1
            return {"total": 100.0 - calls["n"]}      # 单调下降 -> 触发早停

        hist = L.run_training(model, data, cfg, eval_fn=eval_fn, keep_best=True)
        self.assertEqual(hist["stopped_reason"], "early_stop")
        self.assertEqual(hist["best_epoch"], 0)
        self.assertLessEqual(hist["n_epochs_run"], 4)

    def test_learning_reduces_loss_on_learnable_signal(self):
        """5000 行级别的过拟合测试（E1/P1 §9 的"学不动就查损失"判据）。"""
        t, data = self._data(n_rows=2048, n_wells=8)
        cfg = L.TrainConfig(epochs=6, batch_size=256, patience=99, seed=0, hidden=64,
                            layers=2, dropout=0.0, amp_dtype="fp32")
        model = build_model(n_features=F.N_FEATURES, hidden=64, layers=2, dropout=0.0, seed=0)
        y_true = M.label_scale_stack(t.y_por, t.y_perm_z, t.y_sw)

        def eval_fn(m):
            pred = L.predict_torch(m, data, cfg)
            return {"total": M.score_of(y_true, M.decode_continuous(pred), t.mask)["total"]}

        first = float(eval_fn(model)["total"])
        hist = L.run_training(model, data, cfg, eval_fn=eval_fn, keep_best=True,
                              scaler_params={"s_por": 4.0, "s_sw": 10.0})
        after = float(hist["best_total"])
        self.assertGreater(after, first + 1.0,
                           f"6 个 epoch 后真实分数应上升（{first:.2f} -> {after}）")
        self.assertTrue(all(np.isfinite(r["total"]) for r in hist["epochs"]))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTotalLossScalerPlumbing(unittest.TestCase):
    """E1/P1 §5 步 4：`L_aux` 必须能接收训练折的 `s_por`/`s_sw`。"""

    def _batch(self, out, n=64):
        t = synth_fold(n_rows=n, n_wells=2)
        return {
            "por": torch.from_numpy(t.y_por), "perm_z": torch.from_numpy(t.y_perm_z),
            "sw": torch.from_numpy(t.y_sw), "mask": torch.from_numpy(t.mask),
            "y_atom": torch.from_numpy(t.y_atom), "y_joint": torch.from_numpy(t.y_joint),
        }

    def _model_out(self, n=64):
        m = build_model(n_features=F.N_FEATURES, hidden=16, layers=1, dropout=0.0, seed=0)
        x = torch.from_numpy(synth_fold(n_rows=n, n_wells=2).X)
        return m(x)

    def test_s_sw_changes_aux_term(self):
        from src.losses.score_aligned import total_loss
        out, batch = self._model_out(), self._batch(None)
        _, p_small = total_loss(out, batch, s_por=4.0, s_sw=1.0, use_align=True, use_aux=True)
        _, p_big = total_loss(out, batch, s_por=4.0, s_sw=1000.0, use_align=True, use_aux=True)
        self.assertNotAlmostEqual(p_small["aux"], p_big["aux"], places=6,
                                  msg="s_sw 必须真的影响 L_aux（此前被 total_loss 丢弃）")

    def test_defaults_match_legacy_values(self):
        from src.losses.score_aligned import aux_loss, total_loss
        out, batch = self._model_out(), self._batch(None)
        _, parts = total_loss(out, batch, use_align=False, use_aux=True)
        ref = float(aux_loss(batch["por"], out["por"], batch["perm_z"], out["perm_z"],
                             batch["sw"], out["sw"], batch["mask"], s_por=11.34, s_sw=20.0))
        self.assertAlmostEqual(parts["aux"], ref, places=6)

    def test_huber_beta_is_forwarded(self):
        from src.losses.score_aligned import total_loss
        out, batch = self._model_out(), self._batch(None)
        _, a = total_loss(out, batch, use_align=False, s_por=4.0, s_sw=10.0, huber_beta=0.1)
        _, b = total_loss(out, batch, use_align=False, s_por=4.0, s_sw=10.0, huber_beta=5.0)
        self.assertNotAlmostEqual(a["aux"], b["aux"], places=6)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestCheckpoint(unittest.TestCase):
    def _model(self):
        return build_model(n_features=F.N_FEATURES, hidden=16, layers=1, dropout=0.0, seed=1)

    def test_roundtrip_bf16_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best.pt"
            m = self._model()
            CK.save_checkpoint(p, m, meta={"stage": "E1", "fold": 0, "epoch": 3,
                                          "target_scalers": {"sw_mu": 80.0},
                                          "model": {"n_features": F.N_FEATURES, "hidden": 16,
                                                    "layers": 1, "dropout": 0.0}},
                               extra={"row_scaler": {"median": [0.0] * F.N_FEATURES}})
            man = CK.read_manifest(p)
            self.assertEqual(man["dtype"], "bfloat16")
            self.assertEqual(man["fold"], 0)
            self.assertGreater(man["n_params"], 0)
            # 读回一个新模型，参数必须逐张量一致（bf16 量化误差内）
            m2 = self._model()
            CK.load_checkpoint(p, model=m2)
            for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
                self.assertEqual(k1, k2)
                # bf16 只有 ~8 位尾数（39.8 -> 39.75），因此按 bf16 精度比较
                self.assertTrue(torch.allclose(v1.float(), v2.float(), atol=0.25, rtol=1e-2),
                                f"{k1} 载入后不一致")

    def test_verify_resumable_and_resume_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "last.pt"
            m = self._model()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            CK.save_checkpoint(p, m, meta={"epoch": 7}, optimizer=opt)
            info = CK.verify_resumable(p, self._model)
            self.assertTrue(info["ok"])
            m2 = self._model()
            opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
            rr = CK.load_for_resume(p, m2, optimizer=opt2)
            self.assertEqual(rr["epoch"], 7)
            self.assertTrue(rr["optimizer_restored"])

    def test_rotate_keeps_only_three(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            m = self._model()
            for name in ("best.pt", "last.pt", "last_prev.pt", "epoch5.pt", "epoch6.pt"):
                CK.save_checkpoint(d / name, m)
            removed = CK.rotate(d)
            left = sorted(f.name for f in d.glob("*.pt"))
            self.assertEqual(left, ["best.pt", "last.pt", "last_prev.pt"])
            self.assertIn("epoch5.pt", removed)

    def test_scheduler_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "last.pt"
            m = self._model()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
            sched.step()
            CK.save_checkpoint(p, m, meta={"epoch": 4}, optimizer=opt, scheduler=sched)
            m2 = self._model()
            opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
            rr = CK.load_for_resume(p, m2, optimizer=opt2, scheduler=sched2)
            self.assertEqual(rr["epoch"], 4)
            self.assertTrue(rr["scheduler_restored"])
            self.assertAlmostEqual(float(sched2.last_epoch), float(sched.last_epoch))

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                CK.load_checkpoint(Path(td) / "nope.pt")
            with self.assertRaises(FileNotFoundError):
                CK.read_manifest(Path(td) / "nope.pt")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestMetricsDecoding(unittest.TestCase):
    def test_sw_is_never_clipped_to_unit_scale(self):
        pred = {"por": np.array([5.0]), "perm_z": np.array([0.0]),
                "sw": np.array([82.0]), "q_atom": np.zeros((1, 3)), "q_joint": np.zeros(1)}
        cont = M.decode_continuous(pred)
        self.assertAlmostEqual(float(cont[0, 2]), 82.0, places=6)
        # SW 低于 1 的输入只做 [0,100] 软保护，不会被压到 [0,1]
        pred2 = dict(pred, sw=np.array([0.8]))
        self.assertAlmostEqual(float(M.decode_continuous(pred2)[0, 2]), 0.8, places=6)

    def test_perm_is_always_positive(self):
        pred = {"por": np.array([1.0]), "perm_z": np.array([-30.0]),
                "sw": np.array([50.0]), "q_atom": np.zeros((1, 3)), "q_joint": np.zeros(1)}
        self.assertGreater(float(M.decode_continuous(pred)[0, 1]), 0.0)

    def test_atom_gate_replaces_exactly_without_interpolation(self):
        cont = np.array([[9.0, 0.5, 50.0]])
        q = np.array([[0.99, 0.99, 0.99]])
        g = M.atom_gate(cont, q, tau=[0.5, 0.5, 0.5])
        np.testing.assert_allclose(g[0], [C.ATOM_VALUES["POR"], C.ATOM_VALUES["PERM"],
                                          C.ATOM_VALUES["SW"]])
        g2 = M.atom_gate(cont, q, tau=[0.999, 0.999, 0.999])
        np.testing.assert_allclose(g2, cont)          # 未过阈值 -> 完全等于连续头

    def test_placeholder_report_only_counts_atomic_rows(self):
        y_true = np.array([[0.1, 0.01, 99.9], [7.0, 1.0, 60.0]])
        y_pred = np.array([[0.1, 0.01, 99.9], [9.9, 0.0, 12.0]])
        y_atom = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        mask = np.ones((2, 3), dtype=bool)
        rep = M.atomic_rows_report(y_true, y_pred, y_atom, mask)
        self.assertEqual(rep["rows"]["POR"], 1)
        self.assertAlmostEqual(rep["hit_rate"]["POR"], 1.0)
        self.assertAlmostEqual(M.placeholder_min_acc({"atomic_rows": rep}), 1.0)

    def test_label_scale_stack_percent_scale(self):
        out = M.label_scale_stack(np.array([1.0]), np.array([0.0]), np.array([50.0]))
        np.testing.assert_allclose(out[0], [1.0, 1.0, 50.0])

    def test_jsonable_handles_nan_and_numpy(self):
        got = M.jsonable({"a": np.float32(1.5), "b": float("nan"),
                          "c": np.array([1, 2]), "d": (np.int64(3),)})
        self.assertEqual(got["a"], 1.5)
        self.assertIsNone(got["b"])
        self.assertEqual(got["c"], [1, 2])
        self.assertEqual(got["d"], [3])


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestDeviceResolution(unittest.TestCase):
    """`resolve_device` 必须按 hardware 优先级走（NPU 优先），显式指定则原样使用。"""

    def test_explicit_device_is_respected(self):
        self.assertEqual(str(L.resolve_device(L.TrainConfig(device="cpu"))), "cpu")

    def test_auto_uses_hardware_detection(self):
        from src import hardware as HW
        cfg = L.TrainConfig(device="auto")
        got = str(L.resolve_device(cfg))
        kind = HW.detect_accelerator(torch)
        self.assertTrue(got.startswith(kind), f"auto 应解析为 {kind}*，实际 {got}")

    def test_amp_context_disabled_on_cpu(self):
        ctx = L.amp_context(L.TrainConfig(amp_dtype="bf16"), torch.device("cpu"))
        with ctx:
            a = torch.ones(4)
        self.assertEqual(a.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
