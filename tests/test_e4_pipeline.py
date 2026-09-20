"""E4/P0 端到端（**torch 门控**）：PatchTF 入口脚本真能跑出可复算的报告与体检。

用**合成井**跑完整链路：缓存 → 两阶段训练（inner 选 epoch/τ）→ OOF →
还原/拼缝体检 → 搜索表/消融（可选）→ 报告与 Gate 落盘。

断言的硬契约（E4/P0 §7）：
  * 报告键齐全（E4_patchtf.json / E4_param_budget.json / E4_gate.json）；
  * `reconstruction_check.ok is True`（patchify/unpatchify 逐点还原）；
  * `chunk_vs_full_recon.ok is True`（分块 vs 整井无拼缝跳变）；
  * 无 E3 基线时 **不伪造 delta**：`delta_vs_e3 is None`、`passed is None`、`nogo is True`；
  * `selection_score_only is True`（选择信号只来自 inner-OOF）；
  * 仓库的 `versions/candidates.json` **不被写**（smoke 不登记候选）。
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _synth_cache as SC  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402

REPO_CANDIDATES = V4 / "versions" / "candidates.json"


def _load_train_patchtf():
    spec = importlib.util.spec_from_file_location(
        "v4_e4_train_patchtf", str(V4 / "E4" / "code" / "train_patchtf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestE4Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        tr_wells, va_wells = SC.fold0_wells(n_val=16, n_train=16)
        cls.root = root
        cls.cache = root / "cache"
        SC.build_cache(cls.cache, tr_wells + va_wells, n_rows=40, seed=11)
        cls.mod = _load_train_patchtf()
        cls.reports = root / "reports"
        cls.runs = root / "runs"
        cls.scalers = root / "scalers"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _run(self, *extra: str) -> dict:
        argv = ["--smoke", "--folds", "0", "--max-wells", "8", "--epochs", "2",
                "--d-model", "16", "--n-layers", "1", "--n-heads", "4",
                "--chunk", "96", "--overlap", "32",
                "--cache-root", str(self.cache), "--reports-dir", str(self.reports),
                "--run-root", str(self.runs), "--scalers-dir", str(self.scalers), *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.mod.main(argv)
        self.assertEqual(rc, 0, buf.getvalue()[-2000:])
        return json.loads((self.reports / "E4_gate.json").read_text(encoding="utf-8"))

    def test_torch_arch_is_registered(self):
        from src.training.seq_loop import build_seq_model
        m = build_seq_model("patchtf", 8, patch_len=16, stride=8, d_model=16, n_layers=1,
                            n_heads=4, max_tokens=32)
        self.assertEqual(m.summary()["attn_api"], "sdpa")

    def test_smoke_pipeline_reports_and_contracts(self):
        before = _sha(REPO_CANDIDATES) if REPO_CANDIDATES.is_file() else None
        gate = self._run()
        self.assertEqual(gate["stage"], "E4")
        self.assertTrue(gate["reconstruction_check"]["ok"])
        self.assertTrue(gate["chunk_vs_full_recon"]["ok"])
        self.assertIsNone(gate["passed"])
        self.assertTrue(gate["nogo"])
        for name in ("E4_patchtf.json", "E4_param_budget.json", "E4_gate.json",
                     "training_time_log.json"):
            self.assertTrue((self.reports / name).is_file(), name)
        metrics = json.loads((self.reports / "E4_patchtf.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["selection_score_only"])
        self.assertTrue(metrics["reconstruction_check"]["ok"])
        self.assertTrue(metrics["chunk_vs_full_recon"]["ok"])
        self.assertIsNone(metrics["delta_vs_e3"])
        self.assertIsNone(metrics["paired_ci_vs_e3"])
        self.assertTrue(metrics["placeholder_rows"]["hit_rate"])
        self.assertGreater(metrics["model"]["n_params"], 0)
        budget = json.loads((self.reports / "E4_param_budget.json").read_text(encoding="utf-8"))
        self.assertIn("patchtf", budget)
        self.assertEqual(gate["gate_id"], "E4_P0_gate")
        self.assertTrue(gate["checks"]["contract_ok"])
        self.assertTrue(gate["checks"]["no_label_leak"])
        self.assertTrue(gate["checks"]["length_contract_ok"])
        self.assertFalse(gate["checks"]["baseline_available"])
        if before is not None:
            self.assertEqual(_sha(REPO_CANDIDATES), before, "smoke 不得写仓库 candidates.json")

    def test_search_uses_inner_only_and_records_table(self):
        search = self.root / "search.json"
        search.write_text(json.dumps([
            {"patch_len": 16, "stride": 8, "d_model": 16, "n_layers": 1, "n_heads": 4},
            {"patch_len": 32, "stride": 16, "d_model": 16, "n_layers": 1, "n_heads": 4},
        ]), encoding="utf-8")
        self._run("--search", str(search), "--search-folds", "0"
                  , "--channel-independence-ablation", "--rel-pos-ablation")
        metrics = json.loads((self.reports / "E4_patchtf.json").read_text(encoding="utf-8"))
        self.assertEqual(len(metrics["search_table"]), 2)
        for row in metrics["search_table"]:
            self.assertTrue(row["exploratory"] and row["selection_score_only"])
            self.assertIsInstance(row["inner_oof_total"], float)
        self.assertIsNotNone(metrics["channel_independence_ablation"])
        self.assertEqual(len(metrics["channel_independence_ablation"]), 2)
        self.assertEqual(len(metrics["rel_pos_ablation"]), 2)
        gate = json.loads((self.reports / "E4_gate.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["checks"]["ci_and_rel_pos_ablated"])


class TestE4ScriptContract(unittest.TestCase):
    """口径层（不需 torch）：入口脚本与 `run_train.sh` 的接线必须存在。"""

    def test_run_train_dispatches_e4(self):
        src = (V4 / "run_train.sh").read_text(encoding="utf-8")
        self.assertIn("E4) python3", src)
        self.assertIn("E4/code/train_patchtf.py", src)

    def test_help_lists_e4_knobs(self):
        import subprocess
        out = subprocess.run([sys.executable, str(V4 / "E4" / "code" / "train_patchtf.py"),
                              "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        for flag in ("--patch-len", "--stride", "--d-model", "--n-layers", "--n-heads",
                     "--channel-independent", "--no-channel-independent", "--rel-pos",
                     "--search", "--channel-independence-ablation", "--rel-pos-ablation",
                     "--capacity-ablation", "--e3-oof", "--seam-tol"):
            self.assertIn(flag, out.stdout, flag)


if __name__ == "__main__":
    unittest.main()
