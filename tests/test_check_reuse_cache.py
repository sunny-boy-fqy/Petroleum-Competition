"""检查 /data 复用缓存的脚本测试。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
TOOL = V4 / "tools" / "check_reuse_cache.py"


def _run(root: Path):
    return subprocess.run([sys.executable, str(TOOL), "--remote-root", str(root)],
                          capture_output=True, text=True)


class TestCheckReuseCache(unittest.TestCase):
    def _make_complete(self, root: Path) -> None:
        art = root / "v4" / "artifacts"
        mir = root / "v4" / "mirror"
        st = root / "v4" / "state"
        for split, n in (("train", 80), ("test", 10)):
            d = art / "cache" / "raw" / split
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / f"w{i}.npz").write_bytes(b"x")
        feat = art / "cache" / "feat" / "FX_win_w11-51-201_smsmmtc" / "train"
        feat.mkdir(parents=True, exist_ok=True)
        (feat / "w0.npz").write_bytes(b"f")
        e1 = mir / "run_root" / "E1"
        e1.mkdir(parents=True, exist_ok=True)
        (e1 / "oof.npz").write_bytes(b"o")
        (e1 / "last.pt").write_bytes(b"p")
        reps = mir / "reports"
        (mir / "run_root" / "E2").mkdir(parents=True, exist_ok=True)
        (reps / "E2_work").mkdir(parents=True, exist_ok=True)
        for name in ("E0_data_card.json", "E0_folds.json", "E0_local_contract_gate.json",
                     "E1_gate.json", "E2_ablation.json", "E2_gate.json", "E2_best_spec.json"):
            (reps / name).write_text("{}", encoding="utf-8")
        (mir / "run_root" / "E2" / "oof_FX_win_w11-51-201_smsmmtc.npz").write_bytes(b"o")
        st.mkdir(parents=True, exist_ok=True)
        (st / "all_pipeline_progress.json").write_text(json.dumps({
            "tasks": {str(i): {"status": "done"} for i in range(1, 6)},
        }), encoding="utf-8")

    def test_complete_fixture_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_complete(root)
            r = _run(root)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("RESULT: OK", r.stdout)

    def test_missing_fixture_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(Path(td))
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("RESULT: INCOMPLETE", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
