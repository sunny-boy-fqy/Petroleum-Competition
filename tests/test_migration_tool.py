"""旧 checkpoint 迁移工具测试。

覆盖：torch_npu 旧格式警告对应的 ``tools/migrate_checkpoints.py``。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.portability import HAS_TORCH  # noqa: E402

if HAS_TORCH:
    import torch


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestMigrateCheckpoints(unittest.TestCase):
    def test_migrates_old_and_skips_already_format2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "E3"
            root.mkdir(parents=True)
            p = root / "last.pt"
            torch.save({"state_dict": {"w": torch.zeros(2)}, "dtype": "float32",
                        "format": 1}, p)
            mp = p.with_suffix(".manifest.json")
            mp.write_text(json.dumps({"epoch": 7, "torch": "2.7.0"}), encoding="utf-8")

            r = subprocess.run(
                [sys.executable, str(V4 / "tools" / "migrate_checkpoints.py"),
                 "--root", str(root)],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            payload = torch.load(p, map_location="cpu", weights_only=False)
            self.assertEqual(payload["state_dict"]["w"].shape, (2,))
            man = json.loads(mp.read_text(encoding="utf-8"))
            self.assertEqual(int(man["checkpoint_format"]), 2)
            self.assertEqual(int(man["epoch"]), 7)

            # 已迁移文件第二次必须跳过，不再重写。
            r2 = subprocess.run(
                [sys.executable, str(V4 / "tools" / "migrate_checkpoints.py"),
                 "--root", str(root)],
                capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
            self.assertIn("skipped=1", r2.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
