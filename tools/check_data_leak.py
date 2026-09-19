#!/usr/bin/env python3
"""全量输入-标签泄漏回归（90 口井）。

背景（审查 B1）：`parse.py` 曾用 `inputs = arr[:, 1:15]`，把索引 14 的 **POR 标签**
当成第 14 个输入，80 口训练井全部泄漏。修复后必须有全量回归防止再次发生。

检查项（逐井）：
  1. `inputs.shape[1] == 13`（13 条曲线；DEPTH 单独作为深度通道）
  2. 任何输入列与任何目标列**不完全相等**（atol=1e-9，比例 > 0.999 即判违规）
  3. 训练井目标列存在；测试井 `inputs` 与训练井列数一致

    python3 v4/tools/check_data_leak.py [--data ../data]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C               # noqa: E402
from src.data import labels as L             # noqa: E402
from src.data import parse as P              # noqa: E402

ATOL = 1e-9
RATIO_THRESHOLD = 0.999


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(V4.parent / "data"))
    ap.add_argument("--min-rows", type=int, default=20,
                    help="目标非缺测行数少于此值时跳过该目标的等值比较")
    args = ap.parse_args()

    root = Path(args.data)
    violations: list[str] = []
    checked = 0
    for split, with_targets in (("train", True), ("test", False)):
        for f in sorted((root / split).glob("*.txt")):
            rec = P.parse_well(f, with_targets=with_targets)
            checked += 1
            n_in = rec.inputs.shape[1]
            if n_in != C.N_INPUT:
                violations.append(f"{rec.well_id}: inputs has {n_in} cols, expected {C.N_INPUT}")
                continue
            if not with_targets or rec.targets is None:
                continue
            for j, tname in enumerate(C.TARGET_COLUMNS):
                m = ~L.missing_masks(rec.targets)[:, j]
                if int(m.sum()) < args.min_rows:
                    continue
                for k in range(C.N_INPUT):
                    ratio = float(np.isclose(rec.inputs[m, k], rec.targets[m, j], atol=ATOL).mean())
                    if ratio > RATIO_THRESHOLD:
                        violations.append(
                            f"{rec.well_id}: {tname} == inputs[:,{k}] (ratio={ratio:.4f})")

    print(f"检查井数: {checked}（train+test），输入列数要求: {C.N_INPUT}")
    print(f"违规: {len(violations)}")
    for v in violations[:10]:
        print("  -", v)
    print("RESULT:", "OK" if not violations else "FAIL")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
