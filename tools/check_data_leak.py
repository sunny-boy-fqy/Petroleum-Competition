#!/usr/bin/env python3
"""全量输入-标签泄漏回归（90 口井）。

背景（审查 B1）：`parse.py` 曾用 `inputs = arr[:, 1:15]`，把索引 14 的 **POR 标签**
当成第 14 个输入，80 口训练井全部泄漏。修复后必须有全量回归防止再次发生。

检查项（逐井）：
  1. `inputs.shape[1] == 13`（13 条曲线；DEPTH 单独作为深度通道）
  2. 任何输入列与任何目标列**不完全相等**（atol=1e-9，比例 > 0.999 即判违规）
  3. 训练井目标列存在；测试井 `inputs` 与训练井列数一致

覆盖性断言（R5-H1，**必须**）：
  4. 训练井 = 80、测试井 = 10（`src.constants.EXPECTED_N_TRAIN_WELLS` 等）
     —— 否则"0 口井、0 违规、RESULT: OK"会变成**假绿**：一条防止 80 口井标签泄漏的
     回归在数据缺失时反而宣告通过（`git archive` 解包目录、路径写错、目录为空时都会命中）。

    python3 v4/tools/check_data_leak.py [--data ../data]
    python3 v4/tools/check_data_leak.py --data /data/v4/data
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
EXPECTED = {"train": C.EXPECTED_N_TRAIN_WELLS, "test": C.EXPECTED_N_TEST_WELLS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(V4.parent / "data"))
    ap.add_argument("--min-rows", type=int, default=20,
                    help="目标非缺测行数少于此值时跳过该目标的等值比较")
    ap.add_argument("--expect-train", type=int, default=C.EXPECTED_N_TRAIN_WELLS,
                    help="必须检查到的训练井数（覆盖性断言；0 井必须 FAIL）")
    ap.add_argument("--expect-test", type=int, default=C.EXPECTED_N_TEST_WELLS,
                    help="必须检查到的测试井数（覆盖性断言；0 井必须 FAIL）")
    args = ap.parse_args()

    root = Path(args.data)
    expect = {"train": args.expect_train, "test": args.expect_test}
    violations: list[str] = []
    counts = {"train": 0, "test": 0}

    if not root.is_dir():
        print(f"!! 数据目录不存在：{root}")
        print("RESULT: FAIL")
        return 1

    for split, with_targets in (("train", True), ("test", False)):
        split_dir = root / split
        for f in sorted(split_dir.glob("*.txt")):
            rec = P.parse_well(f, with_targets=with_targets)
            counts[split] += 1
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

    # ---- 覆盖性断言（R5-H1）：不看井数就等于在空目录上"空转通过"
    for split in ("train", "test"):
        if counts[split] != expect[split]:
            violations.append(
                f"coverage[{split}]: 只检查到 {counts[split]} 口井，期望 {expect[split]}"
                f"（--data 是否指向正确的数据目录？空目录/错目录必须 FAIL）")

    print(f"检查井数: {counts['train']}+{counts['test']}={sum(counts.values())}"
          f"（期望 train={expect['train']} / test={expect['test']}），"
          f"输入列数要求: {C.N_INPUT}")
    print(f"违规: {len(violations)}")
    for v in violations[:10]:
        print("  -", v)
    print("RESULT:", "OK" if not violations else "FAIL")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
