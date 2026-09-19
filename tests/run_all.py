#!/usr/bin/env python3
"""一键运行 v4 口径层测试（unittest，只依赖标准库 + numpy）。

    python3 v4/tests/run_all.py            # 全部
    python3 v4/tests/run_all.py -v         # 详细
    python3 v4/tests/run_all.py test_parse # 指定模块

设计说明（R2-H5）：这些测试**不需要 torch**，覆盖数据/评分/契约/Gate 四组口径，
可在本机（开发机）与云端同样的方式运行。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
sys.path.insert(0, str(V4))
sys.path.insert(0, str(HERE))


def main() -> int:
    loader = unittest.TestLoader()
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        names = sys.argv[1:]
        suite = loader.loadTestsFromNames(names)
        argv = [sys.argv[0]] + [a for a in sys.argv[2:]]
    else:
        suite = loader.discover(str(HERE), pattern="test_*.py")
        argv = sys.argv
    runner = unittest.TextTestRunner(verbosity=2 if "-v" in argv else 1)
    result = runner.run(suite)
    print(f"\n运行 {result.testsRun} 项，失败 {len(result.failures)}，错误 {len(result.errors)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
