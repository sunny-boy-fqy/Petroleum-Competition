#!/usr/bin/env python3
"""一键运行 v4 口径层测试（unittest，只依赖标准库 + numpy）。

    python3 v4/tests/run_all.py            # 全部
    python3 v4/tests/run_all.py -v         # 详细
    python3 v4/tests/run_all.py test_parse # 指定模块

设计说明（R2-H5）：口径层测试**不需要 torch**，覆盖数据/评分/契约/Gate/计划统计/平台脚本
六组口径，可在本机（开发机）与云端同样的方式运行。
需要 torch 的用例（`test_losses.py` / `test_heads.py`）在没有 torch 时自动 **skip**，
装了 torch 就会真正执行 —— 因此同一个入口既能做本机快速校验，也能做 GPU 机全量校验。

解释器选择（R3）
----------------
本机 `/usr/bin/python3` 是 PEP-668 externally-managed，**没有 numpy**，直接跑会在 import
阶段抛一堆 `ModuleNotFoundError`，看起来像测试失败，其实是环境问题。因此入口先做一次
**显式预检**：缺 numpy 时打印可执行的补救命令并返回 **2**（区别于测试失败的 1），
避免把"环境缺失"误读成"测试挂了"。

推荐（本机）：`../v2/.venv/bin/python tests/run_all.py`
装了 CPU torch 的校验 venv：`./.venv-torch/bin/python tests/run_all.py`
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
sys.path.insert(0, str(V4))
sys.path.insert(0, str(HERE))

# 本机可用的、带 numpy 的解释器候选（按优先级）
_PY_CANDIDATES = (
    V4 / ".venv-torch" / "bin" / "python",
    V4.parent / "v2" / ".venv" / "bin" / "python",
    V4 / ".venv" / "bin" / "python",
)


def preflight() -> int | None:
    """返回 None 表示可以继续；返回整数表示应直接以该退出码结束。"""
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        found = [str(p) for p in _PY_CANDIDATES if p.is_file()]
        print("=" * 74)
        print(f"ENV ERROR: 当前解释器没有 numpy，口径层测试无法运行。")
        print(f"  interpreter : {sys.executable}")
        print(f"  python      : {sys.version.split()[0]}")
        print("这是**环境问题，不是测试失败**（退出码 2 与测试失败的 1 区分）。")
        if found:
            print("请改用下列解释器之一重跑：")
            for p in found:
                print(f"  {p} {Path(__file__).relative_to(V4.parent)}")
        else:
            print("未找到带 numpy 的 venv；可先创建："
                  "python3 -m venv .venv && .venv/bin/pip install numpy==1.26.4")
        print("=" * 74)
        return 2
    return None


def main() -> int:
    rc = preflight()
    if rc is not None:
        return rc

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
    print(f"\n运行 {result.testsRun} 项，失败 {len(result.failures)}，错误 {len(result.errors)}，"
          f"跳过 {len(result.skipped)}")
    if result.skipped:
        names = {t.id().split(".")[-2] for t, _ in result.skipped}
        print(f"跳过模块（缺 torch 时属正常）：{sorted(names)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
