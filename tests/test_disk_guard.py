#!/usr/bin/env python3
"""`disk_guard` 五路径单测（四审 R4-H3，**monkeypatch，不触碰真实磁盘**）。

四审用 monkeypatch 复现了两个缺陷：

  1. 初次 `cleanup`、cleanup 之后降到 `save_and_exit` 时，`capacity_hook` 调用次数是 **0**
     —— 违反模块 docstring 的"free < 5 GB → 先回调 hook 保存 last.pt，再抛错"。
     后果是磁盘快满时**最后一个 checkpoint 不会被保存**，`--resume` 无从续训。
  2. `allow_soft` 的位置在 cleanup 分支里，可能把 `abort`（free < 3 GB）也软接受
     —— 违反"free < 3 GB 绝不继续写盘"。

本模块把 `disk_state` / `cleanup` 替换成可控序列，覆盖
`ok / cleanup→ok / cleanup→save_and_exit / cleanup→abort / cleanup→cleanup(allow_soft) /
save_and_exit 直达 / abort 直达` 七条路径，断言异常、hook 调用次数与 `risk_accepted` 标记。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.data import disk_guard as DG  # noqa: E402


def _state(level: str, free_gb: float) -> DG.DiskState:
    return DG.DiskState(
        path="/fake", total_gb=100.0, used_gb=100.0 - free_gb,
        free_gb=free_gb, level=level,
    )


class _Harness(unittest.TestCase):
    """把 disk_state/cleanup 打桩成可控序列的公共夹具。"""

    def setUp(self) -> None:
        self._orig_state = DG.disk_state
        self._orig_cleanup = DG.cleanup
        self.hook_calls: list[str] = []
        self.cleanup_calls: list[bool] = []
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        DG.disk_state = self._orig_state
        DG.cleanup = self._orig_cleanup

    def _patch(self, sequence: list[tuple[str, float]]) -> dict:
        """sequence = [(level, free_gb), ...]；最后一次会重复返回。"""
        calls = {"n": 0}

        def fake_state(path="/", min_gb=8.0, cleanup_gb=5.0, abort_gb=3.0):
            i = min(calls["n"], len(sequence) - 1)
            calls["n"] += 1
            return _state(sequence[i][0], sequence[i][1])

        def fake_cleanup(verbose: bool = False) -> list[str]:
            self.cleanup_calls.append(bool(verbose))
            return ["/fake/cache/old.pt (12.0 MB)"]

        DG.disk_state = fake_state
        DG.cleanup = fake_cleanup
        return calls

    def _hook(self):
        def h():
            self.hook_calls.append("saved")
        return h

    def _assert(self, sequence, *, expect_raise: bool, hook_calls: int,
                allow_soft: bool = False, verbose: bool = False):
        calls = self._patch(sequence)
        if expect_raise:
            with self.assertRaises(DG.DiskBudgetError) as ctx:
                DG.assert_disk_headroom(8.0, "/fake", self._hook(),
                                        verbose=verbose, allow_soft=allow_soft)
            st = None
            err = str(ctx.exception)
        else:
            st = DG.assert_disk_headroom(8.0, "/fake", self._hook(),
                                         verbose=verbose, allow_soft=allow_soft)
            err = ""
        self.assertEqual(len(self.hook_calls), hook_calls,
                         f"capacity_hook 调用次数应为 {hook_calls}，实际 {len(self.hook_calls)}"
                         + (f"（异常：{err}）" if err else ""))
        return st, calls


class TestDiskGuardPaths(_Harness):
    def test_ok_returns_without_cleanup_or_hook(self):
        st, calls = self._assert([("ok", 20.0)], expect_raise=False, hook_calls=0)
        self.assertEqual(st.level, "ok")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(self.cleanup_calls, [])

    def test_cleanup_then_ok(self):
        st, calls = self._assert(
            [("cleanup", 6.5), ("ok", 12.0)], expect_raise=False, hook_calls=0)
        self.assertEqual(st.level, "ok")
        self.assertEqual(calls["n"], 2)          # 初次 + cleanup 后复测
        self.assertEqual(len(self.cleanup_calls), 1)

    def test_cleanup_then_save_and_exit_calls_hook_then_raises(self):
        """R4-H3 核心修复：cleanup 之后降到 save_and_exit 必须**先保存再抛错**。"""
        st, calls = self._assert(
            [("cleanup", 6.5), ("save_and_exit", 4.2)],
            expect_raise=True, hook_calls=1)
        self.assertIsNone(st)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(self.cleanup_calls), 1)

    def test_cleanup_then_abort_never_calls_hook(self):
        self._assert([("cleanup", 6.5), ("abort", 2.1)],
                     expect_raise=True, hook_calls=0)

    def test_cleanup_then_abort_ignores_allow_soft(self):
        """allow_soft 只能接受"高于 save_and_exit 安全线"的风险，绝不能吞掉 abort。"""
        self._assert([("cleanup", 6.5), ("abort", 2.1)],
                     expect_raise=True, hook_calls=0, allow_soft=True)

    def test_cleanup_then_cleanup_allow_soft_marks_risk_accepted(self):
        st, calls = self._assert(
            [("cleanup", 6.5), ("cleanup", 6.9)],
            expect_raise=False, hook_calls=0, allow_soft=True)
        self.assertEqual(st.level, "cleanup")
        self.assertTrue(any("risk_accepted" in a for a in st.actions))
        # cleanup 实际删除项必须被保留在 actions 里（可审计）
        self.assertTrue(any("old.pt" in a for a in st.actions))

    def test_cleanup_then_cleanup_without_allow_soft_raises(self):
        self._assert([("cleanup", 6.5), ("cleanup", 6.9)],
                     expect_raise=True, hook_calls=0, allow_soft=False)

    def test_direct_save_and_exit_calls_hook(self):
        self._assert([("save_and_exit", 4.0)], expect_raise=True, hook_calls=1)

    def test_direct_abort_no_hook_even_with_allow_soft(self):
        self._assert([("abort", 2.0)], expect_raise=True, hook_calls=0, allow_soft=True)

    def test_hook_failure_does_not_mask_disk_error(self):
        """hook 自身抛错时仍必须以 DiskBudgetError 结束（磁盘问题不能被 hook 掩盖）。"""
        self._patch([("save_and_exit", 4.0)])

        def bad_hook():
            self.hook_calls.append("attempted")
            raise RuntimeError("checkpoint write failed")

        with self.assertRaises(DG.DiskBudgetError):
            DG.assert_disk_headroom(8.0, "/fake", bad_hook, verbose=False)
        self.assertEqual(len(self.hook_calls), 1)


class TestDiskGuardHelpers(unittest.TestCase):
    def test_worst_level_ordering(self):
        self.assertEqual(DG.worst_level(["ok", "cleanup"]), "cleanup")
        self.assertEqual(DG.worst_level(["ok", "save_and_exit", "cleanup"]), "save_and_exit")
        self.assertEqual(DG.worst_level(["ok", "abort", "save_and_exit"]), "abort")
        self.assertEqual(DG.worst_level([]), "ok")

    def test_level_severity_covers_all_levels(self):
        for lv in ("abort", "save_and_exit", "cleanup", "ok"):
            self.assertIn(lv, DG.LEVEL_SEVERITY)

    def test_measure_path_degrades_to_existing_ancestor(self):
        """本机没有 /data 时也要能测量：退回到最近存在的祖先并记录实际路径。"""
        missing = Path("/definitely/not/here/v4/data")
        st = DG.measure_path(missing, min_gb=0.0)
        self.assertEqual(st.path, str(missing))
        self.assertIn(st.level, DG.LEVEL_SEVERITY)
        if not missing.exists():
            self.assertTrue(any("path_missing" in a for a in st.actions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
