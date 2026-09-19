#!/usr/bin/env python3
"""30 GB 云端磁盘守卫。

用途
----
云端机器只有 30 GB 磁盘，而一次 `pip install torch==2.4.0+cu124` 就要吃掉约 7 GB。
因此所有训练/推理脚本都必须在「启动时 / 每个 epoch 结束 / 每次落 checkpoint 前」
调用本模块，按剩余空间执行分级动作。

分级动作（默认阈值）
--------------------
    free >= 8 GB  ->  ok，什么都不做
    5 <= free < 8 ->  cleanup：删除 last_prev.pt、cache/tmp/*、旧特征版本目录
                      **cleanup 后重新测量**；若仍 < 8 GB，默认**抛 DiskBudgetError**
                      （与文档"8 GB 硬门禁"一致）；若调用方显式传
                      `allow_soft=True`，则仅记录 `risk_accepted` 并继续。
    free < 5 GB   ->  save_and_exit：先回调 capacity_hook 保存 last.pt，再抛错（可用 --resume 续训）
    free < 3 GB   ->  abort：不回调 hook，立即抛 DiskBudgetError，绝不继续写盘

接口
----
    from disk_guard import assert_disk_headroom, disk_report, register_cleanup_paths
    assert_disk_headroom()                 # 默认 min_gb=8.0，超限时自动清理
    rep = disk_report()                    # 用于写 training_time_log
    with disk_guard(capacity_hook=save_fn):  # 上下文管理器，抛出前先回调保存
        ...

命令行
------
    python3 src/data/disk_guard.py --min-free-gb 8 \
        --data-root /data --path /data --path /code/workspace \
        --report "/data,/code/workspace" --json reports/E0_disk_budget.json

`--path` 可重复（默认 `/`）；`--data-root`（缺省回退 `$V4_DATA_ROOT`）是**权威**路径：
JSON 顶层 `level/free_gb/...` 描述它，`paths` 列出每个被测挂载点，`worst_level` 是全部
挂载点中最差的 level（`abort` < `save_and_exit` < `cleanup` < `ok`）。退出码只看 primary
（data root）的 level，因此下游 `disk_budget_ok` 读到的就是 `/data` 配额而非根文件系统。

只依赖标准库（本机无 torch 也能跑）。
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

GB = 1024**3

# level 严重度：数值越小越严重（abort 最差）。用于计算 worst_level。
LEVEL_SEVERITY = {"abort": 0, "save_and_exit": 1, "cleanup": 2, "ok": 3}

# ---------------------------------------------------------------- 全局登记
_CLEANUP_CANDIDATES: list[Path] = []
_CACHE_DIRS: list[Path] = []


def register_cleanup_paths(paths: Iterable[Path | str]) -> None:
    """登记「可以安全删除」的路径（checkpoint 旧版本、临时目录等）。"""
    for p in paths:
        _CLEANUP_CANDIDATES.append(Path(p))


def register_cache_dirs(paths: Iterable[Path | str]) -> None:
    """登记缓存目录；磁盘紧张时按 mtime 从旧到新逐个删除其中的文件。"""
    for p in paths:
        _CACHE_DIRS.append(Path(p))


# ---------------------------------------------------------------- 数据类
@dataclass
class DiskState:
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    level: str  # ok | cleanup | save_and_exit | abort
    actions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "total_gb": round(self.total_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "level": self.level,
            "actions": self.actions,
        }


class DiskBudgetError(RuntimeError):
    """磁盘低于 abort 阈值时必须用这个异常终止，便于上层区分。"""


# ---------------------------------------------------------------- 核心
def _du(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def disk_state(path: Path | str = "/", min_gb: float = 8.0,
               cleanup_gb: float = 5.0, abort_gb: float = 3.0) -> DiskState:
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / GB
    state = DiskState(
        path=str(path),
        total_gb=usage.total / GB,
        used_gb=usage.used / GB,
        free_gb=free_gb,
        level="ok",
    )
    if free_gb < abort_gb:
        state.level = "abort"
    elif free_gb < cleanup_gb:
        state.level = "save_and_exit"
    elif free_gb < min_gb:
        state.level = "cleanup"
    return state


def measure_path(path: Path | str, min_gb: float = 8.0,
                 cleanup_gb: float = 5.0, abort_gb: float = 3.0) -> DiskState:
    """测量 `path` 所在文件系统。

    与 `disk_state` 的唯一差别：当 `path` 本身不存在时（例如本机开发环境没有 /data），
    退回到**最近的已存在祖先目录**测量，并在 `actions` 中记录实际测量的是哪个目录，
    而不是直接抛 FileNotFoundError。返回的 `state.path` 始终是调用方请求的路径。
    """
    requested = Path(path)
    probe = requested
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    if not probe.exists():
        raise FileNotFoundError(
            f"neither {requested} nor any ancestor exists; cannot measure disk usage"
        )
    state = disk_state(probe, min_gb=min_gb, cleanup_gb=cleanup_gb, abort_gb=abort_gb)
    state.path = str(requested)          # 报告请求的路径，而非探测用的祖先目录
    if probe != requested:
        state.actions.append(f"path_missing: measured nearest existing ancestor {probe}")
    return state


def worst_level(levels: Iterable[str]) -> str:
    """返回一组 level 中最严重的一个（abort < save_and_exit < cleanup < ok）。"""
    levels = list(levels)
    if not levels:
        return "ok"
    return min(levels, key=lambda lv: LEVEL_SEVERITY.get(lv, 99))


def cleanup(verbose: bool = False) -> list[str]:
    """返回被删除的路径列表；删除顺序：登记的可删路径 -> 缓存目录中最旧的文件。"""
    removed: list[str] = []
    for p in list(_CLEANUP_CANDIDATES):
        if p.exists():
            size = _du(p)
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                removed.append(f"{p} ({size / 1024**2:.1f} MB)")
                if verbose:
                    print(f"[disk_guard] removed {p} ({size / 1024**2:.1f} MB)")
            except OSError:
                pass
    # 缓存目录：按 mtime 升序删 25%（至少 1 个文件）
    for d in _CACHE_DIRS:
        if not d.is_dir():
            continue
        files = sorted(
            (f for f in d.rglob("*") if f.is_file()),
            key=lambda f: f.stat().st_mtime,
        )
        n = max(1, len(files) // 4)
        for f in files[:n]:
            size = f.stat().st_size
            try:
                f.unlink()
                removed.append(f"{f} ({size / 1024**2:.1f} MB)")
                if verbose:
                    print(f"[disk_guard] removed cache {f} ({size / 1024**2:.1f} MB)")
            except OSError:
                pass
    return removed


def assert_disk_headroom(
    min_gb: float = 8.0,
    path: Path | str = "/",
    capacity_hook: Callable[[], None] | None = None,
    verbose: bool = True,
    allow_soft: bool = False,
) -> DiskState:
    """在训练循环的关键位置调用。

    - free < min_gb : 执行 cleanup，然后重新测量；若仍 < cleanup 阈值则继续降级
    - free < 5 GB   : 先调用 capacity_hook() 保存状态，再抛 DiskBudgetError
    - free < 3 GB   : 不调用 hook，直接抛 DiskBudgetError（避免写盘把机器彻底打爆）
    """
    st = disk_state(path, min_gb=min_gb)
    if st.level == "ok":
        return st

    if st.level == "cleanup":
        if verbose:
            print(f"[disk_guard] free={st.free_gb:.2f} GB < {min_gb} GB -> cleanup")
        st.actions = cleanup(verbose=verbose)
        st = disk_state(path, min_gb=min_gb)
        if st.level == "ok":
            return st
        if allow_soft:
            # 显式接受风险：cleanup 后仍低于 min_gb，但高于 save_and_exit 阈值
            st.actions.append("risk_accepted: still below min_gb after cleanup")
            if verbose:
                print(f"[disk_guard] WARNING: free={st.free_gb:.2f} GB still < {min_gb} GB "
                      f"(risk_accepted=True)")
            return st
        raise DiskBudgetError(
            f"disk free={st.free_gb:.2f} GB on {st.path} 仍低于硬门禁 {min_gb} GB "
            f"（cleanup 已删除 {len(st.actions)} 项）。请清理 cache/ 与旧 checkpoint 后重试；"
            "如确需继续，请显式传 allow_soft=True 并在 Gate 中记录 risk_accepted。"
        )

    if st.level in ("save_and_exit", "abort"):
        if st.level == "save_and_exit" and capacity_hook is not None:
            if verbose:
                print(f"[disk_guard] free={st.free_gb:.2f} GB -> save_and_exit (calling hook)")
            try:
                capacity_hook()
            except Exception as exc:  # hook 失败不掩盖磁盘问题
                print(f"[disk_guard] capacity_hook failed: {exc!r}")
        raise DiskBudgetError(
            f"disk free={st.free_gb:.2f} GB on {st.path} (level={st.level}); "
            f"cleanup removed {len(st.actions)} path(s). "
            "请清理 cache/ 与旧 checkpoint 后使用 --resume 继续。"
        )
    return st


def disk_report(path: Path | str = "/") -> dict:
    st = disk_state(path)
    return st.as_dict()


class disk_guard:
    """上下文管理器：在 with 块内任何位置超限都能带着 hook 优雅退出。"""

    def __init__(self, path: Path | str = "/", min_gb: float = 8.0,
                 capacity_hook: Callable[[], None] | None = None):
        self.path = path
        self.min_gb = min_gb
        self.hook = capacity_hook

    def __enter__(self) -> "disk_guard":
        assert_disk_headroom(self.min_gb, self.path, self.hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


# ---------------------------------------------------------------- CLI
def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="v4 disk guard (30 GB cloud budget)")
    ap.add_argument("--path", action="append", default=None, dest="paths",
                    help="要检查的挂载点；可多次传入（例如 --path /data --path /code/workspace）。"
                         "不传时默认只检查 /。")
    ap.add_argument("--data-root", default=None, dest="data_root",
                    help="权威数据根（云端通常是 /data，即 $V4_DATA_ROOT）。给定后顶层 "
                         "level/free_gb/total_gb/used_gb/actions 描述该路径所在文件系统；"
                         "缺省时回退到环境变量 $V4_DATA_ROOT，再缺省则用第一个 --path（或 /）。")
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--cleanup", action="store_true", help="只清理，不做阈值判定")
    ap.add_argument("--report", type=str, default=None,
                    help="额外统计该目录占用（可多次用逗号分隔）")
    args = ap.parse_args()

    if args.cleanup:
        removed = cleanup(verbose=True)
        print(f"removed {len(removed)} path(s)")

    # 权威 data root：显式 --data-root > $V4_DATA_ROOT > None
    data_root = args.data_root or os.environ.get("V4_DATA_ROOT") or None

    # 待检查路径：data root 优先，随后是 --path；去重保序
    requested: list[str] = []
    if data_root:
        requested.append(data_root)
    if args.paths:
        requested.extend(args.paths)
    if not requested:
        requested = ["/"]

    seen: set[str] = set()
    ordered: list[str] = []
    for raw in requested:
        key = os.path.normpath(str(Path(raw).expanduser()))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(raw)

    states = [measure_path(p, min_gb=args.min_free_gb) for p in ordered]

    # 顶层字段描述 primary（data root）；找不到时退回第一个路径
    primary = states[0]
    if data_root:
        dr = os.path.normpath(str(Path(data_root).expanduser()))
        for st in states:
            if os.path.normpath(str(Path(st.path).expanduser())) == dr:
                primary = st
                break

    payload = primary.as_dict()
    payload["paths"] = [s.as_dict() for s in states]
    payload["worst_level"] = worst_level(s.level for s in states)
    payload["primary_path"] = primary.path
    payload["data_root_checked"] = bool(data_root)
    if args.report:
        sizes = {}
        for p in args.report.split(","):
            pp = Path(p.strip())
            if pp.exists():
                sizes[str(pp)] = round(_du(pp) / GB, 3)
        payload["dir_sizes_gb"] = sizes
    payload["ts"] = time.time()

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 退出码只看 primary（data root）的 level，与 disk_budget_ok 的口径一致
    return 0 if primary.level == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
