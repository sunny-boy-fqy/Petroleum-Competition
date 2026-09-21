#!/usr/bin/env python3
"""同步训练恢复所需的小状态文件（checkpoint / scaler / OOF / 报告）。

为什么需要它
------------
训练期数据/缓存写本地 `/code/workspace/v4/*`，网络盘 `/data` 只放最终模型。
但平台任务不能一次跑完时，本地盘会随任务结束丢失。为了让新任务能续跑，
只同步“小体积、可恢复训练”的文件：

    *.pt / *.pth / *.ckpt   每折 best/last/last_prev
    *.npz                   OOF / 必要小分片
    *.json / *.md / *.csv    scaler、manifest、Gate 报告

大 cache/原始数据不同步；新任务从 `/data/v4_data.tar.gz` 重新解压并重建 cache。

用法::

    python3 tools/sync_state.py --src /code/workspace/v4/runs --dst /data/v4/mirror/runs --once
    python3 tools/sync_state.py --src /data/v4/mirror/runs --dst /code/workspace/v4/runs --once
    python3 tools/sync_state.py --src /code/workspace/v4/runs --dst /data/v4/mirror/runs --interval 300
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_PATTERNS = ("*.pt", "*.pth", "*.ckpt", "*.npz", "*.json", "*.md", "*.csv")
SKIP_DIR_NAMES = {"tb", "logs", "__pycache__", ".git", ".v4cache", ".v4_runtime"}


def _match(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _copy_if_needed(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    try:
        src_stat = src.stat()
    except OSError:
        return False
    try:
        dst_stat = dst.stat()
    except OSError:
        dst_stat = None
    if (dst_stat is not None and dst_stat.st_mtime >= src_stat.st_mtime
            and dst_stat.st_size == src_stat.st_size):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp",
                                    dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def sync_once(src: Path, dst: Path, patterns: tuple[str, ...]) -> int:
    if not src.is_dir():
        return 0
    copied = 0
    for root, dirs, files in os.walk(src):
        # 稳定遍历顺序，便于日志复现；同时跳过明显的无关大目录。
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES)
        root_p = Path(root)
        rel = root_p.relative_to(src)
        for name in sorted(files):
            if not _match(name, patterns):
                continue
            sp = root_p / name
            dp = dst / rel / name
            if _copy_if_needed(sp, dp):
                copied += 1
                print(f"[sync] {sp} -> {dp}", flush=True)
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 小状态文件同步器")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--patterns", default=",".join(DEFAULT_PATTERNS),
                    help="逗号分隔的 glob，默认 checkpoint/OOF/报告")
    ap.add_argument("--interval", type=float, default=0.0,
                    help=">0 时循环同步（后台 mirror 用）")
    ap.add_argument("--once", action="store_true", help="只同步一次（默认 interval=0 时自动）")
    args = ap.parse_args(argv)

    src, dst = Path(args.src).expanduser().resolve(), Path(args.dst).expanduser().resolve()
    patterns = tuple(p.strip() for p in args.patterns.split(",") if p.strip())
    if not patterns:
        print("!! --patterns 不能为空", file=sys.stderr)
        return 2

    if args.interval <= 0 or args.once:
        n = sync_once(src, dst, patterns)
        print(f"[sync] done: {n} file(s) -> {dst}")
        return 0

    # 后台 mirror：异常不退出，下一轮继续；SIGTERM/KeyboardInterrupt 正常退出。
    while True:
        try:
            sync_once(src, dst, patterns)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001 - 后台任务必须尽量存活
            print(f"[sync] WARN: {exc!r}", file=sys.stderr, flush=True)
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
