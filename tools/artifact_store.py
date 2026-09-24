#!/usr/bin/env python3
"""可复用训练成果存储（大成果层）。

与 ``tools/sync_state.py`` 的分工
--------------------------------
- ``sync_state.py``：小状态层，持久化 checkpoint / OOF / report / scaler 到
  ``/data/v4/mirror``；
- ``artifact_store.py``：大成果层，补齐 ``cache/raw``、``cache/feat``、
  ``runs/**/*.pkl``、``runs/**/*.onnx``、``runs/**/*.zip``、候选/注册表 JSON 到
  ``/data/v4/artifacts``。

两者合起来覆盖“所有有用的训练成果”：

- 模型参数、checkpoint、OOF、报告 → ``sync_state.py``
- 特征缓存、E3/E4 折级 ``.pkl``、E10 提交包、候选注册表 → ``artifact_store.py``

设计原则
--------
- **增量复制**：目标文件大小相同且 mtime 不旧时跳过，避免每次重复写大文件；
- **原子写出**：先写同目录临时文件，再 ``os.replace``；
- **预算保护**：默认总上限 24 GiB，超过时列出最大文件并拒绝发布；``--max-gb 0`` 关闭；
- **不处理原始数据**：原始数据仍从 ``/data/v4_data.tar.gz`` 部署；
- **不处理日志/TensorBoard**：``tb/``、``logs/``、``*.log`` 默认排除。

用法::

    python3 tools/artifact_store.py publish --local-root "$V4_DATA_ROOT" --remote-root "$V4_NETWORK_ROOT"
    python3 tools/artifact_store.py restore --local-root "$V4_DATA_ROOT" --remote-root "$V4_NETWORK_ROOT"
    python3 tools/artifact_store.py status  --remote-root "$V4_NETWORK_ROOT"
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

FORMAT_VERSION = 1
DEFAULT_LOCAL_ROOT = os.environ.get("V4_DATA_ROOT") or "/code/workspace"
DEFAULT_REMOTE_ROOT = os.environ.get("V4_NETWORK_ROOT") or "/data"
DEFAULT_MAX_GB = 24.0

SKIP_DIR_NAMES = {"tb", "logs", "__pycache__", ".git", ".v4cache", ".v4_runtime", "tmp"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".pyd", ".tmp", ".log")


@dataclass(frozen=True)
class Source:
    name: str
    src: str                          # relative to --local-root
    dst: str                          # relative to <remote-root>/v4/artifacts
    suffixes: tuple[str, ...] = ()    # empty = all files
    exclude_names: frozenset[str] = field(default_factory=frozenset)
    exclude_globs: tuple[str, ...] = ()

    def local_dir(self, local_root: Path) -> Path:
        return local_root / self.src

    def remote_dir(self, artifact_root: Path) -> Path:
        return artifact_root / self.dst


SOURCES: tuple[Source, ...] = (
    Source(
        name="cache",
        src="v4/cache",
        dst="cache",
        exclude_globs=("tmp/*", "*/tmp/*"),
    ),
    Source(
        name="run_extras",
        src="v4/runs",
        dst="run_root",
        suffixes=(".pkl", ".onnx", ".zip"),
    ),
    Source(
        name="state",
        src="v4/state",
        dst="state",
        suffixes=(".json",),
        exclude_names=frozenset({"all_pipeline_progress.json"}),
    ),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024.0
    return f"{n} B"


def _excluded(source: Source, rel: str, name: str) -> bool:
    if name in source.exclude_names:
        return True
    if Path(name).suffix.lower() in SKIP_SUFFIXES:
        return True
    for pat in source.exclude_globs:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def _iter_local(source: Source, local_root: Path) -> Iterator[tuple[Path, str]]:
    base = source.local_dir(local_root)
    if not base.is_dir():
        return
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES)
        root_p = Path(root)
        for name in sorted(files):
            p = root_p / name
            rel = p.relative_to(base).as_posix()
            if _excluded(source, rel, name):
                continue
            if source.suffixes and p.suffix.lower() not in source.suffixes:
                continue
            if p.is_file():
                yield p, rel


def _iter_remote(source: Source, artifact_root: Path) -> Iterator[tuple[Path, str]]:
    base = source.remote_dir(artifact_root)
    if not base.is_dir():
        return
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES)
        root_p = Path(root)
        for name in sorted(files):
            p = root_p / name
            rel = p.relative_to(base).as_posix()
            if _excluded(source, rel, name):
                continue
            if source.suffixes and p.suffix.lower() not in source.suffixes:
                continue
            if p.is_file():
                yield p, rel


def _copy_if_needed(src: Path, dst: Path) -> bool:
    """把 src 原子复制到 dst；目标已同大小且不旧时跳过。"""
    try:
        if src.resolve() == dst.resolve():
            return False
    except OSError:
        pass
    try:
        st = src.stat()
    except OSError:
        return False
    try:
        ds = dst.stat()
        if ds.st_size == st.st_size and ds.st_mtime_ns >= st.st_mtime_ns:
            return False
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _collect(source: Source, local_root: Path) -> list[tuple[Path, str, int]]:
    out: list[tuple[Path, str, int]] = []
    for p, rel in _iter_local(source, local_root):
        try:
            out.append((p, rel, int(p.stat().st_size)))
        except OSError:
            continue
    return out


def publish(args: argparse.Namespace) -> int:
    local_root = Path(args.local_root).expanduser().resolve()
    remote_root = Path(args.remote_root).expanduser().resolve()
    artifact_root = remote_root / "v4" / "artifacts"
    if not local_root.is_dir():
        print(f"[artifact_store] local-root 不存在：{local_root}", file=sys.stderr)
        return 2

    entries: list[tuple[Source, Path, Path, int]] = []
    source_stats: dict[str, dict] = {}
    for source in SOURCES:
        files = _collect(source, local_root)
        bytes_total = sum(n for _, _, n in files)
        source_stats[source.name] = {
            "src": str(source.local_dir(local_root)),
            "dst": str(source.remote_dir(artifact_root)),
            "files": len(files),
            "bytes": int(bytes_total),
        }
        for p, rel, n in files:
            entries.append((source, p, source.remote_dir(artifact_root) / rel, n))

    total_bytes = sum(n for _, _, _, n in entries)
    total_gb = total_bytes / 1024 ** 3
    if args.max_gb and args.max_gb > 0 and total_gb > float(args.max_gb):
        print(f"[artifact_store] 预算超限：{total_gb:.2f} GiB > {args.max_gb:.2f} GiB",
              file=sys.stderr)
        for source, p, dst, n in sorted(entries, key=lambda x: -x[3])[:10]:
            print(f"  {_fmt_bytes(n):>10}  {source.name}/{p.relative_to(source.local_dir(local_root))}",
                  file=sys.stderr)
        return 3

    copied = 0
    copied_bytes = 0
    failed = 0
    for source, p, dst, n in entries:
        try:
            if _copy_if_needed(p, dst):
                copied += 1
                copied_bytes += n
                source_stats[source.name]["copied"] = source_stats[source.name].get("copied", 0) + 1
                if not args.quiet:
                    print(f"[artifact_store] + {source.name}/{p.relative_to(source.local_dir(local_root))}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[artifact_store] WARN copy failed: {p} -> {dst}: {exc!r}",
                  file=sys.stderr)

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": _now_iso(),
        "local_root": str(local_root),
        "remote_root": str(remote_root),
        "artifact_root": str(artifact_root),
        "max_gb": float(args.max_gb),
        "total_files": len(entries),
        "total_bytes": int(total_bytes),
        "copied_files": copied,
        "copied_bytes": int(copied_bytes),
        "failed_files": failed,
        "sources": source_stats,
    }
    try:
        _write_json_atomic(artifact_root / "manifest.json", manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"[artifact_store] WARN 无法写 manifest：{exc!r}", file=sys.stderr)
        failed += 1

    print("[artifact_store] publish done")
    print(f"  local_root : {local_root}")
    print(f"  artifact   : {artifact_root}")
    print(f"  files      : {len(entries)}  copied={copied}  failed={failed}")
    print(f"  bytes      : {_fmt_bytes(total_bytes)}  copied={_fmt_bytes(copied_bytes)}")
    return 0 if failed == 0 else 4


def restore(args: argparse.Namespace) -> int:
    local_root = Path(args.local_root).expanduser().resolve()
    remote_root = Path(args.remote_root).expanduser().resolve()
    artifact_root = remote_root / "v4" / "artifacts"
    if not artifact_root.is_dir():
        print(f"[artifact_store] 无 artifact 目录，跳过 restore：{artifact_root}")
        return 0

    copied = 0
    failed = 0
    for source in SOURCES:
        local_base = source.local_dir(local_root)
        for p, rel in _iter_remote(source, artifact_root):
            dst = local_base / rel
            try:
                if _copy_if_needed(p, dst):
                    copied += 1
                    if not args.quiet:
                        print(f"[artifact_store] <- {source.name}/{rel}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[artifact_store] WARN restore failed: {p} -> {dst}: {exc!r}",
                      file=sys.stderr)
    print("[artifact_store] restore done")
    print(f"  local_root : {local_root}")
    print(f"  artifact   : {artifact_root}")
    print(f"  copied     : {copied}  failed={failed}")
    return 0 if failed == 0 else 4


def status(args: argparse.Namespace) -> int:
    remote_root = Path(args.remote_root).expanduser().resolve()
    artifact_root = remote_root / "v4" / "artifacts"
    manifest = artifact_root / "manifest.json"
    if manifest.is_file():
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            print(f"[artifact_store] manifest: {manifest}")
            for k in ("created_at", "local_root", "total_files", "total_bytes",
                      "copied_files", "copied_bytes", "failed_files"):
                print(f"  {k}: {doc.get(k)}")
            for name, st in (doc.get("sources") or {}).items():
                print(f"  {name}: files={st.get('files')} bytes={st.get('bytes')} "
                      f"copied={st.get('copied', 0)}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[artifact_store] WARN manifest 读取失败：{exc!r}", file=sys.stderr)
    total_files = 0
    total_bytes = 0
    print(f"[artifact_store] artifact: {artifact_root}")
    for source in SOURCES:
        files = list(_iter_remote(source, artifact_root))
        n = sum((p.stat().st_size for p, _ in files), 0)
        total_files += len(files)
        total_bytes += n
        print(f"  {source.name}: files={len(files)} bytes={n}")
    print(f"  total: files={total_files} bytes={total_bytes}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT,
                        help=f"本地运行时根（默认 {DEFAULT_LOCAL_ROOT}）")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT,
                        help=f"云盘根（默认 {DEFAULT_REMOTE_ROOT}）")
    parser.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB,
                        help="发布总大小上限 GiB；0=不限制（默认 24）")
    parser.add_argument("--quiet", action="store_true", help="只打印摘要，不逐文件打印")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="持久化/恢复 v4 可复用训练成果")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("publish", "restore", "status"):
        sp = sub.add_parser(cmd, help=f"{cmd} 可复用成果")
        _add_common(sp)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "publish":
        return publish(args)
    if args.cmd == "restore":
        return restore(args)
    if args.cmd == "status":
        return status(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
