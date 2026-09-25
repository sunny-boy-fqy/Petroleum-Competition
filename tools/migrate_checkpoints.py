#!/usr/bin/env python3
"""把旧 torch/torch_npu 写出的 checkpoint 重新保存为当前 torch 格式。

背景：Ascend 上加载早期任务留下的 ``*.pt`` 时，torch_npu 会提示
"The current version of the file storing weights is old ... please use newer torch
to re-store the weight file."。该警告不阻塞当前训练，但未来 torch 版本可能停止加载。

本工具做最小、保内容的迁移：

    load(payload) -> 原子写回同一路径

payload（state_dict/optimizer/scheduler/manifest 所需标量）原样保留；同时把相邻
``*.manifest.json`` 标记为 ``checkpoint_format=2`` 并更新 ``torch`` / ``bytes``。
``checkpoint_format>=2`` 的文件直接跳过，因此可以安全地在每个任务启动时调用。

用法::

    python3 tools/migrate_checkpoints.py --root "$V4_RUN_ROOT/E3"
    python3 tools/migrate_checkpoints.py --root /code/workspace/v4/runs --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _atomic_torch_save(payload, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.pt",
                                    dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        try:
            os.chmod(tmp, path.stat().st_mode)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _mark_manifest(path: Path, torch_version: str) -> bool:
    mp = path.with_suffix(".manifest.json")
    if not mp.is_file():
        return False
    try:
        man = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return False
    man["checkpoint_format"] = 2
    man["migrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    man["torch"] = torch_version
    man["bytes"] = int(path.stat().st_size)
    tmp = mp.with_name(mp.name + ".tmp")
    tmp.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, mp)
    return True


def _iter_checkpoints(root: Path):
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*.pt")):
        if ".tmp." in p.name or p.name.endswith(".tmp.pt"):
            continue
        if p.is_file():
            yield p


def migrate_one(path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """返回 ``(changed, reason)``；changed=False 表示已跳过。"""
    import torch

    # NPU 环境里 checkpoint 可能含 Ascend storage；先注册 torch_npu 再加载。
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass

    mp = path.with_suffix(".manifest.json")
    if mp.is_file():
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
            if int(man.get("checkpoint_format", 0) or 0) >= 2:
                return False, "already-format2"
        except Exception:
            pass
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"load-failed: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return False, "not-a-dict"
    if dry_run:
        return True, "dry-run"
    _atomic_torch_save(payload, path)
    marked = _mark_manifest(path, str(getattr(torch, "__version__", "unknown")))
    return True, ("migrated" if marked else "migrated(no-manifest)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="迁移旧 checkpoint 格式")
    ap.add_argument("--root", action="append", default=None,
                    help="待迁移目录，可重复；默认 $V4_RUN_ROOT 或 /code/workspace/v4/runs")
    ap.add_argument("--quiet", action="store_true", help="只打印摘要")
    ap.add_argument("--dry-run", action="store_true", help="只扫描，不写文件")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    roots = [Path(x).expanduser() for x in (args.root or [])]
    if not roots:
        default = os.environ.get("V4_RUN_ROOT") or "/code/workspace/v4/runs"
        roots = [Path(default)]
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[migrate] 需要 torch：{exc!r}", file=sys.stderr)
        return 2

    changed = skipped = failed = 0
    for root in roots:
        for p in _iter_checkpoints(root):
            ok, why = migrate_one(p, dry_run=args.dry_run)
            if why == "already-format2":
                skipped += 1
                continue
            if ok:
                changed += 1
                if not args.quiet:
                    print(f"[migrate] {p} -> {why}", flush=True)
            else:
                failed += 1
                print(f"[migrate] WARN {p}: {why}", file=sys.stderr, flush=True)
    print(f"[migrate] roots={[str(r) for r in roots]} changed={changed} "
          f"skipped={skipped} failed={failed} dry_run={args.dry_run}", flush=True)
    return 0 if failed == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
