#!/usr/bin/env python3
"""把 v4 仓库打包成平台“zip 上传代码”可用的文件。

原则
----
- 只打包 git 跟踪 + 未被 .gitignore 忽略的新增文件；
- 包内路径以仓库根为根（run_train.sh 在 zip 根目录，不套一层 v4/）；
- 排除 .git / .venv / .v4cache / __pycache__ / 大产物等；
- 默认输出到 /mnt/d/tmp/Petroleum-Competition/；
- **默认先清除目标目录中的旧 zip**，保证交付目录只保留本次最新包；
  如需保留旧包，显式传 `--keep-old`。

WSL 兼容
--------
若 /mnt/d 是只读挂载（本机 WSL 常见），程序会先用 WSL 临时目录生成 zip，
再通过 powershell.exe 复制到 Windows 的 D:\\tmp\\Petroleum-Competition\\。
这样不需要给 WSL mount 加写权限。

用法::

    python3 tools/pack_code_zip.py
    python3 tools/pack_code_zip.py --out-dir /mnt/d/tmp/Petroleum-Competition
    python3 tools/pack_code_zip.py --name v4_code_src.zip
    python3 tools/pack_code_zip.py --keep-old     # 调试用，不推荐交付时使用
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_OUT_DIR = "/mnt/d/tmp/Petroleum-Competition"

# 这些是仓库根级运行时/临时目录，即使误被 git 列出也不进包。
SKIP_TOP_LEVEL = {
    ".git", ".venv", ".venv-torch", ".v4cache", ".v4_runtime", "__pycache__",
    "cache", "runs", "artifacts", "submission", "experiments",
    "models", "tb", "logs", "dist",
}
SKIP_SUFFIXES = (".pyc", ".pyo", ".pyd")


def _run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def git_rev(repo: Path) -> str:
    try:
        return _run_git(repo, "rev-parse", "--short", "HEAD")
    except Exception:
        return "nogit"


def git_dirty(repo: Path) -> bool:
    try:
        return bool(_run_git(repo, "status", "--porcelain"))
    except Exception:
        return False


def collect_files(repo: Path) -> list[str]:
    """返回相对仓库根的 POSIX 路径列表（tracked + 未忽略的新增文件）。"""
    out = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z",
         "--cached", "--others", "--exclude-standard"]
    )
    rels = [x.decode("utf-8") for x in out.split(b"\0") if x]
    seen: set[str] = set()
    files: list[str] = []
    for rel in rels:
        rel = rel.replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        parts = Path(rel).parts
        if not parts or parts[0] in SKIP_TOP_LEVEL:
            continue
        if "__pycache__" in parts:
            continue
        if rel.endswith(SKIP_SUFFIXES):
            continue
        if (repo / rel).is_file():
            files.append(rel)
    return sorted(files)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_zip(repo: Path, out_path: Path, files: list[str]) -> None:
    """原子地写出 zip；out_path 的父目录必须已存在且可写。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=out_path.name + ".", suffix=".tmp", dir=str(out_path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as zf:
            for rel in files:
                zf.write(repo / rel, arcname=rel)
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)


def clean_old_zips(out_dir: str | Path, pattern: str = "*.zip") -> list[Path]:
    """删除目标目录中的旧 zip（默认交付纪律：一个目录只留最新包）。"""
    d = Path(out_dir)
    removed: list[Path] = []
    if not d.is_dir():
        return removed
    for p in sorted(d.glob(pattern)):
        if p.is_file():
            p.unlink()
            removed.append(p)
    return removed


def _clean_windows_zips(win_dir: str) -> int:
    """WSL `/mnt/d` 只读回退时，在 Windows 侧删除旧 zip。

    返回删除数量；失败返回 -1。
    """
    ps = _powershell()
    if not ps:
        return -1
    cmd = (
        f"$d = '{win_dir}'; "
        f"New-Item -ItemType Directory -Force -Path $d | Out-Null; "
        f"$old = @(Get-ChildItem -LiteralPath $d -Filter '*.zip' -File "
        f"-ErrorAction SilentlyContinue); "
        f"$n = @($old).Count; "
        f"$old | Remove-Item -Force -ErrorAction SilentlyContinue; "
        f"Write-Output $n; exit 0"
    )
    try:
        r = subprocess.run([ps, "-NoProfile", "-Command", cmd],
                           check=True, capture_output=True, text=True)
        return int((r.stdout or "0").strip().splitlines()[-1])
    except Exception:
        return -1


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".v4_write_", delete=True):
            return True
    except OSError:
        return False


def _wsl_to_windows_path(path: Path) -> str | None:
    """`/mnt/d/tmp/x` -> `D:\\tmp\\x`；非 /mnt/<盘> 返回 None。"""
    m = re.match(r"^/mnt/([A-Za-z])/(.*)$", path.as_posix())
    if not m:
        return None
    return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\")


def _powershell() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("pwsh")


def _copy_to_windows_via_powershell(src: Path, dst_win: str) -> bool:
    ps = _powershell()
    if not ps:
        return False
    try:
        src_win = subprocess.check_output(
            ["wslpath", "-w", str(src)], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return False
    cmd = (
        f"$src = '{src_win}'; $dst = '{dst_win}'; "
        f"New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null; "
        f"Copy-Item -LiteralPath $src -Destination $dst -Force; "
        f"if (-not (Test-Path -LiteralPath $dst)) {{ exit 7 }}"
    )
    try:
        subprocess.run([ps, "-NoProfile", "-Command", cmd],
                       check=True, capture_output=True, text=True)
        return True
    except Exception:
        return False


def _print_summary(repo: Path, final_display: str, files: list[str],
                   archive_for_stats: Path, removed: list[Path] | None = None,
                   removed_count: int | None = None) -> None:
    size_mb = archive_for_stats.stat().st_size / 1024 ** 2
    print(f"[pack] root      : {repo}")
    print(f"[pack] zip       : {final_display}")
    print(f"[pack] files     : {len(files)}")
    print(f"[pack] size      : {size_mb:.2f} MiB")
    print(f"[pack] sha256    : {sha256_file(archive_for_stats)}")
    n_removed = len(removed or []) if removed_count is None else removed_count
    print(f"[pack] old zip   : {n_removed} removed"
          + (f" ({', '.join(p.name for p in removed)})" if removed else ""))
    print(f"[pack] git rev   : {git_rev(repo)}{' (dirty)' if git_dirty(repo) else ''}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 代码 zip 打包器")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                    help="仓库根（默认本脚本上一级）")
    ap.add_argument("--out-dir", default=os.environ.get("V4_ZIP_OUT_DIR", DEFAULT_OUT_DIR),
                    help=f"输出目录（默认 {DEFAULT_OUT_DIR}）")
    ap.add_argument("--name", default=None,
                    help="zip 文件名；缺省带 git short sha 与 dirty 标记")
    ap.add_argument("--keep-old", action="store_true",
                    help="保留目标目录旧 zip；默认删除（交付纪律要求清旧包）")
    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    if not (repo / "run_train.sh").is_file():
        print(f"!! {repo} 下没有 run_train.sh，不是 v4 仓库根", file=sys.stderr)
        return 2

    files = collect_files(repo)
    if not files:
        print(f"!! 没有收集到任何文件：{repo}", file=sys.stderr)
        return 3

    if args.name:
        name = args.name if args.name.lower().endswith(".zip") else args.name + ".zip"
    else:
        rev = git_rev(repo)
        dirty = "_dirty" if git_dirty(repo) else ""
        name = f"Petroleum-Competition_v4_{rev}{dirty}.zip"

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_path = out_dir / name

    # 正常 Linux / 可写挂载：先清旧 zip，再原子写新包。
    if _is_writable_dir(out_dir):
        removed = [] if args.keep_old else clean_old_zips(out_dir)
        build_zip(repo, out_path, files)
        _print_summary(repo, str(out_path), files, out_path, removed=removed)
        return 0

    # WSL 的 /mnt/d 只读：先在 WSL 临时目录打包，再走 powershell.exe 复制到 D:。
    win_dir = _wsl_to_windows_path(out_dir)
    if win_dir is None or not _powershell():
        print(f"!! 输出目录不可写，且无法回退到 Windows 复制：{out_dir}", file=sys.stderr)
        return 4

    # 注意：WSL 的 /tmp 是 tmpfs，Windows 侧 \\wsl.localhost 可能看不到；
    # 必须把临时 zip 放在仓库内（.v4cache 已被 gitignore，且 Windows 可读）。
    tmp_base = repo / ".v4cache"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="pack_zip_", dir=str(tmp_base))
    tmpdir = Path(tmpdir_obj.name)
    try:
        removed_win: list[Path] = []
        removed_win_count = 0
        if not args.keep_old:
            removed_win_count = _clean_windows_zips(win_dir)
            if removed_win_count < 0:
                print("!! 无法清除 Windows 目标目录旧 zip", file=sys.stderr)
                return 6
        tmp_zip = tmpdir / name
        build_zip(repo, tmp_zip, files)
        win_target = win_dir + "\\" + name
        if not _copy_to_windows_via_powershell(tmp_zip, win_target):
            print("!! powershell.exe 复制到 Windows D: 失败", file=sys.stderr)
            return 5
        _print_summary(repo, win_target + "  (Windows)", files, tmp_zip,
                       removed=removed_win, removed_count=removed_win_count)
    finally:
        tmpdir_obj.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
