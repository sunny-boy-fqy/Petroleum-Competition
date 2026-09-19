#!/usr/bin/env python3
"""E0 环境自检：PyTorch 2.4.0 / CUDA 12.6 / Python 3.11 / A100 / 30 GB 磁盘。

设计原则
--------
1. 只依赖标准库 + （可选的）torch；**不 import numpy/pandas**，保证在本机无 GPU、
   无 torch 的开发机上也能跑出一份 "环境不满足" 的明确报告，而不是 ImportError。
2. 任何一项 hard 检查失败 -> exit code 非 0，训练脚本应在启动时调用它并拒绝继续。
3. 结果可写成 JSON，供 E0 Gate 的 mandatory_check `env_ok` 读取。

用法
----
    python E0/code/check_env.py                      # 人类可读
    python E0/code/check_env.py --json reports/E0_env.json
    python E0/code/check_env.py --min-free-gb 8      # 磁盘门槛（默认 8）
    python E0/code/check_env.py --allow-non-a100     # 本机开发模式，降级为 warn
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_PY = (3, 11)
EXPECTED_TORCH = "2.4.0"
EXPECTED_CUDA_MAJOR_MINOR = (12, 6)
MIN_FREE_GB_DEFAULT = 8.0
DISK_BUDGET_GB = 30.0


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, level: str, detail: str) -> None:
        self.checks.append(
            {"name": name, "ok": bool(ok), "level": level, "detail": detail}
        )

    def hard_failures(self) -> list[dict]:
        return [c for c in self.checks if c["level"] == "hard" and not c["ok"]]

    def warn_failures(self) -> list[dict]:
        return [c for c in self.checks if c["level"] == "warn" and not c["ok"]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v4 cloud/local environment check")
    p.add_argument("--json", type=str, default=None, help="write JSON report here")
    p.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB_DEFAULT)
    p.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="downgrade GPU/torch checks from hard to warn (local dev machine)",
    )
    p.add_argument(
        "--profile", choices=("base", "full"), default="full",
        help="base: 只检查环境（依赖缺失与数据缺失降为 warn，用于首次 --mode env）；"
             "full: 环境 + 数据健康（数据缺失为 hard，用于 --mode data / 训练前）",
    )
    p.add_argument(
        "--disk-path", action="append", default=None,
        help="额外检查这些挂载点的剩余空间（可多次）。默认还会检查 $V4_DATA_ROOT",
    )
    return p.parse_args()


def check_python(rep: Report) -> None:
    v = sys.version_info
    ok = (v.major, v.minor) == EXPECTED_PY
    rep.add(
        "python_version",
        ok,
        "hard",
        f"python {v.major}.{v.minor}.{v.micro} (expected {EXPECTED_PY[0]}.{EXPECTED_PY[1]})",
    )


def check_torch(rep: Report, allow_non_a100: bool) -> dict:
    info: dict = {"torch_available": False}
    level = "warn" if allow_non_a100 else "hard"
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on environment
        rep.add("torch_import", False, level, f"cannot import torch: {exc!r}")
        return info
    info["torch_available"] = True
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None

    ok_ver = torch.__version__.split("+")[0] == EXPECTED_TORCH
    rep.add(
        "torch_version",
        ok_ver,
        level,
        f"torch {torch.__version__} (expected {EXPECTED_TORCH})",
    )

    cuda_ver = getattr(torch.version, "cuda", None)
    want_cuda = ".".join(str(x) for x in EXPECTED_CUDA_MAJOR_MINOR)
    if cuda_ver:
        got_mm = ".".join(str(cuda_ver).split(".")[:2])
        rep.add("cuda_version", got_mm == want_cuda, level,
                f"torch.version.cuda={cuda_ver} (expected {want_cuda} driver)")
    else:
        rep.add("cuda_version", False, level, "torch.version.cuda is None (CPU-only wheel?)")

    cuda_avail = torch.cuda.is_available()
    info["cuda_available"] = cuda_avail
    rep.add("cuda_available", cuda_avail, level, f"torch.cuda.is_available()={cuda_avail}")
    if not cuda_avail:
        return info

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    info.update({"gpu_name": name, "gpu_capability": list(cap), "gpu_total_gb": round(total_gb, 1)})
    rep.add("gpu_is_a100", cap == (8, 0), level, f"{name} sm_{cap[0]}{cap[1]} {total_gb:.1f} GiB")
    bf16 = bool(torch.cuda.is_bf16_supported())
    info["bf16_supported"] = bf16
    rep.add("bf16_supported", bf16, level, f"torch.cuda.is_bf16_supported()={bf16}")
    info["torch_source"] = "platform image (do NOT pip install torch)"
    return info


def check_py_deps(rep: Report, allow_non_a100: bool, profile: str = "full") -> dict:
    """可选依赖探测。

    profile=base（首次 --mode env，依赖尚未安装）-> warn
    profile=full（安装完成后 / 训练前）           -> hard
    本机 --allow-non-a100 始终 warn。
    """
    level = "warn" if (allow_non_a100 or profile == "base") else "hard"
    info: dict = {}
    expected = {
        "numpy": "1.26.4",
        "pandas": "2.2.3",
        "scipy": "1.13.1",
        "sklearn": "1.5.2",
        "pyarrow": "17.0.0",
        "einops": "0.8.0",
        "onnx": "1.16.2",
        "onnxruntime": "1.18.1",
    }
    for mod, want in expected.items():
        try:
            m = __import__(mod)
            got = getattr(m, "__version__", "?")
            info[mod] = got
            major_minor = ".".join(got.split(".")[:2])
            want_mm = ".".join(want.split(".")[:2])
            rep.add(
                f"dep_{mod}",
                major_minor == want_mm,
                "warn",  # 依赖小版本差异不阻塞；用 lock 文件保证一致性
                f"{mod} {got} (lock says {want})",
            )
        except Exception as exc:
            rep.add(f"dep_{mod}", False, level, f"missing {mod}: {exc!r}")
    return info


def check_disk(rep: Report, path: Path, min_free_gb: float) -> dict:
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    used_gb = usage.used / (1024**3)
    ok = free_gb >= min_free_gb
    rep.add(
        "disk_headroom",
        ok,
        "hard",
        f"{path}: total={total_gb:.1f} GiB used={used_gb:.1f} GiB free={free_gb:.1f} GiB "
        f"(require >= {min_free_gb} GiB)",
    )
    if total_gb > DISK_BUDGET_GB * 1.5:
        # 本机磁盘通常远大于 30 GB；只是提示，不判定失败
        rep.add(
            "disk_budget_context",
            True,
            "warn",
            f"filesystem total {total_gb:.1f} GiB (cloud budget is {DISK_BUDGET_GB:.0f} GiB); "
            "只检查 free >= min_free_gb",
        )
    return {
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "free_gb": round(free_gb, 2),
        "min_free_gb_required": min_free_gb,
    }


def check_repo(rep: Report, root: Path, profile: str = "full") -> dict:
    """数据与折引用是否就位。

    H4 修正：支持云端布局（代码在 /code/workspace/v4、数据在 /data/v4/data）。
    解析顺序：$V4_DATA_ROOT/v4/data -> <v4>/../data；折文件用 src/validation/folds.py 的解析器。
    本机（无 V4_DATA_ROOT）仍走 <v4>/../data。
    """
    info = {}
    import os as _os
    droot = _os.environ.get("V4_DATA_ROOT")
    if droot:
        train_dir = Path(droot) / "v4" / "data" / "train"
        test_dir = Path(droot) / "v4" / "data" / "test"
    else:
        train_dir = root.parent / "data" / "train"
        test_dir = root.parent / "data" / "test"
    info["data_root_env"] = droot
    level = "hard" if profile == "full" else "warn"
    sys.path.insert(0, str(root))
    try:
        from src.validation.folds import find_folds_file  # noqa: PLC0415
        folds = find_folds_file(root)
    except Exception:
        folds = root / "versions" / "reference" / "well_folds.json"
    n_train = len(list(train_dir.glob("*.txt"))) if train_dir.is_dir() else 0
    n_test = len(list(test_dir.glob("*.txt"))) if test_dir.is_dir() else 0
    info.update({"n_train_files": n_train, "n_test_files": n_test, "folds_exists": folds.is_file()})
    rep.add("data_train_80", n_train == 80, level,
            f"{train_dir} has {n_train} wells (expect 80)")
    rep.add("data_test_10", n_test == 10, level,
            f"{test_dir} has {n_test} wells (expect 10)")
    rep.add("well_folds_present", folds.is_file(), level,
            f"{folds} exists={folds.is_file()}")
    return info


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]  # .../v4
    paths: list[Path] = []
    if args.disk_path:
        paths += [Path(x).resolve() for x in args.disk_path]
    paths.append(root)
    import os as _os
    if _os.environ.get("V4_DATA_ROOT"):
        paths.append(Path(_os.environ["V4_DATA_ROOT"]))
    if Path("/data").exists() and Path("/data") not in paths:
        paths.append(Path("/data"))
    # 去重保序
    seen: set[str] = set()
    disk_paths = [p for p in paths if not (str(p) in seen or seen.add(str(p)))]

    rep = Report()
    check_python(rep)
    torch_info = check_torch(rep, args.allow_non_a100)
    deps_info = check_py_deps(rep, args.allow_non_a100, args.profile)
    disk_infos = []
    for dp in disk_paths:
        d = check_disk(rep, dp, args.min_free_gb)
        d["path"] = str(dp)
        disk_infos.append(d)
    disk_info = disk_infos[0] if len(disk_infos) == 1 else disk_infos
    repo_info = check_repo(rep, root, args.profile)

    hard = rep.hard_failures()
    warn = rep.warn_failures()

    print("=" * 78)
    print("v4 environment check")
    print("=" * 78)
    for c in rep.checks:
        mark = "OK  " if c["ok"] else ("FAIL" if c["level"] == "hard" else "WARN")
        print(f"[{mark}] {c['name']:<22} {c['detail']}")
    print("-" * 78)
    print(f"platform : {platform.platform()}")
    print(f"executable: {sys.executable}")
    print(f"hard failures: {len(hard)}   warnings: {len(warn)}")
    if hard:
        print("\nHARD FAILURES -> 训练脚本必须拒绝启动：")
        for c in hard:
            print(f"  - {c['name']}: {c['detail']}")
    print("=" * 78)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expected": {
                "python": f"{EXPECTED_PY[0]}.{EXPECTED_PY[1]}",
                "torch": EXPECTED_TORCH,
                "cuda": f"{EXPECTED_CUDA_MAJOR_MINOR[0]}.{EXPECTED_CUDA_MAJOR_MINOR[1]}",
                "disk_budget_gb": DISK_BUDGET_GB,
            },
            "allow_non_a100": args.allow_non_a100,
            "profile": args.profile,
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch": torch_info,
            "deps": deps_info,
            "disk": disk_info,
            "repo": repo_info,
            "checks": rep.checks,
            "hard_failures": len(hard),
            "warnings": len(warn),
            "passed": len(hard) == 0,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json report -> {out}")

    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
