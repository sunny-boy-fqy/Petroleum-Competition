#!/usr/bin/env python3
"""E0 环境自检：PyTorch 2.7.1+cu128 / CUDA 12.8 / Python 3.11 / A100 / 30 GB 磁盘。

CUDA 语义（R4-B1 + R5-B1，务必读）
---------------------------------
本项目的 **CUDA 12.8** 指**平台驱动的 CUDA 能力**（`nvidia-smi` 头部
`CUDA Version: 12.8`），**不是** PyTorch 的运行时版本。平台镜像是
`torch==2.7.1+cu128`（见 `versions/locks/cloud.txt`），该 wheel 编译期的
`torch.version.cuda` 是 **12.8**。

历史教训：四审时把版本写死成 torch 2.4.0+cu124，代码便拿 `torch.version.cuda`
硬比驱动声明值 → 云端 `run_train.sh --mode env` 必然 `[FAIL]`→`exit 11`，
`E0_env.json`/`E0_disk_budget.json` 永远产不出来、`E0_cloud_gate` 永远 blocked。
因此现在是**声明值 + 可接受集合 + 硬底线**三层口径，而不是把某个具体小版本钉死：

  - `cuda_runtime_version`（**hard**）：`torch.version.cuda` 存在且 major == 12
    （= "CUDA-enabled wheel + 12.x runtime"，这是唯一真正会破坏训练的条件）；
  - `cuda_runtime_declared`（**warn**）：runtime 是否等于**声明值** 12.8
    （等价地落在 `ACCEPTED_CUDA_RUNTIMES` 内）。cu126 / cu128 两种官方 wheel 都能跑，
    因此小版本漂移只提示、不阻塞 —— 这正是上次 Gate 挂掉的根因；
  - `cuda_driver_version`（**warn/advisory**）：`nvidia-smi` 报的驱动 CUDA 能力 >= 12.8，
    取不到只提示（驱动由平台保证，程序无法也不应修改）。


设计原则
--------
1. 只依赖标准库 + （可选的）torch；**不 import numpy/pandas**，保证在本机无 GPU、
   无 torch 的开发机上也能跑出一份 "环境不满足" 的明确报告，而不是 ImportError。
2. 任何一项 hard 检查失败 -> exit code 非 0，训练脚本应在启动时调用它并拒绝继续。
3. 结果可写成 JSON，供 E0 Gate 的 mandatory_check `env_ok` 读取。
4. 依赖分两档（R3 修复）：
   - **required**（训练/分析主路径）：numpy/pandas/scipy/sklearn/einops。
     `--profile full` 下缺失 = hard。**版本不钉死**（见下）。
   - **optional / degradable**：onnx/onnxruntime/pyarrow 等有文档化降级路径的依赖。
     `--profile full` 下缺失 = warn，并记入 JSON 顶层 `degraded_paths`
     （例如 ONNX 导出不可用时改用原生 torch checkpoint 做推理），**不阻塞训练**。
   注意：本模块 `OPTIONAL_PY_DEPS` 里 optional 依赖的版本号是**参考值（advisory）**，
   仅用于提示版本漂移，不作为门禁；真实版本一致性由 `versions/locks/cloud.txt` 保证。

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
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_PY = (3, 11)
EXPECTED_TORCH = "2.7.1"
# R4-B1 + R5-B1：`torch.version.cuda`（编译期 runtime）≠ 平台驱动的 CUDA 能力。
#   - 声明值：镜像 `torch==2.7.1+cu128` 的 runtime = 12.8；
#   - 硬底线：只要是 CUDA-enabled wheel 且 major == 12（12.x runtime 都能跑本项目的算子）；
#   - 可接受集合：cu128 / cu126 两种官方 wheel 都接受，小版本漂移只 warn（见模块 docstring）。
EXPECTED_CUDA_RUNTIME = (12, 8)                  # 声明值（warn 判定）
ACCEPTED_CUDA_RUNTIME_MAJOR = 12                 # hard：runtime major
ACCEPTED_CUDA_RUNTIMES = ((12, 8), (12, 6))      # cu128 / cu126 官方 wheel
MIN_CUDA_DRIVER = (12, 8)                        # advisory：nvidia-smi 的 "CUDA Version"
MIN_FREE_GB_DEFAULT = 8.0
DISK_BUDGET_GB = 30.0

# 依赖分档（R3 修复 + R5-M1）。
#   REQUIRED_PY_DEPS：训练/分析主路径硬依赖；profile=full 且缺失 -> hard。
#   OPTIONAL_PY_DEPS：有文档化降级路径；profile=full 且缺失 -> warn + degraded_paths。
# **版本一律不钉死**（值为 None 表示"只查是否存在，不比较版本"）：torch 的 wheel 本身
# 不依赖 numpy（2.7.1 的 Requires-Dist 无 numpy），因此 numpy 也在 required 里由 pip 补装；
# 把某个具体小版本写成硬约束会在
# 镜像升级时误报。精确版本一致性由 `versions/locks/cloud_frozen.txt`（云端
# `pip freeze` 回填）保证，本模块只做**存在性**门禁。
REQUIRED_PY_DEPS: dict[str, str | None] = {
    "numpy": None,
    "pandas": None,
    "scipy": None,
    "sklearn": None,
    "einops": None,
}
OPTIONAL_PY_DEPS: dict[str, str | None] = {
    # 分片缓存是 .npz（numpy），不需要 pyarrow；仅当未来改用 parquet 才需要
    "pyarrow": None,
    # CPU 推理的兜底路径；主路径是 torch.load(map_location="cpu")
    "onnx": None,
    "onnxruntime": None,
    # 平台"迭代曲线"观测（只观测，不参与模型选择）
    "tensorboard": None,
}
# 缺失 optional 依赖时启用的降级路径（写进 JSON 的 degraded_paths）
DEGRADATION_PATHS = {
    "pyarrow": "parquet-based shards disabled; the .npz shard cache is used instead",
    "onnx": "ONNX export/serving path disabled; use the native torch checkpoint for inference",
    "onnxruntime": "ONNX runtime unavailable; native torch/CPU inference path is used instead",
    "tensorboard": "TensorBoard curves unavailable; JSONL scalars under $TENSORBOARD_LOGDIR are used",
}

FOLDS_FALLBACK_RELPATH = ("versions", "reference", "v1_well_folds.json")


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


def _mm(text) -> str:
    """把版本串规范成 `major.minor`（用于比较与展示）。"""
    return ".".join(str(text).split(".")[:2])


def _mm_tuple(text) -> tuple[int, int]:
    parts = str(text).split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (-1, -1)


def parse_cuda_driver_from_smi(text: str) -> str | None:
    """从 `nvidia-smi` 输出里解析**驱动 CUDA 能力**（头部 `CUDA Version: 12.6`）。

    纯函数（可单测）：给不定格式的 smi 文本，返回 `"12.6"` 或 `None`。
    """
    m = re.search(r"CUDA\s+Version\s*:\s*(\d+\.\d+)", text or "")
    return m.group(1) if m else None


def query_cuda_driver(timeout: float = 10.0) -> str | None:
    """调用 `nvidia-smi` 查询驱动 CUDA 能力；任何失败都返回 None（advisory）。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(                 # noqa: S603 - 固定可执行名，无 shell
            [exe], capture_output=True, text=True, timeout=timeout, check=False)
    except Exception:                          # pragma: no cover - 依赖环境
        return None
    return parse_cuda_driver_from_smi(proc.stdout or "")


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

    # R4-B1 + R5-B1：hard 只要求"CUDA-enabled wheel + 12.x runtime"；
    # 是否等于声明值（12.8）另行 warn，避免再次把 Gate 钉死在某个 wheel 小版本上。
    cuda_ver = getattr(torch.version, "cuda", None)
    declared = _mm("%d.%d" % EXPECTED_CUDA_RUNTIME)
    accepted = tuple("%d.%d" % v for v in ACCEPTED_CUDA_RUNTIMES)
    if cuda_ver:
        got_mm = _mm(cuda_ver)
        ok_hard = _mm_tuple(cuda_ver)[0] == ACCEPTED_CUDA_RUNTIME_MAJOR
        rep.add("cuda_runtime_version", ok_hard, level,
                f"torch.version.cuda={cuda_ver} (hard 要求 runtime major=="
                f"{ACCEPTED_CUDA_RUNTIME_MAJOR}；声明值 {declared}，可接受 {list(accepted)})")
        info["cuda_runtime_matches_declared"] = got_mm == declared
        rep.add("cuda_runtime_declared", got_mm == declared, "warn",
                f"torch.version.cuda={got_mm} vs 声明 {declared}（cu128 镜像；"
                f"可接受 {list(accepted)} 内的任意值，小版本漂移不阻塞）")
    else:
        info["cuda_runtime_matches_declared"] = False
        rep.add("cuda_runtime_version", False, level,
                "torch.version.cuda is None (CPU-only wheel?)")
        rep.add("cuda_runtime_declared", False, "warn",
                f"无法判定 runtime 是否为 {declared}（torch.version.cuda 为 None）")

    # R4-B1：驱动能力只做 advisory（warn），缺失/偏低都不阻塞训练。
    drv = query_cuda_driver()
    info["cuda_driver_version"] = drv
    want_drv = _mm("%d.%d" % MIN_CUDA_DRIVER)
    if drv:
        rep.add("cuda_driver_version", _mm_tuple(drv) >= MIN_CUDA_DRIVER, "warn",
                f"nvidia-smi CUDA Version={drv} (平台声明 {want_drv}；"
                "驱动能力由平台保证，advisory 不阻塞)")
    else:
        rep.add("cuda_driver_version", False, "warn",
                "nvidia-smi 不可用或未报 CUDA Version（advisory，不影响 hard 判定）")

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


def check_py_deps(rep: Report, allow_non_a100: bool,
                  profile: str = "full") -> tuple[dict, dict]:
    """依赖探测，返回 (已安装版本 info, degraded_paths)。

    分档语义（R3 修复 + R5-M1）：
      - required（numpy/pandas/scipy/sklearn/einops）：
        profile=full 且缺失 -> **hard**（训练/分析主路径不可用）。
      - optional/degradable（pyarrow/onnx/onnxruntime/tensorboard）：
        缺失一律 -> **warn**，并记入 degraded_paths；即使 profile=full 也不阻塞，
        因为 PLAN 有文档化的降级路径（例如 ONNX 不可用时改用原生 torch checkpoint；
        pyarrow 不可用时用 `.npz` 分片；tensorboard 不可用时写 JSONL 标量）。
      - profile=base：required 也降为 warn（保持既有行为；首次 --mode env 依赖尚未安装）。
      - `--allow-non-a100` 只影响 GPU/torch 检查，不影响依赖分档。

    **版本不钉死**（R5-M1）：`REQUIRED_PY_DEPS` 的值是 `None` = 只查存在性。
    torch 的 wheel 不依赖 numpy（2.7.1 Requires-Dist 无 numpy），numpy 也在 required 里
    由 pip 补装；把某个具体小版本
    写成硬约束会在镜像升级时误报。精确版本一致性由 `versions/locks/cloud_frozen.txt`
    （云端 `pip freeze` 回填）保证。
    """
    required_level = "warn" if profile == "base" else "hard"
    info: dict = {}
    degraded: dict[str, str] = {}

    for mod, want in {**REQUIRED_PY_DEPS, **OPTIONAL_PY_DEPS}.items():
        optional = mod in OPTIONAL_PY_DEPS
        try:
            m = __import__(mod)
            got = getattr(m, "__version__", "?")
            info[mod] = got
            if want is None:                       # 只查存在性（R5-M1）
                rep.add(f"dep_{mod}", True, "warn",
                        f"{mod} {got} (存在性检查通过；版本不钉死)"
                        + ("  [optional/advisory]" if optional else ""))
                continue
            major_minor = ".".join(got.split(".")[:2])
            want_mm = ".".join(want.split(".")[:2])
            rep.add(
                f"dep_{mod}",
                major_minor == want_mm,
                "warn",  # 依赖小版本差异不阻塞；用 lock 文件保证一致性
                f"{mod} {got} (lock says {want})"
                + ("  [optional/advisory]" if optional else ""),
            )
        except Exception as exc:
            if optional:
                action = DEGRADATION_PATHS.get(
                    mod, f"{mod} unavailable; documented fallback path is used")
                degraded[mod] = action
                rep.add(f"dep_{mod}", False, "warn",
                        f"missing optional {mod}: {exc!r} -> DEGRADED: {action}")
            else:
                rep.add(f"dep_{mod}", False, required_level, f"missing {mod}: {exc!r}")
    return info, degraded


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
        # R3 修复：实际文件名是 versions/reference/v1_well_folds.json（此前写成 well_folds.json）
        folds = root.joinpath(*FOLDS_FALLBACK_RELPATH)
    n_train = len(list(train_dir.glob("*.txt"))) if train_dir.is_dir() else 0
    n_test = len(list(test_dir.glob("*.txt"))) if test_dir.is_dir() else 0
    # R6-M2：井数契约值必须来自 src/constants.py（唯一事实源，只依赖标准库），
    # 不得在本文件里硬编码 80 / 10 —— 否则常量一改、Gate 口径就会漂移。
    try:
        from src import constants as _C  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - 仓库不完整时明确报错，不静默放行
        raise SystemExit(f"check_env: 无法导入 src.constants（仓库不完整？）：{exc!r}")
    exp_train, exp_test = _C.EXPECTED_N_TRAIN_WELLS, _C.EXPECTED_N_TEST_WELLS
    info.update({"n_train_files": n_train, "n_test_files": n_test, "folds_exists": folds.is_file()})
    rep.add("data_train_80", n_train == exp_train, level,
            f"{train_dir} has {n_train} wells (expect {exp_train})")
    rep.add("data_test_10", n_test == exp_test, level,
            f"{test_dir} has {n_test} wells (expect {exp_test})")
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
    deps_info, degraded_paths = check_py_deps(rep, args.allow_non_a100, args.profile)
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
    print(f"hard failures: {len(hard)}   warnings: {len(warn)}   "
          f"degraded deps: {len(degraded_paths)}")
    if hard:
        print("\nHARD FAILURES -> 训练脚本必须拒绝启动：")
        for c in hard:
            print(f"  - {c['name']}: {c['detail']}")
    if degraded_paths:
        print(f"\nDEGRADED PATHS（{len(degraded_paths)} 个可选依赖缺失 -> 启用降级路径，不阻塞训练）：")
        for mod, action in degraded_paths.items():
            print(f"  - {mod}: {action}")
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
                # R4-B1/R5-B1：明确区分 runtime 与 driver，避免再次拿 torch.version.cuda 比驱动
                "cuda_runtime": _mm("%d.%d" % EXPECTED_CUDA_RUNTIME),
                "cuda_runtime_accepted": [_mm("%d.%d" % v) for v in ACCEPTED_CUDA_RUNTIMES],
                "cuda_runtime_hard_major": ACCEPTED_CUDA_RUNTIME_MAJOR,
                "cuda_driver_min": _mm("%d.%d" % MIN_CUDA_DRIVER),
                "disk_budget_gb": DISK_BUDGET_GB,
            },
            "allow_non_a100": args.allow_non_a100,
            "profile": args.profile,
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch": torch_info,
            "deps": deps_info,
            "degraded_paths": degraded_paths,
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
