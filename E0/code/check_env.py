#!/usr/bin/env python3
"""E0 环境自检：Ascend 910B / CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64。

目标平台（2026-09-20 变更，**唯一事实源 = `src/hardware.py`**）
-------------------------------------------------------------
规格 `Ascend910B-1-64G`：Ascend 910B × 1（64 GB HBM）/ 4000m vCPU / 16 GiB RAM /
64 GiB 磁盘；镜像 = CANN 8.3rc2 + PyTorch 2.8.0 + torch_npu 2.8.0 + Python 3.11 / **arm64**。

三层口径（沿用 R4-B1 的教训：**不要把某个具体小版本钉成硬门禁**）
----------------------------------------------------------------
| 项 | hard（会阻塞训练） | warn（只提示） |
|---|---|---|
| Python | major.minor == 3.11 | micro 差异 |
| torch | 可导入且 major.minor == 2.8 | 与声明值 `2.8.0` 完全相等 |
| torch_npu | 可导入且 major.minor == 2.8（**NPU 上缺它就没有 `torch.npu`**） | 与声明值完全相等 |
| CANN | 可探测且 major.minor == (8, 3) | 与声明值 `8.3rc2` 归一化后相等（rc 后缀漂移不阻塞） |
| 加速器 | `torch.npu.is_available()` 为真且设备名含 910 | — |
| 架构 | `uname -m` ∈ {aarch64, arm64}（仅目标机；本机开发用开关降级） | — |
| bf16 | 910B 上真实跑一次 bf16 小算子 | — |

历史教训：四审时把版本写死成 torch 2.4.0+cu124，代码便拿 `torch.version.cuda`
硬比驱动声明值 → 云端 `run_train.sh --mode env` 必然 `[FAIL]`→`exit 11`，
`E0_env.json`/`E0_disk_budget.json` 永远产不出来、`E0_cloud_gate` 永远 blocked。
因此这里始终是**声明值 + 可接受集合 + 硬底线**三层，而不是钉死具体小版本。

CUDA 路径**保留**（`check_torch_and_accel` 会按实际加速器分支）：若将来换回 NVIDIA
机器，`torch.version.cuda` 的 hard 底线仍是 "CUDA-enabled wheel + major == 12"。


设计原则
--------
1. 只依赖标准库 + （可选的）torch/torch_npu；**不 import numpy/pandas**，保证在本机无 NPU、
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
   仅用于提示版本漂移，不作为门禁；真实版本一致性由 `versions/locks/cloud_frozen.txt` 保证。

用法
----
    python E0/code/check_env.py                      # 人类可读
    python E0/code/check_env.py --json reports/E0_env.json
    python E0/code/check_env.py --min-free-gb 8      # 磁盘门槛（默认 8）
    python E0/code/check_env.py --allow-non-target-device   # 本机开发：设备/架构/torch 降为 warn
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import hardware as HW  # noqa: E402

EXPECTED_PY = HW.PLATFORM["python"]
EXPECTED_TORCH = HW.PLATFORM["torch"]
EXPECTED_TORCH_NPU = HW.PLATFORM["torch_npu"]
EXPECTED_CANN = HW.PLATFORM["cann"]
EXPECTED_ARCH = HW.PLATFORM["arch"]
TARGET_ACCELERATOR = HW.PLATFORM["accelerator"]
ACCEL_MODEL = HW.PLATFORM["accelerator_model"]

# CUDA 兜底口径（仅当实际加速器是 cuda 时使用；见模块 docstring）
ACCEPTED_CUDA_RUNTIME_MAJOR = 12
ACCEPTED_CUDA_RUNTIMES = ((12, 8), (12, 6))
EXPECTED_CUDA_RUNTIME = (12, 8)
MIN_CUDA_DRIVER = (12, 8)

MIN_FREE_GB_DEFAULT = 8.0
DISK_BUDGET_GB = float(HW.PLATFORM["disk_gb"])
RAM_BUDGET_GB = float(HW.PLATFORM["ram_gb"])

# 依赖分档（R3 修复 + R5-M1）。
#   REQUIRED_PY_DEPS：训练/分析主路径硬依赖；profile=full 且缺失 -> hard。
#   OPTIONAL_PY_DEPS：有文档化降级路径；profile=full 且缺失 -> warn + degraded_paths。
# **本表版本一律不钉死**（值为 None 表示"只查是否存在，不比较版本"）。
# 注意范围：这句话只覆盖**额外轻量包**；`torch`/`torch_npu`/CANN/Python 由镜像**硬约束**
# （`torch_version`/`torch_npu_version`/`cann_version` 是 hard 检查，声明值见 src/hardware.py），
# 两者不矛盾。
# torch 的 wheel 本身
# 不依赖 numpy（2.7.1/2.8.0 的 Requires-Dist 均无 numpy），因此 numpy 也在 required 里由 pip 补装；
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
        "--allow-non-target-device",
        "--allow-non-a100",              # 兼容旧名（四审时期的叫法）
        dest="allow_non_target_device",
        action="store_true",
        help="把设备/架构/torch 检查从 hard 降为 warn（本机开发机用；不影响依赖分档）",
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


def parse_cann_version_from_text(text: str) -> str | None:
    """从 CANN 的版本文本里解析版本号（纯函数，可单测）。

    支持三类来源的写法：
      - `version.cfg` / `ascend_toolkit_install.info`：`version=8.3.RC2` / `Version=8.3.rc2`
      - `npu-smi info`：`CANN Version: 8.3.RC2`
      - `torch_npu.version.cann`：`8.3.rc2`
    没有匹配时返回 None（调用方判失败，不静默放过）。
    """
    t = text or ""
    m = re.search(r"(?:cann[_\s]*version|version)\s*[:=]\s*([0-9]+\.[0-9][0-9A-Za-z.\-]*)", t, re.I)
    if m:
        return HW.normalize_cann(m.group(1))
    m = re.search(r"\b(\d+\.\d+(?:\.\s*rc\d+)?)\b", t, re.I)
    return HW.normalize_cann(m.group(1)) if m else None


def _cann_from_files() -> str | None:
    """从 CANN 安装目录里的 version 文件读取（不依赖 torch_npu）。"""
    cands: list[Path] = []
    for env in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        v = os.environ.get(env)
        if v:
            cands += [Path(v) / "version.cfg", Path(v) / "ascend_toolkit_install.info",
                      Path(v).parent / "version.cfg"]
    cands += [Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
              Path("/usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info")]
    for c in cands:
        try:
            if c.is_file():
                got = parse_cann_version_from_text(c.read_text(encoding="utf-8", errors="ignore"))
                if got:
                    return got
        except Exception:                      # pragma: no cover - 环境相关
            continue
    return None


def query_cann_version() -> tuple[str | None, str]:
    """尽力探测 CANN 版本，返回 `(version, source)`。

    顺序：`torch_npu.version.cann` → CANN 安装目录 version 文件 → `npu-smi info`。
    全部失败返回 `(None, "unavailable")`。
    """
    try:
        import torch_npu                        # noqa: PLC0415
        v = getattr(getattr(torch_npu, "version", None), "cann", None)
        if v:
            return HW.normalize_cann(v), "torch_npu.version.cann"
    except Exception:
        pass
    v = _cann_from_files()
    if v:
        return v, "ascend_toolkit/version.cfg"
    exe = shutil.which("npu-smi")
    if exe:
        try:
            proc = subprocess.run([exe, "info"], capture_output=True,
                                  text=True, timeout=10, check=False)  # noqa: S603
            v = parse_cann_version_from_text(proc.stdout or "")
            if v:
                return v, "npu-smi info"
        except Exception:                      # pragma: no cover
            pass
    return None, "unavailable"


def check_arch(rep: Report, allow_non_target: bool) -> dict:
    """编译架构：目标机必须是 aarch64（arm64 轮子）。"""
    level = "warn" if allow_non_target else "hard"
    machine = HW.norm_arch()
    ok = HW.arch_matches(machine)
    rep.add("machine_arch", ok, level,
            f"uname -m={machine} (目标 {EXPECTED_ARCH}；arm64/aarch64 视为一致"
            + ("；本机开发模式只 warn" if allow_non_target else "") + ")")
    return {"machine": machine, "expected": EXPECTED_ARCH, "ok": bool(ok)}


def _bf16_probe(torch_mod, accel: str) -> bool:
    """真实跑一次 bf16 小算子（910B 支持 bf16；报告必须来自实测而不是假定）。"""
    try:
        dev = torch_mod.device(HW.device_string(accel))
        a = torch_mod.ones((8, 8), dtype=torch_mod.bfloat16, device=dev)
        b = torch_mod.ones((8, 8), dtype=torch_mod.bfloat16, device=dev)
        c = a @ b
        return float(c.float().sum().item()) == 64.0 * 1.0 and str(c.dtype).endswith("bfloat16")
    except Exception:
        return False


def check_torch_and_accel(rep: Report, allow_non_target: bool) -> dict:
    """torch / torch_npu / CANN / 加速器设备检查（hard 与 warn 三层，见模块 docstring）。"""
    info: dict = {"torch_available": False, "target_accelerator": TARGET_ACCELERATOR,
                  "platform": HW.describe()}
    level = "warn" if allow_non_target else "hard"
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on environment
        rep.add("torch_import", False, level, f"cannot import torch: {exc!r}")
        return info
    info["torch_available"] = True
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = getattr(torch.version, "cuda", None)

    want_mm = HW.version_major_minor(EXPECTED_TORCH)
    got_mm = HW.version_major_minor(torch.__version__)
    rep.add("torch_version", got_mm == want_mm, level,
            f"torch {torch.__version__} (hard: major.minor=={want_mm[0]}.{want_mm[1]})")
    rep.add("torch_version_declared", torch.__version__.split("+")[0] == EXPECTED_TORCH, "warn",
            f"torch {torch.__version__} vs 声明 {EXPECTED_TORCH}（patch/构建串漂移不阻塞）")

    # torch_npu：NPU 路径的硬依赖（它负责注册 torch.npu）
    try:
        import torch_npu  # noqa: PLC0415
        tn_ver = getattr(torch_npu, "__version__", "?")
        info["torch_npu_version"] = tn_ver
        rep.add("torch_npu_version", HW.version_major_minor(tn_ver) == want_mm, level,
                f"torch_npu {tn_ver} (hard: major.minor=={want_mm[0]}.{want_mm[1]}，"
                f"必须与 torch 同小版本)")
        rep.add("torch_npu_version_declared", str(tn_ver).split("+")[0] == EXPECTED_TORCH_NPU,
                "warn", f"torch_npu {tn_ver} vs 声明 {EXPECTED_TORCH_NPU}")
    except Exception as exc:
        info["torch_npu_version"] = None
        rep.add("torch_npu_version", False, level,
                f"cannot import torch_npu: {exc!r}（目标机为 Ascend NPU，缺它则 torch.npu 不可用）")

    cann, cann_src = query_cann_version()
    info["cann_version"] = cann
    info["cann_source"] = cann_src
    want_cann_mm = tuple(HW.PLATFORM["cann_accepted_major_minor"])
    cann_mm = HW.cann_major_minor(cann) if cann else (-1, -1)
    rep.add("cann_version", cann is not None and cann_mm == want_cann_mm, level,
            f"CANN {cann} (from {cann_src}; hard: major.minor=={want_cann_mm[0]}.{want_cann_mm[1]})")
    rep.add("cann_version_declared", HW.normalize_cann(cann) == HW.normalize_cann(EXPECTED_CANN),
            "warn", f"CANN {cann} vs 声明 {EXPECTED_CANN}（rc/补丁漂移不阻塞）")

    accel = HW.detect_accelerator(torch)
    info["accelerator"] = accel
    info["accelerator_is_target"] = accel == TARGET_ACCELERATOR
    rep.add("accelerator_available", accel == TARGET_ACCELERATOR, level,
            f"detect_accelerator={accel}（目标 {TARGET_ACCELERATOR}；"
            f"torch.npu.is_available 决定 NPU 路径）")
    if accel == "cpu":
        info["torch_source"] = "platform image (do NOT pip install torch/torch_npu)"
        return info

    frontend = torch.npu if accel == "npu" else torch.cuda
    try:
        name = frontend.get_device_name(0)
        props = frontend.get_device_properties(0)
        total_gb = float(getattr(props, "total_memory", 0)) / (1024 ** 3)
        info.update({"device_name": name, "device_total_gb": round(total_gb, 1),
                     "device_count": int(frontend.device_count())})
    except Exception as exc:                   # pragma: no cover - 平台相关
        info["device_query_error"] = repr(exc)
        rep.add("device_query", False, level, f"无法读取设备信息：{exc!r}")
        return info
    if accel == "npu":
        rep.add("device_is_910b", ACCEL_MODEL.lower() in str(name).lower(), level,
                f"{name}（目标 {ACCEL_MODEL}）{total_gb:.1f} GiB HBM；count={info['device_count']}")
    else:
        cap = torch.cuda.get_device_capability(0)
        info["gpu_capability"] = list(cap)
        rep.add("device_is_910b", False, "warn",
                f"{name} sm_{cap[0]}{cap[1]} —— 实际是 CUDA 机器，与声明的 Ascend 目标不符")
    bf16 = _bf16_probe(torch, accel)
    info["bf16_supported"] = bf16
    rep.add("bf16_supported", bf16, level, f"实测 {accel} 上 bf16 matmul={bf16}")
    if accel == "cuda":
        cuda_ver = getattr(torch.version, "cuda", None)
        declared = _mm("%d.%d" % EXPECTED_CUDA_RUNTIME)
        if cuda_ver:
            rep.add("cuda_runtime_version",
                    _mm_tuple(cuda_ver)[0] == ACCEPTED_CUDA_RUNTIME_MAJOR, "warn",
                    f"torch.version.cuda={cuda_ver}（CUDA 兜底口径：major=="
                    f"{ACCEPTED_CUDA_RUNTIME_MAJOR}；声明 {declared}）")
        drv = query_cuda_driver()
        info["cuda_driver_version"] = drv
        rep.add("cuda_driver_version", bool(drv) and _mm_tuple(drv) >= MIN_CUDA_DRIVER, "warn",
                f"nvidia-smi CUDA Version={drv}（advisory）")
    info["torch_source"] = "platform image (do NOT pip install torch/torch_npu)"
    return info


def check_torch(rep: Report, allow_non_target: bool) -> dict:
    """向后兼容别名（旧调用点/单测仍可用）。"""
    return check_torch_and_accel(rep, allow_non_target)


def check_py_deps(rep: Report, allow_non_target_device: bool,
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
                # review R7：level 描述的是"这项检查若失败有多严重"，因此 required 依赖
                # 通过时也应记为 required_level（此前一律记 "warn"，语义误导）。
                rep.add(f"dep_{mod}", True, "warn" if optional else required_level,
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
        # 本机磁盘通常远大于目标机的 64 GiB；只是提示，不判定失败
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

    H4 修正：支持云端布局（代码在 /code/workspace/<仓库名>、数据在 /data/v4/data）。
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
    arch_info = check_arch(rep, args.allow_non_target_device)
    torch_info = check_torch_and_accel(rep, args.allow_non_target_device)
    deps_info, degraded_paths = check_py_deps(rep, args.allow_non_target_device, args.profile)
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
            "expected": HW.describe(),
            "allow_non_target_device": args.allow_non_target_device,
            "arch": arch_info,
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
