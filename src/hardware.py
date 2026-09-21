"""目标平台硬件画像（**唯一事实源**：Ascend 910B / CANN / torch_npu / arm64）。

为什么单独一个模块
------------------
1. 2026-09-20 平台规格变更：**A100 + CUDA 12.8 + torch 2.7.1/x86_64**
   → **Ascend 910B(64G) + CANN 8.3rc2 + torch 2.8.0 + torch_npu 2.8.0 + py3.11/arm64**。
   镜像版本、设备类型、编译架构、CANN 版本这一组事实被 `check_env.py`（门禁）、
   `training/loop.py`（设备解析）、`docs/` 与生成的计划同时引用，必须有**单一来源**，
   否则改一处漏一处（E0-R2 的"SW 双尺度"就是这么漂移的）。
2. 本模块**只依赖标准库**，因此无 torch / 非 arm64 的开发机也能导入它并跑口径层测试。

数量口径纪律（**必须先读，这里踩过坑**）
---------------------------------------
平台资源行 `Ascend910B-1-64G | Ascend 910B * 1 | 4000m vCPU | 16G | 64Gi` 的三个数**不是一回事**：

| 量 | 值 | 出处 |
|---|---|---|
| **显存（HBM）** | **64 GB** | 资源规格名里的 `-64G` / 资源表"显存"列 `64Gi` |
| **内存（系统 RAM）** | **16 GB** | 资源表"内存"列 `16G` |
| **云盘 `/data` 配额** | **30 GB** | 云盘页面"总容量"（官方文档 §我的云盘：`0 GB / 30GB`，可申请扩容） |

2026-09-20 的规格变更里，agent 一度把**云盘**也写成 64 GiB —— 那会让全部磁盘纪律
（`DISK_BUDGET_GB`、特征缓存上限、checkpoint 滚动窗口、`assert_disk_headroom`）
建立在错误容量上。因此本模块把三者命名为 `device_memory_gb` / `host_ram_gb` /
`cloud_disk_gb`，并由 `tests/test_hardware.py` 同时锁定数值与"文档不得混淆"。

术语纪律（沿用 R4-B1 的教训）
-----------------------------
- `torch==2.8.0` 是**镜像预装的框架版本**（hard：major.minor 一致）；
- `torch_npu==2.8.0` 必须与 torch **同小版本**，否则 `torch.npu` 不可用（hard）；
- `CANN 8.3rc2` 是**运行时/工具包版本**（hard：major.minor == 8.3，rc 后缀只做 warn），
  与"驱动版本"不是一回事 —— 不要把某一个具体小版本钉成硬门禁，
  这与当初"拿 torch.version.cuda 硬比驱动"是同类错误。
- `arm64/aarch64` 是**编译架构**：镜像里的 wheel 必须是 aarch64 轮子；本机 x86_64
  开发机在 `--allow-non-target-device` 下只 warn。
"""
from __future__ import annotations

import platform as _platform

# ---------------------------------------------------------------- 声明值
PLATFORM: dict = {
    "resource_spec": "Ascend910B-1-64G",
    "accelerator": "npu",              # 主加速器种类：npu | cuda | cpu
    "accelerator_vendor": "Huawei Ascend",
    "accelerator_model": "910B",
    "accelerator_count": 1,
    # ⚠️ 三个"看起来都是大小"的量必须分清（2026-09-20 曾把云盘误改成 64 GiB）：
    #   device_memory_gb = **显存**（910B 的 HBM），来自资源规格名 `Ascend910B-1-64G`
    #   host_ram_gb      = **内存**（系统 RAM），平台资源表的"内存"列 = 16G
    #   cloud_disk_gb    = **云盘 /data 配额**（持久存储），平台云盘页显示 30 GB
    "device_memory_gb": 64,
    "cpu_millicores": 4000,
    "host_ram_gb": 16,
    "cloud_disk_gb": 30,
    "arch": "aarch64",                 # 平台机器架构（arm64）
    "os": "linux",
    "python": (3, 11),
    "torch": "2.8.0",
    "torch_npu": "2.8.0",
    "cann": "8.3rc2",
    "cann_accepted_major_minor": (8, 3),
}

# 设备解析优先级（有 NPU 就用 NPU；其次 CUDA；最后 CPU 兜底，便于本机跑链路测试）
ACCELERATOR_PRIORITY: tuple[str, ...] = ("npu", "cuda", "cpu")

# 镜像/依赖层面**绝不 pip 安装**的包名前缀（setup_deps.sh 与 frozen_pins.py 共用）
FORBIDDEN_INSTALL_PREFIXES: tuple[str, ...] = (
    "torch", "torch_npu", "torch-npu", "torch_npu-", "npu", "ascend", "cann",
    "nvidia-", "cuda-", "triton", "apex", "deepspeed", "flash-attn", "xformers",
)


def normalize_cann(text) -> str:
    """把 CANN 版本串规范成小写无分隔形式：`8.3.RC2`/`8.3rc2`/`8.3.rc2` → `8.3rc2`。

    纯函数（可单测）：CANN 在不同来源里的写法不一致
    （`torch_npu.utils.get_cann_version("CANN")`、`npu-smi`、`ascend_*_install.info`），必须先归一化再比较。
    """
    s = str(text or "").strip().lower()
    for junk in (" ", "\t", "-", "_"):
        s = s.replace(junk, "")
    s = s.replace(".rc", "rc").replace(".RC", "rc")
    # 去掉尾部多余的 .0（如 8.3.0 -> 8.3）
    parts = s.split("rc")[0].rstrip(".")
    rc = ("rc" + s.split("rc")[1]) if "rc" in s else ""
    parts = ".".join(p for p in parts.split(".") if p != "")
    if parts.endswith(".0") and rc == "":
        parts = parts[:-2]
    return parts + rc


def cann_major_minor(text) -> tuple[int, int]:
    """`8.3rc2` → `(8, 3)`；无法解析时返回 `(-1, -1)`（调用方判失败，不静默放过）。"""
    s = normalize_cann(text)
    head = s.split("rc")[0]
    parts = head.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        try:
            return int(parts[0]), 0
        except (IndexError, ValueError):
            return (-1, -1)


def norm_arch(machine: str | None = None) -> str:
    """把 `uname -m` 规范成 `aarch64` / `x86_64` / 其它原样返回（小写）。"""
    m = (machine or _platform.machine() or "").strip().lower()
    if m in ("arm64", "aarch64", "armv8l", "armv8"):
        return "aarch64"
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    return m


def arch_matches(machine: str | None = None, expected: str | None = None) -> bool:
    return norm_arch(machine) == norm_arch(expected or PLATFORM["arch"])


def version_major_minor(text) -> tuple[int, int]:
    parts = str(text or "").split("+")[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (-1, -1)


def detect_accelerator(torch_mod=None, try_import_accel: bool = True) -> str:
    """返回实际可用的加速器：`"npu"` / `"cuda"` / `"cpu"` / `"none"`。

    NPU 必须先 `import torch_npu`（它负责把 `torch.npu` 注册进来）；
    `try_import_accel=False` 时跳过该导入 —— 单测用假 torch 对象注入桩实现时用这个开关。
    """
    if torch_mod is None:
        try:
            import torch as torch_mod          # noqa: PLC0415
        except Exception:
            return "none"
    if try_import_accel:
        try:
            import torch_npu                   # noqa: F401,PLC0415
        except Exception:
            pass
    npu = getattr(torch_mod, "npu", None)
    if npu is not None:
        try:
            if bool(npu.is_available()):
                return "npu"
        except Exception:
            pass
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None:
        try:
            if bool(cuda.is_available()):
                return "cuda"
        except Exception:
            pass
    return "cpu"


def device_string(kind: str, index: int = 0) -> str:
    """`("npu", 0)` → `"npu:0"`；`("cpu", _)` → `"cpu"`。"""
    k = str(kind)
    if k in ("cpu", "none"):
        return "cpu"
    return f"{k}:{int(index)}"


def describe() -> dict:
    """可 JSON 序列化的画像摘要（写进 E0_env.json / 报告与文档）。"""
    d = dict(PLATFORM)
    d["python"] = "%d.%d" % PLATFORM["python"]
    d["cann_accepted_major_minor"] = list(PLATFORM["cann_accepted_major_minor"])
    d["accelerator_priority"] = list(ACCELERATOR_PRIORITY)
    d["local_machine"] = norm_arch()
    return d


def is_forbidden_install(name: str) -> bool:
    """包名是否属于"镜像预装的加速栈"（绝不 pip 安装/升级）。"""
    n = str(name or "").strip().lower()
    if not n:
        return False
    base = n.split("==")[0].split(">=")[0].split("<")[0].split("!=")[0].split("[")[0].strip()
    return any(base == p or base.startswith(p) for p in FORBIDDEN_INSTALL_PREFIXES)
