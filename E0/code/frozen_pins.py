#!/usr/bin/env python3
"""把 `pip freeze` 的输出收敛成**本项目直接依赖的精确钉子**。

版本策略（为什么需要这个脚本）
------------------------------
`requirements.txt` / `versions/locks/cloud.txt` 刻意**不钉小版本**：包名相同，但
`python 3.11 + 平台镜像`组合下 pip 实际解析出哪个版本，我在开发机上无法预知；
猜错会在镜像构建时失败或引入 ABI 冲突（与四审"把 CUDA runtime 小版本钉死"同类风险）。

真正能钉准的只有**实机事实**：在镜像/训练任务里跑一次 `pip freeze`，把输出写进
`versions/locks/cloud_frozen.txt`，重建镜像或复现提交时用它精确安装。
`E0/code/setup_deps.sh --from-frozen` 即走这条路。

安全性质（白名单式）
--------------------
只输出 `KEEP` 里的包，因此结果**在构造上不可能**包含 `torch` / `torch_npu` /
`nvidia-*` / `cuda-*` / `triton` / `ascend*` / `cann*`（pip 一律不许触碰镜像预装的
加速栈 —— 目标平台已从 CUDA 换成 **Ascend + CANN**）；`pip` / `setuptools` /
`wheel` 与全部传递依赖也会被忽略（它们由 pip 按依赖关系自行解析）。

    python3 v4/E0/code/frozen_pins.py --frozen versions/locks/cloud_frozen.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 直接依赖（required 5 个 + 可选推荐 1 个），与 check_env.REQUIRED_PY_DEPS 对齐
KEEP: tuple[str, ...] = ("numpy", "pandas", "scipy", "scikit-learn", "einops", "tensorboard")
# 禁止由本项目 pip 安装的系列（仅用于回归断言：结果集必须与它们无交集）。
# 单一事实源 = src/hardware.py::FORBIDDEN_INSTALL_PREFIXES（含 torch_npu/npu/ascend/cann）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import hardware as _HW  # noqa: E402

FORBIDDEN: tuple[str, ...] = tuple(_HW.FORBIDDEN_INSTALL_PREFIXES)


def _norm(name: str) -> str:
    """PEP 503 归一化：`scikit_learn` / `Scikit-Learn` / `scikit.learn` 视为同一个包。"""
    return name.strip().lower().replace("_", "-").replace(".", "-")


def parse_freezes(text: str, keep: tuple[str, ...] = KEEP) -> dict[str, str]:
    """从 `pip freeze` 文本里挑出 keep 中的直接依赖，返回 `{归一化名: 发行名==版本}`。"""
    want = {_norm(k): k for k in keep}
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue                      # `-e ...` / `--index-url ...` 之类的行
        name, sep, ver = line.partition("==")
        if not sep:
            continue                      # 无精确版本的（`pkg @ file://...`）跳过
        ver = ver.split(";", 1)[0].split("#", 1)[0].strip()
        if not ver:
            continue
        key = _norm(name)
        if key in want and key not in pins:      # 首次出现优先，不重复
            pins[key] = f"{want[key]}=={ver}"
    return pins


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 pip freeze 提取直接依赖的精确版本")
    ap.add_argument("--frozen", required=True, help="pip freeze 输出文件")
    ap.add_argument("--keep", default=",".join(KEEP),
                    help="逗号分隔的直接依赖白名单（默认 = KEEP）")
    args = ap.parse_args(argv)

    path = Path(args.frozen)
    if not path.is_file():
        print(f"!! frozen lock 不存在：{path}", file=sys.stderr)
        return 2
    keep = tuple(x.strip() for x in args.keep.split(",") if x.strip())
    pins = parse_freezes(path.read_text(encoding="utf-8"), keep)

    for name in keep:
        if _norm(name) not in pins:
            print(f"# 注意：frozen lock 里没有 {name} —— 不会被安装", file=sys.stderr)
    if not pins:
        print(f"!! {path} 里没有任何直接依赖（{', '.join(keep)}）", file=sys.stderr)
        return 1
    print("\n".join(pins[k] for k in sorted(pins)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
