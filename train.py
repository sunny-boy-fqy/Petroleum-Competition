#!/usr/bin/env python3
"""v4 统一训练入口（rules.md §8.3 要求"仓库内有可运行的训练入口"）。

两条用法
--------
1. **按阶段转发**（推荐；参数原样透传）：

       python train.py --stage E6 --folds all
       python train.py --stage E10 --aggregate full_retrain --epochs 30

   转发到 `E<stage>/code/<脚本>.py`（映射见 `STAGE_SCRIPTS`），子进程退出码原样返回。

2. **跑 v4 配置**（`configs/v4.yaml` 的默认值 + CLI 覆盖）：

       python train.py --config configs/v4.yaml
       python train.py --config configs/v4.yaml --set training.epochs=10 --set model.hidden=64

   只做**浅层键覆盖**（`a.b=c`），覆盖值以字符串交给对应脚本；不做 YAML 语义推断
   （避免"配置写错却静默用默认值"）。不依赖 PyYAML：用一个极小的解析器读关键块。

退出码：0 成功；4 缺数据/配置；5 缺 torch；其它 = 子脚本退出码。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parent
sys.path.insert(0, str(V4))

STAGE_SCRIPTS = {
    "E1": "E1/code/train_row.py",
    "E3": "E3/code/train_seq.py",
    "E4": "E4/code/train_patchtf.py",
    "E5": "E5/code/head_por.py",
    "E6": "E6/code/train_state.py",
    "E7": "E7/code/ablate_loss.py",
    "E8": "E8/code/train_mmoe.py",
    "E10": "E10/code/final_train.py",
}
DEFAULT_CONFIG = V4 / "configs" / "v4.yaml"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v4 训练入口（阶段转发 / 配置默认值）")
    ap.add_argument("--stage", default=None, choices=sorted(STAGE_SCRIPTS),
                    help="要运行的阶段；省略时用 --config 里的默认阶段")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖配置项（可重复），如 --set training.epochs=10")
    ap.add_argument("--list-stages", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="其余参数原样透传给阶段脚本")
    return ap


def parse_flat_yaml(path: Path) -> dict[str, str]:
    """极小 YAML 读取：只支持 `a: v`、`a:\\n  b: v` 与行内 `{k: v, ...}` 的浅层键值。"""
    out: dict[str, str] = {}
    if not Path(path).is_file():
        return out
    section = ""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if ":" not in line:
            continue
        key, _, val = line.strip().partition(":")
        if indent == 0 and not val.strip():
            section = key.strip()
            continue
        full = f"{section}.{key.strip()}" if section else key.strip()
        out[full] = val.strip()
    return out


def apply_overrides(cfg: dict[str, str], sets: list[str]) -> dict[str, str]:
    for item in sets:
        if "=" not in item:
            raise SystemExit(f"[train] --set 需要 KEY=VALUE，got {item!r}")
        k, _, v = item.partition("=")
        cfg[k.strip()] = v.strip()
    return cfg


def to_cli_args(cfg: dict[str, str]) -> list[str]:
    """把配置里**阶段脚本认识**的键翻成 CLI（只翻确定存在的映射，避免拼错静默忽略）。"""
    mapping = {
        "features.version": "--spec",
        "model.hidden": "--hidden",
        "model.layers": "--layers",
        "model.dropout": "--dropout",
        "training.epochs": "--epochs",
        "training.lr": "--lr",
        "training.weight_decay": "--weight-decay",
        "training.seed": "--seed",
        "training.batch_size": "--batch-size",
        "training.amp_dtype": "--amp-dtype",
        "atomic.two_stage": None,              # 布尔开关由脚本自身默认（两阶段恒开）
        "decode.missing_mode": None,           # 只读常量（constants.SCORE_MISSING_MODE）
    }
    args: list[str] = []
    for key, flag in mapping.items():
        if flag is None or key not in cfg:
            continue
        args += [flag, cfg[key]]
    return args


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_stages:
        for k in sorted(STAGE_SCRIPTS):
            print(f"{k:<4} {STAGE_SCRIPTS[k]}")
        return 0
    cfg = apply_overrides(parse_flat_yaml(Path(args.config)), args.set)
    stage = args.stage or ("E10" if cfg.get("aggregate") == "full_retrain" else "E6")
    script = V4 / STAGE_SCRIPTS[stage]
    if not script.is_file():
        print(f"[train] FATAL: 阶段脚本不存在 {script}", file=sys.stderr)
        return 4
    passthrough = [a for a in args.rest if a != "--"]
    cmd = [sys.executable, str(script), *to_cli_args(cfg), *passthrough]
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    print(f"[train] stage={stage}  config={args.config}  "
          f"overrides={len(args.set)}", flush=True)
    return int(subprocess.run(cmd).returncode)


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
