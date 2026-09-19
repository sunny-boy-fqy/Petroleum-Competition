#!/usr/bin/env python3
"""v4 推理入口（官方口径）。

    python predict.py --data_dir ./data --output result.json
    python predict.py --list-versions
    python predict.py --use-version CONST --data_dir ./data/test --output /tmp/r.json
    python predict.py --data_dir ./data/test --output /tmp/r.json --validate-only

设计约束（PLAN.md §9.2）：
  - 只读 `--data_dir`，不联网、不训练、不写 `--output` 以外的路径；
  - 推理主路径 **CPU-only**、确定性；
  - 输出的 SW 保持训练标签尺度（**禁止裁剪到 [0,1]**）；PERM 严格为正；
  - 生成后自动跑 `validate_payload` 契约校验。

当前实现状态：模型尚未训练，因此除 `CONST`（常数基线，用于契约自检）之外的版本
都会**明确报错**，不会静默输出无意义结果。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parent
sys.path.insert(0, str(V4))

from src import constants as C                      # noqa: E402
from src.data import parse as P                     # noqa: E402
from src.inference import contract as CT            # noqa: E402
from src.portability import describe                # noqa: E402
from src.versioning import registry as REG          # noqa: E402


# ---------------------------------------------------------------- 版本表
def _versions() -> dict[str, dict]:
    """可运行版本表：来自 `versions/registry.json`（审查 M5：不再硬编码）。"""
    return REG.versions()


DEFAULT_VERSION = "PD1"


# ---------------------------------------------------------------- 预测器
def predict_const(test_dir: Path) -> dict:
    """常数基线预测：为每口井逐行输出占位常量。"""
    result_data = []
    for rec in P.load_split(test_dir, with_targets=False):
        preds = [
            {
                "depth": round(float(d), C.DEPTH_DECIMALS),
                "POR": C.PLACEHOLDER["POR"],
                "PERM": C.PLACEHOLDER["PERM"],
                "SW": C.PLACEHOLDER["SW"],
            }
            for d in list(rec.depth)
        ]
        result_data.append({"logId": rec.well_id, "predictions": preds})
    return result_data


PREDICTORS = {"CONST": predict_const}


def build_payload(version: str, data_dir: Path, model_name: str | None = None) -> dict:
    info = _versions()[version]
    if not info["available"]:
        raise SystemExit(
            f"[predict] version '{version}' is registered but NOT trained yet.\n"
            f"  desc: {info['desc']}\n"
            "  先完成对应阶段（见 v4/PLAN.md §七）后再运行；"
            "如需校验提交契约，请用 --use-version CONST。"
        )
    fn = PREDICTORS[version]
    return {
        "modelId": "",
        "modelName": model_name or f"v4-{version}",
        "version": "1.0",
        "resultData": fn(data_dir),
    }


def cmd_list_versions() -> int:
    vs = _versions()
    for line in REG.list_lines():
        print(line)
    print(f"default: {DEFAULT_VERSION}  (可用: {REG.available_versions()})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 reservoir-parameter predictor")
    ap.add_argument("--list-versions", action="store_true")
    ap.add_argument("--use-version", default=None)
    ap.add_argument("--data_dir", "--data-dir", dest="data_dir", default=None,
                    help="官方参数名 --data_dir（同时兼容 --data-dir）")
    ap.add_argument("--output", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--expected-rows", type=int, default=C.EXPECTED_N_TEST_ROWS)
    ap.add_argument("--expected-wells", type=int, default=C.EXPECTED_N_TEST_WELLS)
    ap.add_argument("--validate-only", action="store_true",
                    help="只校验 --data_dir 是否可用，不写结果")
    ap.add_argument("--print-deps", action="store_true")
    args = ap.parse_args(argv)

    if args.print_deps:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0

    if args.list_versions:
        return cmd_list_versions()

    if args.data_dir is None:
        ap.error("--data_dir is required (official CLI, rules.md §6.3)")

    data_dir = Path(args.data_dir)
    # 官方 --data_dir 可能指向 data/（内含 test/）或直接指向测试井目录
    if (data_dir / "test").is_dir():
        test_dir = data_dir / "test"
    elif any(data_dir.glob("*.txt")):
        test_dir = data_dir
    else:
        ap.error(f"no *.txt wells found under {data_dir}")

    version = args.use_version or DEFAULT_VERSION
    if version not in _versions():
        ap.error(f"unknown version '{version}'; run --list-versions")

    n_test = len(list(test_dir.glob("*.txt")))
    if n_test != args.expected_wells:
        print(f"[predict] WARNING: found {n_test} wells, contract expects "
              f"{args.expected_wells}", file=sys.stderr)

    if args.validate_only:
        print(json.dumps({"data_dir": str(data_dir), "test_dir": str(test_dir),
                          "n_wells": n_test, "version": version,
                          "available": _versions()[version]["available"]},
                         ensure_ascii=False, indent=2))
        return 0

    if args.output is None:
        ap.error("--output is required")

    payload = build_payload(version, test_dir, args.model_name)

    res = CT.validate_payload(payload, test_dir=test_dir, expected_rows=args.expected_rows)
    if not res.ok:
        print("[predict] CONTRACT VALIDATION FAILED:", file=sys.stderr)
        for e in res.errors[:20]:
            print("  -", e, file=sys.stderr)
        return 3

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "version": version,
        "model_name": payload["modelName"],
        "output": str(out),
        "n_wells": res.stats["n_wells"],
        "n_rows": res.stats["n_rows"],
        "contract_ok": res.ok,
        "warnings": res.warnings,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
