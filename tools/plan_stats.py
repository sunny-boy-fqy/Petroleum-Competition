#!/usr/bin/env python3
"""计划行数统计（唯一事实源，供文档引用）。

**为什么需要它**：一审/二审两次出现文档手写行数与实际不符（R2-H6）。
本工具用 shell 展开顺序统计，输出可被 CI/提交前检查复算的数字。

    python3 v4/tools/plan_stats.py [--json reports/E0_plan_stats.json] [--check <md 文件>]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]


def count(pattern: str) -> dict:
    files = sorted(V4.glob(pattern))
    total = 0
    per: dict[str, int] = {}
    for f in files:
        n = len(f.read_text(encoding="utf-8").splitlines())
        per[str(f.relative_to(V4))] = n
        total += n
    return {"pattern": pattern, "files": len(files), "lines": total, "per_file": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--check", default=None,
                    help="校验某个 md 文件中出现的行数声明是否与实际一致")
    args = ap.parse_args()

    main_p = count("PLAN.md")
    stage = count("E*/PLAN.md")
    p_level = count("E*/P*/PLAN.md")
    total = main_p["lines"] + stage["lines"] + p_level["lines"]
    payload = {
        "total_plan_files": main_p["files"] + stage["files"] + p_level["files"],
        "total_lines": total,
        "main": {"files": main_p["files"], "lines": main_p["lines"]},
        "stage": {"files": stage["files"], "lines": stage["lines"],
                  "avg": stage["lines"] // max(stage["files"], 1)},
        "p_level": {"files": p_level["files"], "lines": p_level["lines"],
                    "avg": p_level["lines"] // max(p_level["files"], 1)},
        "per_file": {**main_p["per_file"], **stage["per_file"], **p_level["per_file"]},
    }
    print(f"总计划 PLAN.md      : {main_p['lines']} 行")
    print(f"阶段计划 E*/PLAN.md : {stage['files']} 份 / {stage['lines']} 行 "
          f"(均 {stage['lines'] // max(stage['files'],1)})")
    print(f"P 级计划 E*/P*/     : {p_level['files']} 份 / {p_level['lines']} 行 "
          f"(均 {p_level['lines'] // max(p_level['files'],1)})")
    print(f"合计                : {payload['total_plan_files']} 份 / {total} 行")

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = V4 / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json -> {out}")

    rc = 0
    if args.check:
        md = Path(args.check)
        text = md.read_text(encoding="utf-8")
        # 检查是否出现与实测不符的魔法数字
        claims = {
            "阶段": stage["lines"], "P 级": p_level["lines"], "合计": total,
        }
        found = re.findall(r"([\d,]{3,})\s*行", text)
        nums = {int(x.replace(",", "")) for x in found}
        stale = {k: v for k, v in claims.items() if v not in nums}
        if stale:
            print(f"WARN: {md} 未声明或声明不一致 -> {stale}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
