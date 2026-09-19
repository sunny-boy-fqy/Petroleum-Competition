#!/usr/bin/env python3
"""校验 versions/status.json 与实物一致（审查 H3）。

    python3 v4/tools/check_status.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
VALID = {"pending", "in_progress", "done", "no_go", "blocked"}
VALID_STAGE = VALID | {"done_local_pending_cloud_env"}


def main() -> int:
    st = json.loads((V4 / "versions" / "status.json").read_text(encoding="utf-8"))
    errs: list[str] = []
    for s in st["stages"]:
        if s["status"] not in VALID_STAGE:
            errs.append(f"{s['stage']}: illegal stage status {s['status']!r}")
        for p in s.get("p", []):
            if p.get("status") not in VALID:
                errs.append(f"{s['stage']}/{p.get('id')}: illegal P status {p.get('status')!r}")
            if p.get("status") == "done" and not p.get("evidence"):
                errs.append(f"{s['stage']}/{p.get('id')}: done 但无 evidence")
        # P 计划文件必须存在
        for p in s.get("p", []):
            f = V4 / s["stage"] / p["id"] / "PLAN.md"
            if not f.is_file():
                errs.append(f"missing plan file: {f.relative_to(V4)}")
    # 无孤儿 P 目录
    for d in sorted(V4.glob("E*/P*")):
        if d.is_dir() and not (d / "PLAN.md").is_file():
            errs.append(f"orphan P dir (no PLAN.md): {d.relative_to(V4)}")
    for e in errs:
        print("FAIL:", e)
    print(f"阶段数: {len(st['stages'])}  P 数: {sum(len(s.get('p', [])) for s in st['stages'])}")
    print("RESULT:", "OK" if not errs else "FAIL")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
