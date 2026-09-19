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
    # 计划行数与文档声明一致（R2-H6）
    import subprocess, json as _json
    try:
        js = V4 / "reports" / "E0_plan_stats.json"
        subprocess.run([sys.executable, str(V4 / "tools" / "plan_stats.py"),
                        "--json", str(js)], capture_output=True, text=True, check=True)
        tot = int(_json.loads(js.read_text(encoding="utf-8"))["total_lines"])
    except Exception as exc:  # 统计失败不阻塞
        tot = None
        errs.append(f"plan_stats failed: {exc!r}")
    if tot is not None:
        plan_md = (V4 / "PLAN.md").read_text(encoding="utf-8")
        if f"{tot:,} 行" not in plan_md and f"{tot} 行" not in plan_md:
            errs.append(f"PLAN.md 未声明实测计划总行数 {tot:,}（运行 tools/sync_plan_stats.py）")
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
            # R2-M7：done 的 P 其 evidence 文件必须存在
            if p.get("status") == "done":
                for ev in p.get("evidence", []):
                    ev = ev.split("::")[0].strip().strip("`")
                    if ev.startswith("$"):
                        continue          # 运行时目录（/data）不校验
                    if not (V4 / ev).exists():
                        errs.append(f"{s['stage']}/{p['id']}: evidence not found: {ev}")
        # 阶段 Gate 报告文件必须存在
        g = s.get("gate") or {}
        for key in ("local", "cloud"):
            gk = g.get(key) if isinstance(g, dict) else None
            if isinstance(gk, dict) and gk.get("report"):
                if not (V4 / gk["report"]).is_file():
                    errs.append(f"{s['stage']}: gate report missing: {gk['report']}")
            elif isinstance(g, dict) and g.get("report"):
                if not (V4 / g["report"]).is_file():
                    errs.append(f"{s['stage']}: gate report missing: {g['report']}")
        # stage=done 时不得有 pending 的 P
        if s.get("status") == "done":
            bad = [p["id"] for p in s.get("p", []) if p.get("status") != "done"]
            if bad:
                errs.append(f"{s['stage']}: status=done 但 P {bad} 未 done")
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
