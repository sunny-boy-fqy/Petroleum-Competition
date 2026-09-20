#!/usr/bin/env python3
"""校验 `versions/status.json` 与实物一致（审查 H3 / R3-C2 / R3-C3 / R3-H4 / R3-M4）。

检查项
------
A. **合法性**：stage/P 状态取值合法；`done` 的 P 必须有 evidence，且 evidence 文件存在。
B. **计划行数分项**（R3-C2/M4）：调用 `tools/plan_stats.py` 的**同一份**声明正则，
   校验 `PLAN.md` / `README.md` / `docs/PROJECT_FILES.md` / `status.json` 的阶段、P、平均、总量、
   以及 README 里的 Gate `N/M` 计数。旧实现只查总量，因此「阶段 711 / P 4,982」被漏检。
C. **状态语义一致性**（R3-H4）：
   - `stage.status == pending` ⇒ 其所有 P 必须 `pending`；
   - `stage.status == in_progress` ⇒ 至少一个 P 不是 `pending`；
   - `stage.status == done` ⇒ 所有 P 都 `done`；
   - `project.phase` 必须等于**第一个未 done 的阶段**（E0 云端 Gate 未过时不得指向 E1）。
D. **Gate 报告一致性**（R3-M4）：stage.gate 里声明的 `mandatory_passed/total`
   必须与 `report` 指向的报告里实际 checks 数量一致；`passed` 字段必须与报告一致。
E. **E0/P1 缓存证据**（R3-M5）：`done` 的 P1 必须把 `$V4_CACHE_ROOT/manifest.json` 列入 evidence。
F. 无孤儿 P 目录。

    python3 v4/tools/check_status.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

VALID = {"pending", "in_progress", "done", "no_go", "blocked"}
VALID_STAGE = VALID | {"done_local_pending_cloud_env"}

# 阶段 Gate 报告里"一份 mandatory check 表"可能出现的字段名
_CHECK_FIELDS = ("mandatory_checks", "checks")


def _report_checks(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                             # noqa: BLE001
        return None
    for k in _CHECK_FIELDS:
        v = d.get(k)
        if isinstance(v, dict):
            return v
    return None


def check_plan_counts(errs: list[str]) -> None:
    """B：行数分项 + status 摘要 + 证据 JSON（复用 plan_stats 的声明正则）。"""
    from tools import plan_stats as PS
    try:
        payload = PS.measure()
        payload["gate"] = PS._local_gate_counts()
    except Exception as exc:                                      # noqa: BLE001
        errs.append(f"plan_stats 统计失败: {exc!r}")
        return
    errs.extend(PS.check_docs(payload))
    errs.extend(PS.check_status_summary(payload))
    # R4-H2：reports/E0_plan_stats.json 必须与 measure() 逐字段相等
    errs.extend(PS.check_evidence(payload))


def check_stage_semantics(st: dict, errs: list[str]) -> None:
    """C：阶段/P 状态语义 + project.phase。"""
    stages = st["stages"]
    for s in stages:
        ps = s.get("p", []) or []
        status = s.get("status")
        non_pending = [p["id"] for p in ps if p.get("status") != "pending"]
        if status == "pending" and non_pending:
            errs.append(f"{s['stage']}: status=pending 但 P {non_pending} 非 pending")
        if status == "in_progress" and ps and not non_pending:
            errs.append(f"{s['stage']}: status=in_progress 但所有 P 都是 pending"
                        f"（应至少有一个 P 已开始，或把阶段改为 pending）")
        if status == "done":
            bad = [p["id"] for p in ps if p.get("status") != "done"]
            if bad:
                errs.append(f"{s['stage']}: status=done 但 P {bad} 未 done")

    expected_phase = next((s["stage"] for s in stages if s.get("status") != "done"), None)
    phase = st.get("project", {}).get("phase")
    if expected_phase is not None and phase != expected_phase:
        errs.append(f"project.phase={phase!r} 与「第一个未 done 的阶段」{expected_phase!r} 不一致"
                    f"（未完成的阶段不得被跳过）")


def check_gates(st: dict, errs: list[str]) -> None:
    """D：Gate 报告与 status 声明的项数/结论必须一致。"""
    for s in st["stages"]:
        g = s.get("gate") or {}
        entries = []
        if isinstance(g, dict):
            for key in ("local", "cloud"):
                if isinstance(g.get(key), dict):
                    entries.append((f"{s['stage']}.{key}", g[key]))
            if g.get("report"):
                entries.append((s["stage"], g))
        for label, gk in entries:
            rep = gk.get("report")
            if not rep:
                continue
            f = V4 / rep
            if not f.is_file():
                errs.append(f"{label}: gate report missing: {rep}")
                continue
            checks = _report_checks(f)
            if checks is None:
                # 报告没有可解析的 check 表（如 cloud gate 的 mandatory_checks 形状不同）。
                # review R7：此前无条件 `continue`，于是 status 声明的项数被**静默**跳过核对；
                # 若声明了 mandatory_total/passed 却核不了，必须显式报错。
                if "mandatory_total" in gk or "mandatory_passed" in gk:
                    errs.append(f"{label}: {rep} 缺少可解析的 check 表，"
                                f"无法核对 status 声明的 mandatory_total/passed")
                continue
            n_pass = sum(1 for v in checks.values() if v)
            if "mandatory_total" in gk and int(gk["mandatory_total"]) != len(checks):
                errs.append(f"{label}: status 声明 mandatory_total={gk['mandatory_total']}，"
                            f"但 {rep} 实际有 {len(checks)} 项")
            if "mandatory_passed" in gk and int(gk["mandatory_passed"]) != n_pass:
                errs.append(f"{label}: status 声明 mandatory_passed={gk['mandatory_passed']}，"
                            f"但 {rep} 实际通过 {n_pass} 项")
            if gk.get("passed") is not None:
                try:
                    rep_passed = json.loads(f.read_text(encoding="utf-8")).get("passed")
                except Exception:                                 # noqa: BLE001
                    rep_passed = None
                if rep_passed is not None and bool(gk["passed"]) != bool(rep_passed):
                    errs.append(f"{label}: status passed={gk['passed']} 与 {rep} 的 "
                                f"passed={rep_passed} 不一致")


def check_cache_evidence(st: dict, errs: list[str]) -> None:
    """E：R3-M5 —— 声称已产出缓存证据的 P 必须把 cache manifest 列进 evidence。"""
    for s in st["stages"]:
        for p in s.get("p", []) or []:
            blob = " ".join(p.get("findings", []) or [])
            ev = " ".join(p.get("evidence", []) or [])
            claims_cache = ("缓存" in blob or "cache" in blob.lower())
            if p.get("status") == "done" and claims_cache and "manifest.json" not in ev:
                errs.append(f"{s['stage']}/{p['id']}: findings 声称已产出缓存，但 evidence 未列 "
                            f"`$V4_CACHE_ROOT/manifest.json`")


def main() -> int:
    st = json.loads((V4 / "versions" / "status.json").read_text(encoding="utf-8"))
    errs: list[str] = []

    check_plan_counts(errs)

    for s in st["stages"]:
        if s["status"] not in VALID_STAGE:
            errs.append(f"{s['stage']}: illegal stage status {s['status']!r}")
        for p in s.get("p", []) or []:
            if p.get("status") not in VALID:
                errs.append(f"{s['stage']}/{p.get('id')}: illegal P status {p.get('status')!r}")
            if p.get("status") == "done" and not p.get("evidence"):
                errs.append(f"{s['stage']}/{p.get('id')}: done 但无 evidence")
        # P 计划文件必须存在 + done 的 P 其 evidence 文件必须存在
        for p in s.get("p", []) or []:
            f = V4 / s["stage"] / p["id"] / "PLAN.md"
            if not f.is_file():
                errs.append(f"missing plan file: {f.relative_to(V4)}")
            if p.get("status") == "done":
                for ev in p.get("evidence", []):
                    ev = ev.split("::")[0].strip().strip("`")
                    if ev.startswith("$"):
                        continue          # 运行时目录（/data）不校验
                    if not (V4 / ev).exists():
                        errs.append(f"{s['stage']}/{p['id']}: evidence not found: {ev}")

    check_stage_semantics(st, errs)
    check_gates(st, errs)
    check_cache_evidence(st, errs)

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
