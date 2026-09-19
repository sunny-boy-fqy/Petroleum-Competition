#!/usr/bin/env python3
"""计划行数统计（唯一事实源，供文档引用）。

**为什么需要它**：一审/二审/三审三次出现文档手写行数与实际不符（R2-H6、R3-C2）。
本工具用 shell 展开顺序统计，输出可被 CI/提交前检查复算的数字。

R3-C2 修复
----------
旧版 `--check` 只把文档里出现的**所有** `N 行` 数字收集起来，判断实测值是否在其中。
后果：`PLAN.md` 里「阶段 711 / P 4,982」是错的，但文档别处恰好出现了正确的
「合计 6,425 行」，于是 `711/4,982` 被完全漏检。

现在改为**带上下文的逐条断言**：每个文档里每一条行数声明都有专属正则，
正则必须**匹配到**（`pattern not found` 也算失败），且捕获到的数字必须等于实测值。
另新增 `--fix`（= `sync_plan_stats.py` 的文档同步逻辑）。

    python3 v4/tools/plan_stats.py                      # 只统计
    python3 v4/tools/plan_stats.py --json reports/E0_plan_stats.json
    python3 v4/tools/plan_stats.py --check              # 校验 PLAN/README/PROJECT_FILES
    python3 v4/tools/plan_stats.py --check PLAN.md      # 只校验指定文档
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- 统计
def count(pattern: str) -> dict:
    files = sorted(V4.glob(pattern))
    total = 0
    per: dict[str, int] = {}
    for f in files:
        n = len(f.read_text(encoding="utf-8").splitlines())
        per[str(f.relative_to(V4))] = n
        total += n
    return {"pattern": pattern, "files": len(files), "lines": total, "per_file": per}


def measure() -> dict:
    main_p = count("PLAN.md")
    stage = count("E*/PLAN.md")
    p_level = count("E*/P*/PLAN.md")
    total = main_p["lines"] + stage["lines"] + p_level["lines"]
    total_files = main_p["files"] + stage["files"] + p_level["files"]
    return {
        "total_plan_files": total_files,
        "total_lines": total,
        "main": {"files": main_p["files"], "lines": main_p["lines"],
                 "avg": main_p["lines"] // max(main_p["files"], 1)},
        "stage": {"files": stage["files"], "lines": stage["lines"],
                  "avg": stage["lines"] // max(stage["files"], 1)},
        "p_level": {"files": p_level["files"], "lines": p_level["lines"],
                    "avg": p_level["lines"] // max(p_level["files"], 1)},
        "per_file": {**main_p["per_file"], **stage["per_file"], **p_level["per_file"]},
    }


# --------------------------------------------------------------------- 断言
@dataclass(frozen=True)
class Claim:
    """一条文档行数声明：正则必须匹配，且捕获组必须等于实测值。"""
    path: str
    pattern: str
    expect: tuple[tuple[str, str], ...]      # (捕获组名, payload 取值路径)


def _claims() -> list[Claim]:
    """所有行数声明的位置与形状（R3-C2：逐条带上下文断言）。"""
    M, S, P = "main", "stage", "p_level"
    return [
        Claim("PLAN.md",
              r"\| 总计划 `PLAN\.md` \| (?P<files>\d+) \| \*\*(?P<lines>[\d,]+) 行\*\* \|",
              (("files", f"{M}.files"), ("lines", f"{M}.lines"))),
        Claim("PLAN.md",
              r"\| 阶段计划 `E\*/PLAN\.md` \| (?P<files>\d+) \| 平均 (?P<avg>\d+) 行"
              r"（合计 (?P<lines>[\d,]+)）",
              (("files", f"{S}.files"), ("avg", f"{S}.avg"), ("lines", f"{S}.lines"))),
        Claim("PLAN.md",
              r"\| P 级子计划 `E\*/P\*/PLAN\.md` \| (?P<files>\d+) \| \*\*平均 (?P<avg>\d+) 行\*\*"
              r"（合计 (?P<lines>[\d,]+)）",
              (("files", f"{P}.files"), ("avg", f"{P}.avg"), ("lines", f"{P}.lines"))),
        Claim("PLAN.md",
              r"\| 计划文件合计 \| (?P<files>\d+) \| \*\*(?P<lines>[\d,]+) 行\*\* \|",
              (("files", "total_plan_files"), ("lines", "total_lines"))),
        Claim("README.md",
              r"总计划 (?P<main>[\d,]+) 行 \+ (?P<sf>\d+) 个阶段计划（(?P<stage>[\d,]+) 行）"
              r"\+ (?P<pf>\d+) 个 P 级详细计划（(?P<p>[\d,]+) 行），\*\*合计 (?P<total>[\d,]+) 行\*\*",
              (("main", f"{M}.lines"), ("sf", f"{S}.files"), ("stage", f"{S}.lines"),
               ("pf", f"{P}.files"), ("p", f"{P}.lines"), ("total", "total_lines"))),
        Claim("README.md",
              r"E0 口径层已实现并通过本地契约 Gate (?P<n>\d+)/(?P<d>\d+)",
              (("n", "gate.mandatory_passed"), ("d", "gate.mandatory_total"))),
        Claim("docs/PROJECT_FILES.md",
              r"├── PLAN\.md\s+总计划（(?P<lines>[\d,]+) 行，唯一权威）",
              (("lines", f"{M}.lines"),)),
        Claim("docs/PROJECT_FILES.md",
              r"\| 计划文件（`PLAN\.md`） \| 1（总，(?P<main>[\d,]+) 行）"
              r"\+ (?P<sf>\d+)（阶段，(?P<stage>[\d,]+) 行）"
              r"\+ (?P<pf>\d+)（P，(?P<p>[\d,]+) 行）= \*\*(?P<files>\d+) 份 / (?P<total>[\d,]+) 行\*\* \|",
              (("main", f"{M}.lines"), ("sf", f"{S}.files"), ("stage", f"{S}.lines"),
               ("pf", f"{P}.files"), ("p", f"{P}.lines"),
               ("files", "total_plan_files"), ("total", "total_lines"))),
        Claim("docs/PROJECT_FILES.md",
              r"\| P 级计划平均篇幅 \| \*\*(?P<avg>\d+) 行\*\*（合计 (?P<lines>[\d,]+)；",
              (("avg", f"{P}.avg"), ("lines", f"{P}.lines"))),
        # R3-C3：E0 数据卡里的 Gate 项数也是"实物声明"，必须由报告实测（此前写死 10/10）
        Claim("E0/docs/data_card.md",
              r"`reports/E0_local_contract_gate\.json` \| ✅ passed（mandatory (?P<n>\d+)/(?P<d>\d+)）",
              (("n", "gate.mandatory_passed"), ("d", "gate.mandatory_total"))),
    ]


def _dig(payload: dict, path: str):
    cur = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _to_int(text: str):
    return int(str(text).replace(",", ""))


def check_docs(payload: dict, paths: list[str] | None = None) -> list[str]:
    """返回错误列表（空 = 所有文档声明与实测一致）。"""
    errs: list[str] = []
    wanted = set(paths) if paths else None
    if wanted is not None:
        # 请求校验一个**没有任何声明**的文件，等同于校验被静默跳过 -> 必须报错
        known = {c.path for c in _claims()}
        for p in sorted(wanted - known):
            errs.append(f"{p}: 没有登记任何行数声明（无法校验；缺少声明本身即错误）")
    seen: dict[str, int] = {}
    for c in _claims():
        if wanted is not None and c.path not in wanted:
            continue
        f = V4 / c.path
        if not f.is_file():
            errs.append(f"{c.path}: 文件缺失")
            continue
        text = f.read_text(encoding="utf-8")
        hits = list(re.finditer(c.pattern, text))
        if not hits:
            errs.append(f"{c.path}: 未找到行数声明（pattern not found）: {c.pattern[:70]}…")
            continue
        seen[c.path] = seen.get(c.path, 0) + len(hits)
        for m in hits:
            for group, key in c.expect:
                want = _dig(payload, key)
                got = _to_int(m.group(group))
                if want is None:
                    errs.append(f"{c.path}: 内部错误，payload 缺少 {key}")
                elif got != int(want):
                    line = text[: m.start()].count("\n") + 1
                    errs.append(f"{c.path}:{line}: 声明 {group}={got} 与实测 {key}={want} 不一致")
    return errs


def check_status_summary(payload: dict) -> list[str]:
    """status.json 的 summary 也必须声明实测总量。"""
    sp = V4 / "versions" / "status.json"
    if not sp.is_file():
        return ["versions/status.json: 文件缺失"]
    try:
        d = json.loads(sp.read_text(encoding="utf-8"))
    except Exception as exc:                                     # noqa: BLE001
        return [f"versions/status.json: 无法解析 {exc!r}"]
    summary = str(d.get("project", {}).get("summary", ""))
    tot = payload["total_lines"]
    errs: list[str] = []
    if f"{tot:,} 行" not in summary and f"{tot} 行" not in summary:
        errs.append(f"versions/status.json: project.summary 未声明实测总行数 {tot:,}")
    if f"{payload['total_plan_files']} 份" not in summary:
        errs.append(f"versions/status.json: project.summary 未声明实测计划份数 "
                    f"{payload['total_plan_files']} 份")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--check", nargs="*", default=None,
                    help="校验文档中的行数声明（默认 PLAN.md / README.md / docs/PROJECT_FILES.md）")
    args = ap.parse_args()

    payload = measure()
    payload["gate"] = _local_gate_counts()
    print(f"总计划 PLAN.md      : {payload['main']['lines']} 行")
    print(f"阶段计划 E*/PLAN.md : {payload['stage']['files']} 份 / {payload['stage']['lines']} 行 "
          f"(均 {payload['stage']['avg']})")
    print(f"P 级计划 E*/P*/     : {payload['p_level']['files']} 份 / {payload['p_level']['lines']} 行 "
          f"(均 {payload['p_level']['avg']})")
    print(f"合计                : {payload['total_plan_files']} 份 / {payload['total_lines']} 行")
    if payload["gate"]:
        g = payload["gate"]
        print(f"E0 本地契约 Gate    : {g['mandatory_passed']}/{g['mandatory_total']} mandatory PASS")

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = V4 / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json -> {out}")

    rc = 0
    if args.check is not None:
        paths = args.check or None
        errs = check_docs(payload, paths)
        # 默认（未显式指定文件）时额外校验 status.json 摘要
        if paths is None:
            errs += check_status_summary(payload)
        if errs:
            for e in errs:
                print(f"WARN: {e}")
            print(f"RESULT: FAIL（{len(errs)} 条声明不一致；运行 tools/sync_plan_stats.py 修复）")
            rc = 1
        else:
            scope = ", ".join(sorted(set(paths))) if paths else "PLAN.md, README.md, docs/PROJECT_FILES.md, versions/status.json"
            print(f"RESULT: OK（{scope} 的行数声明全部与实测一致）")
    return rc


def _local_gate_counts() -> dict | None:
    """把 E0 本地契约 Gate 的项数并入 payload（README 的 N/M 断言需要）。

    R3-C3：报告里的字段是 `mandatory_checks`（dict），旧代码只看 `checks`，
    于是文档里的 `10/10` 永远对不上实物报告。
    """
    p = V4 / "reports" / "E0_local_contract_gate.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return None
    checks = d.get("mandatory_checks") or d.get("checks")
    if not isinstance(checks, dict):
        return None
    return {"mandatory_passed": sum(1 for v in checks.values() if v),
            "mandatory_total": len(checks),
            "declared_passed": d.get("passed")}


if __name__ == "__main__":
    raise SystemExit(main())
