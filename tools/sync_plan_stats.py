#!/usr/bin/env python3
"""把 tools/plan_stats.py 的实测行数同步进文档（幂等，可重复执行）。

R3-C2 修复
----------
旧版正则要求列尾紧接 `|`，而 `PLAN.md` 的表格该列后还有 `✅ 完成`，于是
`pattern not found` 被**静默吞掉**：只有「合计」那一行被更新，阶段 711 / P 4,982 原样留着。

现在同步逻辑**直接复用 `plan_stats.py::_claims()` 的正则**，逐条：
  - 匹配不到 -> 打印 `WARN: pattern not found` 且最终返回非零；
  - 匹配到   -> 把捕获组里的数字替换为实测值（保留原千分位风格）。
因此「校验用的正则」与「同步用的正则」不可能再漂移。

    python3 v4/tools/sync_plan_stats.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from tools.plan_stats import _claims, _dig, _local_gate_counts, measure  # noqa: E402


def _subst(text: str, pattern: str, payload: dict,
           expect: tuple[tuple[str, str], ...]) -> tuple[str, int]:
    """把正则捕获组里的数字替换成实测值；返回 (新文本, 命中次数)。"""
    def repl(m: re.Match) -> str:
        out = m.group(0)
        spans: list[tuple[int, int, str]] = []
        for group, key in expect:
            want = _dig(payload, key)
            if want is None:
                continue
            orig = m.group(group)
            new = f"{int(want):,}" if "," in orig else str(int(want))
            s, e = m.span(group)
            spans.append((s - m.start(), e - m.start(), new))
        for s, e, new in sorted(spans, key=lambda t: -t[0]):   # 从后往前，避免偏移错乱
            out = out[:s] + new + out[e:]
        return out

    return re.subn(pattern, repl, text)


def main() -> int:
    payload = measure()
    payload["gate"] = _local_gate_counts()
    rc = 0
    touched: dict[str, int] = {}
    cache: dict[str, str] = {}

    for c in _claims():
        f = V4 / c.path
        if not f.is_file():
            print(f"WARN: {c.path} 不存在")
            rc = 1
            continue
        text = cache.get(c.path) or f.read_text(encoding="utf-8")
        new, n = _subst(text, c.pattern, payload, c.expect)
        cache[c.path] = new
        touched[c.path] = touched.get(c.path, 0) + n
        if n == 0:
            print(f"WARN: pattern not found in {c.path}: {c.pattern[:70]}…")
            rc = 1

    for path, text in cache.items():
        f = V4 / path
        if f.read_text(encoding="utf-8") != text:
            f.write_text(text, encoding="utf-8")
            print(f"updated {path}（{touched.get(path, 0)} 条声明）")

    # status.json 摘要（单一格式，避免模糊正则）
    sp = V4 / "versions" / "status.json"
    d = json.loads(sp.read_text(encoding="utf-8"))
    g = payload.get("gate") or {}
    gate_txt = (f"E0 本地契约 Gate {g['mandatory_passed']}/{g['mandatory_total']} PASS"
                if g else "E0 本地契约 Gate 待复算")
    d["project"]["summary"] = (
        f"{gate_txt}（云端 Gate 待 P0 实机）；"
        f"计划全部完成（{payload['total_plan_files']} 份 / {payload['total_lines']:,} 行，"
        f"由 tools/plan_stats.py 实测）；"
        f"三审 R3-C1..C3/H1..H5/M1..M5、四审 R4-B1..B3/H1..H3/M1..M6、"
        f"五审 R5-B1/H1/H2/M1/M2/L1..L3 与六审 R6-M1/M2/L1..L3 已修复并回归验证；"
        f"改进 proposal 已落进 PLAN.md / E6 与 src/"
    )
    d["project"]["phase"] = "E0"          # R3-H4：云端 Gate 通过前不得指向 E1
    sp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("updated versions/status.json（summary + project.phase=E0）")

    # R4-H2：证据 JSON 由**同一次实测**直接落盘，杜绝"手工副本过期"
    ep = V4 / "reports" / "E0_plan_stats.json"
    ep.parent.mkdir(parents=True, exist_ok=True)
    ep.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated reports/E0_plan_stats.json（{payload['total_plan_files']} 份 / "
          f"{payload['total_lines']:,} 行，与 measure() 同源）")

    print(f"RESULT: {'OK' if rc == 0 else 'FAIL'}（{payload['total_plan_files']} 份 / "
          f"{payload['total_lines']:,} 行）")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
