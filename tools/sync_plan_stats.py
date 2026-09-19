#!/usr/bin/env python3
"""把 tools/plan_stats.py 的实测行数同步进文档（幂等，可重复执行）。

    python3 v4/tools/sync_plan_stats.py
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]

def main() -> int:
    subprocess.run([sys.executable, str(V4 / "tools" / "plan_stats.py"),
                    "--json", str(V4 / "reports" / "E0_plan_stats.json")],
                   check=True, capture_output=True)
    st = json.loads((V4 / "reports" / "E0_plan_stats.json").read_text(encoding="utf-8"))
    m, s_, p_, tot = st["main"]["lines"], st["stage"]["lines"], st["p_level"]["lines"], st["total_lines"]
    rules = [
        (V4 / "PLAN.md", [
            (r"\| 总计划 `PLAN\.md` \| 1 \| \*\*[\d,]+ 行\*\* \|",
             f"| 总计划 `PLAN.md` | 1 | **{m} 行** |"),
            (r"\| 阶段计划 `E\*/PLAN\.md` \| \d+ \| 平均 \d+ 行（合计 [\d,]+）\|",
             f"| 阶段计划 `E*/PLAN.md` | {st['stage']['files']} | 平均 {st['stage']['avg']} 行（合计 {s_:,}）|"),
            (r"\| P 级子计划 `E\*/P\*/PLAN\.md` \| \d+ \| \*\*平均 \d+ 行\*\*（合计 [\d,]+）\|",
             f"| P 级子计划 `E*/P*/PLAN.md` | {st['p_level']['files']} | **平均 {st['p_level']['avg']} 行**（合计 {p_:,}）|"),
            (r"\| 计划文件合计 \| \d+ \| \*\*[\d,]+ 行\*\* \| \u2705 \|",
             f"| 计划文件合计 | 46 | **{tot:,} 行** | \u2705 |"),
        ]),
        (V4 / "README.md", [
            (r"总计划 \d+ 行 \+ 12 个阶段计划（[\d,]+ 行）\+ 33 个 P 级详细计划（[\d,]+ 行），\*\*合计 [\d,]+ 行\*\*",
             f"总计划 {m} 行 + 12 个阶段计划（{s_:,} 行）+ 33 个 P 级详细计划（{p_:,} 行），**合计 {tot:,} 行**"),
        ]),
        (V4 / "docs" / "PROJECT_FILES.md", [
            (r"├── PLAN\.md\s+总计划（[\d,]+ 行，唯一权威）",
             f"├── PLAN.md                   总计划（{m} 行，唯一权威）"),
            (r"\| 计划文件（`PLAN\.md`） \| [^|]*\|",
             f"| 计划文件（`PLAN.md`） | 1（总，{m} 行）+ 12（阶段，{s_:,} 行）+ 33（P，{p_:,} 行）= **46 份 / {tot:,} 行** |"),
            (r"\| P 级计划平均篇幅 \| [^|]*\|",
             f"| P 级计划平均篇幅 | **{st['p_level']['avg']} 行**（合计 {p_:,}；由 `tools/plan_stats.py` 统计） |"),
        ]),
    ]
    changed = 0
    for path, subs in rules:
        t = path.read_text(encoding="utf-8"); o = t
        for pat, rep in subs:
            t2, n = re.subn(pat, rep, t)
            if n == 0:
                print(f"WARN: pattern not found in {path.name}: {pat[:48]}...")
            t = t2
        if t != o:
            path.write_text(t, encoding="utf-8"); changed += 1
            print(f"updated {path.relative_to(V4)}")
    # status.json 摘要
    sp = V4 / "versions" / "status.json"
    d = json.loads(sp.read_text(encoding="utf-8"))
    d["project"]["summary"] = re.sub(r"计划全部完成（[^）]*）",
                                     f"计划全部完成（46 份 / {tot:,} 行，由 tools/plan_stats.py 实测）",
                                     d["project"]["summary"])
    sp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT: {changed + 1} 处已同步（总 {tot:,} 行 / 46 份）")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
