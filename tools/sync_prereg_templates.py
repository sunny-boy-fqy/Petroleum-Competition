#!/usr/bin/env python3
"""把 33 份 P 级子计划里的 Gate 预注册 JSON 同步到 `versions/prereg_templates/`。

**为什么需要它**（R3 发现的一处静默漂移）
------------------------------------------
`versions/prereg_templates/*.json` 没有任何工具生成，是**手工副本**，因此已经与
`E*/P*/PLAN.md` 里的 ```json 块**不一致**：
  * 模板目录里的 33 份文件全部**缺少 `gate_type` 字段**（而 P 级计划里有）；
  * `E0_P0` 的模板写 `min_hard_pass: 6`，而 P 级计划写 `max_hard_failures: 0`；
  * 模板目录的 `mandatory_checks` 没有包含 P 级计划里追加的项。

三审正是拿 `versions/prereg_templates/E9_P2_*.json` 与 P 级计划两份不同的文件讨论
`gate_type`，导致结论自相矛盾。本工具把 **P 级 PLAN.md 的 ```json 块**确立为唯一事实源，
模板目录改为**生成物**，并在 `--check` 下校验零漂移。

    python3 v4/tools/sync_prereg_templates.py            # 同步（幂等）
    python3 v4/tools/sync_prereg_templates.py --check    # 只校验
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

TEMPLATES = V4 / "versions" / "prereg_templates"
JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.S)


def extract(plan: Path) -> dict:
    m = JSON_BLOCK.search(plan.read_text(encoding="utf-8"))
    if m is None:
        raise ValueError(f"{plan} 里没有 ```json 块")
    return json.loads(m.group(1))


def discover() -> list[tuple[Path, Path, dict]]:
    out = []
    for plan in sorted(V4.glob("E*/P*/PLAN.md")):
        stage = plan.parent.parent.name          # E6
        pstage = plan.parent.name                # P0
        out.append((plan, TEMPLATES / f"{stage}_{pstage}_gate_prereg.json", extract(plan)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    items = discover()
    if not items:
        print("FAIL: 未发现任何 E*/P*/PLAN.md")
        return 1
    TEMPLATES.mkdir(parents=True, exist_ok=True)

    rc = 0
    written = 0
    for plan, dst, data in items:
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not dst.is_file():
                print(f"FAIL: 缺失模板 {dst.relative_to(V4)}（来源 {plan.relative_to(V4)}）")
                rc = 1
            elif dst.read_text(encoding="utf-8") != text:
                print(f"FAIL: 模板与 P 级计划漂移 {dst.relative_to(V4)}"
                      f"（运行 tools/sync_prereg_templates.py 修复）")
                rc = 1
        else:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                dst.write_text(text, encoding="utf-8")
                written += 1

    # 孤儿模板（没有对应 P 级计划）
    known = {d.name for _, d, _ in items}
    for f in sorted(TEMPLATES.glob("*.json")):
        if f.name not in known:
            print(f"FAIL: 孤儿模板（无对应 P 级计划）{f.relative_to(V4)}")
            rc = 1

    if args.check:
        print(f"RESULT: {'OK' if rc == 0 else 'FAIL'}（{len(items)} 份模板与 P 级计划一致）")
    else:
        print(f"RESULT: OK（{len(items)} 份模板；本次写入 {written} 份）")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
