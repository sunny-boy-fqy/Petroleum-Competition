#!/usr/bin/env python3
"""由 `reports/E0_data_card.json` 生成 `E0/docs/data_card.md` 的统计表。

**为什么**：二审 R2-B4 发现手写的统计表与 JSON/实测不一致（POR<1: 587 vs 576、
PERM min: 0.000 vs 0.01、SW<1: 11 vs 0、直方图完全不符）。
本工具用 marker 注入，保证"文档数字 = JSON 数字"。

    python3 v4/tools/gen_data_card_md.py            # 就地更新 md
    python3 v4/tools/gen_data_card_md.py --check    # 只校验是否一致
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN:AUTO_STATS (由 tools/gen_data_card_md.py 生成，勿手改) -->"
END = "<!-- END:AUTO_STATS -->"


def build(card: dict) -> str:
    tr = card["train"]
    te = card["test"]
    sc = card["constant_baseline"]["missing_mode_drop"]
    L: list[str] = [BEGIN, "", "### 自动生成的统计表（来源：`reports/E0_data_card.json`）", ""]
    L.append("| 项 | 训练 | 测试 |")
    L.append("|---|---:|---:|")
    L.append(f"| 井数 | {tr['n_wells']} | {te['n_wells']} |")
    L.append(f"| 行数 | {tr['n_rows']:,} | {te['n_rows']:,} |")
    L.append(f"| 缺测行（三目标全缺） | {tr['n_missing_rows']:,} | {te['n_missing_rows']} |")
    L.append(f"| 联合常量占位行 | {tr['n_placeholder_rows']:,} | {te['n_placeholder_rows']} |")
    L.append(f"| 有效行 | {tr['n_valid_rows']:,} | {te['n_valid_rows']} |")
    L.append(f"| 非规范 schema 井 | {len(tr['noncanonical_schema_wells'])} | "
             f"{len(te['noncanonical_schema_wells'])} |")
    L.append(f"| 单井行数 min/median/max | {tr['rows_per_well']['min']} / "
             f"{tr['rows_per_well']['median']} / {tr['rows_per_well']['max']} | — |")
    L.append("")

    L.append("#### 目标分布（训练井，按切片）")
    L.append("")
    L.append("| 目标 | 切片 | n | min | p01 | median | p99 | max | <1 行数 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in ("POR", "PERM", "SW"):
        nm = card["target_stats"][t]["non_missing"]
        va = card["target_stats"][t]["valid_rows_only"]
        L.append(f"| {t} | 非缺测 | {nm['n']:,} | {nm['min']:.4f} | {nm['p01']:.4f} | "
                 f"{nm['median']:.4f} | {nm['p99']:.4f} | {nm['max']:.4f} | — |")
        L.append(f"| {t} | 仅有效行 | {va['n']:,} | {va['min']:.4f} | {va['p01']:.4f} | "
                 f"{va['median']:.4f} | {va['p99']:.4f} | {va['max']:.4f} | "
                 f"{va.get('n_lt_1', '—')} |")
    L.append("")

    L.append("#### 常数基线评分（drop 口径，冻结）")
    L.append("")
    L.append("| 目标 | Acc |")
    L.append("|---|---:|")
    L.append(f"| POR | {sc['acc_por']:.7f} |")
    L.append(f"| PERM | {sc['acc_perm']:.7f} |")
    L.append(f"| SW | {sc['acc_sw']:.7f} |")
    L.append(f"| **Total** | **{sc['total']:.7f}** |")
    L.append("")
    L.append(f"自洽校验：`100×(0.30·POR + 0.35·PERM + 0.35·SW) = "
             f"{card['score_consistency']['total_from_weights']:.7f}`，"
             f"与上报 total 一致 = **{card['score_consistency']['consistent']}**"
             f"（口径 `{card['score_consistency']['missing_mode_frozen']}`）。")
    L.append("")
    L.append(f"输入泄漏回归：抽样 {card['input_leak_regression']['checked_wells']} 井 "
             f"passed={card['input_leak_regression']['passed']}，"
             f"全量 {card['input_leak_regression'].get('full_90_wells', {}).get('checked', '—')} 井 "
             f"violations={card['input_leak_regression'].get('full_90_wells', {}).get('violations', '—')}。")
    L.append("")
    cache = card.get("shard_cache") or {}
    L.append(f"分片缓存：built={cache.get('built')}，"
             f"{cache.get('mb', '—')} MB，input_cols_ok={cache.get('input_cols_ok')}。")
    L.append("")
    L.append(END)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=str(V4 / "reports" / "E0_data_card.json"))
    ap.add_argument("--md", default=str(V4 / "E0" / "docs" / "data_card.md"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    block = build(card)
    md = Path(args.md)
    text = md.read_text(encoding="utf-8")

    if BEGIN in text and END in text:
        a = text.index(BEGIN)
        b = text.index(END) + len(END)
        new = text[:a] + block + text[b:]
    else:
        new = text.rstrip() + "\n\n---\n\n" + block + "\n"

    if args.check:
        if new.strip() == text.strip():
            print("RESULT: OK（文档统计与 JSON 一致）")
            return 0
        print("RESULT: FAIL（文档统计与 JSON 不一致，请运行不带 --check 的命令刷新）")
        return 1

    md.write_text(new, encoding="utf-8")
    print(f"已更新 {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
