#!/usr/bin/env python3
"""把 E0 的执行状态写回 E0 的 PLAN.md（生成器会覆盖，故需可重复执行）。

    python3 v4/E0/code/mark_status.py
"""
from __future__ import annotations

from pathlib import Path

V4 = Path(__file__).resolve().parents[2]

E0_STATUS = """> **执行状态（2026-09-19）：本地契约层已完成；阶段整体 `in_progress`（云端 Gate 待 P0）。**
> **E0 本地契约 Gate：10/10 mandatory PASS**（`reports/E0_local_contract_gate.json`）；
> **E0 云端 Gate：`blocked_pending_cloud_run`**（`reports/E0_cloud_gate.json`，需 A100 任务跑 `run_train.sh --mode env`）。
> 已复算并冻结的事实：
> - 训练 **80 井 / 730,268 行**；测试 **10 井 / 95,948 行**（= 契约值）
> - 标签状态：缺测 **6,700** / 联合常量占位 **487,225（66.719%）** / 有效 **236,343**
> - 常数基线 **70.490735**（`drop` 口径）命中公开锚点 70.4907 ±1e-4；`mask` 口径为 69.843218
> - 折指纹 `f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e`（80 井 / 5 折，每折 16 井）
> - 提交契约自检：6 个负样例全部被拒绝；`CONST` 端到端 10 井 / 95,948 行 / 1.4 s（单核 CPU）
> - **两个硬发现**：①3 口训练井 schema 非规范（20/21/16 列，27,080 行，3.71%），必须按表头名解析；
>   ②评分分母口径为「逐目标排除缺测」（`drop`），选错会系统性低 0.65 分
>
> 详见 [`E0/docs/data_card.md`](docs/data_card.md) 与 [`versions/status.json`](../versions/status.json)。
> 唯一**未完成**的 P 是 **P0（云端环境与磁盘实测）**，需 A100 训练任务实机运行。"""

ANCHOR = "> 本目录是最小可执行单元"


def main() -> None:
    f = V4 / "E0" / "PLAN.md"
    s = f.read_text(encoding="utf-8")
    # 状态块一旦写入即视为人工维护内容；如需刷新用 --force
    import sys as _sys
    force = "--force" in _sys.argv
    if force and "> **执行状态（2026-09-19）" in s:
        a = s.index("> **执行状态（2026-09-19）")
        b = s.index("> 详见 [`E0/docs/data_card.md`]", a)
        s = s[:a] + E0_STATUS + "\n>\n" + s[b:]
        f.write_text(s, encoding="utf-8")
        print("E0/PLAN.md: status refreshed (--force)")
        return
    if "执行状态（2026-09-19）" not in s:
        old = "> 阶段性质：**契约冻结阶段。不训练任何模型。**"
        assert old in s
        s = s.replace(old, old + "\n>\n" + E0_STATUS, 1)
        f.write_text(s, encoding="utf-8")
        print("E0/PLAN.md: status written")
    else:
        print("E0/PLAN.md: status already present")

    # P 级状态由 docs/gen_p_details.py 的 `status` 字段生成（唯一来源），此处不再重复插入
    for pid in ("P0", "P1", "P2", "P3"):
        g = V4 / "E0" / pid / "PLAN.md"
        t = g.read_text(encoding="utf-8")
        print(f"E0/{pid}/PLAN.md: status line = "
              f"{'present' if '**状态**' in t else 'MISSING -> 请运行 gen_p_details.py'}")


if __name__ == "__main__":
    main()
