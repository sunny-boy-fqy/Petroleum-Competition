#!/usr/bin/env python3
"""把 E0 的执行状态写回 E0 的 PLAN.md（生成器会覆盖，故需可重复执行）。

    python3 v4/E0/code/mark_status.py
"""
from __future__ import annotations

from pathlib import Path

V4 = Path(__file__).resolve().parents[2]

E0_STATUS = """> **执行状态（2026-09-19）：已完成。** E0 Gate **6/6 mandatory PASS**（`reports/E0_gate.json`）。
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

P_STATUS = {
    "P0": "> **状态：待云端执行**（本机无 GPU / 无 torch，无法替代）。需在平台训练任务以 A100 + 预装镜像运行 `run_train.sh --mode env`。",
    "P1": "> **状态：已完成（2026-09-19）。** 证据：`reports/E0_data_card.json`、`E0/docs/data_card.md`、`versions/folds_sha256.json`、`artifacts/E0/folds.json`。",
    "P2": "> **状态：已完成（2026-09-19）。** 证据：`src/score.py`、`reports/E0_data_card.json::constant_baseline`（drop=70.490735，mask=69.843218）。",
    "P3": "> **状态：已完成（2026-09-19）。** 证据：`predict.py`、`src/inference/contract.py`、`reports/E0_contract_tests.json`、`versions/candidates.json`。",
}

ANCHOR = "> 本目录是最小可执行单元"


def main() -> None:
    f = V4 / "E0" / "PLAN.md"
    s = f.read_text(encoding="utf-8")
    if "执行状态（2026-09-19）" not in s:
        old = "> 阶段性质：**契约冻结阶段。不训练任何模型。**"
        assert old in s
        s = s.replace(old, old + "\n>\n" + E0_STATUS, 1)
        f.write_text(s, encoding="utf-8")
        print("E0/PLAN.md: status written")
    else:
        print("E0/PLAN.md: status already present")

    for pid, txt in P_STATUS.items():
        g = V4 / "E0" / pid / "PLAN.md"
        t = g.read_text(encoding="utf-8")
        if txt in t:
            print(f"E0/{pid}/PLAN.md: already marked")
            continue
        assert ANCHOR in t, g
        t = t.replace(ANCHOR, txt + "\n>\n" + ANCHOR, 1)
        g.write_text(t, encoding="utf-8")
        print(f"E0/{pid}/PLAN.md: marked")


if __name__ == "__main__":
    main()
