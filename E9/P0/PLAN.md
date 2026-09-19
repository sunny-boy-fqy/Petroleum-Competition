# E9/P0 OOF 汇总与护栏判定

> 所属阶段：[E9](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

汇总全部候选的 80 井 OOF、逐目标 Acc、bootstrap CI，运行 `choose_submission.py` 护栏。

## 2. 为什么需要这一步

护栏防止"本地漂亮但弱于历史锚点"的候选被提交。

## 3. 代码

- `E9/code/aggregate_oof.py`
- `E9/code/choose_submission.py`

## 4. 产物

- `reports/E9_validation_report.json`、`reports/E9_submission_decision.json`

## 5. 完成判据

- 守卫下限 = max(75, 80.382479−1.0, A 榜 82.2757−0.5 若已知)；判定可复算

## 6. 禁止事项

- 用 selection_score_only 数字伪装独立确认

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E9_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E9_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
