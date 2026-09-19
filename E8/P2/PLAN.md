# E8/P2 集成与 Gate

> 所属阶段：[E8](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

多 seed/快照/多结构集成，报告成员同源性与加权配对 bootstrap，判定 Gate。

## 2. 为什么需要这一步

集成的收益必须扣除同源性；`资料库/08` §0.3 第 4 层。

## 3. 代码

- `src/ensemble/blend.py`
- `E8/code/ensemble.py`

## 4. 产物

- `experiments/E8/ensemble/**`
- `reports/E8_ensemble_report.json`、`reports/E8_gate.json`

## 5. 完成判据

- 集成 ≥ 最佳单成员且 CI 下界 > 0；同源报告齐备

## 6. 禁止事项

- 用同源模型平均制造假增益

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E8_P2_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E8_P2_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
