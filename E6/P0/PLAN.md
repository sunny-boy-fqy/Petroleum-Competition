# E6/P0 联合常量状态头

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

训练 H0 联合占位状态分类头（BCE），评估 AUC 与逐目标原子 precision/recall。

## 2. 为什么需要这一步

占位是 66.7% 的行，状态判别质量直接决定白送分是否守住。

## 3. 代码

- `src/models/state_head.py`
- `E6/code/train_state.py`

## 4. 产物

- `models/E6/state_*.pt`
- `reports/E6_atomic_report.json`

## 5. 完成判据

- AUC ≥ 0.97；原子 recall/precision 双报

## 6. 禁止事项

- 把三个目标的占位状态分别独立建模（丢失联合性）

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E6_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E6_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
