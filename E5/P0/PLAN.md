# E5/P0 POR 窄带精修

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

针对 ±0.008 容差带设计 POR 头（0.1+softplus 起步）、观测加权与窄带误差分析。

## 2. 为什么需要这一步

POR 是容差最窄的目标，需要与其他目标完全不同的精度策略。

## 3. 代码

- `E5/code/head_por.py`

## 4. 产物

- `experiments/E5/por/oof.npz`
- `reports/E5_per_target.json`

## 5. 完成判据

- POR 连续切片 Acc 提升且原子带内占比上升；不牺牲 PERM/SW

## 6. 禁止事项

- 用全局回归头直接压 POR 到 0.1

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E5_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E5_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
