# E2/P0 物理与交会特征

> 所属阶段：[E2](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现 Wyllie/密度/中子孔隙度、GR/SP 泥质指数、RT/RXO 比值、PE 骨架、DEN-CNL 差等物理派生列。

## 2. 为什么需要这一步

物理先验是约束与归纳偏置，不是万能解；必须消融验证（`资料库/03` §8.2）。

## 3. 代码

- `src/features/physics.py`
- `E2/code/ablate_groups.py`

## 4. 产物

- `reports/E2_ablation.json`（组 `F_phys`）

## 5. 完成判据

- 组消融给出增量或 NO-GO；派生列公式全部登记

## 6. 禁止事项

- 使用目标值构造派生列

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E2_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E2_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
