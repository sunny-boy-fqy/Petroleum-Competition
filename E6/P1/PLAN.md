# E6/P1 原子门 τ 搜索（inner-OOF）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

在 inner-OOF 上网格/贝叶斯搜索逐目标门限 τ_t，报告误判代价分解。

## 2. 为什么需要这一步

τ 决定"判常量还是判连续"，是纯 DL 管线唯一保护屏障的开关。

## 3. 代码

- `src/inference/atomic_gate.py`
- `E6/code/search_tau.py`

## 4. 产物

- `reports/E6_tau_search.json`

## 5. 完成判据

- τ 只在 inner-OOF 选；误判代价表完整；占位行 Acc ≥ 0.99

## 6. 禁止事项

- 用 outer 折或 A 榜选 τ
- 在常量与连续值之间插值

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E6_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E6_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
