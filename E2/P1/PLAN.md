# E2/P1 窗口与井级特征

> 所属阶段：[E2](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现居中多尺度窗口统计（窗长 11/51/201 点）与井级聚合特征，处理 16 GiB 内存纪律。

## 2. 为什么需要这一步

v1 已证明窗口特征是稳定增益；井级聚合是唯一的井间信号通路。

## 3. 代码

- `src/features/window.py`
- `src/features/well.py`
- `E2/code/build_features.py`

## 4. 产物

- `cache/feat/F2/**`
- `reports/E2_mem_profile.json`
- `reports/E2_feature_provenance.csv`

## 5. 完成判据

- 缓存 < 2 GB；峰值内存 < 12 GiB；窗口特征消融为正

## 6. 禁止事项

- 使用非居中（因果）窗口
- 在验证井上 fit 分位数

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E2_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E2_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
