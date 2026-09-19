# E1/P0 行级输入管线

> 所属阶段：[E1](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

构建 `F_raw(14) + F_miss(14+1) + F_depth(3)` 逐行张量，按井分片落盘，检查无泄漏。

## 2. 为什么需要这一步

输入管线的正确性决定了后面所有对比是否有意义。

## 3. 代码

- `src/features/basic.py`
- `E1/code/build_row_features.py`

## 4. 产物

- `cache/raw/*.npz`
- `reports/E1_row_features.json`

## 5. 完成判据

- 行数与 E0 数据卡一致；缺失指示位与掩码一致；缓存 < 100 MB

## 6. 禁止事项

- 在整表上 fit 标准化参数
- 把井身份作为特征

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E1_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E1_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
