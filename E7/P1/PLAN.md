# E7/P1 解码与后处理

> 所属阶段：[E7](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现温度校正、逐目标偏置、分位数收缩等解码手段，并在 inner-OOF 上选择。

## 2. 为什么需要这一步

零模型改动的换分手段，成本最低、风险最小。

## 3. 代码

- `E7/code/decode_search.py`
- `src/inference/decode.py`

## 4. 产物

- `reports/E7_decode_search.json`

## 5. 完成判据

- 解码策略在 inner-OOF 选定并冻结；对契约无副作用

## 6. 禁止事项

- 用 A 榜选解码参数

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E7_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E7_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
