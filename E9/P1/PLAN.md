# E9/P1 16 井体检与泄漏终审

> 所属阶段：[E9](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

对 top-2 候选在 `folds_confirm` 上各跑一次；完成特征来源、折、标准化、伪标签四类泄漏审计。

## 2. 为什么需要这一步

16 井是 v1-exposed 的次级体检；泄漏审计是防"本地虚高"的最后一道关卡。

## 3. 代码

- `E9/code/confirm_check.py`
- `E9/code/leakage_audit.py`

## 4. 产物

- `reports/E9_confirm.json`、`reports/E9_leakage_audit.json`

## 5. 完成判据

- 无 > 1.5 分崩坏；四类泄漏审计全部通过或标出残余风险

## 6. 禁止事项

- 把 confirm 当独立确认宣称显著增益

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E9_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E9_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
