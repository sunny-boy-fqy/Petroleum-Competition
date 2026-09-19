# E1/P1 行级 MLP + 对齐损失

> 所属阶段：[E1](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

训练多任务 MLP（共享 3 层 + 3 头 + 状态头），三段式损失，5 折 OOF，硬 Gate ≥ 78.0。

## 2. 为什么需要这一步

这是纯 DL 的分母与损失实现的验证器。

## 3. 代码

- `src/losses/score_aligned.py`
- `src/models/row_mlp.py`
- `E1/code/train_row.py`

## 4. 产物

- `models/E1/pd0.pt`
- `experiments/E1/P1/pd0/oof.npz`
- `reports/E1_gate.json`

## 5. 完成判据

- OOF ≥ 78.0、5 折同向、占位行 Acc ≥ 0.98、loss 无 NaN、可 `--resume`

## 6. 禁止事项

- 用 MSE 当主损失
- 用 loss 值而非真实评分做早停

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E1_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E1_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
