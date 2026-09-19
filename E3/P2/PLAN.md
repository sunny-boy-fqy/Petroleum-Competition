# E3/P2 与行级对照 + 感受野消融 + Gate

> 所属阶段：[E3](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

同折同头对比序列主干与 E1 行级模型；做感受野消融（depth 3/5、dilation_max 64/512）；判定 Gate ≥ 81.0。

## 2. 为什么需要这一步

这是"上下文是否被利用"的判据，也是 E4/E5 是否值得继续的前提。

## 3. 代码

- `E3/code/compare_row_vs_seq.py`
- `E3/code/rf_ablation.py`
- `E3/code/gate.py`

## 4. 产物

- `experiments/E3/P2/*/oof.npz`
- `reports/E3_gate.json`
- `reports/E3_receptive_field_ablation.json`

## 5. 完成判据

- OOF ≥ 81.0；相对行级模型 CI 下界 > 0；消融表完整

## 6. 禁止事项

- 跳过消融直接堆容量
- 用 outer 折选超参

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E3_P2_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E3_P2_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
