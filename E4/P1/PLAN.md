# E4/P1 多尺度融合

> 所属阶段：[E4](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

把 CNN 主干与 PatchTF 的输出按门控或 concat 融合送头，比较融合与单主干。

## 2. 为什么需要这一步

多尺度是最常见的稳定增益来源，但必须证明超过最好单主干。

## 3. 代码

- `src/models/multiscale.py`
- `E4/code/fuse_multiscale.py`

## 4. 产物

- `experiments/E4/P1/*/oof.npz`
- `reports/E4_gate.json`

## 5. 完成判据

- 融合 ≥ 单主干最佳（CI 下界 > 0），否则 NO-GO 并保留 E3

## 6. 禁止事项

- 用两个高度同源的分支冒充多尺度

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E4_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E4_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
