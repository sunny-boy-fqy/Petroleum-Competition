# E5/P1 PERM log 域精修

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

PERM 在 log10 域建模，处理长尾与数量级误差，输出裁剪到 [-6,6] 并保证 >0。

## 2. 为什么需要这一步

PERM 权重 35%、历史探索最少、边际收益最高（`资料库/12` §3.3）。

## 3. 代码

- `E5/code/head_perm.py`

## 4. 产物

- `experiments/E5/perm/oof.npz`

## 5. 完成判据

- PERM Acc 提升（CI 下界 > 0）；无 ≤0 或非有限输出

## 6. 禁止事项

- 线性域建模 PERM
- 用 ReLU 输出导致零梯度

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E5_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E5_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
