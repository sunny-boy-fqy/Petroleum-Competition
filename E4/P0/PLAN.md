# E4/P0 Patch Transformer 主干

> 所属阶段：[E4](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现 PatchTST 式通道独立 Transformer（patch=32/stride=16、d=256、6 层、8 头、相对位置编码）。

## 2. 为什么需要这一步

长程依赖与通道独立建模是 CNN 的天然补充。

## 3. 代码

- `src/models/patchtf.py`
- `E4/code/train_patchtf.py`

## 4. 产物

- `models/E4/patchtf_*.pt`

## 5. 完成判据

- 与 E3 同数据同折可比；注意力走 `F.scaled_dot_product_attention`

## 6. 禁止事项

- 把曲线轴当图像轴做 2D 卷积
- 依赖 flash-attn

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E4_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E4_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
