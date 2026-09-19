# E3/P1 1D U-Net 与 TCN 实现

> 所属阶段：[E3](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现两种主干（depthwise 可分离卷积 + 残差 + 上采样 skip），多任务头，bf16 训练。

## 2. 为什么需要这一步

`资料库/08` §0.3 的核心推荐；两者头对头才能知道哪种归纳偏置更适合本数据。

## 3. 代码

- `src/models/unet1d.py`
- `src/models/tcn.py`
- `E3/code/train_seq.py`

## 4. 产物

- `models/E3/{unet,tcn}_fold*.pt`

## 5. 完成判据

- 参数量/显存/单折耗时记录；前向输出长度与输入严格一致

## 6. 禁止事项

- 使用 2.5+ API
- 引入需编译的 CUDA 扩展

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E3_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E3_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
