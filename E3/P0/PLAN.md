# E3/P0 序列数据集与分块策略

> 所属阶段：[E3](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现按井分块（chunk）数据集：变长井切 chunk、边界 overlap、按需读取分片、worker 内存自检。

## 2. 为什么需要这一步

整井 4k–13k 点无法一次性进模型；分块策略决定感受野与吞吐的平衡。

## 3. 代码

- `src/data/seq_dataset.py`

## 4. 产物

- `reports/E3_seq_dataset.json`

## 5. 完成判据

- worker 常驻 < 300 MB；chunk 边界无标签错位；可复现的采样顺序

## 6. 禁止事项

- 把整井常驻内存
- 打乱时破坏井内深度顺序

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E3_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E3_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
