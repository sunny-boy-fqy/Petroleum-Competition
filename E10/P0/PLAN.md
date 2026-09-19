# E10/P0 全量重训与 CPU 导出

> 所属阶段：[E10](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

用 E9 通过候选的配置在全部 80 井上重训（或折集成），导出 CPU 可加载权重与 ONNX 兜底。

## 2. 为什么需要这一步

复现要求"训练+推理"可独立运行；CPU 推理保证评测机兼容。

## 3. 代码

- `E10/code/final_train.py`
- `E10/code/export_cpu.py`

## 4. 产物

- `models/v4/final/*.pt`、`models/v4/final/*.onnx`

## 5. 完成判据

- CPU 加载并前向成功；两次前向 sha256 一致

## 6. 禁止事项

- 导出依赖 GPU
- 把训练日志写进模型目录导致包体膨胀

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E10_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E10_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
