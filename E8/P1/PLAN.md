# E8/P1 井级分支与 transductive 消融

> 所属阶段：[E8](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现 H4 井级 attention-pool 偏置分支；测试推理期 transductive 适配（伪标签/井级统计对齐），两者都只作消融。

## 2. 为什么需要这一步

工程曲线只提供井间信号；v1/v2 的井级路线 NO-GO 需在序列主干条件下重验一次。

## 3. 代码

- `src/models/well_head.py`
- `E8/code/well_branch.py`
- `E8/code/pseudo_label.py`

## 4. 产物

- `reports/E8_well_branch.json`、`reports/E8_transductive.json`

## 5. 完成判据

- 给出明确的采纳/NO-GO 结论与 CI；伪标签不得使用测试标签

## 6. 禁止事项

- 把 transductive 适配伪装成"训练时改进"

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E8_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E8_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
