# E6/P2 PD1 完整管线与 Gate

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

组装 数据→主干→头→原子门→契约 的完整 PD1，产出 OOF、result.zip 与 manifest，判定 Gate ≥ 82.0。

## 2. 为什么需要这一步

这是 v4 第一个可作为提交候选的完整管线。

## 3. 代码

- `E6/code/build_pd1.py`
- `E6/code/gate.py`

## 4. 产物

- `experiments/E6/P2/pd1/{result.json,result.zip,cv.json,manifest.json}`
- `reports/E6_gate.json`

## 5. 完成判据

- OOF ≥ 82.0；契约通过；本机 CPU 可跑 `--use-version PD1`

## 6. 禁止事项

- 在管线中混入未冻结的特征版本

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E6_P2_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E6_P2_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
