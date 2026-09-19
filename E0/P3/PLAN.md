# E0/P3 提交契约与版本路由

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

实现 `predict.py`（官方 `--data_dir`/`--output`）、`validate_payload`、候选注册表与 manifest，并在干净目录冒烟。

## 2. 为什么需要这一步

提交格式错误是零分风险；契约必须早于模型存在。

## 3. 代码

- `predict.py`
- `src/inference/contract.py`
- `src/versioning/registry.py`
- `E0/code/run_all.py`

## 4. 产物

- `reports/E0_contract_tests.json`
- `versions/candidates.json`
- `reports/E0_gate.json`

## 5. 完成判据

- 正/负样例都被正确判定；`python predict.py --list-versions` 在干净目录可用；E0 Gate 全 mandatory 通过

## 6. 禁止事项

- 契约校验依赖 torch
- 允许 logId 缺失或行数不符通过

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E0_P3_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E0_P3_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
