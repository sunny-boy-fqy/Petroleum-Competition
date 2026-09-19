# E9 诚实验证与提交护栏

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E9/src/、v4/E9/code/`　|　产物 `v4/experiments/E9/、v4/models/E9/`

> 阶段性质：**验证与仲裁阶段。不出新模型。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

汇总 80 井 OOF、做 16 井次级体检、泄漏终审、A 榜短名单仲裁与 `choose_submission.py` 提交护栏判定。

## 2. 为什么这么设计

1. 本地 80 井 OOF 是主判据；16 井 confirm 只是 v1-exposed 的次级体检，不能当独立确认。
2. A 榜只有 5 口井、噪声带约 ±0.02，只能用于短名单仲裁与崩坏体检（`资料库/12` §3.5）。
3. 提交前必须有护栏，防止"本地漂亮但明显弱于历史锚点"的候选被提交。

## 3. 输入

- 全部候选的 OOF 与 result.zip
- `reports/E0_a_board_log.json`（历史 A 榜锚点）

## 4. 产物

- `reports/E9_validation_report.json`、`reports/E9_leakage_audit.json`
- `reports/E9_a_board_log.json`、`reports/E9_submission_decision.json`
- `reports/E9_gate.json`

## 5. 代码归属

- `E9/code/aggregate_oof.py`、`E9/code/confirm_check.py`、`E9/code/leakage_audit.py`、`E9/code/choose_submission.py`

## 6. P 级子计划

- [P0 OOF 汇总与提交护栏判定](P0/PLAN.md)
- [P1 16 井次级体检与泄漏终审](P1/PLAN.md)
- [P2 A 榜短名单仲裁（预算制 ≤3 次）](P2/PLAN.md)

## 7. 完成判据（Gate）

- 候选 OOF ≥ `max(75, 80.382479 − 1.0, A 榜锚点 82.2757 − 0.5 若已知)`，否则判定回退。
- 泄漏审计覆盖：井级折、特征来源表、标准化 fit 范围、伪标签来源。
- 16 井体检未出现 > 1.5 分的崩坏。

## 8. 禁止事项

- 用 A 榜做细粒度调参
- 在看到 confirm 结果后更换候选或阈值

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.8 / torch 2.7.1 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
