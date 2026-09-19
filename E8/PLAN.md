# E8 多任务、井级分支与集成

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E8/src/、v4/E8/code/`　|　产物 `v4/experiments/E8/、v4/models/E8/`

> 阶段性质：**增益放大阶段。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

实现 MMoE 式任务平衡、井级分支（H4，只作消融）、多 seed/快照/多结构集成，以及 transductive 伪标签（只作消融）。

## 2. 为什么这么设计

1. `资料库/08` §1.4：多任务硬共享通常优于三个独立模型，但必须解决权重失衡（MMoE 是标准解法）。
2. 井内近似常数的工程曲线（CAL/DEVI/AZIM/BIT/CASE）只提供井间区分度（`资料库/08` §0.1-4），井级分支是唯一合法的井间信号通路。
3. v1 E8–E11 与 v2 E6/E7 的井级/序列 NO-GO 是在**无 GPU、无序列主干**的条件下得出的；E8 在已有序列主干的前提下重试一次，但必须用消融说话。

## 3. 输入

- E6/E7 冻结管线
- `资料库/08` §1.4、§8（生成模型/伪标签）

## 4. 产物

- `src/models/mmoe.py`、`src/models/well_head.py`、`src/ensemble/`
- `models/E8/**`、`experiments/E8/**`
- `reports/E8_gate.json`、`reports/E8_ensemble_report.json`

## 5. 代码归属

- `E8/code/train_mmoe.py`、`E8/code/well_branch.py`、`E8/code/ensemble.py`、`E8/code/pseudo_label.py`

## 6. P 级子计划

- [P0 MMoE 任务平衡](P0/PLAN.md)
- [P1 井级分支与 transductive 消融](P1/PLAN.md)
- [P2 集成与 Gate](P2/PLAN.md)

## 7. 完成判据（Gate）

- 集成 ≥ 最佳单成员且加权配对 bootstrap 95% CI 下界 > 0。
- 同源性报告（成员间相关系数）齐备，同源平均不得计为增益。
- 井级分支与伪标签各自给出采纳/NO-GO 结论。

## 8. 禁止事项

- 用测试集标签（不存在）
- 把同源模型平均包装成"集成增益"

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.6 / torch 2.4.0 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
