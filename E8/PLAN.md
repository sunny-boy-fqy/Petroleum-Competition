# E8 多任务、井级分支与集成

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E8/src/、v4/E8/code/`　|　产物 `v4/experiments/E8/、v4/models/E8/`

> 阶段性质：**增益放大阶段。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

实现 MMoE 式任务平衡、井级分支（H4，小容量强正则、只作消融）、EMA/SWA 与同折 top-k 快照集成、多 seed/多结构集成，以及 transductive 伪标签（只作消融）。

## 2. 为什么这么设计

1. `资料库/08` §1.4：多任务硬共享通常优于三个独立模型，但必须解决权重失衡（MMoE 是标准解法）。
2. 井内近似常数的工程曲线（CAL/DEVI/AZIM/BIT/CASE）只提供井间区分度（`资料库/08` §0.1-4），井级分支是唯一合法的井间信号通路。
3. v1 E8–E11 与 v2 E6/E7 的井级/序列 NO-GO 是在**无 GPU、无序列主干**的条件下得出的；E8 在已有序列主干的前提下重试一次，但必须用消融说话。
4. 改进 proposal §5：EMA 要用真实 `score.py` 在 inner-OOF 上逐 epoch 评估并保存 `ema.pt`，SWA 只作对照（BN 统计需谨慎）；同折 top-k 快照融合权重只能在 inner-OOF 选；**增益 CI 含 0 即 NO-GO**，不得用同源平均冒充增益。

## 3. 输入

- E6/E7 冻结管线
- `资料库/08` §1.4、§8（生成模型/伪标签）

## 4. 产物

- `src/models/mmoe.py`、`src/models/well_head.py`、`src/ensemble/`
- `models/E8/**`（含 `ema.pt`/`last.pt`/`best.pt`）、`experiments/E8/**`
- `reports/E8_gate.json`、`reports/E8_ensemble_report.json`（逐折 delta + CI）

## 5. 代码归属

- `E8/code/train_mmoe.py`、`E8/code/well_branch.py`、`E8/code/ensemble.py`、`E8/code/pseudo_label.py`

## 6. P 级子计划

- [P0 MMoE 任务平衡](P0/PLAN.md)
- [P1 井级分支与 transductive 消融（各一次）](P1/PLAN.md)
- [P2 集成（EMA/SWA / 快照 / 多结构）与 Gate](P2/PLAN.md)

## 7. 完成判据（Gate）

- 集成 ≥ 最佳单成员且加权配对 bootstrap 95% CI 下界 > 0。
- 同源性/成员相关性报告齐备，同源平均不得计为增益；增益 CI 含 0 的策略一律 NO-GO。
- 井级分支与伪标签各自给出采纳/NO-GO 结论；井级分支保持小容量强正则且校准参数只在 inner-OOF 选。
- EMA（decay∈{0.99,0.999,0.9995}）逐 epoch 用真实 `score.py` 在 inner-OOF 评估，`ema.pt` 与 `last.pt`/`best.pt` 并存；SWA 仅作对照并报告 BN 统计处理。

## 8. 禁止事项

- 用测试集标签（不存在）
- 把同源模型平均包装成"集成增益"
- 把 CI 含 0 的增益打包成"有效"
- 让井级校准参数在 inner-OOF 之外拟合

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 NPU/GPU）负责代码与契约，云端（1×Ascend 910B 64GB，CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64，**30 GB 云盘（/data）**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。

---

> **WP8/WP11 更新（2026-09）**：参考 SPWLA 2021：
> - WP8：新增类型井选择（`src/data/type_well.py`）与井间输入分布匹配
>   （`src/data/well_adapt.py`），E8 all 先产出 `E8_type_well.json`；
> - WP9：MICE/KNN 插补、异常权重、KS 代表采样作为 E8 前置数据消融；
> - WP11：链式目标（`src/training/chained.py`）与 GBDT 一阶成员
>   （`src/models/gbdt.py`）加入 E8 候选；所有融合仍必须先融合再硬切换，
>   集成权重只允许 inner-OOF 选。
