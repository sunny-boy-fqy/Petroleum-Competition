# E7 评分对齐损失与解码后处理

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E7/src/、v4/E7/code/`　|　产物 `v4/experiments/E7/、v4/models/E7/`

> 阶段性质：**损失与解码的收尾优化阶段。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

对三段式损失（align / aux / ph）做完整权重组消融与退火策略调参；实现**归一化 `L_aux`、容差边界聚焦权重、PERM 官方截断消融**与逐目标期望分解码，全部在 inner-OOF 上选择。

## 2. 为什么这么设计

1. `资料库/12` §2 明确指出：评分截断使超过容差阈值的点不再产生梯度收益，把容量让给"临界点"才是最优策略——这只能通过损失权重与解码调节实现。
2. 对齐损失在训练早期梯度稀疏，退火策略直接决定能否收敛到好的局部解。
3. 解码阶段是"零模型改动换分"的手段，成本最低、风险最小。
4. 改进 proposal §4/§7：`L_aux` 若仍用绝对 Smooth L1，SW 的 99.9 会主导梯度；边界聚焦（κ/σ）与 PERM 截断必须用**同结构对照**消融，解码要按逐目标期望分而不是拍脑袋偏置。

## 3. 输入

- E6 完整管线
- `资料库/12` §2.2–2.4

## 4. 产物

- `src/losses/*` 的最终配置、`reports/E7_loss_ablation.json`
- `reports/E7_decode_search.json`、`reports/E7_gate.json`

## 5. 代码归属

- `E7/code/ablate_loss.py`、`E7/code/decode_search.py`

## 6. P 级子计划

- [P0 三段式损失消融（L_aux 归一化 / 边界聚焦 / PERM 截断）](P0/PLAN.md)
- [P1 解码与后处理（逐目标期望分 / 敏感性）](P1/PLAN.md)

## 7. 完成判据（Gate）

- 对齐损失优于纯 aux 损失（**同结构对照**，CI 下界 > 0）。
- 三段式权重（含 λ₁ 退火曲线）的消融表完整。
- 归一化 `L_aux`（训练折 `s_por`/`s_sw`）与 PERM 官方截断的消融完成。
- 边界聚焦权重消融表完整：`κ∈{0.5,1.0,2.0}` × `σ∈{0.15,0.25,0.35}`（仅作可选增强，默认关）。
- 逐目标期望分解码在 inner-OOF 上选定并冻结，附敏感性曲线。
- 所有对照除被消融项外结构完全相同，且从不只报整体 Total。

## 8. 禁止事项

- 用 outer 折或 A 榜选择损失权重/解码参数
- 同时改多个损失项导致无法归因
- 只报整体 Total 而隐藏单目标或切片的退化
- 把边界聚焦权重设为默认开启（它只是 E7 的可选消融项）

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.8 / torch 2.7.1 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
