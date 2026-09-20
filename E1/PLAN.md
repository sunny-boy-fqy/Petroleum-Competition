# E1 纯 DL 行级基线

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E1/src/、v4/E1/code/`　|　产物 `v4/experiments/E1/、v4/models/E1/`

> 阶段性质：**分母建立阶段。允许弱，必须正确。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

用 24 维行级输入 + MLP（**无任何序列上下文**）对接三段式对齐损失，建立纯深度学习的绝对下限、容量标定与训练曲线基线；同时把**连续头参数化与损失修正**（POR 可到 0、SW 训练折仿射归一化、`L_aux` 逐目标尺度归一化、`masked_mean` 防 NaN、PERM log 空间官方截断）在这一层就冻结；硬 Gate OOF Total ≥ 78.0。

## 2. 为什么这么设计

1. 在引入序列主干之前，必须先知道"只看当前深度点"能拿到多少分，否则无法证明序列上下文的价值（`资料库/08` §0.3 第 1 层）。
2. 行级基线训练极快（分钟级），是验证损失实现、数据管线、OOF 流程是否正确的最高性价比手段。
3. v2 E4/P3 的 CPU MLP 是 NO-GO，但那是逐点+无 NPU/GPU+小容量；E1 要在 Ascend 910B 上给出"正确实现下的行级上限"，作为 E3 的对照。
4. 改进 proposal（§3/§4）：若连续头仍用 `0.1+softplus(g)` 并被 SW 的 99.9 主导 `L_aux`，E3/E4 只会更快地优化一个错误目标；因此 POR/SW 参数化、`L_aux` 归一化、`masked_mean` 防 NaN 与 PERM 截断必须在 E1 就定稿，并配 `POR=0`/`POR<0.1` 切片单测。

## 3. 输入

- E0 冻结的数据卡、折、评分器
- `资料库/12` §2.4 的对齐损失参考实现

## 4. 产物

- `models/E1/pd0.pt`、`experiments/E1/P1/pd0/oof.npz`
- `reports/E1_loss_curve.csv`、`reports/E1_gate.json`
- `versions/candidates.json::E1_PD0`

## 5. 代码归属

- `src/features/basic.py`（F_raw + F_miss + F_depth）
- `src/losses/score_aligned.py`
- `src/models/row_mlp.py`
- `E1/code/train_row.py`、`E1/code/eval_oof.py`

## 6. P 级子计划

- [P0 行级输入管线与分片缓存](P0/PLAN.md)
- [P1 行级 MLP + 评分对齐损失 + 5 折 OOF（硬 Gate ≥ 78.0）](P1/PLAN.md)

## 7. 完成判据（Gate）

- 80 井 5 折 OOF Total **≥ 78.0** 且 5 折方向一致。
- 占位行上逐目标 Acc **≥ 0.98**。
- loss 曲线无 NaN；训练可 `--resume`。
- `atomic_precision_reported` 与 `contract_ok` 均写入 Gate。
- 连续头参数化与损失修正落地：POR 能输出 0、SW 训练折仿射归一化可反变换、`L_aux` 用训练折稳健尺度归一化并写入 manifest、`masked_mean` NaN 单测与 `POR=0`/`POR<0.1` 切片单测通过。

## 8. 禁止事项

- 加入任何窗口/序列特征（那属于 E2/E3）
- 用 outer 折标签做早停或选阈值
- POR 连续头使用 `0.1 + softplus(g)`（锁死下界，无法输出实测的 0.0）
- 把 SW 当 `[0,1]` 小数尺度或做任何 ×100 换算
- 在整表上 fit `s_por`/`s_sw`/`sw_mu`/`sw_sigma`（只能训练折 fit）
- 在 `masked_mean` 里让缺测位置的 NaN 污染均值

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 NPU/GPU）负责代码与契约，云端（1×Ascend 910B 64GB，CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64，**30 GB 云盘（/data）**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
