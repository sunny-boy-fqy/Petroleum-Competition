# E6 联合常量状态与原子门

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E6/src/、v4/E6/code/`　|　产物 `v4/experiments/E6/、v4/models/E6/`

> 阶段性质：**完整管线打通阶段（PD1 候选诞生）。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

把 H0 联合占位头升级为**逐目标原子头 `q_por/q_perm/q_sw`（主保护）+ 辅助 `q_joint`（可选高置信硬门禁，默认关）**，逐目标独立硬切换，用 inner-OOF **官方总分**选择门限 τ_t；完整 PD1 管线 OOF ≥ 82.0。

## 2. 为什么这么设计

1. 66.719% 的行是联合常量占位，占约 66.72 分的白送分；任何软融合都会把 POR 推出 ±0.008 容差带。
2. 占位状态可从输入预测（v1 的原子门已验证），因此应把"是否输出常量"做成显式可学习决策，而不是让回归头勉强逼近。
3. 硬切换保证了原子点的精确性，是纯 DL 管线（无 B0 patch 隔离）下唯一的保护屏障。
4. 改进 proposal §1/§2：单目标原子行并不少（SW 31,030 / PERM 7,373 / POR 157），一个 joint 概率同时决定三目标会漏保护非 joint 原子行、又误伤 joint 行里的非原子目标；因此必须逐目标原子头，且 τ_t 要按官方总分（而非 F1/Acc）选。

## 3. 输入

- E3/E4/E5 的逐行表示与预测
- `资料库/12` §3.4

## 4. 产物

- `src/models/row_mlp.py`（`RowMLP` 的 `q_joint + q_por/q_perm/q_sw` 五个头）、`src/inference/atomic_gate.py`
- `models/E6/pd1_*.pt`、`experiments/E6/P2/pd1/{result.json,result.zip,cv.json}`
- `reports/E6_gate.json`、`reports/E6_atomic_report.json`

## 5. 代码归属

- `E6/code/train_state.py`、`E6/code/search_tau.py`、`E6/code/build_pd1.py`

## 6. P 级子计划

- [P0 逐目标原子头 + 辅助 joint 头（两阶段训练）](P0/PLAN.md)
- [P1 逐目标原子门 τ_t 搜索（inner-OOF 官方总分）](P1/PLAN.md)
- [P2 PD1 完整管线组装与硬 Gate（≥82.0）](P2/PLAN.md)

## 7. 完成判据（Gate）

- OOF Total **≥ 82.0**（超过 B0 本地锚点 80.382479）。
- **逐目标**占位行 Acc **≥ 0.99**、recall **≥ 0.98**，并上报 precision/F1；τ_t 的选择过程可复算（只用 inner-OOF 官方总分）。
- 提交契约通过（10 井 / 95,948 行）且 `predict.py --use-version PD1` 可在本机 CPU 跑通。
- 两阶段训练记录完整；`joint_guard` 默认关闭且启用与否有 inner-OOF 证据。
- **原子/连续之间无任何插值**，`no_atom_continuous_interpolation` 为 true。

## 8. 禁止事项

- 在占位与连续分支之间做线性插值
- 用 outer 折或 A 榜选 τ
- 用 F1/准确率而非官方总分选 τ_t
- 默认启用 `joint_guard`
- 把 joint 头当作三目标的唯一保护（丢失逐目标原子行）

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.8 / torch 2.7.1 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
