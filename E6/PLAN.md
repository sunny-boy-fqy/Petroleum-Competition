# E6 联合常量状态与原子门

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 ``v4/E6/src/`、`v4/E6/code/``　|　产物 ``v4/experiments/E6/`、`v4/models/E6/``

> 阶段性质：**完整管线打通阶段（PD1 候选诞生）。**

## 1. 目标

实现 H0 联合常量状态头与逐目标学习型原子门（硬切换），使占位行精确命中，并用 inner-OOF 选择门限 τ；完整 PD1 管线 OOF ≥ 82.0。

## 2. 为什么这么设计

1. 66.719% 的行是联合常量占位，占约 66.72 分的白送分；任何软融合都会把 POR 推出 ±0.008 容差带。
2. 占位状态可从输入预测（v1 的原子门已验证），因此应把"是否输出常量"做成显式可学习决策，而不是让回归头勉强逼近。
3. 硬切换保证了原子点的精确性，是纯 DL 管线（无 B0 patch 隔离）下唯一的保护屏障。

## 3. 输入

- E3/E4/E5 的逐行表示与预测
- `资料库/12` §3.4

## 4. 产物

- `src/models/state_head.py`、`src/inference/atomic_gate.py`
- `models/E6/pd1_*.pt`、`experiments/E6/P2/pd1/{result.json,result.zip,cv.json}`
- `reports/E6_gate.json`、`reports/E6_atomic_report.json`

## 5. 代码归属

- `E6/code/train_state.py`、`E6/code/search_tau.py`、`E6/code/build_pd1.py`

## 6. P 级子计划

- [P0 联合常量状态头](P0/PLAN.md)
- [P1 原子门 τ 搜索（inner-OOF）](P1/PLAN.md)
- [P2 PD1 完整管线与 Gate](P2/PLAN.md)

## 7. 完成判据（Gate）

- OOF Total **≥ 82.0**（超过 B0 本地锚点 80.382479）。
- 占位行逐目标 Acc **≥ 0.99**；τ 的选择过程可复算（只用 inner-OOF）。
- 提交契约通过（10 井 / 95,948 行）且 `predict.py --use-version PD1` 可在本机 CPU 跑通。

## 8. 禁止事项

- 在占位与连续分支之间做线性插值
- 用 outer 折选 τ

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.6 / torch 2.4.0 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
