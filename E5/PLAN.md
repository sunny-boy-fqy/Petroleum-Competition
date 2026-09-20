# E5 逐目标精修

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E5/src/、v4/E5/code/`　|　产物 `v4/experiments/E5/、v4/models/E5/`

> 阶段性质：**分目标攻坚阶段（三个目标互相独立）。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

针对 POR（±0.008 窄带）、PERM（log 域长尾）、SW（**单尺度**：占位尖峰 99.9 + 有效分布 8.3–99.9）分别设计专用连续头与解码，逐目标独立判定增益。

## 2. 为什么这么设计

1. 总分按目标可加（`rules.md` §7.4），边际收益排序为 PERM > SW ≈ POR（`资料库/12` §3.3）。
2. POR 的容差带只有 ±0.008，精度要求与另外两个目标完全不同；SW 的占位尖峰 99.9 与有效分布（同一标签尺度）形状差异极大，单头线性回归会被尖峰拉扯。
3. 目标独立化让失败可隔离：某一个目标退步不会污染其他两个。
4. 改进 proposal §3：POR 连续头必须能输出 ~0（实测 576 行 <1、186 行 <0.1），SW 连续头必须用**训练折仿射归一化**再反变换回标签尺度，PERM 必须用官方 `max(ẑ−z, log10(eps))` 截断。

## 3. 输入

- E3/E4 冻结主干
- `资料库/12` §3.4（占位尖峰与有效分布）

## 4. 产物

- `src/models/heads.py`、`experiments/E5/{por,perm,sw}/oof.npz`
- `reports/E5_gate.json`、`reports/E5_per_target.json`

## 5. 代码归属

- `E5/code/head_por.py`、`E5/code/head_perm.py`、`E5/code/head_sw.py`

## 6. P 级子计划

- [P0 POR 窄带精修（±0.008）](P0/PLAN.md)
- [P1 PERM log 域精修（长尾与数量级）](P1/PLAN.md)
- [P2 SW 单尺度精修（训练折仿射归一化 + q_sw 硬切换）](P2/PLAN.md)

## 7. 完成判据（Gate）

- 至少一个目标的连续切片 Acc 显著提升（CI 下界 > 0），且其他目标不退步超过 0.01。
- SW 单尺度守卫通过：连续头用训练折 `sw_mu/sw_sigma` 仿射归一化训练、输出反变换回标签尺度（百分数），`SW_SMALL_BRANCH` 永久关闭，无任何 ×100 换算。
- POR 参数化消融表完整（`por_max·sigmoid` 推荐 / `softplus(g)−softplus(g0)` 备选），POR 连续头能输出 0。
- PERM 使用官方 log 空间截断，并给出「低估尾部 vs 官方评分器」一致性报告。

## 8. 禁止事项

- 把 SW 全局裁剪到 [0,1]（禁止；实测会掉约 23 分）
- 用某一目标的增益掩盖另一目标的退化
- POR 连续头使用 `0.1 + softplus(g)`
- 对 SW 做任何 ×100 尺度换算

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 NPU/GPU）负责代码与契约，云端（1×Ascend 910B 64GB，CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64，**64 GiB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
