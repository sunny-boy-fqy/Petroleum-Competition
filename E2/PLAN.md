# E2 特征工程与数据管线

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 ``v4/E2/src/`、`v4/E2/code/``　|　产物 ``v4/experiments/E2/`、`v4/models/E2/``

> 阶段性质：**特征与吞吐阶段。每组特征必须独立消融。**

## 1. 目标

建立 `F_phys`/`F_win`/`F_well` 三组特征、数据增强与按井分片缓存，并把 16 GiB 内存与 30 GB 磁盘的工程约束固化为可复用的数据管线。

## 2. 为什么这么设计

1. 序列主干需要稠密数值输入；物理交会特征（`资料库/01`/`02`）与窗口统计（`资料库/07` §6）在 v1 已被证明有效（C1→C1W +1.0562）。
2. 16 GiB 系统内存是真正的瓶颈：必须把"按井分片 + 按需读取 + 即时增强"写成管线，否则 E3 一开始就会 OOM。
3. 特征一旦进入训练就必须冻结版本；先定义再实验是防止"看 OOF 后加列"的唯一办法。

## 3. 输入

- E0 数据卡与分片
- `资料库/01`、`02`（物理公式）、`资料库/07` §6/§10（窗口与增强）

## 4. 产物

- `src/features/physics.py`、`src/features/window.py`、`src/features/well.py`
- `cache/feat/F2/**`、`reports/E2_feature_provenance.csv`
- `reports/E2_ablation.json`、`reports/E2_mem_profile.json`

## 5. 代码归属

- `E2/code/build_features.py`、`E2/code/ablate_groups.py`、`E2/code/mem_profile.py`

## 6. P 级子计划

- [P0 物理与交会特征](P0/PLAN.md)
- [P1 窗口与井级特征](P1/PLAN.md)
- [P2 增强与吞吐标定](P2/PLAN.md)

## 7. 完成判据（Gate）

- 三组特征各自在行级 MLP 上有消融结果（增量或 NO-GO 均须登记）。
- `reports/E2_feature_provenance.csv` 登记每个派生列的公式与来源，无标签派生列。
- 峰值常驻内存 < 12 GiB、on-disk 缓存 < 2 GB。

## 8. 禁止事项

- 任何使用目标值的派生特征
- 跨井 fit 的标准化参数（必须只在训练折 fit）

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.6 / torch 2.4.0 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
