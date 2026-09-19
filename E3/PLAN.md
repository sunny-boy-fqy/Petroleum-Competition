# E3 深度序列主干

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E3/src/、v4/E3/code/`　|　产物 `v4/experiments/E3/、v4/models/E3/`

> 阶段性质：**主线上限阶段（本计划的核心赌注）。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

实现 1D U-Net 与 TCN 两种深度序列主干（窗口→窗口，**全段 seq2seq、无滑窗中心点**），与行级模型同头对比；硬 Gate：OOF ≥ 81.0 **且**序列主干必须显著优于同头行级模型。

## 2. 为什么这么设计

1. `资料库/08` §0.3 把 1D U-Net / TCN 列为"最可能冲高分的结构"，而前代因 CPU 限制从未真正训练过。
2. 0.1 m 采样意味着 10 m 储层段 = 100 点；没有数百点的感受野，模型看不到层段结构，只能逐点外推。
3. 必须先做**感受野消融**：若缩小感受野不降分，说明上下文没被用上，此时扩容是浪费算力，应先修数据/结构。
4. 改进 proposal §6：U-Net 的 depthwise-separable + 空洞卷积会带来 **padding 边界伪影**，chunk 拼接会带来**接缝跳变**；TCN 在离线任务里必须**非因果**，dilation_max=512（≈50 m 感受野）要先验证有效而不是盲目继续加。

## 3. 输入

- E2 特征管线
- `资料库/08` §4/§6（1D-CNN 与 TCN 细节）

## 4. 产物

- `src/models/unet1d.py`、`src/models/tcn.py`、`src/data/seq_dataset.py`
- `models/E3/{unet,tcn}_fold*.pt`、`experiments/E3/P2/*/oof.npz`
- `reports/E3_gate.json`、`reports/E3_receptive_field_ablation.json`、`reports/E3_boundary_report.json`

## 5. 代码归属

- `E3/code/train_seq.py`、`E3/code/rf_ablation.py`、`E3/code/compare_row_vs_seq.py`

## 6. P 级子计划

- [P0 序列数据集与分块（chunk）策略](P0/PLAN.md)
- [P1 1D U-Net 与 TCN 主干实现](P1/PLAN.md)
- [P2 行级对照 + 感受野消融 + 硬 Gate（≥81.0）](P2/PLAN.md)

## 7. 完成判据（Gate）

- OOF Total **≥ 81.0**；相对同头行级模型的加权配对井级 bootstrap 95% CI 下界 > 0。
- 感受野消融表：depth∈{3,5}、dilation_max∈{64,512} 的对照结果齐备，并给出 512 是否已达饱和的结论。
- **padding 边界伪影检查**：逐目标比较井首/井尾 10 m 与井中段的 Acc，差值超过阈值必须给出解释或修正。
- **全段 seq2seq** 输出（逐行同长，不用滑窗中心点）+ **chunk 重叠推理与加权拼接**（接缝处无跳变）。
- TCN 全程**非因果**；5 折全部同向；checkpoint 可 `--resume`；`disk_guard` 全程未触发 abort。

## 8. 禁止事项

- 使用 ImageNet/自然图像预训练权重
- 把验证井的数据用于训练折统计
- 跳过感受野消融直接堆容量
- 把 TCN 做成因果卷积（本任务是离线整井推理，因果化会白白丢失未来上下文）
- 用滑窗中心点预测代替全段 seq2seq 输出
- 在 chunk 拼接处不做重叠加权（会造成接缝跳变）

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.8 / torch 2.7.1 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
