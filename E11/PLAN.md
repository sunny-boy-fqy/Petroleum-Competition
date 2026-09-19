# E11 归档与复盘

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E11/src/、v4/E11/code/`　|　产物 `v4/experiments/E11/、v4/models/E11/`

> 阶段性质：**收尾阶段。不出候选、不提交。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

把 v4 的代码/模型/候选/报告/A 榜历史归档为可独立理解的知识包，并写出三口径一致性复盘与下一代方向储备。

## 2. 为什么这么设计

1. v1 的 `v1.md` 与 v2 的复盘是后续版本最有价值的输入；v4 必须留下同等质量的复盘。
2. 失败路线（NO-GO）与成功路线同等重要——它们决定下一代是否重复踩坑。

## 3. 输入

- v4 全部 reports 与 experiments
- A 榜日志、B 榜结果（若可得）

## 4. 产物

- `reports/E11_archive_manifest.json`（含 sha256）
- `reports/E11_retrospective.md`、`reports/E11_next_directions.md`

## 5. 代码归属

- `E11/code/archive.py`、`E11/code/retrospective.py`

## 6. P 级子计划

- [P0 资产归档](P0/PLAN.md)
- [P1 复盘与下一代方向](P1/PLAN.md)

## 7. 完成判据（Gate）

- 归档清单完整且每项有 sha256；可在不含 v1/v2/v3 的目录中独立理解。
- 复盘含三口径对照表、路线有效性表、资源统计表（训练时长/磁盘/A 榜配额）。
- 下一代方向清单含明确触发条件。

## 8. 禁止事项

- 删除任何候选或报告（含 rejected）
- 把 B 榜成绩用于事后调参并写入复盘之外的地方

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.6 / torch 2.4.0 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
