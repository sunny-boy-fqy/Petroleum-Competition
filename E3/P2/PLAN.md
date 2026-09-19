# E3/P2 行级对照 + 感受野消融 + 硬 Gate（≥81.0）

> 所属阶段：[E3](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：**判据阶段**：决定序列路线是否继续　|　**依赖**：E3/P1、E1/P1

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

同折同头对比序列主干与行级 MLP；做感受野消融（U-Net depth 3/5、TCN dilation 上限 64/512）；判定硬 Gate：OOF ≥ 81.0 **且** 序列主干显著优于同头行级模型。

## 2. 为什么需要这一步

1. 这是"上下文是否真被利用"的唯一判据，也是 E4/E5 是否值得继续的前提；
2. 若缩小感受野不降分，说明主干没学到长程结构，此时**扩容是浪费机时**，应先修数据/结构/损失；
3. 前代所有序列/井级尝试都是 NO-GO，因此本次必须给出比"分数提高了"更硬的证据：同折、同头、同特征的受控对照 + 感受野消融。

## 3. 输入契约

- E3/P1 的 5 折 OOF、E1/P1 的行级 OOF
- E0 的评分器与 bootstrap 工具

## 4. 输出契约

- `$V4_RUN_ROOT/E3/oof.npz`（合并 5 折，逐行）
- `$V4_REPORTS_DIR/E3_receptive_field_ablation.json`
- `$V4_REPORTS_DIR/E3_row_vs_seq.json`
- `$V4_REPORTS_DIR/E3_gate.json`

## 5. 执行步骤

1. 预注册 `E3_P2_gate_prereg.json`（含 `min_delta`、`primary_metric`、`candidate_budget`）
2. 跑 U-Net 与 TCN 各 5 折（可并行任务拆分），汇总 OOF
3. 把**同一份行级头**装在行级特征上训练一遍（同折同超参），作为受控对照
4. 感受野消融：depth∈{3,5} × dilation_max∈{64,512}，各跑 fold0+1 筛查
5. 计算序列 vs 行级的 paired cluster bootstrap（按井行数加权，1000 次）
6. 逐折 delta、逐井非退化比例、占位行 Acc、连续切片 Acc 一并上报
7. 写 Gate 并判定

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 主判据 | OOF Total ≥ 81.0 | 硬 Gate | 低于此值不得进入 E4 |
| 显著性判据 | 序列 vs 行级 bootstrap CI 下界 > 0 | 冻结 | 受控对照必须有 |
| 消融预算 | fold0+1 筛查，胜者再跑全 5 折 | 冻结 | 省机时且避免看全折后挑结构 |
| 感受野变量 | depth {3,5} × dilation {64,512} | 冻结 | 4 个组合 |

## 7. 完成判据

- OOF Total **≥ 81.0**
- 序列主干相对**同头同特征**行级模型 CI 下界 > 0（若为负，判 NO-GO 并记录证据）
- 感受野消融表完整；若缩小感受野不降分，必须给出"上下文未被利用"的结论与修正计划
- 5 折 delta 全部同向；`atomic_precision_reported` 与 `contract_ok` 为 true

## 8. 禁止事项

- 跳过消融直接堆容量
- 用 outer 折（含 fold0）验证标签选超参——fold0 只作资源筛查，其结论必须标 `selection_score_only`
- 在未通过 CI 判据的情况下宣称"序列有效"

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 序列不优于行级 | CI 含 0 或为负 | 先查数据分块/覆盖、归一化、损失；最多两次结构修订后降级为 NO-GO |
| 分数高但来自容量而非上下文 | 缩小感受野分数不降 | 以感受野消融结论为准，判定上下文未被利用 |
| 机时超支 | 单折耗时持续增长 | 先在 fold0+1 筛查，胜者才跑全折 |

## 10. 停止规则

- Gate 未过 → 触发"最多两次结构修订"规则；仍不过则记录 NO-GO 并把主线降级为行级 + 手工窗口特征，供 E5/E6 继续

## 11. 代码归属

- `E3/code/compare_row_vs_seq.py`
- `E3/code/rf_ablation.py`
- `E3/code/gate.py`

## 12. 复算与证据

- `reports/E3_gate.json`、`reports/E3_row_vs_seq.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E3
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E3_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E3_P2_gate",
  "stage": "E3",
  "p_stage": "P2",
  "candidate_budget": 4,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "oof_total",
  "baseline_version": "E1_PD0",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "oof_total_min": 81.0
  },
  "mde_units": 80,
  "multiplicity": "holm",
  "mandatory_checks": [
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "no_label_leak"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
