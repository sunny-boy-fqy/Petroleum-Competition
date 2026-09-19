# E9/P1 16 井次级体检与泄漏终审

> 所属阶段：[E9](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：验证：泄漏是红线　|　**依赖**：E9/P0
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

对 top-2 候选在 v2 冻结的 16 井 `folds_confirm` 上各跑一次；完成四类泄漏审计（折维度/特征来源/标准化 fit 范围/伪标签来源）。

## 2. 为什么需要这一步

1. 16 井来自 v1 训练集，是 **v1-exposed** 的次级体检，只能防崩坏、不能宣称独立确认；
2. `资料库/12` §3.5 与 §8.5 列出完整泄漏清单；本地 OOF 虚高的主要来源就是同井相邻点跨折与折外 fit；
3. 泄漏审计是提交前的最后一道关卡，失败即放弃该候选。

## 3. 输入契约

- top-2 候选权重
- `../v2/artifacts/E0/`（folds_confirm）
- `资料库/12` §3.5

## 4. 输出契约

- `$V4_REPORTS_DIR/E9_confirm.json`（16 井体检分数）
- `$V4_REPORTS_DIR/E9_leakage_audit.json`

## 5. 执行步骤

1. 在 16 井上推理 top-2 候选（权重不重训），评分并与 80 井 OOF 对比
2. 判定崩坏：16 井分数较 80 井 OOF 下降 > 1.5 分即触发复核
3. 泄漏审计 1：折维度——确认所有折都是井维度，无同井跨折
4. 泄漏审计 2：特征来源——provenance CSV 中无目标派生列
5. 泄漏审计 3：标准化 fit 范围——所有 scaler/分位数参数只在训练折 fit
6. 泄漏审计 4：伪标签来源——transductive/伪标签只用输入分布或训练折模型输出
7. label-shuffle 阴性对照：随机打乱标签后 OOF 应接近常数基线

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| confirm 井数 | 16 | 冻结 | v1-exposed，标注清楚 |
| 崩坏阈值 | 1.5 分 | 冻结 | 超过则复核 |
| 审计项 | 4 类 | 冻结 | 缺一不可 |

## 7. 完成判据

- 16 井体检完成并标注 `v1_exposed=true`、`not_independent_confirmation=true`
- 四类泄漏审计全部有结论（通过或标注残余风险）
- label-shuffle 对照分数接近常数基线（证明无标签泄漏）
- 无候选出现 > 1.5 分崩坏（若有则排除该候选）

## 8. 禁止事项

- 把 16 井体检当独立确认宣称显著增益
- 用 confirm 结果回头调阈值/权重/结构
- 跳过任何一类泄漏审计

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 16 井噪声大 | 分数波动被误读 | 只做崩坏判定，不做增益判定 |
| 审计疏漏 | 泄漏进入提交 | 审计清单固定 4 类，逐项写结论与证据 |

## 10. 停止规则

- 发现高风险泄漏 → 该候选立即作废，E10 走回退协议

## 11. 代码归属

- `E9/code/confirm_check.py`
- `E9/code/leakage_audit.py`

## 12. 复算与证据

- `reports/E9_confirm.json`、`reports/E9_leakage_audit.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E9
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 12.5 选择协议（H1：inner-OOF only）

**所有超参/阈值/早停/结构选择只允许用 inner 折**（`$V4_REPORTS_DIR/E0_folds.json::inner`）。

| 用途 | 允许的数据 | 禁止 |
|---|---|---|
| 超参/阈值/λ/τ/集成权重选择 | 该 outer 折的 inner-OOF | outer 验证折标签 |
| 早停 | inner-OOF 的真实 `score.py` 分数 | outer 折分数、loss 值 |
| 结构/特征筛查（省机时） | 可先用 fold0 做**资源预检** | 预检结论不得进入 Gate 数值 |

> 若某步骤确实只能看 outer 折（例如最终 OOF 汇总），该步骤**不得**反过来影响任何选择；
预检性质的 fold0 结果必须在报告中标 `exploratory=true`、`selection_score_only=true`。

## 13. Gate 预注册要点

预注册文件：`v4/reports/E9_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E9_P1_gate",
  "stage": "E9",
  "p_stage": "P1",
  "created_at": "<ISO8601，写盘时填写>",
  "primary_metric": "no_high_risk_leak",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0
  },
  "alpha": 0.05,
  "multiplicity": "none",
  "candidate_budget": 1,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "pilot_std": null,
  "mde_units": 80,
  "min_detectable_effect": null,
  "planned_task_training_h": 1.0,
  "mandatory_checks": [
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "checkpoint_resumable",
    "no_label_leak"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
