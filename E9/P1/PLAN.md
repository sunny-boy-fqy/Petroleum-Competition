# E9/P1 16 井次级体检与泄漏终审

> 所属阶段：[E9](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：验证：泄漏是红线　|　**依赖**：E9/P0

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

## 13. Gate 预注册要点

预注册文件：`v4/reports/E9_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E9_P1_gate",
  "stage": "E9",
  "p_stage": "P1",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "no_high_risk_leak",
  "mandatory_checks": [
    "leakage_audit_complete",
    "confirm_no_breakdown"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
