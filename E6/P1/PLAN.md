# E6/P1 原子门 τ 搜索（inner-OOF，硬切换）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：开关设定：决定"输出常量还是连续"　|　**依赖**：E6/P0

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

在 inner-OOF 上逐目标搜索门限 τ_t，实现**硬切换**解码，并报告误判代价分解。

## 2. 为什么需要这一步

1. τ 决定每个点是走常量分支还是连续分支，是纯 DL 管线唯一保护屏障的开关；
2. 误判代价不对称：把有效行判成常量会立刻丢分，把占位行判成连续同样丢分，两者代价需分别量化；
3. **禁止在常量与连续之间线性插值**：POR 容差仅 ±0.008，插值必然出带。

## 3. 输入契约

- E6/P0 的 q 概率（inner-OOF）
- E3–E5 的连续预测（inner-OOF）

## 4. 输出契约

- `src/inference/atomic_gate.py`
- `$V4_REPORTS_DIR/E6_tau_search.json`
- `三个 τ 值写入 `versions/candidates.json::PD1.atomic.tau``

## 5. 执行步骤

1. 对每个目标，在 inner-OOF 上网格搜索 τ ∈ [0.05,0.95]（步长 0.01）
2. 目标函数 = 该目标的官方 Acc（drop 口径）
3. 报告：τ 曲线、最优 τ、误判代价分解（FP 代价 vs FN 代价）
4. 验证 τ 的稳定性：不同 inner 折选出的 τ 是否接近（方差过大则不可靠）
5. 在三目标上分别确定 τ，并记录到候选注册表

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| τ 搜索范围 | [0.05, 0.95]，步长 0.01 | 冻结 | 逐目标独立 |
| 目标函数 | 该目标官方 Acc | 冻结 | 不是 F1 |
| 硬切换 | q ≥ τ → 输出精确常量 | 冻结 | 禁止插值 |
| 稳定性判据 | 不同 inner 折最优 τ 的极差 ≤ 0.2 | 冻结 | 超限则用更保守 τ |

## 7. 完成判据

- 三个 τ 都只在 inner-OOF 上选出，过程可复算
- 误判代价分解表完整
- τ 稳定性通过（跨 inner 折极差 ≤ 0.2），否则取更保守值并说明
- 占位行逐目标 Acc ≥ 0.99（这是 Gate 硬条件）

## 8. 禁止事项

- 用 outer 折或 A 榜选 τ
- 在常量与连续输出之间做线性插值
- 用 F1 而非官方 Acc 作为 τ 的目标函数

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| τ 过拟合 inner | inner 最优但 outer 变差 | 报告 τ 敏感性曲线；取平坦区间的中点 |
| 误判代价不对称被忽视 | 总分下降但 F1 上升 | 以官方 Acc 为目标函数 |

## 10. 停止规则

- τ 搜索若无法让占位 Acc ≥ 0.99 → 回到 E6/P0 加强 H0

## 11. 代码归属

- `src/inference/atomic_gate.py`
- `E6/code/search_tau.py`

## 12. 复算与证据

- `reports/E6_tau_search.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E6
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E6_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E6_P1_gate",
  "stage": "E6",
  "p_stage": "P1",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "atomic_f1",
  "thresholds": {
    "min_atomic_acc": 0.99
  },
  "mandatory_checks": [
    "atomic_precision_reported",
    "inner_only_selection"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
