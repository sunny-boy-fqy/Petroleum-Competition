# E6/P1 原子门 τ 搜索（inner-OOF，硬切换）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：开关设定：决定"输出常量还是连续"　|　**依赖**：E6/P0
>
> **状态**：⏸ 待执行　

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

预注册文件：`v4/reports/E6_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E6_P1_gate",
  "stage": "E6",
  "p_stage": "P1",
  "created_at": "<ISO8601，写盘时填写>",
  "primary_metric": "atomic_f1",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "min_atomic_acc": 0.99
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
