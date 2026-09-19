# E10/P2 提交执行与归档登记

> 所属阶段：[E10](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：唯一外部动作　|　**依赖**：E10/P1
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

执行提交（预算内），登记平台返回；把提交指纹与结果写入候选注册表用于赛后复盘。

## 2. 为什么需要这一步

1. 提交是唯一不可逆的外部动作，必须完全可追溯（zip sha256、时间、返回分数）；
2. 赛后 B 榜出来后需要把"本地 OOF / A 榜 / B 榜"三口径对齐分析，因此提交当时的配置指纹必须被记录；
3. 提交后不得再改动候选文件（否则破坏可追溯性）。

## 3. 输入契约

- E10/P1 的 submission zip
- 平台提交入口（人工操作）

## 4. 输出契约

- `$V4_REPORTS_DIR/E10_submission_log.json`
- `versions/candidates.json`（`status=submitted/frozen_best` + a_board_score）

## 5. 执行步骤

1. 确认当日额度与提交包 sha256
2. 人工/脚本提交 `result.zip` 与 `submission_code.zip`
3. 记录返回（分数/时间/错误信息）到 submission_log
4. 更新候选注册表状态；**冻结该候选文件**（不再修改）
5. 若提交的是 fallback 包，显式标注 `choice=b0_fallback` 与原因

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 每日额度 | ≤5 次 | 硬约束 | — |
| 提交后 | 候选冻结 | 冻结 | 修改必须新建 candidate_id |

## 7. 完成判据

- 提交记录含 zip sha256、时间、返回分数或错误
- 候选注册表状态一致（`submitted`/`frozen_best`）
- 提交的包文件被标记为不可变（记录 sha256 并停止修改）

## 8. 禁止事项

- 提交后修改候选文件
- 根据 A 榜返回立刻改模型再提交（每日额度有限，且属于 A 榜过拟合）

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 提交格式错 | 平台报错 | 契约校验 + 干净目录复现已前置 |
| 额度用尽 | 无法提交 | 预算制：E9 最多 3 次，留额度给 E10 |

## 10. 停止规则

- 提交失败 → 立即用 fallback 包重试（若额度允许），否则次日

## 11. 代码归属

- `E10/code/submit.py`

## 12. 复算与证据

- `reports/E10_submission_log.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E10
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

预注册文件：`v4/reports/E10_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E10_P2_gate",
  "stage": "E10",
  "p_stage": "P2",
  "created_at": "<ISO8601，写盘时填写>",
  "primary_metric": "submission_recorded",
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
