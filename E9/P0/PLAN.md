# E9/P0 OOF 汇总与提交护栏判定

> 所属阶段：[E9](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：验证：不出新模型　|　**依赖**：E8/P2
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

汇总全部候选的 80 井 OOF、逐目标 Acc、bootstrap CI，运行 `choose_submission.py` 护栏，输出可提交短名单与决策记录。

## 2. 为什么需要这一步

1. 护栏防止"本地漂亮但明显弱于历史锚点"的候选被提交；
2. 本地 80 井 OOF 是**主判据**，必须与历史锚点同折可比（同一 `folds.json`）；
3. 所有本地分数必须标 `selection_score_only=true`——因为它参与了折内选择。

## 3. 输入契约

- 各候选的 `oof.npz` 与 `cv.json`
- `constants.B0_LOCAL_OOF/B0_A_BOARD/GUARDRAIL_*`

## 4. 输出契约

- `$V4_REPORTS_DIR/E9_validation_report.json`（全部候选对比表）
- `$V4_REPORTS_DIR/E9_submission_decision.json`
- `versions/candidates.json`（status 更新）

## 5. 执行步骤

1. 汇总每个候选的 OOF：总分、逐目标、连续切片、占位行、逐折 delta、bootstrap CI
2. 计算护栏下限 `guardrail_floor = max(75.0, B0_LOCAL_OOF − 1.0, B0_A_BOARD − 0.5)`
3. 对每个候选判定：OOF ≥ guardrail_floor 且（若 A 榜已知）A ≥ B0_A_BOARD − 0.5
4. 输出短名单（≤3）与每个候选的 `choice/reason/candidate_oof/guardrail_floor`
5. 把 `selection_score_only=true` 与口径 `missing_mode=drop` 写入报告头部

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `guardrail_floor` | max(75, 80.382479−1.0, 82.2757−0.5) | 冻结 | 常量在 `constants.py` |
| `protocol_matched` | false | 冻结 | v1 同折 parity 不可复算，仅外部参照 |
| bootstrap | 按井行数加权 cluster，1000 次 | 冻结 | 与各 Gate 口径一致 |

## 7. 完成判据

- 每个候选的护栏判定可复算（脚本 + 输入 sha256 + 输出）
- 短名单 ≤3 且每个都有明确进入理由
- 报告显式标注 `selection_score_only` 与 `missing_mode`
- 未过护栏的候选被标记 `rejected` 且保留在注册表（负资产）

## 8. 禁止事项

- 用未标口径的分数做比较
- 把 `selection_score_only` 数字伪装成独立确认
- 静默丢弃未过护栏的候选

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 候选间口径不一致 | 对比失真 | 统一用同一评分器与同一折，报告中标注口径字段 |
| 护栏把所有候选挡掉 | 只能回退 B0 | 这正是护栏的目的；按 §9.5 走回退协议 |

## 10. 停止规则

- 短名单为空 → 直接进入 E10 的回退协议分支

## 11. 代码归属

- `E9/code/aggregate_oof.py`
- `E9/code/choose_submission.py`

## 12. 复算与证据

- `reports/E9_submission_decision.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E9
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

预注册文件：`v4/reports/E9_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E9_P0_gate",
  "stage": "E9",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "guardrail_pass",
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
    "no_label_leak",
    "guardrail_evaluated"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
