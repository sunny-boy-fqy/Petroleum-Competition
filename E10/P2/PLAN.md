# E10/P2 提交执行与归档登记

> 所属阶段：[E10](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：唯一外部动作　|　**依赖**：E10/P1

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

## 13. Gate 预注册要点

预注册文件：`v4/reports/E10_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E10_P2_gate",
  "stage": "E10",
  "p_stage": "P2",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "submission_recorded",
  "mandatory_checks": [
    "submission_log_complete"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
