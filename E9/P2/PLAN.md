# E9/P2 A 榜短名单仲裁（预算制 ≤3 次）

> 所属阶段：[E9](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：外部仲裁：只看崩坏，不调参　|　**依赖**：E9/P1
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

按预算制提交 ≤3 次（每日上限 5 次，留 1 次余量给最终提交），登记 `E9_a_board_log.json`，只做短名单仲裁与崩坏体检。

## 2. 为什么需要这一步

1. A 榜只有 5 口井，噪声带约 ±0.02 分（`v3/PLAN.md` 继承资产表），**不能**用于细粒度调参，否则线上会掉 3–10 分（`资料库/12` §3.5）；
2. A 榜的价值是筛掉"本地好、线上崩"的候选（例如 SW 被裁剪、原子门误判）；
3. 额度是共享硬瓶颈（5 次/日），必须预算制。

## 3. 输入契约

- E9/P0 的短名单 result.zip
- 历史 A 榜锚点（B0=82.2757、E10=81.8576、E13=82.3035）

## 4. 输出契约

- `$V4_REPORTS_DIR/E9_a_board_log.json`（候选/时间/分数/用途）
- `短名单的 A 榜排序结果`

## 5. 执行步骤

1. 挑选最多 3 个最多样化的短名单候选（不是分数最高的 3 个）
2. 逐个提交并记录 `candidate_id/zip_sha256/submit_time/a_board_score`
3. 对比锚点：任一次低于 B0 锚点 0.10 以上 → 立即回退该路线
4. 汇总 A 榜排序与本地 OOF 排序的一致性（三口径一致性分析的第一部分）
5. **记录但不据反馈修改任何模型**

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 预算 | ≤3 次/日（留 1 次） | 冻结 | 每日上限 5 次 |
| 仲裁判据 | 不崩坏（≥ 锚点 − 0.10） | 冻结 | 不做细粒度比较 |
| 候选多样性 | 不同主干/不同原子门策略 | 冻结 | 避免 3 个几乎相同的候选 |

## 7. 完成判据

- 每次提交都有完整记录（含 zip sha256 与用途）
- 无候选低于锚点 0.10 以上（若有则记录回退）
- 明确声明"A 榜只做仲裁，不用于调参"
- A 榜与本地 OOF 的一致性判断写入报告

## 8. 禁止事项

- 用 A 榜反馈调整模型/阈值/权重
- 单日超过 5 次
- 把 A 榜分数当作优于本地 OOF 的证据

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| A 榜波动误判 | ±0.02 噪声被当成真实差异 | 只做 ±0.10 级别的崩坏判定 |
| 额度浪费 | 提交了 3 个几乎相同的候选 | 先做候选多样性检查 |

## 10. 停止规则

- 当日额度用尽 → 候选进 pending 队列，次日再提交

## 11. 代码归属

- `E9/code/submit_batch.py`

## 12. 复算与证据

- `reports/E9_a_board_log.json`

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

预注册文件：`v4/reports/E9_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E9_P2_gate",
  "stage": "E9",
  "p_stage": "P2",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "a_board_no_breakdown",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "max_degradation": 0.1
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
