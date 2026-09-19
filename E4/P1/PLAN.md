# E4/P1 多尺度融合（CNN × Transformer）与 Gate

> 所属阶段：[E4](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：融合候选：必须超过最佳单主干才算增益　|　**依赖**：E4/P0、E3/P2
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

把 CNN 主干与 PatchTF 的逐行表示按门控或 concat 融合后送同一组头，判定是否超过**最佳单主干**；否则 NO-GO 并保留 E3 结构。

## 2. 为什么需要这一步

1. 多尺度是最常见的稳定增益来源，但必须证明超过最好单主干，否则只是参数变多；
2. CNN 的局部形态与注意力的长程依赖在测井上确实互补（`资料库/08` §0.3 第 3–4 层）；
3. 融合层参数量小，是"低成本换分"的候选。

## 3. 输入契约

- E3/P1 的 U-Net/TCN 权重与逐行表示
- E4/P0 的 PatchTF

## 4. 输出契约

- `src/models/multiscale.py`
- `$V4_RUN_ROOT/E4/oof.npz`
- `$V4_REPORTS_DIR/E4_gate.json`

## 5. 执行步骤

1. 实现三种融合：① 门控加权（可学习标量/向量门）；② concat + 1×1 卷积降维；③ 表示层平均
2. 冻结两个主干的预训练权重先做快速筛查（只训融合层与头）
3. 胜出方案再解冻联合微调（小 lr）
4. 与最佳单主干做同折 paired bootstrap
5. 写 Gate：融合 ≥ 最佳单主干且 CI 下界 > 0

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 融合方式 | 门控（默认） | 门控/concat/平均 | inner-OOF 选（fold0 仅资源预检） |
| 融合层 lr | 1e-3（冻结主干）/ 2e-4（解冻） | — | 解冻时用更小 lr |
| 候选数 | 3 | — | multiplicity=holm 校正 |

## 7. 完成判据

- 融合方案相对最佳单主干的 paired bootstrap CI 下界 > 0，否则判 NO-GO 并保留 E3
- 同源性报告：两主干 OOF 预测的相关系数（过高说明融合收益可疑）
- 参数量与耗时增量记录完整

## 8. 禁止事项

- 用两个高度同源的分支冒充多尺度
- 在未与单主干对照的情况下宣称融合有效

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 融合无增益 | CI 含 0 | 判 NO-GO，保留 E3 主干；把预算让给 E5/E6 |
| 同源性高 | 两主干预测相关 > 0.99 | 检查是否实现同一结构；若确实同源则融合无意义 |

## 10. 停止规则

- 融合 CI 上界 ≤ 0 → NO-GO，E5 直接基于 E3 主干

## 11. 代码归属

- `src/models/multiscale.py`
- `E4/code/fuse_multiscale.py`

## 12. 复算与证据

- `reports/E4_gate.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E4
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

预注册文件：`v4/reports/E4_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E4_P1_gate",
  "stage": "E4",
  "p_stage": "P1",
  "created_at": "<ISO8601，写盘时填写>",
  "primary_metric": "oof_total",
  "primary_threshold_key": "min_delta",
  "baseline_version": "E3_best",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0
  },
  "alpha": 0.05,
  "multiplicity": "holm",
  "candidate_budget": 3,
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
