# E8/P2 集成（多 seed / 快照 / 多结构）与 Gate

> 所属阶段：[E8](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：增益放大：必须扣除同源性　|　**依赖**：E8/P0–P1
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

构建多 seed、快照集成与多结构（U-Net/TCN/PatchTF）集成，报告成员同源性与按井行数加权的 paired bootstrap，判定集成是否真增益。

## 2. 为什么需要这一步

1. `资料库/08` §0.3 第 4 层：多模型 Stacking/加权融合 + 快照集成是标准提分手段；
2. **同源平均不构成增益**：若成员间预测相关 > 0.99，融合只是降低方差而非提升上限；
3. 集成的收益必须用统计检验而非点估计确认。

## 3. 输入契约

- E3/E4/E5/E6/E7 的冻结成员
- E0 的 bootstrap 工具

## 4. 输出契约

- `src/ensemble/blend.py`、`E8/code/ensemble.py`
- `$V4_RUN_ROOT/E8/ensemble/oof.npz`
- `$V4_REPORTS_DIR/E8_ensemble_report.json`、`$V4_REPORTS_DIR/E8_gate.json`

## 5. 执行步骤

1. 枚举可用成员（多 seed 权重、不同主干的权重、快照 checkpoint）
2. 计算成员间 OOF 预测相关矩阵，标记同源簇
3. 实现三种融合：平均、加权（inner-OOF 选权）、线性 stacking（inner-OOF 训）
4. 与最佳单成员做 paired bootstrap（按井行数加权，1000 次）
5. 报告：集成 OOF、最佳单成员 OOF、delta、CI、逐折方向
6. 写 Gate：集成 ≥ 最佳单成员 且 CI 下界 > 0

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 成员数上限 | 5 | 3–5（内存受限时 2） | 总计划 §3.4.1 收缩规则 |
| 融合权重 | inner-OOF 选择 | 平均/加权/stacking | 禁止用 outer 选 |
| 同源判据 | 相关系数 > 0.99 视为同源 | 冻结 | 同源成员不计入增益证据 |

## 7. 完成判据

- 集成 ≥ 最佳单成员 且 CI 下界 > 0
- 成员同源性矩阵与同源簇标注完整
- 5 折 delta 方向一致（至少 4/5）
- `E8_gate.json` 全 mandatory 通过

## 8. 禁止事项

- 用同源模型平均制造假增益
- 用 outer 折选融合权重
- 成员数超过磁盘/内存可承受范围

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 集成无增益 | CI 含 0 | 判 NO-GO，保留最佳单成员作为最终候选 |
| stacking 过拟合 inner | inner 好 outer 差 | 限制 stacking 自由度（只用线性 + 强正则） |

## 10. 停止规则

- 集成 CI 上界 ≤ 0 → 保留最佳单成员，记录 NO-GO

## 11. 代码归属

- `src/ensemble/blend.py`
- `E8/code/ensemble.py`

## 12. 复算与证据

- `reports/E8_ensemble_report.json`、`reports/E8_gate.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E8
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

预注册文件：`v4/reports/E8_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E8_P2_gate",
  "stage": "E8",
  "p_stage": "P2",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "oof_total",
  "primary_threshold_key": "min_delta",
  "baseline_version": "E6_PD1",
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
