# v4 Gate 预注册模板

> 所有 P 级 Gate 必须**在实验前**产出 `v4/reports/<GATE_ID>_gate_prereg.json`，字段按本模板。预注册后不得修改阈值，只能新建修订号（`<GATE_ID>_r2`）。
>
> 校验器：`v4/src/validation/gates.py`（`validate_prereg` / `effective_threshold` / `aggregate_gate`）。

## 1. 必填字段

```json
{
  "gate_id": "E3_P2_gate",
  "stage": "E3",
  "p_stage": "P2",
  "created_at": "YYYY-MM-DDTHH:MM:SS",
  "primary_metric": "oof_total",
  "primary_threshold_key": "min_delta",
  "baseline_version": "E1_PD0",
  "baseline_artifact": "experiments/E1/P1/pd0/oof.npz",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
  "alpha": 0.05,
  "multiplicity": "none",
  "candidate_budget": 1,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "pilot_std": null,
  "mde_units": 80,
  "min_detectable_effect": null,
  "mandatory_checks": [
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "checkpoint_resumable",
    "no_label_leak"
  ],
  "decisions_locked": ["architecture", "loss_weights", "feature_version"],
  "planned_task_training_h": 20.0,
  "notes": ""
}
```

## 2. 字段语义

| 字段 | 含义 | 规则 |
|---|---|---|
| `primary_metric` | 主判据 | `oof_total`（默认）/ `<target>_acc` / `atomic_f1` / `confirm_non_inferiority` |
| `primary_threshold_key` | 主阈值键名 | 必须在 `thresholds` 中存在 |
| `baseline_version` / `baseline_artifact` / `baseline_manifest_sha256` | 比较对象 | 必须是**已冻结**的候选或 `PD0` 行级基线；禁止候选自比 |
| `thresholds.min_delta` | 主阈值 | 预注册写死；不得事后调整 |
| `thresholds.min_effect_floor` | 最小效应下限 | 即使 `allow_underpowered=true` 也不得跳过 |
| `multiplicity` | 多重比较校正 | `none` 时 `candidate_budget` 必须为 1；否则 `holm`/`bonferroni`/`fdr_bh` |
| `pilot_std` / `mde_units` / `min_detectable_effect` | 功效 | `min_detectable_effect` 必须等于 `mde_two_sided(pilot_std, mde_units)`；`pilot_std` 未知时可写 `null` 并显式 `allow_underpowered=true` + 理由 |
| `mandatory_checks` | 强制检查 | 必须包含上表 6 项；缺项即 `validate_prereg` 失败 |
| `planned_task_training_h` | 单任务计划时长 | 仅作记录（D1 已放宽 100h 硬门禁），但必须 > 0 且写入时间日志 |
| `decisions_locked` | 本次实验期间冻结的决策 | 违反即作废该 Gate |

## 3. 判定逻辑

```
effective_threshold = max(thresholds[primary_threshold_key], thresholds.min_effect_floor)
passed = (delta >= effective_threshold)
       AND (paired_bootstrap_ci_low > 0)          # 阶段 Gate；E9 用 ci_low > -margin
       AND all(mandatory_checks == true)
```

- `allow_underpowered=true` 只跳过 MDE 兜底，**不跳过** `min_effect_floor`。
- 任何 mandatory check 失败 → `passed=false`，无论 delta 多高。
- E9 例外：`primary_metric=confirm_non_inferiority`，判定只认预注册 `non_inferiority_margin`。

## 4. 必须随 Gate 一并上报的内容

1. `ΔOOF Total` + **逐折 delta** + **逐井非退化比例**；
2. **按井行数加权**的配对井级 cluster bootstrap 95% CI（1000 次）；
3. 逐目标 Acc（POR/PERM/SW）与**连续切片** Acc（排除原子行后）；
4. **占位原子报告**：逐目标 Acc、precision、recall、F1、τ 值、误判代价分解；
5. `label_scale_summary`（POR/SW/PERM 的分布与量纲声明，**必须显式声明 SW 为单一标签尺度（百分数，实测 8.305–99.9）且未做 [0,1] 归一化**）；
6. 特征来源表变更（若有）；
7. `training_time_log.json` 的本任务条目与 `disk_free_gb_end`；
8. 所有产物的 sha256 与复算命令。

## 5. Gate 目录约定

| 文件 | 说明 |
|---|---|
| `reports/<GATE_ID>_gate_prereg.json` | 预注册（实验前） |
| `reports/<GATE_ID>_gate.json` | 判定结果（实验后） |
| `reports/<GATE_ID>_metrics.json` | 原始指标与 bootstrap 明细 |
| `experiments/<E>/<P>/<candidate_id>/` | `result.json` / `result.zip` / `cv.json` / `manifest.json` |
