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
| `gate_type` | Gate 类型 | `delta` / `absolute` / `boolean` / `non_inferior`；**唯一事实源**是 `src/validation/gates.py::infer_gate_type`（生成器 `docs/gen_p_details.py` 直接 import 它，不再自己抄一份 boolean 指标清单） |
| `primary_metric` | 主判据 | `oof_total`（默认）/ `<target>_acc` / `atomic_f1` / `state_auc` / `confirm_non_inferiority`；`boolean` 类判据（`env_hard_checks_passed`、`a_board_no_breakdown`、`cpu_inference_ok`、`row_pipeline_ok`、`seq_pipeline_ok`、`seq_train_ok` 等）按 `infer_gate_type` 自动归为 `boolean` |
| `primary_threshold_key` | 主阈值键名 | 必须在 `thresholds` 中存在 |
| `baseline_version` / `baseline_artifact` / `baseline_manifest_sha256` | 比较对象 | 必须是**已冻结**的候选或 `PD0` 行级基线；禁止候选自比 |
| `thresholds.min_delta` | 主阈值 | 预注册写死；不得事后调整 |
| `thresholds.min_effect_floor` | 最小效应下限 | 即使 `allow_underpowered=true` 也不得跳过 |
| `thresholds.min_*` / `max_*` / `abs_*` | **绝对门槛**（可多条并存） | `min_*` 要求 `have >= want`；`max_*` 与 `abs_*` 要求 `have <= want`（`abs_*` 比较的是**绝对偏差**，不是分数）。方向清单与 `result` 字段映射的**唯一事实源**是 `src/validation/gates.py`（`MIN_ABSOLUTE_KEYS` / `MAX_ABSOLUTE_KEYS` / `METRIC_RESULT_FIELDS`），本模板不复制清单以免漂移；声明了阈值却拿不到对应 `result` 字段 → **判为失败**（不可复算的 Gate 不能算过）。**绝对门槛对所有 Gate 类型生效，包括 `boolean`** |
| `thresholds` 键名合法性 | 方向已知 | 任何不在上述两组清单、也不属于 `min_delta`/`min_effect_floor`/`non_inferiority_margin` 的 `min_*`/`max_*`/`abs_*` 键会被 `validate_prereg` **直接拒绝**（防止拼错键名被静默忽略） |
| `multiplicity` | 多重比较校正 | `none` 时 `candidate_budget` 必须为 1；否则 `holm`/`bonferroni`/`fdr_bh` |
| `pilot_std` / `mde_units` / `min_detectable_effect` | 功效 | `min_detectable_effect` 必须等于 `mde_two_sided(pilot_std, mde_units)`；`pilot_std` 未知时可写 `null` 并显式 `allow_underpowered=true` + 理由 |
| `mandatory_checks` | 强制检查 | 必须包含上表 6 项；缺项即 `validate_prereg` 失败 |
| `planned_task_training_h` | 单任务计划时长 | 仅作记录（D1 已放宽 100h 硬门禁），但必须 > 0 且写入时间日志 |
| `decisions_locked` | 本次实验期间冻结的决策 | 违反即作废该 Gate |

## 3. 判定逻辑

```
gate_type            = infer_gate_type(primary_metric, gate_type)   # 唯一事实源
effective_threshold = max(thresholds[primary_threshold_key], thresholds.min_effect_floor)
passed = (delta >= effective_threshold)
       AND (paired_bootstrap_ci_low > 0)          # 阶段 Gate；E9 用 ci_low > -margin
       AND all(mandatory_checks == true)
       AND all(绝对门槛 ok)                        # min_* 走 >=，max_*/abs_* 走 <=；取不到指标即判失败
```

- **绝对门槛的方向与取值（R3-H1）**：`min_*` 要求 `have >= want`，`max_*` / `abs_*` 要求 `have <= want`。
  旧实现把所有绝对键一律当**下界**，导致 `max_hard_failures` / `max_point_diff` / `max_minutes` /
  `max_memory_gb` / `max_degradation` / `abs_tolerance` 方向反转、并在 `boolean` Gate 下被整体忽略；
  现已修正为**按方向分组**且**对所有 Gate 类型统一强制**（例如 E10/P0 的 `max_minutes`/`max_memory_gb`、
  E10/P1 的 `max_point_diff` 现在真正生效）。
- **result 字段映射**：每个门槛键经 `METRIC_RESULT_FIELDS` 映射到 `result` 字段（如
  `min_auc → result["auc"]`、`min_atomic_acc → result["atomic_acc"]`、`max_minutes → result["minutes"]`、
  `max_memory_gb → result["memory_gb"]`、`max_point_diff → result["point_diff"]`、
  `max_degradation → result["degradation"]`、`min_atom_recall → result["atom_recall"]`；
  `abs_tolerance` 比较绝对偏差，映射到 `abs_diff`/`score_diff`/`abs_error`）。
  **声明了阈值但 result 未提供匹配字段 ⇒ 该项判为失败**（宁严勿松）。
- **键名合法性**：`min_*`/`max_*`/`abs_*` 中方向未知的键由 `validate_prereg` 报错（模板也会被查），
  因此键名拼错会**响亮失败**而不是被静默跳过。权威清单见 `src/validation/gates.py`。
- `allow_underpowered=true` 只跳过 MDE 兜底，**不跳过** `min_effect_floor`。
- 任何 mandatory check 失败 → `passed=false`，无论 delta 多高。
- E9 例外：`primary_metric=confirm_non_inferiority`，判定只认预注册 `non_inferiority_margin`。
- E6 追加的 mandatory：`per_target_atom_acc_reported`、`per_target_atom_precision_recall_f1_reported`、
  `joint_atom_auc_reported`、`tau_t_inner_oof_only`、`no_atom_continuous_interpolation`、`input_no_label_leak_full`。

## 4. 必须随 Gate 一并上报的内容

1. `ΔOOF Total` + **逐折 delta** + **逐井非退化比例**；
2. **按井行数加权**的配对井级 cluster bootstrap 95% CI（1000 次）；
3. 逐目标 Acc（POR/PERM/SW）与**连续切片** Acc（排除原子行后）；
4. **占位原子报告**：**逐目标** Acc、precision、recall、F1、`τ_t` 值（附 τ-总分曲线与最宽平台）、
   误判代价分解，以及 **joint atom Acc/AUC/AP**；
5. **逐目标 continuous slice Acc** 与总分分解（**禁止只报 Overall Total**）；
6. `label_scale_summary`（POR/SW/PERM 的分布与量纲声明，**必须显式声明 SW 为单一标签尺度（百分数，实测 8.305–99.9，`SW<1` 为 0 行，双尺度假设已在 E0-R2 证伪、`SW_SMALL_BRANCH` 永久关闭）且未做 [0,1] 归一化**）；
7. `inner_only_selection` 证据：所有超参/阈值/`τ_t`/解码/融合权重的选择只用了 inner-OOF；
8. 集成类 Gate：**成员相关性/共同来源**矩阵与同源簇标注（同源平均不得计为增益；增益 CI 含 0 即 NO-GO）；
9. 特征来源表变更（若有）；
10. `training_time_log.json` 的本任务条目与 `disk_free_gb_end`；
11. 所有产物的 sha256 与复算命令。

**禁止**（出现在报告中即 Gate 作废）：在原子与连续分支之间做插值；对 SW 做全局 `[0,1]` 裁剪
（只允许 `[0,100]` 软裁剪）；在 inner-OOF 之外选 τ/超参/解码参数；把同源平均或 CI 含 0 的差异当作"增益"；
删除占位行；在验证折/测试井上 fit scaler。

> E0 本地契约 Gate 现为 **12/12（含 cache）** mandatory：原 10 项 + `shard_cache_built` + `shard_cache_input_cols_ok`；
> 其 `disk_budget_ok` 读取 **DATA-ROOT（`$V4_DATA_ROOT` = `/data`）** 级别。

## 5. Gate 目录约定

| 文件 | 说明 |
|---|---|
| `reports/<GATE_ID>_gate_prereg.json` | 预注册（实验前） |
| `reports/<GATE_ID>_gate.json` | 判定结果（实验后） |
| `reports/<GATE_ID>_metrics.json` | 原始指标与 bootstrap 明细 |
| `experiments/<E>/<P>/<candidate_id>/` | `result.json` / `result.zip` / `cv.json` / `manifest.json` |
