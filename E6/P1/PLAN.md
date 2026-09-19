# E6/P1 逐目标原子门 τ_t 搜索（inner-OOF 官方总分）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：开关设定：决定"输出常量还是连续"，按官方总分优化　|　**依赖**：E6/P0
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

在 inner-OOF 上**逐目标**搜索门限 `τ_t`，目标函数是**官方总分**`τ_t* = argmax_τ 100·w_t·Acc_t(τ)`（不是 F1/准确率的点估计），取最宽平台中点实现**硬切换**解码，并报告误判代价分解。

## 2. 为什么需要这一步

1. τ 决定每个点是走常量分支还是连续分支，是纯 DL 管线唯一保护屏障的开关；
2. 误判代价不对称：把有效行判成常量会立刻丢分，把占位行判成连续同样丢分，两者代价需分别量化；
3. **禁止在常量与连续之间线性插值**：POR 容差仅 ±0.008，插值必然出带；
4. 改进 proposal §2 A4/§7 F1：Total 是三目标加权和、硬切换逐目标独立，因此 τ_t 可逐目标搜；目标函数必须是官方总分（含权重 w_t），并且要取**最宽平台的中点**而非 argmax 尖峰，否则 inner 过拟合；`joint_guard` 只是可选门禁，默认关闭。

## 3. 输入契约

- E6/P0 的 `q_por/q_perm/q_sw/q_joint`（inner-OOF）
- E3–E5 的连续预测（inner-OOF）
- 官方评分权重 `w = (0.30, 0.35, 0.35)`

## 4. 输出契约

- `src/inference/atomic_gate.py`（逐目标 `τ_t` 硬切换 + 可选 `joint_guard`）
- `$V4_REPORTS_DIR/E6_tau_search.json`（τ-总分曲线、平台、逐目标 atomic P/R/F1/acc、连续切片 acc、误判代价分解）
- `三个 τ 值与 `joint_guard` 开关写入 `versions/candidates.json::PD1.atomic` 与 manifest`

## 5. 执行步骤

1. 对每个目标 t，在 inner-OOF 上网格搜索 τ∈[0.05,0.95]（步长 0.01），网格与容差**只从 `src/inference/atomic_gate.py::default_tau_grid()` / `DEFAULT_PLATEAU_TOL` 取**（禁止在本脚本或计划里另抄一份数字）
2. 目标函数 `100·w_t·Acc_t(τ)`（官方 drop 口径）—— 调 `select_tau_per_target` 时 **不传 `score_fn`**（默认即 `official_score_fns()`，直接包装 `src/score.py` 的 `acc_relative`/`acc_perm`）；**禁止**自写 0/1 容差准确率冒充官方目标（R4-M4）
3. 记录**最宽平台**（连续满足 `score ≥ max−ε` 的区间）并取其中点，而不是 argmax 尖峰
4. 报告：`τ_t` vs inner-OOF Total 曲线、平台区间、逐目标 atomic precision/recall/F1/acc、连续切片 Acc、误判代价分解（有效判 atom vs atom 判连续）
5. 评估可选 `joint_guard`：仅当 `q_joint > τ_joint_high` 时三目标全输出 atom；默认关闭，只有它在 inner-OOF 上提升 Total 且 CI 下界 > 0 才启用，并把决定写入 manifest
6. 验证 τ 稳定性：不同 inner 折选出的 τ 是否接近（方差过大则取更保守值并说明）
7. 在三目标上分别确定 τ 并记录到候选注册表；**断言原子/连续之间无任何插值**

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| τ 搜索范围 | [0.05, 0.95]，步长 0.01 | 冻结 | 逐目标独立 |
| 目标函数 | `100·w_t·Acc_t(τ)`（官方总分） | 冻结 | **不是 F1/准确率** |
| 平台选择 | 最宽平台中点（`max−ε`） | ε=1e-3 | 避免 argmax 尖峰过拟合 inner |
| 硬切换 | q > τ → 输出精确常量（严格大于） | 冻结 | 禁止插值；符号与 atomic_gate.py 一致 |
| `joint_guard` | **关闭（默认）** | 开/关 | 仅当 inner-OOF Total 提升且 CI 下界 > 0 才开 |
| `τ_joint_high` | — | inner 搜索 | 仅 joint_guard 启用时使用 |
| 稳定性判据 | 不同 inner 折最优 τ 的极差 ≤ 0.2 | 冻结 | 超限则用更保守 τ |

## 7. 完成判据

- 三个 τ 都只在 inner-OOF 上按**官方总分**选出，过程可复算；`tau_t_inner_oof_only` 为 true
- 报告 τ-总分曲线与**最宽平台**，所选 τ 落在平台中点（附平台宽度）
- 逐目标 atomic precision/recall/F1/acc 与连续切片 Acc 全部上报
- 误判代价分解表完整
- τ 稳定性通过（跨 inner 折极差 ≤ 0.2），否则取更保守值并说明
- 占位行逐目标 Acc ≥ 0.99；`oof_total_min ≥ 81.0`
- `joint_guard` 默认关闭；若启用，必须有 inner-OOF Total 提升 + CI 下界 > 0 的证据并写入 manifest
- `no_atom_continuous_interpolation` 为 true（代码级断言，无插值/混合尺度算术）

## 8. 禁止事项

- 用 outer 折或 A 榜选 τ
- 在常量与连续输出之间做线性插值或混合尺度算术
- 用 F1 而非官方总分作为 τ 的目标函数
- 取 argmax 尖峰而不看平台
- 默认启用 `joint_guard`

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| τ 过拟合 inner | inner 最优但 outer 变差 | 报告 τ 敏感性曲线；取最宽平台中点 |
| joint_guard 误伤 | 非 joint 行三目标全被覆盖 | 默认关闭；仅 inner-OOF 正增益 + CI 下界 > 0 才启用 |
| 误判代价不对称被忽视 | 总分下降但 F1 上升 | 以官方总分为目标函数，报告代价分解 |

## 10. 停止规则

- τ 搜索若无法让占位 Acc ≥ 0.99 → 回到 E6/P0 加强逐目标原子头

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
  "gate_type": "delta",
  "primary_metric": "atomic_f1",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "min_atomic_acc": 0.99,
    "min_atom_acc": 0.99,
    "min_atom_recall": 0.98,
    "oof_total_min": 81.0
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
    "per_target_atom_acc_reported",
    "per_target_atom_precision_recall_f1_reported",
    "joint_atom_auc_reported",
    "tau_t_inner_oof_only",
    "no_atom_continuous_interpolation",
    "input_no_label_leak_full"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
