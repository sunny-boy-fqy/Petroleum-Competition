# E6/P0 逐目标原子头 + 辅助 joint 头（两阶段训练）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：保护屏障：66.7% 的白送分 + 单目标原子行都靠逐目标原子头守住　|　**依赖**：E5/P2（三个连续头定型）
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

训练**逐目标原子头** `q_por/q_perm/q_sw`（主保护）+ **辅助** `q_joint`（可选高置信硬门禁，默认关），并做**两阶段训练**：stage 1 训练主干 + 原子头（`L_atom + λ_joint·L_joint` + 极小连续 fallback 0.05）；stage 2 冻结原子头（或 `q_head_lr_mult=0.05–0.1`）训练连续头（按切片加权，**权重永不为 0**）。评估逐目标 AUC / 原子 Acc/Precision/Recall/F1 与 joint atom AUC。

## 2. 为什么需要这一步

1. 487,225 行（66.719%）是联合常量占位，占约 66.72 分的白送分；
2. v1 的原子门已被证明可从输入预测（E7 的 +0.2015 主要来自此），因此应把"是否输出常量"做成**显式可学习决策**，而不是让回归头勉强逼近；
3. 本项目不做 B0 patch 隔离（用户决策 D3），原子头是**唯一**的占位保护屏障，因此它的质量直接决定管线是否安全；
4. 改进 proposal §1/§2：**单个 joint 头不够**——SW 单目标原子 31,030 行、PERM 7,373 行、POR 157 行并非 joint；joint 头对它们漏保护，又会误伤 joint 行中的非原子目标，且无法满足按目标验收的 Gate；
5. 改进 proposal §5 D1：两步训练能避免连续损失把刚学好的原子边界冲掉；非 joint 原子行加权比整行过采样更精确，避免不同目标互相干扰。

## 3. 输入契约

- E3/E4 主干逐行表示或 E1 行级特征
- E0 的占位标签与三目标 mask

## 4. 输出契约

- `src/models/state_head.py`（`q_joint + q_por/q_perm/q_sw` 五个头）、`$V4_RUN_ROOT/E6/state/{foldk}.pt`（含 stage 1/2 元数据）
- `$V4_REPORTS_DIR/E6_atomic_report.json`（逐目标 AUC / Acc / Precision / Recall / F1、joint atom AUC、两阶段曲线）

## 5. 执行步骤

1. 实现原子头：`q_t = sigmoid(Linear(d→1))`（`t∈{por,perm,sw}`）+ `q_joint = sigmoid(Linear(d→1))`；标签 `y_atom[:,t] = (y[:,t]==占位值[t]) & ~missing[:,t]`、`y_joint = y_atom.all(1) & ~missing.any(1)`
2. 实现 **stage 1**：训练主干 + 原子头，损失 `L_atom + λ_joint·L_joint`（外加极小连续 fallback 0.05）；监控 per-target atomic Acc/Precision/Recall
3. 实现 per-target 非 joint 原子行加权：`w_t = 1 + α·y_atom[:,t]·(¬y_joint)`，`L_atom_t = Σ(BCE(q_t,y_atom_t)·w_t·mask_t)/Σ(w_t·mask_t)`
4. 实现 **stage 2**：冻结原子头（或 `q_head_lr_mult=0.05–0.1`），训练连续头；按切片加权：joint 行 0.1–0.3、非 joint 原子行 0.1–0.3（作为 fallback）、有效连续行 1.0，**永不置 0**
5. 总损失：`L = L_cont + λ_joint·L_joint + λ_atom·L_atom`，默认 `λ_atom=0.5`、`λ_joint=0.2`，per-target `pos_weight` 1.0（搜 1.0/1.5/2.0），非 joint 原子行权重 `α` 1.0（搜 1.0/2.0/3.0）
6. 用 BCE 训练（缺测行不参与）；报告 AUC、PR-AUC 与 joint atom AUC/AP
7. 报告**逐目标**原子 Acc/Precision/Recall/F1（τ=0.5 与最优 τ 两处）
8. 做 label-shuffle 阴性对照 + 全量输入泄漏回归（`input_no_label_leak_full`），确认指标不是来自泄漏

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 头结构 | `q_joint + q_por/q_perm/q_sw` | 冻结 | 逐目标原子是主保护 |
| `λ_atom` | 0.5 | 0.2/0.5/1.0 | inner 选择 |
| `λ_joint` | 0.2 | 0.1/0.2/0.5 | inner 选择 |
| per-target `pos_weight` | 1.0 | 1.0/1.5/2.0 | inner 选择 |
| 非 joint 原子行权重 `α` | 1.0 | 1.0/2.0/3.0 | inner 选择 |
| `q_head_lr_mult` | 0.05–0.1（stage 2） | 0（冻结）/0.05/0.1 | inner 选择 |
| stage 2 切片权重 | joint 0.1–0.3 / 非 joint 原子 0.1–0.3 / 有效 1.0 | 冻结 | **永不为 0** |
| 正负样本 | 全量（占位 66.7%） | 全量/过采样有效 | 过采样需消融 |

## 7. 完成判据

- 逐目标原子 Acc **≥ 0.99**、recall **≥ 0.98**，precision/F1 全部上报；`state_auc ≥ 0.97` 且 PR-AUC 报告完整
- **joint atom AUC/AP** 单独上报（`joint_atom_auc_reported`）
- label-shuffle 对照下 AUC ≈ 0.5（证明非泄漏）；`input_no_label_leak_full` 为 true
- 两阶段训练记录完整：stage 2 后原子 Acc 不下降（冻结或低 lr 生效）
- 连续头切片权重非 0 且写入配置；`pos_weight`/`α`/`λ_atom`/`λ_joint` 均只在 inner-OOF 选

## 8. 禁止事项

- 用测试集或验证折标签训练原子头
- 把原子头当作"裁剪器"直接覆盖回归输出而不经 τ 判定
- 用单个 joint 头覆盖三目标（必须逐目标原子头）
- 把连续头切片权重设为 0（原子误判时连续头必须能 fallback）
- 在 stage 2 让原子头以全 lr 继续更新（会冲掉原子边界）

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 逐目标原子头互相干扰 | 某目标 recall 上升、另两个下降 | 独立 loss 权重 + per-target sample weight；必要时先不共享原子头 |
| 原子头学不到占位 | AUC < 0.9 | 检查特征是否包含足够区分信息；加井级池化 |
| 原子头过拟合 | inner AUC 高 outer 低 | 减容量 + dropout + 折内早停 |
| 两阶段第二段遗忘原子头 | stage 2 后 atom Acc 下降 | 冻结或极低 lr；inner-OOF 监控 atom Acc；必要时联合微调 |

## 10. 停止规则

- AUC < 0.9 且无改善 → 记录 NO-GO，改用固定常量策略并重新评估总分上限

## 11. 代码归属

- `src/models/state_head.py`
- `E6/code/train_state.py`

## 12. 复算与证据

- `reports/E6_atomic_report.json`

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

预注册文件：`v4/reports/E6_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E6_P0_gate",
  "stage": "E6",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "state_auc",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "min_auc": 0.97,
    "min_atom_acc": 0.99,
    "min_atom_recall": 0.98
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
