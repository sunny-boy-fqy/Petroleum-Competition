# E6/P2 PD1 完整管线组装与硬 Gate（≥82.0）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：**完整管线诞生**：本计划第一个可提交候选　|　**依赖**：E6/P0–P1、E5/P2
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

组装 数据→主干→**逐目标原子头 + 辅助 joint 头**→逐目标硬切换→连续后处理→契约 的完整 PD1 管线，产出 5 折 OOF、测试集 `result.json`/`result.zip`、manifest 与 cv 报告；**硬 Gate：OOF Total ≥ 82.0**；并上报 proposal §7 F5 要求的全部逐目标指标。

## 2. 为什么需要这一步

1. 这是 v4 第一个"端到端可跑、可提交、可复现"的候选；
2. ≥82.0 意味着超过历史锚点 B0 的本地 OOF 80.382479，是纯 DL 路线成立的最低证据；
3. 只有完整管线才能暴露"训练能跑但推理契约不过"这类问题（前代多次踩坑）；
4. 改进 proposal §7 F5/§11：Gate 必须能复算**逐目标** atomic Acc/P/R/F1、τ_t、连续切片 Acc、joint atom AUC/AP、总分分解、误判代价矩阵与 `inner_only_selection` 证据。

## 3. 输入契约

- E3/E4 冻结主干权重
- E5 的三个连续头
- E6 的逐目标原子头 `q_por/q_perm/q_sw`、`q_joint` 与 `τ_t`
- E0 的契约与评分器

## 4. 输出契约

- `models/E6/pd1_fold{k}.pt` + `models/E6/pd1_config.json`
- `experiments/E6/P2/pd1/{oof.npz,cv.json,result.json,result.zip,manifest.json}`
- `$V4_REPORTS_DIR/E6_gate.json`、`$V4_REPORTS_DIR/E6_atomic_report.json`（逐目标 + joint + τ 曲线）
- `versions/candidates.json::PD1`（status=local_only→shortlisted）

## 5. 执行步骤

1. 实现统一推理器 `src/inference/predictor.py`：加载配置与权重 → 逐井前向 → 逐目标硬切换 → 连续后处理 → 解码
2. 在 5 折上各自推理出 OOF（训练时已产出，此处复核逐行对齐）
3. 对 10 口测试井推理：平均 5 折权重（或按核验过的最优折），产出 result.json
4. 跑契约校验（10 井 / 95,948 行 / depth 对齐 / PERM>0 / 无 NaN / SW 单尺度 `[0,100]` 守卫）
5. 本机 CPU 冒烟 `predict.py --use-version PD1 --data_dir ../data --output /tmp/r.json`
6. 汇总 OOF 评分并上报 proposal §7 F5 全部条目：逐目标 `atomic_acc/precision/recall/F1`、逐目标 `τ_t`、逐目标 continuous slice Acc、joint atom Acc/AUC/AP、总分分解与阈值曲线、误判代价矩阵、`inner_only_selection` 证据
7. 写 manifest（config 哈希/数据指纹/折指纹/代码哈希 + `joint_guard` 开关决定）并注册候选
8. 写 Gate 并判定 ≥ 82.0

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 折权重聚合 | 5 折平均 | 平均/最优折/加权 | inner 决定，冻结后不改 |
| `τ_t` | E6/P1 选定值（官方总分平台中点） | 冻结 | 写入 candidate registry |
| `joint_guard` | E6/P1 决定（默认关） | 冻结 | 写入 manifest |
| 推理精度 | fp32（CPU） | fp32/fp16 | 提交侧必须 fp32 保证确定性 |
| `num_folds` | 5 | 冻结 | 与 folds.json 一致 |

## 7. 完成判据

- OOF Total **≥ 82.0**（硬 Gate），且 `min_atom_acc ≥ 0.99`、`min_atom_recall ≥ 0.98`、`min_joint_atom_auc ≥ 0.90`
- 契约全绿；`predict.py --use-version PD1` 在本机 CPU 可跑通并输出 95,948 行
- 逐目标原子 Acc/P/R/F1、连续切片 Acc、joint atom AUC/AP、τ_t 与误判代价矩阵全部上报
- `tau_t_inner_oof_only`、`no_atom_continuous_interpolation`、`input_no_label_leak_full` 均为 true
- manifest 写全 config/data/folds/code 四类指纹与 `joint_guard` 决定；候选已注册
- 5 折 delta 全部同向；`disk_budget_ok`、`training_time_log_valid`、`checkpoint_resumable` 为 true

## 8. 禁止事项

- 在管线中混入未冻结的特征版本
- 推理阶段读取任何标签
- 把 5 折权重聚合方式在看到 OOF 后临时更换
- 在原子与连续之间插值，或对 SW 做全局 `[0,1]` 裁剪
- 只报 Overall Total 而省略逐目标/逐切片指标

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 训练能跑但推理契约不过 | result.json 行数/字段错 | 契约前置到训练脚本每次落盘时校验 |
| 低于 82.0 | 纯 DL 未超过树模型锚点 | 按总计划 §9.5 回退协议准备 B0 fallback；同时保留 PD1 为 `local_only` 候选供 E7/E8 继续改进 |
| 逐目标指标被总分掩盖 | 总分达标但某目标原子 recall 低 | Gate 强制逐目标阈值 + 误判代价矩阵 |
| 折间差异大 | 逐折 delta 方向不一致 | 检查折内标准化与早停；报告逐折而非只报总分 |

## 10. 停止规则

- Gate < 82.0 → 冻结当前最强候选为 `PD-pre`，E7/E8 继续改进；若 E8 结束仍 < 82.0，E10 走回退协议

## 11. 代码归属

- `E6/code/build_pd1.py`
- `E6/code/gate.py`
- `src/inference/predictor.py`
- `src/inference/atomic_gate.py`

## 12. 复算与证据

- `reports/E6_gate.json`、`reports/E6_atomic_report.json`、`reports/E6_tau_search.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E6
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

预注册文件：`v4/reports/E6_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E6_P2_gate",
  "stage": "E6",
  "p_stage": "P2",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "oof_total",
  "primary_threshold_key": "min_delta",
  "baseline_version": "E1_PD0",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "oof_total_min": 82.0,
    "min_atom_acc": 0.99,
    "min_atom_recall": 0.98,
    "min_joint_atom_auc": 0.9
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
    "input_no_label_leak_full",
    "cpu_inference_ok"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
