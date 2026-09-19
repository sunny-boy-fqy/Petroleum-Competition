# E1/P1 行级 MLP + 评分对齐损失 + 5 折 OOF（硬 Gate ≥ 78.0）

> 所属阶段：[E1](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：**分母建立阶段**：允许弱，必须正确　|　**依赖**：E1/P0

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

训练多任务 MLP（共享主干 + POR/PERM/SW 三头 + 联合占位头），用三段式对齐损失，跑完 80 井按井 5 折 OOF，产出逐行预测与官方口径评分；**硬 Gate：OOF Total ≥ 78.0、5 折同向、占位行逐目标 Acc ≥ 0.98**。

## 2. 为什么需要这一步

1. 在引入序列主干前必须知道"只看当前深度点"的上限，否则无法证明 E3 序列上下文的价值（`资料库/08` §0.3 第 1 层）；
2. 行级基线训练极快（分钟级），是验证损失实现、数据管线、OOF 流程是否正确的最高性价比手段；
3. v2 E4/P3 的 CPU MLP 是 NO-GO，但那是**逐点 + 无 GPU + 小容量**；E1 要给出"正确实现下的行级上限"作为 E3 的严格对照；
4. `资料库/12` §2.3 指出纯对齐损失早期信号稀疏，因此必须用三段式 + 用**真实评分**早停。

## 3. 输入契约

- E1/P0 的行级特征与标签
- `src/losses/score_aligned.py`
- E0 的评分器与折

## 4. 输出契约

- `models/E1/pd0_fold{k}.pt`（5 折权重，bf16 state_dict）
- `$V4_RUN_ROOT/E1/oof.npz`（well_id/depth/y_true/y_pred/q_ph，逐行）
- `$V4_REPORTS_DIR/E1_metrics.json`（逐折/逐目标/连续切片/bootstrap CI）
- `$V4_REPORTS_DIR/E1_loss_curve.csv`（每 epoch 训练/验证真实分数）
- `$V4_REPORTS_DIR/E1_gate.json`

## 5. 执行步骤

1. 预注册 `E1_P1_gate_prereg.json`（阈值、候选数、bootstrap 设置、mandatory checks）
2. 实现 `train_row.py`：`--resume`、`--time-budget-h`、每 epoch checkpoint、每 epoch 调 `assert_disk_headroom(8.0)`、写 `training_time_log.json`
3. 跑 fold0 小规模冒烟（`--max-wells 8 --epochs 2 --smoke`）确认链路与显存/内存
4. 全 5 折训练：bf16、AdamW、余弦退火、梯度裁剪 1.0；λ₁ 从 1.0 退火到 0.1
5. 每 epoch 在**验证折**上用真实 `score.py` 算分（早停依据，不用 loss 值）
6. 汇总 OOF → `score_arrays(..., missing_mode="drop")` → 逐目标 Acc 与 Total
7. 逐折 delta、逐井非退化比例、按井行数加权 paired cluster bootstrap（1000 次）
8. 评估占位行逐目标 Acc/precision/recall，写入 Gate 的 `atomic_precision_reported`
9. 写 `E1_gate.json` 并判定是否 ≥ 78.0

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `hidden` | 256 | 128/256/512 | 在 fold0+1 上选，选后冻结 |
| `layers` | 2 | 1/2/3 | 同上 |
| `dropout` | 0.1 | 0.0/0.1/0.2 | 同上 |
| `lr` | 2e-3 | 5e-4/1e-3/2e-3/5e-3 | AdamW，余弦退火到 1e-4 |
| `weight_decay` | 1e-4 | 0/1e-5/1e-4/1e-3 | 同上 |
| `batch_size` | 4096 | 2048/4096/8192 | 内存允许下尽量大（显存不是约束） |
| `epochs` | 40 | 20–80 | 结合早停（patience 5） |
| `λ1` 退火 | 1.0 → 0.1 | 线性，前 60% epoch | `资料库/12` §2.3 建议 |
| `λ2`（占位 BCE） | 0.2 | 0.1/0.2/0.3 | 同上 |
| `alpha`/`beta` | 1e-3 / 20 | 冻结 | Charbonnier / softplus 平滑参数 |
| `seed` | 42 | 42/1337 | 固定；多 seed 视为集成成员（E8） |

## 7. 完成判据

- **OOF Total ≥ 78.0**（硬 Gate，低于此值视为实现 bug，先排查不扩容）
- 5 折 delta **全部同向**（相对全常量基线）
- 占位行逐目标 Acc **≥ 0.98**，且 `atomic_precision_reported` 写入 Gate
- 加权配对井级 cluster bootstrap 95% CI 下界 > 0
- loss 曲线无 NaN；训练可 `--resume` 且 `disk_budget_ok`、`training_time_log_valid` 均为 true
- CONTIN 连续切片（排除占位行）逐目标 Acc 一并上报

## 8. 禁止事项

- 加入任何窗口/序列特征（属于 E2/E3）
- 用 outer 折或 A 榜选超参、阈值、早停点
- 用 loss 值替代真实评分做模型选择
- 为冲分而删除占位行或裁剪 SW

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 损失实现有误导致学不动 | loss 长时间不降或 Total < 70 | 先用 `--smoke` 在 5000 行上做过拟合测试（应能拟合到接近满分） |
| 标准化泄漏 | OOF 虚高、A 榜落差大 | 折内 fit 断言 + 单元测试 |
| 占位行学坏 | 占位 Acc < 0.98、Total 卡在 70 出头 | 提高 λ₂ 或对占位行过采样；检查 SW 尺度换算 |
| PERM 长尾崩塌 | PERM Acc < 0.85 | 确认在 log10 空间监督；检查 clip 范围 |
| 内存/磁盘被打爆 | 训练中途被杀 | `num_workers=4`、checkpoint 滚动淘汰、`disk_guard` |

## 10. 停止规则

- OOF < 78.0 时**禁止扩容**：先做"5000 行过拟合测试"与"折内一致性检查"
- 连续 2 次 NaN → 回退上一 checkpoint 并减半 lr
- 单折耗时超软预算 3 倍 → 减 epoch 或减宽度

## 11. 代码归属

- `E1/code/train_row.py`
- `E1/code/eval_oof.py`
- `src/models/row_mlp.py`
- `src/losses/score_aligned.py`
- `src/training/loop.py`

## 12. 复算与证据

- `reports/E1_metrics.json`、`reports/E1_gate.json`、`versions/candidates.json::E1_PD0`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E1
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E1_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E1_P1_gate",
  "stage": "E1",
  "p_stage": "P1",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "oof_total",
  "baseline_version": "CONST",
  "thresholds": {
    "min_delta": 7.5,
    "min_effect_floor": 0.0,
    "oof_total_min": 78.0
  },
  "pilot_std": null,
  "mde_units": 80,
  "min_detectable_effect": null,
  "multiplicity": "none",
  "mandatory_checks": [
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "checkpoint_resumable",
    "no_label_leak"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
