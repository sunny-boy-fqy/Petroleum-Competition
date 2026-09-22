# E1/P1 行级 MLP + 评分对齐损失 + 5 折 OOF（硬 Gate ≥ 78.0）

> 所属阶段：[E1](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：**分母建立阶段**：允许弱，必须正确　|　**依赖**：E1/P0
>
> **状态**：▶ 进行中　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

训练多任务 MLP（共享主干 + POR/PERM/SW 连续头 + 逐目标原子头 `q_por/q_perm/q_sw` + 辅助 `q_joint`），用三段式对齐损失（**含训练折尺度归一化的 `L_aux`**），跑完 80 井按井 5 折 OOF，产出逐行预测与官方口径评分；**硬 Gate：OOF Total ≥ 78.0、5 折同向、占位行逐目标 Acc ≥ 0.98**。

## 2. 为什么需要这一步

1. 在引入序列主干前必须知道"只看当前深度点"的上限，否则无法证明 E3 序列上下文的价值（`资料库/08` §0.3 第 1 层）；
2. 行级基线训练极快（分钟级），是验证损失实现、数据管线、OOF 流程是否正确的最高性价比手段；
3. v2 E4/P3 的 CPU MLP 是 NO-GO，但那是**逐点 + 无 GPU + 小容量**；E1 要给出"正确实现下的行级上限"作为 E3 的严格对照；
4. `资料库/12` §2.3 指出纯对齐损失早期信号稀疏，因此必须用三段式 + 用**真实评分**早停；
5. 改进 proposal §3/§4：连续头的**参数化**（POR 可到 0、SW 训练折仿射归一化反变换）与损失修正（`L_aux` 按目标尺度归一化、`masked_mean` 防 NaN、PERM 官方 log 截断）必须在本层定稿，否则 E3/E4 只是在更快地优化错误目标。

## 3. 输入契约

- E1/P0 的行级特征与标签
- `src/losses/score_aligned.py`
- E0 的评分器与折

## 4. 输出契约

- `models/E1/pd0_fold{k}.pt`（5 折权重，bf16 state_dict；含 `q_por/q_perm/q_sw/q_joint` 头）
- `$TENSORBOARD_LOGDIR/E1_pd0/`（平台可见的迭代曲线；由 `src/training/tb_logger.py::RunLogger` 写 TensorBoard + JSONL）
- `$V4_RUN_ROOT/E1/oof.npz`（well_id/depth/y_true/y_pred/q_atom[3]/q_joint，逐行）
- `$V4_REPORTS_DIR/E1_metrics.json`（逐折/逐目标/连续切片/bootstrap CI）
- `$V4_REPORTS_DIR/E1_loss_curve.csv`（每 epoch 训练/验证真实分数）
- `$V4_REPORTS_DIR/E1_gate.json`
- `连续头尺度参数（`s_por`/`s_sw`/`sw_mu`/`sw_sigma`/`por_max`）随 checkpoint manifest 落盘`

## 5. 执行步骤

1. 预注册 `E1_P1_gate_prereg.json`（阈值、候选数、bootstrap 设置、mandatory checks）
2. 实现 `train_row.py`：`--resume`、`--time-budget-h`、每 epoch checkpoint、每 epoch 调 `assert_disk_headroom(8.0)`、写 `training_time_log.json`
3. 实现连续头参数化：POR 用 `por_cont = por_max·sigmoid(g)`（备选 `softplus(g)−softplus(g0)`，**禁止 `0.1+softplus(g)`**）；SW 用 `sw_cont = sw_mu + sw_sigma·head_out`（`head_out` 在归一化空间训练）；PERM 用 `zhat = 6·tanh(g)`，初始化自 `perm_z_median ≈ −0.08`
4. 实现 `L_aux = 0.30·SmoothL1((por_hat−y)/s_por) + 0.35·SmoothL1(zhat−z) + 0.35·SmoothL1((sw_hat−y)/s_sw)`，`s_por`/`s_sw` 只取训练折有效标签的稳健尺度并写入 manifest
5. 修正 `masked_mean`：先 `x = where(m>0, x, 0)` 再 `(x·m).sum()/m.sum().clamp_min(1.0)`，并加 NaN 单测（`x=(nan,2.0)`、`m=(0,1)` → 2.0）
6. PERM 对齐项在 log 空间显式实现 `d = clamp_min(zhat−z, log10(eps))`，与官方 `max(ŷ/y,ε)` 对齐
7. 每 epoch 用 `RunLogger` 写 TensorBoard（`TENSORBOARD_LOGDIR`，持久在 /data）：`loss/align|aux|ph`、`score/oof_total`、`score/acc_{por,perm,sw}`、`atomic/*`、`lr`
8. 跑 fold0 小规模冒烟（`--max-wells 8 --epochs 2 --smoke`）确认链路与显存/内存
9. 全 5 折训练：bf16、AdamW、余弦退火、梯度裁剪 1.0；λ₁ 从 1.0 退火到 0.1
10. 每 epoch 在**该 outer 折的 inner-OOF** 上用真实 `score.py` 算分（早停依据，不用 loss 值；outer 验证折只在最后推理一次）
11. 汇总 OOF → `score_arrays(..., missing_mode="drop")` → 逐目标 Acc 与 Total
12. 逐折 delta、逐井非退化比例、按井行数加权 paired cluster bootstrap（1000 次）
13. 评估占位行逐目标 Acc/precision/recall，写入 Gate 的 `atomic_precision_reported`；并单独上报 `POR=0` 与 `POR<0.1` 切片的 Acc（单测覆盖该切片构造）
14. 写 `E1_gate.json` 并判定是否 ≥ 78.0

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `hidden` | 256 | 128/256/512 | 在该 outer 折的 inner-OOF 上选，选后冻结（fold0 仅作资源预检） |
| `layers` | 2 | 1/2/3 | 同 hidden，inner-OOF 选择 |
| `dropout` | 0.1 | 0.0/0.1/0.2 | inner-OOF 选择 |
| `lr` | 2e-3 | 5e-4/1e-3/2e-3/5e-3 | inner-OOF 选择；AdamW，余弦退火到 1e-4 |
| `weight_decay` | 1e-4 | 0/1e-5/1e-4/1e-3 | 同上 |
| `batch_size` | 4096 | 2048/4096/8192 | 内存允许下尽量大（显存不是约束） |
| `epochs` | 40 | 20–80 | 结合早停（patience 5） |
| `λ1` 退火 | 1.0 → 0.1 | 线性，前 60% epoch | `资料库/12` §2.3 建议 |
| `λ2`（原子 BCE） | 0.2 | 0.1/0.2/0.3 | 同上 |
| POR 参数化 | `por_max·sigmoid(g)` | sigmoid / softplus 偏移 | inner 选择；**禁用 `0.1+softplus`** |
| SW 参数化 | 训练折仿射归一化 + 反变换 | 冻结 | `sw_mu=median`，`sw_sigma=IQR/1.349` |
| 边界聚焦权重 | **关闭（默认）** | 仅 E7 可选消融 | 不属于 E1 默认配方 |
| `alpha`/`beta` | 1e-3 / 20 | 冻结 | Charbonnier / softplus 平滑参数 |
| `seed` | 42 | 42/1337 | 固定；多 seed 视为集成成员（E8） |

## 7. 完成判据

- **OOF Total ≥ 78.0**（硬 Gate，低于此值视为实现 bug，先排查不扩容）
- 5 折 delta **全部同向**（相对全常量基线）
- 占位行逐目标 Acc **≥ 0.98**，且 `atomic_precision_reported` 写入 Gate
- 加权配对井级 cluster bootstrap 95% CI 下界 > 0
- loss 曲线无 NaN；训练可 `--resume` 且 `disk_budget_ok`、`training_time_log_valid` 均为 true
- CONTIN 连续切片（排除占位行）逐目标 Acc 一并上报
- **连续头参数化验证**：POR 能输出 ~0（`POR=0`/`POR<0.1` 切片单测通过），SW 训练折仿射归一化（`sw_cont = sw_mu + sw_sigma·head_out`）可反变换且无 ×100 换算
- **损失修正验证**：`L_aux` 使用训练折 `s_por`/`s_sw` 且写入 manifest；`masked_mean` NaN 单测通过；PERM 对齐项使用 `clamp_min(zhat−z, log10(eps))`
- 边界聚焦权重保持**默认关闭**（只作为 E7 的可选消融项）

## 8. 禁止事项

- 加入任何窗口/序列特征（属于 E2/E3）
- 用 outer 折或 A 榜选超参、阈值、早停点
- 用 loss 值替代真实评分做模型选择
- 为冲分而删除占位行或裁剪 SW
- POR 连续头使用 `0.1 + softplus(g)`（锁死下界，无法输出实测 0.0）
- 把 SW 当 `[0,1]` 尺度或对连续头做 ×100 换算
- 在整表上 fit `s_por`/`s_sw`/`sw_mu`/`sw_sigma`（只能训练折 fit）
- 默认开启边界聚焦权重

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 损失实现有误导致学不动 | loss 长时间不降或 Total < 70 | 先用 `--smoke` 在 5000 行上做过拟合测试（应能拟合到接近满分） |
| 标准化泄漏 | OOF 虚高、A 榜落差大 | 折内 fit 断言 + 单元测试 |
| 占位行学坏 | 占位 Acc < 0.98、Total 卡在 70 出头 | 提高 λ₂ 或对占位行过采样；检查 SW 是否被误做尺度换算 |
| PERM 长尾崩塌 | PERM Acc < 0.85 | 确认在 log10 空间监督；检查 `clamp_min(zhat−z, log10(eps))` 与 clip 范围 |
| POR 连续头锁死在下界 | `POR=0`/`POR<0.1` 切片 Acc 恒为 0 | 换用 `por_max·sigmoid(g)`；确认已删除 `0.1+softplus` |
| SW 梯度被 99.9 主导 | POR/PERM 早期不降 | 确认 `L_aux` 已按 `s_sw` 归一化 |
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
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E1
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

预注册文件：`v4/reports/E1_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E1_P1_gate",
  "stage": "E1",
  "p_stage": "P1",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "oof_total",
  "primary_threshold_key": "min_delta",
  "baseline_version": "CONST",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 7.5,
    "min_effect_floor": 0.0,
    "oof_total_min": 78.0,
    "min_same_direction_folds": 5,
    "min_placeholder_acc": 0.98
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

---

> **WP0 更新（2026-09）**：
> - 预注册新增 `min_placeholder_acc: 0.98` 与 `min_same_direction_folds: 5`，两者都进入
>   `aggregate_gate` 的绝对门槛；读取旧预注册缺键时拒绝静默沿用。
> - 早停/选 epoch 由“连续头分数”改为 **gated 代理分（τ=0.5）**；日志同时输出
>   `innerOOF_gated@0.5` 与 `innerOOF_cont`。
> - 重跑验收：`oof_total ≥ 78` **且** `placeholder_min_acc ≥ 0.98`；否则不得进入 E3。
