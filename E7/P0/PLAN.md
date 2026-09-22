# E7/P0 三段式损失消融（L_aux 归一化 / 边界聚焦 / PERM 截断）

> 所属阶段：[E7](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：损失配方冻结　|　**依赖**：E6/P2（PD1 基线）
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

对 `L_align / L_aux / L_ph` 三组权重与 `λ₁` 退火曲线做完整消融，并单独消融**归一化 `L_aux`**（训练折 `s_por`/`s_sw`）、**容差边界聚焦权重**（`κ∈{0.5,1.0,2.0}`、`σ∈{0.15,0.25,0.35}`，默认关）与 **PERM 官方 log 截断**（`clamp_min(ẑ−z, log10(eps))`）；全部为**同结构对照**，确定唯一损失配方并冻结。

## 2. 为什么需要这一步

1. `资料库/12` §2.3 指出纯对齐损失早期梯度稀疏（大量点落在容忍域外，梯度≈0），必须靠 aux 提供早期梯度；
2. 评分 `max(0,·)` 截断意味着超过容差阈值的点不再产生梯度收益，把容量让给"临界点"是理论最优——这只能通过损失权重实现；
3. 配方必须消融确定，不能凭感觉设 λ；
4. 改进 proposal §4：`L_aux` 若用绝对 Smooth L1，SW 的 99.9 会主导梯度，必须按训练折尺度归一化；边界聚焦（κ/σ）与 PERM 截断都必须用同结构对照消融，且**不能只报整体 Total**。

## 3. 输入契约

- E6/P2 的 PD1 管线（结构冻结）
- `资料库/12` §2.2–2.4
- E1/P0 的训练折 `s_por`/`s_sw`

## 4. 输出契约

- `src/losses/score_aligned.py`（最终配方）
- `$V4_REPORTS_DIR/E7_loss_ablation.json`（含 `L_aux` 归一化、边界聚焦、PERM 截断三张子表）
- `versions/configs/loss_v1.json`（冻结配置）

## 5. 执行步骤

1. 对照实验 1：纯 align vs 纯 aux vs align+aux
2. 对照实验 2：λ₁ ∈ {0.1,0.3,1.0} × 退火曲线 ∈ {常数, 线性到 0.1, 余弦}
3. 对照实验 3：λ₂ ∈ {0.1,0.2,0.3} 对占位 Acc 的影响
4. 对照实验 4：`L_aux` 绝对 Smooth L1 vs 训练折尺度归一化 `(·)/s_por`、`(·)/s_sw`（同结构对照）
5. 对照实验 5：容差边界聚焦 `w=1+κ·exp(−((r−1)²)/(2σ²))`，`κ∈{0.5,1.0,2.0}` × `σ∈{0.15,0.25,0.35}`（默认关；缺失 mask=0，原子行单独切片处理）
6. 对照实验 6：PERM 对齐项 `clamp_min(ẑ−z, log10(eps))` 开/关（同结构对照）
7. 对照实验 7（可选）：加 `L_phys`（物理软约束）并消融其 λ₃
8. 所有对照在该 outer 折的 **inner-OOF** 上做（省机时可先 fold0 资源预检），胜者跑全 5 折确认
9. 用**真实评分**而非 loss 值选择配方；每个对照同时报告逐目标与连续切片，禁止只报 Overall Total

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| λ₁（aux） | 1.0 → 0.1（线性，前 60% epoch） | 见对照 2 | inner-OOF 选择 |
| λ₂（原子 BCE） | 0.2 | 0.1/0.2/0.3 | 同上 |
| `L_aux` 归一化 | 训练折 `s_por`/`s_sw` 归一化 | 归一化 / 绝对（对照） | inner-OOF 选择 |
| 边界聚焦 `κ` | 0（默认关） | 0.5/1.0/2.0 | inner-OOF 选择；默认不启用 |
| 边界聚焦 `σ` | — | 0.15/0.25/0.35 | 同上 |
| PERM 截断 | `clamp_min(ẑ−z, log10(eps))` | 开/关 | 与官方评分器对齐 |
| λ₃（物理） | 0（默认关） | 0/0.02/0.05 | 必须消融；`资料库/03` 提醒 KC 只能定性 |
| `alpha`/`beta` | 1e-3 / 20 | 1e-3~1e-2 / 10~30 | 平滑参数 |
| `huber_beta` | 1.0 | 0.5/1.0/2.0 | aux 损失 |

## 7. 完成判据

- 对齐损失 ≥ 纯 aux 损失（同结构对照，CI 下界 > 0）
- 三段式权重的完整消融表（含退火曲线）
- **归一化 `L_aux`** 消融完成，且证明 SW 99.9 不再主导梯度（逐目标早期 loss 曲线为证）
- **边界聚焦**消融表完整：`κ∈{0.5,1.0,2.0}` × `σ∈{0.15,0.25,0.35}`（默认关，采纳需 CI 下界 > 0）
- **PERM 官方截断**消融完成，且低估尾部与官方评分器一致性通过
- 每个对照除被消融项外结构完全相同，且逐目标/逐切片指标齐全（**不得只报 Overall Total**）
- 最终配方写入 `versions/configs/loss_v1.json` 并冻结；每个对照都能复算（脚本 + 命令 + 产物 sha256）

## 8. 禁止事项

- 同时改多个损失项导致无法归因
- 用 loss 值而非真实评分选配方
- 在看到 outer 折结果后调整 λ
- 只报整体 Total 而隐藏单目标/切片退化
- 默认启用边界聚焦（它只是可选消融项）

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 对照实验爆炸 | 组合数过多 | 分层做：先定性（哪一项有用），再定量（λ 搜 3 档） |
| 边界聚焦过拟合 | inner 提升 outer 下降 | 只取平坦区；默认关；报告逐目标与切片 |
| 物理损失引入偏差 | POR/SW 变好但 PERM 变差 | λ₃ 只在所有目标都不退化时才采纳 |

## 10. 停止规则

- 若 align 与 aux 无差异，保留简单配方（align+aux+ph 默认值）并记录

## 11. 代码归属

- `E7/code/ablate_loss.py`

## 12. 复算与证据

- `reports/E7_loss_ablation.json`、`versions/configs/loss_v1.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E7
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

预注册文件：`v4/reports/E7_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E7_P0_gate",
  "stage": "E7",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "oof_total",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0
  },
  "alpha": 0.05,
  "multiplicity": "holm",
  "candidate_budget": 8,
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

> **WP4 更新（2026-09）**：新增 exp8 臂，测试 PERM 不对称损失
> `perm_over_weight ∈ {1.5, 3.0}`（官方对高估无上界、对低估有 ε 截断）。
> 所有不对称权重只在 inner-OOF 选；若 `perm_acc` 的 paired CI 下界 ≤0，则不采纳。
