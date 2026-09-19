# E8/P1 井级分支与 transductive 消融（各一次）

> 所属阶段：[E8](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：历史 NO-GO 路线在新条件下的受控重验　|　**依赖**：E8/P0
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现 H4 井级 attention-pool 偏置分支（**辅助、小容量、强正则**，所有校准参数只在 inner-OOF 选）；实现推理期 transductive 适配（伪标签/井级统计对齐）；**两者都只作消融（强制）**，给出明确的采纳/NO-GO 结论。

## 2. 为什么需要这一步

1. 工程曲线（CAL/DEVI/AZIM/BIT/CASE）在井内近常数，只提供**井间**区分度（`资料库/08` §0.1-4），井级分支是唯一合法的井间信号通路；
2. v1 E8–E11 与 v2 E6 的井级/域适应均为 NO-GO，但那些结论是在**无序列主干、CPU-only**条件下取得的；E8 在已有序列主干的前提下只重验一次；
3. transductive 适配需要使用测试井的**输入分布**（合法，标签不可见），但必须与实际提升严格区分，不能把"用了测试输入"包装成"训练改进"；
4. 改进 proposal §6 E6：80 井上井级校准极易过拟合，因此井级分支只能小容量、强正则，**必须做开/关消融**，校准参数只能由 inner-OOF 决定。

## 3. 输入契约

- E6/E7 冻结管线
- `资料库/04` §九（地理因素多数不可获得）、`资料库/09` §10

## 4. 输出契约

- `src/models/well_head.py`、`E8/code/well_branch.py`、`E8/code/pseudo_label.py`
- `$V4_REPORTS_DIR/E8_well_branch.json`、`$V4_REPORTS_DIR/E8_transductive.json`

## 5. 执行步骤

1. 实现 H4：主干输出做井级 attention-pool → 井向量 → 预测逐目标井级偏置 Δ_t→ `ŷ + λ·Δ_t`（λ 由 inner-OOF 选）；**容量受限、强正则**（小 hidden、weight decay、dropout）
2. **强制消融**井级分支：开/关，报告逐目标与总分 delta + CI + 逐井非退化比例
3. 实现 transductive 适配：用测试井输入做特征分布对齐（如逐井分位数映射），**禁止使用任何标签**
4. 消融 transductive：开/关，明确标注"该增益来自推理期使用了测试输入分布"
5. 所有井级偏差/校准参数只在 inner-OOF 上选；两个方向各自给出采纳/NO-GO 与 CI

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 井级偏置 λ | 0（默认关） | inner 搜索 [0, 0.5] | 选中后冻结 |
| 井级分支容量 | 小（≤主干 1/8 宽） | 冻结上限 | 80 井过拟合风险 |
| 正则 | 强（weight decay + dropout 0.2） | 冻结 | 同上 |
| 池化方式 | attention-pool | mean/max/attention | inner 选择 |
| transductive 方式 | 逐井分位数映射 | 无/分位数映射/井均值对齐 | 仅消融 |
| 候选数 | 各 2–3 | — | holm 校正 |

## 7. 完成判据

- 井级分支**开/关消融**完成（强制），两个方向都给出明确结论（采纳或 NO-GO）+ CI
- 井级分支容量与正则在预注册上限内，校准参数只在 inner-OOF 选
- transductive 结论中显式标注其合法性与局限（使用了测试输入分布）
- 井级分支若采纳，必须证明不是井身份泄漏（无 `logId` 特征、无逐井拟合标签）

## 8. 禁止事项

- 使用测试集标签（不存在，任何形式的伪标签都必须来自训练折模型输出）
- 把 transductive 适配说成"训练时改进"
- 用井身份作为特征
- 跳过井级分支的开/关消融
- 用大容量井级分支或在 inner-OOF 之外拟合校准参数

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 井级偏置过拟合 80 井 | inner 提升 outer 下降 | λ 小范围搜索 + 只取平坦区；报告逐井非退化比例 |
| transductive 引入分布假设错误 | A 榜崩坏 | 只对 top-2 候选做，并做 16 井体检 |

## 10. 停止规则

- 任一方向 CI 含 0 → 判 NO-GO，记录证据（这是 v1/v2 同类路线的第二次受控重验）

## 11. 代码归属

- `src/models/well_head.py`
- `E8/code/well_branch.py`
- `E8/code/pseudo_label.py`

## 12. 复算与证据

- `reports/E8_well_branch.json`、`reports/E8_transductive.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E8
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

预注册文件：`v4/reports/E8_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E8_P1_gate",
  "stage": "E8",
  "p_stage": "P1",
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
  "candidate_budget": 6,
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
    "inner_only_selection"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
