# E8/P1 井级分支与 transductive 消融（各一次）

> 所属阶段：[E8](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：历史 NO-GO 路线在新条件下的受控重验　|　**依赖**：E8/P0

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现 H4 井级 attention-pool 偏置分支；实现推理期 transductive 适配（伪标签/井级统计对齐）；**两者都只作消融**，给出明确的采纳/NO-GO 结论。

## 2. 为什么需要这一步

1. 工程曲线（CAL/DEVI/AZIM/BIT/CASE）在井内近常数，只提供**井间**区分度（`资料库/08` §0.1-4），井级分支是唯一合法的井间信号通路；
2. v1 E8–E11 与 v2 E6 的井级/域适应均为 NO-GO，但那些结论是在**无序列主干、CPU-only**条件下取得的；E8 在已有序列主干的前提下只重验一次；
3. transductive 适配需要使用测试井的**输入分布**（合法，标签不可见），但必须与实际提升严格区分，不能把"用了测试输入"包装成"训练改进"。

## 3. 输入契约

- E6/E7 冻结管线
- `资料库/04` §九（地理因素多数不可获得）、`资料库/09` §10

## 4. 输出契约

- `src/models/well_head.py`、`E8/code/well_branch.py`、`E8/code/pseudo_label.py`
- `$V4_REPORTS_DIR/E8_well_branch.json`、`$V4_REPORTS_DIR/E8_transductive.json`

## 5. 执行步骤

1. 实现 H4：主干输出做井级 attention-pool → 井向量 → 预测逐目标井级偏置 Δ_t→ `ŷ + λ·Δ_t`（λ 由 inner-OOF 选）
2. 消融井级分支：开/关，报告逐目标与总分 delta
3. 实现 transductive 适配：用测试井输入做特征分布对齐（如逐井分位数映射），**禁止使用任何标签**
4. 消融 transductive：开/关，明确标注"该增益来自推理期使用了测试输入分布"
5. 两个方向各自给出采纳/NO-GO 与 CI

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 井级偏置 λ | 0（默认关） | inner 搜索 [0, 0.5] | 选中后冻结 |
| 池化方式 | attention-pool | mean/max/attention | inner 选择 |
| transductive 方式 | 逐井分位数映射 | 无/分位数映射/井均值对齐 | 仅消融 |
| 候选数 | 各 2–3 | — | holm 校正 |

## 7. 完成判据

- 两个方向都给出明确结论（采纳或 NO-GO）+ CI
- transductive 结论中显式标注其合法性与局限（使用了测试输入分布）
- 井级分支若采纳，必须证明不是井身份泄漏（无 `logId` 特征、无逐井拟合标签）

## 8. 禁止事项

- 使用测试集标签（不存在，任何形式的伪标签都必须来自训练折模型输出）
- 把 transductive 适配说成"训练时改进"
- 用井身份作为特征

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
bash /code/workspace/v4/run_train.sh --mode stage --stage E8
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E8_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E8_P1_gate",
  "stage": "E8",
  "p_stage": "P1",
  "candidate_budget": 6,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "oof_total",
  "thresholds": {
    "min_delta": 0.0
  },
  "multiplicity": "holm",
  "mandatory_checks": [
    "no_label_leak",
    "inner_only_selection"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
