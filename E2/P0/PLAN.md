# E2/P0 物理与交会特征（F_phys）

> 所属阶段：[E2](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：特征组候选：必须独立消融，未过则 NO-GO　|　**依赖**：E1/P1（行级基线与评分口径）

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现孔隙度类（Wyllie/密度/中子）、泥质类（GR/SP 指数）、流体类（RT/RXO、log10 RT）、骨架类（PE、DEN-CNL 差）共约 14 列物理派生特征，并在行级基线上做**单组消融**。

## 2. 为什么需要这一步

1. `资料库/01` §2–§5 与 `资料库/02` 给出成体系的岩石物理公式，是领域归纳偏置的合法来源；
2. 物理先验是**约束**不是万能解：`资料库/03` 明确 Kozeny–Carman 只能定性，`资料库/13` 指出渗透率两倍以内已算很好，因此必须消融而不能硬编码进模型；
3. 特征一旦进入训练就必须冻结版本（先定义再实验），否则会出现"看 OOF 后加列"的选择偏差。

## 3. 输入契约

- E1/P0 的行级张量
- `资料库/01` §2–§6、`资料库/02`、`资料库/16`（交会图版）

## 4. 输出契约

- `src/features/physics.py`（每个派生列一个纯函数 + 公式注释）
- `$V4_REPORTS_DIR/E2_ablation.json`（组级 delta 与 CI）
- `$V4_REPORTS_DIR/E2_feature_provenance.csv`（列名/公式/来源登记）

## 5. 执行步骤

1. 按公式逐个实现派生列（Wyllie 声波孔隙度、密度孔隙度、中子孔隙度、GR 指数 IGR、SP 指数、RT/RXO 比值、log10 RT、PE 骨架指示、DEN-CNL 差、AC-DEN 交会等）
2. 对可能除零/负数的公式加数值保护（如 `(AC-ACma)/(ACf-ACma)` 的夹取）
3. 缺失输入传播为缺失输出（不填 0），并同步生成缺失指示位
4. 在行级 MLP 上做"F1 vs F1+F_phys"单组消融（同折同超参，只改特征）
5. 登记每个派生列的公式与出处到 provenance CSV
6. 给出采纳/NO-GO 结论与 bootstrap CI

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| AC 骨架/流体时差 | ACma=55.5, ACf=189 μs/ft→换算为 μs/m | 依据 `资料库/01` | 需按数据单位换算 |
| DEN 骨架/流体密度 | ρma=2.65, ρf=1.0 g/cm³ | 依据 `资料库/01` | 同上 |
| GR 泥质基线 | GRmin/GRmax 由**训练折**分位数确定 | 折内 fit | 禁止用全量分位数 |
| 消融判据 | 组级 delta 的 CI 下界 > 0 | 冻结 | 否则 NO-GO |

## 7. 完成判据

- 每个派生列有公式出处与数值保护，且无目标值参与构造
- 组级消融给出 delta 与 CI，明确采纳或 NO-GO
- provenance CSV 覆盖全部新增列（列名 → 公式 → 依据）
- 派生特征在测试井上同样可计算（不依赖标签）

## 8. 禁止事项

- 使用目标值（POR/PERM/SW）构造任何派生列
- 用全量数据确定 GRmin/GRmax 等分位数参数
- 把物理公式当成硬约束直接替换模型输出（那属于 E7 的 L_phys 消融）

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 单位换算错误 | 派生孔隙度出现 0–1 之外的离谱值 | 对每个派生列做物理区间检查并记录越界比例 |
| 公式引入泄漏 | 消融提升异常大 | provenance 审计 + label-shuffle 检查 |
| NO-GO 被硬塞进模型 | 特征表膨胀但无增量 | Gate 强制要求显式 NO-GO 记录 |

## 10. 停止规则

- 单组消融 CI 上界 ≤ 0 时标记 NO-GO，不得进入 F2

## 11. 代码归属

- `src/features/physics.py`
- `E2/code/ablate_groups.py`

## 12. 复算与证据

- `reports/E2_feature_provenance.csv`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E2
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E2_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E2_P0_gate",
  "stage": "E2",
  "p_stage": "P0",
  "candidate_budget": 3,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "target_acc",
  "thresholds": {
    "min_delta": 0.0
  },
  "multiplicity": "holm"
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
