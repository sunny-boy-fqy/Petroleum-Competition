# E6/P0 联合常量状态头（H0）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：保护屏障：66.7% 的白送分靠它守住　|　**依赖**：E5/P2（三个目标头定型）

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

训练联合占位状态分类头（BCE 监督 `POR=0.1 ∧ PERM=0.01 ∧ SW=99.9`），评估 AUC 与逐目标原子 precision/recall/F1。

## 2. 为什么需要这一步

1. 487,225 行（66.719%）是联合常量占位，占约 66.72 分的白送分；
2. v1 的原子门已被证明可从输入预测（E7 的 +0.2015 主要来自此），因此应把"是否输出常量"做成**显式可学习决策**，而不是让回归头勉强逼近；
3. 本项目不做 B0 patch 隔离（用户决策 D3），H0 是**唯一**的占位保护屏障，因此它的质量直接决定管线是否安全。

## 3. 输入契约

- E3/E4 主干逐行表示或 E1 行级特征
- E0 的占位标签

## 4. 输出契约

- `src/models/state_head.py`、`$V4_RUN_ROOT/E6/state/{foldk}.pt`
- `$V4_REPORTS_DIR/E6_atomic_report.json`（AUC/PR 曲线/逐目标原子指标）

## 5. 执行步骤

1. 实现 H0：`Linear(d→1)`，可用"逐行 + 井内平均池化"拼接增强井级信息
2. 用 BCE 训练（占位/有效/缺测三类的处理：缺测行不参与）
3. 报告 AUC 与 PR-AUC（占位类不平衡，PR 更重要）
4. 报告逐目标原子 precision/recall/F1（在 τ=0.5 与最优 τ 两处）
5. 做 label-shuffle 阴性对照，确认 AUC 不是来自泄漏

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| H0 输入 | 逐行表示（默认） | 逐行/逐行+井级池化 | inner 选择 |
| 正负样本 | 全量（占位 66.7%） | 全量/过采样有效 | 过采样需消融 |
| `pos_weight` | 1.0 | 1.0/1.5/2.0 | inner 选择 |

## 7. 完成判据

- AUC ≥ 0.97 且 PR-AUC 报告完整
- label-shuffle 对照下 AUC ≈ 0.5（证明非泄漏）
- 逐目标原子 precision/recall/F1 全部上报（`atomic_precision_reported`）

## 8. 禁止事项

- 用测试集或验证折标签训练 H0
- 把 H0 当作"裁剪器"直接覆盖回归输出而不经 τ 判定

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| H0 学不到占位 | AUC < 0.9 | 检查特征是否包含足够区分信息；加井级池化 |
| H0 过拟合 | inner AUC 高 outer 低 | 减容量 + dropout + 折内早停 |

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

## 13. Gate 预注册要点

预注册文件：`v4/reports/E6_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E6_P0_gate",
  "stage": "E6",
  "p_stage": "P0",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "state_auc",
  "thresholds": {
    "min_auc": 0.97
  },
  "mandatory_checks": [
    "atomic_precision_reported",
    "no_label_leak"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
