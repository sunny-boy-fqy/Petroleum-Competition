# E0/P2 官方评分器复算与分母口径冻结

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：度量工具冻结：评分器错了，后面全部结论作废　|　**依赖**：E0/P1（数据卡与标签状态）

> **状态：已完成（2026-09-19）。** 证据：`src/score.py`、`reports/E0_data_card.json::constant_baseline`（drop=70.490735，mask=69.843218）。
>
> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

按 `rules.md` §7.3 实现三目标评分器，用全常量 (0.1, 0.01, 99.9) 复算，**确定官方分母口径**（逐目标排除缺测 vs 全行），并冻结为项目唯一口径。

## 2. 为什么需要这一步

1. 评分器是所有 Gate 的度量工具，必须与官方逐点一致；
2. 分母口径有歧义：rules 公式写 `1/N Σ`（N=总点数），但实测常数基线在**逐目标排除缺测**口径下 = 70.490735（命中公开锚点 70.4907 ±1e-4），而在全行口径下 = 69.843218（低 0.65 分）——0.65 分足以改变 Gate 判定；
3. `资料库/12` §1 明确 `eps` 只用于防除零，取 1e-3 而非 1e-6 可稳定梯度。

## 3. 输入契约

- `rules.md` §7.3–7.4
- `资料库/12` §1（逐条解析）、§3.3（分数预算表）
- E0/P1 的 730,268 行标签与缺测掩码

## 4. 输出契约

- `$V4_REPORTS_DIR/E0_score_check.json`（两种口径的逐目标 Acc 与 Total）
- `src/score.py`（冻结实现，默认 `missing_mode="drop"`）
- `src/constants.py::SCORE_MISSING_MODE/SCORE_WEIGHTS/DELTA_POR/DELTA_SW`

## 5. 执行步骤

1. 实现 `acc_relative`（POR/SW）与 `acc_perm`（log10）两个原子函数
2. 实现 `score_arrays(y_true, y_pred, missing, missing_mode)` 返回逐目标 Acc 与 Total
3. 两种口径各跑一次常数基线，记录到 `E0_score_check.json`
4. 核对 POR/SW/PERM 三个 Acc 与 `资料库/12` §3.3 预算表自洽（占位白送 66.72）
5. 把命中锚点的口径写入 `constants.SCORE_MISSING_MODE` 并加单元测试锁定
6. 写 4 个边界单测：y=0、ŷ=0、ŷ/y=10、缺测行

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `missing_mode` | drop（冻结） | drop/mask | mask 仅作诊断对照，报告必须标注 |
| `eps` | 1e-3 | 冻结 | `资料库/12` §2.4 提示 2 |
| PERM 比值下限 | eps=1e-3 | 冻结 | `log10(max(ŷ/y, eps))` |
| 权重 | POR 0.30 / PERM 0.35 / SW 0.35 | 冻结 | rules §7.4 |
| 容差 | POR δ=0.08 / SW δ=0.05 | 冻结 | rules §7.3 |

## 7. 完成判据

- 常数基线在冻结口径下 = **70.490735 ± 1e-4**（锚点 70.4907）
- 另一种口径的数字同时记录（69.843218）并在数据卡标注差异
- 逐目标 Acc 与预算表自洽：POR 0.6736 / PERM 0.7467 / SW 0.7421（±0.002）
- `src/score.py` 在**没有 torch** 的环境下可导入并运行

## 8. 禁止事项

- 用第三方库的 `mean_squared_error` / `r2_score` 代替官方公式
- 把缺测行计入分母（除非显式标 `missing_mode="mask"`）
- 更换口径后不重算全部历史数字

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 口径选错 | 所有 OOF 数字系统性偏低 0.65 分 | 以锚点 70.4907 命中的口径为准，并写进 constants + 单测 |
| eps 取 1e-6 导致梯度爆炸 | 训练早期 loss NaN | 固定 eps=1e-3 |
| PERM 出现 0 或负值 | log10 报错或 -inf | `max(·, eps)` + ŷ>0 由模型层保证 |

## 10. 停止规则

- 锚点未命中时禁止进入 E1；必须先修正评分器或数据卡

## 11. 代码归属

- `src/score.py`
- `E0/code/run_all.py`

## 12. 复算与证据

- `reports/E0_data_card.json::constant_baseline`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E0
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E0_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E0_P2_gate",
  "stage": "E0",
  "p_stage": "P2",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "constant_baseline_anchor",
  "thresholds": {
    "abs_tolerance": 0.0001
  },
  "mandatory_checks": [
    "constant_baseline_anchor_hit"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
