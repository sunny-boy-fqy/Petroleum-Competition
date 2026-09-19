# E0/P2 官方评分器复算与分母口径冻结

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：度量工具冻结：评分器错了，后面全部结论作废　|　**依赖**：E0/P1（数据卡与标签状态）
>
> **状态**：✅ 已完成　　证据：`reports/E0_score_check.json`（两种口径 + 逐目标 + 恒等式校验）、`reports/E0_data_card.json::constant_baseline`

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
- 逐目标 Acc 与预算表自洽：POR 0.6735824 / PERM 0.7072023 / SW 0.7294624（实测值，见 `reports/E0_score_check.json`）
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

- `reports/E0_score_check.json`（两种口径 + 逐目标 + 恒等式校验）
- `reports/E0_data_card.json::constant_baseline`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode env    # P0：环境+磁盘（先装依赖再硬校验）
bash /code/workspace/v4/run_train.sh --mode data   # 部署数据到 /data/v4/data
bash /code/workspace/v4/run_train.sh --mode e0     # P1-P3：口径复算 + 分片缓存
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

预注册文件：`v4/reports/E0_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E0_P2_gate",
  "stage": "E0",
  "p_stage": "P2",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "constant_baseline_anchor",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "abs_tolerance": 0.0001
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
