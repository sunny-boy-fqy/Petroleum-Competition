# E7/P1 解码与后处理（温度/偏置/收缩）

> 所属阶段：[E7](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：零模型改动的换分手段　|　**依赖**：E7/P0

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现并选择推理期解码手段：逐目标偏置校正、温度/收缩、分位数收缩，全部在 inner-OOF 上选定并冻结。

## 2. 为什么需要这一步

1. 解码在**不重训模型**的前提下换分，成本最低、风险最小；
2. 评分对每个目标有独立的最优"保守/激进"倾向（例如在容忍带边界附近，向众数偏移可提高期望分）；
3. 但解码参数极易过拟合 inner，因此必须做敏感性分析并只取平坦区间。

## 3. 输入契约

- E6/P2 的 PD1 逐行预测（inner-OOF）
- E0 的评分器

## 4. 输出契约

- `src/inference/decode.py`
- `$V4_REPORTS_DIR/E7_decode_search.json`
- `versions/configs/decode_v1.json`

## 5. 执行步骤

1. 实现逐目标偏置 `ŷ ← ŷ + b_t`，在 inner-OOF 上搜索 b_t（小范围）
2. 实现收缩 `ŷ ← μ_t + α_t(ŷ − μ_t)`，搜索 α_t ∈ [0.9, 1.1]
3. 实现分位数收缩（`资料库/09` §4 思路）：把预测往训练折分位数靠拢
4. 做参数敏感性热图，只采纳平坦区中点
5. 冻结 `decode_v1.json` 并复算 OOF 确认增益
6. 验证解码不破坏占位行的精确输出（原子门在解码之后仍生效）

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 偏置 b_t | 0（默认） | ±0.005（POR）/ ±0.02（SW）/ ±0.05（z） | inner 选择 |
| 收缩 α_t | 1.0 | [0.9, 1.1] | inner 选择 |
| 分位数收缩 | 关 | 开/关 + 目标分位 | inner 选择 |
| 敏感性判据 | 最优邻域 ±1 档内 Acc 变化 < 0.005 | 冻结 | 否则不采纳 |

## 7. 完成判据

- 解码增益在 inner-OOF 上可复算，且 CI 下界 > 0
- 敏感性热图显示所选参数位于平坦区
- 占位行仍精确输出常量（原子门优先级高于解码）
- `decode_v1.json` 冻结并被 PD1 管线读取

## 8. 禁止事项

- 用 outer 折或 A 榜选解码参数
- 让解码覆盖原子门的常量输出
- 采纳落在敏感性尖峰上的参数

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 过拟合 inner | inner 提升 outer 下降 | 只取平坦区；报告内外一致性 |
| 解码破坏原子精确性 | 占位 Acc 下降 | 解码在原子门之前/之后的位置做单测固定 |

## 10. 停止规则

- 若解码增益 CI 含 0，判 NO-GO 并保持恒等解码

## 11. 代码归属

- `E7/code/decode_search.py`
- `src/inference/decode.py`

## 12. 复算与证据

- `reports/E7_decode_search.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E7
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E7_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E7_P1_gate",
  "stage": "E7",
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
    "inner_only_selection",
    "atomic_precision_reported"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
