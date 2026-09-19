# E5/P0 POR 窄带精修（±0.008）

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：单目标攻坚：POR（权重 30%，容差最窄）　|　**依赖**：E3/P2 或 E4/P1（冻结主干）

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

针对 POR 的极窄容差带（0.08×0.1 = ±0.008）设计专用头与训练策略，提升 POR 连续切片准确率且不牺牲其他目标。

## 2. 为什么需要这一步

1. POR 权重 30% 但**容差带最窄**：占位 0.1 的允许误差只有 ±0.008，任何抖动都会掉出带外；
2. POR 头从 `0.1 + softplus(g)` 起步可保证非负且离占位值近，减少初期震荡；
3. `资料库/12` §3.3 指出 POR 全空间 9.71 分，是三个目标中上限最小的，因此策略应是"守住占位 + 精修有效段"而不是全面重构。

## 3. 输入契约

- E3/E4 的冻结主干逐行表示
- E1/P1 的 POR 基线 OOF

## 4. 输出契约

- `src/models/heads.py::PorHead`（精修版）
- `$V4_RUN_ROOT/E5/por/oof.npz`
- `$V4_REPORTS_DIR/E5_por.json`（连续切片 Acc、带内占比、逐折 delta）

## 5. 执行步骤

1. 统计 POR 误差分布：落在 ±0.008 带内的比例、带外距离分布（定位问题在偏移还是方差）
2. 试验三种 POR 参数化：`0.1+softplus`、`sigmoid·0.5`、直接线性（作对照）
3. 试验误差加权：对接近带边界的样本加大权重（可微权重，不改标签）
4. 只在 inner-OOF 上选参数化与权重，outer 折只推理一次
5. 报告 POR 的**连续切片**（排除占位行）Acc 与占位行 Acc 的跷跷板效应

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| POR 参数化 | `0.1 + softplus(g)` | softplus/sigmoid/线性 | inner 选择 |
| 带边加权 | 关（默认） | 开/关 + 权重 2/5 | inner 选择 |
| δ（容差） | 0.08（官方） | 冻结 | 不得改动 |
| 候选数 | 3–4 | — | holm 校正 |

## 7. 完成判据

- POR 连续切片 Acc 提升且 CI 下界 > 0
- PERM/SW 不退化超过 0.01（总分为准的跷跷板检查）
- POR 占位行 Acc 仍 ≥ 0.99

## 8. 禁止事项

- 改动 POR 容差或评分权重
- 用全局裁剪把 POR 压到 [0,0.4]（会破坏占位值 0.1 之外的物理含义）
- 为提升 POR 而牺牲 PERM/SW

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 带边加权导致占位过拟合 | 占位 Acc 上升但有效段下降 | 报告两个切片并做跷跷板检查 |
| POR 头震荡 | 连续段预测呈锯齿 | 提高 λ_align 中 POR 的有效样本权重；检查 BN 统计 |

## 10. 停止规则

- POR 连续切片连续 3 次无正增量 → 该方向停止，转 PERM/SW

## 11. 代码归属

- `E5/code/head_por.py`

## 12. 复算与证据

- `reports/E5_por.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E5
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E5_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E5_P0_gate",
  "stage": "E5",
  "p_stage": "P0",
  "candidate_budget": 4,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "por_acc",
  "thresholds": {
    "min_delta": 0.0
  },
  "multiplicity": "holm"
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
