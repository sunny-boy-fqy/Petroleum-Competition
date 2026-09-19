# E5/P1 PERM log 域精修（长尾与数量级）

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：单目标攻坚：PERM（权重 35%，边际收益最高）　|　**依赖**：E5/P0

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

在 log10 域精修 PERM：处理长尾与数量级误差，输出经 tanh 夹到 [-6,6] 保证正有限，提升 PERM 连续切片准确率。

## 2. 为什么需要这一步

1. PERM 权重 35%、历史探索最少、边际收益最高（`资料库/12` §3.3 排序 PERM > SW ≈ POR）；
2. 评分是 `|log10(ŷ/y)|`，相差 10 倍即得 0——**必须在数量级上正确**，绝对误差无意义；
3. `资料库/13` 指出测井预测渗透率"落在真值两倍内已算很好"，因此目标是量级正确而非过拟合 RMSE。

## 3. 输入契约

- E3/E4 冻结主干表示
- E1/P1 的 PERM 基线 OOF

## 4. 输出契约

- `E5/code/head_perm.py`
- `$V4_RUN_ROOT/E5/perm/oof.npz`
- `$V4_REPORTS_DIR/E5_perm.json`

## 5. 执行步骤

1. 分析 z 空间误差分布：σ(z)、落在 |Δz|<1 的比例、长尾方向（低估/高估）
2. 试验量化分桶辅助损失（把 z 分箱做 soft 分类，再求期望）与纯回归对照
3. 实现分位数/异方差辅助头（`资料库/09` §4）以改善尾部
4. 确保输出 `10^clip(z)` 严格 > 0 且有限（契约层会二次校验）
5. 只在 inner-OOF 上选方案

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| z 输出 | `6·tanh(g)` | tanh/clip/线性 | tanh 保证有界 |
| 辅助损失 | Smooth L1（默认） | Smooth L1 / 分桶 soft-CE / 分位数 | inner 选择 |
| clip 范围 | [-6, 6] | 冻结 | `constants.PERM_LOG_MIN/MAX` |
| 候选数 | 3–5 | — | holm 校正 |

## 7. 完成判据

- PERM 连续切片 Acc 提升且 CI 下界 > 0
- 无 ≤0 或非有限输出（契约自动校验）
- z 空间误差分布改善（σ(z) 或尾部比例）有数据支撑

## 8. 禁止事项

- 线性域建模 PERM
- 用 ReLU 输出 PERM（0 处零梯度）
- 改动 PERM 的评分公式或权重

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 长尾被平均掩盖 | 整体 Acc 微升但尾部更差 | 分位数报告：按真值分箱统计 Acc |
| 分桶边界引入偏差 | 分桶方案的 OOF 不稳定 | 分桶边界只在训练折确定并冻结 |

## 10. 停止规则

- PERM 连续切片连续 3 次无正增量 → 转 SW

## 11. 代码归属

- `E5/code/head_perm.py`

## 12. 复算与证据

- `reports/E5_perm.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E5
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E5_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E5_P1_gate",
  "stage": "E5",
  "p_stage": "P1",
  "candidate_budget": 5,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "perm_acc",
  "thresholds": {
    "min_delta": 0.0
  },
  "multiplicity": "holm"
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
