# E5/P1 PERM log 域精修（长尾与数量级）

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：单目标攻坚：PERM（权重 35%，边际收益最高）　|　**依赖**：E5/P0
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

在 log10 域精修 PERM：处理长尾与数量级误差，对齐官方评分器的截断 `1−|log10(max(ŷ/y, ε))|`（即 `clamp_min(ẑ−z, log10(eps))`），输出经 tanh 夹到 [-6,6] 保证正有限；输出层初始化使 `perm_z ≈ −0.08`（训练折有效 PERM 的 log10 中位数）。

## 2. 为什么需要这一步

1. PERM 权重 35%、历史探索最少、边际收益最高（`资料库/12` §3.3 排序 PERM > SW ≈ POR）；
2. 评分是 `|log10(ŷ/y)|`，相差 10 倍即得 0——**必须在数量级上正确**，绝对误差无意义；
3. `资料库/13` 指出测井预测渗透率"落在真值两倍内已算很好"，因此目标是量级正确而非过拟合 RMSE；
4. 改进 proposal §3 B3/§4 C2：初始化若停在 0 附近，早期相对误差极大；且官方对**极端低估**有 `max(ŷ/y, ε)` 截断，损失必须在 log 空间显式复算这一截断，否则低估尾部的梯度与得分与官方不一致。

## 3. 输入契约

- E3/E4 冻结主干表示
- E1/P1 的 PERM 基线 OOF
- E1/P0 的训练折 `perm_z_median ≈ −0.08`

## 4. 输出契约

- `E5/code/head_perm.py`
- `$V4_RUN_ROOT/E5/perm/oof.npz`
- `$V4_REPORTS_DIR/E5_perm.json`
- `$V4_REPORTS_DIR/E5_perm_tail.json`（低估尾部 vs 官方评分器一致性报告）

## 5. 执行步骤

1. 分析 z 空间误差分布：σ(z)、落在 |Δz|<1 的比例、长尾方向（低估/高估）
2. 输出层初始化到训练折有效 PERM 的 `log10` 中位数（≈ −0.08），而不是 0
3. 实现官方截断对齐：`d = clamp_min(zhat − z, log10(eps))`，再套平滑绝对损失；写单测对比「对齐损失」与官方 `acc_perm` 在极端低估处的单调性
4. 试验量化分桶辅助损失（把 z 分箱做 soft 分类，再求期望）与纯回归对照
5. 实现分位数/异方差辅助头（`资料库/09` §4）以改善尾部
6. 确保输出 `10^clip(z)` 严格 > 0 且有限（契约层会二次校验）
7. 写「PERM 低估尾部 vs 官方评分器」一致性报告：按真值分箱比较预测的官方 Acc 与对齐损失的排序一致性
8. 只在 inner-OOF 上选方案

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| z 输出 | `6·tanh(g)`，初始化到 `perm_z_median ≈ −0.08` | tanh/clip/线性 | tanh 保证有界 |
| 对齐截断 | `clamp_min(zhat−z, log10(eps))` | 冻结 | 与官方 `max(ŷ/y,ε)` 对齐 |
| 辅助损失 | Smooth L1（默认） | Smooth L1 / 分桶 soft-CE / 分位数 | inner 选择 |
| clip 范围 | [-6, 6] | 冻结 | `constants.PERM_LOG_MIN/MAX` |
| 候选数 | 3–5 | — | holm 校正 |

## 7. 完成判据

- PERM 连续切片 Acc 提升且 CI 下界 > 0
- 无 ≤0 或非有限输出（契约自动校验）
- z 空间误差分布改善（σ(z) 或尾部比例）有数据支撑
- 输出层初始化落在 `perm_z_median ≈ −0.08` 附近（不是 0）
- 对齐损失使用官方截断 `clamp_min(zhat−z, log10(eps))`，并有单测证明低估尾部与官方评分器一致
- `E5_perm_tail.json` 给出按真值分箱的「低估尾部 vs 官方 Acc」一致性结论

## 8. 禁止事项

- 线性域建模 PERM
- 用 ReLU 输出 PERM（0 处零梯度）
- 改动 PERM 的评分公式或权重
- 在 log 空间忽略官方 `max(ŷ/y, ε)` 截断（会让极端低估的梯度与官方不一致）

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

预注册文件：`v4/reports/E5_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E5_P1_gate",
  "stage": "E5",
  "p_stage": "P1",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "perm_acc",
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
  "candidate_budget": 5,
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
