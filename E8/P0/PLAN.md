# E8/P0 MMoE 任务平衡

> 所属阶段：[E8](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：多任务结构候选　|　**依赖**：E7/P1（损失与解码冻结）
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

用 MMoE 替换硬共享主干，比较逐目标 Acc 与梯度冲突指标，判定是否优于硬共享。

## 2. 为什么需要这一步

1. `资料库/08` §1.4：多任务硬共享通常优于三个独立模型，但**必须解决权重失衡**；MMoE 允许任务部分共享，在任务相关性弱时比硬共享更稳；
2. POR/PERM/SW 由同一套岩石物理关系耦合（Archie/Kozeny–Carman/Wyllie），共享表示相当于额外归纳偏置与隐式增强；
3. 但三个目标的**扰动敏感性**不同（POR 容差 ±0.008 极窄），硬共享可能让梯度互相干扰。

## 3. 输入契约

- E6/P2 的冻结结构与特征
- `资料库/08` §1.4（MMoE 公式）

## 4. 输出契约

- `src/models/mmoe.py`
- `$V4_RUN_ROOT/E8/mmoe/oof.npz`
- `$V4_REPORTS_DIR/E8_mmoe.json`（逐目标 + 梯度冲突指标）

## 5. 执行步骤

1. 实现 MMoE：E 个专家 + 每任务独立门控 softmax 加权
2. 对照：硬共享（E6 结构）vs MMoE（E=4）vs 完全独立三模型
3. 计算梯度冲突指标（任务间梯度余弦相似度）与逐目标 Acc
4. 判定：至少一个目标提升且无目标退化
5. 胜者跑全 5 折并汇总 OOF

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 专家数 E | 4 | 2/4/8 | inner-OOF 选择 |
| 专家容量 | 与硬共享主干同宽 | — | 保证参数量可比 |
| 门控温度 | 1.0 | 0.5/1.0/2.0 | 影响路由锐度 |

## 7. 完成判据

- 逐目标 Acc 表完整（不是只报总分）
- MMoE 与硬共享的 CI 对比明确（采纳或 NO-GO）
- 梯度冲突指标有数据支撑结论

## 8. 禁止事项

- 只报告总分而隐藏单目标退化
- 在参数量差异巨大的情况下比较两种结构

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 参数量不对等 | 结论不可比 | 固定总参数量，只改共享结构 |
| 门控塌陷 | 所有任务走同一专家 | 报告门控熵；熵过低则加负载均衡损失 |

## 10. 停止规则

- MMoE 无正增量 → 保留硬共享，记录 NO-GO

## 11. 代码归属

- `src/models/mmoe.py`
- `E8/code/train_mmoe.py`

## 12. 复算与证据

- `reports/E8_mmoe.json`

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

预注册文件：`v4/reports/E8_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E8_P0_gate",
  "stage": "E8",
  "p_stage": "P0",
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
  "candidate_budget": 3,
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
