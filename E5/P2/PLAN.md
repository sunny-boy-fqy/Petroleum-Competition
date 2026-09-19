# E5/P2 SW 单尺度精修（占位 99.9 与有效 8.3–99.9 同尺度）

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：单目标攻坚：SW（权重 35%，结构最特殊）　|　**依赖**：E5/P1
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现 SW 的占位/有效双分支混合输出（`q̂·99.9 + (1−q̂)·f_valid`，**同一标签尺度**），用 BCE 监督占位分支，并验证不引入任何尺度换算；提升 SW 连续切片准确率。

## 2. 为什么需要这一步

1. 占位峰（99.9）与有效峰（实测 8.3–99.9，中位 82.8）**同尺度但分布形状完全不同**，单头线性回归仍会被占位尖峰拉扯（`资料库/12` §3.4 的双峰会震荡结论在结构上成立）；
2. **E0-R2 修正**：SW 不是 `[0,1]` 双尺度（审查 B2/R2-B4 实测 SW<1 为 0 行，min 8.305）——因此**禁止**任何 ×100 换算；`constants.SW_SMALL_BRANCH=False`，有效分支直接用标签尺度监督；
3. `资料库/12` §3.4 指出在 `q̂` 灰色地带向 99.9 偏移可换期望分——这是该指标允许的"下注"。

## 3. 输入契约

- E3/E4 冻结主干表示
- E1/P1 的 SW 基线 OOF

## 4. 输出契约

- `E5/code/head_sw.py`
- `$V4_RUN_ROOT/E5/sw/oof.npz`
- `$V4_REPORTS_DIR/E5_sw.json`

## 5. 执行步骤

1. 实现双分支：`q̂=sigmoid(g0)`（可与 H0 共享或独立）、`f` 为**标签尺度**的有效分支输出、混合 `q̂·99.9 + (1−q̂)·f`
2. 单测锁定尺度：`q̂=1` 时输出必须精确 99.9；`q̂=0` 时输出等于 `f`（**不做任何倍数换算**）；并断言 `constants.SW_SMALL_BRANCH is False`
3. 试验灰色地带偏移策略（在 inner-OOF 上选阈值）
4. 报告 SW 两个切片的 Acc：占位行、有效行（实测 8.3–99.9）
5. 确认**绝不做全局 [0,1] 裁剪**（会掉约 23 分）

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 有效分支输出 | 标签尺度（**不乘 100**） | 冻结 | `constants.SW_SMALL_BRANCH=False` |
| 占位分支 | 常数 99.9 | 冻结 | 不参与梯度（只作为混合常量） |
| 灰色地带阈值 | 0.3–0.5 内选 | inner 选择 | 向 99.9 下注 |
| 候选数 | 3–4 | — | holm 校正 |

## 7. 完成判据

- 尺度单测通过（`q̂=1 → 99.9`；`q̂=0 → 输出等于有效分支，无倍数换算）
- SW 有效行与占位行的 Acc 均报告；有效行 Acc 提升且 CI 下界 > 0
- 契约校验通过：SW 输出不被裁剪，且与标签尺度一致
- 占位行 SW Acc ≥ 0.99

## 8. 禁止事项

- 全局裁剪 SW 到 [0,1]（`资料库/12` §0 结论 2：直接损失约 23 分）
- 对有效分支做 ×100 换算（E0-R2 已证伪双尺度假设）
- 用占位行样本训练有效分支

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 误用双尺度换算 | SW 有效段预测整体偏大 100 倍 | 单测断言 SW_SMALL_BRANCH=False + 契约层范围检查 |
| 双分支失衡 | 占位/有效一侧塌陷 | 分别报告两切片 Acc；调整 λ₂ |
| 灰色地带过拟合 | inner 提升 outer 下降 | 阈值只在 inner 选并报告敏感性 |

## 10. 停止规则

- SW 有效行连续 3 次无正增量 → 该方向停止，转 E6

## 11. 代码归属

- `E5/code/head_sw.py`

## 11.5 接口与实现约定

- **接口语义（R2-M3）**：`head_sw` 的 `f_valid` 必须输出**标签尺度**（百分数，实测 8.3–99.9），不得是归一化值；`RowMLP` 用 `sw_affine_w/b` 做输出层仿射，训练脚本需先调用 `init_from_stats(sw_median=82.8, sw_std≈20)`。

## 12. 复算与证据

- `reports/E5_sw.json`

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

预注册文件：`v4/reports/E5_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E5_P2_gate",
  "stage": "E5",
  "p_stage": "P2",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "sw_acc",
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
  "candidate_budget": 4,
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
