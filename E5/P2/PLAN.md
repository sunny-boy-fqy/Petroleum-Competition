# E5/P2 SW 单尺度精修（训练折仿射归一化 + q_sw 硬切换）

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：单目标攻坚：SW（权重 35%，单一百分数尺度）　|　**依赖**：E5/P1
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

把 SW 连续头改为**训练折仿射归一化**训练、输出反变换回标签尺度：`sw_norm = (sw − sw_mu)/sw_sigma`、`sw_cont = sw_mu + sw_sigma·head_out`，其中 `sw_mu = median(训练折有效 sw)`、`sw_sigma = IQR(训练折有效 sw)/1.349`；占位由**逐目标原子头 `q_sw`** 精确决定 `SW = 99.9`；提升 SW 连续切片准确率。

## 2. 为什么需要这一步

1. **E0-R2 已证伪双尺度假设**：SW 是**单一标签尺度**（百分数），实测有效 SW 为 min 8.305 / median 82.805 / max 99.9，`SW<1` 的行数为 **0**——`SW_SMALL_BRANCH` 作为遗留对照路径**永久关闭**，不存在 `[0,1]` 有效分支，也禁止任何 ×100 换算；
2. 占位尖峰（99.9）与有效分布（8.3–99.9）**在同一尺度上形状差异极大**，线性头从 0 附近起步会让早期相对误差损失极大，因此连续头必须在归一化空间训练、输出再反变换；
3. SW 的原子判定交给独立 `q_sw`（不是 joint 头）：实测 SW 单目标原子行有 31,030 行，joint 头覆盖不到，必须逐目标保护；
4. 占位与连续是**互斥的硬切换**（`SW = 99.9 if q_sw > τ_sw else sw_cont`），τ_sw 由 E6/P1 在 inner-OOF 上用官方总分选；允许在 `[0,100]` 内做软裁剪，**严禁**全局压到 `[0,1]`。

## 3. 输入契约

- E3/E4 冻结主干表示
- E1/P1 的 SW 基线 OOF
- E1/P0 的训练折 `sw_mu`/`sw_sigma`

## 4. 输出契约

- `E5/code/head_sw.py`（归一化空间连续头 + 反变换）
- `$V4_RUN_ROOT/E5/sw/oof.npz`（含 `sw_cont` 与 `q_sw`）
- `$V4_REPORTS_DIR/E5_sw.json`（占位行/有效行切片 Acc、逐折 delta、逐折 `sw_mu/sw_sigma`）

## 5. 执行步骤

1. 在训练折内统计 `sw_mu = median(valid sw)` 与 `sw_sigma = IQR(valid sw)/1.349`（备选 std），写入 manifest
2. 连续头在归一化空间监督：`sw_norm = (sw − sw_mu)/sw_sigma`，输出 `head_out`，前向还原 `sw_cont = sw_mu + sw_sigma·head_out`；输出层 bias 初始化为 0（初始输出≈sw_mu，不是 0）
3. 实现逐目标原子头 `q_sw = sigmoid(g)`（在 E6/P0 与其它原子头一起训练），硬切换 `SW = 99.9 if q_sw > τ_sw else sw_cont`（τ_sw 来自 E6/P1）
4. 单测锁定：`q_sw > τ` 时输出**精确** 99.9；否则输出等于 `sw_cont`；断言 `constants.SW_SMALL_BRANCH is False`，且全程**无 ×100、无插值、无混合尺度算术**
5. 允许输出在 `[0,100]` 内做软裁剪，但**断言不存在 `[0,1]` 全局裁剪**
6. 报告 SW 两个切片的 Acc：占位行、有效行（实测 8.3–99.9）
7. 逐折报告 `sw_mu/sw_sigma`，确认只由训练折决定且折间差异有记录

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 连续头训练空间 | 归一化 `(sw−sw_mu)/sw_sigma` | 冻结 | 输出必反变换回标签尺度 |
| `sw_mu` | `median(训练折有效 sw)` | 冻结 | 写入 manifest/scaler JSON |
| `sw_sigma` | `IQR(训练折有效 sw)/1.349`（备选 std） | 冻结 | 写入 manifest/scaler JSON |
| 连续输出裁剪 | `[0,100]` 软裁剪（允许） | 冻结 | **禁止** `[0,1]` 全局裁剪 |
| 原子判定 | `q_sw` 硬切换（τ_sw 由 E6/P1 定） | 冻结 | 禁止插值/混合尺度 |
| 候选数 | 3–4 | — | holm 校正 |

## 7. 完成判据

- 尺度单测通过：`q_sw > τ` → 精确 99.9；否则 → `sw_mu + sw_sigma·head_out`，无倍数换算
- SW 有效行与占位行的 Acc 均报告；有效行 Acc 提升且 CI 下界 > 0
- 契约校验通过：SW 输出只允许 `[0,100]` 软裁剪，**绝不被压到 `[0,1]`**，且与标签尺度一致
- 占位行 SW Acc ≥ 0.99
- `sw_mu/sw_sigma` 只由训练折统计并可反变换；逐折参数记录完整
- `SW_SMALL_BRANCH is False` 单测通过（遗留双尺度对照永久关闭）

## 8. 禁止事项

- 把 SW 全局裁剪到 [0,1]（`资料库/12` §0 结论 2：直接损失约 23 分）
- 对连续头做 ×100 换算（E0-R2 已证伪双尺度假设）
- 在原子 99.9 与连续输出之间做插值或混合尺度算术
- 用占位行样本训练连续头而不加权重区分
- 在整表/验证折上 fit `sw_mu/sw_sigma`

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 残留双尺度换算 | SW 有效段预测整体偏大 100 倍 | 单测断言 `SW_SMALL_BRANCH=False` + 契约层 `[0,100]` 范围检查 |
| 归一化参数跨折不一致 | 逐折分数方差大 | 只允许训练折 fit；报告每折 `mu/sigma`；用鲁棒统计量 |
| 原子/连续误判 | 占位或有效一侧塌陷 | 分别报告两切片 Acc；τ_sw 由 E6/P1 在 inner-OOF 上用官方总分选 |

## 10. 停止规则

- SW 有效行连续 3 次无正增量 → 该方向停止，转 E6

## 11. 代码归属

- `E5/code/head_sw.py`

## 11.5 接口与实现约定

- **接口语义**：`head_sw` 的输出 `head_out` 位于**归一化空间**（零均值单位尺度），`sw_cont = sw_mu + sw_sigma·head_out` 才是标签尺度；`sw_mu/sw_sigma` 只由训练折有效 SW 决定，写入 checkpoint manifest 与 scaler JSON，推理时反变换。`RowMLP` 不再需要 `sw_affine_w/b` 的 ×100 语义；`constants.SW_SMALL_BRANCH` 恒为 `False`（遗留对照，永久关闭）。

## 12. 复算与证据

- `reports/E5_sw.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E5
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
    "no_label_leak",
    "sw_scale_unit_test"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
