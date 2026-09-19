# E1/P0 行级输入管线与分片缓存

> 所属阶段：[E1](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：数据管线：为 E1–E8 共用，必须先冻结　|　**依赖**：E0/P1（分片写入）、E0/P2（评分）
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

构建并缓存 `F1 = 13 条曲线 + DEPTH 原始值 + 13+1 缺失位 + 4 深度编码 = 32 维` 行级张量与三目标标签，落盘为按井分片，并验证内存占用符合 16 GiB 预算。

## 2. 为什么需要这一步

1. 输入管线的正确性决定后面所有对比是否有意义：特征与标签必须逐行对齐；
2. 16 GiB 系统内存是真正的瓶颈，必须把"按井分片 + 按需读取"固化为管线，否则 E3 一开始就 OOM；
3. 标准化参数只能在训练折 fit（`资料库/07` §5、§8.6），因此管线必须支持"折内 fit"接口。

## 3. 输入契约

- `$V4_CACHE_ROOT/raw|labels/<well>.npz`（E0/P1 产出）
- `src/validation/folds.py`（outer/inner 折）

## 4. 输出契约

- `src/features/basic.py`（`build_row_features`/`build_labels`/`decode_predictions`）
- `src/data/row_dataset.py`（折内拼接 + 标准化）
- `$V4_REPORTS_DIR/E1_row_features.json`（维度、缺失率、内存峰值、耗时）

## 5. 执行步骤

1. 实现 `build_row_features`：14 原始 + 14 缺失位 + 缺失比例 + 相对深度 + 深度步长 + 深度序号
2. 实现 `build_labels`：POR/SW 原尺度、PERM 转 log10、三目标 mask、联合占位标签
3. 实现 `RowScaler`：**只在训练折 fit** 的中位数填补 + 均值/标准差标准化，输出 JSON 参数
4. 实现折内数据装配：outer 训练折做 train、outer 验证折做推理，保证行级对齐
5. 实测内存：单折激活内存峰值与常驻内存，写入报告
6. 缓存体积核对：`raw`+`labels` 合计应远小于 0.2 GB

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 特征维度 | 32 | 冻结（F1） | 含 4 个深度编码列 |
| 标准化 | 中位数填补 + 零均值单位方差 | 冻结 | 参数仅在训练折 fit |
| PERM 变换 | log10, clip[-6,6] | 冻结（F1） | `constants.PERM_LOG_MIN/MAX` |
| 分片格式 | npz(compressed), float32 | 冻结 | 原子写：tmp → rename |

## 7. 完成判据

- 特征/标签逐行对齐：`X.shape[0] == y.shape[0] == mask.shape[0]` 对全部 90 井成立
- 折内装配后训练/验证行的井集合与 `folds.json` 完全一致（无井级泄漏）
- `RowScaler` 参数可序列化为 JSON 并在推理时复现
- 常驻内存 < 6 GiB、缓存 < 0.2 GB（实测写入报告）

## 8. 禁止事项

- 在全部 80 井上 fit 标准化参数（必须折内 fit）
- 把井身份（`logId`）或折号作为特征
- 把占位行从训练集中剔除

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 特征与标签错位 | 训练 loss 不下降或分数异常低 | 断言行数一致 + 抽查若干井的 depth 对齐 |
| 标准化泄漏 | OOF 虚高 | `RowScaler` 只接受训练折索引，接口层拒绝整表 fit |
| 内存峰值过高 | 被 OOM killer 杀 | 按井拼接而非全量 concat；`num_workers=4` |

## 10. 停止规则

- 行级对齐断言失败时，禁止进入 E1/P1

## 11. 代码归属

- `src/features/basic.py`
- `src/data/row_dataset.py`
- `src/data/dataset.py`

## 12. 复算与证据

- `reports/E1_row_features.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E1
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

预注册文件：`v4/reports/E1_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E1_P0_gate",
  "stage": "E1",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "primary_metric": "row_pipeline_ok",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0
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
