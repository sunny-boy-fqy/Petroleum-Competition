# E3/P0 序列数据集与分块（chunk）策略

> 所属阶段：[E3](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：数据管线：序列建模的地基　|　**依赖**：E2/P2（吞吐标定）、E0/P1（分片）
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现按井分块的序列数据集：变长井切 chunk、边界 overlap、按需读取分片、worker 内存自检；确定 chunk 长度与 overlap 策略。

## 2. 为什么需要这一步

1. 整井 4,473–13,078 点无法一次性进模型（尤其 16 GiB 系统内存下），必须分块；
2. 分块策略直接决定**有效感受野**与吞吐的平衡：chunk 太短则 U-Net 看不到层段尺度，太长则内存与时间不可控；
3. 边界效应会导致 chunk 接缝处预测跳变，必须用 overlap + 加权拼接处理。

## 3. 输入契约

- E1/P0 分片、E2 特征缓存
- `资料库/08` §1.2（窗口→窗口 seq2seq）
- `资料库/07` §8.4（深度序列划分）

## 4. 输出契约

- `src/data/seq_dataset.py`
- `$V4_REPORTS_DIR/E3_seq_dataset.json`（chunk 统计、worker 内存、采样顺序可复现性）

## 5. 执行步骤

1. 实现 `SeqChunker`：按井切定长 chunk（默认 1024 点），相邻 chunk overlap 128 点
2. 实现 `SeqDataset.__getitem__`：打开分片 → 取 chunk → 即时算/读特征 → 关闭，**禁止把分片缓存在全局字典**
3. 每 worker 内存自检（< 300 MB），超限即报错而不是静默增长
4. 实现 `--smoke` 模式：前 8 井、1 折、2 epoch
5. 验证采样顺序在固定 seed 下可复现；验证 chunk 边界无标签错位
6. 实测 5 折吞吐并写入报告

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| chunk 长度 | 1024 点（≈102 m） | 512/1024/2048 | 与 U-Net 深度耦合，E3/P2 消融 |
| overlap | 128 点 | 0/64/128/256 | 拼接时按距离加权 |
| `num_workers` | 4 | 2–6 | 16 GiB 内存约束 |
| `prefetch_factor` | 2 | 1–4 | 过高会挤爆内存 |
| `pin_memory` | true | — | 加速 H2D（显存不是瓶颈） |

## 7. 完成判据

- chunk 数×长度 ≈ 井长（覆盖完整，无丢点）
- 每个 worker 常驻 < 300 MB；`num_workers=4` 时总内存 < 4 GiB
- 固定 seed 下两次运行的采样顺序一致（可复现）
- chunk 边界处的 (x, y, mask) 逐行对齐（单测）

## 8. 禁止事项

- 把整井常驻内存或缓存在全局字典
- 打乱时破坏井内深度顺序（同一 chunk 内必须有序）
- 让 chunk 覆盖出现空洞（丢点会直接丢分）

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| chunk 覆盖丢点 | OOF 行数 < 730,268 | 覆盖性单测：所有井的 chunk 拼接后等于原长 |
| overlap 拼接权重错误 | 接缝处预测跳变 | 用三角/汉宁权重并按权重归一 |
| worker 内存膨胀 | 被 OOM kill | 每 worker 自检 + `persistent_workers=False` |

## 10. 停止规则

- 覆盖性单测失败时禁止进入 E3/P1

## 11. 代码归属

- `src/data/seq_dataset.py`

## 12. 复算与证据

- `reports/E3_seq_dataset.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E3
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

预注册文件：`v4/reports/E3_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E3_P0_gate",
  "stage": "E3",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "seq_pipeline_ok",
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
    "no_label_leak",
    "seq_pipeline_ok"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
