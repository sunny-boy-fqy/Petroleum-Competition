# E2/P2 数据增强与吞吐标定

> 所属阶段：[E2](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：正则化与预算标定：为 E3 的序列主干提供可行性依据　|　**依赖**：E2/P1
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现曲线随机掩码、深度抖动、段置换等增强；用单折小规模实验标定点/秒吞吐与 batch 上限，给出 E3 每折耗时的可靠估计。

## 2. 为什么需要这一步

1. 80 井太少（730k 行但只有 80 个独立井），正则化与增强是防过拟合的主要手段；
2. E3 的序列主干比行级模型贵 1–2 个数量级，若不在 E2 标定吞吐，E3 的预算承诺就是猜的；
3. 增强必须**保持标签语义**：占位行的三目标一致性不能被破坏。

## 3. 输入契约

- E2/P0–P1 特征
- `资料库/07` §10（数据增强）

## 4. 输出契约

- `src/data/augment.py`
- `$V4_REPORTS_DIR/E2_throughput.json`（点/秒、显存/内存峰值、每折耗时估计）

## 5. 执行步骤

1. 实现增强：① 曲线通道随机置缺（模拟仪器失效）；② 深度轴小抖动（±2 点）；③ 井内随机段裁剪/重采样；④ 高斯噪声（按曲线量纲缩放）
2. 确认增强后标签仍逐行对应（对同一行做特征扰动，不移动标签）
3. 在 fold0 上跑不同 batch/length 组合，记录吞吐与峰值资源
4. 外推 5 折 × N epoch 的总耗时，写 throughput 报告
5. 给出 E3 的 `planned_task_training_h` 建议值

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 通道掩码概率 | 0.05 | 0.0/0.05/0.1 | 折内选择，不改标签 |
| 深度抖动 | ±2 点 | 0/±2/±5 | 只在特征侧 |
| 高斯噪声 σ | 各曲线训练折标准差的 1% | 0/1%/3% | 同上 |
| 增强开关 | 默认开 | 开/关 | 消融验证是否真的提升 OOF |

## 7. 完成判据

- 增强开关的消融结果（提升或 NO-GO）明确
- 吞吐报告给出点/秒与每折耗时估计，误差 < 30%
- 增强不改变任何标签行（单测：增强前后 y 与 mask 逐行相等）

## 8. 禁止事项

- 增强改变标签或行数
- 用验证/测试井数据估计噪声尺度
- 无消融地默认开启全部增强

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 增强破坏占位一致性 | 占位 Acc 下降 | 只对输入特征做扰动；标签与 mask 不动 |
| 吞吐估计过于乐观 | E3 训练超预算 | 留 2× 余量，并在 E3/P1 首个 fold 复核 |

## 10. 停止规则

- 吞吐估计若使 E3 单折 > 软预算 3 小时，先降低 chunk 长度再进入 E3

## 11. 代码归属

- `src/data/augment.py`
- `E2/code/throughput.py`

## 12. 复算与证据

- `reports/E2_throughput.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E2
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

预注册文件：`v4/reports/E2_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E2_P2_gate",
  "stage": "E2",
  "p_stage": "P2",
  "created_at": "<ISO8601，写盘时填写>",
  "primary_metric": "target_acc",
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
  "candidate_budget": 2,
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
