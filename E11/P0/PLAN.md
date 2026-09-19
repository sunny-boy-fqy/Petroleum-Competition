# E11/P0 资产归档（含 sha256）

> 所属阶段：[E11](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：可独立理解的知识包　|　**依赖**：E10/P2
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

生成代码/模型/候选/报告的清单与 sha256，确保归档可在**不含 v1/v2/v3** 的目录中独立理解。

## 2. 为什么需要这一步

1. 归档是下一代的输入；无哈希的归档无法验证，也无法判断"哪个数字对应哪份权重"；
2. git 仓库只放代码，权重与产物在 `/data`，因此归档清单必须把两侧关联起来；
3. 失败候选（rejected）与成功候选同等重要——它们是负知识资产。

## 3. 输入契约

- `v4/` 全部报告与代码、`/data/v4/**` 的产物
- `versions/candidates.json`

## 4. 输出契约

- `$V4_REPORTS_DIR/E11_archive_manifest.json`
- `v4/docs/PROJECT_FILES.md`（目录树 + 文件用途 + 数量统计）

## 5. 执行步骤

1. 遍历 `v4/` 与关键 `/data/v4` 产物，逐项记录路径/大小/sha256/用途
2. 生成目录树文档（层级/文件数/职责），标注"提交必需"与"仅开发"
3. 核对候选注册表里每个候选都能在归档中找到对应权重与结果
4. 确认归档不依赖 v1/v2/v3 即可理解（除 E0 冻结引用件外）
5. 删除任何重复/临时产物（保留有分析价值的）

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 哈希算法 | sha256 | 冻结 | — |
| 保留策略 | 保留全部候选（含 rejected） | 冻结 | 负资产 |

## 7. 完成判据

- 清单完整且每项有 sha256
- 每个候选都能追溯到权重 + result.json + cv.json
- `PROJECT_FILES.md` 目录树与实际一致（文件数一致）

## 8. 禁止事项

- 删除任何候选或报告（含 rejected）
- 在归档中丢失"哪个数字来自哪份权重"的关联

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 清单与实物漂移 | 哈希对不上 | 归档脚本从文件系统实时计算，不手写 |

## 10. 停止规则

- —（收尾阶段无停止条件）

## 11. 代码归属

- `E11/code/archive.py`

## 12. 复算与证据

- `reports/E11_archive_manifest.json`、`docs/PROJECT_FILES.md`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E11
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

预注册文件：`v4/reports/E11_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E11_P0_gate",
  "stage": "E11",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "archive_complete",
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
