# E10/P0 全量重训与 CPU 推理导出

> 所属阶段：[E10](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：交付：训练+推理双入口　|　**依赖**：E9/P2
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

用 E9 通过候选的配置在全部 80 井上重训（或改用 5 折权重集成），导出 CPU 可加载权重与 ONNX 兜底；验证两次前向完全一致。

## 2. 为什么需要这一步

1. `rules.md` §8.3 要求"训练 + 推理"可独立运行、结果一致；因此训练入口必须真实可跑，即使评测时不需要；
2. 评测机不保证有 GPU，推理必须在 CPU 上稳定运行（fp32、确定性）；
3. 全量重训 vs 折集成的选择必须在看到 E9 结果前预注册，避免事后择优。

## 3. 输入契约

- E9 通过的候选配置与权重
- 80 井全量数据

## 4. 输出契约

- `models/v4/final/*.pt`（fp32，CPU 可加载）
- `models/v4/final/*.onnx`（可选兜底）
- `$V4_REPORTS_DIR/E10_final_train.json`（训练日志/耗时/磁盘）

## 5. 执行步骤

1. 按预注册选择聚合方式（全量重训 或 5 折权重集成）
2. 全量重训时固定 seed、固定 epoch 数（不做早停，因为无验证折）
3. 导出：`state_dict` → fp32 `.pt`；同时尝试 `torch.onnx.export`（失败不阻塞）
4. CPU 冒烟：加载 → 前向 → 与训练期预测逐点比对（容差 ≤1e-6）
5. 两次独立前向的 sha256 必须一致（确定性验证）
6. 记录峰值内存与单次推理耗时（目标 < 30 min、< 8 GiB）

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 聚合方式 | 5 折权重集成（默认） | 全量重训/折集成 | **预注册**后不改 |
| 精度 | fp32 | fp32/fp16 | 提交侧必须 fp32 |
| 确定性 | 固定 seed + 确定性算子 | — | 两次运行 sha256 一致 |
| ONNX | 尝试导出 | 导出/跳过 | 失败不阻塞主路径 |

## 7. 完成判据

- CPU 加载并前向成功，两次结果 sha256 一致
- 单次推理 < 30 min 且峰值内存 < 8 GiB
- 训练入口 `train.py` 在 A100 上可跑通（不要求评测时执行）
- 导出的权重与配置哈希写入 manifest

## 8. 禁止事项

- 导出依赖 GPU 的权重
- 在推理阶段做任何训练
- 把训练日志/中间产物写进提交包

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 全量重训无验证折 | 无法早停，可能过拟合 | 固定 epoch 数（用折内平均最优 epoch）+ 强正则 |
| ONNX 导出失败 | 兜底路径不可用 | 不阻塞：主路径仍是 torch CPU |
| CPU 推理超时 | > 30 min | 减成员数或减 chunk；必要时用单折权重 |

## 10. 停止规则

- CPU 推理 > 30 min 或内存 > 8 GiB → 简化模型（减宽度/减成员）后重试

## 11. 代码归属

- `E10/code/final_train.py`
- `E10/code/export_cpu.py`
- `train.py`

## 12. 复算与证据

- `reports/E10_final_train.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E10
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

预注册文件：`v4/reports/E10_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E10_P0_gate",
  "stage": "E10",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "cpu_inference_ok",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "max_minutes": 30,
    "max_memory_gb": 8
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
    "cpu_inference_ok",
    "deterministic_output"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
