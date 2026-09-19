# E10/P0 全量重训与 CPU 推理导出

> 所属阶段：[E10](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：交付：训练+推理双入口　|　**依赖**：E9/P2

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

## 13. Gate 预注册要点

预注册文件：`v4/reports/E10_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E10_P0_gate",
  "stage": "E10",
  "p_stage": "P0",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "cpu_inference_ok",
  "thresholds": {
    "max_minutes": 30,
    "max_memory_gb": 8
  },
  "mandatory_checks": [
    "cpu_inference_ok",
    "deterministic_output"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
