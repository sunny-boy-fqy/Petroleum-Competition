# E6/P2 PD1 完整管线组装与硬 Gate（≥82.0）

> 所属阶段：[E6](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：**完整管线诞生**：本计划第一个可提交候选　|　**依赖**：E6/P0–P1、E5/P2

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

组装 数据→主干→头→原子门→契约 的完整 PD1 管线，产出 5 折 OOF、测试集 `result.json`/`result.zip`、manifest 与 cv 报告；**硬 Gate：OOF Total ≥ 82.0**。

## 2. 为什么需要这一步

1. 这是 v4 第一个"端到端可跑、可提交、可复现"的候选；
2. ≥82.0 意味着超过历史锚点 B0 的本地 OOF 80.382479，是纯 DL 路线成立的最低证据；
3. 只有完整管线才能暴露"训练能跑但推理契约不过"这类问题（前代多次踩坑）。

## 3. 输入契约

- E3/E4 冻结主干权重
- E5 的三个目标头
- E6 的 H0 与 τ
- E0 的契约与评分器

## 4. 输出契约

- `models/E6/pd1_fold{k}.pt` + `models/E6/pd1_config.json`
- `experiments/E6/P2/pd1/{oof.npz,cv.json,result.json,result.zip,manifest.json}`
- `$V4_REPORTS_DIR/E6_gate.json`
- `versions/candidates.json::PD1`（status=local_only→shortlisted）

## 5. 执行步骤

1. 实现统一推理器 `src/inference/predictor.py`：加载配置与权重 → 逐井前向 → 原子门 → 解码
2. 在 5 折上各自推理出 OOF（训练时已产出，此处复核逐行对齐）
3. 对 10 口测试井推理：平均 5 折权重（或按核验过的最优折），产出 result.json
4. 跑契约校验（10 井 / 95,948 行 / depth 对齐 / PERM>0 / 无 NaN）
5. 本机 CPU 冒烟 `predict.py --use-version PD1 --data_dir ../data --output /tmp/r.json`
6. 汇总 OOF 评分：逐目标 Acc、连续切片、占位 Acc、bootstrap CI
7. 写 manifest（config 哈希/数据指纹/折指纹/代码哈希）并注册候选
8. 写 Gate 并判定 ≥ 82.0

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 折权重聚合 | 5 折平均 | 平均/最优折/加权 | inner 决定，冻结后不改 |
| τ | E6/P1 选定值 | 冻结 | 写入 candidate registry |
| 推理精度 | fp32（CPU） | fp32/fp16 | 提交侧必须 fp32 保证确定性 |
| `num_folds` | 5 | 冻结 | 与 folds.json 一致 |

## 7. 完成判据

- OOF Total **≥ 82.0**（硬 Gate）
- 契约全绿；`predict.py --use-version PD1` 在本机 CPU 可跑通并输出 95,948 行
- 占位行逐目标 Acc ≥ 0.99；连续切片 Acc 一并上报
- manifest 写全 config/data/folds/code 四类指纹；候选已注册
- 5 折 delta 全部同向；`disk_budget_ok`、`training_time_log_valid`、`checkpoint_resumable` 为 true

## 8. 禁止事项

- 在管线中混入未冻结的特征版本
- 推理阶段读取任何标签
- 把 5 折权重聚合方式在看到 OOF 后临时更换

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 训练能跑但推理契约不过 | result.json 行数/字段错 | 契约前置到训练脚本每次落盘时校验 |
| 低于 82.0 | 纯 DL 未超过树模型锚点 | 按总计划 §9.5 回退协议准备 B0 fallback；同时保留 PD1 为 `local_only` 候选供 E7/E8 继续改进 |
| 折间差异大 | 逐折 delta 方向不一致 | 检查折内标准化与早停；报告逐折而非只报总分 |

## 10. 停止规则

- Gate < 82.0 → 冻结当前最强候选为 `PD-pre`，E7/E8 继续改进；若 E8 结束仍 < 82.0，E10 走回退协议

## 11. 代码归属

- `E6/code/build_pd1.py`
- `E6/code/gate.py`
- `src/inference/predictor.py`
- `src/inference/atomic_gate.py`

## 12. 复算与证据

- `reports/E6_gate.json`、`reports/E6_atomic_report.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E6
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E6_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E6_P2_gate",
  "stage": "E6",
  "p_stage": "P2",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "oof_total",
  "baseline_version": "E1_PD0",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "oof_total_min": 82.0
  },
  "mde_units": 80,
  "mandatory_checks": [
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "checkpoint_resumable",
    "cpu_inference_ok"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
