# E2/P1 窗口与井级特征（F_win / F_well）+ 内存纪律

> 所属阶段：[E2](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：特征组候选 + 数据管线扩容　|　**依赖**：E2/P0
>
> **状态**：▶ 进行中　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现居中多尺度窗口统计（窗长 11/51/201 点 ≈ 1.1/5.1/20.1 m）与井级聚合特征，并在 16 GiB 内存约束下落盘缓存、完成单组消融。

## 2. 为什么需要这一步

1. v1 已证明滚动窗口是稳定增益（C1→C1W 提升 +1.0562，5/5 折同向），这是"上下文有效"的最强历史证据；
2. 工程曲线（CAL/DEVI/AZIM/BIT/CASE）在井内近常数（`资料库/08` §0.1-4），井级聚合是**唯一的井间信号通路**；
3. 窗口特征也是最贵的一组，必须按需生成、按版本目录落盘，否则 30 GB 云盘与 16 GiB 内存都扛不住。

## 3. 输入契约

- E1/P0 分片
- `资料库/07` §6（窗口与中心窗口要求）、§10（增强）

## 4. 输出契约

- `src/features/window.py`、`src/features/well.py`
- `$V4_CACHE_ROOT/feat/F2_win/<well>.npz`
- `$V4_REPORTS_DIR/E2_mem_profile.json`（峰值内存/缓存体积/耗时）
- `$V4_REPORTS_DIR/E2_ablation.json`（追加 F_win/F_well 两组）

## 5. 执行步骤

1. 实现居中窗口统计：mean/std/min/max/trend（对窗内做线性回归斜率）与覆盖率
2. 实现井级聚合：各曲线井内均值/标准差/分位数、井长、平均采样间隔、井斜均值
3. 对每口井按需生成并原子落盘；换特征版本时**先删旧版本目录**
4. 做单组消融（F1+F_win、F1+F_well、F1+F_win+F_well）
5. 实测每折训练的内存峰值与缓存体积，写入 mem_profile
6. 确认无跨折/跨井泄漏（窗口只用同井邻域，井级统计只用训练折井）

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 窗长 | {11, 51, 201} 点 | 可加 {5, 1001} | 对应 1.1/5.1/20.1 m |
| 统计量 | mean/std/min/max/trend/coverage | 可裁剪 | trend = 窗内线性斜率 |
| 窗口类型 | **居中**（离线任务无因果约束） | 冻结 | `资料库/07` §6 第 917 行 |
| 缺失处理 | 窗内有效值统计 + coverage 列 | 冻结 | 不填 0 |
| 缓存上限 | 2.0 GB（可用 <12 GB 时降至 0.5 GB） | 按实测调整 | 总计划 §3.4.1 |

## 7. 完成判据

- F_win 或 F_well 至少一组消融 CI 下界 > 0（否则 NO-GO 并记录）
- 缓存体积 < 2 GB（收缩模式 < 0.5 GB），峰值常驻内存 < 12 GiB
- 窗口统计严格居中，trend 计算无未来信息泄漏（对本任务无因果约束，但保持与 v1 一致）
- provenance CSV 覆盖窗口/井级列

## 8. 禁止事项

- 使用非居中（因果）窗口（与 v1 的 C1W 口径不一致，无法对照）
- 用验证井/测试井数据计算井级统计参数
- 在缓存目录无限累积历史特征版本

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 窗口特征让内存爆掉 | worker 被 OOM kill | 按需读取 + LRU + `num_workers=4` + 每 worker <300 MB 自检 |
| 缓存撑爆磁盘 | free < 8 GB | 版本目录先删后建；`disk_guard` 每 epoch 检查 |
| 井级统计泄漏 | OOF 虚高 | 只允许训练折井参与井级统计参数计算 |

## 10. 停止规则

- 内存或磁盘触达阈值时，先降级特征组（去掉最贵窗口）再继续

## 11. 代码归属

- `src/features/window.py`
- `src/features/well.py`
- `E2/code/build_features.py`

## 12. 复算与证据

- `reports/E2_mem_profile.json`、`reports/E2_feature_provenance.csv`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E2
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

预注册文件：`v4/reports/E2_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E2_P1_gate",
  "stage": "E2",
  "p_stage": "P1",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
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
