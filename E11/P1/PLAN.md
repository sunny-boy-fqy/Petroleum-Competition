# E11/P1 复盘与下一代方向

> 所属阶段：[E11](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：知识沉淀　|　**依赖**：E11/P0

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

写三口径（本地 CV / A 榜 / B 榜）一致性分析、路线有效性表、资源统计与下一代方向储备。

## 2. 为什么需要这一步

1. v1 的 `v1.md` 与 v2 的复盘是后续版本最有价值的输入（v3/v4 都直接引用了它们）；
2. 资源统计（训练时长/磁盘/A 榜配额）决定下一代预算分配；
3. 必须区分**确定性结论**（只依赖规则/数据/评分）与**经验性结论**（依赖当前模型家族），否则下一代会把经验当定律。

## 3. 输入契约

- 全部 `reports/E*.json`、A/B 榜历史、`training_time_log.json`

## 4. 输出契约

- `$V4_REPORTS_DIR/E11_retrospective.md`
- `$V4_REPORTS_DIR/E11_next_directions.md`

## 5. 执行步骤

1. 三口径对照表：每个候选的本地 OOF / A 榜 / B 榜（若有）与排序一致性
2. 路线有效性表：每个方向的结论（采纳/NO-GO）+ 证据 + 是否可复算
3. 资源统计：单任务训练时长分布、磁盘峰值、A 榜配额使用、镜像构建次数
4. 下一代方向：未触发的余量、未验证的假设、需要什么**新信息源**才允许重开 NO-GO 方向
5. 明确标注确定性 vs 经验性结论

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 结论分类 | 确定性/经验性 | 冻结 | 必须逐条标注 |
| B 榜 | 赛后才有 | 冻结 | 未回收则明确标缺失 |

## 7. 完成判据

- 含三口径对照表、路线有效性表、资源统计表
- 下一代方向清单每条都有触发条件
- 所有结论标注确定性/经验性
- B 榜缺失时明确标注而非猜测

## 8. 禁止事项

- 把经验性结论写成确定性结论
- 用 B 榜成绩做赛后调参并写入复盘之外的地方

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 复盘流于形式 | 无具体数字 | 每张表都必须含可核验数字与来源 |

## 10. 停止规则

- —

## 11. 代码归属

- `E11/code/retrospective.py`

## 12. 复算与证据

- `reports/E11_retrospective.md`、`reports/E11_next_directions.md`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E11
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E11_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E11_P1_gate",
  "stage": "E11",
  "p_stage": "P1",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "retrospective_complete",
  "mandatory_checks": [
    "three_way_table",
    "route_table",
    "resource_table"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
