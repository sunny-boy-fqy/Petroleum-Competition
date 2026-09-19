# E5/P2 SW 双峰精修（占位 99.9 vs 有效 [0,1]）

> 所属阶段：[E5](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：单目标攻坚：SW（权重 35%，结构最特殊）　|　**依赖**：E5/P1

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现 SW 的双分支混合输出（`q̂·99.9 + (1-q̂)·σ(f)·100`），用 BCE 监督占位分支，并验证双尺度换算正确；提升 SW 连续切片准确率。

## 2. 为什么需要这一步

1. SW 的两个峰（99.9 与 [0,1]）相距上百个标准差，单头线性回归会在峰间震荡（`资料库/12` §3.4）；
2. **同一列两种尺度**是最容易造成约 10 分静默损失的坑：有效分支必须 ×100 才与占位分支同尺度；
3. `资料库/12` §3.4 指出在 `q̂` 灰色地带（0.3–0.5）向 99.9 偏移可换期望分——这是该指标允许的"下注"。

## 3. 输入契约

- E3/E4 冻结主干表示
- E1/P1 的 SW 基线 OOF

## 4. 输出契约

- `E5/code/head_sw.py`
- `$V4_RUN_ROOT/E5/sw/oof.npz`
- `$V4_REPORTS_DIR/E5_sw.json`

## 5. 执行步骤

1. 实现双分支：`q̂=sigmoid(g0)`（可与 H0 共享或独立）、`f` 为有效分支 logit、输出 `q̂·99.9 + (1-q̂)·sigmoid(f)·100`
2. 单测锁定尺度：把 (q̂=0, f=0) 的输出与 0.5×100=50 比较；把 (q̂=1, f=任意) 的输出与 99.9 比较
3. 试验灰色地带偏移策略（在 inner-OOF 上选阈值）
4. 报告 SW 两个切片的 Acc：占位行、[0,1] 有效行
5. 确认**绝不做全局 [0,1] 裁剪**（会掉约 23 分）

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 有效分支激活 | sigmoid × 100 | 冻结 | `constants.SW_VALID_SCALE` |
| 占位分支 | 常数 99.9 | 冻结 | 不参与梯度（只作为混合常量） |
| 灰色地带阈值 | 0.3–0.5 内选 | inner 选择 | 向 99.9 下注 |
| 候选数 | 3–4 | — | holm 校正 |

## 7. 完成判据

- 双尺度单测通过（锁定 ×100 换算）
- SW 有效行与占位行的 Acc 均报告；有效行 Acc 提升且 CI 下界 > 0
- 契约校验通过：SW 输出不被裁剪，且与标签尺度一致
- 占位行 SW Acc ≥ 0.99

## 8. 禁止事项

- 全局裁剪 SW 到 [0,1]（`资料库/12` §0 结论 2：直接损失约 23 分）
- 在有效分支输出百分数后再乘 100（重复换算）
- 用占位行样本训练有效分支

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 尺度换算错误 | SW Acc 掉约 10 分 | 双尺度单测 + 契约层断言 |
| 双分支失衡 | 占位/有效一侧塌陷 | 分别报告两切片 Acc；调整 λ₂ |
| 灰色地带过拟合 | inner 提升 outer 下降 | 阈值只在 inner 选并报告敏感性 |

## 10. 停止规则

- SW 有效行连续 3 次无正增量 → 该方向停止，转 E6

## 11. 代码归属

- `E5/code/head_sw.py`

## 12. 复算与证据

- `reports/E5_sw.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E5
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E5_P2_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E5_P2_gate",
  "stage": "E5",
  "p_stage": "P2",
  "candidate_budget": 4,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "sw_acc",
  "thresholds": {
    "min_delta": 0.0
  },
  "multiplicity": "holm",
  "mandatory_checks": [
    "sw_scale_unit_test",
    "contract_ok"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
