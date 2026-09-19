# E7/P0 三段式损失消融与退火策略

> 所属阶段：[E7](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：损失配方冻结　|　**依赖**：E6/P2（PD1 基线）

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

对 `L_align / L_aux / L_ph` 三组权重与 `λ₁` 退火曲线做完整消融（同结构对照），确定唯一损失配方并冻结。

## 2. 为什么需要这一步

1. `资料库/12` §2.3 指出纯对齐损失早期梯度稀疏（大量点落在容忍域外，梯度≈0），必须靠 aux 提供早期梯度；
2. 评分 `max(0,·)` 截断意味着超过容差阈值的点不再产生梯度收益，把容量让给"临界点"是理论最优——这只能通过损失权重实现；
3. 配方必须消融确定，不能凭感觉设 λ。

## 3. 输入契约

- E6/P2 的 PD1 管线（结构冻结）
- `资料库/12` §2.2–2.4

## 4. 输出契约

- `src/losses/score_aligned.py`（最终配方）
- `$V4_REPORTS_DIR/E7_loss_ablation.json`
- `versions/configs/loss_v1.json`（冻结配置）

## 5. 执行步骤

1. 对照实验 1：纯 align vs 纯 aux vs align+aux
2. 对照实验 2：λ₁ ∈ {0.1,0.3,1.0} × 退火曲线 ∈ {常数, 线性到 0.1, 余弦}
3. 对照实验 3：λ₂ ∈ {0.1,0.2,0.3} 对占位 Acc 的影响
4. 对照实验 4（可选）：加 `L_phys`（物理软约束）并消融其 λ₃
5. 所有对照在 fold0+1 上做，胜者跑全 5 折确认
6. 用**真实评分**而非 loss 值选择配方

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| λ₁（aux） | 1.0 → 0.1（线性，前 60% epoch） | 见对照 2 | inner/fold0+1 选择 |
| λ₂（占位 BCE） | 0.2 | 0.1/0.2/0.3 | 同上 |
| λ₃（物理） | 0（默认关） | 0/0.02/0.05 | 必须消融；`资料库/03` 提醒 KC 只能定性 |
| `alpha`/`beta` | 1e-3 / 20 | 1e-3~1e-2 / 10~30 | 平滑参数 |
| `huber_beta` | 1.0 | 0.5/1.0/2.0 | aux 损失 |

## 7. 完成判据

- 对齐损失 ≥ 纯 aux 损失（同结构对照，CI 下界 > 0）
- 三段式权重的完整消融表（含退火曲线）
- 最终配方写入 `versions/configs/loss_v1.json` 并冻结
- 每个对照都能复算（脚本 + 命令 + 产物 sha256）

## 8. 禁止事项

- 同时改多个损失项导致无法归因
- 用 loss 值而非真实评分选配方
- 在看到 outer 折结果后调整 λ

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 对照实验爆炸 | 组合数过多 | 分层做：先定性（哪一项有用），再定量（λ 搜 3 档） |
| 物理损失引入偏差 | POR/SW 变好但 PERM 变差 | λ₃ 只在所有目标都不退化时才采纳 |

## 10. 停止规则

- 若 align 与 aux 无差异，保留简单配方（align+aux+ph 默认值）并记录

## 11. 代码归属

- `E7/code/ablate_loss.py`

## 12. 复算与证据

- `reports/E7_loss_ablation.json`、`versions/configs/loss_v1.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E7
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E7_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E7_P0_gate",
  "stage": "E7",
  "p_stage": "P0",
  "candidate_budget": 8,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "oof_total",
  "thresholds": {
    "min_delta": 0.0
  },
  "multiplicity": "holm"
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
