# v4 冲分优化详细计划（Score-Max Plan）

> 目标：在**规则允许、可复算、无标签泄漏**的前提下，按收益/代价排序把 OOF 与提交分数推到尽可能高。
> 本文是执行顺序的**唯一计划源**；每个 WP 必须有 inner-OOF 证据、配对 CI 与 Gate 收据才能进入下一步。
> 交付纪律：每次任务完成必须 **更新文档 → git add/commit/push → `tools/pack_code_zip.py` 打包到 `/mnt/d/tmp/Petroleum-Competition/`**。

---

## 0. 当前基线（E1 实测，必须先修）

| 指标 | 实测 | 判定 |
|---|---:|---|
| `oof_total` (gated) | **78.343** | 硬门槛 78.0，边缘通过 |
| `delta_vs_const` | 7.852 | 门槛 7.5，边缘通过 |
| `paired_ci_low` | 7.233 | >0 |
| `folds_all_same_direction` | true | 过 |
| `placeholder_min_acc` | **0.9723** | **声明的硬条件 0.98 失败，且 Gate 未判定** |
| 连续头 innerOOF | 49–54 | 远低于常数基线，但这是**设计预期**（连续头对联合占位降权） |
| 单折 gated | 76.29–80.19 | 折间波动大；fold2/fold4 < 78 |

**已确认的阻断问题（优先级最高）：**

1. **Gate 漏检 `placeholder_min_acc ≥ 0.98`**：`placeholder_ok` 只写进报告，没有进入 `checks`、`result` 或 `aggregate_gate`。
2. **epoch 选择指标错位**：阶段 1 早停/选 `best_epoch` 用的是**连续头**分数（`inner_eval` 返回 `["cont"]`），而 E1 最终分数是 q_atom **硬切换后**的 gated 分数；fold1/2/4 的 `best_epoch` 只有 5/6/2。
3. **原子门控是单一全局 τ + 未校准 q_atom**：τ 搜索虽然优化总分，但无法表达“不同 q 区间/不同 cont 区间的代价差异”；E1 的 `tau=[0.18–0.28, 0.37–0.46, 0.10–0.19]` 长期偏低，说明 q_atom 欠校准/信号弱。
4. **连续头主动放弃占位行**：`joint_cont_weight=0.2` 是为了避免 66.7% 占位行支配连续头，但结果是连续头对占位行近乎无监督；gated 分数完全依赖 q_atom 的质量。

> **结论：先修 1/2/3，再跑 E3。E1 重跑成本约 7 分钟，远低于 E3 的 ~20h。**

---

## 1. 工作包总览（按顺序）

| WP | 名称 | 依赖 | 预计算力 | 预期收益 | 风险 |
|---|---|---|---|---|---|
| **WP0** | Gate 修复 + 选择指标对齐 | 无 | ~7 min（重跑 E1） | 基线可信；placeholder 可判定 | 低 |
| **WP1** | 原子概率校准 + 期望分数决策 | WP0 | 1–2h（E6/E7 离线） | **最大**：直接优化 66.7% 占位行 | 中（需防过拟合） |
| **WP2** | 序列原子分类器 + 非联合原子优化 | WP1 | 5–10h | 大：把 q_atom 从行级弱信号升级为序列强信号 | 中 |
| **WP3** | 多架构 × 多种子集成 + 先融合后硬切换 | WP1/2 | 20–40h | 大且稳：通常 +0.5–1.5 | 中（算力/同源性） |
| **WP4** | PERM 不对称优化 | WP1 | 2–5h | 中：官方公式对高估更严厉 | 低 |
| **WP5** | Transductive / 自训练 | WP2/3 | 5–15h | 中–大：分布适配 | **高（规则/泄漏）** |
| **WP6** | 二阶 stacking / 树集成 | WP3 | 2–6h | 中–大（若规则允许非 DL） | 中（选择偏差） |
| **WP7** | 自监督预训练 / 外部数据 | WP2 | 10–30h | 中：小数据表示学习 | 高（收益不确定） |

---

## 2. WP0：Gate 修复 + 选择指标对齐（立即做）

### 2.1 把 placeholder 纳入 Gate
- `src/validation/gates.py`：
  - `MIN_ABSOLUTE_KEYS += ("min_placeholder_acc",)`
  - `METRIC_RESULT_FIELDS["min_placeholder_acc"] = ("placeholder_min_acc",)`
- `E1/code/train_row.py`：
  - prereg `thresholds` 增加 `"min_placeholder_acc": 0.98`
  - `result` 增加 `"placeholder_min_acc": float(min_ph)`
  - 旧的 `placeholder_ok` 保留为报告字段，但 Gate 判定以阈值为主
- `E1/P1/PLAN.md` 的 JSON 块同步；`tools/sync_prereg_templates.py` 重新生成模板。
- 回归：`tests/test_e1_pipeline.py` / `tests/test_gates.py` 增加“0.97 必须 FAIL”的断言。

### 2.2 早停/选 epoch 用 gated 代理分
- `src/training/fold_runner.py::inner_eval` 改为：
  ```python
  ev = M.evaluate_predictions(y_in, m_in, pred, y_atom=a_in, tau=0.5)
  return ev["gated"]          # 固定 τ=0.5 的 gated 代理分
  ```
  - `best_epoch` 由 gated 代理分选；最终 τ 仍在训练结束后在 inner-val 上单独搜（τ 搜索不受影响）。
  - 日志把 `innerOOF` 标记为 `innerOOF_gated_proxy@0.5`；同时记录 `cont_total` 供对比。
- 回归：E1 smoke 下 `best_epoch` 不应再由连续分决定；增加一个“gated proxy 确实被调用”的单元测试。

### 2.3 验收
- 重跑 E1：
  - `placeholder_min_acc ≥ 0.98`
  - `oof_total ≥ 78.0`
  - `folds_all_same_direction = true`
  - `min_same_direction_folds = 5`
- 不满足 → 不进 E3。

---

## 3. WP1：原子概率校准 + 期望分数决策

### 3.1 目标
把“逐目标全局 τ 硬切换”升级为“**校准后的 q_atom + 逐行/逐桶期望分数动作选择**”。

### 3.2 新模块
- `src/inference/calibration.py`（纯 numpy）
  - `fit_temperature(logits, y, mask)`：最小化 BCE，返回 `T`；`apply_temperature(logits, T)`。
  - `fit_isotonic_pav(scores, y, mask)` / `apply_isotonic`：PAV 单调校准（可选）。
  - `reliability_report(q, y, mask, n_bins)`：ECE / Brier / 分箱可靠性，写 Gate 收据。
- `src/inference/atom_decision.py`（纯 numpy）
  - `ExpectedScoreDecision`：
    - `fit(cont, q_atom, y, mask, score_fns, n_bins=20, min_bin=200, shrink=0.1)`
      - 对每个目标按**校准后 q** 分桶（等频/等宽）；
      - 桶内比较 `predict=ATOM_VALUE` 与 `predict=cont` 的官方平均分；
      - 用 Beta/二项收缩 + **单调约束**（q 越大越倾向 atom）防止小桶过拟合；
      - 输出每桶 `action`（atom/continuous）、`gain`、`n`；
    - `predict(cont, q_atom)`：按桶动作替换，返回 `(pred, actions)`；
    - `to_dict/from_dict`：可写入 `versions/configs/decode_v1.json`。
  - `decision_from_cost_matrix(q, cost_atom, cost_cont)`：当有代价矩阵时的闭式决策（可选）。
- `src/inference/decode.py`：
  - `DecodeConfig` 增加 `atom_calibration: dict`、`action_table: dict[str, list[dict]]`。
  - `apply_atom_decision(cont, q_atom, config)`：先温度校准，再按 action table 决策；无 table 时回退 τ。
- `predict.py` / `E9/confirm_check.py`：
  - 折集成仍“先融合 cont+q_atom”，然后应用 `atom_calibration + action_table`（若存在），否则回退 `tau_fused`。

### 3.3 离线搜索
- `E7/code/decode_search.py`：
  - 新增 `--atom-calibration temperature|isotonic|none`、`--decision-table`、`--decision-bins`。
  - 在内折 OOF 上按折交叉拟合（fold-wise cross-fitting）避免用同一批数据既校准又评估；
  - 报告 `objective`、每目标 ECE、action table、与全局 τ 的 paired delta/CI。
- 只有 `delta > 0` 且 `ci_low > 0` 才采纳 action table；否则保留 τ。

### 3.4 Gate
- `min_atom_acc ≥ 0.99`、`min_atom_recall ≥ 0.98`、`min_placeholder_acc ≥ 0.98`、`oof_total_min` 按阶段。
- 新增 mandatory：`atom_calibration_reported`、`decision_table_inner_only`、`no_atom_continuous_interpolation`。

---

## 4. WP2：序列原子分类器 + 非联合原子优化

### 4.1 目标
把 q_atom 从“行级 MLP 的附带头”升级为**序列主干上的专用分类器**，重点抓：
- 非联合原子行（q_joint 漏掉的 ~31k SW 原子行）；
- 边界负例（真实 SW 95–99.9、POR 0.05–0.2）。

### 4.2 代码
- `src/models/atom_head.py`
  - `AtomClassifierHead(d_in, hidden, layers, dropout)`：逐目标 3 个 logit，支持从序列隐状态 `(B,L,d)` 或行级 `(B,d)` 输入；
  - `build_atom_head(...)`；与 `UNet1D`/`TCN`/`PatchTF` 的 `forward_states` 对接。
- `src/losses/atom_classifier.py`
  - `focal_bce_with_logits(logits, targets, gamma, pos_weight, mask, nonjoint_weight)`
  - `per_target_atom_weights(atom_rates, joint_rates, alpha_nonjoint)`
  - `boundary_hard_negative_weight(cont, y, atom_value, delta)`
  - `atom_calibration_loss(q, y, mask)`：可选 BCE + 分箱校准正则。
- `E3/code/train_atom.py`（新入口）
  - 冻结 E3/E4 主干 → 取隐状态 → 训练专用原子头（可两阶段：先焦点 BCE，再校准/阈值）；
  - 产出 `E3_atom_report.json` + `atom_head_fold{k}.pt` + `inner_oof_atom.npz`；
  - 逐目标 AUC/AP/Acc/P/R/F1、非联合/联合分项、ECE。
- `E6/code/train_state.py`：
  - 允许从 `--atom-head-ckpt` 初始化 q_atom 头；
  - stage2 仍冻结原子头；τ/decision 在 inner-OOF 单独选。

### 4.3 Gate
- `min_atom_auc ≥ 0.97`（逐目标）、`min_nonjoint_atom_recall ≥ 0.95`、`min_placeholder_acc ≥ 0.98`；
- 序列原子头必须 ≥ 同协议行级原子头（paired CI 下界 > 0）。

---

## 5. WP3：多架构 × 多种子集成 + 先融合后硬切换

### 5.1 目标
用**多样性**换稳定分数：RowMLP + U-Net + TCN + PatchTF + MMoE + IndependentHeads，多种子、多折；先融合 cont/q_atom，再统一 decision/tau。

### 5.2 代码
- `src/ensemble/fusion.py`（或扩展 `blend.py`）
  - `fuse_members(members, weights, calibrations)`：连续头按权重、q_atom 按权重、校准后再融合；
  - `select_weights_inner(inner_members, y, mask, ...)`：单纯形/非负最小二乘 + 正则；
  - `correlation_prune(members, threshold)`：同源成员剔除；
  - `paired_well_bootstrap`：已有 `blend.paired_bootstrap_delta`。
- `E8/code/ensemble.py`：
  - `--members` / `--inner-members` 支持多架构；
  - 融合后统一 `atom_decision` / `tau_fused`；
  - 产出 `E8_ensemble_report.json`（同源性、权重、CI、adopted）。
- `E8/code/train_mmoe.py`：`--save-dir` 已接线，补齐候选注册（可选）。
- 多种子入口：`run_train.sh` 增加 `--seed-list` 或 `E3/E4/E6` 的 `--seeds`，每 seed 落不同 tag。

### 5.3 Gate
- `min_ensemble_gain ≥ 0`、`ci_low > 0`、`homology_reported`、`weights_from_inner_only`；
- 集成后必须重新跑 `min_placeholder_acc` / `no_interpolation`。

---

## 6. WP4：PERM 不对称优化

### 6.1 依据
官方 `Acc_PERM = max(0, 1 − |log10(max(ŷ/y, ε))|)`：
- **低估**被 `max(ŷ/y, ε)` 截断：`d = log10(ε)` 后误差不再增长（最多 3）；
- **高估**无上界：`ŷ/y = 10^6` 时误差 6 → 0 分。
因此 PERM 在不确定时应**偏向低估/收缩高值尾部**。

### 6.2 代码
- `src/losses/score_aligned.py`
  - `align_score_log(..., over_weight=1.0, under_weight=1.0)`：`ell = smooth_abs(d) * (over_weight if d>0 else under_weight)`；
  - `aux_loss` 增加 `perm_over_weight`；
  - `TrainConfig` 增加 `perm_over_weight`、`perm_aux_over_weight`（默认 1.0，E7 消融选）。
- `E7/code/ablate_loss.py`：新增 exp8（PERM 不对称）臂，写进 `loss_v1.json`。
- `E10/full_retrain` / E6/E8 消费 recipe 时同步。

### 6.3 Gate
- 在 inner-OOF 上 `perm_acc` 提升且 `paired_ci_low > 0`；
- 监控 `perm_low_tail` 与 `frac_abs_dz_lt_1` 不退化。

---

## 7. WP5：Transductive / 自训练

### 7.1 目标
利用测试输入分布（**无测试标签**）做校准/伪标签/一致性正则，提升跨井泛化。

### 7.2 代码
- `src/training/self_training.py`
  - `select_pseudo_labels(cont, q_atom, calibrated_q, thresholds, atom_values, per_well=True)`；
  - `well_quantile_align`（复用 E8/pseudo_label 已有逻辑，抽成公共件）；
  - `consistency_weights`（q 熵 / 置信度）；
  - 硬规则：测试侧输入 dict 含 `y_true/mask/targets` 等键 → 抛 `PermissionError`。
- `E8/code/pseudo_label.py`：接入自训练轮次（`--self-train-rounds`）、每轮确认折评估、标签键守卫。
- `E9/code/leakage_audit.py`：增加“transductive 只使用输入分布”的证据项。

### 7.3 Gate
- `no_high_risk_leak`、`confirm_no_breakdown`；
- 自训练每轮必须在 inner-OOF + 确认折上有正增益，否则回退。

---

## 8. WP6：二阶 stacking / 树集成

### 8.1 代码
- `src/ensemble/stacking.py`
  - `RidgeStacker`（numpy 闭式解，强制非负 + 归一化）；
  - `LogisticStacker`（numpy 梯度下降，用于原子概率融合）；
  - `HistGBStacker`（可选 `sklearn.ensemble.HistGradientBoosting*`，缺失时显式降级）。
  - 输入特征：各成员 OOF 的 `por/perm_z/sw/q_atom/q_joint` + 深度/井级统计 + E2/F2 子集。
  - 只用 inner-OOF 拟合，outer-OOF 评估；嵌套交叉拟合防止选择偏差。
- `E9/code/aggregate_oof.py`：把 stacking 作为候选融合策略之一；报告 paired CI。

### 8.2 规则检查
- 若比赛规则只允许纯深度学习/禁止二阶模型，本 WP 标记 `not_allowed`，不进入提交候选。

---

## 9. WP7：自监督预训练 / 外部数据

### 9.1 代码
- `src/training/ssl.py`
  - `mask_curve_batch(x, mask, rng, mask_ratio)`：随机掩码曲线片段；
  - `masked_reconstruction_loss(model_out, x, mask)`；
  - `contrastive_well_loss(embeddings, well_ids)`（可选）。
- `E3/code/pretrain_seq.py`（新入口）
  - 在全部 90 井输入曲线（含测试输入，**无标签**）上做 masked curve modeling；
  - 保存 `ssl_backbone.pt` 作为 E3/E4 初始化。
- 外部数据：只在规则允许时接入；单独数据卡与泄漏审计，默认关闭。

### 9.2 Gate
- 预训练后 E3 的 paired CI 下界 > 0，且确认折无 breakdown；否则丢弃。

---

## 10. 执行顺序与预算

| 顺序 | 任务 | 预算 | 验收 |
|---|---|---:|---|
| 0 | WP0 Gate/选择修复 + 重跑 E1 | ~0.5h | placeholder ≥0.98 且 OOF ≥78 |
| 1 | WP1 校准 + 期望分数决策（离线 E7） | 1–2h | paired delta >0 & ci_low>0 |
| 2 | WP2 序列原子头（E3 主干 + E6 两阶段） | 5–10h | min_atom_auc≥0.97，nonjoint recall≥0.95 |
| 3 | WP3 多种子多架构集成 | 20–40h | ensemble CI >0，placeholder≥0.98 |
| 4 | WP4 PERM 不对称 | 2–5h | perm_acc CI >0 |
| 5 | WP5 transductive/自训练 | 5–15h | confirm 无 breakdown + 泄漏审计通过 |
| 6 | WP6 stacking/树集成（规则允许时） | 2–6h | stacking CI >0 |
| 7 | WP7 SSL/外部数据（可选） | 10–30h | E3 CI >0 |
| 8 | E9 复验 + E10 打包提交 | 2–5h | 全部 Gate 通过 |

**原则：每一层没有 paired CI 正增益就回退；E7 之后所有配方冻结；E9 之后不再改模型。**

---

## 11. 风险与红线

1. **测试标签泄漏**：绝对禁止；所有测试侧输入必须经过 `label_like_keys` 守卫。
2. **用确认折/A 榜反馈调参**：禁止；确认折只用于最终复验。
3. **τ/decision/权重选择**：只能来自 inner-OOF；outer 折只能推理一次。
4. **小数据过拟合选择**：所有离线搜索用嵌套交叉拟合 + bootstrap CI + 多重比较校正。
5. **提交包必须自包含、CPU 可推理、确定性**：每次 E10 都要干净目录复现。

---

## 12. 交付纪律（每次任务完成必须执行）

1. 更新本文档 / README / status / 相关 PLAN；
2. `git add -A && git commit -m "..."`；
3. `git push origin HEAD:master HEAD:main`（若云端有任务在跑，先推 feature 分支，任务结束后再合并）；
4. `python3 tools/pack_code_zip.py` 打包到 `/mnt/d/tmp/Petroleum-Competition/`；
5. 在报告中记录 revision、zip sha256、实验结论。

---

## 13. 参考 SPWLA 2021 后的新增工作包（WP8–WP11）

> 详细复盘见 [`SPWLA2021_REVIEW.md`](SPWLA2021_REVIEW.md)。
> 结论：**数据适配 > 模型结构**；冠军用“类型井选择 + 井间自适应 + 线性/KNN”，
> 第 2/4/5 名用 MICE/KS/特征工程/GBDT/Stacking。

| WP | 名称 | 代码落点 | 预计算力 | 预期收益 |
|---|---|---|---|---|
| **WP8** | 类型井选择 + 井间输入分布匹配 | `src/data/type_well.py`、`src/data/well_adapt.py` | 3–8h | 高 |
| **WP9** | MICE/KNN 插补 + 异常权重 + KS 代表 inner split | `src/data/impute.py`、`src/data/outliers.py`、`src/validation/representative.py` | 2–6h | 高 |
| **WP10** | 岩石物理特征扩展（Sw/Vsh/φD/Klogh） | `src/features/physics_ext.py` | 2–4h | 中–高 |
| **WP11** | 链式目标 + GBDT + Stacking | `src/training/chained.py`、`src/models/gbdt.py` | 5–15h | 高 |

**与 WP0–WP7 的关系**：WP8/WP9/WP10 先做（便宜且是数据层），WP11 在 WP3 的集成框架上加入 GBDT/链式成员；WP1 的原子校准仍然优先，因为 66.7% 占位行是 SPWLA 没有的特殊结构。

## 14. 参考项目贡献与引用（必读）

本计划的 WP8–WP11 来自对 SPWLA PDDA SIG 2021 竞赛公开方案的复盘，参考仓库：
<https://github.com/pddasig/Machine-Learning-Competition-2021>；赛后论文：
Fu et al., *Well-Log-Based Reservoir Property Estimation With Machine Learning: A Contest Summary*,
*Petrophysics* 65(01), 108–127, 2024。

**参考项目的主要贡献**：
- 公开了多井测井 ML 竞赛的完整数据、任务与榜单；
- 冠军 UTFE 证明**类型井选择 + 井间自适应**比模型复杂度更关键；
- 亚军 MoLPhy 展示了 MICE、KS 代表采样、测试输入分布匹配与 SuperLearner；
- 第 4 名 Atwah 展示了系统性岩石物理特征工程；
- 论文总结了“数据/适配 > 模型”的可复用结论。

**我们的使用边界**：只参考方法与工程思路；未复制 Volve 数据、标签、代码或固定阈值。
v4 的数据、标签、官方指标与 66.7% 原子占位结构不同，所有参数必须重新在 v4 的
inner-OOF 上拟合。详见 [`SPWLA2021_REVIEW.md`](SPWLA2021_REVIEW.md)。


## 15. start.sh 统一入口

云端只启动一个命令：

```bash
bash "$(find /code/workspace -name start.sh | head -1)" --to all
```

或用参数限制范围：

```bash
start.sh --to E3-main
start.sh --stage E8 --target all
start.sh --wp data-quality
start.sh --wp atom-decision
start.sh --wp gbdt
start.sh --e1-rerun --to E2
```

`start.sh` 自动补前置、复用 `run_train.sh` 的断点续跑与 `/data` mirror；
全部可选值见 `start.sh --list`。

## 16. 第二轮审查修复记要

- **P0**：`placeholder_min_acc` 已改为官方软 Acc（`acc_relative` / `acc_perm` 逐行均值）；
  `SCORE_MAX_PLAN` 里记录的 `0.9723` 是旧“容差带命中率”口径，**必须重跑 E1 复算**。
- **P1**：`expected_value_table` 只接受高 q 后缀且起点 >0；`decode_search` 为 expected_value
  增加 paired CI/gain 门槛，杜绝 τ=0 导致整列强制原子值。
- **P1**：`MatrixImputer` 的 KNN/MICE transform 现在使用 fit 阶段保存的模型，不再静默返回 NaN。
- **P2**：`assert_atom_priority` 现在检查实际管线输出；E6/E8 支持 `--resume`。
- **WP8**：新增 `E8/code/type_well_member.py`，完成“类型井选择 + 井间输入适配”的一阶成员。
- 详细清单见 [`AUDIT_FIXES.md`](AUDIT_FIXES.md)。
