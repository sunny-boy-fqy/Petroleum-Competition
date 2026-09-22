# 第二轮审查修复清单（含 WP8 完成）

> 对应审查报告：`placeholder_min_acc` 口径、`expected_value_table` τ=0 事故、
> `MatrixImputer` 静默不插补、`assert_atom_priority` 恒真、`--resume` 覆盖、
> P3 小问题，以及 WP8 的完整落地。

## 1. P0：`placeholder_min_acc` 改为官方软 Acc

**问题**：`src/training/metrics.py::_hit_rate` 旧实现按 `err <= 1`（进入容差带边缘）
统计，而文档/计划写的是官方软准确率；`y=0.1, pred=0.105` 会被算成 1.0，官方
`acc_relative` 其实约 0.3812。

**修复**：

- `_hit_rate` 改为调用官方 `acc_relative` / `acc_perm`，返回逐行软分均值；
- 旧的“容差带命中率”拆到 `atomic_rows_report["tolerance_hit_rate"]`，只作诊断；
- `atomic_precision_recall` 的阈值比较统一为 `q > tau`（与硬切换一致）；
- 新增回归：`y=0.1, pred=0.105` 的 POR 软 Acc < 0.5，绝不能算 1.0。

**影响**：E1 旧记录 `placeholder_min_acc=0.9723` 是旧宽松口径；新口径需要**重跑 E1**
重新计算。`E1/P1/PLAN.md` 的 Gate 仍是 `min_placeholder_acc=0.98`，但现在必须用真 Acc 判定。

## 2. P1：`expected_value_table` 单调后缀 + 增益门槛

**问题**：旧实现只要求 atom 箱连续，低 q 箱全为 atom 时也能返回 `tau=0.0`；
`predict.py` 会把该目标所有行强制切成原子值；且 expected_value 未加 CI/增益门槛。

**修复**：

- `src/inference/decode.py::expected_value_table`：
  - atom 动作必须是**高 q 侧后缀**：`atoms == range(atoms[0], n_bins)`；
  - 且 `atoms[0] > 0`，否则 `monotone=False, tau_from_table=None`；
  - 若所有箱都是 atom（从 0 开始），直接拒绝，杜绝 τ=0。
- `E7/code/decode_search.py`：
  - expected_value 采纳前计算相对当前 τ 基线的 per-target paired CI 与 gain；
  - 只有 `gain > min_gain` 且 `ci_low > 0` 才写 `expected_value=True / tau_from_table`；
  - 报告新增 `expected_value_gate`。
- 新增回归：低 q 原子表必须被拒绝；高 q 后缀表才能给出正 τ。

## 3. P1：`MatrixImputer` 的 KNN/MICE transform 不再静默返回 NaN

**问题**：`MatrixImputer.fit` 对 KNN/MICE 没有保存 `fill_values`，
`transform` 直接返回原始 NaN。

**修复**：

- 新增 `KNNImputerModel`：fit 保存训练折参考行与列中位数，transform 逐行 KNN 插补；
- `MatrixImputer` 对 KNN/MICE 保存 `_model`，`transform` 必须调用模型；
- MICE 有 sklearn 时保存 `IterativeImputer`，缺失 sklearn 时显式降级 KNN，
  绝不静默返回 NaN；
- 新增 `test_matrix_imputer_knn_transform_no_nan` / `..._mice_transform_no_nan`。

## 4. P2：`assert_atom_priority` 改为检查实际管线输出

**问题**：旧实现分别对 `cont_before/cont_after` 重新硬切换，命中行永远相同，
`atom_priority_preserved` 形同虚设。

**修复**：

- 新增可选参数 `out_actual`：
  - 命中行（q > tau）的实际输出必须逐位等于原子值；
  - 未命中行的实际输出必须逐位等于 `cont_after`；
  - 两者任一不满足 -> `ok=False`。
- `E7/code/decode_search.py` 现在传入真实 `final_pred`；
- 新增“污染一个原子行必须被发现”的回归测试。

## 5. P2：`--resume` 覆盖补全

- **E6**：`run_train.sh` 现在把 `--resume` 只传给 `train_state.py`，不污染
  `search_tau.py` / `build_pd1.py`；`--mode all` 任务 10 也传 `--resume`。
- **E8**：`E8/code/train_mmoe.py` 新增 `--resume`，若
  `<save-dir>/<arm>/fold{k}.pt` 存在则加载权重并跳过该折训练；
  `run_train.sh` E8 只把 `--resume` 传给 `train_mmoe`，其他 E8 子目标不受影响；
  `--mode all` 任务 12 也传 `--resume`。
- **E4**：原本已支持 `--resume`，`run_train.sh` 任务 8 已透传。
- **E5**：目前没有 checkpoint 保存/恢复机制，文档明确标注为“不可折内续训”；
  由于其只训练冻结主干上的小头，重跑成本相对低。
- **E10 full_retrain**：`final_train.py` 已支持 `--resume`；通过
  `start.sh --stage E10 --phase final --aggregate full_retrain` 时可显式传 `--resume`。

## 6. P2：交付卫生

并发写入产生的 `start.sh`、WP8/WP9/WP10/WP11 新文件、测试与文档已全部
`git add -A` 并 commit/push；本轮结束再重新跑全量测试与一致性工具。

## 7. P3：小问题

1. **阈值比较**：`atomic_precision_recall` 改为 `q > tau`，与硬切换一致。
2. **SW 守卫**：`validate_payload` 现在始终 `bounds.get("SW") or (0,100)`，
   部分 `row_scales` 不能绕过 SW 量纲检查；新增回归测试。
3. **boundary_focus 序列兼容**：`_focus` 改用 `atom[..., t]`，同时支持
   `(B,3)` 与 `(B,L,3)`；新增序列模式测试。
4. **Ascend 真机未实测**：仍是最大未量化风险；必须在长训前跑
   `start.sh --stage env`、`start.sh --stage smoke`（或 `--to e0`）与最小 NPU
   前向/保存/加载冒烟。

## 8. WP8 完成情况（向后兼容）

- `src/data/type_well.py`：KL/DTW/签名类型井选择；
- `src/data/well_adapt.py`：线性/分位井间输入匹配；
- `E8/code/type_well_report.py`：类型井报告；
- **新增** `E8/code/type_well_member.py`：逐验证井
  “选类型井 → 在类型井子集上训练 GBDT/链式 → 验证输入适配 → 预测”的一阶成员；
  产出 E8 集成兼容 OOF；默认独立入口，不改变任何既有训练路径；
- `start.sh --wp type-well` / `--wp type-well-adapt` 可分别启动报告和适配成员；
- `tests/test_wp_reports.py` 已端到端跑通合成 cache 下的适配成员。

## 9. 复现与验证命令

```bash
# 全量测试
.venv-torch/bin/python tests/run_all.py
../v2/.venv/bin/python tests/run_all.py test_gbdt test_impute test_wp_reports

# 一致性
python3 tools/check_consistency.py
python3 tools/check_status.py
python3 tools/check_pipeline.py
python3 tools/plan_stats.py --check
python3 tools/sync_prereg_templates.py --check
python3 tools/audit_goal.py
python3 tools/final_acceptance.py --quick

# 云端真机最小冒烟（Ascend）
bash "$(find /code/workspace -name start.sh | head -1)" --stage env
bash "$(find /code/workspace -name start.sh | head -1)" --stage smoke
```
