# v4 代码审查发现处置状态

> **原始报告**：[`../../docs/CODE_REVIEW.md`](../../docs/CODE_REVIEW.md)（项目工作区根目录，v4 仓库外）。
> **原始审查基线**：commit `3fa65da`（2026-09-21）。
> **本状态页代码快照**：commit `fafaf7c`（2026-09-24）。
> **本次复核**：`./.venv-torch/bin/python tests/run_all.py` → Ran 892 tests, OK（skipped=3）。
> **用途**：避免把历史审查报告中的问题当成“当前仍全部存在”，也避免在没有证据时宣布“已全部修复”。
>
> 本页只记录代码/测试可复核的处置状态；详细背景与复现命令仍以原始报告为准。

---

## 0. 结论摘要

原始审查共记录：**1 个严重、7 个高、8 个中、6 个低**问题。后续提交
`6b2a580`（修复 C1/H1–H7 与 M 项）、`5faf4e1`、`a3921f6`、`fafaf7c` 等已修复绝大多数问题。

当前仍应特别关注的边界：

1. **M5**：E6 原子阈值 `min_atom_acc=0.99`、`min_atom_recall=0.98` 未调整，实机可达性未证明。
2. **H5 残余**：`E10/code/submit.py` 仍有 3 项硬编码 Gate check（`atomic_precision_reported`、
   `checkpoint_resumable`、`no_label_leak`），且 `training_time_log_valid` 仍只查文件存在；
   需要统一改为从 `src/validation/evidence.py` 读取。
3. **L1**：数据缓存仍使用 `np.load(..., allow_pickle=True)`；需要显式安全边界或改为 `allow_pickle=False`。
4. **L3**：项目根 `rules.md` 的旧“须过滤”表述与 §5.3 项目决策冲突；该文件在 v4 仓库外，需在项目级维护时同步。

---

## 1. 逐条状态

| ID | 原问题摘要 | 状态 | 当前证据/说明 |
|---|---|---|---|
| C1 | 提交包不自包含，官方默认推理命令失败 | ✅ 已修复 | `E10/code/build_submission.py` 生成包内 `versions/registry.json`，权重写相对路径；`tests/test_e10_package.py` 覆盖 |
| H1 | 折集成因 `row_scaler.fit_wells` 整字典比较而默认拒绝 | ✅ 已修复 | `E6/code/build_pd1.py`、`E10/code/final_train.py` 改为容许多折标尺；`tests/test_e10_final_train.py`、`tests/test_predict_pd1.py` |
| H2 | E2 物理特征缓存绕过折内 `phys_params`，造成分布泄漏 | ✅ 已修复 | `src/data/row_dataset.py::_well_feature_matrix` 对含 `phys` 的 spec 强制使用折内参数；`tests/test_features_e2.py` |
| H3 | `--resume` / checkpoint 承诺不成立 | ✅ 已修复 | 每 epoch 写 `last.pt`，保存实际 epoch，恢复 scheduler，seq 路径接入 resume，磁盘 hook 接入；`tests/test_training_core.py`、`tests/test_e1_pipeline.py`、`tests/test_seq_pipeline.py` |
| H4 | E6 delta Gate `paired_ci_low=0.0` 恒不过；原子指标误用 `y>=0.5` | ✅ 已修复 | `E6/code/search_tau.py` 使用 `inner_oof.y_atom` 计算原子指标与真实配对 CI；`tests/test_e6_tau.py`、`tests/test_atomic_gate.py` |
| H5 | E9/E10 多个 mandatory check 硬编码/恒真 | 🟡 部分修复 | E9 已统一走 `src/validation/evidence.py`；`tests/test_evidence.py`、`tests/test_e9_leakage.py`。**残余**：`E10/code/submit.py` 仍硬编码 3 项 True，`training_time_log_valid` 仍只查文件存在 |
| H6 | 候选/版本注册表写在临时目录，跨任务丢失 | ✅ 已修复 | `run_train.sh` 将 candidates/registry/state 放到本地 runtime，并随最终产物 publish；`V4_CANDIDATES`/`V4_REGISTRY` 可覆盖 |
| H7 | `--mode all` 名实不符、E6/P2 未接线 | ✅ 已修复 | `run_train.sh --mode all` 已串行 14 个任务，支持 `--through` 断点续跑；E6 `p0/p1/p2/all` 已接线 |
| M1 | E9 对 E1/E3 OOF 取错预测列 | ✅ 已修复 | `E9/code/aggregate_oof.py` 统一 OOF schema 读取规则；`tests/test_e9_aggregate.py` |
| M2 | `atom_rates` 生产路径未按折拟合 | ✅ 已修复 | `fit_target_scalers` 从折内 `perm_z` 反变换 PERM 统计原子率；不再依赖 `y_perm=None` 的默认回落 |
| M3 | 提交契约不校验 POR/PERM 范围 | ✅ 已修复 | `src/inference/contract.py` 增加 POR/PERM/SW 物理边界与 SW 尺度守卫；`tests/test_contract.py` |
| M4 | 提交包含 `train.py` 但阶段脚本没打包 | ✅ 已修复 | `build_submission.py` 增加 `E*/code` 白名单并校验训练入口可运行 |
| M5 | E6 原子阈值可能不可达，且与指标定义不匹配 | 🔴 未修复/待实机验证 | `E6/code/train_state.py` 与 `E6/code/search_tau.py` 仍写 `min_atom_acc=0.99`、`min_atom_recall=0.98`；云端未跑，不能确认可达 |
| M6 | 磁盘守卫 hook/自动清理未接线 | ✅ 已修复 | `fold_runner.py`/`seq_loop.py` 注册 `last_prev.pt` 与 cache tmp，传入 `capacity_hook`；`tests/test_disk_guard.py`、`tests/test_training_core.py` |
| M7 | `feature_report` 对 F2 列名越界/失真 | ✅ 已修复 | `src/data/row_dataset.py::feature_report` 从 scaler/spec 取列名，仅在 F1 宽度时用 `FEATURE_NAMES` |
| M8 | `bootstrap_ci` 对 `n=0` 未防护 | ✅ 已修复 | `src/validation/folds.py::bootstrap_ci` 显式抛 `ValueError` |
| L1 | `allow_pickle=True` | 🔴 未修复 | `src/data/dataset.py`、`src/data/row_dataset.py`、`src/features/groups.py` 仍有 `np.load(..., allow_pickle=True)`；需记录可信输入边界或改列名读取 |
| L2 | `tools/audit_goal.py` 只查存在性 | 🟡 文档缓解 | 工具语义未变；README/本状态页已避免把“文件存在”写成“验收通过” |
| L3 | 文档数字漂移 | 🟡 部分修复 | v4 侧已同步（含 `dataset.py` 显存口径）；项目根 `rules.md:254` 旧“须过滤”与 §5.3 决策冲突，该文件在 v4 仓库外，需项目级维护修复 |
| L4 | `--mode all` 名实不符 | ✅ 已修复 | 同 H7 |
| L5 | `last_prev.pt` 永不生效 | ✅ 已修复 | `save_last` 在写新 `last.pt` 前调用 `CK.rotate()`，备份上一版 |
| L6 | 多仓库结构未说清 | 🟡 文档已解释 | v4 README/docs 已声明 `v4/` 是独立 git 仓库根；项目工作区根 README 属于上层治理 |

---

## 2. 提交前必须复查

```bash
cd v4

# 全量本地回归
./.venv-torch/bin/python tests/run_all.py

# 文档/计划/状态一致性
python3 tools/check_status.py
python3 tools/plan_stats.py --check

# 关键交付链回归（可单独跑）
./.venv-torch/bin/python -m unittest tests.test_e10_package tests.test_e10_final_train \
    tests.test_e6_tau tests.test_evidence tests.test_features_e2 tests.test_training_core
```

> 任何一次文档或代码变更后，如无法运行上述命令，必须在交付说明里写明“未验证”和原因，
> 不得默认标记为通过。
