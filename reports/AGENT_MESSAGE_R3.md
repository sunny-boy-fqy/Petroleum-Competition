# 给 AGENT 的三审返工消息（可直接复制）

请先阅读 `v4/reports/V4_PLAN_REVIEW_3.md`（三审报告）和
`v4/reports/V4_PLAN_IMPROVEMENT_PROPOSAL.md`（计划改进建议），然后一次性完成两类工作：

## 一、修完三审提出的问题

1. **清干净 SW 旧口径残留**：全仓删除“SW<1 仅 11 行”“SW 有效值 [0,1]”“SW 双尺度”等错误/冲突表述，统一为“SW 是单一标签尺度（百分数，实测有效 8.305–99.9，SW<1=0）”；修改 `docs/gen_plans.py`，避免重新生成时回退。
2. **修计划行数同步**：当前 `PLAN.md:55-56` 写阶段 711 / P 4,982，实测 708 / 4,988。修 `tools/sync_plan_stats.py` 的正则，使 `PLAN.md`、`README.md`、`docs/PROJECT_FILES.md`、`versions/status.json` 的阶段/P/总量全部由 `tools/plan_stats.py` 一致生成；把 `plan_stats --check` 纳入 `check_status.py`。
3. **修 E0 Gate 计数与证据路径**：`E0_local_contract_gate.json` 当前 12/12，但 status/README/E0/PLAN 仍写 10/10，必须统一；`E0_data_card.json::shard_cache.cache_root` 不能是 `/tmp/v4cache_final`，必须使用 `$V4_CACHE_ROOT`（云端 `/data/v4/cache`）；status P1 的 evidence 补 `$V4_CACHE_ROOT/manifest.json`。
4. **修 Gate 语义**：`gates.aggregate_gate` 必须区分 `min_*`（>=）与 `max_*`（<=）；为 `state_auc`、`atomic_f1`、`por_acc/perm_acc/sw_acc`、`cpu_inference_ok`、`clean_dir_reproduce`、`a_board_no_breakdown` 等定义明确的 result 字段映射；实测 E9/P2 的 `max_degradation=0.1`、degradation=0.01 必须判 pass，E10 的 `max_minutes/max_memory_gb/max_point_diff` 必须真正生效。为 E9_P2/E10_P0/E10_P1 增加单测。
5. **修磁盘预算路径**：`run_train.sh` 与 `E0/code/setup_deps.sh` 的 `disk_guard` 必须显式检查 `$DATA_ROOT`，`E0_disk_budget.json::level` 必须对应 `/data` 配额，不能默认检查 `/`；`E0_cloud_gate` 的 `disk_budget_ok` 只取该文件。
6. **修状态一致性**：E0 cloud gate 通过前，`versions/status.json` 的 `project.phase` 应为 E0，E1 stage 应为 `pending`；`check_status.py` 增加 stage/P 状态一致性、Gate 报告内容一致性检查。
7. **修其他小项**：`check_env` fallback 折文件名应为 `v1_well_folds.json`；full profile 下 onnx/onnxruntime 等具备降级路径的可选依赖改为 warn + degraded_paths；`run_e0` 回拷全部 `reports/E0_*.json` 到 repo 或在文档中明确持久化来源。

## 二、把改进 proposal 真正落进计划与实现

按 `V4_PLAN_IMPROVEMENT_PROPOSAL.md` 的 §9 修改清单执行：

1. **原子建模层次**：
   - `PLAN.md §5.1/§5.3/§6.4`、`E6/P0/P1/P2` 从“H0 联合头”升级为 `q_joint + q_por + q_perm + q_sw + 连续头`；
   - 每目标独立硬切换 `τ_t`，`τ_t` 在 inner-OOF 上用官方总分期望增量选择，不按 F1；
   - `joint_guard` 默认关闭，只有 inner-OOF Total 正增益且 CI 下界 >0 才启用；
   - 输出契约与 Gate 增加逐目标 atomic Acc/Precision/Recall/F1、τ_t 曲线、误判代价。
2. **输出参数化**：
   - POR 改为可到 0 的参数化（`por_max*sigmoid(g)` 或 `softplus(g)-softplus(g0)`），初始化按有效 POR 分位数；
   - SW 连续头使用训练折内 `mu/sigma` 仿射归一化，输出反变换到标签尺度；原子分支精确输出 99.9；
   - 修改 `src/models/row_mlp.py`、`E5/P0/P2`、`src/features/basic.py` 与 manifest/scaler 落盘。
3. **损失修正**：
   - `masked_mean` 已改，继续保持；
   - `align_score_log` 已加 `max(zhat-z, log10(eps))`，继续保持；
   - `L_aux` 改为按目标尺度归一化，避免 SW 99.9 主导；增加 boundary-focus 权重作为 E7 消融；
   - 更新 `E7/P0/P1` 的消融表与 Gate。
4. **训练策略**：
   - E6/P0 增加两阶段训练：先原子头，再冻结/慢更新原子头训连续头；
   - E8/P2 增加 EMA/SWA 与同折 top-k 快照集成，权重只在 inner-OOF 选，报告同源相关性；
   - 非 joint 单目标原子行用 per-target sample weight，不删除 joint 行。
5. **序列主干优化**：
   - E3/E4 加入 U-Net padding 伪影检查、TCN 非因果、PatchTF overlap/边界推理、门控/FiLM 融合、连续分支平滑（不跨原子边界）、井级小容量强正则。
6. **验收**：
   - `tests/run_all.py` 全绿；
   - `tools/check_data_leak.py` 90 井 0 违规；
   - `tools/plan_stats.py --check PLAN.md` 无 WARN；
   - `tools/gen_data_card_md.py --check` OK；
   - `gates` 对 E9_P2/E10 模板的 max_* 语义测试通过；
   - PLAN/README/E0/status 全部同步；
   - proposal 中列出的文件全部有对应修改和证据。

完成后请附上：修改文件清单、关键 diff/公式、所有验收命令输出摘要、以及仍未闭环的 blocked 项。
