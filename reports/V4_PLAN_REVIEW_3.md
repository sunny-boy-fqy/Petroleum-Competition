# v4 三审报告

- 审查对象：二审返工提交（`ca4e89c`）及其后的平台提交（`bf2ed47`）后的当前工作树。
- 审查方式：当前提交不依赖 GPU/torch 的可复算项全部实跑；数值部分使用 `../v2/.venv`。
- 三审结论：**二审的大部分关键修复真实生效，但仍有若干会误导实现/导致 Gate 误判的问题；并且《V4_PLAN_IMPROVEMENT_PROPOSAL.md》尚未落进计划与模型实现。**

---

## 0. 二审返工已验证生效的部分

| 二审项 | 当前证据 |
|---|---|
| R2-B1 云端启动顺序 | `run_train.sh` 已改为先 `setup_deps`，`check_env --profile base` 允许数据/依赖未就绪；`--mode data` 后做 `full` 校验 |
| R2-B3 缓存接线 | `run_e0` 默认传 `--with-cache`；`run_stage --stage E0` 已实现；本地 Gate 含 cache 两项 |
| R2-B5 测试井 schema | `reports/E0_data_card.json::test.noncanonical_schema_wells` 已为 **0**；parse 的 `missing_cols` 按 split 计算 |
| R2-B6 提交契约 | `validate_payload` 已做每井行数、井数、SW 尺度守卫；`predict.py` 调用 `depth_alignment_report` |
| R2-H1 全量泄漏 | `run_all.py` 已内嵌 90 井全量泄漏检查；`tools/check_data_leak.py` 实跑 90 井 0 违规 |
| R2-H2 masked NaN | `masked_mean` 已用 `torch.nan_to_num` |
| R2-H3 PERM 截断 | `align_score_log` 已做 `d = max(zhat-z, log10(eps))` |
| R2-H4/H6 工具 | `gates.py` 增加 `gate_type`/boolean；`check_status.py`、`plan_stats.py`、`gen_data_card_md.py` 已存在 |
| 自动测试 | `tests/run_all.py` 实跑 **52 项全部通过** |
| 数据/计划工具 | `check_data_leak`、`check_consistency`、`check_status`、`gen_data_card_md --check` 均通过 |
| CONST 端到端 | `predict.py --use-version CONST` 10 井 / 95,948 行通过契约与深度对齐 |

---

## 1. 三审阻断项

### R3-C1. SW 尺度残留错误未清干净，且 PLAN 内部仍自相矛盾

**证据**

仍存在“SW<1 仅 11 行”的错误数字：

- `PLAN.md:26`
- `PLAN.md:37`
- `PLAN.md:389`
- `src/constants.py:47`

但当前事实源明确为 **0 行**：

- `reports/E0_data_card.json::target_stats.SW.valid_rows_only.n_lt_1 = 0`
- `README.md:142`、`E0/P1/PLAN.md:21`、`E5/P2/PLAN.md:20` 已改为 0。

仍存在“SW 有效值域 [0,1]”的旧表述：

- `PLAN.md:124`：仍写 `SW 占位 99.9 与有效值域 [0,1] 相距极远，需双分支`
- `PLAN.md:666`：风险表仍写 `SW 尺度混淆（99.9 vs [0,1]）`
- `E0/PLAN.md:31`：仍写“同一列混用 99.9 与 [0,1] 两种尺度”
- `docs/gen_plans.py`：仍保留多段旧双尺度文本；如果运行该生成器会把错误写回
- `src/data/labels.py:107` 仍保留 `[0,1]` 旧分支说明（作为对照路径，需明确默认关闭）

**影响**：计划层面对 SW 的量纲仍有互相否定的描述，实现者可能再次按 [0,1] 建模。

**修复**：全仓统一为“SW 单一标签尺度，实测有效 8.305–99.9，SW<1 = 0”；历史对照路径明确标为 deprecated/default off；修改生成器，避免回退。

### R3-C2. 计划行数统计仍未同步，工具本身也漏检

**证据**

实测：

```text
PLAN.md            729 行
E*/PLAN.md         12 份 / 708 行（均 59）
E*/P*/PLAN.md      33 份 / 4,988 行（均 151）
合计               46 份 / 6,425 行
```

但 `PLAN.md:55-56` 仍写：

- 阶段计划：合计 **711**（实际 708）
- P 级计划：合计 **4,982**（实际 4,988）

`tools/plan_stats.py --check PLAN.md` 已明确警告：

```text
WARN: PLAN.md 未声明或声明不一致 -> {'阶段': 708, 'P 级': 4988}
```

`tools/sync_plan_stats.py` 也没有修复：

- 它的 `阶段计划` / `P 级子计划` 正则要求列尾紧接 `|`，而 `PLAN.md` 中该列后还有 `✅ 完成`，导致 pattern not found；
- 运行后只更新了总量，没有修正行 55/56。

此外 `tools/check_status.py` 只检查“总量 6,425”是否出现，不检查阶段/P 分量，因此这类错误不会被它拦截。

**影响**：计划完成度表仍不可信；R2-H6 未闭环。

**修复**：修正同步工具正则；让 `check_status.py` 同时校验 PLAN/README/PROJECT_FILES 的阶段/P 分项；把 `plan_stats --check` 纳入提交前检查。

### R3-C3. E0 本地 Gate 的文档计数与实物报告不一致

**证据**

- `reports/E0_local_contract_gate.json` 当前有 **12 项** mandatory checks：
  `... + shard_cache_built + shard_cache_input_cols_ok`，passed=true；
- `versions/status.json` 写 `mandatory_passed: 10`、`mandatory_total: 10`，summary 写 `10/10 PASS`；
- `README.md:132` 写“本地契约 Gate 10/10”；
- `E0/PLAN.md:8` 写“10/10 mandatory PASS”；
- `E0/docs/data_card.md:167` 写“mandatory 10/10”。

另外当前 tracked `reports/E0_data_card.json::shard_cache.cache_root = "/tmp/v4cache_final"`；
这是本机临时路径，不是文档声明的 `$V4_CACHE_ROOT` 或 `/data/v4/cache`，作为提交证据不可复现。

**影响**：Gate 数量口径不一致；cache 证据的持久化路径与计划不符。

**修复**：
1. 统一文档/status/PLAN 为“12/12（含 cache）”，或者明确区分“无 cache 10 项 / 含 cache 12 项”；
2. E0 证据中的 cache root 必须使用 `$V4_CACHE_ROOT` 相对/持久路径，不能用 `/tmp/...`；
3. status E0 P1 的 evidence 增加 `$V4_CACHE_ROOT/manifest.json`，并确保该文件在云端 /data 存在。

---

## 2. 三审高优先级问题

### R3-H1. Gate 的 max_* / 指标专属阈值语义仍然错误

**证据 1：max_* 被当成下界**

`src/validation/gates.py::aggregate_gate` 对 `ABSOLUTE_KEYS` 一律执行：

```python
ok = have is not None and have >= want
```

但其中包含：

- `max_hard_failures`（应为 `<=`）
- `max_point_diff`（应为 `<=`）
- `max_minutes`（应为 `<=`）
- `max_memory_gb`（应为 `<=`）
- `max_degradation`（应为 `<=`）

实测 E9/P2 模板：

```text
gate_type = delta
thresholds = {min_delta:0, max_degradation:0.1}
传入 degradation=0.01（本应 pass，因为 0.01 <= 0.1）
aggregate_gate 返回 passed=False，reason:
  absolute_checks.max_degradation: required=0.1, have=0.01, ok=False
```

即 `max_degradation` 被错误地当作“必须 ≥0.1”。

**证据 2：指标专属绝对阈值没有独立结果字段**

- E6/P0：`primary_metric=state_auc`，阈值 `min_auc=0.97`；
- E6/P1：`primary_metric=atomic_f1`，阈值 `min_atomic_acc=0.99`；
- E10/P0：`max_minutes=30`、`max_memory_gb=8`；
- E10/P1：`max_point_diff=1e-6`。

当前 `aggregate_gate` 在 delta 分支只接受 `result["score"]` 或 `result["oof_total"]` 作为绝对分数来源，
没有 `auc`、`atomic_acc`、`minutes`、`memory_gb`、`point_diff` 等字段映射。
E10/P0/P1 的 `cpu_inference_ok`/`clean_dir_reproduce` 会被推断为 boolean，
max_* 阈值直接不参与判定。

**影响**：E6/E9/E10 等 Gate 可能在真实结果上误判或漏判。

**修复**：
1. 将绝对值键拆分为 `min_*` 与 `max_*` 两组，分别用 `>=`/`<=`；
2. 为每个指标定义 result 字段名映射（如 `state_auc -> result["auc"]`，`atomic_f1 -> result["atomic_f1"]`，`cpu_inference_ok -> result["minutes"]/["memory_gb"]`）；
3. 给这些模板补 `gate_type` 与真正的 mandatory checks；
4. 增加单测覆盖 E9/P2、E10/P0、E10/P1 三类模板。

### R3-H2. `E0_disk_budget.json` 仍检查 `/`，不是 `/data`

**证据**

- `run_train.sh:111-113`：`disk_guard.py --min-free-gb 8 --report "$HERE,$DATA_ROOT" --json "$REPORTS_DIR/E0_disk_budget.json"`，没有 `--path`，默认路径 `/`；
- `E0/code/setup_deps.sh:21` 与 `:74` 同样没有 `--path`；
- `disk_guard._main` 默认 `--path "/"`，所以 `E0_disk_budget.json::level` 是根文件系统的余量，不是 `/data` 的 30 GB 配额；
- `run_all.py` 的 cloud gate 直接读取该 JSON 的 `level` 作为 `disk_budget_ok`。

虽然 `check_env` 已经会对 `$DATA_ROOT` 做磁盘检查，因此 env 失败会阻止 gate，但 `disk_budget_ok` 这一项自身的语义仍然错误，且可能“根盘充足、/data 不足/相反”时给出错误信号。

**修复**：显式传 `--path "$DATA_ROOT"`（或分别检查）并让 `E0_disk_budget.json` 含 `paths`/每挂载点 level；`disk_budget_ok` 只取 `$DATA_ROOT` 的结果。

### R3-H3. 《V4_PLAN_IMPROVEMENT_PROPOSAL.md》没有落进计划/实现

**证据**

当前 `PLAN.md`、`E6/P0`、`src/models/row_mlp.py` 仍是 joint-only 结构：

- `PLAN.md:370`：`H0 联合常量状态头`；
- `E6/P0/PLAN.md:15`：训练联合占位状态分类头；
- `src/models/row_mlp.py`：只有 `head_ph`，没有 `q_por/q_perm/q_sw`；
- 没有 per-target atom head、没有 `τ_t` 期望分解码；
- `PLAN.md:373` 的 H3 仍是旧的 `q*99.9 + (1-q)*sigmoid(f)*100`；
- `row_mlp.py` 仍使用 `por = 0.1 + softplus(g)`，未做 POR/SW 参数化修正；
- `aux_loss` 仍是绝对 Smooth L1，无 SW fold 归一化/边界聚焦；
- 无 EMA/SWA/快照训练与两阶段训练计划。

**影响**：即使 E0 修复完成，进入 E6 后仍会按旧结构训练，无法获得逐目标原子保护、POR 低值表示、SW 稳定训练等主要增量。

**修复**：按 proposal §9 的文件清单，直接修改 `PLAN.md`、`E1/P1`、`E5/*`、`E6/*`、`E7/*`、`E8/*`、`src/models`、`src/losses`、`src/inference`、`src/training` 和预注册模板。

### R3-H4. `versions/status.json` 与阶段状态语义不一致

**证据**

- `project.phase = "E1"`；
- E0 stage status = `in_progress`（P0 blocked）；
- E1 stage status = `in_progress`，但 E1 的两个 P 都是 `pending`。

即：E0 未完成、E1 未开始，却同时把 E1 标为 in_progress，并让 project phase 指向 E1。

**修复**：E0 cloud gate 通过前，project.phase 应为 E0；E1 应为 `pending`。`check_status.py` 应增加“stage in_progress 时至少一个 P 非 pending”的一致性检查。

### R3-H5. `${REPORTS_DIR}` 的部分 E0 产物没有回拷到仓库，e0 后 git 证据可能不刷新

**证据**

`run_train.sh::run_e0` 只复制：

- `E0_local_contract_gate.json`
- `E0_cloud_gate.json`
- `E0_data_card.json`

没有复制：

- `E0_score_check.json`
- `E0_gate_prereg.json`
- `E0_contract_tests.json`
- `E0_folds.json`

云端持久化本身没问题（/data 保留），但仓库中的 E0 证据不会随云端复算刷新。
如果以后以 git 里的 `reports/` 作为审查依据，会出现数据卡与 score_check/prereg 不一致。

**修复**：把 `reports/E0_*.json` 全量从 `$REPORTS_DIR` 同步到 `$HERE/reports/`，并在文档中明确哪些以 `/data` 为准、哪些以 git 为准。

---

## 3. 三审中优先级问题/优化项

### R3-M1. `check_env` fallback 折文件名错误

`E0/code/check_env.py::check_repo` 在导入失败时回退到：

```python
folds = root / "versions" / "reference" / "well_folds.json"
```

实际文件名是 `v1_well_folds.json`。虽正常路径会走 `find_folds_file`，fallback 是错的。

### R3-M2. 可选依赖在 `profile=full` 仍然是 hard

`check_py_deps` 在 full 下把 `onnx` / `onnxruntime` 等缺失记为 hard。但这些包在计划中有 portability 降级路径；如果 `setup_deps.sh` 因为磁盘或平台网络失败没有装上，训练会被一个“可选兜底依赖”卡死。建议 full 下只对训练/推理主路径必需的依赖 hard，onnx/onnxruntime 等列为 warn 并写 `degraded_paths`。

### R3-M3. 自动测试覆盖仍有空档

`tests/` 52 项通过，但没有覆盖：

- `tools/plan_stats.py --check` 的行数分项一致性；
- `gates.aggregate_gate` 的 `max_*` 语义（R3-H1）；
- E9_P2 / E10_P0 / E10_P1 模板的真实聚合；
- `run_train.sh::run_env` 的 base/full profile 顺序；
- `E0_data_card` cache root 持久路径。

### R3-M4. `check_status.py` 漏检文档与报告的不一致

它只检查总行数，不检查：

- `PLAN.md` 的阶段/P 分项；
- `reports/E0_local_contract_gate.json` 的 checks 数量与 status 里的 `mandatory_passed/total` 是否一致；
- E0 P1 的 cache evidence 文件是否存在。

### R3-M5. `E0/P1/PLAN.md` 状态与 evidence 仍有小缺口

E0 P1 标 done，但 status 的 evidence 没有 `$V4_CACHE_ROOT/manifest.json`，而 P1 的 findings 说缓存已产出。建议补入 evidence，并区分“本地证据”和“云端持久证据”。

---

## 4. 三审验收命令

代理在修改后至少应跑通：

```bash
# 1) 口径层与契约
python3 v4/tests/run_all.py
python3 v4/tools/check_data_leak.py --data ../data
python3 v4/tools/check_consistency.py
python3 v4/tools/check_status.py
python3 v4/tools/plan_stats.py --check PLAN.md
python3 v4/tools/gen_data_card_md.py --check
python3 v4/predict.py --use-version CONST --data_dir ../data --output /tmp/r.json

# 2) E0 一键复算（含 cache）
V4_CACHE_ROOT=/tmp/v4cache R3 python3 v4/E0/code/run_all.py --with-cache \
  --cache-root /tmp/v4cache --out /tmp/E0_data_card.json

# 3) 计划修改 proposal 落地后的静态检查
grep -R "SW<1 仅 11\|SW<1.*11\|有效值域 \[0,1\]" PLAN.md E0 README.md src docs
grep -R "q_atom\|q_por\|q_perm\|q_sw" PLAN.md E6 src
```

验收标准：

- [ ] `plan_stats --check PLAN.md` 无 WARN；
- [ ] PLAN/README/E0 文档与 `E0_local_contract_gate.json` 的 gate 项数一致；
- [ ] 全仓 SW 只有“单一标签尺度、SW<1=0”一种表述；
- [ ] `gates.aggregate_gate` 对 E9_P2/E10 模板语义正确，max_* 不被当上下界；
- [ ] E0 disk budget 明确来自 `$DATA_ROOT`；
- [ ] status project.phase 与 E0/E1 stage 状态一致；
- [ ] proposal 中的逐目标原子头、POR/SW 参数化、loss 修正、两阶段/EMA/快照进入 PLAN 与 P 级计划。

---

## 5. 结论

二审修复的工程基础（泄漏、契约、Gate 类型、测试、缓存接线、数据卡脚本）总体是有效的；
但三审仍发现：

1. SW 旧表述和错误数字仍有残留；
2. 计划行数分项未同步，sync/check 工具本身有漏洞；
3. Gate 的 max_* 与指标专属阈值语义仍有错误；
4. 磁盘预算检查仍指向 `/`；
5. 状态/证据计数不一致；
6. **最重要的模型改进 proposal 尚未落到 PLAN 和代码。**

因此，三审状态为：**E0 工程层接近可交付，但模型计划尚未按改进 proposal 升级，Gate 语义仍需修复；修复完成前不应进入 E1/E6 训练。**
