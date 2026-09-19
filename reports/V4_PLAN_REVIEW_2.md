# v4 二审报告：代码/计划硬逻辑 + 实现逻辑与疏漏

- 审查范围：二审后的所有计划、E0 实现代码、Gate 校验器、预注册模板、版本注册表、
  run_train 编排、数据卡与复算证据。
- 审查方式：静态代码审查 + 逐项实跑复算（本机无 torch，数值部分用 `../v2/.venv`）。
- 二审结论：**一审修复有真实成果，但仍不批准进入 E1。**
  发现 5 个新的执行/门禁阻断项、若干口径残留错误，以及多处可优化但会实际影响训练/提交的实现疏漏。

---

## 0. 一审修复中已经真实生效的部分

| 项 | 证据 |
|---|---|
| B1 输入泄漏 | `parse.py` 已改为 13 曲线 + DEPTH 分离；`tools/check_data_leak.py --data ../data` 实跑 **90 井 / 0 违规**；train/test cache 输入列均为 13 |
| B3 非规范井 | `reports/E0_data_card.json`、`dist/v4_data_manifest.json` 均已正确记录：17,426 行多列 + 9,654 行缺 CASE |
| B4 双 Gate | `E0_local_contract_gate` 与 `E0_cloud_gate` 已拆分；status E0 已改为 `in_progress` |
| B5 候选分数 | `versions/candidates.json` 已按实测回填；`tools/check_consistency.py` 通过 |
| H4 部分 | `check_env.py` 已支持 `V4_DATA_ROOT`、CUDA 12.6 断言、硬失败非零退出 |
| H5 部分 | `disk_guard` 在 cleanup 后仍低于 min_gb 时会报错，`allow_soft` 才接受风险 |
| H1 部分 | 33 份 P 计划新增 §12.5，`fold0+1` 仅保留为资源预检说明，步骤/参数表大多已改为 inner-OOF |
| H2 部分 | `src/validation/gates.py` 已存在，33 份预注册模板可被 `validate_prereg` 做字段级通过 |

---

## 1. 新的执行/门禁阻断项

### R2-B1. `run_train.sh --mode env/all` 首次必失败，云端 Gate 无法自然闭环

**证据**

- `run_train.sh:72-92`：`run_env` 的执行顺序是
  1. `check_env.py`（硬失败即 `exit 11`）
  2. `setup_deps.sh`
- 但 `check_env.py` 把“缺可选依赖”和“缺 `/data/v4/data` 数据”都判为 hard：
  - `check_py_deps` 对 `einops/onnx/onnxruntime/scipy/sklearn/pyarrow` 缺失在云端是 hard；
  - `check_repo` 要求 `$V4_DATA_ROOT/v4/data/train` 有 80 井、`test` 有 10 井，否则 hard fail。
- 官方文档/README 的任务顺序是 **env → data → e0**；因此第一次 `--mode env` 时：
  - 数据尚未部署 → `data_train_80`/`data_test_10` 必然失败；
  - 可选依赖尚未安装 → 更多 hard failure；
  - `run_env` 在调用 `setup_deps` 之前就退出，`--mode all` 永远到不了 `--mode data`。
- 即使显式设 `V4_ALLOW_ENV_FAILURE=1` 让流程继续，`setup_deps.sh` 成功后的最终 `check_env.py`
  仍写向**临时 repo** 的 `reports/E0_env.json`，而不是 `$V4_REPORTS_DIR/E0_env.json`；
  `run_all.py` 的 cloud gate 读取 `/data/v4/reports/E0_env.json`，看到的仍是第一次失败的报告。

**影响**：按文档操作时 P0 无法通过，`E0_cloud_gate` 无法变为 passed，整个云端流程卡死。

**修复建议**
1. `run_env` 先安装依赖，再跑持久化路径的 `check_env`；或把依赖检查降为 warn，安装后追加硬断言。
2. 把数据/折文件检查从“环境自检”拆出，作为 `--mode data` 后的独立 `data_health` 检查。
3. 明确启动顺序：首次要么 `data → env → e0`，要么 `env` 允许“数据未就绪”的 non-hard 状态。
4. `setup_deps.sh` 一律写 `$V4_REPORTS_DIR`，并同时保留 `cloud_frozen.txt` 到持久目录。

### R2-B2. Gate 校验器/模板不能执行计划中的绝对硬门槛

**证据**

- `src/validation/gates.py:124-129` 的 `effective_threshold` 只取
  `thresholds[primary_threshold_key]`；所有 P 模板都写 `primary_threshold_key="min_delta"`。
- 实测：E3/P2 模板的 `oof_total_min=81.0` 被完全忽略；
  用 `delta=0.0, paired_ci_low=0.1` 调 `aggregate_gate`，结果是 **passed=true**。
- E0/P0 是确定性检查（`env_hard_checks_passed`），但 `aggregate_gate` 仍强制要求 `paired_ci_low`；
  实测传入无 CI 的结果会 `passed=false`，即使所有 mandatory checks 为真。
- 33 份模板虽然字段齐全，但 `baseline_version`、`baseline_artifact`、`baseline_manifest_sha256`
  大量为 `<已冻结候选或 CONST>`、`<sha256>` 占位符；`validate_prereg` 只查字段存在，不查内容，
  因此“33 份全部通过”只是浅层通过。
- 实际 `reports/E0_gate_prereg.json` **没有通过新校验器**：
  缺 `p_stage`、`baseline_manifest_sha256`，且 `planned_task_training_h=0.0` 也不满足 `>0`。

**影响**：Gate 会在未达到硬门槛时被判 passed；确定性 Gate 会被错误判 failed。
预注册发布物与计划中的阈值语义脱节。

**修复建议**
1. 建立 `primary_metric -> primary_threshold_key` 的显式映射，或让 `effective_threshold` 同时考虑
   `oof_total_min`、`min_auc`、`abs_tolerance`、`min_atomic_acc` 等绝对阈值。
2. 区分 `metric_kind = delta / absolute / boolean / deterministic`，boolean 类型不要求 paired CI。
3. `validate_prereg` 增加非占位、可解析、`gate_id` 与 stage/p_stage 一致、基线 sha 非空等检查。
4. 修复 `run_all.py` 写出的 E0 预注册，或明确 E0 的 supersede 机制不参与通用 validator。

### R2-B3. B6 缓存仍未进入官方执行路径，E1 实际会缺输入

**证据**

- `E0/P1/PLAN.md` 第 5 步要求 `run_all.py --with-cache`；
- `run_train.sh:100-105` 的 `run_e0` 只调用 `run_all.py --train-dir ... --test-dir ... --out ...`，
  **没有 `--with-cache`**；
- `run_train.sh:118-127` 的 `run_stage` 只有 E1/E3，没有 E0；而 E0/P0/P1/P2/P3 的复算命令却写
  `run_train.sh --mode stage --stage E0`，实际会走 `*)` 报“尚未实现”。
- tracked 证据 `reports/E0_data_card.json::shard_cache = {"built": false}`；
  `reports/E0_local_contract_gate.json` 只有 10 项 mandatory，不含 cache 两项；
  status/P1 却写“缓存 32.4 MB，90 井输入列校验通过”。

**影响**：官方 `--mode e0` 后 E1/P0 找不到分片，直接阻塞；P1 的完成状态与 tracked 证据不一致。

**修复建议**
1. `run_e0` 默认传 `--with-cache`，并把 `manifest.json`、缓存体积、输入列校验写入 E0 报告。
2. 实现 `run_stage --stage E0`，或删除 E0 P 计划中的该命令。
3. E0 本地 Gate 只有在 `shard_cache_built && shard_cache_input_cols_ok` 为真时才 pass；
   否则 P1 不能标记 done。

### R2-B4. SW 修复未闭环：口径残留 + 数据卡文档数字错误

**证据**

仍然在现行代码/计划中写 SW 有效值 `[0,1]` 或“双尺度”：

- `src/data/labels.py:9`
- `src/models/row_mlp.py:9`
- `src/features/basic.py:112`
- `E0/PLAN.md:34`
- `PLAN.md:113`
- `PLAN.md:655`
- `docs/gate_template.md` §4.5 仍要求“显式声明 SW 的双尺度处理”
- `PLAN.md:32` 的 E0-R1 历史块仍写“20/21 列缺 CASE”，与已修正的 E0-R2 冲突。

更关键的是“SW<1 仅 11 行”这个修复数字本身是错的：

- `reports/E0_data_card.json::target_stats.SW.valid_rows_only.n_lt_1 = 0`
- 全量有效 SW：min 8.305、median 82.805、max 99.9，没有 SW<1 的有效行。
- 但 `PLAN.md:26/378`、`README.md:142`、`E0/P1/PLAN.md:21`、`src/constants.py:47`、
  `docs/gen_p_details.py:93` 都写“SW<1 仅 11 行”。

`E0/docs/data_card.md` 的统计表也与 JSON/实测不一致：

| 项 | 文档写法 | 实测 |
|---|---:|---:|
| POR `<1` | 587 | **576** |
| POR p01 | 1.738 | **1.742** |
| PERM `<1` | 122,727 | **122,716** |
| PERM min | 0.000 | **0.01** |
| PERM p99 | 54.993 | **54.9964** |
| SW `<1` | 11 | **0** |
| SW 直方图 | `[5,0,...,236337]` | `[5,228,1578,4271,7805,12980,24818,49173,60651,74822]` |

另外 `E0/P2/PLAN.md:58` 仍写期望逐目标
`POR 0.6736 / PERM 0.7467 / SW 0.7421`，而实际为
`0.6735824 / 0.7072023 / 0.7294624`。

**影响**：文档和代码注释会继续误导 SW 归一化/裁剪；数据卡“实证”不可信。

**修复建议**
1. 以 `reports/E0_data_card.json` 为唯一事实源，用脚本生成 `E0/docs/data_card.md` 的统计表。
2. 全仓统一 SW 为“单一标签尺度（百分数）”，只保留“禁止全局裁剪到 [0,1]”的正确表述。
3. 把 SW<1 的错误数字全部改为 0；更新生成器，避免重新生成时回退。
4. 更新 E0/P2 的期望值与实际值一致。

### R2-B5. `parse_well` 对测试井错误地报告“缺 POR/PERM/SW”，数据卡把 10 口测试井都列为非规范 schema

**证据**

- `parse_well` 对 `with_targets=False` 仍用 `wanted = list(C.COLUMNS)` 计算 `missing_cols`，
  于是测试井的 `missing_columns = ("POR","PERM","SW")`。
- `reports/E0_data_card.json::test.noncanonical_schema_wells` 当前长度为 **10**，
  每口测试井的 `missing_columns` 都写着 `["POR","PERM","SW"]`。
- 测试井本来就没有标签列，这不是“schema 非规范”。

**影响**：数据卡测试集 schema 诊断失真；下游若按 `missing_columns` 判断输入列缺失会被误导。

**修复建议**
- `missing_cols` 只针对该 split 的期望列计算：
  - `with_targets=True`：`C.COLUMNS`
  - `with_targets=False`：`DEPTH + C.INPUT_COLUMNS`
- 加单测：测试井 `missing_columns == []`（除非真有输入曲线缺失），训练集异常井恰好 3 口。

### R2-B6. 提交契约没有实现自己声明的深度对齐、SW 尺度、10 井硬校验

**证据**

- `src/inference/contract.py:11` 声明要检查“depth 与输入深度对齐”，但
  `validate_payload` 只检查深度严格递增；真正的 `depth_alignment_report` 从未被
  `predict.py` 调用（`predict.py` 只调用 `validate_payload`）。
- `depth_alignment_report` 只在输入文件存在时可用；当前主路径没有调用。
- `validate_payload` 对 `test_dir` 井数不符只 `warnings.append`，不置 `ok=False`；
  `predict.py` 的 `--expected-wells` 参数根本没有传给契约校验。
- 契约不检查每个井的预测行数是否等于输入行数；只检查总行数 95,948 和每井深度递增。
- 契约没有 SW 尺度保护：一个把 SW 预测成 0.4（但真实标签在 8.3–99.9）的结果可以顺利通过。
  文档中“禁止裁剪到 [0,1]”没有对应的自动检查。

**影响**：格式契约无法防止最常见的零分风险：深度错位、井数/井行数错、SW 量纲错。

**修复建议**
1. `predict.py` 在 `validate_payload` 通过后，再调用 `depth_alignment_report`，任何井
   `ok=False` 即非零退出。
2. 契约中把井数、每井行数、深度逐行对齐作为硬错误；`expected_wells` 正式传入。
3. 增加标签尺度分布守卫：例如 SW 预测中位数若 <1.0 直接判失败（或至少 hard warning 后 fail），
   并可选做 SW 输出范围 `[0,100]` 的软告警。

---

## 2. 高优先级实现逻辑问题

### R2-H1. 全量 90 井泄漏回归没有进入 Gate，只做了 18 井抽样

- `E0/code/run_all.py` 的 `input_leak_regression` 只遍历 `tr_records[:12] + te_records[:6]`，
  `scope` 明确写 sample；
- 真正覆盖 90 井的 `tools/check_data_leak.py` 是独立脚本，E0 local Gate 的
  `input_no_label_leak` 只绑定抽样结果。
- 修复：把全量 leak 检查作为 `run_all.py` 的强制步骤（或让 Gate 直接调用该工具的库函数）。

### R2-H2. `masked_mean` 会让 masked NaN 污染整个 loss

- `src/losses/score_aligned.py:55-60`：
  `(x * m).sum() / m.sum()`；如果 `x` 在 `m=0` 的行是 NaN，`NaN*0 = NaN`，
  该 batch loss 直接变 NaN。
- 当前标签缺测主要是 `-99999` 有限值，不会触发；但解析器允许 NaN 标签，且未来特征/标签变换
  可能产生 NaN。用 NaN mask 本应屏蔽，却反而污染训练。
- 修复：`x = torch.where(m > 0, x, torch.zeros_like(x))` 后再求和，或
  `torch.nan_to_num(x) * m`。

### R2-H3. `align_score_log` 与官方 PERM 评分并不严格同构

- 官方：`1 - |log10(max(ŷ/y, ε))|`，对极端低估会在 `log10(ε)=-3` 处截断。
- 当前：`1 - |ẑ - z|`，没有对 `max(ratio, ε)` 的截断。
- 当 `ŷ/y < 1e-3` 时，当前 loss 使用的误差大于官方误差，梯度鼓励的方向与官方评分不完全一致。
- 修复：在 log 空间实现 `d = zhat - z; d = max(d, log10(eps)); s = max(0, 1-|d|)` 的平滑版本，
  或者对 ratio 先 `clamp_min(eps)` 再取 log。

### R2-H4. gate 判定器对 boolean/deterministic Gate 不适用

- `E0_P0_gate`、`E1_P0_gate`、`E10_P0_gate` 等主指标不是 delta/CI，而是 `_ok` 布尔值；
- `aggregate_gate` 对所有非 E9 Gate 统一要求 `paired_ci_low > 0`，会直接把这类 Gate 判 false；
- 实测 `E0_P0_gate` 在 mandatory 全绿、delta=0、ci_low=None 时 passed=false。
- 修复：增加 `gate_type = delta / absolute / boolean`；boolean 只查 mandatory checks，
  不要求 delta/CI。

### R2-H5. 缺少自动化测试；计划中的单测没有落地

- 全仓无 `tests/`、`test_*.py`、`*_test.py`。
- `E0/P2` 要求“4 个边界单测”，`E5/P2` 要求 SW 尺度单测，当前只有 `run_all.py` 内联 selftest
  和几个工具脚本，没有可回归的 pytest/unittest 套件。
- 修复：至少新增
  `tests/test_parse.py`（列布局/测试缺列/畸形井/哨兵）、
  `tests/test_score.py`（边界/两种 missing_mode）、
  `tests/test_contract.py`（深度对齐/井数/行数/SW 尺度）、
  `tests/test_gates.py`（metric 类型/阈值）。
  纯口径测试不需要 torch。

### R2-H6. 计划完成度/状态/证据仍不一致

- 实测行数：`PLAN.md` 718；12 阶段计划合计 **711**；33 P 计划合计 **4,941**；
  总计 **6,370** 行。
- 文档仍写：阶段 1,412、P 9,880、总计 12,010/12,022；`docs/PROJECT_FILES.md` 甚至仍有 661/224。
- `versions/status.json::project.phase = "E1"`，同时 E0 stage 为 `in_progress`、E1 stage 也为
  `in_progress`；阶段语义混乱。
- status P1 标记 done 并引用 cache，但 tracked E0 证据中 `shard_cache.built = false`。
- 修复：加 `tools/check_plan_stats.py` 或扩展 `check_status.py`，把行数、阶段状态、
  P 证据文件存在性做成可失败校验；统一 README/PLAN/PROJECT_FILES 的生成源。

---

## 3. 中优先级优化/疏漏

### R2-M1. `RowMLP` 的 POR 参数化下界仍锁在 0.1

- `por = 0.1 + softplus(g)` 永远 ≥0.1；实测有效 POR 有 576 行 <1、186 行 <0.1，
  其中不少是 0.0。模型无法表示这些真实值。
- H0 只解决“联合占位行”，不能解决“非占位但 POR 很小”的行。
- 建议：改成 `por = softplus(g) - softplus(b0)` 类型的有下界但可到 0 的参数化，
  或先用 `sigmoid` 映射到数据范围；初始化应使输出接近有效 POR 的中位数而不是 0.1。

### R2-M2. SW 头初始化和标签尺度可能需要显式归一化

- `RowMLP.head_sw` 是线性头，bias 初始为 0；有效 SW 为 8.3–99.9，早期输出在 0 附近，
  相对误差损失会非常大。
- 建议：训练时对 SW 做固定仿射归一化（如 `(sw-mean)/std`）并在输出层反变换；或初始化
  bias 到训练集中位数，并明确是否对 [0,100] 做物理裁剪。

### R2-M3. PERM/occupancy 的 BCE 与分类头初始化未审查

- H0 的 `head_ph` bias 初始为 0，占位率 66.7%，BCE 可训练但前期会在 0.5 附近；
  可考虑 bias 初始化为占位先验 logit。
- 当前 `RowMLP` 没有 H3 双分支输出，E5/P2 的 H3 结构尚未实现；接口需要对后继实现
  明确 `sw` 是标签尺度还是归一化后的值，避免再次出现量纲错误。

### R2-M4. `run_train.sh --mode all` 会吞掉训练阶段失败

- `run_train.sh:141` 使用 `run_stage || true`，所以 E1 未实现或训练失败时 `--mode all`
  仍可能以成功状态结束。
- 建议：显式区分“仅前置检查的 all”和“完整训练 all”；或把 `|| true` 去掉并让失败可见。

### R2-M5. 磁盘门禁路径不正确

- `run_train.sh:87` 的 disk_guard 未传 `--path`，默认检查 `/`；`setup_deps.sh` 也用
  `shutil.disk_usage('/')`。但 30 GB 配额很可能在 `/data` 挂载点。
- 修复：对 `$DATA_ROOT` 和 `/code/workspace` 分别检查、分别写入 `E0_disk_budget.json`；
  `run_env` 至少检查 `$DATA_ROOT`。

### R2-M6. `setup_deps.sh` 与 P0 输出契约不一致

- `cloud_frozen.txt` 写向临时 repo，任务结束即丢；P0 要求它作为持久证据。
- `check_env.py` 的 `deps` 只有 8 个模块（不含 torch 本身），P0 要求“9 个可选依赖”。
- 依赖安装使用了范围约束（`pandas>=2.0,<3` 等），而 lock 文件是精确版本；二者不一定一致。
- 建议：写 lock 到 `$V4_REPORTS_DIR`，用 lock 中的精确版本安装，明确 torch 单独由镜像提供。

### R2-M7. `tools/check_consistency.py`/`check_status.py` 校验仍偏浅

- `check_consistency` 允许 `cv.missing_mode = None`，而注册表规则要求必须是 `drop`；
  也没有校验候选字段 schema、`result_zip` 存在性。
- `check_status` 不校验 gate report 文件是否存在、stage status 与 P status 是否自洽、
  计划行数与文档统计是否一致。
- 建议补强，使这两个工具真正成为提交前门禁。

### R2-M8. 文档与生成器仍会回退错误结论

- `docs/gen_p_details.py:93`、`docs/gen_plans.py` 仍包含旧的 SW<1=11/双尺度文本；
  重新生成 P 计划会把错误写回。
- `PLAN.md:32` 的 E0-R1 历史块没有标注 superseded，容易被误读为现行契约；
  建议明确写“本段已被 E0-R2 取代”。
- `E0/P3/PLAN.md` 仍写“`versions/candidates.json` 被 `predict.py` 读取”，
  实际 `predict.py` 已改为读 `versions/registry.json`。

### R2-M9. `run_train.sh` 的 `--mode e0` / P0 命令与文档不一致

- E0/P0/P1/P2/P3 的计划命令都是 `--mode stage --stage E0`，但脚本没有 E0 case；
- README 又说 `all` 串行全部，实际只到 E1 且可能失败；
- 修复命令行契约，或者在脚本中增加 E0/E2/E4–E11 的显式 case。

---

## 4. 建议的修复顺序

1. **R2-B1**：修 `run_train.sh` 的 env/data/deps 顺序与持久化路径；让云端 Gate 能自然通过。
2. **R2-B3**：`run_e0` 默认建缓存；补 `stage E0` 或统一命令；让 P1 证据与 tracked 状态一致。
3. **R2-B2 + R2-H4**：修 Gate 校验器的 metric/threshold 语义，boolean Gate 不要求 CI；
   修复 E0 实际预注册并重跑 33 份模板的语义校验。
4. **R2-B5**：修 `parse_well` 测试集 `missing_columns`；补测试；重跑 E0 数据卡。
5. **R2-B4 + R2-H6**：用数据卡 JSON 生成文档统计；统一 SW<1=0、单尺度表述；
   修正 P2 期望值和所有行数声明。
6. **R2-B6 + R2-H1 + R2-H2 + R2-H3**：契约深度对齐/尺度守卫、全量泄漏 Gate、
   `masked_mean` NaN、PERM 对齐损失。
7. **R2-M1 ~ R2-M9**：训练/初始化/测试/工具优化。

---

## 5. 二审通过条件

- [ ] 首次 `bash run_train.sh --mode env`、`--mode data`、`--mode e0` 可按文档顺序通过，
      且 `E0_cloud_gate` 能在真实云端数据写入后变为 passed；
- [ ] `--mode e0` 默认生成 `$V4_CACHE_ROOT/manifest.json`，E0 local Gate 含 cache 两项；
- [ ] `gates.aggregate_gate` 能对 E3/P2 的 `delta=0` 判 false，对 E0/P0 的 boolean 检查正常判定；
- [ ] `reports/E0_gate_prereg.json` 通过 `validate_prereg`；
- [ ] `reports/E0_data_card.json::test.noncanonical_schema_wells` 为空；
- [ ] 全仓无“SW<1=11”“SW 双尺度 [0,1]”残留；文档统计与 JSON 逐项一致；
- [ ] 计划行数由脚本统计且与实际 `wc -l` 一致；
- [ ] 提交契约真正检查 10 井/每井行数/深度逐行对齐/SW 标签尺度；
- [ ] 新增至少 parse/score/contract/gates 四组自动测试。

> 结论：一审的 B1/B3/B5/H1/H2 修复是有效的，但二审发现的关键问题集中在
> **云端启动闭环、Gate 语义、E0 缓存路径、SW/数据卡残留和提交契约实际强度**。
> 这些问题修复前，v4 仍不应启动 E1。
