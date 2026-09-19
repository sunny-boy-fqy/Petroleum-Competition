# v4 总计划审查报告

- 审查对象：`v4/PLAN.md`、`E*/PLAN.md`、`E*/P*/PLAN.md`、`versions/status.json`、
  `versions/candidates.json`、E0 已实现代码与门禁产物。
- 审查方式：静态交叉核对 + 本机可复算项实跑（不依赖 GPU/torch；数值复算使用
  `../v2/.venv` 的 numpy）。
- 结论：**计划框架完整，但当前状态不能批准进入 E1 训练。** 存在 6 个必须先修复的
  阻断项（其中 2 个会直接导致标签泄漏/推理崩溃），以及若干高优先级口径与门禁问题。
- 本报告只记录审查结论；未修改 `PLAN.md`、E0 代码或既有 E0 产物。

---

## 0. 已通过的核验项

| 核验 | 结果 |
|---|---|
| 计划文件数量 | 1 总计划 + 12 阶段计划 + 33 P 级计划；33 个 P 计划均含 1–13 节 |
| Markdown 链接 | 55 个 md 文件内链全部存在 |
| 冻结折文件 | `tools/verify_reference.py` 输出 `RESULT: OK`；文件 sha256 `f7c2c58b...d94b87e` |
| 数据行数 | train 80 井 / 730,268 行；test 10 井 / 95,948 行 |
| 常数基线 | `drop` = **70.490735**（命中锚点）；`mask` = 69.843218 |
| 提交契约 | `predict.py --use-version CONST` 端到端 10 井 / 95,948 行，`contract_ok=true`，约 1.3 s CPU |
| 三状态计数 | 缺测 6,700 / 占位 487,225 / 有效 236,343 |

---

## 1. 阻断项（必须修复后才能开 E1）

### B1. 训练输入含标签泄漏，测试输入列数不一致（最高优先级）

**证据**

- `src/data/parse.py:147`：`inputs = arr[:, 1:15]`。
  对训练井（17 列重排后）这段是 `GR...CASE, POR`，即 **第 14 个输入特征就是 POR 标签**；
  对测试井（14 列）同样的切片只得到 13 列。
- 实测：训练分片 `inputs[:,13]` 与 `targets[:,0]`（POR）在非缺测行上逐点相等；
  测试分片 shape 为 `(n,13)`。
- `src/features/basic.py:58` 做 `X[:, 0:14] = inputs`；对测试 13 列直接抛
  `ValueError: could not broadcast input array from shape (n,13) into shape (n,14)`。

**影响**

- E1 会在“偷看 POR 标签”的条件下训练，E1 Gate ≥78.0 及后续所有序列对照全部失真；
- 最终 `predict.py` 在测试集上会因列数不匹配直接失败。

**必须动作**

1. 明确“14 列输入”的规范定义：
   - 方案 A：输入 = `arr[:, :14]`（DEPTH + 13 条曲线），训练/测试均 14 列；
   - 方案 B：输入 = `arr[:, 1:14]`（13 条曲线，DEPTH 单独用），并把全文的
     `F_raw(14)` 改为 13、相应调整 `FEATURE_NAMES`。
2. 统一 `constants.INPUT_COLUMNS`、`parse.py`、`dataset.py`、`features/basic.py` 的列语义。
3. 加回归单测：train/test 输入列数相同；输入列与任何目标列不相交；对 90 口井全部通过。
4. 重新生成/校验 cache，重跑 E0 与 E1 Gate。

### B2. SW“有效值域 [0,1]”与真实数据不符

**证据**

- `src/constants.py:45`：`SW_VALID_RANGE=(0.0,1.0)`；`PLAN.md:103/366`、
  `E0/docs/data_card.md`、`E5/P2/PLAN.md` 均声明 SW 有效值是 [0,1]、占位 99.9 是“双尺度”。
- 对 80 口训练井实测（非缺测且非联合占位行）：
  - SW min = **8.305**，max = **99.9**，median = **99.9**；
  - SW < 1 的行数为 **0**；SW < 10 的行数仅 5。
- `rules.md` 的 JSON 示例 `SW:0.426` 是模板示例；`v1.md` 与 v1 数据统计明确按训练标签
  百分数尺度（0–100）建模和提交。

**影响**

- “双峰相距上百个标准差”“有效分支必须 ×100”的诊断前提错误；
- 任何按该前提做的 SW 裁剪/归一化/监督都会引入额外偏差；
- E0 数据卡没有输出任何目标分布/范围统计，因此这个错误一直未被 Gate 发现。

**必须动作**

1. 数据卡增加三目标 min/max/median/分位数（按占位/有效切片分别统计）；
2. 将文档改为“标签尺度：POR/SW 均为训练标签原尺度（训练集中 SW 约 8.3–99.9，占位 99.9）；
   模型内部可用 sigmoid 归一化分支，但输出层必须换算回标签尺度”；
3. 增加单测：有效 SW 范围检查、占位分支精确 99.9、输出不被裁剪到 [0,1]；
4. 重新评估 H3/E5/P2 的监督与解码设计。

### B3. 非规范 schema 描述错误：20/21 列井并未缺 CASE

**证据**

- 实际表头：
  - `42f2870b` 20 列：`DEPTH,GR,K,U,CGR,PE,...,BIT,CASE,POR,PERM,SW`，**含 CASE**，多 K/U/CGR；
  - `b7eb1274` 21 列：在上一基础上多 TH，**含 CASE**；
  - `c7611b01` 16 列：**唯一真正缺 CASE 的井**，共 9,654 行。
- `reports/E0_data_card.json` 与 `dist/v4_data_manifest.json` 的 `missing_columns` 正确：
  前两口为 `[]`，只有第三口为 `["CASE"]`。
- 但 `PLAN.md`、`README.md`、`E0/docs/data_card.md`、`src/data/parse.py` docstring 均写成
  “20/21 列缺 CASE”，把 27,080 行全部描述为 CASE 缺失。

**影响**

- E0/P1 的“3 口畸形井 CASE 缺失补 NaN”验收条件无法按字面通过；
- 容易写出错误的 schema 单测或填补逻辑。

**必须动作**：修正所有文档表述；明确区分
“非规范列数共 27,080 行（17,426 行多列 + 9,654 行缺 CASE）”。

### B4. E0 Gate 与执行状态自相矛盾：P0 未过，E0 却标记 done/passed

**证据**

- `E0/PLAN.md` 的 Gate 完成判据要求 `check_env.py` hard 全过、`disk_budget_ok`；
- `E0/P0/PLAN.md` 状态为“待云端执行”；`versions/status.json` 中 E0/P0 为 `blocked`；
- 但 `reports/E0_gate.json` 的 mandatory 只含 6 个“本机口径”检查，
  不含 `env_ok`、`disk_budget_ok`、`training_time_log_valid`；
- `PLAN.md:458` 写“E0 状态：已完成（E0 Gate 6/6 PASS）”，同时 `status.json` 又写
  “P0 环境实测待云端”；
- E0/P1 的依赖写的是 `E0/P0（环境可用）`，但实际已完成于 P0 之前。

**影响**

- “E0 已完成”的对外结论不成立；后续 P 的 blocked/pending 状态会被错误解除；
- 环境/磁盘风险没有真实验收就进入训练。

**必须动作**

1. 将当前 Gate 明确改名为 `E0_local_contract_gate`，只声明本机口径层通过；
2. 新增 `E0_cloud_gate`，必须包含 `env_hard_checks_passed`、`disk_budget_ok`，
   在 P0 通过前 E0 阶段保持 `in_progress/blocked`；
3. 更新 `PLAN.md` 零之二表、README、`versions/status.json`；
4. 明确 E0/P1 可在无 GPU 时先做本地复算，但必须记录该例外。

### B5. 候选注册表的 CONST 逐目标分数错误，与总分为 70.490735 不自洽

**证据**

- `versions/candidates.json` 中 CONST：POR=0.67352、PERM=0.74666、SW=0.74210，total=70.490735；
- 按官方权重计算：`100×(0.30×0.67352 + 0.35×0.74666 + 0.35×0.74210)=72.3122`，
  不是 70.490735；
- E0 实测 drop 口径：POR=0.6735824、PERM=0.7072023、SW=0.7294624；
  `E0/P2/PLAN.md` 却把 PERM/SW 写成 0.7467/0.7421，超出其自述 ±0.002 容差。

**影响**

- “唯一事实源”候选注册表数据不可信；后续 Gate/汇总可能复现出与总分矛盾的逐目标值。

**必须动作**

1. 用 `src/score.py` 重算并回填 CONST 的逐目标值；
2. E0/P2 的完成判据改为可直接复算的精确值；
3. 增加一致性检查：`abs(100·Σw_t·acc_t − total) < 1e-6`。

### B6. raw/labels 分片缓存没有 P 级步骤负责生成

**证据**

- `E0/P1/PLAN.md` 输出契约包含
  `$V4_CACHE_ROOT/raw/<split>/<well>.npz` 与 `labels/<well>.npz`；
- `E1/P0/PLAN.md` 输入契约直接依赖这些分片；
- 但 `E0/code/run_all.py` 只解析数据卡/评分/折/契约，**不调用** `src/data/dataset.py::build_cache`；
- E1/P0 的步骤也只描述“实现 build_row_features/RowScaler”，没有“生成 raw/labels 分片”；
- 当前仓库 `cache/` 不存在。

**影响**

- E1/P0 按计划执行时会因缺少输入分片而阻塞，或临时补写脚本导致口径漂移。

**必须动作**：把 `build_cache` 明确写入 E0/P1（或 E1/P0 第 1 步），
产出并校验 `cache/manifest.json`（90 井、维度、行数、目标不进入输入）。

---

## 2. 高优先级问题

### H1. 内层验证协议与多个 P 计划冲突：outer 折被用于早停/选超参

- `PLAN.md §6.3` 要求“每个 outer 折内再切 3 个 inner 折；阈值、λ、早停、集成权重只在
  inner-OOF 上选；outer 折只推理一次”。
- 但计划中多处直接使用 outer 验证折：
  - `E1/P1/PLAN.md:42`：“每 epoch 在**验证折**上用真实 `score.py` 算分（早停依据）”；
  - `E1/P1/PLAN.md:52`：hidden 在 `fold0+1` 上选；
  - `E3/P2/PLAN.md:38/49/71`：感受野消融先在 `fold0+1` 筛查，胜者再跑全 5 折；
  - `E4/P0`、`E7/P0`、`E8/P0` 同样使用 `fold0+1` 选择。
- 这些做法会让最终 OOF 带选择偏差，且与各 P 自己“禁止用 outer 折选超参”的条款矛盾
  （例如 E1/P1 §8 禁止用 outer 折早停，但 §5 正是这么写的）。

**建议**：所有调参/早停/结构选择改为只用 `artifacts/E0/folds.json` 中的 inner 折；
`fold0+1` 只能用于“不进入最终 Gate 数值”的资源预检，且必须标记 `exploratory=true`。

### H2. P 级 Gate 预注册模板普遍缺必填项，且校验器不存在

- `docs/gate_template.md` 要求 6 个 mandatory checks：
  `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`、
  `checkpoint_resumable`、`no_label_leak`。
- 实际 33 个 P 计划 JSON 模板中：
  - 29 个未包含全部 4 个核心 checks；8 个连 `mandatory_checks` 字段都没有；
  - 10 个没有 `thresholds`；28 个没有 `baseline_version`；全部没有
    `primary_threshold_key`、`baseline_artifact`、`baseline_manifest_sha256`、`created_at`、
    `planned_task_training_h`；
  - `docs/PROJECT_FILES.md` 声明有 `src/validation/gates.py`，但该文件不存在。
- 各 P 末尾都写“prereg_extra 已按 P 的性质补齐”，但 JSON 示例中并未补齐。

**建议**：生成每份 P 的完整预注册模板（可由生成器统一填默认值），
实现 `src/validation/gates.py` 并在 Gate 前实际写盘、校验。

### H3. 计划篇数/行数/状态字段与实际仓库不符

- 声明：`PLAN.md` 661 行；P 计划平均 224 行、合计约 7,600 行；46 个计划文件合计 10,336 行。
- 实测：`PLAN.md` **694 行**；P 计划 33 份共 **3,822 行**，平均 **115.8 行**；
  46 个计划文件合计 **5,222 行**。
- `versions/status.json` 的 E0 stage 状态为 `done_local_pending_cloud_env`，
  不在文件自己声明的 `pending/in_progress/done/no_go/blocked` 枚举内。
- `README.md` 与 `docs/training_tasks.md` 写分支 `main`，当前仓库分支为 `master`。
- `E0/docs/data_card.md` 写单井行数 median 9,504，数据卡 JSON 为 9,547。

**建议**：把完成度表改为“由脚本统计生成”，避免手写行数；统一状态枚举。

### H4. `check_env.py` 在云端不可按预期通过，且未真正检查部分硬项

- `E0/code/check_env.py:181-183` 硬编码检查 `root.parent/data/train` 与
  `root.parent/v1/src/well_folds.json`；云端代码在 `/code/workspace/v4`，
  数据在 `/data/v4/data`，父目录没有 `data/` 与 `v1/`，因此会报 hard failure。
  脚本也没有读取 `V4_DATA_ROOT`，与 `bootstrap_data.sh`、`configs/paths.yaml` 的契约不一致。
- `run_train.sh` 的 `run_env` 用 `|| log` 忽略了 `check_env` 的非零退出码，
  即使环境检查失败也继续。
- `check_env.py` 没有实际比较 `torch.version.cuda` 与 `EXPECTED_CUDA_MAJOR_MINOR=(12,6)`；
  `E0_env.json` 的 `deps` 只收集 numpy/pandas/scipy/sklearn/pyarrow，与 P0 要求的
  “9 个依赖”不符。
- P0 宣称“hard 检查 6/6”，实际脚本的 hard 检查数量更多且内容不同。

**建议**：把数据/折检查移到独立脚本或改为读取环境变量；环境检查只保留环境项；
`run_env` 对 hard failure 必须非零退出；补 CUDA 版本与全部依赖探测。

### H5. `assert_disk_headroom(8.0)` 与“8 GB 硬门禁”语义不一致

- 文档反复写“`assert_disk_headroom(8.0)` 是硬门禁”。
- `src/data/disk_guard.py:156-190` 的实际行为：`free < 8 GB` 只触发 cleanup；
  只有 `free < 5 GB` 才 `save_and_exit`、`free < 3 GB` 才 abort；
  cleanup 后若仍为 6–8 GB，函数正常返回，不抛错。
- `disk_guard.py` CLI 在 level != ok 时返回 2，这与训练循环内行为不一致。

**建议**：明确三级阈值语义；若 8 GB 是硬门禁，则 cleanup 后仍 <8 GB 必须报错或写
`risk_accepted`，并在 Gate 中体现。

---

## 3. 中优先级问题

| ID | 问题 | 建议 |
|---|---|---|
| M1 | `E1/PLAN.md:11` 写“24 维输入”，总计划/`features/basic.py`/`E1/P0` 写 32 维；B1 修复时一并统一 | 以 `FEATURE_NAMES` 为准生成文档 |
| M2 | `RowMLP` 的 POR 头 `0.1+softplus(g)` 在 bias=0 时输出约 0.793，不能“从 0.1 起步”，也不能表示训练集中 186 行 <0.1 的有效 POR | 初始化 bias 到使输出≈0.1，或改用带下界校正的参数化；加单测 |
| M3 | `run_all.py` 的 E0 anchor 检查允许 `drop` 或 `mask` 任一命中；`constants.SCORE_MISSING_MODE` 未被 Gate 强制；数据卡 `folds.n_folds` 为 null | Gate 必须显式断言 `missing_mode=="drop"`；修 `folds.n_folds` |
| M4 | `run_all.export_folds` 写 `artifacts/E0/folds.json` 与 `versions/folds_sha256.json` 到仓库路径；`.gitignore` 忽略 `artifacts/`，云端克隆后不存在；`source_path` 写绝对路径 | 改为写 `V4_REPORTS_DIR` 或作为可重算产物；去掉绝对路径 |
| M5 | `E0/P3` 声称 `versions/candidates.json` 被 `predict.py` 读取，实际 `predict.py` 硬编码版本表；`src/versioning/registry.py` 不存在 | 实现 registry 或修改 P3 输出契约 |
| M6 | 缺失声明产物：`reports/E0_score_check.json`、`src/versioning/registry.py`、`artifacts/E0/folds.json` | 要么产出并纳入 git，要么从输出契约删除 |
| M7 | 空目录 `E1/P2`、`E4/P2`、`E7/P2` 存在但无 `PLAN.md`，未出现在阶段计划/status | 删除或补计划 |
| M8 | `docs/PROJECT_FILES.md`、README 把 `train.py`、`configs/v4.yaml`、`gates.py`、`unet1d.py` 等未来文件列为已存在；`run_train.sh --mode all` 实际只到 E1 | 文档明确“现状 vs 计划”，`all` 改为逐步编排或更名 |
| M9 | `requirements.txt` 含 markdown/`python==3.11`，不是可执行 pip 文件 | 生成纯 requirements 文件，或明确其仅为文档声明（rules 可能要求可读环境声明） |
| M10 | `parse.py` 中 `take = [...]` 为死代码；docstring 的 CASE 描述同 B3 | 清理并修正文档 |

---

## 4. 建议的修复顺序

1. **B1**：修正输入列切片与 14/13 维定义，加 train/test shape 与无标签泄漏断言。
2. **B6**：把 `build_cache` 纳入 E0/P1 或 E1/P0，并生成 `cache/manifest.json`。
3. **B2/B3**：重写数据卡的目标统计与 3 口井 schema 描述，修正 `constants`/标签文档。
4. **B4**：拆分 local/cloud E0 Gate，更新 status/README/PLAN 完成度状态。
5. **B5**：重算并修正 candidates.json、E0/P2 期望值，增加总分一致性校验。
6. **H1–H5**：修订所有 P 的 inner-OOF 选择协议、Gate 模板、行数/状态、env 检查、磁盘门禁。
7. 上述全部完成后，重跑 `E0/code/run_all.py` 与 `tools/verify_reference.py`，
   生成本地复审记录；在 P0 云端 Gate 通过前，E0 不得标记完成、E1 不得启动。

---

## 5. 复审通过条件

- [ ] B1–B6 全部有代码/文档修复与复算证据；
- [ ] `E0_local_contract_gate` 与 `E0_cloud_gate` 分离且均通过；
- [ ] 数据卡新增三目标范围统计，SW 尺度描述与真实数据一致；
- [ ] candidates.json 逐目标值与总分自洽；
- [ ] 所有 P 的 Gate 预注册模板通过 `validate_prereg`；
- [ ] 所有选择/早停只使用 inner 折；
- [ ] `versions/status.json` 使用合法枚举，且与计划完成度表一致。

> 审查结论：**当前 v4 总计划文档的架构和阶段划分基本完整，但执行门禁、数据契约和
> 输入管线存在会导致实验结论无效或提交失败的关键缺陷。修复完成并通过复审前，不应开始 E1。**
