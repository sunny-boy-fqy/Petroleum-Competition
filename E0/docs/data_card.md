# E0 数据卡与口径冻结（E0-R2）

> 复算命令：`python3 v4/E0/code/run_all.py`（本机可跑，**不需要 torch**）
> 事实源：`reports/E0_data_card.json`、`reports/E0_score_check.json`、
> `reports/E0_local_contract_gate.json`、`reports/E0_cloud_gate.json`
>
> **E0-R2 修订说明**：本文件在 E0-R1 之后根据独立审查（`reports/V4_PLAN_REVIEW.md`）
> 修正了三处会直接导致实验结论无效的错误：①输入列泄漏；②SW 尺度误判；③非规范井描述错误。
> 每条都附复算证据。

---

## 0. 三个已修正的严重问题（含复现证据）

### 0.1 输入列泄漏：第 14 个"输入"曾是 POR 标签（已修复）

**现象**：`parse.py` 用 `inputs = arr[:, 1:15]` 取输入，而规范 17 列布局中索引 **14 正是 POR**，
于是 80 口训练井的 `inputs[:, 13]` 与 `targets[:, 0]` 逐点相等（实测相关系数 1.0）。

```
train inputs.shape: (5690, 14)  targets.shape: (5690, 3)
inputs[:,13] 与 POR 逐点相等比例: 1.000000   ->  泄漏
odd well(20 列) inputs.shape: (7879, 14)
```

**同时暴露第二个 bug**：测试井只有 14 列（无目标），同一段切片只得到 **13** 列，
`build_row_features` 会因形状不匹配直接抛 `ValueError` —— 即提交侧必然失败。

**修复**（`src/data/parse.py`、`src/constants.py`、`src/features/basic.py`）：

1. 明确规范布局常量：`IDX_DEPTH=0`、`IDX_CURVE_START=1`、`IDX_CURVE_STOP=1+13=14`、目标 = 末 3 列；
2. `INPUT_COLUMNS = COLUMNS[1:14]`（**13 条曲线**，DEPTH 单独作为深度通道），`N_INPUT=13`；
3. 列布局**与 `with_targets` 无关**：无论训练/测试，输出恒为规范 17 列（缺失目标列填 NaN），
   保证训练与测试的输入通道位置完全一致；
4. 硬断言：`n_in == 13`、`target_start >= 14`（目标区间不得与输入区间重叠）；
5. 回归测试：90 口井逐井检查"输入列数 = 13"且"任何输入列与任何目标列不完全相等" → **全部通过**；
   该检查已纳入 E0 的 mandatory check `input_no_label_leak`。

**教训（已写入计划纪律）**：**禁止按列位置猜测语义**；任何切片都必须由 `constants.py` 的
布局常量推导，并配一条"输入与目标不相交"的断言。

### 0.2 SW 尺度误判：SW 是**单一标签尺度**，不是 [0,1]（已修复）

**原错误假设**：文档写"SW 占位 99.9（百分数），有效值 [0,1]（小数），需 ×100 双尺度换算"。

**实测反驳**（80 口训练井，非缺测且非联合占位行）：

| 目标 | min | p01 | median | p99 | max | <1 的行数 |
|---|---:|---:|---:|---:|---:|---:|
| POR | 0.000 | 1.738 | 11.339 | 22.551 | 33.177 | 587 |
| PERM | 0.000 | 0.010 | 0.834 | 54.993 | 1064.683 | 122,727 |
| **SW** | **8.305** | 31.979 | **82.805** | 99.900 | 99.900 | **11** |

SW 直方（有效行，10 档）：`[5, 0, 0, 0, 0, 0, 0, 0, 0, 236337]` —— 有效 SW 几乎全部在 80–99.9。
**结论**：SW 与 POR/PERM 一样是**单一标签尺度（百分数）**。

**修复**：`constants.SW_LABEL_RANGE=(0,100)`、`SW_SMALL_BRANCH=False`（默认关闭）；
`labels.sw_decode` 默认不再乘 100；新增 `labels.sw_scale_report` 把实测范围写入数据卡；
E5/P2 的"双尺度单测"改为"单尺度断言 + 明确禁止全局裁剪到 [0,1]"。

> 保留 `SW_SMALL_BRANCH` 开关与旧换算路径，仅用于**对照实验**，默认关闭。

### 0.3 非规范井描述错误（已修复）

| 井（前 8 位） | 列数 | 差异 | 是否缺 CASE | 行数 |
|---|---:|---|---|---:|
| `42f2870b` | 20 | 多 `K, U, CGR` | **否（含 CASE）** | 7,879 |
| `b7eb1274` | 21 | 多 `TH, K, U, CGR` | **否（含 CASE）** | 9,547 |
| `c7611b01` | 16 | — | **是（唯一）** | 9,654 |

即 **17,426 行是"多列"，9,654 行是"缺 CASE"**（此前文档误把 27,080 行全部写成缺 CASE）。
`reports/E0_data_card.json` 与 `dist/v4_data_manifest.json` 的数据一直是对的，只有文档写错。

---

## 1. 复算事实（`reports/E0_data_card.json`）

| 项 | 值 |
|---|---|
| 训练井 / 行数 | 80 / **730,268** |
| 测试井 / 行数 | 10 / **95,948**（= 契约值） |
| 缺测行（三目标全缺） | **6,700**（0.917%） |
| 逐目标缺测 | POR 6,701 / PERM 6,711 / SW 6,711 |
| 部分缺测行 | 12 |
| **联合常量占位行** | **487,225**（66.719%） |
| 有效行（无缺且非占位） | **236,343**（32.364%） |
| 单井行数 | min 4,473 / median 9,547 / max 13,078 |
| 重复深度行 / 非递增深度行 | 0 / 0 |
| 输入曲线数（规范） | **13**（GR…CASE）+ DEPTH 单独作为深度通道 |
| 输入泄漏回归 | **通过**（90 井：13 列输入、输入与目标不相交） |

## 2. 评分口径（`reports/E0_score_check.json`）

常数基线 (0.1, 0.01, 99.9) 在 80 井上的实测：

| 分母口径 | acc_por | acc_perm | acc_sw | Total | 与锚点 70.4907 |
|---|---:|---:|---:|---:|---|
| **`drop`（冻结采用）** | 0.6735824 | 0.7072023 | 0.7294624 | **70.490735** | 命中（±1e-4） |
| `mask`（仅诊断） | — | — | — | 69.843218 | 差 −0.6475 |

自洽校验：`100×(0.30×0.6735824 + 0.35×0.7072023 + 0.35×0.7294624) = 70.490735` ✅
（该恒等式已由 `tools/check_consistency.py` 对**所有候选**强制校验。）

> **口径冻结**：`constants.SCORE_MISSING_MODE = "drop"`；数据卡的 mandatory check
> `missing_mode_is_drop` 与 `score_total_consistent` 会阻止口径漂移。

常数基线逐目标命中（非缺测行内）：

| 目标 | 命中行数 / 非缺测行数 | 说明 |
|---|---:|---|
| POR | 487,382 / 723,567 | 占位全中 + 157 行真值恰在 ±8% 带内 |
| PERM | 540,273 / 723,557 | 占位全中 + 53,048 行真值落在 [0.1, 10] |
| SW | 536,941 / 723,557 | 占位全中 + 49,716 行真值落在 99.9±4.995 |

> 这解释了"全常量"能拿 70.49 分，也是 A 榜"常数对冲有效"的机制性原因（`资料库/12` §3.3）。

## 3. 折协议

| 项 | 值 |
|---|---|
| 源文件 | `versions/reference/v1_well_folds.json`（随 git 冻结；云端另备 `$V4_DATA_ROOT/v4/data/folds/`） |
| sha256 | `f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e` |
| 井数 / 折数 | 80 / 5（每折 16 井） |
| 导出 | `$V4_REPORTS_DIR/E0_folds.json`（outer + 每折 inner 3 折）、`versions/folds_sha256.json` |

## 4. 分片缓存（E1 的输入；审查 B6）

`run_all.py --with-cache` 产出 `cache/raw/<split>/<well>.npz` 与 `cache/labels/<well>.npz`，
并写 `cache/manifest.json`：

| 项 | 实测 |
|---|---|
| 缓存体积 | **32.4 MB**（90 井，float32 + 压缩） |
| manifest 计数 | 80 训练井 / 730,268 行；10 测试井 / 95,948 行 |
| 输入列校验 | 全部井 `inputs.shape[1] == 13` ✅ |
| mandatory checks | `shard_cache_built`、`shard_cache_input_cols_ok` |

云端用法（数据在 `/data`）：

```bash
python3 /code/workspace/v4/E0/code/run_all.py --with-cache \
  --train-dir /data/v4/data/train --test-dir /data/v4/data/test \
  --cache-root /data/v4/cache --out /data/v4/reports/E0_data_card.json
```

## 5. 提交契约（已单测）

`src/inference/contract.py::validate_payload` 的 6 项负样例全部被正确拒绝
（`reports/E0_contract_tests.json`）：PERM≤0 / 缺顶层键 / 行数不符 / depth 乱序 /
大写 `DEPTH` / 含 NaN。端到端冒烟：

```bash
python3 predict.py --use-version CONST --data_dir ../data --output /tmp/r.json
# -> 10 wells / 95,948 rows / contract_ok=true / ~1.4 s（单核 CPU）
```

版本解析来自 `versions/registry.json`（`src/versioning/registry.py`）；
`PD1` 未训练时 `predict.py` **明确报错**而非静默输出常数。

## 6. 双 Gate（审查 B4）

| Gate | 范围 | 报告 | 当前状态 |
|---|---|---|---|
| `E0_local_contract_gate` | 本机可复算的口径层 | `reports/E0_local_contract_gate.json` | ✅ passed（mandatory 10/10） |
| `E0_cloud_gate` | 云端环境与磁盘 | `reports/E0_cloud_gate.json` | ⏸ `blocked_pending_cloud_run` |

`E0_cloud_gate` 的 mandatory：`env_hard_checks_passed`、`disk_budget_ok`。
满足方式：平台训练任务执行 `run_train.sh --mode env` 产出 `E0_env.json` 与
`E0_disk_budget.json`，再重跑 `run_all.py`。

> **E0 阶段只有在两个 Gate 都通过后才算完成**；仅本地通过时 `versions/status.json`
> 记为 `in_progress`。

## 7. 对本机开发环境的说明

本机（开发机）**没有 torch、没有 GPU**，因此：

- `E0_local_contract_gate` 的全部检查都只用标准库 + numpy，可本机复算；
- `E0_cloud_gate` 必须在平台的 A100 训练任务里跑（见 `docs/platform_setup.md`）；
- 这是**计划内的例外**：E0/P1–P3 的本地复算先于 E0/P0 完成，P0 的状态为 `blocked`，
  不影响本机侧的契约层开发。

## 8. 对后续阶段的硬性影响

1. **输入是 13 条曲线 + DEPTH**：`F1 = 13 曲线 + DEPTH 原值 + 13+1 缺失位 + 4 深度编码 = 32 维`；
   任何文档/代码写 14 条曲线都是错的。
2. **SW 按单一标签尺度处理**：禁止 ×100 双尺度换算（除非显式开启对照开关）；
   **禁止全局裁剪到 [0,1]**。
3. **3 口非规范井**：只有 `c7611b01` 缺 CASE（记为 NaN + 缺失位）；`42f2870b`/`b7eb1274`
   多出的 K/U/TH/CGR 忽略但记录在 `extra_columns`（未来可作为"新信息源"启用）。
4. **评分一律 `drop` 口径**，报告必须标注；`mask` 只作对照。
5. **契约测试是提交前置**：`predict.py` 生成结果后自动跑 `validate_payload`，不通过即非零退出。
