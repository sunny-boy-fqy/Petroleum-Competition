# E0/P1 数据卡、哨兵规则与标签三状态

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：契约冻结：产出**唯一事实源**的数据卡　|　**依赖**：E0/P0（环境可用）

> **状态：已完成（2026-09-19）。** 证据：`reports/E0_data_card.json`、`E0/docs/data_card.md`、`versions/folds_sha256.json`、`artifacts/E0/folds.json`。
>
> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

用自写解析器读取 90 口井，冻结缺失哨兵规则、标签三状态判据与 SW 双尺度常量，产出数据卡、按井折导出与指纹；**必须查清 3 口非规范 schema 井的正确解析方式**。

## 2. 为什么需要这一步

1. 口径是唯一事实源：解析错一列，后面所有分数都不可比；
2. **实测发现 3 口训练井表头非官方 17 列**（`42f2870b` 20 列含 K/U/CGR、`b7eb1274` 21 列含 TH/K/U/CGR、`c7611b01` 16 列缺 CASE），共 27,080 行（3.71%），且都在 80 井折内、三目标齐全——按列位置解析会错位或丢行；
3. **SW 同一列混用 99.9（百分数）与 [0,1]（小数）两种尺度**，混淆会造成约 10 分静默损失；
4. 开发过程中已实际触发一次 numpy 越界切片静默截断（`arr[:,15:18]` 在 17 列数组上返回 2 列，丢掉 SW 整列），必须用断言防回归。

## 3. 输入契约

- `$V4_DATA_ROOT/v4/data/train/*.txt`（80 井，含表头/单位/数据三段）
- `$V4_DATA_ROOT/v4/data/test/*.txt`（10 井，无标签）
- `rules.md` §5.1–5.3
- `资料库/12` §3.1（占位分布）

## 4. 输出契约

- `$V4_REPORTS_DIR/E0_data_card.json`（计数/状态/深度诊断/schema 异常清单）
- `artifacts/E0/folds.json`（outer 5 折 + 每折 inner 3 折）
- `versions/folds_sha256.json`（折指纹）
- `$V4_CACHE_ROOT/raw/<split>/<well>.npz`、`labels/<well>.npz`（按井分片）

## 5. 执行步骤

1. `python3 E0/code/run_all.py`（默认从 `V4_DATA_ROOT` 取数据）
2. 核对 `train_rows`=730,268、`test_rows`=95,948、`state_counts`={missing:6700, placeholder:487225, valid:236343}
3. 核对 `noncanonical_schema_wells` 恰好 3 口，且 `missing_columns`/`extra_columns` 与实测一致
4. 核对 `folds_sha256`=f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e 且 fold_sizes 各 16 井
5. `python3 tools/verify_reference.py` 必须输出 `RESULT: OK`
6. 用 `V4_DATA_ROOT` 指向云端 `/data` 再跑一次，确认路径契约生效

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 缺失哨兵 | {-99999, -9999, NaN, 任何 < -1000} | 冻结，不可改 | `src/constants.py::SENTINELS/MISSING_LT` |
| 占位常量 | (POR=0.1, PERM=0.01, SW=99.9) | 冻结 | `constants.PLACEHOLDER` |
| 占位判等容差 | 1e-9（绝对） | 冻结 | 防止浮点写成 0.1000000001 时误判 |
| 折数 | 5（outer）/ 3（inner） | 冻结 | 与 v1 同折以保证历史锚点可比 |

## 7. 完成判据

- `E0_data_card.json` 全部计数与上表一致（逐项 equality，不是近似）
- 3 口畸形井在三目标上**零丢失**，且 `CASE` 缺失以 NaN + 缺失指示位表示
- `folds.json` 的 80 井与数据目录文件名集合**完全一致**（不重不漏）
- `verify_reference.py` 输出 `RESULT: OK`

## 8. 禁止事项

- 按固定列位置解析（必须按表头名映射）
- 删除占位行 / 把 3 口畸形井排除出训练
- 用 `pd.read_csv` 默认行为直接吃表头（单位行会被当成数据）
- 在没有列数断言的情况下做数组切片

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 把 3 口畸形井按位置解析 | POR/PERM/SW 三列错位或行数少 27,080 | 硬断言 `arr.shape[1]==n_out` + 逐井 schema 清单入数据卡 |
| numpy 越界切片静默截断 | SW 列被丢且不报错 | 目标列用 `n_out-len(TARGETS)` 动态计算 + `targets.shape[1]==3` 断言 |
| 单位行被当作数据行 | 行数多 80/10，深度出现 'm' | `next(reader)` 跳过第 2 行；非数值 token 记 NaN 并计数 |
| 折文件被 autocrlf 改写 | 字节 sha256 不一致 | `verify_reference.py` 同时校验内容指纹（与行尾无关） |

## 10. 停止规则

- 数据卡计数与实测不符时，禁止进入 E0/P2
- 发现新的 schema 变体（>5 种）时，暂停并重新设计解析契约

## 11. 代码归属

- `src/data/parse.py`
- `src/data/labels.py`
- `src/data/dataset.py`
- `src/validation/folds.py`
- `E0/code/run_all.py`
- `tools/verify_reference.py`

## 12. 复算与证据

- `E0/docs/data_card.md`（含 3 口井的实测表）

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E0
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E0_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E0_P1_gate",
  "stage": "E0",
  "p_stage": "P1",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "data_card_recomputable",
  "mandatory_checks": [
    "data_card_recomputable",
    "row_counts_match",
    "folds_fingerprint_present"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
