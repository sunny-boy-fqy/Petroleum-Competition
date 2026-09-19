# E0/P3 提交契约、版本路由与干净目录冒烟

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：交付契约冻结：格式错误 = 零分风险　|　**依赖**：E0/P1（数据）、E0/P2（评分）
>
> **状态**：✅ 已完成　　证据：`reports/E0_contract_tests.json`（6 负样例全拒绝）、`versions/registry.json`、`versions/candidates.json`

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现官方 CLI 的 `predict.py`、`result.json` 结构校验、候选注册表与 manifest，并在**只含代码与数据**的干净目录完成冒烟（含常数基线端到端）。

## 2. 为什么需要这一步

1. `rules.md` §6.1 规定 JSON 结构禁止增删改字段，小写 `depth`，10 井 95,948 行；格式错一次就浪费一次每日 5 次的提交额度；
2. `rules.md` §8.4：复现失败直接取消资格，因此契约必须早于模型存在并可自动化校验；
3. 契约校验必须**不依赖 torch**，否则本机（无 GPU）无法在提交前自检。

## 3. 输入契约

- `rules.md` §6.1–6.3、§8
- `data/测试数据返回结果格式.json`（官方模板）
- `资料库/12` §4（提交规范）、§5（复现要求）

## 4. 输出契约

- `v4/predict.py`（`--data_dir/--output/--use-version/--list-versions`）
- `src/inference/contract.py`（`validate_payload`/`validate_file`/`depth_alignment_report`）
- `versions/registry.json`（可运行版本事实源）+ `versions/candidates.json`（候选事实源）
- `$V4_REPORTS_DIR/E0_contract_tests.json`（正/负样例自检）

## 5. 执行步骤

1. 实现 `validate_payload`：顶层键集合、logId 集合与文件名一致、行数、逐行键名、depth 严格递增、有限性、PERM>0、禁止 SW 裁剪
2. 实现 `predict.py`：识别 `--data_dir` 指向 `data/` 或测试井目录两种形态；生成后自动调用契约校验，失败即非零退出
3. 支持 `--use-version CONST` 走常数基线（用于契约自检，不参与评分竞争）
4. 跑 6 个负样例单测（PERM≤0 / 缺顶层键 / 行数不符 / depth 乱序 / 大写 DEPTH / NaN）
5. 在只含 `v4/` 与 `data/` 的干净目录执行 `python3 predict.py --data_dir ./data --output result.json`
6. 建立 `versions/candidates.json` 空表与 schema 注释

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 预期测试井数 | 10 | 冻结 | `constants.EXPECTED_N_TEST_WELLS` |
| 预期测试行数 | 95,948 | 冻结 | `constants.EXPECTED_N_TEST_ROWS` |
| depth 小数位 | 1 | 冻结 | 相对输入深度对齐，容差 1e-6 |

## 7. 完成判据

- 6 个负样例**全部被正确拒绝**，正样例通过（`E0_contract_tests.json::passed=true`）
- 干净目录下 `python3 predict.py --use-version CONST --data_dir ./data --output result.json` 在一次运行内产出 10 井 / 95,948 行且 `contract_ok=true`（实测 ≈1.4 s，单核 CPU）
- `predict.py` 在**无 torch** 环境可运行；`--list-versions` 正确区分可用/未训练版本
- `versions/registry.json` 建立且被 `predict.py` 读取（`src/versioning/registry.py`）

## 8. 禁止事项

- 契约校验依赖 torch 或网络
- 静默裁剪 SW / POR 到物理区间
- 允许 logId 缺失、行数不符、深度错位通过校验
- 把未训练版本当作可用版本静默输出常数

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 官方 `--data_dir` 指向 `data/` 而代码假设指向 `test/` | 找不到井文件 | `predict.py` 同时支持两种形态并打印实际使用的目录 |
| JSON 键名大小写不一致 | 评测字段解析失败 | 契约强制小写 `depth`，并有负样例单测锁定 |
| 浮点序列化差异 | 两次运行 sha256 不同 | 固定小数位与序列化参数，E10 做两次运行一致性校验 |

## 10. 停止规则

- 契约自检未全绿，禁止任何候选进入 `submitted` 状态

## 11. 代码归属

- `predict.py`
- `src/inference/contract.py`
- `src/versioning/registry.py`

## 12. 复算与证据

- `reports/E0_contract_tests.json`（6 负样例全拒绝）
- `versions/registry.json`
- `versions/candidates.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode env    # P0：环境+磁盘（先装依赖再硬校验）
bash /code/workspace/v4/run_train.sh --mode data   # 部署数据到 /data/v4/data
bash /code/workspace/v4/run_train.sh --mode e0     # P1-P3：口径复算 + 分片缓存
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 12.5 选择协议（H1：inner-OOF only）

**所有超参/阈值/早停/结构选择只允许用 inner 折**（`$V4_REPORTS_DIR/E0_folds.json::inner`）。

| 用途 | 允许的数据 | 禁止 |
|---|---|---|
| 超参/阈值/λ/τ/集成权重选择 | 该 outer 折的 inner-OOF | outer 验证折标签 |
| 早停 | inner-OOF 的真实 `score.py` 分数 | outer 折分数、loss 值 |
| 结构/特征筛查（省机时） | 可先用 fold0 做**资源预检** | 预检结论不得进入 Gate 数值 |

> 若某步骤确实只能看 outer 折（例如最终 OOF 汇总），该步骤**不得**反过来影响任何选择；
预检性质的 fold0 结果必须在报告中标 `exploratory=true`、`selection_score_only=true`。

## 13. Gate 预注册要点

预注册文件：`v4/reports/E0_P3_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E0_P3_gate",
  "stage": "E0",
  "p_stage": "P3",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "contract_selftest_passed",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0
  },
  "alpha": 0.05,
  "multiplicity": "none",
  "candidate_budget": 1,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "pilot_std": null,
  "mde_units": 80,
  "min_detectable_effect": null,
  "planned_task_training_h": 1.0,
  "mandatory_checks": [
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "checkpoint_resumable",
    "no_label_leak"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
