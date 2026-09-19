# E10/P1 打包、干净目录复现与 B0 fallback

> 所属阶段：[E10](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：复现门禁 + 保底策略　|　**依赖**：E10/P0

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

组装 `submission_code_v4.zip`；在**只含 zip 与 data** 的干净目录用官方命令一次性复现；同时预构建 B0 fallback 保险包并在同一干净目录验证。

## 2. 为什么需要这一步

1. `rules.md` §8.4：复现失败即取消晋级资格——这是最高优先级风险；
2. v4 是纯 DL 管线，没有 B0 patch 隔离兜底，因此必须**预先**构建 B0 fallback；等到失败时再建已经来不及（且情绪与时间压力下易出错）；
3. 干净目录验证必须用官方 `--data_dir`（评测命令），不能用开发时的自定义参数。

## 3. 输入契约

- `models/v4/final/`、`v4/src/`、`v4/predict.py`、`v4/train.py`、锁文件
- `../v1/submission_e7/`（fallback 源）

## 4. 输出契约

- `submission/submission_code_v4.zip`、`submission/result.zip`
- `$V4_REPORTS_DIR/E10_reproduce_report.json`
- `submission/submission_code_b0_fallback.zip`、`$V4_REPORTS_DIR/E10_B0_fallback_manifest.json`

## 5. 执行步骤

1. 组装 zip：README/predict.py/train.py/requirements.txt/configs/src/models（含权重）
2. 在干净目录解压，执行 `python predict.py --data_dir ./data --output result.json`
3. 比对：与提交的 result.json 逐点差 ≤1e-6；记录最大差与超差行数
4. 两次独立运行，sha256 必须一致
5. 构建 B0 fallback：从冻结 v1 源码生成官方 `--data_dir` 兼容的自包含包
6. 在**同一干净目录**验证 fallback：结果与冻结 B0 逐点差 ≤1e-9（或字节一致）
7. 写 `inference_verified=true` 与全部指纹

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| 逐点容差 | ≤1e-6 | 冻结 | 浮点级微差可接受 |
| fallback 容差 | ≤1e-9 或字节一致 | 冻结 | v1 包用同版本依赖 |
| 干净目录环境 | 只含 zip + data | 冻结 | 不得有 v4/v1/v2 目录 |
| 两次运行 | sha256 一致 | 冻结 | 确定性要求 |

## 7. 完成判据

- 干净目录复现成功且逐点差 ≤1e-6
- 两次运行 sha256 一致
- B0 fallback 包在同一干净目录验证通过（逐点差 ≤1e-9 或字节一致）
- `E10_reproduce_report.json` 含目录/命令/stdout 摘要/版本/耗时/内存/指纹

## 8. 禁止事项

- 在提交包里依赖 v1/v2/v3 目录或外部数据
- 复现时联网或安装包
- 跳过 fallback 构建
- 用开发时的自定义 CLI 参数代替官方 `--data_dir`

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 复现失败 | 干净目录报错或结果不一致 | 门禁前置到每个候选；失败立即回退 B0 fallback |
| fallback 也失败 | 保险失效 | fallback 现在就建并验证，不等到 E10 当天 |
| zip 体积超限 | 上传失败 | 权重 < 50 MB、代码 < 5 MB，剔训练日志 |

## 10. 停止规则

- 复现不通过 → 不提交 v4，改用 fallback 包，并在报告写明原因

## 11. 代码归属

- `E10/code/build_submission.py`
- `E10/code/verify_inference.py`
- `E10/code/build_b0_fallback.py`

## 12. 复算与证据

- `reports/E10_reproduce_report.json`、`reports/E10_B0_fallback_manifest.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E10
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E10_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E10_P1_gate",
  "stage": "E10",
  "p_stage": "P1",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "clean_dir_reproduce",
  "thresholds": {
    "max_point_diff": 1e-06
  },
  "mandatory_checks": [
    "clean_dir_reproduce",
    "b0_fallback_verified",
    "contract_ok"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
