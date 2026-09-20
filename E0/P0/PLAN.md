# E0/P0 环境与磁盘实测（云端第一次运行）

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：契约前置：不产出模型，只产出**环境事实**　|　**依赖**：无（这是全项目第一步）
>
> **状态**：⛔ 阻塞　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

在云端实机确认 CANN 8.3rc2 / PyTorch 2.8.0 + torch_npu 2.8.0 / Python 3.11 / Ascend 910B(64G) / bf16 可用，并实测云盘（/data，30 GB 配额）的可用余量与分布，产出可复算的 `E0_env.json` 与 `E0_disk_budget.json`。

## 2. 为什么需要这一步

1. 未实测的环境假设会在 E3 训练数小时后才暴露（OOM / 版本不兼容 / 磁盘写满），返工成本是 Ascend 机时；
2. 30 GB 云盘配额是本项目最硬的资源约束，而 `/code/workspace`（临时）与 `/data`（云盘）是否同一文件系统必须实测，不能假设；
3. 镜像内 torch 小版本漂移（2.7 vs 2.8）、以及 **2.6 起 `torch.load(weights_only=True)` 的默认值变更**，都会在训练数小时后才暴露成 AttributeError/反序列化错误，必须前置断言。

## 3. 输入契约

- 云端训练任务（Git 仓库代码来源，Ascend 910B 资源，PyTorch 2.8.0+torch_npu 2.8.0/CANN 8.3rc2/py3.11/arm64 镜像）
- `v4/E0/code/check_env.py`、`v4/E0/code/setup_deps.sh`、`v4/src/data/disk_guard.py`

## 4. 输出契约

- `$V4_REPORTS_DIR/E0_env.json`
- `$V4_REPORTS_DIR/E0_disk_budget.json`（**按挂载点**报告：`paths` + `worst_level` + `primary_path` + `data_root_checked`）
- `versions/locks/cloud_frozen.txt`（镜像构建后 `pip freeze` 快照）
- `$V4_LOG_DIR/train_env_*.log`（stdout 全量日志）

## 5. 执行步骤

1. `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env`
2. 读日志确认 `hard failures: 0`，逐项核对 torch/cuda/gpu/bf16/disk 五行
3. `df -h / /data /code/workspace` 记录三个挂载点的容量与是否同盘，并写入 `E0_disk_budget.json::paths`（Gate 的 `disk_budget_ok` 只看 DATA-ROOT 级别）
4. `du -sh /usr /opt /root 2>/dev/null` 记录镜像本体占用，推算项目可用空间
5. 若 **DATA-ROOT（`$V4_DATA_ROOT` = `/data`）** 可用 < 12 GB，把 `contingency_applied=true` 与收缩项写入 `E0_disk_budget.json`（`worst_level`/`primary_path`/`data_root_checked` 一并落盘）
6. 把 `E0_env.json` 的 `deps.versions` 与 `versions/locks/cloud.txt` 逐项比对，不一致则更新 lock 并提交

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `--min-free-gb` | 8.0 | 8–12 | 磁盘硬门禁；实测后若过紧则上调 |
| `--allow-non-target-device` | false | — | 仅本机开发时开启：**只放宽 设备/架构/torch 检查**，不放宽依赖检查 |
| `NUM_WORKERS` | 4 | 2–6 | 16 GiB 内存下的安全值，见总计划 §3.1-3 |

## 7. 完成判据

- `check_env.py` 输出 **hard failures: 0**（判据是「**所有 hard 级检查全过**」，**不是固定项数**，也不因 `--allow-non-target-device` 而放宽依赖检查——该开关只放宽 设备/架构/torch 检查）；非主路径可选依赖（`onnx` / `onnxruntime`）失败时写入顶层 `degraded_paths` 并降级为 warn，不阻塞训练
- `E0_disk_budget.json` **按挂载点**报告（`paths` + `worst_level` + `primary_path` + `data_root_checked`），且 Gate 的 `disk_budget_ok` 读取 **DATA-ROOT（`$V4_DATA_ROOT` = `/data`）** 级别——不是 `/`
- `E0_env.json` 含全部可选依赖的 `available/versions` 与 `degraded_paths`
- `cloud_frozen.txt` 已生成并与 `versions/locks/cloud.txt` 一致或已更新

## 8. 禁止事项

- `pip install torch` / 升级 CUDA / 用 conda 建环境
- 安装 flash-attn / xformers / apex / deepspeed 等需编译 CUDA 扩展的包
- 在磁盘余量未知的情况下启动任何训练

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 镜像里不是 torch 2.8.0（或 torch_npu 与 torch 小版本不一致） | `torch_version` hard failure | 按镜像实际版本同步改 `check_env.py::EXPECTED_TORCH`/`EXPECTED_TORCH_NPU`/`EXPECTED_CANN`、`versions/locks/cloud.txt` 与 docs；**不要**pip 降级 torch |
| CUDA runtime 与声明值不一致（cu126 wheel） | `cuda_runtime_declared` warn | 只记录事实；hard 底线是 runtime major==12，不因此阻断 Gate（R4-B1 教训） |
| 系统内存被镜像/其它进程占用 | `free -g` 显示可用 < 14 GiB | 把 `num_workers` 降到 2，并在 E2 关闭特征缓存 |
| `/data` 与 `/code` 同盘且总容量仅 30 GB | `df` 显示同一 Filesystem | 按总计划 §3.4.1 安全规则收缩：特征缓存 ≤0.5 GB、集成成员 ≤2、模型宽度减半 |
| pip 计划替换 torch | `setup_deps.sh` 预检命中 | 脚本自动中止（exit 3）；改为只用镜像自带版本 |

## 10. 停止规则

- hard failure 未清零前，禁止进入 E0/P1 之后的任何阶段
- 可用磁盘 < 8 GB 且无法清理时，暂停项目并先与 owner 确认配额

## 11. 代码归属

- `E0/code/check_env.py`
- `E0/code/setup_deps.sh`
- `src/data/disk_guard.py`
- `run_train.sh`

## 12. 复算与证据

- `reports/E0_env.json`（云端实测快照）
- `reports/E0_disk_budget.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env    # P0：环境+磁盘（先装依赖再硬校验）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode data   # 部署数据到 /data/v4/data
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode e0     # P1-P3：口径复算 + 分片缓存
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

预注册文件：`v4/reports/E0_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E0_P0_gate",
  "stage": "E0",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "env_hard_checks_passed",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0,
    "max_hard_failures": 0
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
    "no_label_leak",
    "env_hard_checks_passed"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
