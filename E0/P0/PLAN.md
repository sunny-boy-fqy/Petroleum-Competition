# E0/P0 环境与磁盘实测（云端第一次运行）

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：契约前置：不产出模型，只产出**环境事实**　|　**依赖**：无（这是全项目第一步）

> **状态：待云端执行**（本机无 GPU / 无 torch，无法替代）。需在平台训练任务以 A100 + 预装镜像运行 `run_train.sh --mode env`。
>
> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

在云端实机确认 CUDA 12.6 / PyTorch 2.4.0 / Python 3.11 / A100 sm_80 / bf16 可用，并实测 30 GB 磁盘的可用余量与分布，产出可复算的 `E0_env.json` 与 `E0_disk_budget.json`。

## 2. 为什么需要这一步

1. 未实测的环境假设会在 E3 训练数小时后才暴露（OOM / 版本不兼容 / 磁盘写满），返工成本是 A100 机时；
2. 30 GB 预算是本项目最硬的资源约束，而 `/code/workspace`（临时）与 `/data`（云盘）是否同一文件系统必须实测，不能假设；
3. `torch 2.4.0` 与 `2.5+` 的 API 差异（`torch.nn.attention`、`torch.export` 新签名）会直接导致运行期 AttributeError，必须前置断言。

## 3. 输入契约

- 云端训练任务（Git 仓库代码来源，A100 资源，PyTorch 2.4.0/CUDA 12.6/py3.11 镜像）
- `v4/E0/code/check_env.py`、`v4/E0/code/setup_deps.sh`、`v4/src/data/disk_guard.py`

## 4. 输出契约

- `$V4_REPORTS_DIR/E0_env.json`
- `$V4_REPORTS_DIR/E0_disk_budget.json`
- `versions/locks/cloud_frozen.txt`（镜像构建后 `pip freeze` 快照）
- `$V4_LOG_DIR/train_env_*.log`（stdout 全量日志）

## 5. 执行步骤

1. `bash /code/workspace/v4/run_train.sh --mode env`
2. 读日志确认 `hard failures: 0`，逐项核对 torch/cuda/gpu/bf16/disk 五行
3. `df -h / /data /code/workspace` 记录三个挂载点的容量与是否同盘
4. `du -sh /usr /opt /root 2>/dev/null` 记录镜像本体占用，推算项目可用空间
5. 若可用 < 12 GB，把 `contingency_applied=true` 与收缩项写入 `E0_disk_budget.json`
6. 把 `E0_env.json` 的 `deps.versions` 与 `versions/locks/cloud.txt` 逐项比对，不一致则更新 lock 并提交

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `--min-free-gb` | 8.0 | 8–12 | 磁盘硬门禁；实测后若过紧则上调 |
| `--allow-non-a100` | false | — | 仅本机开发时开启（把 GPU 检查降级为 warn） |
| `NUM_WORKERS` | 4 | 2–6 | 16 GiB 内存下的安全值，见总计划 §3.1-3 |

## 7. 完成判据

- `check_env.py` 的 **hard 检查 6/6 通过**：python 3.11 / torch 2.4.0 / cuda 可用 / A100 sm_80 / bf16 / disk ≥ 8 GiB
- `E0_disk_budget.json` 含 `total_gb/used_gb/free_gb/level`，且 `level=="ok"`
- `E0_env.json` 含全部 9 个可选依赖的 `available/versions`
- `cloud_frozen.txt` 已生成并与 `versions/locks/cloud.txt` 一致或已更新

## 8. 禁止事项

- `pip install torch` / 升级 CUDA / 用 conda 建环境
- 安装 flash-attn / xformers / apex / deepspeed 等需编译 CUDA 扩展的包
- 在磁盘余量未知的情况下启动任何训练

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 镜像里是 torch 2.5+/2.3，不是 2.4.0 | hard failure 直接报错 | 改用镜像内实际版本并同步改 `versions/locks/cloud.txt` 与代码中的 2.4-only 断言 |
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
bash /code/workspace/v4/run_train.sh --mode stage --stage E0
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E0_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E0_P0_gate",
  "stage": "E0",
  "p_stage": "P0",
  "candidate_budget": 1,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "env_hard_checks_passed",
  "thresholds": {
    "min_hard_pass": 6
  },
  "mandatory_checks": [
    "env_hard_checks_passed",
    "disk_budget_ok"
  ]
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
