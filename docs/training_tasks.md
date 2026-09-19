# 训练任务配置表（逐字段照抄）

> 平台字段见 [训练任务文档](http://discovery-staging.intern-ai.org.cn/docs/workbench/training)。
> 启动命令上限 **500 字符**，本表全部命令都远低于该限制。

## 通用字段（所有任务相同）

| 字段 | 值 |
|---|---|
| 代码来源 | **Git 仓库** |
| 仓库地址 | `<你的 git 仓库地址>` |
| 分支 | `main` |
| 资源配置 | **Nvidia A100 × 1**（80 GB 显存 / 4000m vCPU / 16 GiB 内存） |
| 镜像 | 【我的镜像】→ `v4-train-py311-torch271-cu128`（场景 = **训练任务**）<br>未构建前先用官方 PyTorch 2.7.1 / CUDA 12.8 / Python 3.11 镜像 |
| 训练/验证数据集 | 不挂载（数据走云盘 `/data`） |
| 超参数 | 不填（全部通过 `run_train.sh` 参数传递） |

代码会被克隆到 **`/code/workspace/v4/`**，所以启动命令一律以
`bash /code/workspace/v4/run_train.sh` 开头。

---

## 任务 1：`v4-env` — 环境与磁盘自检（必做，≈5 min）

| 字段 | 值 |
|---|---|
| 启动命令 | `bash /code/workspace/v4/run_train.sh --mode env` |
| 运行时长 | 0h30m |
| 产出（持久） | `/data/v4/reports/E0_env.json`、`/data/v4/reports/E0_disk_budget.json`、`/data/v4/logs/*.log` |
| 判据 | 日志中 `hard failures: 0`；`torch 2.7.1` / `A100 sm_80` / `bf16=True` / `free >= 8 GiB` |
| 失败处置 | 若 torch 版本或 GPU 不符 → 检查镜像与资源配置；若磁盘 < 8 GiB → 见 `docs/platform_setup.md` §五 收缩预案 |

## 任务 2：`v4-data` — 部署数据集到云盘（必做一次，≈2 min）

**前置**：本机已运行 `python3 v4/tools/pack_dataset.py`，并把 `v4/dist/v4_data.tar.gz`
上传到云盘（例如 `/v4_data/v4_data.tar.gz`）。若上传到了别处，先把文件移到 `/data` 下任意位置，
再让脚本从该位置解压（脚本默认找 `v4/dist/v4_data.tar.gz`）。

| 字段 | 值 |
|---|---|
| 启动命令 | `bash /code/workspace/v4/run_train.sh --mode data` |
| 运行时长 | 0h30m |
| 产出（持久） | `/data/v4/data/{train,test}/*.txt`、`/data/v4/data/folds/v1_well_folds.json` |
| 判据 | 日志出现 `train wells=80 rows=730268`、`test wells=10 rows=95948`、`RESULT: OK` |
| 幂等 | 可重复执行；每次都会重算 tarball sha256 与行数 |

> 若把数据做成了平台**数据集**并挂载成功，可改用：
> `bash /code/workspace/v4/tools/bootstrap_data.sh --from-dir <挂载目录>`

## 任务 3：`v4-e0` — 口径复算（必做，≈1 min）

| 字段 | 值 |
|---|---|
| 启动命令 | `bash /code/workspace/v4/run_train.sh --mode e0` |
| 运行时长 | 0h30m |
| 产出（持久） | `/data/v4/reports/E0_data_card.json`、`E0_gate.json`、`E0_contract_tests.json` |
| 判据 | `const_baseline_drop: 70.490735`、`anchor_hit: true`、`contract_selftest: true`、`gate_passed: true` |
| 备注 | 这一步**不需要 GPU**，CPU 资源也能跑；但仍建议与训练任务用同一镜像以保证版本一致 |

## 任务 4：`v4-smoke` — 极小规模冒烟（E1 代码实现后，≈5 min）

| 字段 | 值 |
|---|---|
| 启动命令 | `bash /code/workspace/v4/run_train.sh --mode smoke` |
| 运行时长 | 0h30m |
| 判据 | 1 折 / 8 井 / 2 epoch 跑完，产出 OOF，且契约校验通过 |
| 当前状态 | E1 训练脚本尚未实现，脚本会**明确报错**而不是静默跳过 |

## 任务 5+：`v4-E1` … `v4-E8` — 分阶段训练

| 阶段 | 启动命令 | 建议时长（软预算） | 产出 |
|---|---|---|---|
| E1 行级基线 | `bash /code/workspace/v4/run_train.sh --mode stage --stage E1` | 2h | `$RUN_ROOT/E1/*`、OOF |
| E3 序列主干 | `... --stage E3` | 20h | `$RUN_ROOT/E3/*` |
| E8 集成 | `... --stage E8` | 40h | `$RUN_ROOT/E8/*` |

**续训**（平台停止任务后不能直接续跑，必须新建任务）：

```bash
bash /code/workspace/v4/run_train.sh --mode stage --stage E3 --resume
# run_train.sh 把 --resume 透传，训练脚本从 $V4_RUN_ROOT/E3/last.pt 恢复
```

**长任务建议**：单次任务上限 7×24 h，但为降低风险，把 E3/E8 拆成
「每折一个任务」或「每 12 h 一个任务 + `--resume`」，checkpoint 写
`/data/v4/runs/<stage>/{best,last,last_prev}.pt`。

---

## 平台限制速记（来自官方文档）

| 限制 | 值 | 影响 |
|---|---|---|
| 启动命令长度 | ≤ 500 字符 | 全部封装进 `run_train.sh` |
| 单次任务时长 | ≤ 7×24 h | 需 `--resume` 支持拆分 |
| 本地上传代码包 | ≤ 500 MB 且禁含权重/大数据集 | v4 用 git 仓库 + 云盘数据 |
| 云盘持久目录 | `/data` | 所有产物写这里 |
| 代码目录 | `/code/workspace`（临时） | 不写任何需要保留的东西 |
| 镜像数量上限 | 5 个 | 只建 1 个训练镜像 |
| 镜像场景 | 开发机 / 训练任务互不通用 | 建「训练任务」场景 |
| 停止任务后续跑 | 不支持，只能【重新训练】 | 依赖 checkpoint + `--resume` |
