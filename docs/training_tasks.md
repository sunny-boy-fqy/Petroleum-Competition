# 训练任务配置表（逐字段照抄）

> 平台字段见 [训练任务文档](http://discovery-staging.intern-ai.org.cn/docs/workbench/training)。
> 启动命令上限 **500 字符**，本表全部命令都远低于该限制。

> **⚠️ 每次开跑前必须先 push**（协议见 `PLAN.md` §3.1 / `docs/platform_setup.md` 步骤 3）：
> ```bash
> cd /home/fangqiyu/projects/Petroleum-Competition/v4
> git push origin HEAD:master HEAD:main && git log --oneline -1
> ```
> 平台只克隆**已 push** 的代码。**本机**的 remote 与 SSH 认证已配置好
> （`origin` = `git@github.com:sunny-boy-fqy/Petroleum-Competition.git`），
> 不要再 `git init` / `git remote add`。

> **⚠️ 平台【仓库地址】要填 HTTPS，不要填上面那个 SSH 地址**：
> `https://github.com/sunny-boy-fqy/Petroleum-Competition.git`
> 平台侧**没有**你的 SSH 私钥；而且平台表单正则会**直接拒绝** scp 形式的
> `git@github.com:...`（根本提交不了）。仓库是 public，HTTPS 无需任何凭据。
> 任务失败且无日志的排查见 `docs/platform_setup.md` §六-8。

## 通用字段（所有任务相同）

| 字段 | 值 |
|---|---|
| 代码来源 | **Git 仓库** |
| 仓库地址 | `https://github.com/sunny-boy-fqy/Petroleum-Competition.git`（**HTTPS**，不是 `git@…`） |
| 分支 | **`main`**（平台默认；remote 上 `main` 与 `master` 同指一个 commit，填哪个都能拉到） |
| 资源配置 | **Ascend 910B × 1**（64 GB HBM / 4000m vCPU / 16 GiB 内存 / 30 GB 云盘） |
| 镜像 | 【我的镜像】→ `v4-train-py311-torch280-npu280-cann83rc2`（场景 = **训练任务**）<br>未构建前先用官方 PyTorch 2.8.0 + torch_npu 2.8.0 / CANN 8.3rc2 / Python 3.11 / arm64 镜像 |
| 训练/验证数据集 | 不挂载（数据走云盘 `/data`） |
| 超参数 | 不填（全部通过 `run_train.sh` 参数传递） |

代码会被克隆到 **`/code/workspace/<平台决定的目录名>/`**（本仓库的根就是 v4 的内容，
所以目录名**不一定是** `v4`），因此启动命令一律用**位置无关**写法开头：

```bash
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env
```

`run_train.sh` 内部用 `$HERE` 自定位，`--mode env` 的日志第一行会打印真实的
`repo(HERE) = ...`，需要时可据此改用具体路径。

---

## 备用代码来源（Git 拉不到时用；也是 §六-8 的 2×2 实验材料）

`docs/platform_setup.md` §六-8 的判定实验需要「不依赖 Git」的代码来源。两种做法都用同一个
包：`dist/v4_code_src.zip`（`git archive` 产物，**仅 tracked 文件**，约 0.6 MB，**不含** 31 MB
数据 tarball，也未含任何权重）。

| 方式 | 上传物 / 填法 | 启动命令 |
|---|---|---|
| **本地上传** | 上传 `dist/v4_code_src.zip`；代码源选「本地上传」 | 见下方代码块 A |
| **我的云盘** | 把同一个 zip 上传到云盘根 `/`；代码源选「我的云盘」、云盘路径填 `/`。平台会**自动解压到同级目录** `/v4_code_src/` | 见下方代码块 B |

```bash
# A：本地上传（解压位置由平台决定，仍用 find 自定位）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env
# B：我的云盘（路径由"压缩包同级目录 + 包名"决定）
bash /data/v4_code_src/run_train.sh --mode env
```

本机重新生成该包（在仓库根执行；`dist/*.zip` 已被 `.gitignore` 忽略，不会进仓库）：

```bash
cd /home/fangqiyu/projects/Petroleum-Competition/v4
git archive --format=zip --prefix='' -o dist/v4_code_src.zip HEAD
```

---

## 任务 1：`v4-env` — 环境与磁盘自检（必做，≈5 min）

| 字段 | 值 |
|---|---|
| 启动命令 | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env` |
| 运行时长 | 0h30m |
| 产出（持久） | `/data/v4/reports/E0_env.json`、`/data/v4/reports/E0_disk_budget.json`、`/data/v4/logs/*.log` |
| 判据 | 日志中 `hard failures: 0`；`torch 2.8.0` / `Ascend 910B` / `bf16=True` / `free >= 8 GiB` |
| 失败处置 | 若 torch 版本或 GPU 不符 → 检查镜像与资源配置；若磁盘 < 8 GiB → 见 `docs/platform_setup.md` §五 收缩预案 |

## 任务 2：`v4-data` — 部署数据集到云盘（必做一次，≈2 min）

**前置**：本机已运行 `python3 v4/tools/pack_dataset.py`，并把 `v4/dist/v4_data.tar.gz`
（建议连带 `v4_data_manifest.json`）上传到**云盘 `/data/` 根目录**。

> **R5-B1**：`dist/*` 被 `.gitignore` 忽略，云端克隆出来的 repo 里**没有** tarball，
> 所以脚本会在 `$DATA_ROOT/v4_data.tar.gz`、`$DATA_ROOT/dist/v4_data.tar.gz`、
> `$V4/dist/v4_data.tar.gz` 依次搜索；上传到别处时用
> `--tarball <云盘路径>` 或环境变量 `V4_DATA_TARBALL=<云盘路径>` 显式指定。
> 找不到会打印搜索过的全部位置并 **exit 4**。

| 字段 | 值 |
|---|---|
| 启动命令 | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode data` |
| 备选启动命令 | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode data --tarball /data/uploads/v4_data.tar.gz` |
| 运行时长 | 0h30m |
| 产出（持久） | `/data/v4/data/{train,test}/*.txt`、`/data/v4/data/folds/v1_well_folds.json` |
| 判据 | 日志出现 `train wells=80 rows=730268`、`test wells=10 rows=95948`、`RESULT: OK` |
| 幂等 | 可重复执行；manifest 存在时重算 tarball sha256，井数与行数**每次都无条件硬校验**（R5-H2） |

> 若把数据做成了平台**数据集**并挂载成功，可改用：
> `bash "$(dirname "$(find /code/workspace -name run_train.sh | head -1)")/tools/bootstrap_data.sh" --from-dir <挂载目录>

## 任务 3：`v4-e0` — 口径复算（必做，≈1 min）

| 字段 | 值 |
|---|---|
| 启动命令 | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode e0` |
| 运行时长 | 0h30m |
| 产出（持久） | `/data/v4/reports/E0_data_card.json`、`E0_gate.json`、`E0_contract_tests.json` |
| 判据 | `const_baseline_drop: 70.490735`、`anchor_hit: true`、`contract_selftest: true`、`gate_passed: true` |
| 备注 | 这一步**不需要 GPU**，CPU 资源也能跑；但仍建议与训练任务用同一镜像以保证版本一致 |

## 任务 4：`v4-smoke` — 极小规模冒烟（E1 代码实现后，≈5 min）

| 字段 | 值 |
|---|---|
| 启动命令 | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode smoke` |
| 运行时长 | 0h30m |
| 判据 | 1 折 / 8 井 / 2 epoch 跑完，产出 OOF，且契约校验通过 |
| 当前状态 | E1 训练脚本**已实现**（`E1/code/train_row.py`，本机 CPU smoke 六项 mandatory checks 全绿）；80 井 5 折 OOF 待 Ascend 910B 任务 |

## 任务 5+：`v4-E1` … `v4-E8` — 分阶段训练

| 阶段 | 启动命令 | 建议时长（软预算） | 产出 |
|---|---|---|---|
| **E1→E10 全链路** | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode all` | 数十小时（受平台 7×24h 限制，需拆任务） | `/data/v4/{runs,reports,state,logs}` 全链路产物；进度见 `/data/v4/state/all_pipeline_progress.json` |
| E1 行级基线 | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E1` | 2h | `$RUN_ROOT/E1/*`、OOF |
| E3 序列主干 | `... --stage E3` | 20h | `$RUN_ROOT/E3/*` |
| E8 集成 | `... --stage E8` | 40h | `$RUN_ROOT/E8/*` |

**续训**（平台停止任务后不能直接续跑，必须新建任务）：

```bash
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E3 --resume
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
