# v4 在 Intern InkStone（discovery）平台上的落地

> **文档导航**：[v4 文档中心](README.md) · [v4 README](../README.md) · [总计划](../PLAN.md) · [代码审查状态](CODE_REVIEW_STATUS.md)
> **文档类型**：平台落地运行手册。云端目录、任务字段与故障排查；运行入口以 `run_train.sh` / `start.sh` 为准。


> 依据平台官方文档：
> [训练任务](http://discovery-staging.intern-ai.org.cn/docs/workbench/training)、
> [镜像](http://discovery-staging.intern-ai.org.cn/docs/workbench/images)、
> [我的开发机](http://discovery-staging.intern-ai.org.cn/docs/workbench/machine)、
> [为开发机准备数据](http://discovery-staging.intern-ai.org.cn/docs/workbench/data-prepare)。
> 本文把这些约定落到 v4 的具体配置上。**平台文档若与本文件冲突，以平台文档为准。**

> **交付纪律（每次任务完成后必须执行，缺一不可）**：
> 1. 先更新文档（README/PLAN/status/阶段报告等）；
> 2. `git add -A`；
> 3. `git commit`；
> 4. `git push origin HEAD:master HEAD:main`；
> 5. **清除 `/mnt/d/tmp/Petroleum-Competition/` 下的旧 zip**（例如 `rm -f /mnt/d/tmp/Petroleum-Competition/*.zip`；`tools/pack_code_zip.py` 默认也会清旧包）；
> 6. 运行 `python3 tools/pack_code_zip.py` 生成新的完整代码 zip 到
>    `/mnt/d/tmp/Petroleum-Competition/`（WSL 路径，对应 Windows `D:\tmp\Petroleum-Competition\`）。
>
> **没有完成“更新文档 + add + commit + push + 清旧 zip + 打包新 zip”这六步，任务不算完成。**
> **交付目录必须只有本次最新 zip；旧 zip 不清理视为任务未完成。**

---

## 一、平台关键约定（决定了 v4 的目录设计）

| 平台约定 | 出处 | 对 v4 的影响 |
|---|---|---|
| 训练任务代码来源三选一：**Git 仓库 / 本地上传 / 我的云盘** | 训练任务文档 §选择代码来源 | v4 用 **Git 仓库**（本仓库，**HTTPS** 地址） |
| Git 仓库代码被复制到**临时**目录 `/code/workspace`，任务结束即丢；本仓库 zip 上传后，`run_train.sh` 直接位于 `/code/workspace/`，没有外层仓库目录 | 训练任务文档 §Git 仓库 | 仍用 `$(find /code/workspace -name run_train.sh | head -1)` 定位；训练期 runtime 写 `$V4_LOCAL_ROOT/v4/*`，最终模型 publish 到 `/data` |
| **云盘挂载在 `/data`**，任务结束/资源释放后仍保留 | 同上 + §云盘持久化 | 数据、缓存、checkpoint、日志、报告**全部写 `/data/v4/...`** |
| 启动命令最长 **500 字符** | 训练任务文档 §填写启动命令 | 所有编排逻辑封装进 `run_train.sh`，启动命令只有一句 |
| 单次任务运行时长最长 **7×24 h** | 训练任务文档 §设置运行时长 | 与 D1（用户放宽 100h）兼容；但仍要求 checkpoint 可续训 |
| TensorBoard 日志写到环境变量 `TENSORBOARD_LOGDIR` 指向目录 | 训练任务文档 §TensorBoard 日志 | `run_train.sh` 已导出 `TENSORBOARD_LOGDIR=/data/v4/tb`（**持久**）；训练脚本用 `src/training/tb_logger.py::RunLogger` 写入，平台任务详情页可见迭代曲线 |
| 停止任务后**不能直接续跑**，只能用【重新训练】新建任务 | 训练任务文档 §常见问题 3 | 必须实现 `--resume` 并从 `$V4_LOCAL_ROOT/v4/runs/<stage>/last.pt` 恢复 |
| 本地上传的代码包 ≤ 500 MB，**禁止包含模型权重与大数据集** | 训练任务文档 §本地上传 | v4 的 git 仓库只放代码 + 31 MB 数据 tarball 的**生成脚本**；数据集单独上传到云盘 |
| 平台**最多创建 5 个镜像**；镜像使用场景（开发机 / 训练任务）互不通用 | 镜像文档 §常见问题 4、6 | 只构建**一个**「训练任务」场景镜像；开发机侧用官方镜像即可 |
| 镜像可用【快捷安装】(apt/pip) 或【Dockerfile 编辑】 | 镜像文档 §配置构建方式 | v4 用快捷安装即可（只需 pip 轻量包） |

> **克隆目录名不要猜（重要）**：本 git 仓库的**根就是 `v4/` 的内容**（`run_train.sh`、
> `src/`、`E0/`… 直接位于仓库根），所以平台解压出来的目录**不叫 `v4`** ——
> 本场景：仓库根直接是 `/code/workspace/`，没有 `v4/` 或仓库名外层目录。
> 因此**所有启动命令一律写成位置无关形式**：
>
> ```bash
> bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env
> ```
>
> 需要引用仓库内其它文件时先取根目录：
>
> ```bash
> V4="$(dirname "$(find /code/workspace -name run_train.sh | head -1)")"
> python3 "$V4/E0/code/check_env.py" --profile full
> ```
>
> `run_train.sh` 内部用 `$HERE` 自定位，所以仓库放在哪里都能跑；`--mode env` 的日志会打印
> 实测的 `repo(HERE) = ...`，想改用具体路径时照抄那一行即可。

---

## 二、两机三处：代码 / 数据 / 产物

```
本机（开发机，无 GPU）                     云端（Intern InkStone）
├── 写代码、跑口径层单测                    ├── /code/workspace/           ← zip 仓库根（临时！）
├── tools/pack_dataset.py 生成 31 MB 数据包  ├── $V4_LOCAL_ROOT/v4/data/    ← 本地高速盘（训练期）
└── git push                               ├── $V4_LOCAL_ROOT/v4/cache/   ← 本地高速盘（训练期）
                                           ├── $V4_LOCAL_ROOT/v4/runs/    ← 本地 checkpoint/OOF
                                           ├── $V4_LOCAL_ROOT/v4/reports/ ← 本地 Gate 报告
                                           └── /data/v4/final/            ← 网络盘：最终模型
```

**环境变量契约**（`run_train.sh` 会设置好，Python 侧读取）：

| 变量 | 默认 | 用途 |
|---|---|---|
| `V4_LOCAL_ROOT` | 自动：`/code/workspace`，回退 `/workspace` / `$HERE/.v4_runtime` | 本地高速盘运行时根 |
| `V4_NETWORK_ROOT` | `/data` | 网络盘：只读 tarball + 最终模型回写 |
| `V4_DATA_ROOT` | `$V4_LOCAL_ROOT` | 训练期数据根；数据位于 `$V4_DATA_ROOT/v4/data/{train,test}` |
| `V4_RUN_ROOT` | `$V4_DATA_ROOT/v4/runs` | checkpoint / OOF（本地） |
| `V4_CACHE_ROOT` | `$V4_DATA_ROOT/v4/cache` | 特征与张量缓存（本地） |
| `V4_REPORTS_DIR` | `$V4_DATA_ROOT/v4/reports` | Gate 报告 / 数据卡（本地） |
| `V4_LOG_DIR` | `$V4_DATA_ROOT/v4/logs` | 训练日志（本地） |
| `TENSORBOARD_LOGDIR` | `$V4_DATA_ROOT/v4/tb` | 平台迭代曲线（本地） |
| `V4_REPO_ROOT` | `$HERE` | 本次任务的代码目录（临时） |

---

## 三、一次性准备（按顺序）

### 步骤 0：本机生成数据分发包

```bash
cd <项目根>
python3 v4/tools/pack_dataset.py            # 需要 numpy
# -> v4/dist/v4_data.tar.gz          (~30 MB)
# -> v4/dist/v4_data_manifest.json   (逐文件 sha256 + 行数 + 畸形 schema 清单)
```

> `v4/dist/` 默认被 `.gitignore` 忽略（30 MB 二进制不适合进 git）。
> 把它上传到平台**云盘**（即可在 `/data` 看到）即可，推荐放到 **`/data/v4_data.tar.gz`**
> （`$DATA_ROOT` 根，`bootstrap_data.sh` 会自动搜索；见步骤 2 的定位协议）。
> 若放到别处（例如 `/data/uploads/v4_data.tar.gz`），必须显式传 `--tarball <路径>`。

### 步骤 1：构建训练任务镜像（`v4/docs/image_requirements.md`）

只需一次。核心是把 pip 轻量包烘进镜像，这样每个训练任务都能省掉安装时间。
**不要**在镜像里安装/升级 `torch`（保留平台预装版本）。

需要的 pip 包（**由用户自己安装；agent 无法自动安装任何包**，一行一个；版本不钉死）：

```
numpy
pandas
scipy
scikit-learn
einops
tensorboard
```

> 前 5 个是 required，与 `v4/E0/code/check_env.py::REQUIRED_PY_DEPS` 严格一致
> （有单测锁定两者相等）；`tensorboard` 可选，仅用于平台"迭代曲线"观测，缺失自动降级为 JSONL。
> **不需要** `pyarrow`（分片缓存是 `.npz`）、`onnx`/`onnxruntime`（CPU 推理主路径是
> `torch.load(map_location="cpu")`）。

### 步骤 2：把数据放到云盘 `/data`

**R5-B1 必读**：`dist/*.tar.gz` 与 `dist/*.json` 被 `.gitignore` 忽略，所以云端从 Git
克隆出来的 `dist/` **不可能**有 tarball（仓库根 = v4 的内容）。你必须把 tarball 放到云盘，
`bootstrap_data.sh` 会按下面的顺序找它：

| 优先级 | 位置 | 说明 |
|---|---|---|
| 1 | `--tarball <path>` 或 `V4_DATA_TARBALL=<path>` | 显式指定，指向任何位置 |
| 2 | `$V4/dist/v4_data.tar.gz` | 本机开发（`pack_dataset.py` 的产物） |
| 3 | **`$DATA_ROOT/v4_data.tar.gz`** | **云端推荐**：直接把 tarball 传到 `/data` 根 |
| 4 | `$DATA_ROOT/dist/v4_data.tar.gz` | 云端：保留 `dist/` 目录结构上传 |

manifest（`v4_data_manifest.json`，≈23 KB）同法搜索；**建议与 tarball 一起上传**，
这样会额外校验 tarball 的 sha256。**没有 manifest 也不会放宽**：80/10 井与
730,268/95,948 行是**无条件**硬校验（R5-H2）。

两种上传方式任选：

- **A. 平台云盘上传**：把 `v4_data.tar.gz`（可选带上 `v4_data_manifest.json`）传到云盘
  **`/data/` 根目录**，然后在一个训练任务里执行
  `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode data`；
- **B. 开发机直传**：在「我的开发机」里把文件放到云盘目录（`/data`）后再执行同一命令。

若文件在别的位置（例如 `/data/uploads/v4_data.tar.gz`），显式传参即可（启动命令仍 ≤500 字符）：

```bash
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode data --tarball /data/uploads/v4_data.tar.gz
```

校验（应输出 `RESULT: OK`，含 80/10 井、730,268/95,948 行、3 口畸形井 `OK`）：

```bash
V4="$(dirname "$(find /code/workspace -name run_train.sh | head -1)")"
V4_DATA_ROOT=/data bash "$V4/tools/bootstrap_data.sh"
```

> 数据只需部署**一次**；`/data` 持久保留，后续任务直接用。

### 步骤 3：推送代码（**每次开跑前都必须做**）

远程仓库与 SSH 认证**已经配置完成**，不需要再 `git init` / `git remote add`：

| 项 | 值 |
|---|---|
| remote（**仅本机 push 用**） | `origin` = `git@github.com:sunny-boy-fqy/Petroleum-Competition.git` |
| 认证（**仅本机**） | SSH key（`~/.ssh/id_rsa`）；已验证 `ssh -T git@github.com` 返回 `Hi sunny-boy-fqy!` |
| 分支 | 远端同时维护 **`main`** 与 **`master`**，两者指向**同一个 commit**（`main` 是平台「分支」字段的默认值，见 §四） |

```bash
cd /home/fangqiyu/projects/Petroleum-Competition/v4
git status --short                       # 应为空（干净工作树）
git push origin HEAD:master HEAD:main    # 一次推送同时更新两个分支（防止 main 变旧）
git log --oneline -1                     # 记下这个 revision —— 平台任务跑的就是它
```

> **⚠️ 平台【仓库地址】≠ 本机 remote（容易混，但别夸大后果）**
> 平台的 Git 克隆发生在**平台侧**，那里**没有**你的 SSH 私钥，所以平台字段必须填
> **HTTPS**：`https://github.com/sunny-boy-fqy/Petroleum-Competition.git`。
> 补充（2026-09-20 读前端 bundle 得到）：表单的正则
> `/^(https?:\/\/|git@)[\w\-.~/]+(\.git)?$/i` **会直接拒绝** scp 形式的
> `git@github.com:owner/repo.git`（`:` 不匹配），也就是说这个错误根本提交不了；
> 本机 push 继续用 SSH 地址（已配置），两者用途不同，不要混。
> 任务"失败但无日志"的排查见 §六-8（**不要**预设是这一条导致的）。

> **协议（agent 必须遵守）**：每次让用户在平台上开跑训练/评测任务前，agent 给出的操作
> 清单**必须包含**上面这条 `git push`（**双分支**），并写明待推送的 revision。平台只克隆
> **已 push** 的代码，漏掉这一步时任务会静默跑在旧代码上（日志里看不出来）—— 这是本流程
> 最容易错、也最难察觉的一步。见 `PLAN.md` §3.1「代码同步协议」。

---

## 四、创建训练任务（逐字段填写）

| 字段 | 填什么 |
|---|---|
| 任务名称 | `v4-E0-env-check` / `v4-E1-row-baseline` / … |
| 代码来源 | **Git 仓库** |
| 仓库地址 | **HTTPS**：`https://github.com/sunny-boy-fqy/Petroleum-Competition.git`（平台侧无 SSH key；scp 形式 `git@github.com:...` 会被表单正则直接拒绝） |
| 分支 | **`main`**（平台默认值）；远端 `main` 与 `master` 同指一个 commit，填 `master` 也能拉到 |
| **启动命令**（≤500 字符） | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode all` |
| 资源配置 | **Ascend 910B \* 1**（64 GB HBM），4000m vCPU / 16 GiB 内存 |
| 镜像 | 【我的镜像】→ `v4-train-py311-torch280-npu280-cann83rc2`（步骤 1 构建）；未构建则先用官方 PyTorch 2.8.0 + torch_npu 2.8.0 / CANN 8.3rc2 / Python 3.11 / arm64 镜像 |
| 训练数据集 / 验证数据集 | 可不挂载（数据在云盘 `/data`）；若平台数据集功能里有原始井数据，可挂载后在 `run_train.sh` 里加 `--from-dir` |
| 超参数 | 可不填（v4 全部走 `run_train.sh` 的参数）；如平台要求，填 `mode=all` |
| 运行时长 | E0 自检/数据部署：0h30m；E1：2h；E3：20h；E8：40h（软预算，见 PLAN §3.2） |

### 首次运行建议顺序（每次都可独立成任务）

| # | 任务名 | 启动命令 | 预期 |
|---|---|---|---|
| 1 | `v4-bootstrap` | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env` | 打印 torch 2.8.0 / Ascend 910B / bf16 / 本地磁盘剩余；把 `E0_env.json`、`E0_disk_budget.json` 写入本地 runtime 的 `reports/` |
| 2 | `v4-data` | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode data` | 从 `/data` 读 tarball，解压到本地 `$V4_LOCAL_ROOT/v4/data`，`RESULT: OK` |
| 3 | `v4-e0` | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode e0` | 数据卡 + 常数基线 70.490735 + 折指纹 + 契约自检；`E0_gate.json` passed=true |
| 4 | `v4-smoke` | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode smoke` | 1 折 / 2 epoch / 8 井，验证训练链路（需 E1 代码实现后） |
| 5+ | `v4-E1`…`v4-E8` | `bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E1` | 按 PLAN §七 推进 |

> `--mode all` 会串行执行 env → data → e0 → **E1→E10 全链路**，每个阶段自动带 all 子路由；
> `--through N` 可只跑到第 N 个任务（1~14，默认 14）；已完成任务会根据
> `/data/v4/state/all_pipeline_progress.json` 自动跳过，`--fresh` 强制从头重跑。
> 任一步失败立即退出。全链路可能运行数十小时，建议先按任务 1–3 单独确认 env/data/e0，
> 再用 `--through` 拆成多个训练任务。

---

## 五、云盘 30 GB 的实测纪律（重要）

平台给出的 **30 GB** 是**云盘（`/data`）的持久化配额**；训练期大产物现在写本地 `$V4_LOCAL_ROOT/v4/*`，不再写 `/data`。第一个任务（`--mode env`）就会打印 `df -h` 结果。

```bash
df -h / /data /code/workspace        # 运行任务时看，确认容量与是否同一挂载点
du -sh /data/* 2>/dev/null | sort -h # 确认占用分布
V4="$(dirname "$(find /code/workspace -name run_train.sh | head -1)")"
python3 "$V4/src/data/disk_guard.py" --min-free-gb 8 \
        --report /code/workspace,/data --json "$V4_LOCAL_ROOT/v4/reports/disk.json"
```

无论实测结果如何，v4 的规则不变：

1. **训练期大产物只写本地 `$V4_LOCAL_ROOT`**（数据/缓存/checkpoint/报告/日志）；`/data` 网络盘只读 tarball、写最终模型。
2. **checkpoint 滚动淘汰**：每个 run 只留 `best.pt`、`last.pt`、`last_prev.pt`（bf16 存储，单个 ≤1.2 GB）。
3. **特征按版本目录落盘**，换版本先删旧目录；不落任何逐 epoch 的中间张量。
4. 每个 epoch 调 `assert_disk_headroom(8.0)`；<5 GB 时**保存 `last.pt` 后优雅退出**，用【重新训练】+ `--resume` 继续。
5. 若实测 `/data` 可用 < 12 GB：先只跑 `env/data/e0`，把特征缓存上限压到 0.5 GB（改为训练时现算），集成成员数上限降到 2。

**续训流程**（平台停止任务后不能直接续跑，必须新建任务）：

```bash
# 新任务的启动命令
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E3 --resume
# run_train.sh 会把 --resume 透传给训练脚本，从 $V4_RUN_ROOT/E3/last.pt 恢复
```

---

## 五之二、平台集成两件事（官方提示的落地）

### 1. TensorBoard 迭代曲线

`run_train.sh` 已导出：

```bash
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$DATA_ROOT/v4/tb}"   # -> /data/v4/tb（持久）
```

训练脚本统一用 `src/training/tb_logger.py`：

```python
from src.training.tb_logger import RunLogger
with RunLogger(f"E3_unet_fold{fold}") as log:
    log.scalars({"loss/align": a, "loss/aux": b, "loss/ph": c}, step=epoch)
    log.scalars({"score/oof_total": s, "score/acc_por": p1,
                 "score/acc_perm": p2, "score/acc_sw": p3}, step=epoch)
    log.scalar("lr", lr, step=epoch)
```

- 它会同时写 TensorBoard（若 `torch.utils.tensorboard` 可用）与 **JSONL**
  （`$TENSORBOARD_LOGDIR/<run>/scalars.jsonl`，永远可写，便于事后复算）。
- 平台镜像若不带 `tensorboard`，模块**静默降级**，不影响训练。
- 纪律：曲线只用于观测；**早停与模型选择仍以 `src/score.py` 的真实分数为准**。

### 2. 本地高速盘 + 网络盘 `/data`

训练期所有大数据都写本地 `$V4_LOCAL_ROOT/v4/`；`/data` 只保留上传数据包和最终模型：

| 路径 | 内容 |
|---|---|
| `$V4_LOCAL_ROOT/v4/data/` | 数据集（从 `/data` tarball 解压，本地） |
| `$V4_LOCAL_ROOT/v4/cache/` | 分片缓存与特征缓存（本地） |
| `$V4_LOCAL_ROOT/v4/runs/` | checkpoint（`best/last/last_prev.pt`）、OOF（本地） |
| `$V4_LOCAL_ROOT/v4/reports/` | Gate 报告、数据卡、`cloud_frozen.txt`（本地） |
| `$V4_LOCAL_ROOT/v4/logs/` | `run_train.sh` 的 stdout 日志（本地） |
| `/data/v4_data.tar.gz` | 上传的数据分发包（网络盘，只读） |
| `/data/v4/mirror/` | 每 5 分钟增量同步 checkpoint/OOF/报告（网络盘） |
| `/data/v4/final/` | 训练结束后 publish 的最终模型（网络盘） |
| `/data/v4/tb/` | TensorBoard 事件文件 + JSONL |

**网络盘只保留** `/data/v4_data.tar.gz`、`/data/v4/final/`、`/data/v4/submission/` 和小体积进度 JSON。

## 六、注意事项与已知坑

1. **本地 runtime 是临时/高速的**：训练期数据、缓存、checkpoint 写 `$V4_LOCAL_ROOT`；任务结束前会把最终模型 publish 到 `/data/v4/final`。
2. **不要在任务里 `pip install torch`**：会替换镜像内版本并消耗数 GB；`run_train.sh` 的 `setup_deps.sh` 会预检并拒绝触碰 torch/nvidia/cuda 系列。
3. **镜像场景要选「训练任务」**：选成「开发机」后训练任务里看不到该镜像（镜像文档 §常见问题 4）。
4. **平台最多 5 个镜像**：只建一个训练镜像即可，避免占满配额。
5. **数据只需部署一次**：`/data` 持久；重复执行 `--mode data` 是幂等的（会覆盖同名文件并重新校验 sha256）。
6. **回归保护**：解析器必须按**表头名**对齐（3 口井 schema 非规范），不得按列位置——详见 `E0/docs/data_card.md`。
7. **评分口径**：一律 `missing_mode="drop"`（常数基线 70.490735 命中锚点），不得用全行分母口径出报告。
8. **任务失败且日志为空 = 容器从未启动**（2026-09-20 首跑实测；**根因未定**，见下方纪律）。
   卡片的【日志】是**容器内 stdout**，容器没起来时面板只会显示「暂无日志」。注意：平台**没有「错误」这个状态**，
   官方状态枚举只有：`排队中 pending / 运行中 running / 已完成 finished /
   失败 failed / 已停止 stopped / 停止中 stopping / 删除中 deleting / 未知 unknown`
   （来源：平台前端 `trainingTask.status.*`，2026-09-20 直接读取前端 bundle 得到）。

   **平台前端的硬事实（同样读自 bundle，可用于排除假设）**：

   | 事实 | 值 | 推论 |
   |---|---|---|
   | Git 仓库地址校验正则 | `/^(https?:\/\/\|git@)[\w\-.~/]+(\.git)?$/i` | **scp 形式 `git@github.com:owner/repo.git` 会被表单直接拒绝**（`:` 不在字符类里）→ 它不可能成为"任务已创建但失败"的原因；平台侧无 SSH key，所以仍必须用 `https://…` |
   | Git 源「当前工作目录」 | 恒为 **`/code/workspace`**（云盘源为 `/data<云盘路径>`） | 命令在该目录下执行；但**代码被放在其下哪个子目录仍属未定**（官方示例保留仓库名） → 继续用 `find` 自定位 |
   | 「启动命令」校验 | **只有长度 ≤ 500**（无路径/语法校验） | 命令里的 `$( )` 不会被表单拦截 |
   | 日志空态文案 | `暂无日志`；另有 `日志连接失败` 单独文案 | 需要用户区分是"空"还是"连接失败" |

   **容器是否起来过：去【我的云盘】看 `/v4/logs/`。** `run_train.sh` 第 57 行就 `mkdir -p`
   并 `tee` 到 `/data/v4/logs/train_<mode>_<时间>.log`，所以**只要容器起来过，云盘里必有日志**
   （平台日志面板空也不影响）。云盘里没有 → 容器确实没起。

   **无日志的三种可能（按可检验性排序）**：

   | 环节 | 可能原因 | 一次实验即可判定 |
   |---|---|---|
   | 拉镜像 / 起容器 | 自定义镜像被删、构建失效，或与所选资源不兼容（官方镜像文档明确要求"检查所选资源是否与镜像环境兼容"） | 换**官方镜像**（PyTorch 2.8.0 + torch_npu 2.8.0 / CANN 8.3rc2 / py3.11 / arm64）跑同一条探针 |
   | 拉代码 | 平台侧出网到 `github.com` 失败/超时（平台内网 ≠ 本机网络；本机能 push 不代表平台能 clone） | 换**本地上传**（`dist/v4_code_src.zip`）或**我的云盘**代码源跑同一条探针 |
   | 调度 | 该规格暂无资源 | 状态应为 `排队中`，不是 `失败` |

   **零歧义探针**（不含任何 `$( )`、`|`、引号，长度 <200，排除命令解析因素）：

   ```bash
   pwd; ls -la; ls -la /code/workspace; find /code/workspace -maxdepth 3 -name run_train.sh; ls -la /data; ls -la "$V4_LOCAL_ROOT/v4/logs" 2>/dev/null; python -V
   ```

   **2×2 判定法**（两件事各做一次，5 分钟内出结论）：

   | 实验 | 代码来源 | 镜像 | 结论 |
   |---|---|---|---|
   | A | Git 仓库（HTTPS + `main`） | **官方** | 有日志 → 平台能克隆、能起容器，问题在**你的镜像**；无日志 → 拉代码出网成疑 |
   | B | **本地上传**（`dist/v4_code_src.zip`） | **你的** | 有日志 → 你的镜像没问题，问题在**平台克隆 GitHub**；无日志 → 问题在**镜像**或平台本身 |

   纪律：**根因未定之前，不要在文档里写死"就是某某导致的"**。2026-09-20 那次先被误判为
   "平台仓库地址填了 SSH"，随后被上面的正则事实证伪（表单不会放行 scp 形式）——结论必须由
   实验或平台侧证据支撑。

## 6. WP8–WP11 的可选依赖与任务参数

- E7 现在支持 `--phase loss|decode|atom|all`；`atom` 需要 E6 先产出 `inner_oof.npz`。
- E8 `--target all` 会先写 `E8_type_well.json`（类型井报告），再跑 mmoe/well/transductive/ensemble。
- 若希望启用 GBDT 成员（HistGB/LightGBM/XGBoost/CatBoost），在镜像可写层安装对应库；
  缺失时 WP11 只保留 numpy Ridge/sklearn HistGB，并显式报告 `available=false`。
- 所有新增方法仍只使用测试输入；不得使用测试标签，类型井/分布匹配必须留审计记录。


---

## 七、云端统一入口 start.sh（推荐）

把平台启动命令改成：

```bash
bash "$(find /code/workspace -name start.sh | head -1)" --to all
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--to STAGE` | 自动补前置，跑到 STAGE；STAGE 可为 env/data/e0/E1..E10/E3-main/E3-all/all/1..14 |
| `--through N` | 直接透传 run_train.sh 任务号 1..14 |
| `--stage STAGE` | 只跑单阶段；自动先补前置 |
| `--phase/--target` | 传给对应阶段的子路由 |
| `--wp NAME` | 跑 WP 实验：data-quality/petro/type-well/atom-row/atom-decision/loss-full/perm-asym/stacking/gbdt/chained/all |
| `--fresh` | 清进度，从 env 重新开始 |
| `--e1-rerun` | 清旧 E1 预注册/产物后重跑 E1（WP0 新 Gate） |
| `--dry-run` | 只打印命令 |
| `--list` | 列出全部可选值 |

`start.sh` 复用 `run_train.sh` 的 env/data/e0/阶段路由、`/data/v4/mirror` 恢复与 5 分钟状态同步；
它本身不重复实现训练逻辑。
