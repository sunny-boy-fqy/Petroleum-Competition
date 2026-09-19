# v4 在 Intern InkStone（discovery）平台上的落地

> 依据平台官方文档：
> [训练任务](http://discovery-staging.intern-ai.org.cn/docs/workbench/training)、
> [镜像](http://discovery-staging.intern-ai.org.cn/docs/workbench/images)、
> [我的开发机](http://discovery-staging.intern-ai.org.cn/docs/workbench/machine)、
> [为开发机准备数据](http://discovery-staging.intern-ai.org.cn/docs/workbench/data-prepare)。
> 本文把这些约定落到 v4 的具体配置上。**平台文档若与本文件冲突，以平台文档为准。**

---

## 一、平台关键约定（决定了 v4 的目录设计）

| 平台约定 | 出处 | 对 v4 的影响 |
|---|---|---|
| 训练任务代码来源三选一：**Git 仓库 / 本地上传 / 我的云盘** | 训练任务文档 §选择代码来源 | v4 用 **Git 仓库**（本仓库） |
| Git 仓库代码被复制到**临时**目录 `/code/workspace`，任务结束即丢 | 同上 | 启动命令必须基于 `/code/workspace/v4/...` 填写；**代码里不得写入需要保留的东西** |
| **云盘挂载在 `/data`**，任务结束/资源释放后仍保留 | 同上 + §云盘持久化 | 数据、缓存、checkpoint、日志、报告**全部写 `/data/v4/...`** |
| 启动命令最长 **500 字符** | 训练任务文档 §填写启动命令 | 所有编排逻辑封装进 `run_train.sh`，启动命令只有一句 |
| 单次任务运行时长最长 **7×24 h** | 训练任务文档 §设置运行时长 | 与 D1（用户放宽 100h）兼容；但仍要求 checkpoint 可续训 |
| TensorBoard 日志写到环境变量 `TENSORBOARD_LOGDIR` 指向目录 | 训练任务文档 §TensorBoard 日志 | `run_train.sh` 已导出该变量到 `/data/v4/tb` |
| 停止任务后**不能直接续跑**，只能用【重新训练】新建任务 | 训练任务文档 §常见问题 3 | 必须实现 `--resume` 并从 `/data/v4/runs/<stage>/last.pt` 恢复 |
| 本地上传的代码包 ≤ 500 MB，**禁止包含模型权重与大数据集** | 训练任务文档 §本地上传 | v4 的 git 仓库只放代码 + 31 MB 数据 tarball 的**生成脚本**；数据集单独上传到云盘 |
| 平台**最多创建 5 个镜像**；镜像使用场景（开发机 / 训练任务）互不通用 | 镜像文档 §常见问题 4、6 | 只构建**一个**「训练任务」场景镜像；开发机侧用官方镜像即可 |
| 镜像可用【快捷安装】(apt/pip) 或【Dockerfile 编辑】 | 镜像文档 §配置构建方式 | v4 用快捷安装即可（只需 pip 轻量包） |

---

## 二、两机三处：代码 / 数据 / 产物

```
本机（开发机，无 GPU）                     云端（Intern InkStone）
├── 写代码、跑口径层单测                    ├── /code/workspace/v4/   ← git clone（临时！）
├── tools/pack_dataset.py 生成 31 MB 数据包  ├── /data/v4/data/         ← 云盘（持久）
└── git push                               ├── /data/v4/cache/        ← 云盘（持久）
                                           ├── /data/v4/runs/         ← 云盘（持久，checkpoint）
                                           ├── /data/v4/reports/      ← 云盘（持久，Gate 报告）
                                           └── /data/v4/logs/         ← 云盘（持久，stdout 日志）
```

**环境变量契约**（`run_train.sh` 会设置好，Python 侧读取）：

| 变量 | 默认 | 用途 |
|---|---|---|
| `V4_DATA_ROOT` | `/data` | 数据根；数据实际位于 `$V4_DATA_ROOT/v4/data/{train,test}` |
| `V4_RUN_ROOT` | `$V4_DATA_ROOT/v4/runs` | checkpoint / OOF |
| `V4_CACHE_ROOT` | `$V4_DATA_ROOT/v4/cache` | 特征与张量缓存 |
| `V4_REPORTS_DIR` | `$V4_DATA_ROOT/v4/reports` | Gate 报告 / 数据卡 |
| `V4_LOG_DIR` | `$V4_DATA_ROOT/v4/logs` | 训练日志 |
| `TENSORBOARD_LOGDIR` | `$V4_DATA_ROOT/v4/tb` | 平台迭代曲线 |
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
> 把它上传到平台**云盘**（即可在 `/data` 看到）即可，例如放到 `/v4_data/v4_data.tar.gz`。

### 步骤 1：构建训练任务镜像（`v4/docs/image_requirements.md`）

只需一次。核心是把 6 个 pip 轻量包烘进镜像，这样每个训练任务都能省掉安装时间。
**不要**在镜像里安装/升级 `torch`（保留平台预装版本）。

### 步骤 2：把数据放到云盘 `/data`

两种方式任选：

- **A. 平台云盘上传**：把 `v4_data.tar.gz` 传到云盘，然后在一个训练任务里执行
  `bash /code/workspace/v4/run_train.sh --mode data`；
- **B. 开发机直传**：在「我的开发机」里把文件放到云盘目录（`/data`）后再执行同一命令。

校验（应输出 `RESULT: OK`，含 80/10 井、730,268/95,948 行、3 口畸形井 `OK`）：

```bash
V4_DATA_ROOT=/data bash /code/workspace/v4/tools/bootstrap_data.sh
```

> 数据只需部署**一次**；`/data` 持久保留，后续任务直接用。

### 步骤 3：推送代码

```bash
cd <项目根>/v4
git init && git add -A && git commit -m "v4: plan + E0 contract layer + platform scripts"
git remote add origin <你的仓库地址>
git push -u origin main
```

---

## 四、创建训练任务（逐字段填写）

| 字段 | 填什么 |
|---|---|
| 任务名称 | `v4-E0-env-check` / `v4-E1-row-baseline` / … |
| 代码来源 | **Git 仓库** |
| 仓库地址 / 分支 | `<你的仓库地址>` / `main` |
| **启动命令**（≤500 字符） | `bash /code/workspace/v4/run_train.sh --mode all` |
| 资源配置 | **Nvidia A100 \* 1**（80 GB 显存），4000m vCPU / 16 GiB 内存 |
| 镜像 | 【我的镜像】→ `v4-train-py311-torch240-cu126`（步骤 1 构建）；未构建则先用官方 PyTorch 2.4.0 / CUDA 12.6 / Python 3.11 镜像 |
| 训练数据集 / 验证数据集 | 可不挂载（数据在云盘 `/data`）；若平台数据集功能里有原始井数据，可挂载后在 `run_train.sh` 里加 `--from-dir` |
| 超参数 | 可不填（v4 全部走 `run_train.sh` 的参数）；如平台要求，填 `mode=all` |
| 运行时长 | E0 自检/数据部署：0h30m；E1：2h；E3：20h；E8：40h（软预算，见 PLAN §3.2） |

### 首次运行建议顺序（每次都可独立成任务）

| # | 任务名 | 启动命令 | 预期 |
|---|---|---|---|
| 1 | `v4-bootstrap` | `bash /code/workspace/v4/run_train.sh --mode env` | 打印 torch 2.4.0 / A100 sm_80 / bf16 / 磁盘剩余；把 `E0_env.json`、`E0_disk_budget.json` 写入 `/data/v4/reports/` |
| 2 | `v4-data` | `bash /code/workspace/v4/run_train.sh --mode data` | 解压数据到 `/data/v4/data`，`RESULT: OK` |
| 3 | `v4-e0` | `bash /code/workspace/v4/run_train.sh --mode e0` | 数据卡 + 常数基线 70.490735 + 折指纹 + 契约自检；`E0_gate.json` passed=true |
| 4 | `v4-smoke` | `bash /code/workspace/v4/run_train.sh --mode smoke` | 1 折 / 2 epoch / 8 井，验证训练链路（需 E1 代码实现后） |
| 5+ | `v4-E1`…`v4-E8` | `bash /code/workspace/v4/run_train.sh --mode stage --stage E1` | 按 PLAN §七 推进 |

> `--mode all` 会串行执行 env → data → e0 → 首个可用训练阶段，适合单次跑完前置检查。

---

## 五、磁盘 30 GB 的实测纪律（重要）

平台给出的 **30 GB** 是**该资源规格的磁盘配额**，而 `/code/workspace`（临时代码）与 `/data`（云盘）
是否在同一文件系统**必须实测确认**，不能假设。第一个任务（`--mode env`）就会打印 `df -h` 结果。

```bash
df -h / /data /code/workspace        # 运行任务时看，确认容量与是否同一挂载点
du -sh /data/* 2>/dev/null | sort -h # 确认占用分布
python3 /code/workspace/v4/src/data/disk_guard.py --min-free-gb 8 \
        --report /code/workspace,/data --json /data/v4/reports/disk.json
```

无论实测结果如何，v4 的规则不变：

1. **只往 `/data` 写有保留价值的东西**（checkpoint / 缓存 / 报告 / 日志）；`/code/workspace` 只读代码。
2. **checkpoint 滚动淘汰**：每个 run 只留 `best.pt`、`last.pt`、`last_prev.pt`（bf16 存储，单个 ≤1.2 GB）。
3. **特征按版本目录落盘**，换版本先删旧目录；不落任何逐 epoch 的中间张量。
4. 每个 epoch 调 `assert_disk_headroom(8.0)`；<5 GB 时**保存 `last.pt` 后优雅退出**，用【重新训练】+ `--resume` 继续。
5. 若实测 `/data` 可用 < 12 GB：先只跑 `env/data/e0`，把特征缓存上限压到 0.5 GB（改为训练时现算），集成成员数上限降到 2。

**续训流程**（平台停止任务后不能直接续跑，必须新建任务）：

```bash
# 新任务的启动命令
bash /code/workspace/v4/run_train.sh --mode stage --stage E3 --resume
# run_train.sh 会把 --resume 透传给训练脚本，从 $V4_RUN_ROOT/E3/last.pt 恢复
```

---

## 六、注意事项与已知坑

1. **`/code/workspace` 是临时的**：不要在那里放数据集副本或让训练脚本把结果写在那里（会随任务结束丢失）。
2. **不要在任务里 `pip install torch`**：会替换镜像内版本并消耗数 GB；`run_train.sh` 的 `setup_deps.sh` 会预检并拒绝触碰 torch/nvidia/cuda 系列。
3. **镜像场景要选「训练任务」**：选成「开发机」后训练任务里看不到该镜像（镜像文档 §常见问题 4）。
4. **平台最多 5 个镜像**：只建一个训练镜像即可，避免占满配额。
5. **数据只需部署一次**：`/data` 持久；重复执行 `--mode data` 是幂等的（会覆盖同名文件并重新校验 sha256）。
6. **回归保护**：解析器必须按**表头名**对齐（3 口井 schema 非规范），不得按列位置——详见 `E0/docs/data_card.md`。
7. **评分口径**：一律 `missing_mode="drop"`（常数基线 70.490735 命中锚点），不得用全行分母口径出报告。
