# v4：深度学习单主干、评分对齐损失、全量自包含管线

> **一句话**：用 1× A100 80GB 训练一条**纯深度学习**管线（深度序列主干 + **逐目标原子头 `q_por/q_perm/q_sw`** + 联合占位辅助头 `q_joint` + 逐目标连续头），以**与官方评分同构的可微损失**优化，解码用**逐目标硬切换 `τ_t`**（阈值只在 inner-OOF 上按官方总分选），提交**自包含、CPU 可推理**的模型包。

## 快速导航

| 文档 | 内容 |
|---|---|
| [`PLAN.md`](PLAN.md) | **总计划**（架构、约束、Gate、风险、预算）—— 唯一权威 |
| [`资料引用索引.md`](资料引用索引.md) | 每一处引用的可核验定位 |
| [`docs/platform_setup.md`](docs/platform_setup.md) | **平台落地**：代码/数据/产物三处路径、训练任务逐字段填写、`/data` 持久化、续训流程 |
| [`docs/image_requirements.md`](docs/image_requirements.md) | **镜像需求**：基础镜像版本锁定、快捷安装包清单、禁装清单、Dockerfile 追加层 |
| [`E0/docs/data_card.md`](E0/docs/data_card.md) | E0 数据卡与口径冻结（含两个硬发现） |
| [`docs/training_tasks.md`](docs/training_tasks.md) | **训练任务配置表**：每个任务逐字段填写 + 平台限制速记 |
| [`docs/PROJECT_FILES.md`](docs/PROJECT_FILES.md) | **目录树与文件用途**：什么进 git、什么走云盘 |
| [`versions/status.json`](versions/status.json) | **执行状态台账**（阶段/P 级状态唯一事实源） |
| [`versions/candidates.json`](versions/candidates.json) | **候选注册表**（唯一事实源，未登记不得提交） |
| [`versions/registry.json`](versions/registry.json) | 可运行版本注册表（`predict.py` 的版本来源） |
| [`versions/prereg_templates/`](versions/prereg_templates) | 33 份 Gate 预注册模板（全部通过 `gates.py` 校验） |
| [`reports/V4_PLAN_REVIEW.md`](reports/V4_PLAN_REVIEW.md) | 一审报告（B1–B6/H1–H5/M1–M10 已逐条处置） |
| [`reports/V4_PLAN_REVIEW_2.md`](reports/V4_PLAN_REVIEW_2.md) | 二审报告（R2-B1..B6 / H1..H6 / M1..M9 已逐条处置） |
| [`reports/V4_PLAN_REVIEW_3.md`](reports/V4_PLAN_REVIEW_3.md) | **三审报告**（R3-C1..C3 / H1..H5 / M1..M5 已逐条处置） |
| [`reports/V4_PLAN_IMPROVEMENT_PROPOSAL.md`](reports/V4_PLAN_IMPROVEMENT_PROPOSAL.md) | **计划改进建议**（逐目标原子层次 / 输出参数化 / 损失 / 训练策略 / 主干优化，已落进计划与代码） |
| [`docs/gate_template.md`](docs/gate_template.md) | Gate 预注册模板与判定逻辑 |
| `E0/` … `E11/` | 阶段计划与 P 级子计划 |
| `versions/locks/cloud.txt`、`versions/locks/submit.txt` | 训练/推理依赖快照 |

## 本仓库就是平台训练任务的代码来源（Git 仓库）

平台把 git 代码克隆到**临时**目录 `/code/workspace`，云盘挂载在**持久**目录 `/data`。
因此本仓库只放代码，数据/缓存/checkpoint/报告全部写 `/data`：

```
/code/workspace/v4/          <- 本仓库（临时，任务结束即丢）
/data/v4/data/               <- 数据集（一次性部署，永久保留）
/data/v4/{cache,runs,reports,logs,tb}/   <- 缓存 / checkpoint / Gate 报告 / 日志 / TensorBoard
```

平台【启动命令】只填一句：

```bash
bash /code/workspace/v4/run_train.sh --mode all
```

`run_train.sh` 的模式：`env`（环境+磁盘自检+装轻量依赖）、`data`（部署数据集到 `/data`）、
`e0`（口径复算）、`smoke`（极小规模冒烟）、`stage --stage E1`（训练）、`all`（串行全部）。

首次上手顺序见 [`docs/platform_setup.md`](docs/platform_setup.md) §四。

## 云端上手（本机准备 → 平台三次任务）

```bash
# 【本机】① 生成数据分发包（~30 MB，不进 git，上传到平台云盘）
python3 v4/tools/pack_dataset.py
# 【本机】② 校验冻结引用件（折文件 / 数据指纹）
python3 v4/tools/verify_reference.py
# 【本机】③ 推送代码
cd v4 && git add -A && git commit -m "..." && git push

# 【平台】任务1：环境与磁盘自检  bash /code/workspace/v4/run_train.sh --mode env
# 【平台】任务2：部署数据到 /data  bash /code/workspace/v4/run_train.sh --mode data
# 【平台】任务3：口径复算 E0      bash /code/workspace/v4/run_train.sh --mode e0
```

## 环境（两机分离）

| | 本机（开发机） | 云端（Intern InkStone 训练任务） |
|---|---|---|
| 角色 | 写代码、生成数据包、跑口径层单测、组装提交包 | 训练、OOF 推理、集成 |
| 硬件 | 无 GPU、`v2/.venv` 有 numpy/pandas、**无 torch** | **1× A100 80GB**、4000m vCPU、**16 GiB 系统内存**、**30 GB 磁盘** |
| 软件 | 系统 Python 3.12（仅用于口径层） | **CUDA 12.8 / PyTorch 2.7.1 / Python 3.11**（平台镜像预装，**无 conda**，不得改 torch 版本；额外轻量包可 `pip install --no-cache-dir`） |
| 目录 | `../data`、`./reports` | 代码 `/code/workspace/v4`（临时）；数据与产物 `/data/v4/*`（持久） |

**四条铁律**

1. `v4/src/data/`、`v4/src/score.py`、`v4/src/inference/` **不 import torch** —— 本机也能校验数据与提交契约。
2. 数据以按井分片 / 打包件传递；**云端不把整井序列常驻内存**（16 GiB 是瓶颈），`num_workers=4`，每 epoch 调 `assert_disk_headroom(8.0)`。
3. 一切阈值/权重/早停**只在 inner-OOF 上选**；outer 折只推理一次；A 榜只做短名单仲裁。
4. **只往 `/data` 写需要保留的东西**；`/code/workspace` 是临时的，任务结束即丢。

## 执行顺序

```
E0 契约 → E1 行级基线 → E2 特征 → E3 序列主干 → E4 多尺度 → E5 逐目标精修
   → E6 逐目标原子层次 + 原子门（两阶段训练，PD1 诞生）→ E7 损失与解码 → E8 集成（EMA/SWA/快照）
   → E9 验证与护栏 → E10 打包提交 → E11 归档
```

## 关键量化目标

| 里程碑 | 本地 OOF Total（80 井按井 5 折，`drop` 口径） |
|---|---:|
| 全常量基线（**已复算命中**） | 70.490735（锚点 70.4907） |
| 历史锚点（v1 E7 / 树模型） | 80.382479（A 榜 82.2757） |
| E1 行级基线硬 Gate | ≥ 78.0 |
| E3 序列主干硬 Gate | ≥ 81.0 |
| E6 PD1 硬 Gate | ≥ 82.0 |
| E10 冲刺目标 | ≥ 82.5 |

## 训练任务速查（Intern InkStone）

| 字段 | 值 |
|---|---|
| 代码来源 | **Git 仓库**（本仓库）；分支 `main` |
| 启动命令 | `bash /code/workspace/v4/run_train.sh --mode all` |
| 资源 | Nvidia **A100 × 1**（80 GB 显存） |
| 镜像 | 见 [`docs/image_requirements.md`](docs/image_requirements.md)（训练任务场景；torch 2.7.1 / CUDA 12.8 / py3.11） |
| 运行时长 | 自检/数据 0h30m；E1 2h；E3 20h；E8 40h（软预算） |

任务序列（每个都可独立成任务，见 [`docs/platform_setup.md`](docs/platform_setup.md) §四）：

```bash
bash /code/workspace/v4/run_train.sh --mode env    # ① 环境 + 磁盘 + 装轻量依赖
bash /code/workspace/v4/run_train.sh --mode data   # ② 解压数据到 /data/v4/data（只需一次）
bash /code/workspace/v4/run_train.sh --mode e0     # ③ 数据卡 + 常数基线 70.490735 + 折指纹 + 契约自检
bash /code/workspace/v4/run_train.sh --mode smoke  # ④ 极小规模冒烟（E1 实现后）
bash /code/workspace/v4/run_train.sh --mode stage --stage E1 --resume   # ⑤ 训练（可续训）
```

本机（无 torch）也可跑通口径层与提交契约：

```bash
python3 v4/tools/pack_dataset.py       # 生成数据包
python3 v4/tools/verify_reference.py   # 校验折文件 / 数据指纹（RESULT: OK）
python3 v4/E0/code/run_all.py          # 数据卡 + 评分复算 + 契约自检
python3 v4/predict.py --use-version CONST --data_dir ../data --output /tmp/r.json
```

## 当前状态

- [x] **计划全部完成**：总计划 863 行 + 12 个阶段计划（759 行）+ 33 个 P 级详细计划（5,221 行），**合计 6,843 行**（由 `tools/plan_stats.py` 实测）
- [x] 状态台账 `versions/status.json`、候选注册表 `versions/candidates.json`、目录总览 `docs/PROJECT_FILES.md`
- [x] 环境/磁盘自检脚本（`E0/code/check_env.py`、`src/data/disk_guard.py`、`E0/code/setup_deps.sh`）
- [x] 锁文件与 Gate/引用模板
- [x] **E0 口径层已实现并通过本地契约 Gate 13/13**（`reports/E0_local_contract_gate.json`）：按表头名对齐解析器（13 曲线输入 + 无泄漏回归）、三状态标签判据与目标分布统计、官方评分器（drop 口径 70.490735 + 恒等式校验）、按井 5 折导出与指纹、分片缓存（32.4 MB）、提交契约校验、`predict.py` 端到端冒烟（10 井 / 95,948 行 / 1.4 s CPU）
- [ ] E0 云端 Gate（`env_hard_checks_passed` + `disk_budget_ok`）—— 需在平台 A100 任务运行 `run_train.sh --mode env`
- [ ] E1–E10 模型与训练代码（E1/P0 行级管线与 E1/P1 训练器待写）
- [x] **改进 proposal 已落进计划与代码**：`PLAN.md` §5.1/§5.2.1/§5.3/§5.4/§6.4/§8.2/§10 与 E1/E3–E8 的 P 级计划；
      `src/models/row_mlp.py`（`q_joint` + `q_por/q_perm/q_sw` + `por_max·sigmoid(g)` + SW 折内仿射归一化）、
      `src/losses/score_aligned.py`（尺度归一化 `L_aux` / 边界聚焦 / 逐目标原子 BCE）、
      `src/inference/atomic_gate.py`（逐目标硬切换 `τ_t` + 平台区中点选择 + 误判代价报告）
- [x] **口径层与 Gate 语义修复（三审）**：`min_*`/`max_*` 方向分离 + 指标字段映射；`disk_guard --data-root`；
      E0 证据 `cache_root` 改为可复现形式；`plan_stats --check` 逐条断言阶段/P 分项

## E0 已经查出的六个硬事实（详见 [`E0/docs/data_card.md`](E0/docs/data_card.md)）

1. **3 口训练井（27,080 行，3.71%）的表头不是官方 17 列**：`42f2870b`（20 列，多 K/U/CGR，**含 CASE**）、`b7eb1274`（21 列，多 TH/K/U/CGR，**含 CASE**）、`c7611b01`（16 列，**唯一缺 CASE**）。
   → **必须按表头名对齐解析**，禁止按列位置（即 17,426 行"多列" + 9,654 行"缺 CASE"）。
2. **评分分母口径 = 逐目标排除缺测行（`drop`）**：常数基线在此口径下 `70.490735`（命中锚点 70.4907）；全行分母口径为 `69.843218`（低 0.65 分）。
3. **输入列泄漏事故（已修复 + 已加回归）**：`inputs = arr[:, 1:15]` 曾把 **POR 标签**当作第 14 个输入（80 口井全部泄漏），且测试井只有 13 列会导致提交崩溃。现为 **13 条曲线 + DEPTH 分离**，并强制 `input_no_label_leak` 检查。
4. **SW 是单一标签尺度（百分数）**：实测有效 SW 为 min 8.305 / median 82.805 / max 99.9，小于 1 的行数为 **0** → 取消"×100 双尺度"假设；连续头在**训练折内**做仿射归一化并反变换，仍严禁全局裁剪到小数区间。
5. **逐目标原子事件远多于联合原子事件**：`q_joint` 只有 487,225 行，而 SW 原子行 518,255 / PERM 494,598 / POR 487,382 → **单 joint 头会漏保护 31,030 个 SW 原子行**，必须用逐目标原子头（见 `PLAN.md` §5.3/§6.4）。
6. **分片缓存证据可复现**：`E0_data_card.json::shard_cache.cache_root` 现为 `$V4_CACHE_ROOT`（云端 `/data/v4/cache`）或 repo 相对路径，不再是 `/tmp` 临时路径。

> 未实现的部分在 README 与各 `PLAN.md` 中显式列出；**不宣称任何尚未复算的分数**。
