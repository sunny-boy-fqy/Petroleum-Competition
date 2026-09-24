# v4：深度学习单主干、评分对齐损失、全量自包含管线

> **一句话**：用 1× Ascend 910B 64 GB 训练一条**纯深度学习**管线（深度序列主干 +
> **逐目标原子头 `q_por/q_perm/q_sw`** + 联合占位辅助头 `q_joint` + 逐目标连续头），
> 以**与官方评分同构的可微损失**优化，解码用**逐目标硬切换 `τ_t`**（阈值只在 inner-OOF
> 上按官方总分选），最终提交**自包含、CPU 可推理**的模型包。

> **文档入口**：[`docs/README.md`](docs/README.md) 是 v4 文档中心；
> [`PLAN.md`](PLAN.md) 是唯一权威总计划；[`docs/CODE_REVIEW_STATUS.md`](docs/CODE_REVIEW_STATUS.md)
> 记录代码审查发现的当前修复/遗留状态。
>
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

> **参考项目与致谢（必读）**：v4 的冲分计划在方法论上参考了 SPWLA PDDA SIG **2021 PDDA Machine Learning
> Competition** 的公开方案与赛后论文，尤其是冠军 **UTFE** 的“类型井选择 + 井间自适应”、
> **MoLPhy** 的“MICE 插补 + KS 代表采样 + 测试输入分布匹配 + 链式/Stacking”、
> **Atwah_Analytics** 的岩石物理特征工程、**Tomsk** 的后处理规则。我们**只参考方法与工程思路，
> 不复制其数据、标签或代码**；Volve 数据与其许可归 Equinor / 原仓库所有。
> 详见 [`docs/SPWLA2021_REVIEW.md`](docs/SPWLA2021_REVIEW.md) 与 [`资料引用索引.md`](资料引用索引.md)。

---

## 0. 先读：当前状态与已知边界

**本目录 `v4/` 是独立 git 仓库根，也是当前唯一活跃交付源。**

| 项目 | 状态 |
|---|---|
| 代码实现 | E1–E10 模型/训练/推理/打包代码已实现；本次重构后全量测试 **889 项通过（跳过 3）** |
| 本地口径层 | E0 数据契约、评分复算、折指纹、提交契约已通过 |
| 云端训练 | **未完成**：正式 5 折 OOF、E1–E10 数值 Gate、Ascend 实机验证仍需平台任务 |
| 最终提交 | **未生成**：需先完成云端训练与 E10 打包，且以官方 `result.zip` 为准 |
| 代码审查 | 原始报告基线 `3fa65da`；当前 checkout 已包含 C1/H1–H7/M1–M8 的主要修复，但仍有少量遗留项 |

**当前最重要的三条文档**：

1. [`PLAN.md`](PLAN.md)：技术路线、数据契约、Gate 规则、预算与风险，冲突时以它为准。
2. [`docs/README.md`](docs/README.md)：按角色和任务组织的 v4 文档地图。
3. [`docs/CODE_REVIEW_STATUS.md`](docs/CODE_REVIEW_STATUS.md)：审查发现逐条闭环状态；提交前必须复查。

> **不要混淆“代码已实现”和“成绩已达标”**。所有尚未在云端复算的分数、Gate、OOF
> 都只应写成“待云端验证”。

---

## 1. 快速开始

### 1.1 云端一键启动（推荐）

```bash
# 平台启动命令：跑到所有阶段
bash "$(find /code/workspace -name start.sh | head -1)" --to all

# 只跑到 E3-main；自动补 env/data/e0/E1/E2
bash "$(find /code/workspace -name start.sh | head -1)" --to E3-main

# 只跑单个阶段/单个实验（自动补前置）
bash "$(find /code/workspace -name start.sh | head -1)" --stage E8 --target all
bash "$(find /code/workspace -name start.sh | head -1)" --wp atom-decision
```

`start.sh --help` 看全部参数；`start.sh --list` 看所有可用 `--to/--stage/--wp`。
更详细的任务字段见 [`docs/training_tasks.md`](docs/training_tasks.md)。

### 1.2 本机准备与推送协议

> **每次开跑前必须先 push**：平台只克隆**已 push** 的代码；漏推会让平台静默运行旧代码。
> 本机 remote 与 SSH 认证已配置：
> `origin` = `git@github.com:sunny-boy-fqy/Petroleum-Competition.git`。
> 平台【仓库地址】字段必须填 HTTPS：
> `https://github.com/sunny-boy-fqy/Petroleum-Competition.git`（平台侧没有你的 SSH key）。
> 远端 `main` 与 `master` 同指一个 commit，因此一次推两个分支。

```bash
# 【本机】① 生成数据分发包（不进入 git，上传到平台云盘）
python3 tools/pack_dataset.py

# 【本机】② 校验冻结引用件（折文件 / 数据指纹）
python3 tools/verify_reference.py

# 【本机】③ 提交并推送（一次更新两个远端分支）
git add -A
git commit -m "..."
git push origin HEAD:master HEAD:main
git log --oneline -1
```

### 1.3 平台常见任务序列

```bash
# 云端① 环境与磁盘自检
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode env
# 云端② 部署数据集到 /data
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode data
# 云端③ 口径复算 E0
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode e0
# 云端④ 极小规模冒烟
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode smoke
# 云端⑤ 单阶段训练（可续训）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E1 --resume
```

> **恢复训练前清理暂停标志**：平台 `SIGTERM` 或手工 `touch pause.flag` 后，本地会留下
> `/code/workspace/v4/state/pause.flag`。`run_train.sh --mode all/stage/smoke` 启动时会自动清理它，
> 避免新任务在第一个 epoch 边界又立即 `TrainingPaused`。如果确实要保留，设置 `V4_KEEP_PAUSE_FLAG=1`。

### 1.4 本机无 torch 也可跑的口径层

```bash
# 本机：生成/校验数据包与折指纹
python3 tools/pack_dataset.py
python3 tools/verify_reference.py
python3 E0/code/run_all.py
python3 predict.py --use-version CONST --data_dir ../data --output /tmp/result.json
```

---

## 2. 环境、路径与产物边界

| | 本机（开发机） | 云端（Intern InkStone 训练任务） |
|---|---|---|
| 角色 | 写代码、生成数据包、跑口径层单测、组装提交包 | 训练、OOF 推理、集成 |
| 硬件 | 无 NPU/GPU、系统 Python 3.12（仅口径层） | **1× Ascend 910B 64 GB HBM**、4000m vCPU、**16 GiB 系统内存**、**30 GB 云盘 `/data`** |
| 软件 | 系统 Python 3.12（仅口径层） | **CANN 8.3rc2 / PyTorch 2.8.0 + torch_npu 2.8.0 / Python 3.11 / arm64**（平台镜像预装；不得改 torch/torch_npu 版本） |
| 路径 | `../data`、`./reports` | 代码根 `/code/workspace`；训练期运行时写本地高速盘；最终模型写 `/data/v4/final` |

**四条铁律**

1. `src/data/`、`src/score.py`、`src/inference/` **不 import torch**，本机也能校验数据与提交契约。
2. 数据按井分片/打包传递；云端不把整井序列常驻内存，`num_workers=4`，每 epoch 调 `assert_disk_headroom(8.0)`。
3. 一切阈值/权重/早停**只在 inner-OOF 上选**；outer 折只推理一次；A 榜只做短名单仲裁。
4. **训练期只写本地 runtime 目录**；`/data` 只放上传数据包、镜像产物与最终模型。

```
/code/workspace/                         <- Git 仓库根（zip 上传时没有外层文件夹）
$V4_LOCAL_ROOT/v4/data/                  <- 本地解压数据（本地高速盘）
$V4_LOCAL_ROOT/v4/{cache,runs,reports,logs,tb,state}/  <- 缓存/checkpoint/报告/日志/状态
/data/v4_data.tar.gz                     <- 网络盘：数据分发包
/data/v4/final/                          <- 网络盘：最终模型
/data/v4/submission/                     <- 网络盘：可选，E10 提交包
/data/v4/mirror/                         <- 网络盘：checkpoint/OOF/报告增量镜像
```

> 默认本地根优先 `/code/workspace`，再回退 `/workspace` 或 `$HERE/.v4_runtime`。
> 可用 `V4_LOCAL_ROOT` 指定本地盘，用 `V4_NETWORK_ROOT` 指定网络盘（默认 `/data`）。

---

## 3. 运行入口与模式

平台【启动命令】只填一句：

```bash
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode all
```

`run_train.sh` 的模式：

- `env`：环境 + 磁盘自检 + 安装轻量依赖；
- `data`：部署数据集到本地 runtime（只做一次）；
- `e0`：数据卡、常数基线、折指纹、契约自检；
- `smoke`：极小规模冒烟；
- `stage --stage E1`：训练单个阶段；
- `all`：**env → data → e0 → E1 → E2 → … → E10 全链路串行**；每个阶段自动带 all 子路由
  （E3 主模型 + 感受野消融 + 行级对照、E4 三消融、E5 三目标、E6 P0/P1/P2、E8 四路、E9/E10 全部子阶段）。
  `--through N` 只跑到第 N 个任务（1~14，默认 14），已完成任务自动跳过；`--fresh` 强制从头重跑；
  任一步失败立即退出；进度写入 `$STATE_DIR/all_pipeline_progress.json`。

首次上手顺序见 [`docs/platform_setup.md`](docs/platform_setup.md) 第“四、创建训练任务”节。

---

## 4. 执行顺序与量化目标

```
E0 契约 → E1 行级基线 → E2 特征 → E3 序列主干 → E4 多尺度 → E5 逐目标精修
   → E6 逐目标原子层次 + 原子门（两阶段训练，PD1 诞生）→ E7 损失与解码 → E8 集成（EMA/SWA/快照）
   → E9 验证与护栏 → E10 打包提交 → E11 归档
```

| 里程碑 | 本地 OOF Total（80 井按井 5 折，`drop` 口径） |
|---|---:|
| 全常量基线 | 70.490735 |
| 历史锚点（v1 E7 / 树模型） | 80.382479（A 榜 82.2757） |
| E1 行级基线硬 Gate | ≥ 78.0 |
| E3 序列主干硬 Gate | ≥ 81.0 |
| E6 PD1 硬 Gate | ≥ 82.0 |
| E10 冲刺目标 | ≥ 82.5 |

> 表中 Gate 是设计目标；只有云端报告才能证明是否达标。

---

## 5. 当前状态（代码/文档/分数必须分开看）

- [x] **计划全部完成**：总计划 918 行 + 12 个阶段计划（791 行）+ 33 个 P 级详细计划（5,259 行），**合计 6,968 行**（由 `tools/plan_stats.py` 实测）
- [x] 状态台账 `versions/status.json`、候选注册表 `versions/candidates.json`、目录总览 `docs/PROJECT_FILES.md`
- [x] 环境/磁盘自检脚本（`E0/code/check_env.py`、`src/data/disk_guard.py`、`E0/code/setup_deps.sh`）
- [x] 锁文件与 Gate/引用模板
- [x] **E0 口径层已实现并通过本地契约 Gate 13/13**（`reports/E0_local_contract_gate.json`）：
      按表头名对齐解析、三状态标签判据、官方评分器 drop 口径 70.490735、按井 5 折与指纹、
      分片缓存、提交契约校验、`predict.py` CPU 端到端冒烟
- [ ] E0 云端 Gate（`env_hard_checks_passed` + `disk_budget_ok`）—— 需在平台 Ascend 任务运行
- [x] **E1–E10 模型与训练代码已全部实现**（`E1/code` … `E10/code` + `src/`；本地链路/契约测试通过）
- [ ] E1–E10 正式 5 折 OOF 与数值 Gate —— 待云端运行；不要把“代码已实现”误读为“成绩已达标”
- [ ] E10 最终提交包与官方干净目录复现 —— 待云端训练完成后生成并验证
- [x] **可复用成果跨任务持久化**：`tools/artifact_store.py` 自动把 `cache/raw`、`cache/feat`、
      `runs/**/*.pkl`、提交包、候选/注册表写入 `/data/v4/artifacts`，新机器启动时自动恢复
- [x] **E2→E3+ 自动特征选择**：`tools/select_feature_spec.py` 读取 `E2_ablation.json`，
      自动把唯一 ADOPT 的特征组（当前为 `F1+win`）传给 E3–E8/E10，不再全程默认 F1

**审查闭环**：原始审查发现 1 个严重交付缺陷、7 个高优先级、8 个中优先级问题；其中绝大多数已由
`6b2a580`、`5faf4e1`、`fafaf7c` 等提交修复。当前仍需特别关注的遗留项见
[`docs/CODE_REVIEW_STATUS.md`](docs/CODE_REVIEW_STATUS.md)。

---

## 6. E0 已经查出的六个硬事实

详见 [`E0/docs/data_card.md`](E0/docs/data_card.md)。

1. **3 口训练井表头不是官方 17 列**：`42f2870b`（20 列）、`b7eb1274`（21 列）、
   `c7611b01`（16 列，缺 CASE）→ 必须按表头名对齐解析，禁止按列位置。
2. **评分分母口径 = 逐目标排除缺测行（`drop`）**：常数基线 70.490735；全行分母为 69.843218。
3. **输入列泄漏事故已修复**：现为 13 条曲线 + DEPTH 分离，并强制 `input_no_label_leak` 检查。
4. **SW 是单一标签尺度（百分数）**：有效 SW 实测 8.305–99.9，`SW<1` 的行数为 0；禁止全局压到 `[0,1]`。
5. **逐目标原子事件多于联合原子事件**：SW 518,255 / PERM 494,598 / POR 487,382，
   联合只有 487,225 → 必须用逐目标原子头 `q_por/q_perm/q_sw`，不能只靠 `q_joint`。
6. **分片缓存证据可复现**：`E0_data_card.json::shard_cache.cache_root` 必须是
   `$V4_CACHE_ROOT` 或仓库相对路径，不能是 `/tmp` 临时路径。

---

## 7. 文档地图

| 文档 | 内容 |
|---|---|
| [`docs/README.md`](docs/README.md) | **v4 文档中心**：按角色/任务分类的全部入口 |
| [`PLAN.md`](PLAN.md) | 总计划（架构、约束、Gate、风险、预算）——唯一权威 |
| [`docs/CODE_REVIEW_STATUS.md`](docs/CODE_REVIEW_STATUS.md) | 代码审查发现逐条状态与剩余 blocked 项 |
| [`docs/platform_setup.md`](docs/platform_setup.md) | 平台落地：代码/数据/产物路径、任务字段、续训与故障排查 |
| [`docs/image_requirements.md`](docs/image_requirements.md) | 镜像需求：基础镜像、轻量依赖、禁装清单 |
| [`docs/dependencies.md`](docs/dependencies.md) | 依赖声明与版本锁定策略 |
| [`docs/training_tasks.md`](docs/training_tasks.md) | 训练任务配置表（逐字段照抄） |
| [`docs/gate_template.md`](docs/gate_template.md) | Gate 预注册模板与判定逻辑 |
| [`docs/artifact_reuse.md`](docs/artifact_reuse.md) | 跨任务复用：cache / 折结果 / 提交包 / 状态自动持久化与恢复 |
| [`docs/PROJECT_FILES.md`](docs/PROJECT_FILES.md) | 目录树、文件用途、进不进 git、云端边界 |
| [`docs/SCORE_MAX_PLAN.md`](docs/SCORE_MAX_PLAN.md) | 冲分优化计划 WP0–WP11 |
| [`docs/feature_spec_auto_select.md`](docs/feature_spec_auto_select.md) | E2→E3+ 自动选择最优特征版本（F1+win 等） |
| [`docs/SPWLA2021_REVIEW.md`](docs/SPWLA2021_REVIEW.md) | SPWLA 2021 复盘与 v4 启示 |
| [`docs/AUDIT_FIXES.md`](docs/AUDIT_FIXES.md) | 第二轮审查修复清单（历史记录） |
| [`E0/docs/data_card.md`](E0/docs/data_card.md) | E0 数据卡与硬事实 |
| [`versions/status.json`](versions/status.json) | 阶段/P 级执行状态（唯一事实源） |
| [`versions/candidates.json`](versions/candidates.json) | 候选台账（未登记不得提交） |
| [`versions/registry.json`](versions/registry.json) | 可运行版本注册表 |
| [`资料引用索引.md`](资料引用索引.md) | 每一处外部引用的可核验定位 |

---

## 8. 参考项目、贡献与致谢

本项目在方法论上参考以下公开工作；参考的是**方法、流程与经验教训**，不是数据或代码：

| 来源 | 贡献/被参考的部分 | 在 v4 中的落点 |
|---|---|---|
| SPWLA PDDA SIG 2021 竞赛及赛后论文 | 任务设定、前五名方案复盘、“数据适配比模型更重要” | `docs/SPWLA2021_REVIEW.md`、`docs/SCORE_MAX_PLAN.md` |
| 冠军 UTFE | 类型井选择（KL/DTW）、井间自适应、分区沙/泥基线、半监督插值 | `src/data/type_well.py`、`src/data/well_adapt.py` |
| 亚军 MoLPhy | MICE/LGBM 插补、Kennard-Stone 代表采样、测试输入分布匹配、链式目标、SuperLearner | `src/data/impute.py`、`src/validation/representative.py`、`src/training/chained.py`、`src/ensemble/stacking.py` |
| 第 4 名 Atwah_Analytics | Archie/Simandoux/Indonesia、多骨架密度孔隙度、Larionov/Steiber/Clavier Vsh、Klogh、Boruta | `src/features/physics_ext.py` |
| 第 2/4/5 名 | 树模型与 GBDT 集成（ExtraTrees/CatBoost/XGBoost/LightGBM） | `src/models/gbdt.py`、`src/ensemble/stacking.py` |
| 第 3 名 Tomsk | 异常值处理 + 基于 EDA 的后处理规则 | `src/data/outliers.py` |

**没有复制**：Volve 数据、标签、参赛 notebook 的具体代码、SPWLA 的固定阈值/常量。
引用与许可边界详见 [`资料引用索引.md`](资料引用索引.md) 与 [`docs/SPWLA2021_REVIEW.md`](docs/SPWLA2021_REVIEW.md)。

```bibtex
@article{fu2024well,
  title={Well-Log-Based Reservoir Property Estimation With Machine Learning: A Contest Summary},
  author={Fu, Lei and Yu, Yanxiang and Xu, Chicheng and Ashby, Michael and McDonald, Andrew and Pan, Wen and Deng, Tianqi and Szab{\'o}, Istv{\'a}n and Hanzelik, P{\'a}l P and Kalm{\'a}r, Csilla and others},
  journal={Petrophysics}, volume={65}, number={01}, pages={108--127}, year={2024},
  publisher={SPWLA}
}
```
