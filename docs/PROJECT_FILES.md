# v4 项目文件与目录总览

> 用途：跨机协作（本机写代码 / 云端训练）时快速定位"什么东西在哪里、是否进 git、是否需要带到云端"。
> 自动核对：`python3 v4/tools/verify_reference.py` 校验冻结引用件；
> 文件数可用 `git ls-files | wc -l` 与 `find v4 -type f | wc -l` 对照本文档。

---

## 一、顶层结构（`v4/` = git 仓库根 = 提交根）

```
v4/
├── README.md                 总入口：环境、平台速查、当前状态
├── PLAN.md                   总计划（729 行，唯一权威）
├── 资料引用索引.md            每处引用的可核验定位
├── run_train.sh              平台训练任务统一入口（env/data/e0/smoke/stage/all）
├── predict.py                推理入口（官方 --data_dir/--output）
├── train.py                  训练入口（评测时非必需，但必须可跑）
├── requirements.txt          依赖声明（由 versions/locks/ 生成）
├── .gitignore                排除数据包/权重/缓存/运行时产物
│
├── docs/                     项目级文档
│   ├── platform_setup.md        平台落地（/code vs /data、任务配置、续训）
│   ├── image_requirements.md    镜像需求（基础镜像 + pip 清单 + 禁装清单）
│   ├── training_tasks.md        训练任务配置表（逐字段照抄）
│   ├── gate_template.md         Gate 预注册模板与判定逻辑
│   ├── PROJECT_FILES.md         本文件
│   ├── gen_plans.py             [生成器] E 层与 P 层骨架（已被 gen_p_details 取代）
│   └── gen_p_details.py         [生成器] P 级详细子计划（当前有效）
│
├── configs/
│   └── paths.yaml            路径契约（本机/云端两套默认值 + 环境变量覆盖）
│
├── src/                      项目级共享代码（**不 import torch 的口径层也在其中**）
│   ├── constants.py              冻结常量（列序/哨兵/占位/权重/锚点/预算）
│   ├── portability.py            可选依赖探测与降级（numpy/pandas/pyarrow/torch/onnx）
│   ├── score.py                  官方评分器（drop 口径）
│   ├── data/
│   │   ├── parse.py              按表头名对齐的解析器（处理 3 口非规范 schema 井）
│   │   ├── labels.py             三状态判据、SW 尺度校验、PERM log 变换
│   │   ├── dataset.py            按井分片缓存（raw/labels npz）
│   │   └── disk_guard.py         30 GB 磁盘守卫（cleanup/save_and_exit/abort）
│   ├── features/                 F1 行级特征（basic.py）+ E2 特征组（physics/window/well）
│   ├── models/                   row_mlp / unet1d / tcn / patchtf / heads / mmoe
│   ├── losses/score_aligned.py   三段式对齐损失（Charbonnier + softplus）
│   ├── inference/
│   │   ├── contract.py           提交契约校验（10 井/95,948 行/字段/有限性）
│   │   ├── atomic_gate.py        硬切换原子门（E6）
│   │   └── predictor.py          统一推理器
│   ├── validation/
│   │   ├── folds.py              按井折读取 + inner 折 + 加权 cluster bootstrap
│   │   └── gates.py              Gate 预注册校验与聚合判定（已实现）
│   ├── training/tb_logger.py     TensorBoard + JSONL 日志（平台迭代曲线；无 tensorboard 时降级）
│   ├── versioning/registry.py    版本注册表读写（predict.py 的版本来源）
│   └── ensemble/blend.py         集成融合（E8）
│
├── E0/ … E11/                12 个阶段，每层含 PLAN.md + P*/{PLAN.md,code/,docs/}
│
├── versions/                 事实源
│   ├── registry.json            可运行版本注册表
│   ├── prereg_templates/        33 份 Gate 预注册模板（通过 gates.py 校验）
│   ├── candidates.json          **候选注册表（唯一事实源）**
│   ├── status.json              **阶段/P 执行状态台账**
│   ├── folds_sha256.json        折指纹
│   ├── reference/v1_well_folds.json  冻结折文件（随 git）
│   └── locks/{cloud.txt,submit.txt}  训练/推理依赖快照
│
├── reports/                  必须进 git 的**门禁证据**（大产物在 /data）
│   ├── E0_gate.json              E0 Gate 判定（passed=true）
│   ├── E0_data_card.json         数据卡（计数/状态/schema 异常）
│   └── E0_contract_tests.json    契约自检（6 负样例）
│
├── tests/                    口径层测试（**不需要 torch**，52 项）
│   ├── run_all.py                一键运行（unittest discover）
│   ├── test_parse.py             列布局/泄漏回归/畸形井/哨兵/特征/标签（11 项）
│   ├── test_score.py             官方公式边界/两种口径/总分恒等式/锚点（11 项）
│   ├── test_contract.py          井数/每井行数/深度对齐/SW 尺度守卫（14 项）
│   └── test_gates.py             Gate 类型/绝对门槛/模板校验（16 项）
│
├── tools/
│   ├── pack_dataset.py           生成 ~30 MB 自包含数据包
│   ├── bootstrap_data.sh         部署数据到 /data（校验 sha256 + 行数 + 3 口畸形井）
│   ├── verify_reference.py       校验冻结折文件与数据指纹
│   ├── check_consistency.py      候选注册表与评分口径自洽校验
│   ├── check_status.py           状态台账与实物一致性校验
│   ├── check_data_leak.py        全量 90 井输入-标签泄漏回归
│   ├── plan_stats.py             计划行数统计（唯一事实源）
│   ├── sync_plan_stats.py        把实测行数同步进文档
│   └── gen_data_card_md.py       由 JSON 生成数据卡统计表
│
├── artifacts/E0/folds.json   outer + inner 折导出（本地便利副本，可重算；权威副本在 $V4_REPORTS_DIR/E0_folds.json）
├── dist/                     [gitignore] 数据包与清单（上传云盘用）
├── cache/ runs/ logs/ tb/    [gitignore] 云端在 /data/v4/ 下；本机默认在此
├── experiments/ models/ submission/  [gitignore] 候选产物与权重
└── E*/code/                  各 P 的执行脚本（进 git）
```

---

## 二、进不进 git

| 类别 | 进 git | 说明 |
|---|---|---|
| 计划/文档/契约/评分器/解析器/折文件 | ✅ | 复现与协作的最小集 |
| `reports/E0_*.json`（门禁证据） | ✅ | 小而关键，是"已复算"的凭据 |
| `versions/{candidates,status,folds_sha256}.json`、`versions/locks/*` | ✅ | 事实源 |
| 数据包 `dist/*.tar.gz` | ❌ | 30 MB 二进制；走平台云盘 `/data` |
| 权重 `*.pt/*.onnx`、缓存 `*.npz/*.parquet` | ❌ | 体积大；留在 `/data/v4/runs` |
| 训练日志、TensorBoard、实验中间产物 | ❌ | 运行时产物 |
| `cache/ runs/ logs/ tb/ experiments/ models/ submission/` | ❌ | 同上（`.gitignore` 已覆盖） |

---

## 三、云端的四处目录（平台约定）

```
/code/workspace/v4/            git clone（**临时**，任务结束即丢）
/data/v4/data/{train,test}/    数据集（部署一次，永久）
/data/v4/{cache,runs,reports,logs,tb}/
                              缓存 / checkpoint / Gate 报告 / 日志 / TensorBoard（永久）
```

环境变量契约见 `configs/paths.yaml` 与 `docs/platform_setup.md` §二。

---

## 三之二、已实现 vs 计划中（避免"幽灵文件"）

| 类别 | 已实现 | 计划中（未实现，不在仓库） |
|---|---|---|
| `src/` | constants, portability, score, data/{parse,labels,dataset,disk_guard}, features/basic, losses/score_aligned, models/row_mlp, inference/contract, validation/{folds,gates}, versioning/registry | features/{physics,window,well}, models/{unet1d,tcn,patchtf,heads,mmoe,state_head}, inference/{atomic_gate,predictor,decode}, ensemble/blend, losses 的物理项 |
| 根目录 | predict.py, run_train.sh, requirements.txt | train.py, configs/v4.yaml（当前为 `configs/paths.yaml`） |
| 产物 | reports/E0_*.json（7 份）、versions/registry.json、versions/candidates.json、versions/status.json、versions/folds_sha256.json、versions/prereg_templates/（33 份） | 各阶段的 E*.json |
| `run_train.sh` | `env` / `data` / `e0` 可用 | `smoke` / `stage` 需 E1 代码（当前会明确报错） |

## 四、文件数量核对（截至 E0 完成时）

| 项 | 数量 |
|---|---:|
| git 跟踪文件 | 见 `git ls-files \| wc -l` |
| 计划文件（`PLAN.md`） | 1（总，729 行）+ 12（阶段，708 行）+ 33（P，4,988 行）= **46 份 / 6,425 行** |
| P 级计划平均篇幅 | **151 行**（合计 4,988；由 `tools/plan_stats.py` 统计） |
| 代码模块（`v4/src/**/*.py`） | 见 `find v4/src -name '*.py' \| wc -l` |
| E 层脚本（`v4/E*/code/*.py`） | 见 `find v4/E* -name '*.py' \| wc -l` |

> 本文档不手写文件数，避免与实物漂移；以命令行输出为准。
