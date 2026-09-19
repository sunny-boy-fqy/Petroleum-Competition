# v4 项目文件与目录总览

> 用途：跨机协作（本机写代码 / 云端训练）时快速定位"什么东西在哪里、是否进 git、是否需要带到云端"。
> 自动核对：`python3 v4/tools/verify_reference.py` 校验冻结引用件；
> 文件数可用 `git ls-files | wc -l` 与 `find v4 -type f | wc -l` 对照本文档。

---

## 一、顶层结构（`v4/` = git 仓库根 = 提交根）

```
v4/
├── README.md                 总入口：环境、平台速查、当前状态
├── PLAN.md                   总计划（661 行，唯一权威）
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
│   │   ├── labels.py             三状态判据、SW 双尺度、PERM log 变换
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
│   │   └── gates.py              Gate 预注册校验与聚合判定
│   ├── versioning/registry.py    候选注册表读写
│   └── ensemble/blend.py         集成融合（E8）
│
├── E0/ … E11/                12 个阶段，每层含 PLAN.md + P*/{PLAN.md,code/,docs/}
│
├── versions/                 事实源
│   ├── registry.json            可运行版本注册表
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
├── tools/
│   ├── pack_dataset.py           生成 ~30 MB 自包含数据包
│   ├── bootstrap_data.sh         部署数据到 /data（校验 sha256 + 行数 + 3 口畸形井）
│   └── verify_reference.py       校验冻结折文件与数据指纹
│
├── artifacts/E0/folds.json   outer + inner 折导出（本地生成，可重算）
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

## 四、文件数量核对（截至 E0 完成时）

| 项 | 数量 |
|---|---:|
| git 跟踪文件 | 见 `git ls-files \| wc -l` |
| 计划文件（`PLAN.md`） | 1（总）+ 12（阶段）+ 33（P）= **46** |
| P 级计划平均篇幅 | ≈224 行 |
| 代码模块（`v4/src/**/*.py`） | 见 `find v4/src -name '*.py' \| wc -l` |
| E 层脚本（`v4/E*/code/*.py`） | 见 `find v4/E* -name '*.py' \| wc -l` |

> 本文档不手写文件数，避免与实物漂移；以命令行输出为准。
