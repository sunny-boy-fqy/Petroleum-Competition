# v4 项目文件与目录总览

> 用途：跨机协作（本机写代码 / 云端训练）时快速定位"什么东西在哪里、是否进 git、是否需要带到云端"。
> 自动核对：`python3 v4/tools/verify_reference.py` 校验冻结引用件；
> 文件数可用 `git ls-files | wc -l` 与 `find v4 -type f | wc -l` 对照本文档。

---

## 一、顶层结构（`v4/` = git 仓库根 = 提交根）

```
v4/
├── README.md                 总入口：环境、平台速查、当前状态
├── PLAN.md                   总计划（892 行，唯一权威）
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
│   ├── hardware.py               **目标平台画像唯一事实源**（Ascend 910B / CANN 8.3rc2 /
│   │                             torch 2.8.0 + torch_npu 2.8.0 / arm64 / 16 GB RAM / 64 GB 显存 / 30 GB 云盘）
│   ├── portability.py            可选依赖探测与降级（numpy/pandas/pyarrow/torch/onnx）
│   ├── score.py                  官方评分器（drop 口径）
│   ├── data/
│   │   ├── parse.py              按表头名对齐的解析器（处理 3 口非规范 schema 井）
│   │   ├── labels.py             三状态判据、SW 尺度校验、PERM log 变换
│   │   ├── dataset.py            按井分片缓存（raw/labels npz）
│   │   ├── disk_guard.py         30 GB 云盘守卫（cleanup/save_and_exit/abort）
│   │   └── row_dataset.py        E1/P0 行级装配：F1 分片缓存 + **折内** RowScaler/目标尺度
│   ├── features/
│   │   ├── basic.py              F1 行级特征（32 列：13 曲线 + DEPTH + 缺失位 + 深度编码）
│   │   ├── physics.py            F_phys（16 派生 + 16 指示；Wyllie/密度/中子/IGR/电阻率交会）
│   │   ├── window.py             F_win（居中窗 {11,51,201} × 6 统计 = 234 列）
│   │   ├── well.py               F_well（井级 13×5 统计 + 5 标量 = 70 列，逐行广播）
│   │   └── groups.py             **特征组注册表**（FeatureSpec/缓存/溯源/审计）
│   ├── models/
│   │   ├── row_mlp.py            E1 行级 MLP（q_joint + 逐目标原子头 + 连续头）
│   │   ├── heads.py              **序列头**（与 RowMLP 同键，E3 复用同一套损失/解码/τ）
│   │   ├── unet1d.py             E3 1D U-Net（depthwise-separable + 空洞 + 跳连，全段 seq2seq）
│   │   ├── tcn.py                E3 **非因果** TCN（居中 pad、dilation→512、weight norm）
│   │   ├── mmoe.py               E8/P0 MMoE（E 专家 + 每任务门控，**与 SeqHead 同键**；
│   │   │                         瓶颈窄化保证参数量对等 + 门控熵/负载均衡 + 任务梯度余弦）
│   │   ├── well_head.py          E8/P1 井级 attention-pool → 逐目标井级偏置 Δ（容量 ≤ 主干 1/8、
│   │   │                         λ=0 精确恒等、无井身份键审计）
│   │   ├── patchtf.py            E4/P0 Patch Transformer（深度维 patch + 归一化 overlap-add；
│   │   │                         channel-independent / 相对位置开关；SDPA；states 逐行隐状态）
│   │   └── target_heads.py       E5 逐目标头（PorHead 四种参数化+表示能力收据、PermHead
│   │                             截断/桶头/分位、SwHead 单尺度 0–100 + 精确认原子）
│   ├── losses/score_aligned.py   三段式对齐损失（Charbonnier + softplus + 尺度归一化 L_aux + 逐目标原子 BCE + 边界聚焦）
│   ├── inference/（`predict.py` 已接线 PD1：manifest→权重→逐井解码→契约校验）
│   │   ├── contract.py           提交契约校验（10 井/95,948 行/字段/有限性/SW 尺度守卫）
│   │   ├── atomic_gate.py        **逐目标硬切换 τ_t** + joint_guard + 平台区中点选择 + 误判代价（numpy，无 torch）
│   │   └── predictor.py          manifest→权重→逐井预测→提交载荷（CPU 主路径 + 契约校验）
│   ├── validation/
│   │   ├── folds.py              按井折读取 + inner 折 + 加权 cluster bootstrap
│   │   └── gates.py              Gate 预注册校验与聚合判定（已实现）
│   ├── training/
│   │   ├── loop.py               训练循环：bf16 autocast（npu/cuda/cpu）、λ1 退火、梯度裁剪、
│   │   │                         best-epoch 权重写回、时间预算、每 epoch 磁盘守卫、时间日志
│   │   ├── metrics.py            预测→官方分数口径（连续/原子门/占位行命中率/原子头 P·R·F1）
│   │                         + AUC/AP（平均秩实现，E6 逐目标上报；单类标签返回 None）
│   │   ├── fold_runner.py        **两阶段单折协议**（E1/E2/E3 共用：inner-OOF 选 epoch/τ）
│   │   ├── seq_loop.py           E3 序列训练循环（chunk 批 + **分块重叠推理+加权拼接**）
│   │   ├── ema.py                E8/P2 EMA 影子（decay∈{0.99,0.999,0.9995}，逐 step；context_ema 精确还原）
│   │   ├── frozen.py             E5 冻结骨干取逐行隐状态（原生 forward_states / 前向钩子兜底 + 内存收据）
│   │   ├── state_train.py        E6/P0 两阶段件（切片权重永不为 0、阶段 2 冻结/降 lr 原子头、
│   │   │                         标签打乱负对照、输入列/原子标记泄漏审计）
│   │   ├── checkpoint.py         bf16 state_dict + manifest（含连续头标尺与 L_aux 尺度）+ 滚动淘汰 + resume 校验
│   │   └── tb_logger.py          TensorBoard + JSONL 日志（平台迭代曲线；无 tensorboard 时降级）
│   ├── versioning/registry.py    版本注册表读写（predict.py 的版本来源）+ 候选状态机
│   │                             （upsert/status/freeze，submitted 不可覆盖，原子写）
│   │                             + 管线版本写回 register_pipeline/set_version/set_latest
│   │                             （available=True 必须有权重文件；latest 只能指向可用版本）
│   └── ensemble/blend.py         E8 集成融合（纯 numpy）：inner-OOF 单纯形权重、同源性报告、
│                                 井级配对 cluster bootstrap 显著性、EMA/SWA（融合连续头，
│                                 原子硬切换必须在融合之后）
│
├── E0/ … E11/                12 个阶段，每层含 PLAN.md + P*/{PLAN.md,code/,docs/}
│   ├── E1/code/                 train_row.py（5 折 OOF + 两阶段 inner-OOF 选择 + Gate）
│   ├── E2/code/                 build_features.py（F2 缓存/溯源/内存画像）、
│   │                            ablate_groups.py（单组消融 + 增强消融 + 吞吐 + Gate）
│   ├── E3/code/                 train_seq.py（5 折序列 OOF + 边界体检 + Gate）、
│   │                            rf_ablation.py（depth×dilation_max 感受野消融）、
│   │                            compare_row_vs_seq.py（同折同头受控对照 + 配对 CI）
│   ├── E4/code/                 train_patchtf.py（PatchTF 5 折 OOF + 网格搜索/CI·位置·容量消融、
│   │                            还原与拼缝体检、vs E3 paired CI Gate；无基线不伪造 delta）
│   ├── E5/code/                 e5_common.py（冻结骨干/隐状态/两阶段单头训练/配对 CI/Gate 公共件）、
│                                head_por.py（POR 头 + 连续切片·占位行分项 + 0.1 下界反例消融）、
│                                head_perm.py（PERM 三档 z 输出 + 桶头对照 + 尾部单调性报告）、
│                                head_sw.py（SW 单尺度 0–100 契约 + 有效/占位分项 + 禁止尺度反例）、
│   │                            evaluate_targets.py（三目标 OOF 汇总：可复算/井序对齐/联合判据/
│   │                            swing 检查 → E5_per_target.json + E5_gate.json，纯 numpy）
│   └── E6/code/                 train_state.py（原子状态头两阶段：阶段 1 原子+联合头、阶段 2 冻结
│   │                            原子头训连续头 + 切片权重；τ 内折选；逐目标 AUC/Acc/P·R·F1；
│   │                            无插值检查；标签打乱负对照；E6_atomic_report.json + Gate；
│   │                            同时落盘 inner_oof.npz 供 P1 用）、
│                                search_tau.py（纯 numpy：91 点网格 + 最长平台中点、逐目标原子
│                                P·R·F1/Acc、误判代价分解、joint_guard 决策 → E6_tau_search.json
│                                + E6_P1_gate.json + candidates.json::PD1.atomic）
│
├── versions/                 事实源
│   ├── registry.json            可运行版本注册表
│   ├── prereg_templates/        33 份 Gate 预注册模板（**由 tools/sync_prereg_templates.py 从 P 级计划生成**，通过 gates.py 校验）
│   ├── candidates.json          **候选注册表（唯一事实源）**
│   ├── status.json              **阶段/P 执行状态台账**
│   ├── folds_sha256.json        折指纹
│   ├── reference/v1_well_folds.json  冻结折文件（随 git）
│   └── locks/{cloud.txt,submit.txt}  训练/推理依赖快照
│
├── reports/                  必须进 git 的**门禁证据快照**（权威副本在 $V4_REPORTS_DIR=/data/v4/reports）
│   ├── E0_gate.json              本地契约 Gate 判定（mandatory 13/13，passed=true）
│   ├── E0_cloud_gate.json        云端 Gate（blocked_pending_cloud_run）
│   ├── E0_data_card.json         数据卡（计数/状态/schema 异常；shard_cache.cache_root 为可复现形式）
│   ├── E0_score_check.json       常数基线锚点与总分恒等式
│   ├── E0_gate_prereg.json       E0 实际预注册
│   ├── E0_folds.json             折导出
│   ├── E0_contract_tests.json    契约自检（正/负样例）
│   ├── V4_PLAN_REVIEW{,_2,_3}.md 三轮独立审查报告（负资产，保留）
│   └── V4_PLAN_IMPROVEMENT_PROPOSAL.md  计划改进建议（已落地）
│   > `run_train.sh --mode e0` 会把 `$REPORTS_DIR/E0_*.json` **全部**回拷到此处（R3-H5）
│
├── tests/                    口径层测试（**不需要 torch**；torch 相关用例会自动 skip）
│   ├── run_all.py                一键运行（unittest discover）；缺 numpy 时退出码 2（环境问题≠失败）
│   ├── _synth_cache.py           合成井缓存工厂（E1/E3 端到端用例共用）
│   ├── test_parse.py             列布局/泄漏回归/畸形井/哨兵/特征/标签
│   ├── test_score.py             官方公式边界/两种口径/总分恒等式/锚点
│   ├── test_contract.py          井数/每井行数/深度对齐/SW 尺度守卫
│   ├── test_gates.py             Gate 类型/min_*·max_* 方向/指标字段映射/模板校验
│   ├── test_plan_stats.py        行数**分项**一致性 + 漂移必被检出（R3-C2 回归）
│   ├── test_platform_scripts.py  run_env 顺序 / --data-root / E0 回拷 / 折文件名 / 可选依赖降级
│   ├── test_atomic_gate.py       逐目标硬切换 / joint_guard / τ 平台区 / 误判代价（numpy）
│   ├── test_disk_guard.py        30 GB 云盘守卫（cleanup/save_and_exit/abort）
│   ├── test_hardware.py          平台硬件单一事实源 + 显存/内存/云盘**防混淆**回归
│   ├── test_row_dataset.py       E1/P0 行级装配与**折内**标尺（无泄漏）
│   ├── test_training_core.py     损失/指标/两阶段折协议/checkpoint 契约
│   ├── test_features_e2.py       F2 特征组注册表/缓存键/溯源 368 行/无目标派生审计
│   ├── test_models_seq.py        序列头键一致性 / U-Net 跳连 / TCN 非因果探针
│   ├── test_models_e8.py         MMoE 同键+参数量对等+门控熵 / 井级分支容量与 λ=0 恒等 / EMA 影子
│   ├── test_patchtf_e4.py        patch 几何覆盖 / 三种权重精确还原 / 奇数长度同长输出 / CI·位置消融
│   ├── test_e4_pipeline.py       E4 端到端（合成井 → 报告/体检/Gate）+ 入口与 run_train.sh 接线
│   ├── test_target_heads_e5.py   POR 表示能力（0.1 下界禁用臂）/ PERM 截断与桶头 / SW 单尺度与精确原子
│   ├── test_frozen_e5.py         冻结读回等价 / 分块隐状态 / 钩子兜底 / 内存收据
│   ├── test_metrics_auc.py       平均秩 AUC / AP / 单类标签返 None / 逐目标+联合上报
│   ├── test_registry_writes.py   候选状态机：非法状态拒绝、submitted 不可覆盖、原子写
│   ├── test_e5_pipeline.py       E5/P0 端到端（随机骨干预检：报告/分项/消融/不判 PASS）+ 接线
│   ├── test_e5_perm_pipeline.py  E5/P1 端到端（PERM 契约有限且 >0、尾部报告、桶头/linear 臂、路由）
│   ├── test_e5_sw_pipeline.py    E5/P2 端到端（单尺度契约、有效/占位分项、禁止尺度臂暴露代价）
│   ├── test_e5_aggregate.py      E5 汇总（可复算/井序对齐/联合判据/swing 可加性/缺件显式降级）
│   ├── test_state_train_e6.py    E6 两阶段件（切片权重/冻结原子头真的不被更新/负对照/泄漏审计）
│   ├── test_e6_pipeline.py       E6/P0 端到端（逐目标指标齐全、τ 内折、无插值、负对照、可 resume）
│   ├── test_e6_tau.py            E6/P1 τ 搜索（网格/平台、逐目标指标、joint_guard、候选登记、缺件失败）
│   ├── test_predict_pd1.py       版本表写回纪律 + `predict.py --use-version PD1` 端到端（提交契约、
│   │                             确定性、SW 尺度、缺权重/未接线版本明确报错）
│   ├── test_e6_gate.py           E6/P2 Gate（证据齐全才过、缺证据/缺 OOF 一律不判过、绝对门槛）
│   ├── test_seq_pipeline.py      chunk 划分/拼接/权重/stitcher + E3 端到端（OOF + 边界体检，需 torch）
│   ├── test_e1_pipeline.py       E1 端到端（合成井 → Gate/prereg/候选，需 torch）
│   ├── test_ensemble_blend.py    E8 融合纪律：权重只在 inner-OOF / 同源不计增益 / CI 判据 / EMA·SWA
│   ├── test_losses.py            masked_mean NaN / PERM 截断 / L_aux 尺度不变 / 边界聚焦（需 torch）
│   └── test_heads.py             RowMLP 形状 / init_from_stats / POR 可到 0 / SW 标签尺度（需 torch）
│   > 当前：**582 项**；本地无 torch 解释器 172 项 skip、`./.venv-torch` 0 项 skip，两者全绿。
│
├── tools/
│   ├── pack_dataset.py           生成 ~30 MB 自包含数据包
│   ├── bootstrap_data.sh         部署数据到 /data（校验 sha256 + 行数 + 3 口畸形井）
│   ├── verify_reference.py       校验冻结折文件与数据指纹
│   ├── check_consistency.py      候选注册表与评分口径自洽校验
│   ├── check_status.py           状态台账与实物一致性校验
│   ├── check_data_leak.py        全量 90 井输入-标签泄漏回归
│   ├── plan_stats.py             计划行数统计（唯一事实源；`--check` 逐条断言阶段/P/平均/总量）
│   ├── sync_plan_stats.py        把实测行数同步进文档（**复用 plan_stats 的同一份正则**）
│   ├── sync_prereg_templates.py  由 P 级计划的 ```json 块重建 33 份预注册模板（防手工副本漂移）
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
/code/workspace/<仓库名>/      git clone（**临时**，任务结束即丢；目录名由平台决定）
/data/v4/data/{train,test}/    数据集（部署一次，永久）
/data/v4/{cache,runs,reports,logs,tb}/
                              缓存 / checkpoint / Gate 报告 / 日志 / TensorBoard（永久）
```

环境变量契约见 `configs/paths.yaml` 与 `docs/platform_setup.md` §二。

---

## 三之二、已实现 vs 计划中（避免"幽灵文件"）

| 类别 | 已实现 | 计划中（未实现，不在仓库） |
|---|---|---|
| `src/` | constants, portability, score, data/{parse,labels,dataset,disk_guard}, features/basic, losses/score_aligned, models/row_mlp（含 E6 的 `q_joint + q_por/q_perm/q_sw` 五个头）, inference/{contract,atomic_gate}, validation/{folds,gates}, versioning/registry | features/{physics,window,well}, models/{unet1d,tcn,patchtf,heads,mmoe}, inference/{predictor,decode}, ensemble/blend, losses 的物理项 |
| 根目录 | predict.py, run_train.sh, requirements.txt | train.py, configs/v4.yaml（当前为 `configs/paths.yaml`） |
| 产物 | reports/E0_*.json（10 份）、versions/registry.json、versions/candidates.json、versions/status.json、versions/folds_sha256.json、versions/prereg_templates/（33 份） | 各阶段的 E*.json |
| `run_train.sh` | `env` / `data` / `e0` / `smoke` / `stage E1` 均可用 | `--mode all` 会自动续跑 E1（代码已就绪） |

## 四、文件数量核对（截至 E0 完成时）

| 项 | 数量 |
|---|---:|
| git 跟踪文件 | 见 `git ls-files \| wc -l` |
| 计划文件（`PLAN.md`） | 1（总，892 行）+ 12（阶段，759 行）+ 33（P，5,224 行）= **46 份 / 6,875 行** |
| P 级计划平均篇幅 | **158 行**（合计 5,224；由 `tools/plan_stats.py` 统计） |
| 代码模块（`v4/src/**/*.py`） | 见 `find v4/src -name '*.py' \| wc -l` |
| E 层脚本（`v4/E*/code/*.py`） | 见 `find v4/E* -name '*.py' \| wc -l` |

> 本文档不手写文件数，避免与实物漂移；以命令行输出为准。
