# E0 数据、评测与提交契约

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E0/src/、v4/E0/code/`　|　产物 `v4/experiments/E0/、v4/models/E0/`

> 阶段性质：**契约冻结阶段。不训练任何模型。**
>
> **执行状态（2026-09-19）：本地契约层已完成；阶段整体 `in_progress`（云端 Gate 待 P0）。**
> **E0 本地契约 Gate：13/13 mandatory PASS（含 `contract_ok` 与 cache 两项）**
> （`reports/E0_local_contract_gate.json`；项数由 `tools/plan_stats.py` 从报告实测，禁止手写）；
> 与 `reports/E0_gate_prereg.json` 可用同一个 `aggregate_gate` 复算通过（结论见 `reports/E0_gate_result.json`）；
> **E0 云端 Gate：`blocked_pending_cloud_run`**（`reports/E0_cloud_gate.json`，需 A100 任务跑 `run_train.sh --mode env`）。
> 已复算并冻结的事实：
> - 训练 **80 井 / 730,268 行**；测试 **10 井 / 95,948 行**（= 契约值）
> - 标签状态：缺测 **6,700** / 联合常量占位 **487,225（66.719%）** / 有效 **236,343**
> - 常数基线 **70.490735**（`drop` 口径）命中公开锚点 70.4907 ±1e-4；`mask` 口径为 69.843218
> - 折指纹 `f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e`（80 井 / 5 折，每折 16 井）
> - 提交契约自检：**全部负样例均被正确拒绝**，正样例通过；`CONST` 端到端 10 井 / 95,948 行 / 1.4 s（单核 CPU）
>   项数以 `reports/E0_contract_tests.json::n_negative / n_rejected` 为准，本文件不手写数字
> - 分片缓存 **32.39 MB**，90 井输入列校验通过（train+test 两个 split 都查）；`cache_root` 记为**可复现形式**（本地 repo 相对、云端 `$V4_CACHE_ROOT`），不再是 `/tmp` 临时路径
> - **SW 为单一标签尺度**（百分数，实测有效 8.305–99.9，小于 1 的行数为 **0**）——「双尺度 / 有效值归一化到小数区间」的假设已被实测证伪（R3-C1 统一口径）；
>   提交契约对 SW 采用**四重守卫**（SW<1 计数 + SW<8.305 占比 + 非原子行 p05 + 全体中位数），不再只看中位数（R4-B3）
> - **CUDA 语义**：hard 只要求 `torch.version.cuda` 存在且 **major == 12**；声明值
>   **12.8**（torch 2.7.1+cu128 的 runtime）与 `nvidia-smi` 的驱动能力 12.8 分别以
>   warn / advisory 记录；**不得**把 wheel 小版本写成 hard 断言（R4-B1/R5-B1 教训）
> - **依赖**：required 由用户 pip 安装（`numpy/pandas/scipy/scikit-learn/einops`），
>   **版本不钉死**、只查存在性；`pyarrow/onnx/onnxruntime/tensorboard` 为可降级可选（R5-M1）
> - **两个硬发现**：①3 口训练井 schema 非规范（20/21/16 列，27,080 行，3.71%），必须按表头名解析；
>   ②评分分母口径为「逐目标排除缺测」（`drop`），选错会系统性低 0.65 分
>
> 详见 [`E0/docs/data_card.md`](docs/data_card.md) 与 [`versions/status.json`](../versions/status.json)。
> 唯一**未完成**的 P 是 **P0（云端环境与磁盘实测）**，需 A100 训练任务实机运行。

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

把数据解析、缺失哨兵、标签三状态、官方评分器、按井 5 折、提交格式、版本路由与云端环境全部冻结为可复算的契约，使后续一切结论都有唯一口径。

## 2. 为什么这么设计

1. 评分口径与本地评分器实现若与官方有偏差，所有后续迭代都是在优化错误目标；v1 冻结的全常量基线 OOF = **70.4907** 是校验锚点。
2. 标签里 66.719% 是联合常量占位；**SW 是单一标签尺度（百分数，实测有效 8.305–99.9，`SW<1` 为 0 行）**——E0-R2 已证伪“99.9 与 [0,1] 双尺度”假设，`SW_SMALL_BRANCH` 作为遗留对照路径**永久关闭**；把 SW 误当小数尺度是最大的静默失分点，必须在写模型之前用单元测试锁死。
3. 云端是预装镜像（CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64，无 conda）+ 仅 64 GiB 磁盘；环境与磁盘必须在训练开始前实测确认。

## 3. 输入

- `../data/train`（80 井）、`../data/test`（10 井）
- `rules.md` §5–§8
- `../v1/src/well_folds.json`（按井 5 折）
- `资料库/12` §1–§5（口径与工程化落地）

## 4. 产物

- `reports/E0_data_card.json`
- `reports/E0_env.json`、`reports/E0_disk_budget.json`
- `reports/E0_score_check.json`（常数基线 70.4907）
- `reports/E0_gate.json`、`reports/E0_gate_prereg.json`
- `artifacts/E0/folds.json`、`versions/folds_sha256.json`

## 5. 代码归属

- `src/data/parse.py`（自写解析，只依赖标准库+numpy）
- `src/data/labels.py`（三状态判据 + SW 尺度常量）
- `src/score.py`（官方评分复算）
- `src/inference/contract.py`（提交契约校验）
- `E0/code/check_env.py`、`E0/code/setup_deps.sh`
- `E0/code/run_all.py`（一键复算）

## 6. P 级子计划

- [P0 环境与磁盘实测（云端第一次运行）](P0/PLAN.md)
- [P1 数据卡、哨兵规则与标签三状态](P1/PLAN.md)
- [P2 官方评分器复算与分母口径冻结](P2/PLAN.md)
- [P3 提交契约、版本路由与干净目录冒烟](P3/PLAN.md)

## 7. 完成判据（Gate）

- `E0/code/run_all.py` 一条命令复算出：80/10 井、730,268/95,948 行、三状态计数、折指纹、常数基线 70.4907（±1e-4）。
- `check_env.py` 的 hard 检查全过（torch 2.8.0 + torch_npu 2.8.0 / Ascend 910B / CANN 8.3rc2 / bf16 / 磁盘可用 ≥ 8 GB），结果写入 `E0_env.json`。
- 提交契约单测通过：构造的假 `result.json` 能被 `validate_payload` 正确接受/拒绝。
- **全程不需要 torch**（数据与提交侧只依赖标准库+numpy/pandas）。

## 8. 禁止事项

- 训练任何模型
- 修改折分配（必须与 `v1_well_folds.json` 逐字节一致）
- 为省事裁剪 SW 或删除占位行

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 NPU/GPU）负责代码与契约，云端（1×Ascend 910B 64GB，CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64，**64 GiB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
