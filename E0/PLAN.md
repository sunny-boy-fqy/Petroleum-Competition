# E0 数据、评测与提交契约

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 ``v4/E0/src/`、`v4/E0/code/``　|　产物 ``v4/experiments/E0/`、`v4/models/E0/``

> 阶段性质：**契约冻结阶段。不训练任何模型。**

## 1. 目标

把数据解析、缺失哨兵、标签三状态、官方评分器、按井 5 折、提交格式、版本路由与云端环境全部冻结为可复算的契约，使后续一切结论都有唯一口径。

## 2. 为什么这么设计

1. 评分口径与本地评分器实现若与官方有偏差，所有后续迭代都是在优化错误目标；v1 冻结的全常量基线 OOF = **70.4907** 是校验锚点。
2. 标签里 66.719% 是联合常量占位，且 **SW 同一列混用 99.9（百分数）与 [0,1]（小数）两种尺度**——这是最容易造成 ~10 分静默损失的错误，必须在写模型之前用单元测试锁死。
3. 云端是预装镜像（CUDA 12.6 / torch 2.4.0 / py3.11，无 conda）+ 仅 30 GB 磁盘；环境与磁盘必须在训练开始前实测确认。

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

- [P0 环境与磁盘实测](P0/PLAN.md)
- [P1 数据卡、哨兵与标签三状态](P1/PLAN.md)
- [P2 评分器与常数基线复算](P2/PLAN.md)
- [P3 提交契约与版本路由](P3/PLAN.md)

## 7. 完成判据（Gate）

- `E0/code/run_all.py` 一条命令复算出：80/10 井、730,268/95,948 行、三状态计数、折指纹、常数基线 70.4907（±1e-4）。
- `check_env.py` 的 hard 检查全过（torch 2.4.0 / A100 / bf16 / 磁盘可用 ≥ 8 GB），结果写入 `E0_env.json`。
- 提交契约单测通过：构造的假 `result.json` 能被 `validate_payload` 正确接受/拒绝。
- **全程不需要 torch**（数据与提交侧只依赖标准库+numpy/pandas）。

## 8. 禁止事项

- 训练任何模型
- 修改折分配（必须与 `v1_well_folds.json` 逐字节一致）
- 为省事裁剪 SW 或删除占位行

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.6 / torch 2.4.0 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
