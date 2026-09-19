# E10 全量重训、打包与提交

> 所属总计划：[v4/PLAN.md](../PLAN.md)　|　本层代码 `v4/E10/src/、v4/E10/code/`　|　产物 `v4/experiments/E10/、v4/models/E10/`

> 阶段性质：**交付阶段。**

> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。

## 1. 目标

产出最终候选权重与自包含提交包；在干净目录用官方命令一次性复现；执行提交；同时预构建 B0 fallback 保险包。

## 2. 为什么这么设计

1. `rules.md` §6.2/§8.3 要求"训练+推理"可独立复现、结果一致、环境可复现，复现失败直接取消资格。
2. 评测机不保证有 GPU，因此推理必须 CPU 可跑、确定性、< 30 min。
3. v4 是纯 DL 管线，没有 B0 patch 隔离兜底，因此必须**预先**构建 B0 fallback 保险包（从冻结 v1 E7 源码构建、官方 `--data_dir` 兼容）。

## 3. 输入

- E9 通过的候选
- `../v1/submission_e7`（fallback 源）

## 4. 产物

- `submission/submission_code_v4.zip`、`submission/result.zip`
- `reports/E10_reproduce_report.json`、`reports/E10_bench.json`
- `submission/submission_code_b0_fallback.zip`、`reports/E10_B0_fallback_manifest.json`

## 5. 代码归属

- `E10/code/final_train.py`、`E10/code/export_cpu.py`、`E10/code/verify_inference.py`
- `E10/code/build_submission.py`、`E10/code/build_b0_fallback.py`

## 6. P 级子计划

- [P0 全量重训与 CPU 导出](P0/PLAN.md)
- [P1 打包、干净目录复现与 B0 fallback](P1/PLAN.md)
- [P2 提交执行与归档登记](P2/PLAN.md)

## 7. 完成判据（Gate）

- 干净目录（只含提交 zip + data）官方命令一次运行成功，与提交结果逐点差 ≤ 1e-6。
- 两次独立运行 sha256 一致（确定性）。
- CPU 单次推理 < 30 min、峰值内存 < 8 GiB。
- B0 fallback 包在同一干净目录验证通过（byte-identical 或逐点差 ≤ 1e-9）。

## 8. 禁止事项

- 在提交包里依赖 v1/v2/v3 目录
- 依赖 GPU 或联网
- 现场训练

## 9. 通用约束（继承总计划）

- 训练/推理分离：本机（无 GPU）负责代码与契约，云端（1×A100 80GB，CUDA 12.6 / torch 2.4.0 / py3.11，**30 GB 磁盘**）负责训练。
- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。
- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。
- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。
- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。
