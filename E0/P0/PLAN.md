# E0/P0 环境与磁盘实测

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

在云端跑 `check_env.py` / `disk_guard.py` / `setup_deps.sh`，把 torch 2.4.0+cu124、A100 sm_80、bf16、Python 3.11、可用磁盘与 pip freeze 全部落盘为事实。

## 2. 为什么需要这一步

未实测的环境假设会在 E3 训练数小时后才暴露（OOM / 版本不兼容 / 磁盘写满），代价极高。

## 3. 代码

- `E0/code/setup_deps.sh`
- `E0/code/check_env.py`
- `src/data/disk_guard.py`

## 4. 产物

- `reports/E0_env.json`
- `reports/E0_disk_budget.json`
- `versions/locks/cloud_frozen.txt`

## 5. 完成判据

- hard 检查全过；可用磁盘 ≥ 8 GB；若 < 12 GB 则写入 `contingency_applied` 并降低后续预算

## 6. 禁止事项

- 装任何会触碰 torch/nvidia-* 的包
- 在磁盘未知的情况下开始 E3

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E0_P0_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E0_P0_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
