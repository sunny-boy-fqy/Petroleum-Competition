# E10/P1 打包、干净目录复现与 B0 fallback

> 所属阶段：[E10](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

组装 `submission_code_v4.zip`、在只含 zip+data 的干净目录跑官方命令、比对结果；同时预构建 B0 fallback 包并验证。

## 2. 为什么需要这一步

`rules.md` §8.4：复现失败即取消资格；纯 DL 管线必须准备保底包。

## 3. 代码

- `E10/code/build_submission.py`
- `E10/code/verify_inference.py`
- `E10/code/build_b0_fallback.py`

## 4. 产物

- `submission/*.zip`、`reports/E10_reproduce_report.json`、`reports/E10_B0_fallback_manifest.json`

## 5. 完成判据

- 逐点差 ≤ 1e-6；两次运行 sha256 一致；CPU < 30 min；fallback 包验证通过

## 6. 禁止事项

- 干净目录里依赖 v1/v2/v3 文件
- 跳过 fallback 构建

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E10_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E10_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
