# E0/P1 数据卡、哨兵与标签三状态

> 所属阶段：[E0](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。

## 1. 目标

自写解析器读取 90 口井，冻结缺失哨兵规则、三状态判据（缺测/联合常量/有效）、SW 双尺度常量，产出数据卡与折指纹。

## 2. 为什么需要这一步

口径是唯一事实源；SW 的 99.9 与 [0,1] 混列是最大静默失分点。

## 3. 代码

- `src/data/parse.py`
- `src/data/labels.py`
- `E0/code/build_data_card.py`

## 4. 产物

- `reports/E0_data_card.json`
- `artifacts/E0/folds.json`
- `versions/folds_sha256.json`

## 5. 完成判据

- 80/10 井、730,268/95,948 行、三状态计数与 `资料库/12` §3.1 一致；折与 `v1_well_folds.json` 逐字节一致

## 6. 禁止事项

- 解析时保留单位行
- 把 -99999 之外的小负值当有效值

## 7. 执行提示

- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。
- 结果写入本 P 的 `docs/` 与 `v4/reports/E0_P1_*.json`。
- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/E0_P1_gate_prereg.json`。
- 任何结论必须附「复算命令」与「产物 sha256」。
