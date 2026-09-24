# E2 → E3+ 自动特征版本选择

> **文档导航**：[v4 文档中心](README.md) · [v4 README](../README.md) · [总计划](../PLAN.md) · [代码审查状态](CODE_REVIEW_STATUS.md)
> **文档类型**：自动选择规则与接线说明。解决“E2 唯一 ADOPT 特征组被后续阶段忽略”的接线缺口。

## 0. 目的

E2 会做特征组消融，并给出唯一结论：

```text
F1 / F1+phys / F1+win / F1+well / F2
```

在 `E2_ablation.json` 中，每个候选都有：

```json
{
  "spec_key": "FX_win_w11-51-201_smsmmtc",
  "groups": ["F1", "win"],
  "oof_total": 79.4403,
  "delta_vs_f1": 1.0436,
  "paired_ci": [0.8075, 1.2952],
  "decision": "adopted"
}
```

从 E3 开始，流水线自动选择 **paired CI 下界 > 0 且 OOF 最高**的 adopted 特征组；若无 adopted，则回退 `F1`。

## 1. 自动选择程序

```text
tools/select_feature_spec.py
```

读取：

```text
$V4_REPORTS_DIR/E2_ablation.json
```

写出：

```text
$V4_REPORTS_DIR/E2_best_spec.json
```

手动运行：

```bash
python3 tools/select_feature_spec.py --reports-dir "$V4_REPORTS_DIR"
python3 tools/select_feature_spec.py --reports-dir "$V4_REPORTS_DIR" --print
python3 tools/select_feature_spec.py --reports-dir "$V4_REPORTS_DIR" --print-key
```

## 2. run_train 中的自动接线

`run_train.sh --mode all` 中：

- E3–E8 启动前自动执行 `resolve_feature_spec`；
- E3/E4/E5/E6/E7/E8/E10 自动接收：

```bash
--spec "$FEATURE_SPEC"
```

- E3 的 `row-vs-seq` 对照：
  - 若选中的是 `F1`：使用 `runs/E1/oof.npz`；
  - 若选中的是 `F1+win` 等：使用 E2 已产出的同特征行级 OOF：
    - 优先：`$RUN_ROOT/E2/oof_<selected_key>.npz`（run_train 的 E2 work-dir）
    - 回退：`$V4_REPORTS_DIR/E2_work/oof_<selected_key>.npz`（手工直跑 E2 的默认 work-dir）

因此 E2 的 ADOPT 结论不会再被丢掉。

显式覆盖自动选择：

```bash
export V4_FEATURE_SPEC=F1
export V4_FEATURE_SPEC=F1+win
export V4_FEATURE_SPEC_KEY=FX_win_w11-51-201_smsmmtc
```

## 3. fold cache 防混 spec

切换 `F1` 与 `F1+win` 后，输入维度不同。E3 现在在 fold cache 中写入：

```json
{
  "_resume_stamp": {
    "spec": {"groups": ["F1", "win"], "...": "..."},
    "arch": "unet",
    "arch_kwargs": {"base_ch": 64, "depth": 5, "k": 5}
  },
  "result": "..."
}
```

`--resume` 时如果 spec / arch / arch_kwargs 有任何不一致，该 fold cache 会被判定不可用并自动重跑，绝不会把 F1 的结果当成 F1+win 的结果。

同时对 seq_loop 的 checkpoint manifest 增加 `feature_spec` 一致性检查。

## 4. 启动命令

自动选择最优特征版本：

```bash
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode all --through 14
```

不要加 `--fresh`。

只验证 E3-main：

```bash
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode all --through 6
```

## 5. 确认

```bash
python3 tools/select_feature_spec.py \
  --reports-dir "$V4_REPORTS_DIR" --print
```

若输出 `F1+win`，说明 E3+ 会自动使用 E2 的 adopted 特征组。
