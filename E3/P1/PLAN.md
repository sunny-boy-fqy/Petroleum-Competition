# E3/P1 1D U-Net 与 TCN 主干实现

> 所属阶段：[E3](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：主线模型实现：本计划的核心赌注　|　**依赖**：E3/P0
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现两种深度序列主干（1D U-Net 与 **非因果** TCN），输出与输入逐行同长（**全段 seq2seq，不用滑窗中心点**），接多任务头（连续头）+ 逐目标原子头 + 辅助 `q_joint` 头，在 bf16 下训练单折并记录参数量/显存/耗时。

## 2. 为什么需要这一步

1. `资料库/08` §0.3 把 1D U-Net / TCN 列为"最可能冲高分的结构"，而前代因 CPU 限制从未真正训练过（v2 E7 是 NO-GO 但属"无 CPU 可行方案"）；
2. 0.1 m 采样下 10 m 储层段 = 100 点，**没有数百点感受野模型只能逐点外推**；
3. U-Net 的 skip 保留高频细节（薄层），TCN 的空洞卷积给长程依赖，两者归纳偏置互补，必须头对头比较才知道哪个更适合本数据；
4. 改进 proposal §6：本任务是**离线整井**推理，TCN 必须**非因果**；U-Net 用 depthwise 可分离 + 空洞卷积时会产生 **padding 边界伪影**，且 chunk 拼接会产生接缝跳变，必须专门检查。

## 3. 输入契约

- E3/P0 的 `SeqDataset`
- E1/P0 的 F1 特征（作为输入通道）
- `资料库/08` §4（1D-CNN/空洞/深度可分离）、§6（TCN）

## 4. 输出契约

- `src/models/unet1d.py`、`src/models/tcn.py`、`src/models/heads.py`
- `models/E3/{unet,tcn}_fold{k}.pt`
- `$V4_REPORTS_DIR/E3_param_budget.json`（参数量/显存/单折耗时）
- `$V4_REPORTS_DIR/E3_boundary_report.json`（padding 边界伪影 + chunk 拼缝检查）

## 5. 执行步骤

1. 实现 `UNet1D`：depth=5 级下采样（stride 2）+ 同层数上采样 + skip 拼接；每级 2×[**depthwise-separable Conv1d(k=5, groups=C)** → BN → GELU]，下采样块中插入**空洞卷积**扩大感受野；`base_ch` 64→256
2. 实现 `TCN`：残差块 + **非因果**空洞卷积（dilation=2^i, i=0..8, k=3）+ weight norm + 残差；`dilation_max=512` ≈ 50 m 感受野，**先验证有效性**再考虑继续加
3. 实现 `SeqHead`：把主干输出 (B,L,d) 逐行送 POR/PERM/SW 连续头 + `q_atom[3]` + `q_joint`（形状 (B,L)）
4. 确认前向输出长度与输入严格一致（`out.shape[1]==x.shape[1]`）——**全段 seq2seq，禁止滑窗中心点**
5. 实现 chunk 重叠推理与**加权拼接**（三角/汉宁窗，按权重归一），消除接缝跳变
6. 跑 padding 边界伪影检查：逐目标比较井首/井尾 10 m 与井中段的 Acc，记录差值
7. 单折训练（bf16 + 梯度裁剪 + AdamW），记录参数量与峰值显存/内存
8. 跑 `--smoke`（8 井 2 epoch）确认无 NaN、契约通过

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `base_ch` | 64 | 32/64/128 | 显存充足；内存不受影响 |
| U-Net 深度 | 5 | 3/4/5 | 对应感受野 ≈ 数十–上百 m |
| TCN 因果性 | **非因果**（离线任务） | 冻结 | 因果化会白丢未来上下文 |
| TCN 块数/最大 dilation | 9 块 / 512 | 6/9；64/512 | 感受野消融的关键变量 |
| 卷积核 | 5（U-Net）/ 3（TCN） | 3/5/7 | 同上 |
| chunk 拼接权重 | 三角窗（按距离） | 三角/汉宁/等权 | inner-OOF 选择，防接缝跳变 |
| 归一化 | BN（默认） | BN/LN/GN | E3/P2 消融 |
| `dropout` | 0.1 | 0.0/0.1/0.2 | 序列模型更易过拟合 80 井 |
| `lr` | 1e-3 | 3e-4/1e-3/3e-3 | AdamW + 余弦 |
| bf16 | 开启 | bf16/fp32 | A100 支持；fp16 易 NaN |

## 7. 完成判据

- 两种主干都能前向且输出长度与输入一致（全段输出，非中心点）
- 参数量与峰值资源记录完整，单折耗时在软预算内
- `--smoke` 无 NaN、契约通过、checkpoint 可 `--resume`
- TCN 全程**非因果**（单测：将输入尾部置零不应改变头部输出）
- chunk 重叠推理 + 加权拼接实现完成，逐点与整段推理差 < 容差（无缝跳变）
- `E3_boundary_report.json` 给出井首/井尾 10 m vs 井中段的逐目标 Acc 差；若显著偏低，必须给出修正（padding 方式/深度可分离归一化）
- 不使用任何 2.5+ 的 PyTorch API；注意力（若有）走 `F.scaled_dot_product_attention`

## 8. 禁止事项

- 使用 ImageNet/自然图像预训练权重（分布无关，只会引入无关先验）
- 引入 flash-attn / xformers 等需编译的 CUDA 扩展
- 在 16 GiB 内存下把整井喂入模型
- 把 TCN 做成因果卷积（离线任务无因果约束）
- 用滑窗中心点预测代替全段 seq2seq 输出
- 在 chunk 边界不做重叠加权（会留下接缝跳变）

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| 感受野不足 | 序列模型与行级模型分数接近 | E3/P2 的感受野消融会暴露；先增大 depth/dilation 再谈扩容 |
| padding 边界伪影 | 井首/尾 10 m Acc 明显低于中段 | 改 padding 方式（reflect/replicate）、加深可分离块、或在拼缝处加权 |
| 过拟合 80 井 | inner 高 outer 低、折间方差大 | dropout/stochastic depth/weight decay + E2 增强 |
| bf16 数值不稳 | loss 出现 NaN | 梯度裁剪 1.0 + 关键归一化层用 fp32（`autocast` 白名单） |
| 显存充足但内存爆 | 阶段被杀 | chunk + `num_workers=4` + 不缓存分片 |

## 10. 停止规则

- 单折超过软预算 3 倍 → 降 chunk 长度或减 base_ch
- 连续 2 次 NaN → 回退 checkpoint 并减半 lr / 改 fp32 关键层

## 11. 代码归属

- `src/models/unet1d.py`
- `src/models/tcn.py`
- `src/models/heads.py`
- `E3/code/train_seq.py`

## 12. 复算与证据

- `reports/E3_param_budget.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E3
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 12.5 选择协议（H1：inner-OOF only）

**所有超参/阈值/早停/结构选择只允许用 inner 折**（`$V4_REPORTS_DIR/E0_folds.json::inner`）。

| 用途 | 允许的数据 | 禁止 |
|---|---|---|
| 超参/阈值/λ/τ/集成权重选择 | 该 outer 折的 inner-OOF | outer 验证折标签 |
| 早停 | inner-OOF 的真实 `score.py` 分数 | outer 折分数、loss 值 |
| 结构/特征筛查（省机时） | 可先用 fold0 做**资源预检** | 预检结论不得进入 Gate 数值 |

> 若某步骤确实只能看 outer 折（例如最终 OOF 汇总），该步骤**不得**反过来影响任何选择；
预检性质的 fold0 结果必须在报告中标 `exploratory=true`、`selection_score_only=true`。

## 13. Gate 预注册要点

预注册文件：`v4/reports/E3_P1_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E3_P1_gate",
  "stage": "E3",
  "p_stage": "P1",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "boolean",
  "primary_metric": "seq_train_ok",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0
  },
  "alpha": 0.05,
  "multiplicity": "none",
  "candidate_budget": 1,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "pilot_std": null,
  "mde_units": 80,
  "min_detectable_effect": null,
  "planned_task_training_h": 1.0,
  "mandatory_checks": [
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "checkpoint_resumable",
    "no_label_leak",
    "seq_train_ok"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
