# E4/P0 Patch Transformer 主干（通道独立）

> 所属阶段：[E4](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：第二主干候选　|　**依赖**：E3/P2（序列路线确认有效）
>
> **状态**：⏸ 待执行　

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现 PatchTST 式**通道独立 + 相对位置编码**的 Patch Transformer，patch size/stride/overlap 只在 inner-OOF 上搜索，与 E3 的 CNN 主干在**同数据同折**下可比；80 井小数据下**优先小 `d` / 小 `layers`**。

## 2. 为什么需要这一步

1. `资料库/08` §0.3 第 3 层与 §7.5：把深度序列切成 patch 后做通道独立建模，是长序列的低成本高效方案（复杂度从 O(n²) 降到 O((n/P)²)）；
2. CNN 擅长局部形态，注意力擅长长程依赖，两者互补——但必须先证明 Transformer 单体能打平/超过 CNN，才谈融合；
3. `资料库/08` §1.2 指出整井 n≈10⁴ 时原始自注意力不可接受，patch 化是必要前提；
4. 改进 proposal §6：80 井极易过拟合，PatchTF 必须优先小 `d`/小 `layers`；patch/stride/overlap 与重叠 chunk 推理都只能在 inner-OOF 上定。

## 3. 输入契约

- E3/P0 的 `SeqDataset`
- `资料库/08` §7.1/§7.3/§7.5/§7.6

## 4. 输出契约

- `src/models/patchtf.py`
- `models/E4/patchtf_fold{k}.pt`
- `$V4_REPORTS_DIR/E4_patchtf.json`（与 E3 的对照结果 + patch/stride/overlap 搜索表）

## 5. 执行步骤

1. 实现 patch 切分与线性投影（默认 P=32, stride=16, d_model=128, 4 层，小容量起步）
2. 实现通道独立：每条曲线单独作为 token 序列（共享权重），最后沿通道做聚合
3. **保留相对位置编码**并做消融（有/无）
4. 注意力用 PyTorch 2.4 原生 `F.scaled_dot_product_attention`（自动选择 Flash/Memory-Efficient/Math 后端）
5. 输出上采样回逐行长度（patched 输出按 stride overlap-add 还原），并实现**重叠 chunk 推理 + 加权拼接**
6. patch size/stride/overlap 只在 inner-OOF 上搜索（fold0 仅资源预检），胜者跑全 5 折

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `patch_len` | 32 | 16/32/64 | inner-OOF 搜索 |
| `stride` | 16 | 8/16/32 | overlap = patch−stride，inner-OOF 搜索 |
| `d_model` | 128 | 64/128/256 | 80 井优先小 d |
| `n_layers` | 4 | 2/4/6 | 80 井优先小 layers |
| `n_heads` | 8 | 4/8 | d_model 必须整除 |
| 通道独立 | 是 | 是/否 | NO 则退化为多头联合建模，作对照 |
| 相对位置编码 | 开 | 开/关 | 消融证明有贡献或记录 NO-GO |
| chunk 拼接权重 | 三角窗 | 三角/汉宁/等权 | inner-OOF 选，防接缝跳变 |
| `dropout` | 0.1 | 0.0/0.1/0.2 | 80 井易过拟合 |

## 7. 完成判据

- patch 还原后输出长度与输入严格一致（overlap-add 权重归一）
- 与 E3 在同折同数据下可比（同 chunk、同特征、同头）
- 通道独立与相对位置编码保留且各自有消融结论
- patch/stride/overlap 搜索表完整，全部只在 inner-OOF 上选
- 重叠 chunk 推理完成，整段推理与拼接推理逐点差在容差内（无接缝跳变）
- 注意力只使用 2.4 已有签名；无编译扩展依赖
- inner-OOF 筛查结果与资源记录完整（fold0 预检标 exploratory=true）

## 8. 禁止事项

- 把曲线轴当图像轴做 2D 卷积（曲线轴相邻无物理含义，`资料库/08` §1.3）
- 依赖 flash-attn/xformers
- 用 2.5+ 的 `torch.nn.attention` API
- 用 outer 折选 patch/stride/overlap
- 在小数据上直接开大 `d`/`layers` 而不先做容量消融

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| patch 还原错位 | 接缝处预测跳变、行数不符 | overlap-add 单测：常数输入应还原为常数 |
| 通道独立后参数量爆炸 | 显存/时间超预算 | 共享通道权重；必要时减层 |
| 注意力数值不稳 | NaN | Pre-LN + 梯度裁剪 + bf16 关键层 fp32 |

## 10. 停止规则

- 资源预检耗时超过 E3 单折的 3 倍且无优势 → 判 NO-GO，保留 CNN 主干

## 11. 代码归属

- `src/models/patchtf.py`
- `E4/code/train_patchtf.py`

## 12. 复算与证据

- `reports/E4_patchtf.json`

```bash
# 云端（平台训练任务）
bash "$(find /code/workspace -name run_train.sh | head -1)" --mode stage --stage E4
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

预注册文件：`v4/reports/E4_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E4_P0_gate",
  "stage": "E4",
  "p_stage": "P0",
  "created_at": "<ISO8601，写盘时填写>",
  "gate_type": "delta",
  "primary_metric": "target_acc",
  "primary_threshold_key": "min_delta",
  "baseline_version": "<已冻结候选或 CONST>",
  "baseline_artifact": "<基线 OOF 路径>",
  "baseline_manifest_sha256": "<sha256>",
  "thresholds": {
    "min_delta": 0.0,
    "min_effect_floor": 0.0
  },
  "alpha": 0.05,
  "multiplicity": "holm",
  "candidate_budget": 4,
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
    "no_label_leak"
  ],
  "decisions_locked": [],
  "notes": ""
}
```

> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：
> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/`baseline_manifest_sha256` 指向**已冻结**的基线；`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。
> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。
