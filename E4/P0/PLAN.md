# E4/P0 Patch Transformer 主干（通道独立）

> 所属阶段：[E4](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　索引：[资料引用索引](../../资料引用索引.md)

> **性质**：第二主干候选　|　**依赖**：E3/P2（序列路线确认有效）

> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。

---

## 1. 目标

实现 PatchTST 式通道独立 Patch Transformer（patch=32/stride=16、d=256、6 层、8 头、相对位置编码），与 E3 的 CNN 主干在**同数据同折**下可比。

## 2. 为什么需要这一步

1. `资料库/08` §0.3 第 3 层与 §7.5：把深度序列切成 patch 后做通道独立建模，是长序列的低成本高效方案（复杂度从 O(n²) 降到 O((n/P)²)）；
2. CNN 擅长局部形态，注意力擅长长程依赖，两者互补——但必须先证明 Transformer 单体能打平/超过 CNN，才谈融合；
3. `资料库/08` §1.2 指出整井 n≈10⁴ 时原始自注意力不可接受，patch 化是必要前提。

## 3. 输入契约

- E3/P0 的 `SeqDataset`
- `资料库/08` §7.1/§7.3/§7.5/§7.6

## 4. 输出契约

- `src/models/patchtf.py`
- `models/E4/patchtf_fold{k}.pt`
- `$V4_REPORTS_DIR/E4_patchtf.json`（与 E3 的对照结果）

## 5. 执行步骤

1. 实现 patch 切分与线性投影（P=32, stride=16, d_model=256）
2. 实现通道独立：每条曲线单独作为 token 序列（共享权重），最后沿通道做聚合
3. 注意力用 PyTorch 2.4 原生 `F.scaled_dot_product_attention`（自动选择 Flash/Memory-Efficient/Math 后端）
4. 相对位置编码 + Pre-LN + 残差 + FFN(GELU)，dropout 0.1
5. 输出上采样回逐行长度（patched 输出按 stride overlap-add 还原）
6. fold0+1 筛查，胜者跑全 5 折

## 6. 参数与配置

| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |
|---|---|---|---|
| `patch_len` | 32 | 16/32/64 | 与 stride 联动 |
| `stride` | 16 | 8/16/32 | overlap = patch−stride |
| `d_model` | 256 | 128/256/512 | 显存充足 |
| `n_layers` | 6 | 4/6/8 | 同上 |
| `n_heads` | 8 | 4/8 | d_model 必须整除 |
| 通道独立 | 是 | 是/否 | NO 则退化为多头联合建模，作对照 |
| `dropout` | 0.1 | 0.0/0.1/0.2 | 80 井易过拟合 |

## 7. 完成判据

- patch 还原后输出长度与输入严格一致（overlap-add 权重归一）
- 与 E3 在同折同数据下可比（同 chunk、同特征、同头）
- 注意力只使用 2.4 已有签名；无编译扩展依赖
- fold0+1 结果与资源记录完整

## 8. 禁止事项

- 把曲线轴当图像轴做 2D 卷积（曲线轴相邻无物理含义，`资料库/08` §1.3）
- 依赖 flash-attn/xformers
- 用 2.5+ 的 `torch.nn.attention` API

## 9. 风险与对策

| 风险信号 | 早期表现 | 对策 |
|---|---|---|
| patch 还原错位 | 接缝处预测跳变、行数不符 | overlap-add 单测：常数输入应还原为常数 |
| 通道独立后参数量爆炸 | 显存/时间超预算 | 共享通道权重；必要时减层 |
| 注意力数值不稳 | NaN | Pre-LN + 梯度裁剪 + bf16 关键层 fp32 |

## 10. 停止规则

- fold0+1 耗时超过 E3 单折的 3 倍且分数无优势 → 判 NO-GO，保留 CNN 主干

## 11. 代码归属

- `src/models/patchtf.py`
- `E4/code/train_patchtf.py`

## 12. 复算与证据

- `reports/E4_patchtf.json`

```bash
# 云端（平台训练任务）
bash /code/workspace/v4/run_train.sh --mode stage --stage E4
# 本机（口径层，无 torch）
python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py
```

## 13. Gate 预注册要点

预注册文件：`v4/reports/E4_P0_gate_prereg.json`（实验**前**写入，之后不得改阈值，只能新建修订号）。完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。

```json
{
  "gate_id": "E4_P0_gate",
  "stage": "E4",
  "p_stage": "P0",
  "candidate_budget": 4,
  "alpha": 0.05,
  "bootstrap_iters": 1000,
  "bootstrap_unit": "well_row_weighted_cluster",
  "decisions_locked": [],
  "primary_metric": "target_acc",
  "thresholds": {
    "min_delta": 0.0
  },
  "multiplicity": "holm"
}
```

> 所有 Gate 的 `mandatory_checks` 必须包含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`（本 P 的 `prereg_extra` 已按 P 的性质补齐）。
