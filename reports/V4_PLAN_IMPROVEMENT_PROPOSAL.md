# v4 计划修改建议：逐目标原子层次、输出参数化、损失与训练策略

> 依据：用户在二审后提供的《计划改进报告》。
> 性质：**建议稿**，用于把改进项落到 `PLAN.md`、`E*/P*/PLAN.md`、`src/` 接口与 Gate 预注册中。
> 原则：先修正会直接改变优化目标的实现，再改训练策略；先让 E1/E5/E6 的接口正确，再谈 E3/E4 序列主干扩容。

---

## 0. 摘要：建议优先落地的三项

按“对最终 80 井 OOF 的边际收益/实现风险”排序，建议先做：

1. **逐目标原子头 + 期望分解码**
   - 用 `q_por/q_perm/q_sw` 替代“只有一个 joint 头决定三目标”的结构；
   - `q_joint` 保留为辅助损失/可选高置信硬门禁；
   - `τ_t` 在 inner-OOF 上用官方总分搜索，而不是 F1/准确率。
2. **POR / SW 输出参数化与训练折归一化修正**
   - POR 解除 `0.1+softplus(g)` 的下界锁死；
   - SW 用训练折内仿射归一化训练连续头，输出时反变换到标签尺度；
   - 原子分支仍精确输出目标原子值。
3. **PERM/SW 损失修正**
   - `masked_mean` 的 NaN 污染；
   - PERM 的 `max(ŷ/y, ε)` 截断对齐；
   - `L_aux` 改为按目标尺度归一化，避免 SW 99.9 主导辅助损失；
   - 增加“容差边界聚焦”权重作为 E7 消融项。

---

## 1. 数据事实：为什么 joint-only 原子头会误伤

对 80 口训练井全量复算（有效/非缺测行）：

| 原子事件 | 判据 | 行数 | 占全部行 |
|---|---|---:|---:|
| `q_por` | POR=0.1 | 487,382 | 66.740% |
| `q_perm` | PERM=0.01 | 494,598 | 67.728% |
| `q_sw` | SW=99.9 | 518,255 | 70.968% |
| `q_joint` | 三目标同时取原子值 | 487,225 | 66.719% |

组合分解：

| 组合 | 行数 | 对 joint-only 结构的含义 |
|---|---:|---|
| 三目标 joint | 487,225 | joint 头能覆盖 |
| 仅 POR 原子 | 117 | joint 头覆盖不到 |
| 仅 PERM 原子 | 1,906 | joint 头覆盖不到 |
| 仅 SW 原子 | 25,523 | joint 头覆盖不到 |
| POR+SW | 40 | 非 joint |
| PERM+SW | 5,467 | 非 joint |
| 单目标原子总数 | POR 157 / PERM 7,373 / SW 31,030 | joint-only 会漏掉或误伤 |

**结论**：
- joint 头仍然有用，因为 66.7% 的点确实是三目标同时原子；
- 但 31,030 个 SW 原子行、7,373 个 PERM 原子行、157 个 POR 原子行不是 joint；
- 这些行必须由**逐目标原子头**保护；用一个 joint 概率同时决定三目标会：
  - 对 joint 概率低的单目标原子行漏保护；
  - 对 joint 概率高的行强行覆盖三目标，误伤非原子目标；
  - 无法满足“逐目标原子 Acc ≥0.98/0.99”这种按目标验收的 Gate。

---

## 2. 修改 A：原子建模改为“逐目标原子 + 连续”层次结构

### A1. 头结构

在 `PLAN.md §5.3`、`E6/P0/PLAN.md`、`src/models/row_mlp.py`（或后续 `heads.py`）中修改为：

```
共享主干 h
├── q_joint  : Linear(d→1) + sigmoid       # 辅助，可选硬门禁
├── q_por    : Linear(d→1) + sigmoid       # P(POR=0.1)
├── q_perm   : Linear(d→1) + sigmoid       # P(PERM=0.01)
├── q_sw     : Linear(d→1) + sigmoid       # P(SW=99.9)
├── cont_por : POR 连续头
├── cont_perm: PERM log10 连续头
└── cont_sw  : SW 连续头
```

模型输出建议：

```python
{
    "por":       (B,),      # 连续 POR，标签尺度
    "perm_z":    (B,),      # log10(PERM)
    "sw":        (B,),      # 连续 SW，标签尺度
    "q_atom":    (B, 3),    # [q_por, q_perm, q_sw]
    "q_joint":   (B,),      # 辅助
}
```

标签侧新增：

```python
atom_values = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
y_atom[:, t] = (y[:, t] == atom_values[t]) & (~missing[:, t])
y_joint      = y_atom.all(axis=1) & (~missing.any(axis=1))
```

### A2. 损失

在 `src/losses/score_aligned.py` 新增/改造：

```python
L_atom = Σ_t λ_atom_t * masked_bce(q_atom[:, t], y_atom[:, t], mask=~target_missing[:, t])
L_joint = masked_bce(q_joint, y_joint, mask=~all_target_missing)
L = L_cont + λ_joint * L_joint + λ_atom * L_atom
```

建议默认值：

| 项 | 默认 | 搜索范围 |
|---|---:|---:|
| `λ_atom` | 0.5 | 0.2 / 0.5 / 1.0 |
| `λ_joint` | 0.2 | 0.1 / 0.2 / 0.5 |
| per-target `pos_weight` | 1.0 | 1.0 / 1.5 / 2.0 |
| 非 joint 原子行权重 `α` | 1.0 | 1.0 / 2.0 / 3.0 |

**非 joint 原子行加权**（比整行过采样更精确，避免不同目标互相干扰）：

```python
w_atom_t = 1.0 + α * y_atom[:, t] * (~y_joint)
L_atom_t = (BCE(q_t, y_atom_t) * w_atom_t * mask_t).sum() / (w_atom_t * mask_t).sum()
```

### A3. 推理：逐目标硬切换

在 `PLAN.md §6.4`、`E6/P1/PLAN.md`、`src/inference/atomic_gate.py` 中改为：

```python
for t, atom_t in enumerate([0.1, 0.01, 99.9]):
    out[:, t] = np.where(q_atom[:, t] > tau_t[t], atom_t, cont[:, t])
```

- 每个目标独立 `τ_t`；
- 只有高置信 joint 保护作为**可选**门禁：

```python
if joint_guard and q_joint > tau_joint_high:
    out[:] = atom_values
```

- `joint_guard` 默认关闭；是否启用必须由 inner-OOF 总分决定，并在 manifest 记录。
- 不允许在 atom/continuous 之间做线性插值。

### A4. `τ_t` 选择目标必须是官方总分

在 `E6/P1/PLAN.md` 中把“F1/准确率”改为：

```text
对每个 outer 折：
  用该折的 inner-OOF 预测 q_atom、cont
  对每个目标 t：
    for τ in grid:
      pred_t = atom if q_t > τ else cont_t
      score_t(τ) = 官方目标 Acc(pred_t, y_t) 在 inner-OOF 上
    τ_t* = argmax_t 100 * w_t * score_t(τ)
```

因为 Total 对三个目标是加权和，且硬切换是逐目标独立动作，`τ_t` 可逐目标搜索；只有启用 joint_guard 时才需要三目标联合网格。

必须报告：

- `tau_t` 曲线：阈值 vs inner-OOF Total；
- 平台区域：选择平坦区间中点，而不是尖峰；
- `atomic_precision/recall/F1` 与占位行 Acc；
- 误判代价分解：有效行判 atom、atom 行判 continuous 的分数变化。

---

## 3. 修改 B：POR / SW 输出参数化

### B1. POR：解除 0.1 下界

当前问题（已由数据证实）：

- 有效 POR 有 **576 行 <1**，其中 **186 行 <0.1**，不少真值为 `0.0`；
- `por = 0.1 + softplus(g)` 永远 ≥0.1，模型无法表示这些值。

建议改为以下两种之一，优先推荐 B1-2：

**B1-1：可到 0 的 softplus 偏移**

```python
por_cont = torch.clamp(softplus(g) - softplus(g0), min=0.0)
```

- `g0` 可取训练折内有效 POR 的某个低分位数对应的 logit；
- 输出层再 clip 到 `[0, por_max]`；
- 原子分支负责精确 0.1。

**B1-2：数据范围 sigmoid（推荐）**

```python
por_max = 1.2 * valid_por_max_fold          # 例如 33.177 -> ~39.8
por_cont = por_max * sigmoid(g)
```

- 天然有界、非负、可逼近 0；
- 初始化 `g` 使 `por_cont` 接近训练折有效 POR 中位数（约 11.34）；
- 原子分支负责精确 0.1；
- 连续头不再被 0.1 下界锁死。

无论选哪种，都必须：

1. 删除 `0.1 + softplus(g)`；
2. 初始化按有效 POR 分位数，而不是默认为 0.1；
3. 在 `E5/P0/PLAN.md` 增加 POR 参数化消融；
4. 在 `E1/P1` 和 `E5/P0` 的单测中覆盖 `POR=0`、`POR<0.1` 的切片。

### B2. SW：训练折内仿射归一化，而不是把标签当 [0,1]

真实有效 SW 是 **8.305–99.9 的百分数尺度**。线性头从 0 附近起步会导致早期相对误差损失极大。

建议：

1. 在每个训练折内统计：
   ```python
   sw_mu = median(sw_train_valid)
   sw_sigma = IQR(sw_train_valid) / 1.349   # 或 std
   ```
2. 训练连续头时使用：
   ```python
   sw_norm = (sw - sw_mu) / sw_sigma
   sw_norm_hat = head_sw(h)
   sw_cont = sw_mu + sw_sigma * sw_norm_hat
   ```
3. 输出头 bias 初始化为 0，因此初始连续输出接近 `sw_mu`，而不是 0；
4. 原子分支仍精确输出 `99.9`；
5. `sw_mu/sw_sigma` 写进 checkpoint manifest 与 scaler JSON，推理时反变换。

**硬规则**：

- 不允许把 `[0,1]` 当作 SW 标签尺度；
- 允许在 `[0,100]` 内做软裁剪；
- 原子分支必须精确输出 99.9；
- 灰色地带动作选择只在 inner-OOF 上做。

`E5/P2/PLAN.md` 的输出契约应改为：

```python
sw_cont = sw_mu + sw_sigma * sw_norm_hat
sw_final = atom 99.9 if q_sw > tau_sw else sw_cont
```

### B3. PERM 初始化与范围

`perm_z = 6 * tanh(g)` 可以保留，但建议：

- 初始化 `g` 使 `perm_z` 接近训练折有效 PERM 的 `log10` 中位数（约 -0.08）；
- 最终 clip 仍用 `[-6, 6]`，保证提交契约 `PERM > 0`；
- 报告 `z` 空间误差分布，但不再把 PERM 做线性域建模。

---

## 4. 修改 C：损失函数三项优化

### C1. `masked_mean` 防 NaN 污染

当前实现：

```python
return (x * m).sum() / m.sum().clamp_min(1.0)
```

改为：

```python
x = torch.where(m > 0, x, torch.zeros_like(x))
return (x * m).sum() / m.sum().clamp_min(1.0)
```

并新增单测：

```python
x = torch.tensor([float("nan"), 2.0])
m = torch.tensor([0.0, 1.0])
assert masked_mean(x, m).item() == 2.0
```

### C2. `align_score_log` 对齐官方 PERM 截断

官方：`1 - |log10(max(ŷ/y, ε))|`。

建议在 log 空间显式实现：

```python
d = zhat - z
d = torch.clamp_min(d, log10(eps))   # 对应 max(ratio, eps)
ell = smooth_abs(d, alpha)
return 1 - ell + softplus(ell - 1)
```

这样极端低估时的梯度/得分才与官方一致。`E5/P1/PLAN.md` 的完成判据要报告“PERM 低估尾部”与官方评分器的一致性。

### C3. `L_aux` 改为标准化空间并按目标加权

当前 `L_aux` 对 POR/SW 用绝对 Smooth L1，SW=99.9 会主导梯度。

建议：

```python
aux_por  = SmoothL1((por_cont - y_por) / s_por)
aux_perm = SmoothL1(zhat - z)
aux_sw   = SmoothL1((sw_cont - y_sw) / s_sw)
L_aux = 0.30*aux_por + 0.35*aux_perm + 0.35*aux_sw
```

其中 `s_por/s_sw` 只由训练折有效标签的鲁棒尺度决定（IQR 或 std），写入 manifest。

可选增强：

- **按切片加权**：
  - joint placeholder 行在连续头损失中降权（例如 0.1–0.3），因为硬切换后不会使用连续头；
  - non-joint atom 行保留一定权重，作为连续头的 fallback；
  - 有效连续行权重 1.0。
- **不要把权重设为 0**：如果原子头误判，连续头还应该有 fallback。

### C4. 容差边界聚焦（E7 消融项）

官方分不是对有效点等权，而是超过容差后分数归零。可给边界附近点更高权重：

对每个目标：

```python
r = |pred - y| / (delta_t * (|y| + eps))     # PERM 用 r = |zhat - z|
w_boundary = 1 + κ * exp(-((r - 1) ** 2) / (2 * σ ** 2))
```

建议搜索：

- `κ ∈ {0.5, 1.0, 2.0}`
- `σ ∈ {0.15, 0.25, 0.35}`

应用到 `L_align` 或 `L_aux`，但必须：

- 缺失目标 mask = 0；
- 原子行不参与连续边界聚焦，或单独按 atom/continuous 切片处理；
- 在 `E7/P0/PLAN.md` 做同结构对照，不能只报整体 Total。

---

## 5. 修改 D：两阶段训练、EMA/SWA 与快照集成

### D1. 两阶段训练

建议写入 `E6/P0/PLAN.md` 和 `src/training/loop.py`：

**第一阶段：原子头**

- 只训练主干 + `q_por/q_perm/q_sw` + `q_joint`；
- 损失：`L_atom + λ_joint * L_joint`；
- 可选极小的连续 fallback 损失（权重 0.05）；
- 目标准确率以 per-target atomic Acc/Precision/Recall 监控。

**第二阶段：连续头**

- 冻结原子头，或设置 `q_head_lr_mult = 0.05–0.1`；
- 只训练连续头（可同时慢更新主干）；
- 连续损失按目标/切片加权：
  - non-joint atom 行：权重 0.1–0.3（fallback）；
  - joint 行：权重 0.1 或更低；
  - 有效连续行：权重 1.0；
- `τ_t` 选择仍用 inner-OOF 官方总分。

### D2. EMA / SWA

- 每步 EMA：`decay ∈ {0.99, 0.999, 0.9995}`；
- 每 epoch 用 inner-OOF 真实 `score.py` 评估 EMA 权重；
- 保存 `ema.pt` 与普通 `last.pt/best.pt`；
- BN 模型优先 EMA，SWA 作为对照（`torch.optim.swa_utils` 更新 BN 统计需谨慎）；
- 结果写入 `E8_ensemble_report.json`，报告 EMA vs best 的逐折 delta 和 CI。

### D3. 快照集成

- 同一折保存 inner-OOF 得分最高的 top-k（k=2–3）checkpoint；
- 融合权重只在 inner-OOF 上选；
- 必须报告成员相关性/同源性；
- 如果增益 CI 含 0，标记为 NO-GO，不做同源平均包装。

### D4. 类别/切片重采样

- **不删除** placeholder 行；
- 对“非 joint 单目标原子”行使用 per-target sample weight 或 batch 内最低占比采样；
- 连续头训练中 hard joint 行降权，但不置 0；
- 相关权重进入参数表，并报告 effective sample size。

---

## 6. 修改 E：序列主干与融合

在 `E3/P1`、`E3/P2`、`E4/P0`、`E4/P1` 中增加以下检查项。

### E1. U-Net-1D

- 深度可分离卷积 + 空洞卷积；
- padding 边界伪影检查：比较边界 10 m 与井中段的逐目标 Acc；
- seq2seq 全段输出，不做滑窗中心点；
- chunk 重叠推理与加权拼接，避免边界跳变。

### E2. TCN

- 离线任务用**非因果**；
- dilation 上限 512 约等于 50 m 感受野，先验证有效性，不盲目继续加；
- 记录有效感受野消融：`depth∈{3,5}`、`dilation_max∈{64,512}`。

### E3. Patch Transformer

- 通道独立 + 相对位置编码保留；
- patch size/stride/overlap 做 inner-OOF 搜索；
- chunk 边界重叠推理；
- 注意 80 井小数据下的过拟合，优先小 `d`、小 `layers`。

### E4. 多尺度融合

- 优先门控融合或 FiLM，而不是直接 concat；
- 融合头参数量要小；
- 融合只用 inner-OOF 选结构/权重。

### E5. 深度平滑

- 连续分支可加入相邻深度平滑（TV/L2/CRF 式正则）；
- 平滑必须在原子硬切换之前；
- 不能跨越 atom/continuous 边界平滑；
- 平滑强度在 inner-OOF 选择，并报告对原子边界的影响。

### E6. 井级分支

- 只做辅助、小容量、强正则；
- 所有井级偏差/校准参数只在 inner-OOF 选；
- 必须做消融；前代数据表明 80 井上井级校准极易过拟合。

---

## 7. 修改 F：解码与后处理

### F1. 逐目标 `τ_t` 与 joint guard

`E6/P1/PLAN.md` 改为：

1. 对每个目标 `t`，在 inner-OOF 上网格搜索 `τ_t`，目标函数 `argmax 100*w_t*Acc_t(τ)`；
2. 选阈值时同时看：
   - `atomic_precision/recall/F1`
   - `atomic_acc`
   - continuous slice `Acc`
   - 阈值-分数曲线平台区；
3. `joint_guard` 是可选门禁：
   - `q_joint > tau_joint_high` 时全部输出 atom；
   - 默认关闭；
   - 只在 inner-OOF Total 提升且 CI 下界 >0 时启用。
4. `τ_t` 必须在每个 outer 折内由 inner-OOF 选出，outer 折只推理一次。

### F2. SW 灰色地带

- 不按 F1 选阈值；
- 对 `q_sw` 灰区，用 inner-OOF 的期望总分判断；
- 可先按 q 分箱，估计 atom action 与 continuous action 的期望分，再在 bins 上做 isotonic/单调化；
- 最终仍要落成可复算的单调动作表或 τ 数组。

### F3. POR 吸附决策

- `q_por` 高置信 → 精确 0.1；
- 灰区用 inner-OOF 期望分判断“吸附到 0.1”还是“保留连续预测”；
- 不要按固定 F1 或准确率选。

### F4. 连续头后处理

- 逐目标偏置/缩放/分位数收缩；
- 参数只在 inner-OOF 选择；
- 报告敏感性曲线；
- SW 只允许在 `[0,100]` 内软裁剪，严禁全局压到 `[0,1]`；
- 原子分支保持精确值，后处理不能改变原子输出。

### F5. 必须上报

E6/E7 的 Gate 至少报告：

- 逐目标 `atomic_acc / precision / recall / F1`
- 逐目标 `tau_t`
- 逐目标 continuous slice Acc
- joint atom Acc / AUC / AP
- 总分分解与阈值曲线
- 误判代价矩阵
- `inner_only_selection` 证据

---

## 8. 修改 G：工程基础件

### G1. 全量 90 井泄漏回归进 E0/Local Gate

- 把 `tools/check_data_leak.py` 的核心逻辑抽成可 import 函数；
- `run_all.py` 或 `E0_local_contract_gate` 必须执行全量 90 井，不允许 18 井抽样通过；
- Gate mandatory 增加 `input_no_label_leak_full`，并写入报告。

### G2. 提交契约硬校验

`src/inference/contract.py` / `predict.py` 增加：

1. 10 口测试井硬校验；
2. 每口井预测行数 == 输入文件行数；
3. `depth_alignment_report` 必须调用，逐行深度误差 ≤1e-6；
4. SW 标签尺度守卫：
   - 例如 SW 预测中位数 <1.0 判失败；
   - 或对 atom/continuous 分布做范围报告；
5. `--expected-wells` 正式传入校验；
6. 任何一项失败即非零退出，不能只 warning。

### G3. 训练折归一化参数落盘

- SW 的 `mu/sigma`、POR 的 `scale/por_max`、特征 scaler 全部写入 manifest / scaler JSON；
- E1/P0、E5/P0、E5/P2 增加“参数只在训练折 fit，推理可反变换”的单测；
- 禁止跨折 fit。

---

## 9. 对现有计划文件的具体修改清单

| 文件/章节 | 修改内容 |
|---|---|
| `PLAN.md §5.1` | 架构图：H0 拆为 `q_joint + q_atom[3]`，连续头与原子头分开 |
| `PLAN.md §5.2` | 增加 U-Net padding/TCN 非因果/PatchTF overlap/门控融合/平滑/井级小容量要求 |
| `PLAN.md §5.3` | 输出头表：H0→H0joint + H_atom(POR/PERM/SW)；POR/SW 参数化更新；SW 单尺度 |
| `PLAN.md §5.4` | 损失：masked_mean、PERM clamp、aux 归一化、边界聚焦、λ_atom/λ_joint |
| `PLAN.md §6.4` | 原子保护：逐目标硬切换、τ_t 期望分、joint_guard 可选、禁止插值 |
| `PLAN.md §7` | E5/E6/E7/E8 描述同步 |
| `PLAN.md §8.2` | Gate：逐目标 atomic Acc/P/R/F1、τ_t、连续切片、总分 |
| `PLAN.md §10` | 风险表：joint-only 误伤、SW 归一化、POR 下界、NaN loss |
| `E1/P1/PLAN.md` | 连续头参数化、loss 修正、SW fold 归一化、边界聚焦作为可选消融 |
| `E5/P0/PLAN.md` | POR 参数化替代方案与消融 |
| `E5/P1/PLAN.md` | PERM log 空间截断、初始化、尾部报告 |
| `E5/P2/PLAN.md` | SW fold 归一化、q_sw、连续头反变换、单尺度硬约束 |
| `E6/P0/PLAN.md` | 由“联合状态头”改为“逐目标原子头 + joint 辅助”；两阶段训练 |
| `E6/P1/PLAN.md` | τ_t 用官方总分搜索；joint_guard 可选；报告原子 P/R/F1 |
| `E6/P2/PLAN.md` | 组装新头、新解码、新 Gate |
| `E7/P0/PLAN.md` | aux 归一化、边界聚焦、PERM clamp 的消融 |
| `E7/P1/PLAN.md` | 逐目标期望分解码、内层 OOF、敏感性 |
| `E8/P2/PLAN.md` | EMA/SWA、top-k 快照、同源性报告、CI 判据 |
| `src/models/row_mlp.py` 或 `heads.py` | 新增 q_atom[3]、q_joint；POR/SW 参数化 |
| `src/losses/score_aligned.py` | masked_mean、align_score_log、aux、boundary weight |
| `src/inference/atomic_gate.py` | 逐目标 τ_t、joint_guard、期望分搜索 |
| `src/training/loop.py` | 两阶段、EMA/SWA、样本权重、snapshot |
| `src/features/basic.py` / `row_dataset.py` | SW/POR fold scaler 参数与反变换 |
| `src/inference/contract.py` / `predict.py` | 深度对齐、每井行数、10 井、SW 尺度硬校验 |
| `versions/prereg_templates/E6_*.json` | 增加 per-target atomic 指标和 mandatory checks |
| `versions/candidates.json` | atomic 字段改为 `{por_acc,perm_acc,sw_acc, por_precision,...}` |
| `E0/code/run_all.py` | 全量 leak；cache 默认；报告新指标 |
| `tools/check_data_leak.py` | 抽成库函数供 Gate 调用 |

---

## 10. 建议实施顺序

1. **E0 工程基础件**：全量泄漏 Gate、契约硬校验、cache/e0 接线。
2. **E1 连续损失与参数化**：masked_mean、PERM clamp、SW fold 归一化、POR 参数化；
   先保证 E1 基线正确，避免错误目标函数进入 E3 对照。
3. **E5 逐目标连续头**：POR/SW/PERM 三个头按新参数化实现。
4. **E6 原子层次**：q_atom[3] + q_joint、两阶段训练、逐目标 τ_t 期望分解码。
5. **E7 损失/解码消融**：aux 归一化、边界聚焦、后处理敏感性。
6. **E8 训练策略**：EMA/SWA、快照集成、同源性报告。
7. **E3/E4 序列主干优化**：在 E1/E5/E6 固定目标函数与解码后，再投入 U-Net/TCN/PatchTF 优化。

> 理由：如果先堆序列主干而不修原子头、输出参数化和损失，E3/E4 只会更快优化一个错误目标；
> 先把 E1/E5/E6 的目标函数和解码修正确，序列主干才有可比较的增益基线。

---

## 11. 验收与 Gate 修改建议

### E6 Gate 新 mandatory checks

- `per_target_atom_acc_reported`
- `per_target_atom_precision_recall_f1_reported`
- `joint_atom_auc_reported`
- `tau_t_inner_oof_only`
- `no_atom_continuous_interpolation`
- `contract_ok`
- `atomic_precision_reported`
- `disk_budget_ok`
- `training_time_log_valid`
- `checkpoint_resumable`
- `no_label_leak_full`

### E6 量化门槛

- 每个目标 `atomic_acc ≥ 0.99`；
- 每个目标 `atomic_recall ≥ 0.98`（至少 POR/PERM/SW 都要报，不能只看 joint）；
- OOF Total ≥ 82.0；
- `tau_t` 选择过程可复算，且只看 inner-OOF；
- 若 joint_guard 启用，必须证明它在 inner-OOF 上带来正增益且 CI 下界 >0，否则默认关闭。

### E7 Gate

- `align_loss` 在三段式下优于纯 aux；
- 边界聚焦消融表完整；
- `masked_mean NaN` 单测通过；
- PERM 低估尾部与官方评分器一致性报告。

### E8 Gate

- EMA/SWA/快照至少一个策略在 inner-OOF 上 ≥ 最佳单成员；
- CI 下界 >0；
- 成员相关性报告完整；
- 同源平均不得计为增益。

---

## 12. 最小可行版本（如果只做三项）

如果机时/周期紧张，只做以下三项即可获得主要边际收益：

### MVP-1：逐目标原子头 + 期望分解码

- 替换 H0-only 为 `q_atom[3] + q_joint`；
- 每个目标独立 `τ_t`，inner-OOF 用官方总分搜索；
- 先不做 joint_guard（默认关闭）。

### MVP-2：POR / SW 输出参数化修正

- POR 改为 `por_max * sigmoid(g)` 或 `softplus(g)-softplus(g0)`；
- SW 改为训练折内 `mu/sigma` 仿射归一化，输出反变换；
- 原子分支保持精确值。

### MVP-3：PERM/SW 损失修正

- `masked_mean` 防 NaN；
- `align_score_log` 加入 `max(ŷ/y,ε)` 截断；
- `L_aux` 改标准化空间，SW 不再被 99.9 主导。

**MVP 后的 Gate**：
- E6 逐目标原子 Acc ≥0.99；
- E5/E6 连续切片 Acc 不退化；
- 总分相对旧 H0-only 结构有正增益且 CI 下界 >0；
- E0 的 90 井泄漏 Gate + 提交契约硬校验先通过。

---

## 13. 风险与停止规则

| 风险 | 早期信号 | 对策 |
|---|---|---|
| 逐目标原子头互相干扰 | 某目标 atom recall 上升、另两个下降 | 独立 loss 权重；per-target sample weight；先不共享原子头 |
| τ_t 在 inner 上过拟合 | inner 好、outer 差 | 只取平台区中点；报告敏感性；必要时用分箱期望分做单调动作表 |
| SW 归一化参数跨折不一致 | 逐折差异大 | 只允许训练折 fit；报告每折 `mu/sigma`；用鲁棒统计量 |
| POR 参数化后原子 0.1 学不准 | POR atom Acc 下降 | 原子分支独立硬切换；POR 连续头不负责精确 0.1 |
| joint_guard 误伤 | 非 joint 行三目标全被覆盖 | 默认关闭；仅当 inner-OOF Total 正增益且 CI 下界 >0 才启用 |
| 两阶段训练第二段遗忘原子头 | atom Acc 下降 | 冻结或极低 lr；inner-OOF 监控 atom Acc；必要时联合微调 |
| EMA/快照提升不显著 | CI 含 0 | 停止该策略，不包装同源平均 |
| 边界平滑跨原子边界 | 原子行被平滑出容差 | 平滑只作用于连续分支，且按 atom mask 断开 |

---

## 14. 结论

建议把 v4 的模型核心从“**一个 joint 占位头 + 连续头**”升级为：

```text
共享主干
├── q_joint        （辅助/可选高置信门禁）
├── q_atom[3]      （逐目标原子，主保护）
├── cont_por       （修正参数化）
├── cont_perm      （PERM log 空间 + 官方截断）
└── cont_sw        （训练折仿射归一化，输出标签尺度）
        ↓
逐目标硬切换 τ_t，τ_t 由 inner-OOF 官方总分选择
        ↓
连续后处理（inner-OOF 选参，禁止全局 SW [0,1] 裁剪）
```

这条路线比继续加深 RowMLP 更直接地对准官方评分的“原子白送分 + 连续容差”结构，
且所有新增部分都可预注册、可复算、可消融，不引入新的提交风险。
