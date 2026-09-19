# v4 总体计划：深度学习单主干、评分对齐损失、全量自包含管线

> **v4 的定位（一句话）**：放弃“在 v1/v2 树模型上打补丁”的路线，改为**从零构建一条纯深度学习管线**——深度序列主干（长感受野）输出逐行表示，三个目标头 + 联合常量状态头，用**与官方评分同构的可微损失**训练，最终以**自包含、CPU 可推理**的提交包一次性产出 `result.json`。
>
> **本计划与前代的唯一分歧点**：v2/v3 都被“CPU-only + 单任务 ≤100h”锁死在树模型上（v2 的 `资料库/08` §0.3 第 2 层 U-Net/TCN 从未真正训练过，v2 E4/P3 的 CPU MLP 是 NO-GO，但那是在**无序列上下文、无 GPU**条件下取得的结论，**不构成对 GPU 深度序列模型的证伪**）。v4 拥有 **1× A100 80GB**，因此把 v2 `/PLAN.md` 里被降级的 E7 序列模型重新提升为**主线**。
>
> 总原则：**先口径与数据，后模型；先行级基线，后序列主干；先原子保护，后连续精度；先本地诚实 CV，后提交。**

---

## 零、用户决策记录（本次会话确认，优先级高于前代计划）

| 编号 | 决策项 | 用户选择 | 对计划的影响 |
|---|---|---|---|
| D1 | 单任务 wall-clock 上限 | **放宽——忽略 100h，能跑多久跑多久** | 不再设置单任务 100h 硬门禁；改为**软预算 + 强制 checkpoint/可续训**，任何任务被中断都能从最近 checkpoint 恢复。计划中所有“预算”均为软预算。 |
| D5 | 云端软件栈 | **CUDA 12.6 + PyTorch 2.4.0 + Python 3.11 为镜像预装版本，不得变更/升级/另装 CUDA；无 conda；但 `pip` 可用，可安装额外的轻量纯 Python 依赖** | 允许 `pip install numpy/pandas/scipy/...`；**禁止** `pip install torch`（换版本）与任何需现场编译 CUDA 扩展的包（flash-attn/xformers/apex/deepspeed）。所有第三方依赖仍走「探测 + 降级」层，见 §3.3。 |
| D6 | 云端可用磁盘 | **仅 30 GB（含镜像已占部分）** | pip 只装轻量包并立即清缓存；强制「按需生成特征 + checkpoint 滚动淘汰 + 中间产物即时清理」，并用 `du` 实测。见 §3.4。 |
| D2 | 架构 | **深度序列主干 + 行级精度头** | 1D U-Net / TCN / Patch-Transformer 做深度上下文主干；行级精度头做逐点精修；再叠学习型原子门。见 §五。 |
| D3 | 起点与保护 | **从零纯 DL 管线** | 不继承 B0 权重、不做 B0 patch 隔离；**但“联合常量占位必须精确命中”由模型内部的联合状态头承担**（自包含，不依赖 B0）。见 §六.4。 |
| D4 | 交付形态 | **权重随包 + CPU 可推理，训练可选** | 提交包含 `models/` 权重；`predict.py` 纯 CPU、确定性、无网络；`train.py` 可在 A100 上完整重训但不是评测必需。见 §九.3。 |

> D1 的执行纪律：虽然用户允许超 100h，但**每一步训练都必须写 checkpoint 并支持 `--resume`**，且每个 P 的 Gate 报告必须记录 `actual_h`。原因：A100 租用是按时的，且赛题 `rules.md` 原文（“每个模型训练任务 wall-clock 上限 100 小时”）仍是唯一书面规则，保留 ≤100h 可行性是零成本的保险。

> **E0-R2 修订（2026-09-19，独立审查后的修复，最高优先级）**：独立审查（[`reports/V4_PLAN_REVIEW.md`](reports/V4_PLAN_REVIEW.md)）发现三处会直接导致结论无效的错误，已全部修复并加回归测试：
> 1. **输入列泄漏**：`parse.py` 的 `inputs = arr[:, 1:15]` 使索引 14（**POR 标签**）成为第 14 个输入，80 口井全部泄漏；且测试井只得到 13 列（提交必崩）。→ 改为 13 条曲线 + DEPTH 分离，布局常量集中在 `parse.py`，列布局与 `with_targets` 无关，新增 `input_no_label_leak` 回归（90 井全过）。
> 2. **SW 尺度误判**：实测有效 SW 为 min **8.305** / median **82.805** / max **99.9**，SW<1 仅 **11** 行 → SW 是**单一标签尺度（百分数）**，取消"×100 双尺度"前提；`SW_SMALL_BRANCH=False`；仍然严禁全局裁剪到 [0,1]。
> 3. **非规范井描述错误**：只有 `c7611b01` 缺 CASE；`42f2870b`/`b7eb1274` 含 CASE，只是多了 K/U/CGR → 17,426 行"多列" + 9,654 行"缺 CASE"。
>
> 同时修复：E0 Gate 拆分为**本地契约 Gate**（10/10 PASS）与**云端 Gate**（`blocked_pending_cloud_run`）；候选注册表逐目标分数按实测回填并加总分恒等式校验（`tools/check_consistency.py`）；`build_cache` 纳入 E0 产出（32.4 MB，90 井输入列校验通过）；33 份 P 级 Gate 预注册模板全部通过 `src/validation/gates.py::validate_prereg`；新增 `tools/check_status.py` 校验状态台账与实物一致。
>
> **E0-R1 修订（2026-09-19，本机复算后的契约修订）—— ⚠️ 本节已被下方 E0-R2 取代**，保留仅作历史追溯；其中「3 口井都缺 CASE」的表述是错的。
>

>
> **E0-R2 修订（2026-09-19，独立审查后的修复，最高优先级）**：独立审查（[`reports/V4_PLAN_REVIEW.md`](reports/V4_PLAN_REVIEW.md)）发现三处会直接导致结论无效的错误，已全部修复并加回归测试：
> 1. **输入列泄漏**：`parse.py` 的 `inputs = arr[:, 1:15]` 使索引 14（**POR 标签**）成为第 14 个输入，80 口井全部泄漏；且测试井只得到 13 列（提交必崩）。→ 改为 13 条曲线 + DEPTH 分离，布局常量集中在 `parse.py`，列布局与 `with_targets` 无关，新增 `input_no_label_leak` 回归（90 井全过）。
> 2. **SW 尺度误判**：实测有效 SW 为 min **8.305** / median **82.805** / max **99.9**，SW<1 仅 **11** 行 → SW 是**单一标签尺度（百分数）**，取消"×100 双尺度"前提；`SW_SMALL_BRANCH=False`；仍然严禁全局裁剪到 [0,1]。
> 3. **非规范井描述错误**：只有 `c7611b01` 缺 CASE；`42f2870b`/`b7eb1274` 含 CASE，只是多了 K/U/CGR → 17,426 行"多列" + 9,654 行"缺 CASE"。
>
> 同时修复：E0 Gate 拆分为**本地契约 Gate**（10/10 PASS）与**云端 Gate**（`blocked_pending_cloud_run`）；候选注册表逐目标分数按实测回填并加总分恒等式校验（`tools/check_consistency.py`）；`build_cache` 纳入 E0 产出（32.4 MB，90 井输入列校验通过）；33 份 P 级 Gate 预注册模板全部通过 `src/validation/gates.py::validate_prereg`；新增 `tools/check_status.py` 校验状态台账与实物一致。
>
> **E0-R1 修订（2026-09-19，本机复算后的契约修订，最高优先级）**：执行 E0 口径层时发现**两个会静默吃掉分数的硬事实**，已冻结进契约：
> 1. **3 口训练井的表头不是官方 17 列**（`42f2870b` 20 列含 K/U/CGR 缺 CASE；`b7eb1274` 21 列含 TH/K/U/CGR 缺 CASE；`c7611b01` 16 列缺 CASE），共 **27,080 行（3.71%）**，且这 3 口井**都在 80 井折内、三目标齐全**。→ **禁止按列位置解析**，必须按表头名对齐、缺列补 `NaN`、多余列忽略，并对内部列宽做硬断言。开发中已实际触发一次 numpy 越界切片静默截断（丢掉 SW 列且不报错）。
> 2. **评分分母口径确定**：常数基线 (0.1, 0.01, 99.9) 在 `missing_mode="drop"`（逐目标排除缺测行）下 = **70.490735**，命中锚点 70.4907 ±1e-4；在 `"mask"`（全行分母）下 = 69.843218。→ **全项目统一使用 `drop`**，所有 OOF 数字必须标注该口径。
>
> 同时已复算并冻结：训练 80 井 / **730,268 行**；测试 10 井 / **95,948 行**；状态计数 缺测 **6,700** / 占位 **487,225** / 有效 **236,343**；折指纹 sha256 `f7c2c58b…d94b87e`（80 井 / 5 折）。E0 **本地契约 Gate 10/10 通过**（`reports/E0_local_contract_gate.json`）；云端 Gate 待 P0（`E0_cloud_gate.json` 状态 `blocked_pending_cloud_run`）。

---

## 零之二、计划完成度与执行状态

| 层级 | 数量 | 篇幅 | 状态 |
|---|---:|---:|---|
| 总计划 `PLAN.md` | 1 | **729 行** | ✅ 完成 |
| 阶段计划 `E*/PLAN.md` | 12 | 平均 59 行（合计 711） | ✅ 完成 |
| P 级子计划 `E*/P*/PLAN.md` | 33 | **平均 150 行**（合计 4,982） | ✅ 完成（V2 深度：输入/输出契约、执行步骤、参数表、完成判据、禁止事项、风险对策、停止规则、inner-OOF 选择协议、复算命令、Gate 预注册 JSON） |
| 计划文件合计 | 46 | **6,423 行** | ✅ |

> **行数由 `tools/check_status.py` 与 `wc -l` 实测，不手写**（审查 H3：此前手写数字已过期）。

**执行状态唯一事实源**：[`versions/status.json`](versions/status.json)（阶段/P 级状态 + 证据 + 锚点 + 环境）。

### E0（已完成，本机可复算）

| P | 内容 | 状态 | 证据 |
|---|---|---|---|
| P0 | 云端环境与磁盘实测 | ⏸ 待云端（本机无 GPU/torch） | `E0/code/check_env.py`、`setup_deps.sh` |
| P1 | 数据卡、哨兵与标签三状态 | ✅ | `reports/E0_data_card.json`、`E0/docs/data_card.md`、`versions/folds_sha256.json` |
| P2 | 评分器与分母口径冻结 | ✅ | `src/score.py`、常数基线 **70.490735**（锚点 70.4907） |
| P3 | 提交契约、版本路由、干净目录冒烟 | ✅ | `predict.py`、`reports/E0_contract_tests.json`（6 负样例全拒绝） |

**E0 本地契约 Gate：10/10 mandatory PASS**（`reports/E0_local_contract_gate.json`）；
**E0 云端 Gate：`blocked_pending_cloud_run`**（需 E0/P0 在 A100 任务实测，`reports/E0_cloud_gate.json`）。

E0 的两个硬发现（已冻结进契约，详见 §6.1 与 [`E0/docs/data_card.md`](E0/docs/data_card.md)）：

1. **3 口训练井 schema 非规范**（`42f2870b` 20 列含 K/U/CGR、`b7eb1274` 21 列含 TH/K/U/CGR、`c7611b01` 16 列缺 CASE），共 **27,080 行（3.71%）**，且都在 80 井折内、三目标齐全 → **必须按表头名解析**，并对内部列宽做硬断言（开发中曾因 numpy 越界切片静默截断丢掉 SW 整列）。
2. **评分分母口径 = 逐目标排除缺测（`drop`）**：常数基线在此口径下 70.490735（命中锚点），全行分母口径为 69.843218（低 0.65 分）。

### 后续阶段

E1–E11 全部处于 `pending`，按 §七 的顺序执行；每个 Gate 的阈值、mandatory checks 已在对应 P 级计划第 13 节给出预注册模板。

---

## 一、最终目标

1. **赛事目标（`rules.md` §三、§七.4）**：B 榜综合得分 **> 75**（晋级线）。这是唯一外部硬指标。
2. **架构目标**：`v4/` 提供一条**端到端纯深度学习**管线，从 14 条测井曲线直接预测 POR / PERM / SW，不调用 v1/v2/v3 的任何模型或代码。
3. **入口契约**：
   ```bash
   # 训练（在 A100 云端机器上执行，非评测必需）
   python train.py --train-dir data/train --model-dir models/v4 --config configs/v4_pd.yaml
   # 推理（官方口径，评测机执行；纯 CPU、确定性）
   python predict.py --data_dir ./data --output result.json
   # 版本/诊断
   python predict.py --list-versions
   python predict.py --use-version PD1 --data_dir ./data --output result.json
   ```
4. **量化目标（本地诚实 CV，80 井按井 5 折 OOF）**：

   | 里程碑 | 本地 OOF Total | 说明 |
   |---|---:|---|
   | 全常量基线 | 70.4907 | v1 E1 冻结值，v4 必须复算一致 |
   | E1 纯 DL 行级基线 | **≥ 78.0** | 硬 Gate：低于此值说明数据/损失口径有 bug，不得进入 E2 |
   | E3 序列主干 | **≥ 81.0** | 硬 Gate：序列上下文必须带来可测增量 |
   | E6 完整 PD 管线 | **≥ 82.0** | 硬 Gate：必须超过 B0 本地锚点 80.382479 |
   | E10 冻结候选 | **≥ 82.5** | 冲刺目标；未达 82.0 时 E10 启用回退协议（§九.5） |

5. **每个版本的硬契约**（`v4/src/inference/contract.py` 自动校验）：输出 **10 口测试井 / 95,948 行**、`logId` 与文件名一致、深度与输入逐行对齐、小写 `depth`、POR/SW 按训练标签尺度输出（**禁止把 SW 裁剪到 [0,1]**）、PERM 严格为正且有限、无 NaN/Inf、推理阶段不读标签、随机种子固定、纯 CPU 单次运行 < 30 min。

---

## 二、为什么是这条路线：从 v1/v2/v3 提炼的确定性结论

### 2.1 继承的“确定性事实”（不重新论证，直接作为设计输入）

| 事实 | 数值/来源 | 对 v4 的含义 |
|---|---|---|
| 评分是三目标加权相对误差，带 `max(0,·)` 截断 | `rules.md` §7.3–7.4 | 损失必须**与评分同构**（对齐损失），MSE 是错的目标函数 |
| 常量占位 `(0.1, 0.01, 99.9)` 占 **66.719%**（487,225/730,268） | `资料库/12` §3.1 | 占位行是“白送分”，必须精确命中；POR 容差仅 `0.08×0.1 = ±0.008` |
| 占位行白送分 **66.72** | `资料库/12` §3.3 | 任何模型只要退化为“只在有效行预测”就会掉到 ~50 分 |
| PERM 必须 log10 域建模 | `资料库/12` §2.4 提示 3、`资料库/05` §2.3 | 网络输出 `z=log10(PERM)`，`PERM=10^z` |
| SW 占位 99.9 与有效值域 [0,1] 相距极远，需**双分支** | `资料库/12` §3.4 | SW 头 = `q̂·99.9 + (1-q̂)·σ(f)`，`q̂` 由 BCE 监督 |
| 相邻深度点强自相关，必须**按井分组** CV | `资料库/07` §8、`资料库/12` §3.5 | 折维度恒定为井；标准化/分位数只在训练折 fit |
| 评分对齐的三段式可微损失已给出参考实现 | `资料库/12` §2.2–2.4 | v4 直接采用 Charbonnier + softplus 平滑方案，见 §六 |
| 技术梯度推荐：第 2 层 = 1D U-Net / TCN；第 3 层 = Patch 化 Transformer | `资料库/08` §0.3 | v4 的主力正是这两层 |
| 多任务硬共享通常优于三个独立模型，但须解决权重失衡 | `资料库/08` §1.4 | 主干硬共享 + MMoE/加权损失做任务平衡 |
| 单井内工程曲线（CAL/DEVI/AZIM/BIT/CASE）近似常数，只提供**井间**区分度 | `资料库/08` §0.1 第 4 点 | 需要显式“井级特征”分支，不能只做逐点 |
| v2 从零树模型管线最好 dev64 = 80.305222，低于 B0 的 80.382479 | `v2/README.md`、`v2/PLAN.md` §五 | 从零路线在**树模型 + 手工特征**下已触顶 → 需要换假设空间，而不是继续调树 |
| v2 E4/P3 CPU MLP = NO-GO | `v2/PLAN.md` §五（E4 结果） | 该结论的边界条件：**逐点、无序列、CPU、小容量**；v4 不违反该边界（有序列、有 GPU、大容量） |
| A 榜 5 口井噪声带约 ±0.02 分，过拟合 A 榜会掉 3–10 分 | `资料库/12` §3.5、`v3/PLAN.md` §八.1 | A 榜只做短名单仲裁与崩坏体检，不做细粒度调参 |

### 2.2 被 v4 明确**放弃**的路线（负知识资产）

| 放弃项 | 原因（已有证据） |
|---|---|
| B0 patch 隔离 / 在 v1 E7 上打补丁 | 用户决策 D3；且补丁路线的 A 榜增量在噪声带内（+0.0278） |
| 树模型主线（LightGBM/XGBoost） | 作为**辅助基线**保留（E1 需要分母），但不作为最终候选 |
| 手工窗口特征（CW 之类）+ 树 | v2 E3 已证明其收益来自“上下文”，而序列主干以更好的方式提供同一信息 |
| 井级标签校准 / 伪标签（先验） | v1 E8–E11、v2 E6 均为 NO-GO；v4 只在 E8 以“transductive 推理期适配”形式**作为消融**重试一次 |
| 直接裁剪 SW / POR 到物理区间 | `资料库/12` §0 结论 2：直接损失约 23 分 |
| 用 A 榜分数做细粒度调参 | `资料库/12` §3.5 |

---

## 三、算力、环境与两机分离协议

### 3.1 两机分离（本计划最重要的工程约束）

| | 本机（开发机） | 云端（训练机） |
|---|---|---|
| 角色 | 写代码、造数据、跑轻量单测、组装提交包、本地诚实 CV 的记录与汇总 | 训练、全量 OOF 推理、多 seed 集成 |
| 硬件 | 无 GPU；`v2/.venv` 无 torch | **1× A100 80GB**，4000m vCPU，**16 GiB 系统内存**，80 GiB 显存 |
| 必须能力 | **在没有 torch 的情况下**也能校验数据契约、评分口径、提交格式（这些全部用 numpy/pandas 实现） | 能在 ≤16 GiB 系统内存下流式加载数据 |

**由此派生的四条硬性工程规则**：

0. **环境不可变铁律（最高优先级）**：云端镜像**预装且不可替换**——CUDA 12.6、PyTorch 2.4.0、Python 3.11；**没有 conda**。因此：
   - **禁止** `pip install torch` / 升级 CUDA / 更换 Python 版本 / 用 conda 建环境（会破坏镜像一致性，且 30 GB 磁盘容不下第二份 torch）；
   - **允许** `pip install` **额外的轻量纯 Python / 纯 wheel 依赖**（numpy、pandas、scipy、scikit-learn、pyarrow、einops 等），但必须 `--no-cache-dir` 且装完 `pip cache purge`；
   - 所有第三方依赖仍**统一走 `portability` 探测层**（§3.3.2）：装了就用，没装就走 numpy/torch 兜底分支，**任何 `import` 失败都不得变成"请用户去装包"**；
   - 提交包里的 `requirements.txt` 是**声明**（`rules.md` §6.2/§8.3 要求列明环境），默认不被执行；评测环境用的是同一套预装镜像。
1. **代码契约**：`v4/src/data/`、`v4/src/score.py`、`v4/src/inference/contract.py` **只依赖标准库 + numpy/pandas**，不 import torch。这样本机（无 GPU、无 torch）能跑通全部数据与提交侧单测，云端跑训练与推理。
2. **数据载体**：训练数据以一次性生成的按井分片 `npz` 传递（`pack_dataset.py` 产出，730k×17 float32 ≈ 50 MB）。云端只接受本机上传的这份缓存，不依赖任何外部下载。
3. **内存纪律（16 GiB 系统内存 + 80 GiB 显存 = 极端不对称，瓶颈永远在系统内存）**：
   - 训练时**不把整井序列常驻内存**；`Dataset` 按井按需读取（打开分片 → 取所需窗口 → 关闭），**禁止把分片内容缓存在 Python 全局字典里**；
   - 特征预计算缓存使用 `float32` 且**按井分片落盘**（`cache/feat/<F>/<well>.npz`），训练时按需读取；
   - `DataLoader` 使用 `num_workers = 4`（**不是 8**：40 vCPU 但只有 16 GiB RAM，8 个 worker 的 numpy 副本会把内存吃光）、`persistent_workers=False`、`pin_memory=True`、`prefetch_factor=2`；
   - 每个 worker 的常驻内存必须 < 300 MB（用 `torch.utils.data.get_worker_info()` 做断言式自检）；
   - 任何“先全量 concat 再训练”的写法一律禁止（这在 16 GiB 上会 OOM）；
   - 显存侧相反：80 GB 显存允许把 batch 开到内存允许的最大值，用 bf16 提高吞吐（显存不是约束）。
4. **磁盘纪律（30 GB 上限，见 §3.4）**：只装必需轻量包并清缓存；**按需生成特征、按版本目录落盘、checkpoint 滚动淘汰**；把 `TORCH_HOME`/`XDG_CACHE_HOME` 重定向进项目目录以便统一清理；所有训练脚本每 epoch 调 `assert_disk_headroom(8.0)`。

### 3.2 软预算（D1 后的口径）

| 任务类型 | 软预算 | 超限处置 |
|---|---|---|
| E1 行级基线（全 80 井 5 折） | 单折 ≤ 20 min，总计 ≤ 3 h | 减 epoch / 减宽度 |
| E3 序列主干（5 折 × 1 seed） | 单折 ≤ 3 h，总计 ≤ 20 h | 减 chunk 长度 / 减 epoch |
| E6 评分对齐微调 + 解码搜索 | 总计 ≤ 10 h | 冻结主干、只调头 |
| E8 集成（3–5 个成员） | 总计 ≤ 40 h | 减成员数 |
| E10 全量重训 | 单次 ≤ 30 h | 复用 5 折权重做集成替代全量重训 |

> **磁盘预算同样按任务登记**：每个任务结束时把 `df -h /` 剩余量与 `du -sh` 分类占用写入 `training_time_log.json` 的 `disk_free_gb_end` / `disk_breakdown`。任何任务启动前若 `disk_free_gb < 8`，必须先清理 `cache/feat/<旧版本>/`、`models/<旧候选>/last_prev.pt`、`cache/tmp/`，再启动。

### 3.3 云端环境与依赖（pip 可用，但基础栈不可变）

#### 3.3.1 已知基线

```
OS        : Linux x86_64
Python    : 3.11（预装，不得更换/不得用 3.12+ 语法）
PyTorch   : 2.4.0 + cu124（预装，匹配 CUDA 12.6 驱动；不得 pip 改版本）
GPU       : 1× A100 80GB（sm_80），bf16 可用
环境管理  : 无 conda；直接用系统 Python；pip 可装额外轻量包
磁盘      : 30 GB（含镜像本身）
```

**一次性依赖安装（`E0/code/setup_deps.sh`，只装"确实需要且体积小"的包）**：

```bash
export PIP_NO_CACHE_DIR=1
# 基础栈已由镜像提供：torch / numpy；下面只补确实用到的轻量包
python -m pip install --no-cache-dir \
    "pandas>=2.0,<3" "pyarrow>=15" "scipy>=1.11" "scikit-learn>=1.4" \
    "einops>=0.7" "onnx>=1.16" "onnxruntime>=1.17"
python -m pip cache purge          # 30 GB 磁盘，装完立即清
python -m pip freeze > v4/versions/locks/cloud_frozen.txt
python v4/E0/code/check_env.py --json v4/reports/E0_env.json
```

**安装纪律（针对 30 GB）**：
- 只装上表所列；**不装** `tensorboard`/`matplotlib`/`jupyter`/`wandb`/`torchvision`/`timm`（用 CSV+JSON 日志替代绘图与可视化）；
- **不装**任何需要现场编译 CUDA 扩展的包（`flash-attn`/`xformers`/`apex`/`deepspeed`）——注意力统一走 `F.scaled_dot_product_attention`（PyTorch 2.4 原生，自动选择 Flash / Memory-Efficient / Math 后端）；
- `pip install` **不允许**触碰 `torch`、`nvidia-*`、`cuda-*` 系列（会触发版本替换或重复下载数 GB）；
- 每次安装后必须 `pip cache purge` 并跑 `check_env.py`；若 `pip` 解析出要动 torch，**立即中止并按已装版本改代码**，不做任何环境变更。

**PyTorch 2.4.0 兼容红线（写进 `E0/code/check_env.py`，不通过则禁止开始训练）**：

| 项 | 要求 |
|---|---|
| `torch.__version__ == "2.4.0"` | 断言；否则报错退出 |
| `torch.cuda.is_available()` 且 `torch.cuda.get_device_capability() == (8, 0)` | 断言 A100 |
| `torch.cuda.is_bf16_supported()` | 断言，决定用 bf16 而非 fp16 |
| 禁用 2.5+ API | `torch.nn.attention`、`torch.export` 新接口、`torch.compile(fullgraph=True)` 的新参数一律不用 |
| 注意力 | 只用 `F.scaled_dot_product_attention` 的 2.4 已有签名（PyTorch 2.4 内置 Flash / Memory-Efficient / Math 三后端自动选择） |
| `torch.compile` | 可选，必须有 `--no-compile` 开关；编译失败不得中断训练 |
| 优化器/调度器 | 只用原生 `AdamW` + `torch.optim.lr_scheduler` |
| 数据加载 | `DataLoader(num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2)` |

#### 3.3.2 依赖探测与降级层（`v4/src/portability.py`）

**即便 pip 可用，仍必须保留降级层**：一是评测机与本机可能不同（本机无 torch、未必有 pandas/pyarrow），二是"能装"不等于"该依赖可被依赖"（例如 `onnx` 装不上时要有替代方案）。

| 能力 | 首选 | 兜底（必然可用） | 影响 |
|---|---|---|---|
| 数据解析 | `pandas.read_csv` | 标准库 `csv` + `float()` | 无（v4 自写解析器，见 E0） |
| 张量/数值 | `numpy` | **numpy 是硬前提**（torch 依赖它，必然存在） | 无 |
| 分片落盘 | `np.savez_compressed`（必须） | — | 无 |
| 列式 OOF 存储 | `pyarrow.parquet` | `np.savez_compressed` + 列名 JSON | 体积略大，无功能损失 |
| 分位数/标准化 | `numpy.percentile`（自写，训练折内 fit） | — | 无 |
| 模型导出（跨机兜底） | `torch.onnx.export`（2.4 内置） | 权重存 `.npz`（键名/形状清单） | 评测镜像预装 torch 2.4.0，因此 `torch.load(map_location='cpu')` 是主路径；ONNX 与 `.npz` 只是"torch 出现异常"时的兜底，不实现纯 numpy 前向（ROI 太低） |
| 进度/日志 | 标准库 `print` + CSV/JSONL | — | 不依赖 tqdm/tensorboard |
| 绘图 | **不做**（不需要） | — | 不依赖 matplotlib |

**规则**：`import` 可选库统一写成 `try: import X; HAS_X=True except ImportError: HAS_X=False`，并把探测结果写进 `reports/E0_env.json`（`{"pyarrow": true/false, "onnx": ..., "sklearn": ...}`），供后续阶段选择路径。**任何模块不得因为可选库缺失而崩溃**；同时**也不得因为可用就强依赖**（对 OOF/缓存这类产出，必须给出兜底格式）。

### 3.4 云端磁盘预算（30 GB，含镜像占用）

**先测后用，不许假设**。E0 的第一条命令就是测量镜像与文件系统的真实占用：

```bash
df -h /                      # 总容量与已用
du -sh /usr /opt /root 2>/dev/null   # 镜像本体占用
python v4/E0/code/check_env.py --json v4/reports/E0_env.json
python v4/src/data/disk_guard.py --min-free-gb 8 --report /home,/tmp \
       --json v4/reports/E0_disk_budget.json
```

#### 3.4.1 预算表（**可用量 = 30 GB − 镜像已占**；表内为"项目自身"占用）

| 项目 | 预算 | 控制手段 |
|---|---:|---|
| 额外 pip 包（pandas/pyarrow/scipy/sklearn/einops/onnx/onnxruntime，**不含 torch**） | **≤ 0.8 GB** | `--no-cache-dir` + 装完 `pip cache purge`；不装 tensorboard/matplotlib/jupyter/torchvision/timm |
| 数据集（80 训练井 + 10 测试井原始 txt） | **0.05 GB** | 原始文本仅 ~50 MB（730k 行） |
| 预处理分片缓存（`cache/raw/*.npz`） | **0.06 GB** | float32、按井分片 |
| 特征缓存（`cache/feat/<F>/*.npz`） | **≤ 2.0 GB** | 只缓存当前实验使用的组；换版本先删旧目录 |
| Checkpoint 滚动窗口 | **≤ 4.0 GB** | 每 run 只留 `best.pt` / `last.pt` / `last_prev.pt`，bf16 存储 |
| 实验中间产物 / 日志 / OOF | **≤ 1.0 GB** | OOF 只存 6 列 float32；中间张量不落盘 |
| 提交包与结果 | **≤ 1.0 GB** | `result.zip` ~1 MB；代码包 < 5 MB；权重 < 50 MB |
| 缓存重定向（`TORCH_HOME`/`XDG_CACHE_HOME` → 项目内） | **≤ 0.5 GB** | 便于统一清理，避免写到不可控的系统路径 |
| **项目合计** | **≈ 9.4 GB** | **其余留给镜像本体（含 torch ≈5–7 GB）与安全余量** |

**安全规则**：`assert_disk_headroom(8.0)` 是硬门禁。若实测**可用空间 < 12 GB**（即镜像已占 > 18 GB），则**立即收缩**：特征缓存上限降到 0.5 GB（改为训练时现算）、集成成员上限从 5 降到 2、模型宽度上限减半、checkpoint 上限从 4 GB 降到 2 GB，并把该决定写入 `reports/E0_disk_budget.json` 的 `contingency_applied`。若实测可用 < 8 GB，**先不装任何额外包**，只用镜像自带的 torch+numpy 跑 E0/E1，再决定是否装 pandas。

#### 3.4.2 省盘策略

1. **单一数据源**：`pack_dataset.py` 一次性把 90 口井打包为按井分片 `npz`（`cache/raw/<well>.npz`，float32，17 列），之后所有实验都从这里读，不再重复解析 txt。
2. **特征按需生成、按组落盘**：`F_raw`/`F_miss`/`F_depth` 由 raw 分片现算（便宜，不落盘）；只有 `F_win`（最贵）落盘，且按 `特征组 + 版本 + 折` 命名；换版本时**先删旧版本目录再生成**。
3. **不保存多头中间张量**：patch 化、增强、归一化全部在 `Dataset.__getitem__` 内即时完成。
4. **OOF 只存必要列**：`well_id, depth, pred_por, pred_perm_z, pred_sw, q_ph`（6 列 float32）。
5. **禁止**：每个 epoch 的完整验证预测落盘；未压缩的 fp32 模型快照；保留历史候选的权重（只保留 `frozen_best` 与最后 2 个候选）。

#### 3.4.3 Checkpoint 滚动淘汰（`v4/src/models/ckpt.py` 统一实现）

- 每个 run 目录只允许 `best.pt`、`last.pt`、`last_prev.pt`；
- 新 checkpoint 落盘前**先删** `last_prev.pt`（先删后写，禁止先写后删）；
- checkpoint 只存 `state_dict`（**bf16**）+ 量化后的优化器一阶/二阶矩 + epoch + 配置哈希 + 数据/折指纹；**不存源码副本**；
- 单个 checkpoint 上限 **1.2 GB**（≈300M 参数 bf16）；超过则强制减半宽度或开梯度检查点；
- `--time-budget-h` 到点、或磁盘剩余 < 5 GB 时，**先保存 `last.pt` 与日志，再优雅退出**（`--resume` 可续）。

#### 3.4.4 磁盘守卫（E0 Gate 必检）

`v4/src/data/disk_guard.py` 提供 `assert_disk_headroom(min_gb=8.0)`（已实现，含 `cleanup` / `save_and_exit` / `abort` 三级动作与 `disk_guard` 上下文管理器）：

> 不安装：`tensorboard`/`matplotlib`/`jupyter`/`wandb`（用 CSV+JSON 日志替代）、`torchvision`/`timm`（不需要图像侧依赖）、任何 CUDA 编译扩展。

> 每个任务开始时在 `v4/reports/training_time_log.json` 写入 `task/planned_h/started_at/git_rev`，结束时回填 `actual_h`、`best_epoch`、`peak_mem_gb`、`checkpoint_path`。**所有训练脚本必须实现 `--resume`、`--time-budget-h`（到点保存并优雅退出）与每 epoch checkpoint。**

## 四、目录与分层约定

采用与 v2 相同的三级结构（项目层 → E 阶段层 → P 子阶段层）：

| 层级 | 共享代码 | 本层专属代码 | 文档 | 结果/模型 |
|---|---|---|---|---|
| 项目层 | `v4/src/` | `v4/train.py`、`v4/predict.py` | `v4/docs/` | `v4/reports/`、`v4/experiments/`、`v4/models/`、`v4/submission/` |
| E 阶段层 | `v4/E*/src/` | `v4/E*/code/` | `v4/E*/docs/` | `v4/experiments/E*/`、`v4/models/E*/` |
| P 子阶段层 | `v4/E*/P*/code/` | — | `v4/E*/P*/docs/` | `v4/experiments/E*/P*/` |

约定：

- **`v4/` 是提交根**：运行时代码只依赖 `v4/src/`、`v4/E*/src/` 与 manifest 指向的 `models/`、`configs/`；**不依赖 v1/v2/v3 目录**，保证干净目录可独立运行。
- 代码只被当前 P 用 → `P*/code/`；被同层多个 P 用 → `E*/src/`；被多个 E 用 → `v4/src/`。
- 每个候选产物目录 `experiments/E*/P*/<candidate_id>/` 必须含 `result.json`、`result.zip`、`manifest.json`、`cv.json`。
- 每个 P 目录有 `PLAN.md`（最小可执行单元）；每层有 `PLAN.md`；总计划为本文件。
- 资料引用统一登记在 [`资料引用索引.md`](资料引用索引.md)；文档中 `资料库/N` 均指 `../资料/资料库/N`。

---

## 五、模型架构（V4 技术核心）

### 5.1 总体结构：三块拼装

```
                     13 条曲线 + DEPTH（按井，0.1 m）
                                  │
                 ┌────────────────▼─────────────────┐
                 │  A. 数据层 src/data/             │
                 │  去单位行 / 哨兵→NaN / 坏值屏蔽   │
                 │  井内插值(限缺口) / 缺失指示位     │
                 │  分位数裁剪(仅训练折 fit)         │
                 └────────────────┬─────────────────┘
                                  │  x: [B, C=13+1(+K), L]
                 ┌────────────────▼─────────────────┐
                 │  B. 深度序列主干 src/models/     │
                 │  B1: 1D U-Net (dilated depthwise)│
                 │  B2: TCN (因果/非因果空洞卷积)    │
                 │  B3: Patch Transformer (通道独立) │
                 │  输出逐行表示 H: [B, L, d]        │
                 └────────────────┬─────────────────┘
                                  │
                 ┌────────────────▼─────────────────┐
                 │  C. 头 src/models/heads.py       │
                 │  H0 联合常量状态头  → q ∈ (0,1)   │
                 │  H1 POR 头  → p̂  (≈0.1 或连续)   │
                 │  H2 PERM 头 → ẑ = log10(PERM)    │
                 │  H3 SW 双分支 → q̂·99.9+(1-q̂)σ(f)│
                 │  (+ H4 井级分支：井均值/井偏置)   │
                 └────────────────┬─────────────────┘
                                  │
                 ┌────────────────▼─────────────────┐
                 │  D. 解码与后处理 src/inference/  │
                 │  原子门 argmax / 阈值（inner-OOF）│
                 │  多尺度拼接平均 / 集成加权        │
                 │  契约校验（10 井 95,948 行）      │
                 └──────────────────────────────────┘
```

### 5.2 主干候选（按顺序实施，每次只改一处）

| 代号 | 结构 | 感受野 | 关键超参 | 引入阶段 |
|---|---|---|---|---|
| **B1 U-Net-1D** | 4–6 级下采样（stride 2 或 maxpool）+ 同层数上采样 + skip；每级 2 个 `Conv1d(k=5, groups=C)` + GELU + BN | 数百–上千点（≈100 m） | `base_ch=64→256`，`depth=5`，`dropout=0.1` | E3 |
| **B2 TCN** | 空洞因果卷积堆叠，`dilation=2^i`，`i=0..8`，kernel 3，残差块 + weight norm | `2^9≈512` 点 | `channels=128`，`blocks=9` | E3（对照） |
| **B3 PatchTF** | 把深度序列切成 `P=32` 点 patch（stride 16），线性投影成 token，`d=256`，`layers=6`，`heads=8`，**通道独立（CI）** + 相对位置编码 | 全井（chunk 内） | `patch=32`，`d=256`，`L_max=256 tokens` | E4 |
| **B4 MultiScale** | B1 与 B3 并联，输出按可学习门控融合（或直接 concat 送头） | 多尺度 | — | E4 |

**关键设计取舍（有依据，不是口味问题）**：

1. **感受野必须覆盖“层段尺度”而非“点尺度”**：0.1 m 采样下，一个 10 m 的储层段 = 100 点。因此 `2×10^8` 级别的感受野才有意义（B1 的 5 级 = 约 500 点 ≈ 50 m，B2 的 9 块 = 512 点）。E3 的 Gate 之一就是**感受野消融**：把主干深度从 5 减到 3 或把 TCN dilation 上限从 512 减到 64，若分数不降，说明上下文没被用上，必须先修数据/结构再谈扩容。
2. **不用“窗口→中心点”而用“窗口→窗口（seq2seq）”**：一次输出整段，训练吞吐高 5–20×（`资料库/08` §1.2）。逐点滑窗在 730k 点上是算力浪费。
3. **归一化选 BN 而非 LN 作为默认**：BN 的 batch 统计跨井起到正则作用；但**必须验证** LN/GN 在小 batch 下是否更好（E3 消融项）。折内标准化参数只在训练折 fit（`资料库/07` §5）。
4. **不得使用 ImageNet/大规模预训练权重**：测井曲线与自然图像分布无关；预训练只会引入无关先验。若有条件，用**同数据集自监督预训练**（掩码曲线重建，E2 可选 P2）。
5. **工程曲线单独治理**：CAL/DEVI/AZIM/BIT/CASE 在井内近常数（`资料库/08` §0.1-4），把它们同时（a）作为逐点通道送入主干，**且**（b）聚合成井级向量送井级分支 H4。这是“井间泛化信号”的唯一合法来源。

### 5.3 输出头

| 头 | 结构 | 监督信号 | 输出语义 |
|---|---|---|---|
| **H0 联合常量状态头** | `Linear(d→1)` + sigmoid，作用于主干表示的**井内平均池化 + 逐点**拼接 | BCE：`y_ph = 1[POR=0.1 ∧ PERM=0.01 ∧ SW=99.9]` | `q(x)`：该点属于联合占位状态的概率 |
| **H1 POR** | `Linear(d→1)`，输出 `p̂ = 0.1 + softplus(g)`（保证非负且可从 0.1 起步） | 相对误差对齐损失（δ=0.08） | 连续孔隙度 |
| **H2 PERM** | `Linear(d→1)`，输出 `ẑ`（对数域）；`PERM = 10^ẑ`，且 `ẑ` 裁剪到 `[-6, 6]` | 对数域对齐损失 | log10 渗透率 |
| **H3 SW** | 两分支：`q̂ = sigmoid(g₀)`（复用 H0 或独立），`SW = q̂·99.9 + (1-q̂)·sigmoid(f)·100` | 占位 BCE + 相对误差对齐（δ=0.05） | 双峰混合输出 |
| **H4 井级分支** | 主干输出做井级 attention-pool → 井向量 → 预测井级偏置 `Δ_t`（逐目标），以 `p̂_t + λ·Δ_t`（λ 由 inner OOF 选） | 井级均值回归（辅助） | 只在 E8 作为消融/可选增益 |

> **H3 的尺度细节（E0-R2 修正）**：此前假设"SW 占位 99.9（百分数）而有效值 [0,1]（小数）"是**错的**。
> 80 口训练井实测（非缺测且非占位行）：SW **min=8.305 / median=82.805 / max=99.9**，
> SW<1 的行数仅 **11**（占有效行 6.8e-5）→ **SW 与 POR/PERM 一样是单一标签尺度（百分数）**。
> 因此：默认**不做** ×100 换算（`constants.SW_SMALL_BRANCH=False`），SW 头直接输出标签尺度；
> `labels.sw_scale_report` 把实测范围写入数据卡；**仍然严禁把 SW 全局裁剪到 [0,1]**。
> 旧双尺度路径保留为对照开关，默认关闭。

### 5.4 损失（src/losses/score_aligned.py）

严格按 `资料库/12` §2.2–2.4 实现三段式：

```
L = L_align(主) + λ₁ · L_aux(稠密梯度) + λ₂ · L_ph(占位 BCE) + λ₃ · L_phys(可选物理软约束)
```

- `L_align`：`smooth_abs`（Charbonnier，α=1e-3）+ `softplus` 截断（β=20），POR δ=0.08、SW δ=0.05、PERM 在 log10 域 `|ẑ−z|` 上取 `max(0,1−·)`。
- `L_aux`：变换空间的 Smooth L1（POR/SW 用相对误差形式，PERM 用 z 空间），保证早期有稠密梯度；**`λ₁` 从 1.0 退火到 0.1**。
- `L_ph`：H0 的 BCE，`λ₂ = 0.2`。
- `L_phys`（E6 可选）：Archie / Wyllie / Kozeny–Carman 的软约束（`资料库/08` §10.2），`λ₃` 固定 0.05，且**必须消融验证**（`资料库/03` §8.2 提醒 KC 只能定性）。
- **早停与模型选择一律用真实 `score.py` 分数，不用 loss 值**（`资料库/12` §2.3 末）。
- 掩码：缺测行（三目标 = -99999）不参与任何监督；占位行**参与监督**（`rules.md` §5.3 项目决策）。

---

## 六、数据、验证与提交契约（E0 冻结，之后不得修改）

### 6.1 数据解析契约（**E0-R1 修订：按表头名对齐，禁止按列位置解析**）

- 每文件 3 段：第 1 行表头，第 2 行单位（**丢弃**），第 3 行起数据；`utf-8-sig` 读取以吃掉 BOM 与 `\r`。
- **⚠️ E0 实测发现（2026-09-19，本机复算，必须遵守）**：本地 80 口训练井中**有 3 口井的表头与官方 17 列不一致**，共 **27,080 行（3.71%）**：

  | 井（前 8 位） | 列数 | 差异 | 是否缺 `CASE` | 行数 |
  |---|---:|---|---|---:|
  | `42f2870b` | 20 | 多 `K, U, CGR` | 否（含 CASE） | 7,879 |
  | `b7eb1274` | 21 | 多 `TH, K, U, CGR` | 否（含 CASE） | 9,547 |
  | `c7611b01` | 16 | — | **是（唯一）** | 9,654 |

  即 **17,426 行是"多列"，9,654 行是"缺 CASE"**（E0-R2 修正：此前误写为 3 口井都缺 CASE）。
  这 3 口井**三目标（POR/PERM/SW）齐全**，且**都在 `v1_well_folds.json` 的 80 井内**（fold 0/2/3）。
  因此：**必须按表头名建立「列名→列号」映射后逐行重排**；规范表中缺失的列填 `NaN` 并在 `missing_columns` 记录；多余曲线（K/U/TH/CGR）忽略并记录为 `extra_columns`。
  **任何按固定列位置（如 `arr[:, 15:18]`）解析的代码都会错位或静默丢列**（本项目开发过程中已实际触发过一次：numpy 对越界切片静默截断，导致 SW 列被丢掉且不报错）。
- 测试集（10 井）schema 完全一致（14 列规范输入），无畸形井。
- 哨兵：`-99999`、`-9999`、`NaN`，以及任何 `< -1000` 的值 → 缺测（沿用 v2 数据卡的冻结工作假设）。
- 规范列顺序：`DEPTH, GR, PE, SP, CAL, AC, DEN, CNL, RXO, RT, DEVI, AZIM, BIT, CASE | POR, PERM, SW`；
  规范宽度 **17 列**（**13 条曲线** + DEPTH + 3 目标），解析后必须断言 `arr.shape[1] == 17`。
- **列布局常量集中在 `src/data/parse.py`**：`IDX_DEPTH=0`、`IDX_CURVE_START=1`、
  `IDX_CURVE_STOP=14`、目标 = 末 3 列；**布局与 `with_targets` 无关**（测试文件目标列填 NaN），
  并对「目标区间不得与输入区间重叠」加硬断言。
- **⚠️ E0-R2 修正（输入列泄漏事故）**：E0-R1 曾用 `inputs = arr[:, 1:15]`，而索引 14 正是 **POR**，
  导致 80 口训练井 `inputs[:,13] == POR`（逐点相等），且测试井只得到 13 列（提交必崩）。
  已修复为 13 列输入 + 回归测试 `input_no_label_leak`（90 井全过）。
  **教训：禁止按列位置猜测语义，任何切片必须由布局常量推导并配"输入与目标不相交"断言。**
- 训练集：80 井 / **730,268 行**；测试集：10 井 / **95,948 行**（均由 `v4/E0/code/run_all.py` 复算确认）。
- 深度诊断（E0 复算）：重复深度行 0、非递增深度行 0。
- **标签状态三分类**（E0 冻结判据）：
  | 状态 | 判据 | 行数（复算值） | 占比 | 监督方式 |
  |---|---:|---:|---:|---|
  | 缺测 | 三目标同时为哨兵 | 6,700 | 0.917% | 全掩码 |
  | **联合常量占位** | POR=0.1 ∧ PERM=0.01 ∧ SW=99.9 | **487,225** | **66.719%** | 全量参与 |
  | 有效 | 其余（无缺且非占位） | 236,343 | 32.364% | 全量参与 |

  > 说明：`rules.md` §5.3 的「缺测 6,572」采用的是**整行缺测**口径的近似值；本计划以 `run_all.py` 的可复算值为准（缺测 6,700；逐目标缺测 POR 6,701 / PERM 6,711 / SW 6,711；存在 12 行部分缺测）。
- **禁止**：删除占位行；把 SW 全局裁剪到 [0,1]；用井身份（`logId`）作为特征；测试阶段读取任何标签。


### 6.2 特征工程契约（E2）

| 组 | 内容 | 数量 | 阶段 |
|---|---|---|---|
| `F_raw` | 13 条曲线 + DEPTH 原值 | 13 + 1 | E1 |
| `F_miss` | 逐曲线缺失指示位（13）+ DEPTH 缺失位 + 缺失比例 | 13 + 1 + 1 | E1 |
| `F_phys` | 孔隙度类：Wyllie（AC）、密度孔隙度（DEN）、中子（CNL）、`AC-DEN` 交会；泥质：GR 指数、SP 指数；流体：`RT/RXO` 比值、`log10 RT`；骨架：PE；`DEN-CNL` 差 | ~14 | E2 |
| `F_win` | 中心窗口统计（多尺度 mean/std/min/max/trend），窗长 {11, 51, 201} 点（1.1/5.1/20.1 m） | ~14×3×5 | E2 |
| `F_well` | 井级聚合：各曲线井内均值/标准差/分位数、井长、平均采样间隔、井斜均值 | ~20 | E2 |
| `F_depth` | 相对深度、深度分箱、井内归一化深度 | 3 | E1 |

**硬规则**：窗口统计一律**居中窗口**（序列全井可见，离线任务无因果约束，`资料库/07` §6）；`F_win`/`F_well` 的标准化参数只在训练折 fit；特征版本一旦冻结不得在看到 OOF 后加列（改动必须升 `feature_version` 并重跑全部折）。

### 6.3 验证协议

0. **评分分母口径（E0-R1 冻结，已复算确认）**：官方 `N` 为**逐目标排除缺测行后的行数**，即 `missing_mode="drop"`。
   依据：常数基线 (0.1, 0.01, 99.9) 在 `drop` 口径下 = **70.490735**，命中公开锚点 70.4907 ±1e-4；在 `mask`（全行分母）口径下 = 69.843218，低 0.65 分，故官方不是全行分母。
   **所有本地 OOF 数字必须用 `drop` 口径计算**，并在报告中显式标注；`mask` 口径只作为诊断对照保留。
1. **主判据：80 井按井 5 折 GroupKFold**，折分配**逐字节复用** `../v1/src/well_folds.json`
   （sha256 `f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e`，80 井 / 5 折，
   由 `run_all.py` 复算并写入 `versions/folds_sha256.json`），使 v4 的 OOF 与历史锚点**同折可比**。
   所有本地分数标 `selection_score_only=true`。
2. **inner 选择**：每个 outer 折内再切 3 个 inner 折（井维度），一切阈值、λ、早停、集成权重、解码策略**只在 inner-OOF 上选**，outer 折只推理一次。
3. **次级体检**：`../v2/artifacts/E0/` 的 16 井 `folds_confirm` 仅对最终 top-2 候选各跑一次，标注 `v1-exposed`，不作晋级判据。
4. **统计口径**：任何“相对基线”结论必须给出**按井行数加权的配对井级 cluster bootstrap 95% CI**（1000 次重采样），并报告逐折 delta 与逐井非退化比例。
5. **A 榜**：仅用于短名单仲裁（≤4 次/日，留 1 次余量）与崩坏体检；禁止用 A 榜调参。

### 6.4 占位原子保护（D3 下的自包含实现）

因为不做 B0 patch 隔离，占位保护的唯一屏障是模型自身的 H0/H3 头。因此设三条**不可协商**的规则：

1. **解码是硬切换，不是软融合**：`q > τ_t` 时该点输出**精确常量**（POR=0.1 / PERM=0.01 / SW=99.9），否则输出连续分支。禁止在两者之间做线性插值（POR 容差 ±0.008，插值必然出带）。
2. **`τ_t` 只在 inner-OOF 上选**，且必须报告 `atomic_precision` / `atomic_recall` / `atomic_F1` 与“误判代价分解”（把有效行判成常量会立刻丢分，反之亦然）。
3. **Gate 强制上报**：占位行在 OOF 上的**逐目标命中率**（POR/PERM/SW 各自的 Acc），并给出“若全部输出常量”的分数 70.4907 作为下界对照。任何版本只要在占位行上的 Acc 低于 0.98，该目标即 NO-GO。

---

## 七、阶段子计划

执行顺序：**E0 → E1 → E2 ⇒ E3 → E4 → E5 → E6 →（E7 公共件）→ E8 → E9 → E10 → E11**

- [E0 数据、评测与提交契约](E0/PLAN.md) — 数据卡、哨兵、标签状态、评分器复算（70.4907）、按井折、提交契约与 CPU-only 单测。**不训练任何模型。**　**状态：本地契约 Gate 10/10 PASS（`reports/E0_local_contract_gate.json`）；云端 Gate `blocked_pending_cloud_run`（`reports/E0_cloud_gate.json`）。**
- [E1 纯 DL 行级基线](E1/PLAN.md) — **当前阶段（P0/P1 代码待写）**。32 维行级输入 + MLP（无序列上下文），对接对齐损失，建立纯 DL 分母与容量标定；硬 Gate ≥ 78.0。
- [E2 特征工程与数据管线](E2/PLAN.md) — `F_phys`/`F_win`/`F_well` 三组特征、增强策略、按井分片缓存与 16 GiB 内存纪律。
- [E3 深度序列主干](E3/PLAN.md) — 1D U-Net 与 TCN 头对头，含**感受野消融**；硬 Gate ≥ 81.0 且序列主干必须优于同头行级模型。
- [E4 Patch Transformer 与多尺度](E4/PLAN.md) — PatchTST 式通道独立 Transformer；与 CNN 主干的多尺度融合。
- [E5 逐目标精修](E5/PLAN.md) — POR 窄带、PERM log 域、SW 双分支的独立头与专用解码。
- [E6 联合常量状态与原子门](E6/PLAN.md) — H0 状态头 + 学习型原子门 + 精确常量切换；`τ_t` inner-OOF 选择。**PD1 完整管线在此打通**。
- [E7 评分对齐损失与解码](E7/PLAN.md) — 损失三件套的消融与调参、解码后处理、温度/偏置校准。
- [E8 多任务、井级分支与集成](E8/PLAN.md) — MMoE 任务平衡、井级分支（消融）、多 seed/快照/多结构集成、transductive 适配（只作消融）。
- [E9 诚实验证与提交护栏](E9/PLAN.md) — 80 井 OOF 汇总、16 井体检、泄漏终审、A 榜短名单、`choose_submission.py` 护栏。
- [E10 全量重训、打包与提交](E10/PLAN.md) — 全量重训或折集成、ONNX/CPU 推理导出、干净目录一次性复现、提交执行。
- [E11 归档与复盘](E11/PLAN.md) — 资产归档、三口径一致性复盘、下一代方向储备。

---

## 八、Gate、变更与停止规则

### 8.1 统一 Gate 模板（预注册，事后不得改阈值）

所有 P 级 Gate 必须先产出 `v4/reports/<GATE_ID>_gate_prereg.json`，字段：

```json
{
  "gate_id": "E3_gate",
  "primary_metric": "oof_total",          // 或 target_acc / atomic_f1
  "baseline_version": "E1_PD0",
  "baseline_artifact": "experiments/E1/P1/pd0/oof.parquet",
  "baseline_manifest_sha256": "...",
  "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
  "alpha": 0.05,
  "multiplicity": "none",
  "candidate_budget": 1,
  "bootstrap_iters": 1000,
  "pilot_std": null,
  "mde_units": 80,
  "min_detectable_effect": null,
  "mandatory_checks": ["contract_ok", "no_label_leak", "cpu_inference_ok",
                        "training_time_log_valid", "checkpoint_resumable",
                        "atomic_precision_reported"],
  "decisions_locked": ["architecture", "loss_weights", "feature_version"],
  "notes": ""
}
```

规则：

- 多候选必须做 Holm/Bonferroni/FDR 校正；`multiplicity=none` 时 `candidate_budget` 必须为 1。
- `effective_threshold = max(min_delta, min_effect_floor)`；`min_effect_floor` 不允许用 `allow_underpowered` 跳过。
- **所有模型 Gate 的 `mandatory_checks` 必须含 `atomic_precision_reported`（占位行逐目标命中率）与 `contract_ok`。**
- 预注册后只能升修订号（`E3_gate_r2`），不得原地改。

### 8.2 阶段 Gate 硬门槛

| Gate | 硬门槛 | 附加条件 |
|---|---|---|
| E0 | **本地契约 Gate**：数据卡可复算、常数基线 70.4907（±1e-4）、折指纹一致、契约单测、`missing_mode_is_drop`、`score_total_consistent`、`input_no_label_leak`、`target_scale_reported`；**云端 Gate**：`env_hard_checks_passed` + `disk_budget_ok`（实测 ≥8 GB） | 干净目录下 `python predict.py --help` 可运行 |
| E1 | OOF Total **≥ 78.0**；5 折方向一致 | 占位行逐目标 Acc ≥ 0.98；loss 曲线无 NaN |
| E2 | 特征组必须**逐组消融**证明增量 | 特征来源表登记每个派生列的公式；无标签派生列 |
| E3 | OOF Total **≥ 81.0**；**序列主干 > 同头行级模型**（同折同头对比）；感受野消融显示上下文被利用 | 5 折全正；加权配对 bootstrap 95% CI 下界 > 0 |
| E4 | 多尺度 ≥ 单尺度最好者，或明确 NO-GO 并保留 E3 结构 | 参数量/显存报告 |
| E5 | 逐目标：至少一个目标的连续切片 Acc 显著提升 | 不得以其他目标退化换取 |
| E6 | 完整 PD1 管线 OOF **≥ 82.0**；占位行 Acc ≥ 0.99；契约通过 | 原子 precision/recall 双报 |
| E7 | 对齐损失 ≥ 纯 aux 损失（同结构对照） | 三段式权重的消融表 |
| E8 | 集成 ≥ 最佳单成员，且 CI 下界 > 0 | 成员同源性报告（防“同源平均造假增益”） |
| E9 | 80 井 OOF ≥ 82.0 且护栏通过（`choose_submission.py`） | 16 井体检不得崩坏（掉 >1.5 分即触发复核） |
| E10 | 干净目录两次运行结果一致；10 井/95,948 行；CPU 单次 < 30 min | README 两条命令可执行 |

### 8.3 停止规则

- 同一方向连续 **3 个候选**的 delta bootstrap CI 上界 ≤ 0 → 该方向停止，转下一方向。
- E3 若序列主干不优于行级模型 → 先做诊断（数据/感受野/损失），**最多两次结构修订**；仍不优则降级为“行级模型 + 手工窗口特征”并记录 NO-GO 证据。
- 任何训练损失出现 NaN/Inf 连续两次 → 回退上一 checkpoint 并降低学习率/开启梯度裁剪。
- 单个候选的 OOF 提升 < 0.02 且 CI 含 0 → 不计入“有效候选”，不写入候选台账的有效条目。
- 本地资源（云端机器）不可用时，所有依赖该项的 Gate 冻结为 `blocked`，不得用未验证结论替代。

### 8.4 变更规则

| 变更类型 | 规则 |
|---|---|
| 数据/哨兵/折/评分器 | E0 冻结后不得更改；只有官方规则或数据变更才允许 `E0-R2` |
| 特征 | 先定义再实验；看 OOF 后不得加列；变更升 `feature_version` 并重跑全部折与下游 |
| 架构 | 每阶段只允许改一处，且必须有对照实验；架构变更必须新建 candidate_id |
| 阈值/权重/解码 | 只在 **inner-OOF** 上选；outer 折只推理一次 |
| 候选 | 一旦提交 A 榜不得覆盖；修改必须新建 `candidate_id` 并记 `parent` |
| 随机种子 | 必须固定并写入 manifest；多 seed 视为集成成员而非调参手段 |

---

## 九、提交、复现与回退

### 9.1 候选注册表

`v4/versions/candidates.json` 是**唯一事实源**：未登记的候选不得提交。

```json
{
  "candidate_id": "v4-e6-pd1-unet",
  "stage": "E6",
  "arch": "unet1d-b5-multitask",
  "loss": "aligned3stage-lambda10to01-02",
  "feature_version": "F3",
  "created_at": "YYYY-MM-DDTHH:MM:SS",
  "checkpoint": "models/v4/E6/pd1_unet_foldall.pt",
  "result_zip": "experiments/E6/P2/pd1/result.zip",
  "result_zip_sha256": "...",
  "cv": {"total": 0.0, "por": 0.0, "perm": 0.0, "sw": 0.0,
          "fold_deltas": [], "paired_bootstrap_ci": [0.0, 0.0]},
  "atomic": {"por_acc": 0.0, "perm_acc": 0.0, "sw_acc": 0.0,
              "tau": [0.0, 0.0, 0.0]},
  "a_board_score": null,
  "status": "local_only"
}
```
`status ∈ {local_only, shortlisted, submitted, frozen_best, rejected}`。

### 9.2 提交包结构（`submission_code_v4.zip`）

```
submission_code_v4/
├── README.md              # 算法名称/方法/环境/两条命令/文件说明（rules §6.3）
├── predict.py             # 主入口，官方 --data_dir / --output
├── train.py               # 训练入口（可选执行，但必须能跑）
├── requirements.txt       # 环境声明（预装 torch 2.4.0 + 额外轻量包，见 §3.3.1）
├── configs/v4.yaml        # 模型/特征/解码配置
├── src/                   # data/ features/ models/ losses/ inference/ validation/
├── models/                # 训练好的权重 + scaler/分位数参数（JSON）
└── examples/result_example.json
```

硬要求：
1. `predict.py` **只读** `--data_dir` 下的文件，不访问网络、不训练、不写除 `--output` 外的路径（除 `--temp` 显式指定）。
2. 权重是 **CPU 可加载** 的（`map_location='cpu'`），且**优先导出 ONNX** 作为兜底路径（防评测机 torch 版本不匹配）。
3. 单次运行内存峰值 < 8 GiB（16 GiB 机器的安全余量），时间 < 30 min，且**确定性**（固定 seed + `torch.use_deterministic_algorithms` 或明确的浮点容差声明）。
4. `requirements.txt` 声明"预装基础栈（python 3.11 / torch 2.4.0+cu124）+ 额外轻量包（pandas/pyarrow/scipy/sklearn/einops/onnx/onnxruntime）"的确切版本；`versions/locks/cloud.txt`（训练侧）与 `submit.txt`（推理侧）分离，后者不含任何训练专属依赖。

### 9.3 复现门禁（rules §8，最高优先级）

任何 `status` 升到 `frozen_best` 的候选必须：

1. 在**只含 `submission_code_v4.zip` 与 `data/` 的干净目录**中解压并以官方命令生成结果：
   `python predict.py --data_dir ./data --output result.json`
2. 结果与提交的 `result.json` 逐点一致（浮点容差 ≤1e-6；报告最大逐点差与超差行数）；
3. 记录 `reports/E10_reproduce_report.json`：干净目录路径、命令、stdout 摘要、Python/依赖版本、sha256、最大逐点差、耗时、峰值内存。
4. **两次独立运行**结果 sha256 必须一致，否则不得提交。

### 9.4 提交护栏（`v4/E9/code/choose_submission.py`）

即使候选通过全部本地 Gate，提交前仍必须运行护栏：

```
guardrail_floor = max(75.0, B0_LOCAL_OOF − 1.0, B0_A_BOARD − 0.5)
其中 B0_LOCAL_OOF = 80.382479（v1 E7 本地 OOF，protocol_matched=false，仅作外部参照）
     B0_A_BOARD  = 82.2757
若候选 OOF < guardrail_floor 或 A 榜已知且 < B0_A_BOARD − 0.5
   → 不提交该候选，走回退协议（§9.5）
```
输出 `reports/E10_submission_decision.json`（`choice/reason/candidate_oof/guardrail_floor/reference_oof/protocol_matched/`）。

### 9.5 回退协议（v4 必须有，且必须是自包含的）

v4 可能整体失败（纯 DL 在 80 井上不收敛优于树模型）。因此：

- **E10/P1 预先构建 `submission_code_b0_fallback.zip`**：从冻结的 `../v1/submission_e7` 源码构建官方 `--data_dir` 兼容、自包含的回退包（v1 原包只接受 `--data-dir`），并在只含 zip + data 的干净目录中用**包内 v1 requirements 的临时 venv**跑通，确认结果与冻结 `result.json` 字节一致后写 `inference_verified=true`。
- 该包**现在就构建**（不等到失败时），因为它与 v4 开发完全解耦，且是“保底晋级”的唯一保险。
- 回退不代表 v4 作废：报告须明确写“v4 提交 B0 fallback，原因是 X”。

---

## 十、风险登记表

| 风险 | 早期信号 | 预案 |
|---|---|---|
| **占位行被网络学坏** | 占位行 Acc < 0.98，OOF 卡在 70 出头 | H0/H3 状态头 + 硬切换解码；`λ₂` 提高；必要时对占位行做**过采样** |
| **SW 尺度混淆（99.9 vs [0,1]）** | SW Acc 掉 ~10 分 | E0 单元测试锁死；双分支混合输出；契约禁止裁剪 SW |
| **POR 容差过窄（±0.008）** | POR Acc 在 0.6 附近打转 | POR 头从 0.1 起步（`0.1+softplus`）；占位行硬输出 0.1 |
| **16 GiB 系统内存 OOM** | DataLoader 被杀、swap 抖动、训练中途 OOM-kill | `num_workers=4`、禁止全局井缓存、按井分片按需读、每 worker < 300 MB 自检 |
| **30 GB 磁盘打爆** | `pip install` 后剩余 < 5 GB；checkpoint 越攒越多 | 装完 `pip cache purge`；特征按版本目录落盘 + 旧版本即删；checkpoint 滚动窗口（best/last/last_prev）；每 epoch `assert_disk_headroom(8.0)`；E0 Gate 强制 `disk_budget_ok` |
| **PyTorch 2.4 API 不兼容** | 启动即 `AttributeError`/`ImportError` | `check_env.py` 前置断言；禁用 2.5+ API 与需编译扩展（flash-attn/xformers/apex）；注意力统一走 `F.scaled_dot_product_attention` |
| **序列主干不优于行级模型** | E3 delta CI 含 0 或为负 | 感受野消融 → 检查窗口统计是否泄漏/是否被 batch 内打乱；最多两次结构修订后降级 |
| **过拟合（80 井太少）** | inner 高、outer 低；折间方差大 | dropout/stochastic depth/weight decay；曲线随机掩码与深度抖动增强；早停用真实分数 |
| **长尾 PERM 崩塌** | PERM Acc < 0.85 | log10 域 + 分位数损失辅助；输出裁剪到 [-6,6] |
| **训练不稳定（NaN）** | loss 变 NaN | 梯度裁剪 1.0；bf16 而非 fp16；`softplus` 的 `β` 不超过 30 |
| **评测机无 GPU / torch 版本不符** | 干净目录报错 | ONNX 导出兜底路径 + `map_location='cpu'` + 纯 numpy 后处理 |
| **复现失败** | 两次运行 sha256 不一致 | 固定 seed + 确定性算子；不确定时在 README 声明容差并给出逐点最大差 |
| **本地 CV 虚高** | A 榜远低于本地 | 折维度恒定为井；一切 fit 只用训练折；特征来源表审计 |
| **A 榜过拟合** | A 高本地低、换候选就掉 | A 榜只做仲裁；每日额度预算制 |
| **整体失败（不优于树模型）** | E6 Gate < 82.0 | 走 §9.5 回退协议，保底晋级 |

---

## 十一、时间盒与 A 榜配额

| 阶段 | 时间盒 | 计算预算（软） | A 榜配额 | 到期动作 |
|---|---|---|---|---|
| E0 | 0.5 天 | 本机 CPU，分钟级 | 0 | 契约不过不得进 E1 |
| E1 | 1 天 | A100，≤3 h | 0 | OOF < 78.0 → 排查数据/损失，不扩容 |
| E2 | 1 天 | 本机 CPU + A100，≤2 h | 0 | 每组特征必须消融 |
| E3 | 3 天 | A100，≤20 h | 0 | 序列不优于行级 → 诊断，最多两次修订 |
| E4 | 2 天 | A100，≤20 h | 0 | 不优则保留 E3 结构 |
| E5 | 2 天 | A100，≤15 h | 0 | 每目标独立判定 |
| E6 | 2 天 | A100，≤10 h | 0 | OOF < 82.0 → 冻结当前最强候选为 PD-pre |
| E7 | 1.5 天 | A100，≤10 h | 0 | 损失消融必须完整 |
| E8 | 3 天 | A100，≤40 h | 0 | 同源平均不得当增益 |
| E9 | 1 天 | A100 + 本机，≤4 h | **2–3（短名单仲裁）** | 护栏不过 → 回退 |
| E10 | 1 天 | 本机 CPU 打包 + A100 全量重训 ≤30 h | **1（最终确认）** | 复现失败 → 回退 |
| E11 | 0.5 天 | 本机 | 0 | 归档 |

- A 榜每日 ≤5 次；每次提交登记 `v4/reports/<STAGE>_a_board_log.json`（`candidate_id/score/submit_time/用途`）。
- 总时间盒约 19.5 天；E4/E7/E8 可在余量不足时压缩。

---

## 十二、预期提升（仅记录，不作为设计依据）

| 情景 | 条件 | 本地 OOF Total | A/B 榜估计 |
|---|---|---:|---:|
| 保底 | 序列主干未超行级模型 | ≥81.0（纯 DL 行级 + 特征） | 接近但可能略低于 B0 → 走回退 |
| P50 | U-Net 主干 + 状态头 + 对齐损失打通 | 82.0–82.6 | 82.0–82.6 |
| P75 | 多尺度 + 集成 + 井级分支有效 | 82.6–83.2 | 82.5–83.2 |
| 冲刺 | PatchTF + 对齐损失 + 集成同时正增益 | ≥83.2 | >83.5，不保证 |

> 参照系：v1 E7（树 + 手工特征）= 本地 80.3825 / A 榜 82.2757；v2 从零树管线 = dev64 80.3052；v2 E13.5 激进场最佳 A 榜 = 82.3035。**v4 的核心赌注是：序列主干提供的深度上下文 + 与评分同构的损失，能突破树模型 + 手工特征的天花板。**

---

## 十三、为什么这样设计（设计依据索引）

1. **评分是优化目标本身**：`rules.md` §7.3–7.4；`资料库/12` §1（逐条解析）、§2（可微损失设计）→ §5.4 三段式损失。
2. **标签是混合分布（66.7% 占位 + SW 双峰）**：`rules.md` §5.3、`资料库/12` §3.1/§3.4 → §5.3 的 H0/H3 头。
3. **PERM 必须 log 域**：`资料库/12` §2.4 提示 3、`资料库/05` §2.3 → H2 输出对数域。
4. **井级验证是红线**：`资料库/07` §8、`资料库/12` §3.5 → §6.3 折协议。
5. **深度序列模型是第 2 层**：`资料库/08` §0.3 → §5.2 主干候选。
6. **多任务优于独立模型但需平衡**：`资料库/08` §1.4 → MMoE（E8）。
7. **物理先验是约束不是万能解**：`资料库/01`/`02`（公式）、`资料库/03` §8.2（KC 只能定性）、`资料库/13` §5.5 → `L_phys` 为可选且必须消融。
8. **井级/地质信息大多不可直接获得**：`资料库/04` §9 → 只用可观测的井级聚合与工程曲线。
9. **复现是比赛硬要求**：`rules.md` §6/§8、`资料库/12` §5 → §9.2/§9.3。
10. **前代负知识资产**：v1 E8–E11、v2 E6/E7 的 NO-GO 只在“无新信息源”下有效；v4 的 GPU 与新架构即“新的可观测条件”，因此允许重开序列模型方向，但**仍以消融与预注册 Gate 约束**。
