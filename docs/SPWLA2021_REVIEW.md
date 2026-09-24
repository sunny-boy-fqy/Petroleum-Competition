# 参考复盘：SPWLA 2021 PDDA 测井竞赛（Volve）对 v4 的启示

> **文档导航**：[v4 文档中心](README.md) · [v4 README](../README.md) · [总计划](../PLAN.md) · [代码审查状态](CODE_REVIEW_STATUS.md)
> **文档类型**：外部方案复盘。只参考方法与工程思路；不复制外部数据/代码/阈值。


> 参考仓库：`Machine-Learning-Competition-2021`（作者已部署到本机）。
> 参考论文：Fu et al., *Well-Log-Based Reservoir Property Estimation With Machine Learning:
> A Contest Summary*, Petrophysics 65(01), 2024。
> 任务：用常规测井曲线（GR/DEN/NEU/RDEP/DTC 等）预测 VSH / PHIF / SW，评价指标 RMSE。
> 我们：用 13 条曲线 + DEPTH 预测 POR / PERM / SW，官方指标是逐目标相对准确率（含 66.7% 原子占位行）。

---

## 1. 前五名方案一览

| 名次 | 队伍 | 关键方法 | 模型 |
|---|---|---|---|
| 1 | **UTFE** | **类型井选择（KL / DTW）+ 井间自适应 + 分区 + 半监督插值** | 线性回归 / KNN（极简） |
| 2 | **MoLPhy** | **MICE(LGBM) 插值 + KS 代表采样 + DEN/NEU 聚类标志 + 测试输入分布匹配 + 序列目标预测** | ExtraTrees/XGB/LGBM + ExtraTrees meta 的 SuperLearner |
| 3 | **Tomsk** | 对数电阻率、IsolationForest 异常剔除、Robust/MinMax 缩放、FFNN（2 隐层 3 输出）+ **后处理规则** | 小型前馈神经网络 |
| 4 | **Atwah_Analytics** | **大量岩石物理特征工程**（Archie/Simandoux/Indonesia、密度孔隙度多骨架、Vsh 多公式、Klogh）+ Boruta 特征选择 + 序列预测 | ExtraTreesRegressor + CatBoost |
| 5 | **Jaehyuk_Lee** | 缺失/异常处理 + 相关性/特征分析 + 网格搜索 | LightGBM / XGBoost |

**最重要的事实：冠军用的是一元线性回归 + KNN，而不是深度网络。** 论文的讨论结论也很直接：

> “模型本身的选择可能不是成功的关键；更应关注**选择与测试井相似的训练井**、异常值处理、数据质量控制和数据准备。”

---

## 2. 可直接迁移到 v4 的高价值方法

### 2.1 类型井选择 + 井间自适应（冠军方案，最高价值）
- UTFE 对每口测试井，用 **KL 散度**（相似地层）或 **归一化 DTW 距离**（不同深度段）从训练井中选“类型井”；
- 只在类型井上训练，或按类型井做线性/分位映射，降低井间非平稳性；
- VSH 用沙/泥基线：`VSH=(GR - sand)/(shale - sand)`，按 DTW 分区；
- PHIF 用类型井上的密度-孔隙度线性关系；
- SW 用类型井 + KNN。

**对 v4 的启示**：
- 我们现在是“80 口井混在一起训练 + 全局 scaler”；这在井间环境差异大时会传播偏差。
- 应新增 **每口测试井的 type-well 选择**，用类型井做：折内 scaler 参考、输入分布匹配、或单独训练一个“类型井适配”成员。
- 这比继续加深序列主干更可能带来稳定增益，且成本低。

### 2.2 缺失值插补（第 2/4/5 名都做了）
- MoLPhy：**MICE + LGBMRegressor** 逐列迭代插补；
- UTFE：KNNImputer / 相邻已解释段半监督插值；
- Atwah：先用 KNN/迭代插补再用 XGBoost 继续插补；
- Tomsk：缺失行处理 + 后向填充。

**对 v4 的启示**：
- 我们目前 `RowScaler` 用中位数填 NaN + missing indicator；对 13 条曲线缺失较多的行，可能损坏物理特征。
- 应新增 `knn / iterative(MICE) / backfill` 插补选项，并用 inner-OOF 消融决定是否采用。
- 插补必须在**训练折内 fit**，测试/验证只 transform（避免泄漏）。

### 2.3 KS（Kennard-Stone）代表采样 / 数据覆盖
- MoLPhy 用 Kennard-Stone 选训练/验证子集，让验证集覆盖特征空间极端点，而不是随机划分；
- 他们认为“数据 > 模型”，验证集代表性很重要。

**对 v4 的启示**：
- 我们的 inner split 目前是按井随机；可改为 **按井签名做 KS 采样** 选 inner-val，让 inner-OOF 更能代表外折/测试分布；
- outer 折仍必须冻结、按井互斥（不能破坏比赛协议）。

### 2.4 测试输入分布匹配（MoLPhy）
- 发现 train/test 特征分布不匹配后，用**线性变换（乘子+平移）** 把测试输入分布拉到训练分布，RMSE 明显下降。

**对 v4 的启示**：
- 我们已有 `E8/pseudo_label` 的输出分位对齐，但没有系统的**输入特征分布匹配**；
- 可新增 `linear histogram matching` / `quantile matching`，在 inner-OOF 上验证是否提升；
- 必须保留“只用测试输入、无测试标签”的硬审计。

### 2.5 序列/链式目标预测（MoLPhy、Atwah）
- MoLPhy：VSH → PHIF → SW 顺序预测，把前一个预测追加为下一个目标的输入；
- Atwah：ExtraTrees(VSH) → CatBoost(PHIF) → CatBoost(SW)，逐步追加。
- 论文指出这会传播误差，但“目标间相关性带来的信息”通常值得。

**对 v4 的启示**：
- 我们的 POR/PERM/SW 强相关；可新增 **链式模型**：POR → PERM → SW（或 PERM 单独）。
- 关键是使用 **out-of-fold 的前序预测** 作为后续目标特征，避免标签泄漏。

### 2.6 岩石物理特征工程（Atwah，最丰富）
Atwah 第 4 名用了大量领域公式：
- 密度孔隙度：`φD=(ρma-DEN)/(ρma-ρf)`，对石英/方解石/白云石不同骨架；
- 中子-密度组合：`(φD+φN)/2`、气层校正；
- Vsh：GR 线性、Larionov(old/tertiary)、Steiber、Clavier；
- Sw：Archie、Simandoux、Indonesia；
- Klogh（按地层经验系数）；
- 对数电阻率、电阻率比、交集特征等。

**对 v4 的启示**：
- 我们 `src/features/physics.py` 已实现部分（Wyllie、密度、中子、IGR、Vsh、电阻率比等），但缺少：
  - Sw 的 Archie/Simandoux/Indonesia 系列；
  - 多骨架密度孔隙度；
  - Larionov/Steiber/Clavier Vsh；
  - Klogh；
  - 这些特征应作为 `phys` 组的扩展或新增 `petro` 特征组。
- 需要 fold 内拟合 `Rw/Rsh` 等参数，或先给保守默认值 + 让模型学习修正。

### 2.7 Stacking / SuperLearner（第 2/4/5 名）
- MoLPhy：ExtraTrees + XGB + LGBM + Linear/KNN/Bagging/DT → ExtraTrees meta；
- Atwah：ExtraTrees + CatBoost；
- Jaehyuk：LightGBM / XGBoost + GridSearchCV。

**对 v4 的启示**：
- v4 目前是“纯 DL 单主干”，但榜单说明 **树模型 + 特征工程 + 数据适配** 极强；
- 我们已有 WP6 `src/ensemble/stacking.py`（Ridge/Logistic/HistGB），应新增 **GBDT 一阶成员**，并与 DL 成员做 stacking/集成；
- 若规则允许，树模型可以成为最终集成的重要成员，至少作为 OOF 对照。

### 2.8 异常值处理与后处理规则（第 3/4/5 名）
- Tomsk：IsolationForest 去异常行；后处理规则把 RMSE 从 0.069 降到 0.0634；
- 后处理包括：DEN>训练上界 → POR=0.02；GR>训练上界 → VSH=1；RDEP_log<-2 → SW=1；某口井整体 SW=1 等。

**对 v4 的启示**：
- 我们的官方指标也允许对训练分布上下界做**保守裁剪/外推**；
- 可新增 `src/inference/postprocess.py`：折内分位数裁剪、物理区间裁剪、井级一致性修正；
- 所有规则必须用 inner-OOF 验证有正增益才启用，禁止直接套用 SPWLA 的阈值。

### 2.9 模型选择不是关键（论文结论）
论文明确：“model choice might not be the key”；应重点投入：
1. 选与测试井相似的训练井；
2. 异常值/缺失值处理；
3. 输入质量控制和数据准备；
4. 多模型/多井分区建模。

**对 v4 的启示**：
- v4 之前把“深度序列主干”作为核心赌注；参考项目提示应把预算重新分配到 **数据适配 + 特征工程 + 树/线性强基线 + 集成**；
- 序列主干不必放弃，但应从“唯一主线”降级为“多样性成员”，并用 paired CI 决定是否保留。

---

## 3. v4 新增工作包（WP8–WP11）

| WP | 名称 | 核心内容 | 预期收益 | 依赖 |
|---|---|---|---|---|
| **WP8** | 数据适配与类型井 | KL/DTW 选类型井；井间输入直方图/分位匹配；折内适配 | 高 | WP0 |
| **WP9** | 缺失/异常/代表采样 | KNN/MICE 插补；IsolationForest/IQR 异常权重；KS 按井代表 inner split | 高 | WP0 |
| **WP10** | 岩石物理特征扩展 | Archie/Simandoux/Indonesia Sw；多骨架 φD；Larionov/Steiber/Clavier Vsh；Klogh | 中–高 | WP0 |
| **WP11** | 链式目标 + GBDT/Stacking | POR→PERM→SW OOF 链式；GBDT（HistGB/LGBM/XGB）一阶成员；与 DL 成员 stacking | 高 | WP3/WP6 |

**执行顺序建议**（在 WP0–WP3 基础上）：
WP8 → WP9 → WP10 → WP11，并用自己的 OOF/paired CI 与 v4 现有成员比较；
若某 WP 未通过 CI，只保留为多样性成员，不进入最终提交。

---

## 4. 与 v4 的关键差异与红线

| 维度 | SPWLA 2021 | v4 当前 |
|---|---|---|
| 目标 | VSH/PHIF/SW（0–1） | POR/PERM/SW（含原子占位 0.1/0.01/99.9） |
| 指标 | 平均 RMSE | 逐目标相对准确率 + 占位精确命中 |
| 标签 | 连续、无占位 | **66.7% 联合占位行**，官方给精确命中满分 |
| 测试井 | 有隐藏测试井 | 10 口测试井，输入分布可观测 |
| 建模 | 数据适配 + 树/线性/小 NN | 纯 DL 单主干 + 原子门控 |
| CV | 随机/KS | 按井 5 折冻结（必须保持） |

**红线不变**：
- 不允许使用测试标签；
- 类型井/分布匹配只允许用测试**输入**；
- outer 折只推理一次；
- 所有新方法必须在 inner-OOF + confirm 折上有正增益。

---

## 5. 立刻要做的三件事（最高 ROI）

1. **WP8 类型井 + 输入分布匹配**：这是冠军方案的核心，且我们完全没有。
2. **WP9 MICE/KNN 插补 + 异常权重**：参考项目反复强调数据质量 > 模型。
3. **WP11 GBDT + Stacking + 链式目标**：用现成强模型补齐 v4 的“纯 DL 赌注”风险。

> 注意：v4 的 66.7% 占位行是 SPWLA 没有的特殊结构；WP1 的原子校准/期望分数决策仍然必须优先，不能用 SPWLA 方法替代。

## 6. 参考边界、贡献与引用

### 6.1 我们参考了什么

| 参考内容 | 对应 v4 WP | 落点 |
|---|---|---|
| 类型井选择：KL 散度 / 归一化 DTW，为每口测试井找相似训练井 | WP8 | `src/data/type_well.py`、`E8/code/type_well_report.py` |
| 井间自适应：训练/测试井之间的分布对齐、线性/分位匹配 | WP8 | `src/data/well_adapt.py` |
| 缺失值插补：KNN / MICE(LGBM) / 相邻已解释段半监督 | WP9 | `src/data/impute.py` |
| Kennard-Stone 代表采样：让验证集覆盖特征空间 | WP9 | `src/validation/representative.py` |
| 异常值处理：IsolationForest / 稳健统计 / 样本降权 | WP9 | `src/data/outliers.py` |
| 测试输入分布匹配（MoLPhy） | WP8/WP5 | `well_adapt`、`self_training` |
| 链式目标预测：VSH→PHIF→SW，前序预测追加为特征 | WP11 | `src/training/chained.py` |
| 岩石物理特征：Archie/Simandoux/Indonesia、多骨架 φD、Vsh 多公式、Klogh | WP10 | `src/features/physics_ext.py` |
| 树/GBDT 与 SuperLearner stacking | WP6/WP11 | `src/ensemble/stacking.py`、`src/models/gbdt.py` |
| 后处理规则思想（边界/异常区间用领域规则修正） | 规划中 | `docs/SCORE_MAX_PLAN.md` §WP9/后续 |

### 6.2 我们没有参考/没有复制什么

- 未复制 Volve 数据、标签、参赛 notebook 的具体代码、模型权重或提交文件；
- 未套用参考项目的固定阈值（DEN>3、GR 上界、RDEP_log<-2 等）——我们的数据与评分不同；
- 未使用参考项目的 RMSE 指标或随机划分；v4 仍保持自己的按井 5 折冻结协议；
- 未使用任何测试标签；类型井/分布匹配只使用测试**输入**。

### 6.3 参考项目本身的贡献

SPWLA 2021 竞赛及公开方案的主要贡献：

1. 首次系统梳理了“多井测井解释”的机器学习竞赛基准与公开数据（Volve）；
2. 证明在井间非平稳条件下，**类型井选择与数据适配**往往比模型复杂度更有效；
3. 公开了前五名完整方案，覆盖数据清洗、插补、代表采样、特征工程、树集成、小 NN 与后处理；
4. 赛后论文给出了可复用的结论：模型选择不是成功的关键，训练井相似性、异常值处理与数据质量才是；
5. 为后续测井 ML 工作提供了可复现的公开代码与数据（CC BY-NC-SA）。

### 6.4 引用

```bibtex
@article{fu2024well,
  title={Well-Log-Based Reservoir Property Estimation With Machine Learning: A Contest Summary},
  author={Fu, Lei and Yu, Yanxiang and Xu, Chicheng and Ashby, Michael and McDonald, Andrew and Pan, Wen and Deng, Tianqi and Szab{\'o}, Istv{\'a}n and Hanzelik, P{\'a}l P and Kalm{\'a}r, Csilla and others},
  journal={Petrophysics}, volume={65}, number={01}, pages={108--127}, year={2024},
  publisher={SPWLA}
}
```

参考仓库：<https://github.com/pddasig/Machine-Learning-Competition-2021>

### 6.5 WP8 适配成员落地

- 类型井报告：`E8/code/type_well_report.py`；
- **类型井 + 井间自适应一阶成员**：`E8/code/type_well_member.py`；
- 回归：`tests/test_wp_reports.py::test_type_well_member_gbdt`、`tests/test_start_sh.py`；
- 启动：`start.sh --wp type-well`（报告）与 `start.sh --wp type-well-adapt`（成员）。
