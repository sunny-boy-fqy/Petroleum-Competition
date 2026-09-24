# v4 文档中心

> **用途**：把 v4 的说明文档按“先读什么、遇到问题查哪里、事实以谁为准”重新组织。
> 本文件只做导航与阅读顺序；具体内容仍以各专项文档、代码、报告和 `versions/status.json` 为准。
>
> **为什么没有把文件挪到子目录**：大量文档路径被 `PLAN.md`、阶段计划、测试和脚本引用。
> 本次重构保持路径稳定，通过本索引、README 重排和统一页眉解决“混乱”问题，避免制造坏链。

> **交付纪律（每次任务完成后必须执行）**：更新文档 → `git add` → `git commit` →
> `git push origin HEAD:master HEAD:main` → **清除 `/mnt/d/tmp/Petroleum-Competition/` 下的旧 zip** →
> `python3 tools/pack_code_zip.py` 生成最新完整代码包。
> **旧 zip 不清理视为任务未完成；交付目录只保留本次最新 zip。**

---

## 0. 按角色/任务的阅读路径

| 你的目标 | 建议顺序 |
|---|---|
| 第一次了解 v4 | [`../README.md`](../README.md) → 本文 §1 → [`../PLAN.md`](../PLAN.md) §一–§六 |
| 准备云端训练 | [`platform_setup.md`](platform_setup.md) → [`training_tasks.md`](training_tasks.md) → [`image_requirements.md`](image_requirements.md) → [`dependencies.md`](dependencies.md) |
| 改模型/训练策略 | [`../PLAN.md`](../PLAN.md) §五–§七 → [`SCORE_MAX_PLAN.md`](SCORE_MAX_PLAN.md) → [`SPWLA2021_REVIEW.md`](SPWLA2021_REVIEW.md) |
| 查数据/评分口径 | [`../E0/docs/data_card.md`](../E0/docs/data_card.md) → [`../PLAN.md`](../PLAN.md) §六 → `../src/score.py` |
| 查 Gate/预注册 | [`gate_template.md`](gate_template.md) → [`../PLAN.md`](../PLAN.md) §八 → [`../versions/prereg_templates/`](../versions/prereg_templates/) |
| 查文件/目录 | [`PROJECT_FILES.md`](PROJECT_FILES.md) → `../configs/paths.yaml` |
| 提交前复核 | [`CODE_REVIEW_STATUS.md`](CODE_REVIEW_STATUS.md) → [`gate_template.md`](gate_template.md) → [`PROJECT_FILES.md`](PROJECT_FILES.md) |
| 写文档/报告 | 本文 §7、§8 → [`../PLAN.md`](../PLAN.md) §八 |

---

## 1. 事实源层级

发生冲突时，按以下优先级判断：

1. **官方规则 / 数据契约**：`../../rules.md`（v4 仓库外，项目工作区根目录）。
2. **代码与数据**：`../src/`、`../E*/code/`、`../configs/`、`../data/`（运行时）。
3. **执行状态与证据**：`../versions/status.json`、`../versions/candidates.json`、
   `../reports/*.json`、`../versions/registry.json`。
4. **总计划**：[`../PLAN.md`](../PLAN.md)。
5. **专项说明文档**：本目录各文件。
6. **阶段/P 级计划**：`../E*/PLAN.md`、`../E*/P*/PLAN.md`；多为生成物或执行快照。
7. **历史报告/研究资料**：`../reports/*REVIEW*.md`、`../../资料/`。

> **写文档时的硬规则**：不要把尚未运行的数字写成“已达成”；不要把“代码已实现”写成“Gate 已通过”；
> 不要把历史版本（v1/v2/v3）的成绩混入 v4 目标。

---

## 2. 项目级与仓库边界

| 范围 | 文档 | 说明 |
|---|---|---|
| 项目工作区 | `../../README.md`、`../../rules.md`、`../../docs/` | v4 仓库外；解释多版本布局、官方规则、项目级代码审查 |
| v4 仓库 | [`../README.md`](../README.md)、本文 | 当前活跃交付源入口 |
| 总计划 | [`../PLAN.md`](../PLAN.md) | 唯一权威技术计划 |

> v4 是独立 git 仓库根；提交 v4 时 `../../docs/CODE_REVIEW.md` 等根文档**不会自动进入 v4 仓库**。
> 因此 v4 的关键审查结论另见 [`CODE_REVIEW_STATUS.md`](CODE_REVIEW_STATUS.md)。

---

## 3. 入门与总览

| 文档 | 类型 | 说明 |
|---|---|---|
| [`../README.md`](../README.md) | 入口 | 快速开始、当前状态、文档地图 |
| [`../PLAN.md`](../PLAN.md) | 权威计划 | 架构、数据契约、Gate、预算、风险 |
| [`PROJECT_FILES.md`](PROJECT_FILES.md) | 目录索引 | 每个文件的用途、是否进 git、云端边界 |
| [`../E0/docs/data_card.md`](../E0/docs/data_card.md) | 数据事实 | E0 冻结口径与六个硬事实 |

---

## 4. 平台、运行、依赖

| 文档 | 说明 |
|---|---|
| [`platform_setup.md`](platform_setup.md) | `/code/workspace` vs `/data`、任务字段、续训、故障排查 |
| [`training_tasks.md`](training_tasks.md) | 可直接照抄的训练任务配置表 |
| [`image_requirements.md`](image_requirements.md) | 基础镜像、轻量依赖白名单、禁装清单 |
| [`dependencies.md`](dependencies.md) | 依赖声明、版本锁定、降级策略 |
| [`../dist/README.md`](../dist/README.md) | 数据分发包生成与部署 |
| [`../start.sh`](../start.sh) | 云端统一入口（`--to/--stage/--wp/--list/--dry-run`） |
| [`../run_train.sh`](../run_train.sh) | 平台任务统一入口（`env/data/e0/smoke/stage/all`） |

---

## 5. 模型策略、参考与冲分

| 文档 | 说明 |
|---|---|
| [`SCORE_MAX_PLAN.md`](SCORE_MAX_PLAN.md) | WP0–WP11 冲分/数据适配计划 |
| [`SPWLA2021_REVIEW.md`](SPWLA2021_REVIEW.md) | SPWLA 2021 前五名复盘与 v4 启示 |
| [`../资料引用索引.md`](../资料引用索引.md) | 外部引用定位 |
| `../src/`、`../E*/code/` | 真正的模型、损失、特征、训练与推理实现 |

---

## 6. Gate、审查与修复记录

| 文档 | 说明 |
|---|---|
| [`gate_template.md`](gate_template.md) | Gate 预注册模板、字段语义、判定逻辑 |
| [`CODE_REVIEW_STATUS.md`](CODE_REVIEW_STATUS.md) | 项目级代码审查发现的逐条处置/遗留状态 |
| [`AUDIT_FIXES.md`](AUDIT_FIXES.md) | 第二轮审查修复清单（历史记录） |
| `../reports/V4_PLAN_REVIEW*.md` | 计划多轮审查报告（历史记录） |
| `../versions/prereg_templates/` | 33 份 Gate 预注册模板 |

---

## 7. 阶段与 P 级计划

`E0/`–`E11/` 是执行阶段；每级包含：

- `E*/PLAN.md`：阶段目标、Gate、任务拆分；
- `E*/P*/PLAN.md`：P 级详细执行计划（通常由 `docs/gen_p_details.py` 生成）；
- `E*/code/`、`E*/docs/`：实现与阶段证据文档。

阅读建议：

1. 先读 [`../PLAN.md`](../PLAN.md) 的总阶段表；
2. 再读目标阶段的 `E*/PLAN.md`；
3. 只有需要执行细节时才进入 `E*/P*/PLAN.md`；
4. **不要手工编辑生成物**；若确需修改，回到 `../PLAN.md` 或 `gen_plans.py` / `gen_p_details.py`。

---

## 8. 文档维护规则

1. **新增文档必须登记**到本文对应分类；新增 Gate/审查文档同时更新 `../versions/status.json`。
2. 每份说明文档建议包含：目的、事实源、与代码/报告的链接、最后更新说明。
3. 不把“历史报告”当“当前结论”：历史审查/计划文件保留原文，但必须在页眉注明状态。
4. 路径优先使用相对链接；不要移动已被代码或测试引用的文件。
5. 数字声明必须可复算；若不能复算，写“待验证”并给复算命令。
