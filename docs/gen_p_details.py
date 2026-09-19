#!/usr/bin/env python3
"""生成 v4 的 P 级详细子计划（V2 深度）。

每份 P 级 PLAN.md 覆盖：
  0. 元信息（阶段/层级/性质/依赖/代码归属/产物目录）
  1. 目标（可量化）
  2. 为什么需要这一步（依据）
  3. 输入契约（来自谁、字段/格式）
  4. 输出契约（schema 逐字段说明）
  5. 执行步骤（编号、可执行）
  6. 参数与配置表（默认值 + 搜索范围 + 在哪选）
  7. 完成判据（Gate 预注册字段级）
  8. 禁止事项
  9. 风险与对策 / 10. 停止规则 / 11. 复算命令 / 12. 预注册模板

用法： python3 v4/docs/gen_p_details.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # v4/

# =============================================================================
# 详细内容表
# 每个 P: dict(goal, why[list], inputs[list], outputs[list], steps[list],
#              params[list[tuple]], done[list], forbid[list], risk[list],
#              stop[list], prereg_extra[dict], code[list], nature, deps[list],
#              evidence[list])
# =============================================================================

P: dict[str, list[dict]] = {}

# ------------------------------------------------------------------ E0
P["E0"] = [
    dict(
        pid="P0", title="环境与磁盘实测（云端第一次运行）",
        status="blocked",
        nature="契约前置：不产出模型，只产出**环境事实**",
        deps=["无（这是全项目第一步）"],
        goal="在云端实机确认 CUDA 12.6 / PyTorch 2.4.0 / Python 3.11 / A100 sm_80 / bf16 可用，"
             "并实测 30 GB 磁盘的可用余量与分布，产出可复算的 `E0_env.json` 与 `E0_disk_budget.json`。",
        why=["未实测的环境假设会在 E3 训练数小时后才暴露（OOM / 版本不兼容 / 磁盘写满），"
             "返工成本是 A100 机时；",
             "30 GB 预算是本项目最硬的资源约束，而 `/code/workspace`（临时）与 `/data`（云盘）"
             "是否同一文件系统必须实测，不能假设；",
             "`torch 2.4.0` 与 `2.5+` 的 API 差异（`torch.nn.attention`、`torch.export` 新签名）会直接"
             "导致运行期 AttributeError，必须前置断言。"],
        inputs=["云端训练任务（Git 仓库代码来源，A100 资源，PyTorch 2.4.0/CUDA 12.6/py3.11 镜像）",
                "`v4/E0/code/check_env.py`、`v4/E0/code/setup_deps.sh`、`v4/src/data/disk_guard.py`"],
        outputs=["`$V4_REPORTS_DIR/E0_env.json`",
                 "`$V4_REPORTS_DIR/E0_disk_budget.json`",
                 "`versions/locks/cloud_frozen.txt`（镜像构建后 `pip freeze` 快照）",
                 "`$V4_LOG_DIR/train_env_*.log`（stdout 全量日志）"],
        steps=["`bash /code/workspace/v4/run_train.sh --mode env`",
               "读日志确认 `hard failures: 0`，逐项核对 torch/cuda/gpu/bf16/disk 五行",
               "`df -h / /data /code/workspace` 记录三个挂载点的容量与是否同盘",
               "`du -sh /usr /opt /root 2>/dev/null` 记录镜像本体占用，推算项目可用空间",
               "若可用 < 12 GB，把 `contingency_applied=true` 与收缩项写入 `E0_disk_budget.json`",
               "把 `E0_env.json` 的 `deps.versions` 与 `versions/locks/cloud.txt` 逐项比对，不一致则更新 lock 并提交"],
        params=[("`--min-free-gb`", "8.0", "8–12", "磁盘硬门禁；实测后若过紧则上调"),
                ("`--allow-non-a100`", "false", "—", "仅本机开发时开启（把 GPU 检查降级为 warn）"),
                ("`NUM_WORKERS`", "4", "2–6", "16 GiB 内存下的安全值，见总计划 §3.1-3")],
        done=["`check_env.py` 的 **hard 检查 6/6 通过**：python 3.11 / torch 2.4.0 / cuda 可用 / "
              "A100 sm_80 / bf16 / disk ≥ 8 GiB",
              "`E0_disk_budget.json` 含 `total_gb/used_gb/free_gb/level`，且 `level==\"ok\"`",
              "`E0_env.json` 含全部 9 个可选依赖的 `available/versions`",
              "`cloud_frozen.txt` 已生成并与 `versions/locks/cloud.txt` 一致或已更新"],
        forbid=["`pip install torch` / 升级 CUDA / 用 conda 建环境",
                "安装 flash-attn / xformers / apex / deepspeed 等需编译 CUDA 扩展的包",
                "在磁盘余量未知的情况下启动任何训练"],
        risk=[("镜像里是 torch 2.5+/2.3，不是 2.4.0", "hard failure 直接报错", "改用镜像内实际版本并同步改 `versions/locks/cloud.txt` 与代码中的 2.4-only 断言"),
              ("系统内存被镜像/其它进程占用", "`free -g` 显示可用 < 14 GiB", "把 `num_workers` 降到 2，并在 E2 关闭特征缓存"),
              ("`/data` 与 `/code` 同盘且总容量仅 30 GB", "`df` 显示同一 Filesystem", "按总计划 §3.4.1 安全规则收缩：特征缓存 ≤0.5 GB、集成成员 ≤2、模型宽度减半"),
              ("pip 计划替换 torch", "`setup_deps.sh` 预检命中", "脚本自动中止（exit 3）；改为只用镜像自带版本")],
        stop=["hard failure 未清零前，禁止进入 E0/P1 之后的任何阶段",
              "可用磁盘 < 8 GB 且无法清理时，暂停项目并先与 owner 确认配额"],
        code=["E0/code/check_env.py", "E0/code/setup_deps.sh", "src/data/disk_guard.py", "run_train.sh"],
        evidence=["`reports/E0_env.json`（云端实测快照）", "`reports/E0_disk_budget.json`"],
        prereg_extra={"primary_metric": "env_hard_checks_passed", "thresholds": {"min_hard_pass": 6},
                      "mandatory_checks": ["env_hard_checks_passed", "disk_budget_ok"]},
    ),
    dict(
        pid="P1", title="数据卡、哨兵规则与标签三状态",
        status="done",
        nature="契约冻结：产出**唯一事实源**的数据卡",
        deps=["E0/P0（环境可用）"],
        goal="用自写解析器读取 90 口井，冻结缺失哨兵规则、标签三状态判据与目标值域统计，"
             "产出数据卡、按井折导出与指纹；**必须查清 3 口非规范 schema 井的正确解析方式**。",
        why=["口径是唯一事实源：解析错一列，后面所有分数都不可比；",
             "**实测发现 3 口训练井表头非官方 17 列**（`42f2870b` 20 列含 K/U/CGR、`b7eb1274` 21 列含 TH/K/U/CGR、`c7611b01` 16 列缺 CASE），共 27,080 行（3.71%），且都在 80 井折内、三目标齐全——按列位置解析会错位或丢行；",
             "**目标值域必须实测而非假设**：E0-R1 曾误以为 SW 是 `99.9`（百分数）+ `[0,1]`（小数）双尺度，实测有效 SW 为 min 8.305 / median 82.805 / max 99.9、SW<1 仅 11 行 → 实为单一标签尺度（审查 B2）；",
             "开发过程中已实际触发一次 numpy 越界切片静默截断（`arr[:,15:18]` 在 17 列数组上返回 2 列，丢掉 SW 整列），必须用断言防回归。"],
        inputs=["`$V4_DATA_ROOT/v4/data/train/*.txt`（80 井，含表头/单位/数据三段）",
                "`$V4_DATA_ROOT/v4/data/test/*.txt`（10 井，无标签）",
                "`rules.md` §5.1–5.3", "`资料库/12` §3.1（占位分布）"],
        outputs=["`$V4_REPORTS_DIR/E0_data_card.json`（计数/状态/深度诊断/schema 异常清单/目标分布统计/泄漏回归）",
                 "`$V4_REPORTS_DIR/E0_folds.json`（outer 5 折 + 每折 inner 3 折）",
                 "`versions/folds_sha256.json`（折指纹）",
                 "`$V4_CACHE_ROOT/raw/<split>/<well>.npz`、`labels/<well>.npz`（按井分片）"],
        steps=["`python3 E0/code/run_all.py --with-cache`（默认从 `V4_DATA_ROOT` 取数据；"
               "`--with-cache` 同时产出 E1 需要的 raw/labels 分片与 cache/manifest.json）",
               "核对 `train_rows`=730,268、`test_rows`=95,948、`state_counts`={missing:6700, placeholder:487225, valid:236343}",
               "核对 `noncanonical_schema_wells` 恰好 3 口，且 `missing_columns`/`extra_columns` 与实测一致",
               "核对 `folds_sha256`=f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e 且 fold_sizes 各 16 井",
               "`python3 tools/verify_reference.py` 必须输出 `RESULT: OK`",
               "核对 `target_stats` 的 SW 有效切片（min 8.305 / median 82.805）与 `sw_scale` 字段",
               "核对 `input_leak_regression.passed == true`（90 井 13 列输入、输入与目标不相交）",
               "核对 `score_consistency.consistent == true`（总分恒等式）",
               "用 `V4_DATA_ROOT` 指向云端 `/data` 再跑一次，确认路径契约生效"],
        params=[("缺失哨兵", "{-99999, -9999, NaN, 任何 < -1000}", "冻结，不可改", "`src/constants.py::SENTINELS/MISSING_LT`"),
                ("占位常量", "(POR=0.1, PERM=0.01, SW=99.9)", "冻结", "`constants.PLACEHOLDER`"),
                ("占位判等容差", "1e-9（绝对）", "冻结", "防止浮点写成 0.1000000001 时误判"),
                ("折数", "5（outer）/ 3（inner）", "冻结", "与 v1 同折以保证历史锚点可比")],
        done=["`E0_data_card.json` 全部计数与上表一致（逐项 equality，不是近似）",
              "3 口畸形井在三目标上**零丢失**；仅 `c7611b01` 的 CASE 记为 NaN + 缺失位，"
              "另两口的多余曲线记入 `extra_columns`",
              "`input_leak_regression.passed == true` 且 `score_consistency.consistent == true`",
              "`folds.json` 的 80 井与数据目录文件名集合**完全一致**（不重不漏）",
              "`verify_reference.py` 输出 `RESULT: OK`"],
        forbid=["按固定列位置解析（必须按表头名映射）",
                "删除占位行 / 把 3 口畸形井排除出训练",
                "用 `pd.read_csv` 默认行为直接吃表头（单位行会被当成数据）",
                "在没有列数断言的情况下做数组切片"],
        risk=[("把 3 口畸形井按位置解析", "POR/PERM/SW 三列错位或行数少 27,080", "硬断言 `arr.shape[1]==n_out` + 逐井 schema 清单入数据卡"),
              ("numpy 越界切片静默截断", "SW 列被丢且不报错", "目标列用 `n_out-len(TARGETS)` 动态计算 + `targets.shape[1]==3` 断言"),
              ("单位行被当作数据行", "行数多 80/10，深度出现 'm'", "`next(reader)` 跳过第 2 行；非数值 token 记 NaN 并计数"),
              ("折文件被 autocrlf 改写", "字节 sha256 不一致", "`verify_reference.py` 同时校验内容指纹（与行尾无关）")],
        stop=["数据卡计数与实测不符时，禁止进入 E0/P2",
              "发现新的 schema 变体（>5 种）时，暂停并重新设计解析契约"],
        code=["src/data/parse.py", "src/data/labels.py", "src/data/dataset.py",
              "src/validation/folds.py", "E0/code/run_all.py", "tools/verify_reference.py"],
        evidence=["`E0/docs/data_card.md`（含 3 口井实测表与泄漏事故复盘）",
                  "`reports/E0_data_card.json`", "`reports/E0_folds.json`",
                  "`versions/folds_sha256.json`"],
        prereg_extra={"primary_metric": "data_card_recomputable",
                      "mandatory_checks": ["data_card_recomputable", "row_counts_match",
                                           "folds_fingerprint_present"]},
    ),
    dict(
        pid="P2", title="官方评分器复算与分母口径冻结",
        status="done",
        nature="度量工具冻结：评分器错了，后面全部结论作废",
        deps=["E0/P1（数据卡与标签状态）"],
        goal="按 `rules.md` §7.3 实现三目标评分器，用全常量 (0.1, 0.01, 99.9) 复算，"
             "**确定官方分母口径**（逐目标排除缺测 vs 全行），并冻结为项目唯一口径。",
        why=["评分器是所有 Gate 的度量工具，必须与官方逐点一致；",
             "分母口径有歧义：rules 公式写 `1/N Σ`（N=总点数），但实测常数基线在"
             "**逐目标排除缺测**口径下 = 70.490735（命中公开锚点 70.4907 ±1e-4），"
             "而在全行口径下 = 69.843218（低 0.65 分）——0.65 分足以改变 Gate 判定；",
             "`资料库/12` §1 明确 `eps` 只用于防除零，取 1e-3 而非 1e-6 可稳定梯度。"],
        inputs=["`rules.md` §7.3–7.4", "`资料库/12` §1（逐条解析）、§3.3（分数预算表）",
                "E0/P1 的 730,268 行标签与缺测掩码"],
        outputs=["`$V4_REPORTS_DIR/E0_score_check.json`（两种口径的逐目标 Acc 与 Total）",
                 "`src/score.py`（冻结实现，默认 `missing_mode=\"drop\"`）",
                 "`src/constants.py::SCORE_MISSING_MODE/SCORE_WEIGHTS/DELTA_POR/DELTA_SW`"],
        steps=["实现 `acc_relative`（POR/SW）与 `acc_perm`（log10）两个原子函数",
               "实现 `score_arrays(y_true, y_pred, missing, missing_mode)` 返回逐目标 Acc 与 Total",
               "两种口径各跑一次常数基线，记录到 `E0_score_check.json`",
               "核对 POR/SW/PERM 三个 Acc 与 `资料库/12` §3.3 预算表自洽（占位白送 66.72）",
               "把命中锚点的口径写入 `constants.SCORE_MISSING_MODE` 并加单元测试锁定",
               "写 4 个边界单测：y=0、ŷ=0、ŷ/y=10、缺测行"],
        params=[("`missing_mode`", "drop（冻结）", "drop/mask", "mask 仅作诊断对照，报告必须标注"),
                ("`eps`", "1e-3", "冻结", "`资料库/12` §2.4 提示 2"),
                ("PERM 比值下限", "eps=1e-3", "冻结", "`log10(max(ŷ/y, eps))`"),
                ("权重", "POR 0.30 / PERM 0.35 / SW 0.35", "冻结", "rules §7.4"),
                ("容差", "POR δ=0.08 / SW δ=0.05", "冻结", "rules §7.3")],
        done=["常数基线在冻结口径下 = **70.490735 ± 1e-4**（锚点 70.4907）",
              "另一种口径的数字同时记录（69.843218）并在数据卡标注差异",
              "逐目标 Acc 与预算表自洽：POR 0.6736 / PERM 0.7467 / SW 0.7421（±0.002）",
              "`src/score.py` 在**没有 torch** 的环境下可导入并运行"],
        forbid=["用第三方库的 `mean_squared_error` / `r2_score` 代替官方公式",
                "把缺测行计入分母（除非显式标 `missing_mode=\"mask\"`）",
                "更换口径后不重算全部历史数字"],
        risk=[("口径选错", "所有 OOF 数字系统性偏低 0.65 分", "以锚点 70.4907 命中的口径为准，并写进 constants + 单测"),
              ("eps 取 1e-6 导致梯度爆炸", "训练早期 loss NaN", "固定 eps=1e-3"),
              ("PERM 出现 0 或负值", "log10 报错或 -inf", "`max(·, eps)` + ŷ>0 由模型层保证")],
        stop=["锚点未命中时禁止进入 E1；必须先修正评分器或数据卡"],
        code=["src/score.py", "E0/code/run_all.py"],
        evidence=["`reports/E0_score_check.json`（两种口径 + 逐目标 + 恒等式校验）",
                  "`reports/E0_data_card.json::constant_baseline`"],
        prereg_extra={"primary_metric": "constant_baseline_anchor",
                      "thresholds": {"abs_tolerance": 1e-4},
                      "mandatory_checks": ["constant_baseline_anchor_hit"]},
    ),
    dict(
        pid="P3", title="提交契约、版本路由与干净目录冒烟",
        status="done",
        nature="交付契约冻结：格式错误 = 零分风险",
        deps=["E0/P1（数据）、E0/P2（评分）"],
        goal="实现官方 CLI 的 `predict.py`、`result.json` 结构校验、候选注册表与 manifest，"
             "并在**只含代码与数据**的干净目录完成冒烟（含常数基线端到端）。",
        why=["`rules.md` §6.1 规定 JSON 结构禁止增删改字段，小写 `depth`，10 井 95,948 行；"
             "格式错一次就浪费一次每日 5 次的提交额度；",
             "`rules.md` §8.4：复现失败直接取消资格，因此契约必须早于模型存在并可自动化校验；",
             "契约校验必须**不依赖 torch**，否则本机（无 GPU）无法在提交前自检。"],
        inputs=["`rules.md` §6.1–6.3、§8", "`data/测试数据返回结果格式.json`（官方模板）",
                "`资料库/12` §4（提交规范）、§5（复现要求）"],
        outputs=["`v4/predict.py`（`--data_dir/--output/--use-version/--list-versions`）",
                 "`src/inference/contract.py`（`validate_payload`/`validate_file`/`depth_alignment_report`）",
                 "`versions/candidates.json`（候选注册表，唯一事实源）",
                 "`$V4_REPORTS_DIR/E0_contract_tests.json`（正/负样例自检）"],
        steps=["实现 `validate_payload`：顶层键集合、logId 集合与文件名一致、行数、逐行键名、"
               "depth 严格递增、有限性、PERM>0、禁止 SW 裁剪",
               "实现 `predict.py`：识别 `--data_dir` 指向 `data/` 或测试井目录两种形态；"
               "生成后自动调用契约校验，失败即非零退出",
               "支持 `--use-version CONST` 走常数基线（用于契约自检，不参与评分竞争）",
               "跑 6 个负样例单测（PERM≤0 / 缺顶层键 / 行数不符 / depth 乱序 / 大写 DEPTH / NaN）",
               "在只含 `v4/` 与 `data/` 的干净目录执行 `python3 predict.py --data_dir ./data --output result.json`",
               "建立 `versions/candidates.json` 空表与 schema 注释"],
        params=[("预期测试井数", "10", "冻结", "`constants.EXPECTED_N_TEST_WELLS`"),
                ("预期测试行数", "95,948", "冻结", "`constants.EXPECTED_N_TEST_ROWS`"),
                ("depth 小数位", "1", "冻结", "相对输入深度对齐，容差 1e-6")],
        done=["6 个负样例**全部被正确拒绝**，正样例通过（`E0_contract_tests.json::passed=true`）",
              "干净目录下 `python3 predict.py --use-version CONST --data_dir ./data --output result.json` "
              "在一次运行内产出 10 井 / 95,948 行且 `contract_ok=true`（实测 ≈1.4 s，单核 CPU）",
              "`predict.py` 在**无 torch** 环境可运行；`--list-versions` 正确区分可用/未训练版本",
              "`versions/candidates.json` 建立且被 `predict.py` 读取"],
        forbid=["契约校验依赖 torch 或网络",
                "静默裁剪 SW / POR 到物理区间",
                "允许 logId 缺失、行数不符、深度错位通过校验",
                "把未训练版本当作可用版本静默输出常数"],
        risk=[("官方 `--data_dir` 指向 `data/` 而代码假设指向 `test/`", "找不到井文件", "`predict.py` 同时支持两种形态并打印实际使用的目录"),
              ("JSON 键名大小写不一致", "评测字段解析失败", "契约强制小写 `depth`，并有负样例单测锁定"),
              ("浮点序列化差异", "两次运行 sha256 不同", "固定小数位与序列化参数，E10 做两次运行一致性校验")],
        stop=["契约自检未全绿，禁止任何候选进入 `submitted` 状态"],
        code=["predict.py", "src/inference/contract.py", "src/versioning/registry.py"],
        evidence=["`reports/E0_contract_tests.json`（6 负样例全拒绝）",
                  "`versions/registry.json`", "`versions/candidates.json`"],
        prereg_extra={"primary_metric": "contract_selftest_passed",
                      "mandatory_checks": ["contract_selftest", "no_torch_required"]},
    ),
]

# ------------------------------------------------------------------ E1
P["E1"] = [
    dict(
        pid="P0", title="行级输入管线与分片缓存",
        nature="数据管线：为 E1–E8 共用，必须先冻结",
        deps=["E0/P1（分片写入）、E0/P2（评分）"],
        goal="构建并缓存 `F1 = 13 条曲线 + DEPTH 原始值 + 13+1 缺失位 + 4 深度编码 = 32 维` 行级张量与三目标标签，"
             "落盘为按井分片，并验证内存占用符合 16 GiB 预算。",
        why=["输入管线的正确性决定后面所有对比是否有意义：特征与标签必须逐行对齐；",
             "16 GiB 系统内存是真正的瓶颈，必须把\"按井分片 + 按需读取\"固化为管线，否则 E3 一开始就 OOM；",
             "标准化参数只能在训练折 fit（`资料库/07` §5、§8.6），因此管线必须支持\"折内 fit\"接口。"],
        inputs=["`$V4_CACHE_ROOT/raw|labels/<well>.npz`（E0/P1 产出）",
                "`src/validation/folds.py`（outer/inner 折）"],
        outputs=["`src/features/basic.py`（`build_row_features`/`build_labels`/`decode_predictions`）",
                 "`src/data/row_dataset.py`（折内拼接 + 标准化）",
                 "`$V4_REPORTS_DIR/E1_row_features.json`（维度、缺失率、内存峰值、耗时）"],
        steps=["实现 `build_row_features`：14 原始 + 14 缺失位 + 缺失比例 + 相对深度 + 深度步长 + 深度序号",
               "实现 `build_labels`：POR/SW 原尺度、PERM 转 log10、三目标 mask、联合占位标签",
               "实现 `RowScaler`：**只在训练折 fit** 的中位数填补 + 均值/标准差标准化，输出 JSON 参数",
               "实现折内数据装配：outer 训练折做 train、outer 验证折做推理，保证行级对齐",
               "实测内存：单折激活内存峰值与常驻内存，写入报告",
               "缓存体积核对：`raw`+`labels` 合计应远小于 0.2 GB"],
        params=[("特征维度", "32", "冻结（F1）", "含 4 个深度编码列"),
                ("标准化", "中位数填补 + 零均值单位方差", "冻结", "参数仅在训练折 fit"),
                ("PERM 变换", "log10, clip[-6,6]", "冻结（F1）", "`constants.PERM_LOG_MIN/MAX`"),
                ("分片格式", "npz(compressed), float32", "冻结", "原子写：tmp → rename")],
        done=["特征/标签逐行对齐：`X.shape[0] == y.shape[0] == mask.shape[0]` 对全部 90 井成立",
              "折内装配后训练/验证行的井集合与 `folds.json` 完全一致（无井级泄漏）",
              "`RowScaler` 参数可序列化为 JSON 并在推理时复现",
              "常驻内存 < 6 GiB、缓存 < 0.2 GB（实测写入报告）"],
        forbid=["在全部 80 井上 fit 标准化参数（必须折内 fit）",
                "把井身份（`logId`）或折号作为特征",
                "把占位行从训练集中剔除"],
        risk=[("特征与标签错位", "训练 loss 不下降或分数异常低", "断言行数一致 + 抽查若干井的 depth 对齐"),
              ("标准化泄漏", "OOF 虚高", "`RowScaler` 只接受训练折索引，接口层拒绝整表 fit"),
              ("内存峰值过高", "被 OOM killer 杀", "按井拼接而非全量 concat；`num_workers=4`")],
        stop=["行级对齐断言失败时，禁止进入 E1/P1"],
        code=["src/features/basic.py", "src/data/row_dataset.py", "src/data/dataset.py"],
        evidence=["`reports/E1_row_features.json`"],
        prereg_extra={"primary_metric": "row_pipeline_ok",
                      "mandatory_checks": ["row_pipeline_ok", "no_label_leak"]},
    ),
    dict(
        pid="P1", title="行级 MLP + 评分对齐损失 + 5 折 OOF（硬 Gate ≥ 78.0）",
        nature="**分母建立阶段**：允许弱，必须正确",
        deps=["E1/P0"],
        goal="训练多任务 MLP（共享主干 + POR/PERM/SW 三头 + 联合占位头），用三段式对齐损失，"
             "跑完 80 井按井 5 折 OOF，产出逐行预测与官方口径评分；**硬 Gate：OOF Total ≥ 78.0、5 折同向、占位行逐目标 Acc ≥ 0.98**。",
        why=["在引入序列主干前必须知道\"只看当前深度点\"的上限，否则无法证明 E3 序列上下文的价值（`资料库/08` §0.3 第 1 层）；",
             "行级基线训练极快（分钟级），是验证损失实现、数据管线、OOF 流程是否正确的最高性价比手段；",
             "v2 E4/P3 的 CPU MLP 是 NO-GO，但那是**逐点 + 无 GPU + 小容量**；E1 要给出"
             "\"正确实现下的行级上限\"作为 E3 的严格对照；",
             "`资料库/12` §2.3 指出纯对齐损失早期信号稀疏，因此必须用三段式 + 用**真实评分**早停。"],
        inputs=["E1/P0 的行级特征与标签", "`src/losses/score_aligned.py`", "E0 的评分器与折"],
        outputs=["`models/E1/pd0_fold{k}.pt`（5 折权重，bf16 state_dict）",
                 "`$V4_RUN_ROOT/E1/oof.npz`（well_id/depth/y_true/y_pred/q_ph，逐行）",
                 "`$V4_REPORTS_DIR/E1_metrics.json`（逐折/逐目标/连续切片/bootstrap CI）",
                 "`$V4_REPORTS_DIR/E1_loss_curve.csv`（每 epoch 训练/验证真实分数）",
                 "`$V4_REPORTS_DIR/E1_gate.json`"],
        steps=["预注册 `E1_P1_gate_prereg.json`（阈值、候选数、bootstrap 设置、mandatory checks）",
               "实现 `train_row.py`：`--resume`、`--time-budget-h`、每 epoch checkpoint、"
               "每 epoch 调 `assert_disk_headroom(8.0)`、写 `training_time_log.json`",
               "跑 fold0 小规模冒烟（`--max-wells 8 --epochs 2 --smoke`）确认链路与显存/内存",
               "全 5 折训练：bf16、AdamW、余弦退火、梯度裁剪 1.0；λ₁ 从 1.0 退火到 0.1",
               "每 epoch 在**验证折**上用真实 `score.py` 算分（早停依据，不用 loss 值）",
               "汇总 OOF → `score_arrays(..., missing_mode=\"drop\")` → 逐目标 Acc 与 Total",
               "逐折 delta、逐井非退化比例、按井行数加权 paired cluster bootstrap（1000 次）",
               "评估占位行逐目标 Acc/precision/recall，写入 Gate 的 `atomic_precision_reported`",
               "写 `E1_gate.json` 并判定是否 ≥ 78.0"],
        params=[("`hidden`", "256", "128/256/512", "在 fold0+1 上选，选后冻结"),
                ("`layers`", "2", "1/2/3", "同上"),
                ("`dropout`", "0.1", "0.0/0.1/0.2", "同上"),
                ("`lr`", "2e-3", "5e-4/1e-3/2e-3/5e-3", "AdamW，余弦退火到 1e-4"),
                ("`weight_decay`", "1e-4", "0/1e-5/1e-4/1e-3", "同上"),
                ("`batch_size`", "4096", "2048/4096/8192", "内存允许下尽量大（显存不是约束）"),
                ("`epochs`", "40", "20–80", "结合早停（patience 5）"),
                ("`λ1` 退火", "1.0 → 0.1", "线性，前 60% epoch", "`资料库/12` §2.3 建议"),
                ("`λ2`（占位 BCE）", "0.2", "0.1/0.2/0.3", "同上"),
                ("`alpha`/`beta`", "1e-3 / 20", "冻结", "Charbonnier / softplus 平滑参数"),
                ("`seed`", "42", "42/1337", "固定；多 seed 视为集成成员（E8）")],
        done=["**OOF Total ≥ 78.0**（硬 Gate，低于此值视为实现 bug，先排查不扩容）",
              "5 折 delta **全部同向**（相对全常量基线）",
              "占位行逐目标 Acc **≥ 0.98**，且 `atomic_precision_reported` 写入 Gate",
              "加权配对井级 cluster bootstrap 95% CI 下界 > 0",
              "loss 曲线无 NaN；训练可 `--resume` 且 `disk_budget_ok`、`training_time_log_valid` 均为 true",
              "CONTIN 连续切片（排除占位行）逐目标 Acc 一并上报"],
        forbid=["加入任何窗口/序列特征（属于 E2/E3）",
                "用 outer 折或 A 榜选超参、阈值、早停点",
                "用 loss 值替代真实评分做模型选择",
                "为冲分而删除占位行或裁剪 SW"],
        risk=[("损失实现有误导致学不动", "loss 长时间不降或 Total < 70", "先用 `--smoke` 在 5000 行上做过拟合测试（应能拟合到接近满分）"),
              ("标准化泄漏", "OOF 虚高、A 榜落差大", "折内 fit 断言 + 单元测试"),
              ("占位行学坏", "占位 Acc < 0.98、Total 卡在 70 出头", "提高 λ₂ 或对占位行过采样；检查 SW 尺度换算"),
              ("PERM 长尾崩塌", "PERM Acc < 0.85", "确认在 log10 空间监督；检查 clip 范围"),
              ("内存/磁盘被打爆", "训练中途被杀", "`num_workers=4`、checkpoint 滚动淘汰、`disk_guard`")],
        stop=["OOF < 78.0 时**禁止扩容**：先做\"5000 行过拟合测试\"与\"折内一致性检查\"",
              "连续 2 次 NaN → 回退上一 checkpoint 并减半 lr",
              "单折耗时超软预算 3 倍 → 减 epoch 或减宽度"],
        code=["E1/code/train_row.py", "E1/code/eval_oof.py", "src/models/row_mlp.py",
              "src/losses/score_aligned.py", "src/training/loop.py"],
        evidence=["`reports/E1_metrics.json`、`reports/E1_gate.json`、`versions/candidates.json::E1_PD0`"],
        prereg_extra={
            "primary_metric": "oof_total", "baseline_version": "CONST",
            "thresholds": {"min_delta": 7.5, "min_effect_floor": 0.0, "oof_total_min": 78.0},
            "pilot_std": None, "mde_units": 80, "min_detectable_effect": None,
            "multiplicity": "none", "candidate_budget": 1, "bootstrap_iters": 1000,
            "bootstrap_unit": "well_row_weighted_cluster",
            "mandatory_checks": ["contract_ok", "atomic_precision_reported", "disk_budget_ok",
                                 "training_time_log_valid", "checkpoint_resumable", "no_label_leak"],
        },
    ),
]

# ------------------------------------------------------------------ E2
P["E2"] = [
    dict(
        pid="P0", title="物理与交会特征（F_phys）",
        nature="特征组候选：必须独立消融，未过则 NO-GO",
        deps=["E1/P1（行级基线与评分口径）"],
        goal="实现孔隙度类（Wyllie/密度/中子）、泥质类（GR/SP 指数）、流体类（RT/RXO、log10 RT）、"
             "骨架类（PE、DEN-CNL 差）共约 14 列物理派生特征，并在行级基线上做**单组消融**。",
        why=["`资料库/01` §2–§5 与 `资料库/02` 给出成体系的岩石物理公式，是领域归纳偏置的合法来源；",
             "物理先验是**约束**不是万能解：`资料库/03` 明确 Kozeny–Carman 只能定性，"
             "`资料库/13` 指出渗透率两倍以内已算很好，因此必须消融而不能硬编码进模型；",
             "特征一旦进入训练就必须冻结版本（先定义再实验），否则会出现\"看 OOF 后加列\"的选择偏差。"],
        inputs=["E1/P0 的行级张量", "`资料库/01` §2–§6、`资料库/02`、`资料库/16`（交会图版）"],
        outputs=["`src/features/physics.py`（每个派生列一个纯函数 + 公式注释）",
                 "`$V4_REPORTS_DIR/E2_ablation.json`（组级 delta 与 CI）",
                 "`$V4_REPORTS_DIR/E2_feature_provenance.csv`（列名/公式/来源登记）"],
        steps=["按公式逐个实现派生列（Wyllie 声波孔隙度、密度孔隙度、中子孔隙度、GR 指数 IGR、"
               "SP 指数、RT/RXO 比值、log10 RT、PE 骨架指示、DEN-CNL 差、AC-DEN 交会等）",
               "对可能除零/负数的公式加数值保护（如 `(AC-ACma)/(ACf-ACma)` 的夹取）",
               "缺失输入传播为缺失输出（不填 0），并同步生成缺失指示位",
               "在行级 MLP 上做\"F1 vs F1+F_phys\"单组消融（同折同超参，只改特征）",
               "登记每个派生列的公式与出处到 provenance CSV",
               "给出采纳/NO-GO 结论与 bootstrap CI"],
        params=[("AC 骨架/流体时差", "ACma=55.5, ACf=189 μs/ft→换算为 μs/m", "依据 `资料库/01`", "需按数据单位换算"),
                ("DEN 骨架/流体密度", "ρma=2.65, ρf=1.0 g/cm³", "依据 `资料库/01`", "同上"),
                ("GR 泥质基线", "GRmin/GRmax 由**训练折**分位数确定", "折内 fit", "禁止用全量分位数"),
                ("消融判据", "组级 delta 的 CI 下界 > 0", "冻结", "否则 NO-GO")],
        done=["每个派生列有公式出处与数值保护，且无目标值参与构造",
              "组级消融给出 delta 与 CI，明确采纳或 NO-GO",
              "provenance CSV 覆盖全部新增列（列名 → 公式 → 依据）",
              "派生特征在测试井上同样可计算（不依赖标签）"],
        forbid=["使用目标值（POR/PERM/SW）构造任何派生列",
                "用全量数据确定 GRmin/GRmax 等分位数参数",
                "把物理公式当成硬约束直接替换模型输出（那属于 E7 的 L_phys 消融）"],
        risk=[("单位换算错误", "派生孔隙度出现 0–1 之外的离谱值", "对每个派生列做物理区间检查并记录越界比例"),
              ("公式引入泄漏", "消融提升异常大", "provenance 审计 + label-shuffle 检查"),
              ("NO-GO 被硬塞进模型", "特征表膨胀但无增量", "Gate 强制要求显式 NO-GO 记录")],
        stop=["单组消融 CI 上界 ≤ 0 时标记 NO-GO，不得进入 F2"],
        code=["src/features/physics.py", "E2/code/ablate_groups.py"],
        evidence=["`reports/E2_feature_provenance.csv`"],
        prereg_extra={"primary_metric": "target_acc", "thresholds": {"min_delta": 0.0},
                      "multiplicity": "holm", "candidate_budget": 3},
    ),
    dict(
        pid="P1", title="窗口与井级特征（F_win / F_well）+ 内存纪律",
        nature="特征组候选 + 数据管线扩容",
        deps=["E2/P0"],
        goal="实现居中多尺度窗口统计（窗长 11/51/201 点 ≈ 1.1/5.1/20.1 m）与井级聚合特征，"
             "并在 16 GiB 内存约束下落盘缓存、完成单组消融。",
        why=["v1 已证明滚动窗口是稳定增益（C1→C1W 提升 +1.0562，5/5 折同向），"
             "这是\"上下文有效\"的最强历史证据；",
             "工程曲线（CAL/DEVI/AZIM/BIT/CASE）在井内近常数（`资料库/08` §0.1-4），"
             "井级聚合是**唯一的井间信号通路**；",
             "窗口特征也是最贵的一组，必须按需生成、按版本目录落盘，否则 30 GB 磁盘与 16 GiB 内存都扛不住。"],
        inputs=["E1/P0 分片", "`资料库/07` §6（窗口与中心窗口要求）、§10（增强）"],
        outputs=["`src/features/window.py`、`src/features/well.py`",
                 "`$V4_CACHE_ROOT/feat/F2_win/<well>.npz`",
                 "`$V4_REPORTS_DIR/E2_mem_profile.json`（峰值内存/缓存体积/耗时）",
                 "`$V4_REPORTS_DIR/E2_ablation.json`（追加 F_win/F_well 两组）"],
        steps=["实现居中窗口统计：mean/std/min/max/trend（对窗内做线性回归斜率）与覆盖率",
               "实现井级聚合：各曲线井内均值/标准差/分位数、井长、平均采样间隔、井斜均值",
               "对每口井按需生成并原子落盘；换特征版本时**先删旧版本目录**",
               "做单组消融（F1+F_win、F1+F_well、F1+F_win+F_well）",
               "实测每折训练的内存峰值与缓存体积，写入 mem_profile",
               "确认无跨折/跨井泄漏（窗口只用同井邻域，井级统计只用训练折井）"],
        params=[("窗长", "{11, 51, 201} 点", "可加 {5, 1001}", "对应 1.1/5.1/20.1 m"),
                ("统计量", "mean/std/min/max/trend/coverage", "可裁剪", "trend = 窗内线性斜率"),
                ("窗口类型", "**居中**（离线任务无因果约束）", "冻结", "`资料库/07` §6 第 917 行"),
                ("缺失处理", "窗内有效值统计 + coverage 列", "冻结", "不填 0"),
                ("缓存上限", "2.0 GB（可用 <12 GB 时降至 0.5 GB）", "按实测调整", "总计划 §3.4.1")],
        done=["F_win 或 F_well 至少一组消融 CI 下界 > 0（否则 NO-GO 并记录）",
              "缓存体积 < 2 GB（收缩模式 < 0.5 GB），峰值常驻内存 < 12 GiB",
              "窗口统计严格居中，trend 计算无未来信息泄漏（对本任务无因果约束，但保持与 v1 一致）",
              "provenance CSV 覆盖窗口/井级列"],
        forbid=["使用非居中（因果）窗口（与 v1 的 C1W 口径不一致，无法对照）",
                "用验证井/测试井数据计算井级统计参数",
                "在缓存目录无限累积历史特征版本"],
        risk=[("窗口特征让内存爆掉", "worker 被 OOM kill", "按需读取 + LRU + `num_workers=4` + 每 worker <300 MB 自检"),
              ("缓存撑爆磁盘", "free < 8 GB", "版本目录先删后建；`disk_guard` 每 epoch 检查"),
              ("井级统计泄漏", "OOF 虚高", "只允许训练折井参与井级统计参数计算")],
        stop=["内存或磁盘触达阈值时，先降级特征组（去掉最贵窗口）再继续"],
        code=["src/features/window.py", "src/features/well.py", "E2/code/build_features.py"],
        evidence=["`reports/E2_mem_profile.json`、`reports/E2_feature_provenance.csv`"],
        prereg_extra={"primary_metric": "target_acc", "thresholds": {"min_delta": 0.0},
                      "multiplicity": "holm", "candidate_budget": 3,
                      "mandatory_checks": ["disk_budget_ok", "no_label_leak"]},
    ),
    dict(
        pid="P2", title="数据增强与吞吐标定",
        nature="正则化与预算标定：为 E3 的序列主干提供可行性依据",
        deps=["E2/P1"],
        goal="实现曲线随机掩码、深度抖动、段置换等增强；用单折小规模实验标定点/秒吞吐与 batch 上限，"
             "给出 E3 每折耗时的可靠估计。",
        why=["80 井太少（730k 行但只有 80 个独立井），正则化与增强是防过拟合的主要手段；",
             "E3 的序列主干比行级模型贵 1–2 个数量级，若不在 E2 标定吞吐，"
             "E3 的预算承诺就是猜的；",
             "增强必须**保持标签语义**：占位行的三目标一致性不能被破坏。"],
        inputs=["E2/P0–P1 特征", "`资料库/07` §10（数据增强）"],
        outputs=["`src/data/augment.py`",
                 "`$V4_REPORTS_DIR/E2_throughput.json`（点/秒、显存/内存峰值、每折耗时估计）"],
        steps=["实现增强：① 曲线通道随机置缺（模拟仪器失效）；② 深度轴小抖动（±2 点）；"
               "③ 井内随机段裁剪/重采样；④ 高斯噪声（按曲线量纲缩放）",
               "确认增强后标签仍逐行对应（对同一行做特征扰动，不移动标签）",
               "在 fold0 上跑不同 batch/length 组合，记录吞吐与峰值资源",
               "外推 5 折 × N epoch 的总耗时，写 throughput 报告",
               "给出 E3 的 `planned_task_training_h` 建议值"],
        params=[("通道掩码概率", "0.05", "0.0/0.05/0.1", "折内选择，不改标签"),
                ("深度抖动", "±2 点", "0/±2/±5", "只在特征侧"),
                ("高斯噪声 σ", "各曲线训练折标准差的 1%", "0/1%/3%", "同上"),
                ("增强开关", "默认开", "开/关", "消融验证是否真的提升 OOF")],
        done=["增强开关的消融结果（提升或 NO-GO）明确",
              "吞吐报告给出点/秒与每折耗时估计，误差 < 30%",
              "增强不改变任何标签行（单测：增强前后 y 与 mask 逐行相等）"],
        forbid=["增强改变标签或行数", "用验证/测试井数据估计噪声尺度", "无消融地默认开启全部增强"],
        risk=[("增强破坏占位一致性", "占位 Acc 下降", "只对输入特征做扰动；标签与 mask 不动"),
              ("吞吐估计过于乐观", "E3 训练超预算", "留 2× 余量，并在 E3/P1 首个 fold 复核")],
        stop=["吞吐估计若使 E3 单折 > 软预算 3 小时，先降低 chunk 长度再进入 E3"],
        code=["src/data/augment.py", "E2/code/throughput.py"],
        evidence=["`reports/E2_throughput.json`"],
        prereg_extra={"primary_metric": "target_acc", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 2, "multiplicity": "holm"},
    ),
]

# ------------------------------------------------------------------ E3
P["E3"] = [
    dict(
        pid="P0", title="序列数据集与分块（chunk）策略",
        nature="数据管线：序列建模的地基",
        deps=["E2/P2（吞吐标定）、E0/P1（分片）"],
        goal="实现按井分块的序列数据集：变长井切 chunk、边界 overlap、按需读取分片、worker 内存自检；"
             "确定 chunk 长度与 overlap 策略。",
        why=["整井 4,473–13,078 点无法一次性进模型（尤其 16 GiB 系统内存下），必须分块；",
             "分块策略直接决定**有效感受野**与吞吐的平衡：chunk 太短则 U-Net 看不到层段尺度，太长则内存与时间不可控；",
             "边界效应会导致 chunk 接缝处预测跳变，必须用 overlap + 加权拼接处理。"],
        inputs=["E1/P0 分片、E2 特征缓存", "`资料库/08` §1.2（窗口→窗口 seq2seq）",
                "`资料库/07` §8.4（深度序列划分）"],
        outputs=["`src/data/seq_dataset.py`",
                 "`$V4_REPORTS_DIR/E3_seq_dataset.json`（chunk 统计、worker 内存、采样顺序可复现性）"],
        steps=["实现 `SeqChunker`：按井切定长 chunk（默认 1024 点），相邻 chunk overlap 128 点",
               "实现 `SeqDataset.__getitem__`：打开分片 → 取 chunk → 即时算/读特征 → 关闭，"
               "**禁止把分片缓存在全局字典**",
               "每 worker 内存自检（< 300 MB），超限即报错而不是静默增长",
               "实现 `--smoke` 模式：前 8 井、1 折、2 epoch",
               "验证采样顺序在固定 seed 下可复现；验证 chunk 边界无标签错位",
               "实测 5 折吞吐并写入报告"],
        params=[("chunk 长度", "1024 点（≈102 m）", "512/1024/2048", "与 U-Net 深度耦合，E3/P2 消融"),
                ("overlap", "128 点", "0/64/128/256", "拼接时按距离加权"),
                ("`num_workers`", "4", "2–6", "16 GiB 内存约束"),
                ("`prefetch_factor`", "2", "1–4", "过高会挤爆内存"),
                ("`pin_memory`", "true", "—", "加速 H2D（显存不是瓶颈）")],
        done=["chunk 数×长度 ≈ 井长（覆盖完整，无丢点）",
              "每个 worker 常驻 < 300 MB；`num_workers=4` 时总内存 < 4 GiB",
              "固定 seed 下两次运行的采样顺序一致（可复现）",
              "chunk 边界处的 (x, y, mask) 逐行对齐（单测）"],
        forbid=["把整井常驻内存或缓存在全局字典",
                "打乱时破坏井内深度顺序（同一 chunk 内必须有序）",
                "让 chunk 覆盖出现空洞（丢点会直接丢分）"],
        risk=[("chunk 覆盖丢点", "OOF 行数 < 730,268", "覆盖性单测：所有井的 chunk 拼接后等于原长"),
              ("overlap 拼接权重错误", "接缝处预测跳变", "用三角/汉宁权重并按权重归一"),
              ("worker 内存膨胀", "被 OOM kill", "每 worker 自检 + `persistent_workers=False`")],
        stop=["覆盖性单测失败时禁止进入 E3/P1"],
        code=["src/data/seq_dataset.py"],
        evidence=["`reports/E3_seq_dataset.json`"],
        prereg_extra={"primary_metric": "seq_pipeline_ok",
                      "mandatory_checks": ["seq_pipeline_ok", "no_label_leak", "disk_budget_ok"]},
    ),
    dict(
        pid="P1", title="1D U-Net 与 TCN 主干实现",
        nature="主线模型实现：本计划的核心赌注",
        deps=["E3/P0"],
        goal="实现两种深度序列主干（1D U-Net 与 TCN），输出与输入逐行同长，接多任务头与联合占位头，"
             "在 bf16 下训练单折并记录参数量/显存/耗时。",
        why=["`资料库/08` §0.3 把 1D U-Net / TCN 列为\"最可能冲高分的结构\"，"
             "而前代因 CPU 限制从未真正训练过（v2 E7 是 NO-GO 但属\"无 CPU 可行方案\"）；",
             "0.1 m 采样下 10 m 储层段 = 100 点，**没有数百点感受野模型只能逐点外推**；",
             "U-Net 的 skip 保留高频细节（薄层），TCN 的空洞卷积给长程依赖，两者归纳偏置互补，"
             "必须头对头比较才知道哪个更适合本数据。"],
        inputs=["E3/P0 的 `SeqDataset`", "E1/P0 的 F1 特征（作为输入通道）",
                "`资料库/08` §4（1D-CNN/空洞/深度可分离）、§6（TCN）"],
        outputs=["`src/models/unet1d.py`、`src/models/tcn.py`、`src/models/heads.py`",
                 "models/E3/{unet,tcn}_fold{k}.pt",
                 "`$V4_REPORTS_DIR/E3_param_budget.json`（参数量/显存/单折耗时）"],
        steps=["实现 `UNet1D`：depth=5 级下采样（stride 2）+ 同层数上采样 + skip 拼接；"
               "每级 2×[Conv1d(k=5,groups=C) → BN → GELU]；`base_ch` 64→256",
               "实现 `TCN`：残差块 + 空洞卷积（dilation=2^i, i=0..8, k=3）+ weight norm + 残差",
               "实现 `SeqHead`：把主干输出 (B,L,d) 逐行送 POR/PERM/SW 三头 + 占位头（形状 (B,L)）",
               "确认前向输出长度与输入严格一致（`out.shape[1]==x.shape[1]`）",
               "单折训练（bf16 + 梯度裁剪 + AdamW），记录参数量与峰值显存/内存",
               "跑 `--smoke`（8 井 2 epoch）确认无 NaN、契约通过"],
        params=[("`base_ch`", "64", "32/64/128", "显存充足；内存不受影响"),
                ("U-Net 深度", "5", "3/4/5", "对应感受野 ≈ 数十–上百 m"),
                ("TCN 块数/最大 dilation", "9 块 / 512", "6/9；64/512", "感受野消融的关键变量"),
                ("卷积核", "5（U-Net）/ 3（TCN）", "3/5/7", "同上"),
                ("归一化", "BN（默认）", "BN/LN/GN", "E3/P2 消融"),
                ("`dropout`", "0.1", "0.0/0.1/0.2", "序列模型更易过拟合 80 井"),
                ("`lr`", "1e-3", "3e-4/1e-3/3e-3", "AdamW + 余弦"),
                ("bf16", "开启", "bf16/fp32", "A100 支持；fp16 易 NaN")],
        done=["两种主干都能前向且输出长度与输入一致",
              "参数量与峰值资源记录完整，单折耗时在软预算内",
              "`--smoke` 无 NaN、契约通过、checkpoint 可 `--resume`",
              "不使用任何 2.5+ 的 PyTorch API；注意力（若有）走 `F.scaled_dot_product_attention`"],
        forbid=["使用 ImageNet/自然图像预训练权重（分布无关，只会引入无关先验）",
                "引入 flash-attn / xformers 等需编译的 CUDA 扩展",
                "在 16 GiB 内存下把整井喂入模型"],
        risk=[("感受野不足", "序列模型与行级模型分数接近", "E3/P2 的感受野消融会暴露；先增大 depth/dilation 再谈扩容"),
              ("过拟合 80 井", "inner 高 outer 低、折间方差大", "dropout/stochastic depth/weight decay + E2 增强"),
              ("bf16 数值不稳", "loss 出现 NaN", "梯度裁剪 1.0 + 关键归一化层用 fp32（`autocast` 白名单）"),
              ("显存充足但内存爆", "阶段被杀", "chunk + `num_workers=4` + 不缓存分片")],
        stop=["单折超过软预算 3 倍 → 降 chunk 长度或减 base_ch",
              "连续 2 次 NaN → 回退 checkpoint 并减半 lr / 改 fp32 关键层"],
        code=["src/models/unet1d.py", "src/models/tcn.py", "src/models/heads.py",
              "E3/code/train_seq.py"],
        evidence=["`reports/E3_param_budget.json`"],
        prereg_extra={"primary_metric": "seq_train_ok",
                      "mandatory_checks": ["seq_train_ok", "disk_budget_ok",
                                           "checkpoint_resumable", "training_time_log_valid"]},
    ),
    dict(
        pid="P2", title="行级对照 + 感受野消融 + 硬 Gate（≥81.0）",
        nature="**判据阶段**：决定序列路线是否继续",
        deps=["E3/P1、E1/P1"],
        goal="同折同头对比序列主干与行级 MLP；做感受野消融（U-Net depth 3/5、TCN dilation 上限 64/512）；"
             "判定硬 Gate：OOF ≥ 81.0 **且** 序列主干显著优于同头行级模型。",
        why=["这是\"上下文是否真被利用\"的唯一判据，也是 E4/E5 是否值得继续的前提；",
             "若缩小感受野不降分，说明主干没学到长程结构，此时**扩容是浪费机时**，应先修数据/结构/损失；",
             "前代所有序列/井级尝试都是 NO-GO，因此本次必须给出比\"分数提高了\"更硬的证据："
             "同折、同头、同特征的受控对照 + 感受野消融。"],
        inputs=["E3/P1 的 5 折 OOF、E1/P1 的行级 OOF", "E0 的评分器与 bootstrap 工具"],
        outputs=["`$V4_RUN_ROOT/E3/oof.npz`（合并 5 折，逐行）",
                 "`$V4_REPORTS_DIR/E3_receptive_field_ablation.json`",
                 "`$V4_REPORTS_DIR/E3_row_vs_seq.json`",
                 "`$V4_REPORTS_DIR/E3_gate.json`"],
        steps=["预注册 `E3_P2_gate_prereg.json`（含 `min_delta`、`primary_metric`、`candidate_budget`）",
               "跑 U-Net 与 TCN 各 5 折（可并行任务拆分），汇总 OOF",
               "把**同一份行级头**装在行级特征上训练一遍（同折同超参），作为受控对照",
               "感受野消融：depth∈{3,5} × dilation_max∈{64,512}，各跑 fold0+1 筛查",
               "计算序列 vs 行级的 paired cluster bootstrap（按井行数加权，1000 次）",
               "逐折 delta、逐井非退化比例、占位行 Acc、连续切片 Acc 一并上报",
               "写 Gate 并判定"],
        params=[("主判据", "OOF Total ≥ 81.0", "硬 Gate", "低于此值不得进入 E4"),
                ("显著性判据", "序列 vs 行级 bootstrap CI 下界 > 0", "冻结", "受控对照必须有"),
                ("消融预算", "fold0+1 筛查，胜者再跑全 5 折", "冻结", "省机时且避免看全折后挑结构"),
                ("感受野变量", "depth {3,5} × dilation {64,512}", "冻结", "4 个组合")],
        done=["OOF Total **≥ 81.0**",
              "序列主干相对**同头同特征**行级模型 CI 下界 > 0（若为负，判 NO-GO 并记录证据）",
              "感受野消融表完整；若缩小感受野不降分，必须给出\"上下文未被利用\"的结论与修正计划",
              "5 折 delta 全部同向；`atomic_precision_reported` 与 `contract_ok` 为 true"],
        forbid=["跳过消融直接堆容量",
                "用 outer 折（含 fold0）验证标签选超参——fold0 只作资源筛查，"
                "其结论必须标 `selection_score_only`",
                "在未通过 CI 判据的情况下宣称\"序列有效\""],
        risk=[("序列不优于行级", "CI 含 0 或为负", "先查数据分块/覆盖、归一化、损失；最多两次结构修订后降级为 NO-GO"),
              ("分数高但来自容量而非上下文", "缩小感受野分数不降", "以感受野消融结论为准，判定上下文未被利用"),
              ("机时超支", "单折耗时持续增长", "先在 fold0+1 筛查，胜者才跑全折")],
        stop=["Gate 未过 → 触发\"最多两次结构修订\"规则；仍不过则记录 NO-GO 并把主线降级为"
              "行级 + 手工窗口特征，供 E5/E6 继续"],
        code=["E3/code/compare_row_vs_seq.py", "E3/code/rf_ablation.py", "E3/code/gate.py"],
        evidence=["`reports/E3_gate.json`、`reports/E3_row_vs_seq.json`"],
        prereg_extra={
            "primary_metric": "oof_total", "baseline_version": "E1_PD0",
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0, "oof_total_min": 81.0},
            "mde_units": 80, "multiplicity": "holm", "candidate_budget": 4,
            "mandatory_checks": ["contract_ok", "atomic_precision_reported", "disk_budget_ok",
                                 "training_time_log_valid", "no_label_leak"],
        },
    ),
]

# ------------------------------------------------------------------ E4
P["E4"] = [
    dict(
        pid="P0", title="Patch Transformer 主干（通道独立）",
        nature="第二主干候选",
        deps=["E3/P2（序列路线确认有效）"],
        goal="实现 PatchTST 式通道独立 Patch Transformer（patch=32/stride=16、d=256、6 层、8 头、"
             "相对位置编码），与 E3 的 CNN 主干在**同数据同折**下可比。",
        why=["`资料库/08` §0.3 第 3 层与 §7.5：把深度序列切成 patch 后做通道独立建模，"
             "是长序列的低成本高效方案（复杂度从 O(n²) 降到 O((n/P)²)）；",
             "CNN 擅长局部形态，注意力擅长长程依赖，两者互补——但必须先证明 Transformer 单体能打平/超过 CNN，"
             "才谈融合；",
             "`资料库/08` §1.2 指出整井 n≈10⁴ 时原始自注意力不可接受，patch 化是必要前提。"],
        inputs=["E3/P0 的 `SeqDataset`", "`资料库/08` §7.1/§7.3/§7.5/§7.6"],
        outputs=["`src/models/patchtf.py`",
                 "models/E4/patchtf_fold{k}.pt",
                 "`$V4_REPORTS_DIR/E4_patchtf.json`（与 E3 的对照结果）"],
        steps=["实现 patch 切分与线性投影（P=32, stride=16, d_model=256）",
               "实现通道独立：每条曲线单独作为 token 序列（共享权重），最后沿通道做聚合",
               "注意力用 PyTorch 2.4 原生 `F.scaled_dot_product_attention`（自动选择 Flash/Memory-Efficient/Math 后端）",
               "相对位置编码 + Pre-LN + 残差 + FFN(GELU)，dropout 0.1",
               "输出上采样回逐行长度（patched 输出按 stride overlap-add 还原）",
               "fold0+1 筛查，胜者跑全 5 折"],
        params=[("`patch_len`", "32", "16/32/64", "与 stride 联动"),
                ("`stride`", "16", "8/16/32", "overlap = patch−stride"),
                ("`d_model`", "256", "128/256/512", "显存充足"),
                ("`n_layers`", "6", "4/6/8", "同上"),
                ("`n_heads`", "8", "4/8", "d_model 必须整除"),
                ("通道独立", "是", "是/否", "NO 则退化为多头联合建模，作对照"),
                ("`dropout`", "0.1", "0.0/0.1/0.2", "80 井易过拟合")],
        done=["patch 还原后输出长度与输入严格一致（overlap-add 权重归一）",
              "与 E3 在同折同数据下可比（同 chunk、同特征、同头）",
              "注意力只使用 2.4 已有签名；无编译扩展依赖",
              "fold0+1 结果与资源记录完整"],
        forbid=["把曲线轴当图像轴做 2D 卷积（曲线轴相邻无物理含义，`资料库/08` §1.3）",
                "依赖 flash-attn/xformers",
                "用 2.5+ 的 `torch.nn.attention` API"],
        risk=[("patch 还原错位", "接缝处预测跳变、行数不符", "overlap-add 单测：常数输入应还原为常数"),
              ("通道独立后参数量爆炸", "显存/时间超预算", "共享通道权重；必要时减层"),
              ("注意力数值不稳", "NaN", "Pre-LN + 梯度裁剪 + bf16 关键层 fp32")],
        stop=["fold0+1 耗时超过 E3 单折的 3 倍且分数无优势 → 判 NO-GO，保留 CNN 主干"],
        code=["src/models/patchtf.py", "E4/code/train_patchtf.py"],
        evidence=["`reports/E4_patchtf.json`"],
        prereg_extra={"primary_metric": "target_acc", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 4, "multiplicity": "holm"},
    ),
    dict(
        pid="P1", title="多尺度融合（CNN × Transformer）与 Gate",
        nature="融合候选：必须超过最佳单主干才算增益",
        deps=["E4/P0、E3/P2"],
        goal="把 CNN 主干与 PatchTF 的逐行表示按门控或 concat 融合后送同一组头，"
             "判定是否超过**最佳单主干**；否则 NO-GO 并保留 E3 结构。",
        why=["多尺度是最常见的稳定增益来源，但必须证明超过最好单主干，否则只是参数变多；",
             "CNN 的局部形态与注意力的长程依赖在测井上确实互补（`资料库/08` §0.3 第 3–4 层）；",
             "融合层参数量小，是\"低成本换分\"的候选。"],
        inputs=["E3/P1 的 U-Net/TCN 权重与逐行表示", "E4/P0 的 PatchTF"],
        outputs=["`src/models/multiscale.py`",
                 "`$V4_RUN_ROOT/E4/oof.npz`",
                 "`$V4_REPORTS_DIR/E4_gate.json`"],
        steps=["实现三种融合：① 门控加权（可学习标量/向量门）；② concat + 1×1 卷积降维；③ 表示层平均",
               "冻结两个主干的预训练权重先做快速筛查（只训融合层与头）",
               "胜出方案再解冻联合微调（小 lr）",
               "与最佳单主干做同折 paired bootstrap",
               "写 Gate：融合 ≥ 最佳单主干且 CI 下界 > 0"],
        params=[("融合方式", "门控（默认）", "门控/concat/平均", "fold0+1 选"),
                ("融合层 lr", "1e-3（冻结主干）/ 2e-4（解冻）", "—", "解冻时用更小 lr"),
                ("候选数", "3", "—", "multiplicity=holm 校正")],
        done=["融合方案相对最佳单主干的 paired bootstrap CI 下界 > 0，否则判 NO-GO 并保留 E3",
              "同源性报告：两主干 OOF 预测的相关系数（过高说明融合收益可疑）",
              "参数量与耗时增量记录完整"],
        forbid=["用两个高度同源的分支冒充多尺度",
                "在未与单主干对照的情况下宣称融合有效"],
        risk=[("融合无增益", "CI 含 0", "判 NO-GO，保留 E3 主干；把预算让给 E5/E6"),
              ("同源性高", "两主干预测相关 > 0.99", "检查是否实现同一结构；若确实同源则融合无意义")],
        stop=["融合 CI 上界 ≤ 0 → NO-GO，E5 直接基于 E3 主干"],
        code=["src/models/multiscale.py", "E4/code/fuse_multiscale.py"],
        evidence=["`reports/E4_gate.json`"],
        prereg_extra={"primary_metric": "oof_total", "baseline_version": "E3_best",
                      "thresholds": {"min_delta": 0.0}, "candidate_budget": 3,
                      "multiplicity": "holm"},
    ),
]

# ------------------------------------------------------------------ E5
P["E5"] = [
    dict(
        pid="P0", title="POR 窄带精修（±0.008）",
        nature="单目标攻坚：POR（权重 30%，容差最窄）",
        deps=["E3/P2 或 E4/P1（冻结主干）"],
        goal="针对 POR 的极窄容差带（0.08×0.1 = ±0.008）设计专用头与训练策略，"
             "提升 POR 连续切片准确率且不牺牲其他目标。",
        why=["POR 权重 30% 但**容差带最窄**：占位 0.1 的允许误差只有 ±0.008，"
             "任何抖动都会掉出带外；",
             "POR 头从 `0.1 + softplus(g)` 起步可保证非负且离占位值近，减少初期震荡；",
             "`资料库/12` §3.3 指出 POR 全空间 9.71 分，是三个目标中上限最小的，"
             "因此策略应是\"守住占位 + 精修有效段\"而不是全面重构。"],
        inputs=["E3/E4 的冻结主干逐行表示", "E1/P1 的 POR 基线 OOF"],
        outputs=["`src/models/heads.py::PorHead`（精修版）",
                 "`$V4_RUN_ROOT/E5/por/oof.npz`",
                 "`$V4_REPORTS_DIR/E5_por.json`（连续切片 Acc、带内占比、逐折 delta）"],
        steps=["统计 POR 误差分布：落在 ±0.008 带内的比例、带外距离分布（定位问题在偏移还是方差）",
               "试验三种 POR 参数化：`0.1+softplus`、`sigmoid·0.5`、直接线性（作对照）",
               "试验误差加权：对接近带边界的样本加大权重（可微权重，不改标签）",
               "只在 inner-OOF 上选参数化与权重，outer 折只推理一次",
               "报告 POR 的**连续切片**（排除占位行）Acc 与占位行 Acc 的跷跷板效应"],
        params=[("POR 参数化", "`0.1 + softplus(g)`", "softplus/sigmoid/线性", "inner 选择"),
                ("带边加权", "关（默认）", "开/关 + 权重 2/5", "inner 选择"),
                ("δ（容差）", "0.08（官方）", "冻结", "不得改动"),
                ("候选数", "3–4", "—", "holm 校正")],
        done=["POR 连续切片 Acc 提升且 CI 下界 > 0",
              "PERM/SW 不退化超过 0.01（总分为准的跷跷板检查）",
              "POR 占位行 Acc 仍 ≥ 0.99"],
        forbid=["改动 POR 容差或评分权重",
                "用全局裁剪把 POR 压到 [0,0.4]（会破坏占位值 0.1 之外的物理含义）",
                "为提升 POR 而牺牲 PERM/SW"],
        risk=[("带边加权导致占位过拟合", "占位 Acc 上升但有效段下降", "报告两个切片并做跷跷板检查"),
              ("POR 头震荡", "连续段预测呈锯齿", "提高 λ_align 中 POR 的有效样本权重；检查 BN 统计")],
        stop=["POR 连续切片连续 3 次无正增量 → 该方向停止，转 PERM/SW"],
        code=["E5/code/head_por.py"],
        evidence=["`reports/E5_por.json`"],
        prereg_extra={"primary_metric": "por_acc", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 4, "multiplicity": "holm"},
    ),
    dict(
        pid="P1", title="PERM log 域精修（长尾与数量级）",
        nature="单目标攻坚：PERM（权重 35%，边际收益最高）",
        deps=["E5/P0"],
        goal="在 log10 域精修 PERM：处理长尾与数量级误差，输出经 tanh 夹到 [-6,6] 保证正有限，"
             "提升 PERM 连续切片准确率。",
        why=["PERM 权重 35%、历史探索最少、边际收益最高（`资料库/12` §3.3 排序 PERM > SW ≈ POR）；",
             "评分是 `|log10(ŷ/y)|`，相差 10 倍即得 0——**必须在数量级上正确**，绝对误差无意义；",
             "`资料库/13` 指出测井预测渗透率\"落在真值两倍内已算很好\"，因此目标是量级正确而非过拟合 RMSE。"],
        inputs=["E3/E4 冻结主干表示", "E1/P1 的 PERM 基线 OOF"],
        outputs=["`E5/code/head_perm.py`", "`$V4_RUN_ROOT/E5/perm/oof.npz`",
                 "`$V4_REPORTS_DIR/E5_perm.json`"],
        steps=["分析 z 空间误差分布：σ(z)、落在 |Δz|<1 的比例、长尾方向（低估/高估）",
               "试验量化分桶辅助损失（把 z 分箱做 soft 分类，再求期望）与纯回归对照",
               "实现分位数/异方差辅助头（`资料库/09` §4）以改善尾部",
               "确保输出 `10^clip(z)` 严格 > 0 且有限（契约层会二次校验）",
               "只在 inner-OOF 上选方案"],
        params=[("z 输出", "`6·tanh(g)`", "tanh/clip/线性", "tanh 保证有界"),
                ("辅助损失", "Smooth L1（默认）", "Smooth L1 / 分桶 soft-CE / 分位数", "inner 选择"),
                ("clip 范围", "[-6, 6]", "冻结", "`constants.PERM_LOG_MIN/MAX`"),
                ("候选数", "3–5", "—", "holm 校正")],
        done=["PERM 连续切片 Acc 提升且 CI 下界 > 0",
              "无 ≤0 或非有限输出（契约自动校验）",
              "z 空间误差分布改善（σ(z) 或尾部比例）有数据支撑"],
        forbid=["线性域建模 PERM", "用 ReLU 输出 PERM（0 处零梯度）", "改动 PERM 的评分公式或权重"],
        risk=[("长尾被平均掩盖", "整体 Acc 微升但尾部更差", "分位数报告：按真值分箱统计 Acc"),
              ("分桶边界引入偏差", "分桶方案的 OOF 不稳定", "分桶边界只在训练折确定并冻结")],
        stop=["PERM 连续切片连续 3 次无正增量 → 转 SW"],
        code=["E5/code/head_perm.py"],
        evidence=["`reports/E5_perm.json`"],
        prereg_extra={"primary_metric": "perm_acc", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 5, "multiplicity": "holm"},
    ),
    dict(
        pid="P2", title="SW 单尺度精修（占位 99.9 与有效 8.3–99.9 同尺度）",
        nature="单目标攻坚：SW（权重 35%，结构最特殊）",
        deps=["E5/P1"],
        goal="实现 SW 的占位/有效双分支混合输出（`q̂·99.9 + (1−q̂)·f_valid`，**同一标签尺度**），"
             "用 BCE 监督占位分支，并验证不引入任何尺度换算；提升 SW 连续切片准确率。",
        why=["占位峰（99.9）与有效峰（实测 8.3–99.9，中位 82.8）**同尺度但分布形状完全不同**，"
             "单头线性回归仍会被占位尖峰拉扯（`资料库/12` §3.4 的双峰会震荡结论在结构上成立）；",
             "**E0-R2 修正**：SW 不是 `[0,1]` 双尺度（审查 B2 实测 SW<1 仅 11 行）——"
             "因此**禁止**任何 ×100 换算；`constants.SW_SMALL_BRANCH=False`，"
             "有效分支直接用标签尺度监督；",
             "`资料库/12` §3.4 指出在 `q̂` 灰色地带向 99.9 偏移可换期望分——这是该指标允许的\"下注\"。"],
        inputs=["E3/E4 冻结主干表示", "E1/P1 的 SW 基线 OOF"],
        outputs=["`E5/code/head_sw.py`", "`$V4_RUN_ROOT/E5/sw/oof.npz`",
                 "`$V4_REPORTS_DIR/E5_sw.json`"],
        steps=["实现双分支：`q̂=sigmoid(g0)`（可与 H0 共享或独立）、`f` 为**标签尺度**的有效分支输出、"
               "混合 `q̂·99.9 + (1−q̂)·f`",
               "单测锁定尺度：`q̂=1` 时输出必须精确 99.9；`q̂=0` 时输出等于 `f`（**不做任何倍数换算**）；"
               "并断言 `constants.SW_SMALL_BRANCH is False`",
               "试验灰色地带偏移策略（在 inner-OOF 上选阈值）",
               "报告 SW 两个切片的 Acc：占位行、有效行（实测 8.3–99.9）",
               "确认**绝不做全局 [0,1] 裁剪**（会掉约 23 分）"],
        params=[("有效分支输出", "标签尺度（**不乘 100**）", "冻结", "`constants.SW_SMALL_BRANCH=False`"),
                ("占位分支", "常数 99.9", "冻结", "不参与梯度（只作为混合常量）"),
                ("灰色地带阈值", "0.3–0.5 内选", "inner 选择", "向 99.9 下注"),
                ("候选数", "3–4", "—", "holm 校正")],
        done=["尺度单测通过（`q̂=1 → 99.9`；`q̂=0 → 输出等于有效分支，无倍数换算）",
              "SW 有效行与占位行的 Acc 均报告；有效行 Acc 提升且 CI 下界 > 0",
              "契约校验通过：SW 输出不被裁剪，且与标签尺度一致",
              "占位行 SW Acc ≥ 0.99"],
        forbid=["全局裁剪 SW 到 [0,1]（`资料库/12` §0 结论 2：直接损失约 23 分）",
                "对有效分支做 ×100 换算（E0-R2 已证伪双尺度假设）",
                "用占位行样本训练有效分支"],
        risk=[("误用双尺度换算", "SW 有效段预测整体偏大 100 倍", "单测断言 SW_SMALL_BRANCH=False + 契约层范围检查"),
              ("双分支失衡", "占位/有效一侧塌陷", "分别报告两切片 Acc；调整 λ₂"),
              ("灰色地带过拟合", "inner 提升 outer 下降", "阈值只在 inner 选并报告敏感性")],
        stop=["SW 有效行连续 3 次无正增量 → 该方向停止，转 E6"],
        code=["E5/code/head_sw.py"],
        evidence=["`reports/E5_sw.json`"],
        prereg_extra={"primary_metric": "sw_acc", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 4, "multiplicity": "holm",
                      "mandatory_checks": ["sw_scale_unit_test", "contract_ok"]},
    ),
]

# ------------------------------------------------------------------ E6
P["E6"] = [
    dict(
        pid="P0", title="联合常量状态头（H0）",
        nature="保护屏障：66.7% 的白送分靠它守住",
        deps=["E5/P2（三个目标头定型）"],
        goal="训练联合占位状态分类头（BCE 监督 `POR=0.1 ∧ PERM=0.01 ∧ SW=99.9`），"
             "评估 AUC 与逐目标原子 precision/recall/F1。",
        why=["487,225 行（66.719%）是联合常量占位，占约 66.72 分的白送分；",
             "v1 的原子门已被证明可从输入预测（E7 的 +0.2015 主要来自此），因此应把"
             "\"是否输出常量\"做成**显式可学习决策**，而不是让回归头勉强逼近；",
             "本项目不做 B0 patch 隔离（用户决策 D3），H0 是**唯一**的占位保护屏障，"
             "因此它的质量直接决定管线是否安全。"],
        inputs=["E3/E4 主干逐行表示或 E1 行级特征", "E0 的占位标签"],
        outputs=["`src/models/state_head.py`、`$V4_RUN_ROOT/E6/state/{foldk}.pt`",
                 "`$V4_REPORTS_DIR/E6_atomic_report.json`（AUC/PR 曲线/逐目标原子指标）"],
        steps=["实现 H0：`Linear(d→1)`，可用\"逐行 + 井内平均池化\"拼接增强井级信息",
               "用 BCE 训练（占位/有效/缺测三类的处理：缺测行不参与）",
               "报告 AUC 与 PR-AUC（占位类不平衡，PR 更重要）",
               "报告逐目标原子 precision/recall/F1（在 τ=0.5 与最优 τ 两处）",
               "做 label-shuffle 阴性对照，确认 AUC 不是来自泄漏"],
        params=[("H0 输入", "逐行表示（默认）", "逐行/逐行+井级池化", "inner 选择"),
                ("正负样本", "全量（占位 66.7%）", "全量/过采样有效", "过采样需消融"),
                ("`pos_weight`", "1.0", "1.0/1.5/2.0", "inner 选择")],
        done=["AUC ≥ 0.97 且 PR-AUC 报告完整",
              "label-shuffle 对照下 AUC ≈ 0.5（证明非泄漏）",
              "逐目标原子 precision/recall/F1 全部上报（`atomic_precision_reported`）"],
        forbid=["用测试集或验证折标签训练 H0",
                "把 H0 当作\"裁剪器\"直接覆盖回归输出而不经 τ 判定"],
        risk=[("H0 学不到占位", "AUC < 0.9", "检查特征是否包含足够区分信息；加井级池化"),
              ("H0 过拟合", "inner AUC 高 outer 低", "减容量 + dropout + 折内早停")],
        stop=["AUC < 0.9 且无改善 → 记录 NO-GO，改用固定常量策略并重新评估总分上限"],
        code=["src/models/state_head.py", "E6/code/train_state.py"],
        evidence=["`reports/E6_atomic_report.json`"],
        prereg_extra={"primary_metric": "state_auc", "thresholds": {"min_auc": 0.97},
                      "mandatory_checks": ["atomic_precision_reported", "no_label_leak"]},
    ),
    dict(
        pid="P1", title="原子门 τ 搜索（inner-OOF，硬切换）",
        nature="开关设定：决定\"输出常量还是连续\"",
        deps=["E6/P0"],
        goal="在 inner-OOF 上逐目标搜索门限 τ_t，实现**硬切换**解码，并报告误判代价分解。",
        why=["τ 决定每个点是走常量分支还是连续分支，是纯 DL 管线唯一保护屏障的开关；",
             "误判代价不对称：把有效行判成常量会立刻丢分，把占位行判成连续同样丢分，"
             "两者代价需分别量化；",
             "**禁止在常量与连续之间线性插值**：POR 容差仅 ±0.008，插值必然出带。"],
        inputs=["E6/P0 的 q 概率（inner-OOF）", "E3–E5 的连续预测（inner-OOF）"],
        outputs=["`src/inference/atomic_gate.py`", "`$V4_REPORTS_DIR/E6_tau_search.json`",
                 "三个 τ 值写入 `versions/candidates.json::PD1.atomic.tau`"],
        steps=["对每个目标，在 inner-OOF 上网格搜索 τ ∈ [0.05,0.95]（步长 0.01）",
               "目标函数 = 该目标的官方 Acc（drop 口径）",
               "报告：τ 曲线、最优 τ、误判代价分解（FP 代价 vs FN 代价）",
               "验证 τ 的稳定性：不同 inner 折选出的 τ 是否接近（方差过大则不可靠）",
               "在三目标上分别确定 τ，并记录到候选注册表"],
        params=[("τ 搜索范围", "[0.05, 0.95]，步长 0.01", "冻结", "逐目标独立"),
                ("目标函数", "该目标官方 Acc", "冻结", "不是 F1"),
                ("硬切换", "q ≥ τ → 输出精确常量", "冻结", "禁止插值"),
                ("稳定性判据", "不同 inner 折最优 τ 的极差 ≤ 0.2", "冻结", "超限则用更保守 τ")],
        done=["三个 τ 都只在 inner-OOF 上选出，过程可复算",
              "误判代价分解表完整",
              "τ 稳定性通过（跨 inner 折极差 ≤ 0.2），否则取更保守值并说明",
              "占位行逐目标 Acc ≥ 0.99（这是 Gate 硬条件）"],
        forbid=["用 outer 折或 A 榜选 τ",
                "在常量与连续输出之间做线性插值",
                "用 F1 而非官方 Acc 作为 τ 的目标函数"],
        risk=[("τ 过拟合 inner", "inner 最优但 outer 变差", "报告 τ 敏感性曲线；取平坦区间的中点"),
              ("误判代价不对称被忽视", "总分下降但 F1 上升", "以官方 Acc 为目标函数")],
        stop=["τ 搜索若无法让占位 Acc ≥ 0.99 → 回到 E6/P0 加强 H0"],
        code=["src/inference/atomic_gate.py", "E6/code/search_tau.py"],
        evidence=["`reports/E6_tau_search.json`"],
        prereg_extra={"primary_metric": "atomic_f1", "thresholds": {"min_atomic_acc": 0.99},
                      "mandatory_checks": ["atomic_precision_reported", "inner_only_selection"]},
    ),
    dict(
        pid="P2", title="PD1 完整管线组装与硬 Gate（≥82.0）",
        nature="**完整管线诞生**：本计划第一个可提交候选",
        deps=["E6/P0–P1、E5/P2"],
        goal="组装 数据→主干→头→原子门→契约 的完整 PD1 管线，产出 5 折 OOF、测试集 `result.json`/`result.zip`、"
             "manifest 与 cv 报告；**硬 Gate：OOF Total ≥ 82.0**。",
        why=["这是 v4 第一个\"端到端可跑、可提交、可复现\"的候选；",
             "≥82.0 意味着超过历史锚点 B0 的本地 OOF 80.382479，是纯 DL 路线成立的最低证据；",
             "只有完整管线才能暴露\"训练能跑但推理契约不过\"这类问题（前代多次踩坑）。"],
        inputs=["E3/E4 冻结主干权重", "E5 的三个目标头", "E6 的 H0 与 τ", "E0 的契约与评分器"],
        outputs=["`models/E6/pd1_fold{k}.pt` + `models/E6/pd1_config.json`",
                 "`experiments/E6/P2/pd1/{oof.npz,cv.json,result.json,result.zip,manifest.json}`",
                 "`$V4_REPORTS_DIR/E6_gate.json`",
                 "`versions/candidates.json::PD1`（status=local_only→shortlisted）"],
        steps=["实现统一推理器 `src/inference/predictor.py`：加载配置与权重 → 逐井前向 → 原子门 → 解码",
               "在 5 折上各自推理出 OOF（训练时已产出，此处复核逐行对齐）",
               "对 10 口测试井推理：平均 5 折权重（或按核验过的最优折），产出 result.json",
               "跑契约校验（10 井 / 95,948 行 / depth 对齐 / PERM>0 / 无 NaN）",
               "本机 CPU 冒烟 `predict.py --use-version PD1 --data_dir ../data --output /tmp/r.json`",
               "汇总 OOF 评分：逐目标 Acc、连续切片、占位 Acc、bootstrap CI",
               "写 manifest（config 哈希/数据指纹/折指纹/代码哈希）并注册候选",
               "写 Gate 并判定 ≥ 82.0"],
        params=[("折权重聚合", "5 折平均", "平均/最优折/加权", "inner 决定，冻结后不改"),
                ("τ", "E6/P1 选定值", "冻结", "写入 candidate registry"),
                ("推理精度", "fp32（CPU）", "fp32/fp16", "提交侧必须 fp32 保证确定性"),
                ("`num_folds`", "5", "冻结", "与 folds.json 一致")],
        done=["OOF Total **≥ 82.0**（硬 Gate）",
              "契约全绿；`predict.py --use-version PD1` 在本机 CPU 可跑通并输出 95,948 行",
              "占位行逐目标 Acc ≥ 0.99；连续切片 Acc 一并上报",
              "manifest 写全 config/data/folds/code 四类指纹；候选已注册",
              "5 折 delta 全部同向；`disk_budget_ok`、`training_time_log_valid`、`checkpoint_resumable` 为 true"],
        forbid=["在管线中混入未冻结的特征版本",
                "推理阶段读取任何标签",
                "把 5 折权重聚合方式在看到 OOF 后临时更换"],
        risk=[("训练能跑但推理契约不过", "result.json 行数/字段错", "契约前置到训练脚本每次落盘时校验"),
              ("低于 82.0", "纯 DL 未超过树模型锚点", "按总计划 §9.5 回退协议准备 B0 fallback；"
               "同时保留 PD1 为 `local_only` 候选供 E7/E8 继续改进"),
              ("折间差异大", "逐折 delta 方向不一致", "检查折内标准化与早停；报告逐折而非只报总分")],
        stop=["Gate < 82.0 → 冻结当前最强候选为 `PD-pre`，E7/E8 继续改进；"
              "若 E8 结束仍 < 82.0，E10 走回退协议"],
        code=["E6/code/build_pd1.py", "E6/code/gate.py", "src/inference/predictor.py",
              "src/inference/atomic_gate.py"],
        evidence=["`reports/E6_gate.json`、`reports/E6_atomic_report.json`"],
        prereg_extra={
            "primary_metric": "oof_total", "baseline_version": "E1_PD0",
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0, "oof_total_min": 82.0},
            "mde_units": 80,
            "mandatory_checks": ["contract_ok", "atomic_precision_reported", "disk_budget_ok",
                                 "training_time_log_valid", "checkpoint_resumable", "cpu_inference_ok"],
        },
    ),
]

# ------------------------------------------------------------------ E7
P["E7"] = [
    dict(
        pid="P0", title="三段式损失消融与退火策略",
        nature="损失配方冻结",
        deps=["E6/P2（PD1 基线）"],
        goal="对 `L_align / L_aux / L_ph` 三组权重与 `λ₁` 退火曲线做完整消融（同结构对照），"
             "确定唯一损失配方并冻结。",
        why=["`资料库/12` §2.3 指出纯对齐损失早期梯度稀疏（大量点落在容忍域外，梯度≈0），"
             "必须靠 aux 提供早期梯度；",
             "评分 `max(0,·)` 截断意味着超过容差阈值的点不再产生梯度收益，"
             "把容量让给\"临界点\"是理论最优——这只能通过损失权重实现；",
             "配方必须消融确定，不能凭感觉设 λ。"],
        inputs=["E6/P2 的 PD1 管线（结构冻结）", "`资料库/12` §2.2–2.4"],
        outputs=["`src/losses/score_aligned.py`（最终配方）",
                 "`$V4_REPORTS_DIR/E7_loss_ablation.json`",
                 "`versions/configs/loss_v1.json`（冻结配置）"],
        steps=["对照实验 1：纯 align vs 纯 aux vs align+aux",
               "对照实验 2：λ₁ ∈ {0.1,0.3,1.0} × 退火曲线 ∈ {常数, 线性到 0.1, 余弦}",
               "对照实验 3：λ₂ ∈ {0.1,0.2,0.3} 对占位 Acc 的影响",
               "对照实验 4（可选）：加 `L_phys`（物理软约束）并消融其 λ₃",
               "所有对照在 fold0+1 上做，胜者跑全 5 折确认",
               "用**真实评分**而非 loss 值选择配方"],
        params=[("λ₁（aux）", "1.0 → 0.1（线性，前 60% epoch）", "见对照 2", "inner/fold0+1 选择"),
                ("λ₂（占位 BCE）", "0.2", "0.1/0.2/0.3", "同上"),
                ("λ₃（物理）", "0（默认关）", "0/0.02/0.05", "必须消融；`资料库/03` 提醒 KC 只能定性"),
                ("`alpha`/`beta`", "1e-3 / 20", "1e-3~1e-2 / 10~30", "平滑参数"),
                ("`huber_beta`", "1.0", "0.5/1.0/2.0", "aux 损失")],
        done=["对齐损失 ≥ 纯 aux 损失（同结构对照，CI 下界 > 0）",
              "三段式权重的完整消融表（含退火曲线）",
              "最终配方写入 `versions/configs/loss_v1.json` 并冻结",
              "每个对照都能复算（脚本 + 命令 + 产物 sha256）"],
        forbid=["同时改多个损失项导致无法归因",
                "用 loss 值而非真实评分选配方",
                "在看到 outer 折结果后调整 λ"],
        risk=[("对照实验爆炸", "组合数过多", "分层做：先定性（哪一项有用），再定量（λ 搜 3 档）"),
              ("物理损失引入偏差", "POR/SW 变好但 PERM 变差", "λ₃ 只在所有目标都不退化时才采纳")],
        stop=["若 align 与 aux 无差异，保留简单配方（align+aux+ph 默认值）并记录"],
        code=["E7/code/ablate_loss.py"],
        evidence=["`reports/E7_loss_ablation.json`、`versions/configs/loss_v1.json`"],
        prereg_extra={"primary_metric": "oof_total", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 8, "multiplicity": "holm"},
    ),
    dict(
        pid="P1", title="解码与后处理（温度/偏置/收缩）",
        nature="零模型改动的换分手段",
        deps=["E7/P0"],
        goal="实现并选择推理期解码手段：逐目标偏置校正、温度/收缩、分位数收缩，"
             "全部在 inner-OOF 上选定并冻结。",
        why=["解码在**不重训模型**的前提下换分，成本最低、风险最小；",
             "评分对每个目标有独立的最优\"保守/激进\"倾向（例如在容忍带边界附近，"
             "向众数偏移可提高期望分）；",
             "但解码参数极易过拟合 inner，因此必须做敏感性分析并只取平坦区间。"],
        inputs=["E6/P2 的 PD1 逐行预测（inner-OOF）", "E0 的评分器"],
        outputs=["`src/inference/decode.py`", "`$V4_REPORTS_DIR/E7_decode_search.json`",
                 "`versions/configs/decode_v1.json`"],
        steps=["实现逐目标偏置 `ŷ ← ŷ + b_t`，在 inner-OOF 上搜索 b_t（小范围）",
               "实现收缩 `ŷ ← μ_t + α_t(ŷ − μ_t)`，搜索 α_t ∈ [0.9, 1.1]",
               "实现分位数收缩（`资料库/09` §4 思路）：把预测往训练折分位数靠拢",
               "做参数敏感性热图，只采纳平坦区中点",
               "冻结 `decode_v1.json` 并复算 OOF 确认增益",
               "验证解码不破坏占位行的精确输出（原子门在解码之后仍生效）"],
        params=[("偏置 b_t", "0（默认）", "±0.005（POR）/ ±0.02（SW）/ ±0.05（z）", "inner 选择"),
                ("收缩 α_t", "1.0", "[0.9, 1.1]", "inner 选择"),
                ("分位数收缩", "关", "开/关 + 目标分位", "inner 选择"),
                ("敏感性判据", "最优邻域 ±1 档内 Acc 变化 < 0.005", "冻结", "否则不采纳")],
        done=["解码增益在 inner-OOF 上可复算，且 CI 下界 > 0",
              "敏感性热图显示所选参数位于平坦区",
              "占位行仍精确输出常量（原子门优先级高于解码）",
              "`decode_v1.json` 冻结并被 PD1 管线读取"],
        forbid=["用 outer 折或 A 榜选解码参数",
                "让解码覆盖原子门的常量输出",
                "采纳落在敏感性尖峰上的参数"],
        risk=[("过拟合 inner", "inner 提升 outer 下降", "只取平坦区；报告内外一致性"),
              ("解码破坏原子精确性", "占位 Acc 下降", "解码在原子门之前/之后的位置做单测固定")],
        stop=["若解码增益 CI 含 0，判 NO-GO 并保持恒等解码"],
        code=["E7/code/decode_search.py", "src/inference/decode.py"],
        evidence=["`reports/E7_decode_search.json`"],
        prereg_extra={"primary_metric": "oof_total", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 6, "multiplicity": "holm",
                      "mandatory_checks": ["inner_only_selection", "atomic_precision_reported"]},
    ),
]

# ------------------------------------------------------------------ E8
P["E8"] = [
    dict(
        pid="P0", title="MMoE 任务平衡",
        nature="多任务结构候选",
        deps=["E7/P1（损失与解码冻结）"],
        goal="用 MMoE 替换硬共享主干，比较逐目标 Acc 与梯度冲突指标，判定是否优于硬共享。",
        why=["`资料库/08` §1.4：多任务硬共享通常优于三个独立模型，但**必须解决权重失衡**；"
             "MMoE 允许任务部分共享，在任务相关性弱时比硬共享更稳；",
             "POR/PERM/SW 由同一套岩石物理关系耦合（Archie/Kozeny–Carman/Wyllie），"
             "共享表示相当于额外归纳偏置与隐式增强；",
             "但三个目标的**扰动敏感性**不同（POR 容差 ±0.008 极窄），硬共享可能让梯度互相干扰。"],
        inputs=["E6/P2 的冻结结构与特征", "`资料库/08` §1.4（MMoE 公式）"],
        outputs=["`src/models/mmoe.py`", "`$V4_RUN_ROOT/E8/mmoe/oof.npz`",
                 "`$V4_REPORTS_DIR/E8_mmoe.json`（逐目标 + 梯度冲突指标）"],
        steps=["实现 MMoE：E 个专家 + 每任务独立门控 softmax 加权",
               "对照：硬共享（E6 结构）vs MMoE（E=4）vs 完全独立三模型",
               "计算梯度冲突指标（任务间梯度余弦相似度）与逐目标 Acc",
               "判定：至少一个目标提升且无目标退化",
               "胜者跑全 5 折并汇总 OOF"],
        params=[("专家数 E", "4", "2/4/8", "fold0+1 选择"),
                ("专家容量", "与硬共享主干同宽", "—", "保证参数量可比"),
                ("门控温度", "1.0", "0.5/1.0/2.0", "影响路由锐度")],
        done=["逐目标 Acc 表完整（不是只报总分）",
              "MMoE 与硬共享的 CI 对比明确（采纳或 NO-GO）",
              "梯度冲突指标有数据支撑结论"],
        forbid=["只报告总分而隐藏单目标退化",
                "在参数量差异巨大的情况下比较两种结构"],
        risk=[("参数量不对等", "结论不可比", "固定总参数量，只改共享结构"),
              ("门控塌陷", "所有任务走同一专家", "报告门控熵；熵过低则加负载均衡损失")],
        stop=["MMoE 无正增量 → 保留硬共享，记录 NO-GO"],
        code=["src/models/mmoe.py", "E8/code/train_mmoe.py"],
        evidence=["`reports/E8_mmoe.json`"],
        prereg_extra={"primary_metric": "oof_total", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 3, "multiplicity": "holm"},
    ),
    dict(
        pid="P1", title="井级分支与 transductive 消融（各一次）",
        nature="历史 NO-GO 路线在新条件下的受控重验",
        deps=["E8/P0"],
        goal="实现 H4 井级 attention-pool 偏置分支；实现推理期 transductive 适配（伪标签/井级统计对齐）；"
             "**两者都只作消融**，给出明确的采纳/NO-GO 结论。",
        why=["工程曲线（CAL/DEVI/AZIM/BIT/CASE）在井内近常数，只提供**井间**区分度"
             "（`资料库/08` §0.1-4），井级分支是唯一合法的井间信号通路；",
             "v1 E8–E11 与 v2 E6 的井级/域适应均为 NO-GO，但那些结论是在**无序列主干、CPU-only**"
             "条件下取得的；E8 在已有序列主干的前提下只重验一次；",
             "transductive 适配需要使用测试井的**输入分布**（合法，标签不可见），"
             "但必须与实际提升严格区分，不能把\"用了测试输入\"包装成\"训练改进\"。"],
        inputs=["E6/E7 冻结管线", "`资料库/04` §九（地理因素多数不可获得）、`资料库/09` §10"],
        outputs=["`src/models/well_head.py`、`E8/code/well_branch.py`、`E8/code/pseudo_label.py`",
                 "`$V4_REPORTS_DIR/E8_well_branch.json`、`$V4_REPORTS_DIR/E8_transductive.json`"],
        steps=["实现 H4：主干输出做井级 attention-pool → 井向量 → 预测逐目标井级偏置 Δ_t"
               "→ `ŷ + λ·Δ_t`（λ 由 inner-OOF 选）",
               "消融井级分支：开/关，报告逐目标与总分 delta",
               "实现 transductive 适配：用测试井输入做特征分布对齐（如逐井分位数映射），"
               "**禁止使用任何标签**",
               "消融 transductive：开/关，明确标注\"该增益来自推理期使用了测试输入分布\"",
               "两个方向各自给出采纳/NO-GO 与 CI"],
        params=[("井级偏置 λ", "0（默认关）", "inner 搜索 [0, 0.5]", "选中后冻结"),
                ("池化方式", "attention-pool", "mean/max/attention", "inner 选择"),
                ("transductive 方式", "逐井分位数映射", "无/分位数映射/井均值对齐", "仅消融"),
                ("候选数", "各 2–3", "—", "holm 校正")],
        done=["两个方向都给出明确结论（采纳或 NO-GO）+ CI",
              "transductive 结论中显式标注其合法性与局限（使用了测试输入分布）",
              "井级分支若采纳，必须证明不是井身份泄漏（无 `logId` 特征、无逐井拟合标签）"],
        forbid=["使用测试集标签（不存在，任何形式的伪标签都必须来自训练折模型输出）",
                "把 transductive 适配说成\"训练时改进\"",
                "用井身份作为特征"],
        risk=[("井级偏置过拟合 80 井", "inner 提升 outer 下降", "λ 小范围搜索 + 只取平坦区；报告逐井非退化比例"),
              ("transductive 引入分布假设错误", "A 榜崩坏", "只对 top-2 候选做，并做 16 井体检")],
        stop=["任一方向 CI 含 0 → 判 NO-GO，记录证据（这是 v1/v2 同类路线的第二次受控重验）"],
        code=["src/models/well_head.py", "E8/code/well_branch.py", "E8/code/pseudo_label.py"],
        evidence=["`reports/E8_well_branch.json`、`reports/E8_transductive.json`"],
        prereg_extra={"primary_metric": "oof_total", "thresholds": {"min_delta": 0.0},
                      "candidate_budget": 6, "multiplicity": "holm",
                      "mandatory_checks": ["no_label_leak", "inner_only_selection"]},
    ),
    dict(
        pid="P2", title="集成（多 seed / 快照 / 多结构）与 Gate",
        nature="增益放大：必须扣除同源性",
        deps=["E8/P0–P1"],
        goal="构建多 seed、快照集成与多结构（U-Net/TCN/PatchTF）集成，报告成员同源性与"
             "按井行数加权的 paired bootstrap，判定集成是否真增益。",
        why=["`资料库/08` §0.3 第 4 层：多模型 Stacking/加权融合 + 快照集成是标准提分手段；",
             "**同源平均不构成增益**：若成员间预测相关 > 0.99，融合只是降低方差而非提升上限；",
             "集成的收益必须用统计检验而非点估计确认。"],
        inputs=["E3/E4/E5/E6/E7 的冻结成员", "E0 的 bootstrap 工具"],
        outputs=["`src/ensemble/blend.py`、`E8/code/ensemble.py`",
                 "`$V4_RUN_ROOT/E8/ensemble/oof.npz`",
                 "`$V4_REPORTS_DIR/E8_ensemble_report.json`、`$V4_REPORTS_DIR/E8_gate.json`"],
        steps=["枚举可用成员（多 seed 权重、不同主干的权重、快照 checkpoint）",
               "计算成员间 OOF 预测相关矩阵，标记同源簇",
               "实现三种融合：平均、加权（inner-OOF 选权）、线性 stacking（inner-OOF 训）",
               "与最佳单成员做 paired bootstrap（按井行数加权，1000 次）",
               "报告：集成 OOF、最佳单成员 OOF、delta、CI、逐折方向",
               "写 Gate：集成 ≥ 最佳单成员 且 CI 下界 > 0"],
        params=[("成员数上限", "5", "3–5（内存受限时 2）", "总计划 §3.4.1 收缩规则"),
                ("融合权重", "inner-OOF 选择", "平均/加权/stacking", "禁止用 outer 选"),
                ("同源判据", "相关系数 > 0.99 视为同源", "冻结", "同源成员不计入增益证据")],
        done=["集成 ≥ 最佳单成员 且 CI 下界 > 0",
              "成员同源性矩阵与同源簇标注完整",
              "5 折 delta 方向一致（至少 4/5）",
              "`E8_gate.json` 全 mandatory 通过"],
        forbid=["用同源模型平均制造假增益",
                "用 outer 折选融合权重",
                "成员数超过磁盘/内存可承受范围"],
        risk=[("集成无增益", "CI 含 0", "判 NO-GO，保留最佳单成员作为最终候选"),
              ("stacking 过拟合 inner", "inner 好 outer 差", "限制 stacking 自由度（只用线性 + 强正则）")],
        stop=["集成 CI 上界 ≤ 0 → 保留最佳单成员，记录 NO-GO"],
        code=["src/ensemble/blend.py", "E8/code/ensemble.py"],
        evidence=["`reports/E8_ensemble_report.json`、`reports/E8_gate.json`"],
        prereg_extra={"primary_metric": "oof_total", "baseline_version": "E6_PD1",
                      "thresholds": {"min_delta": 0.0}, "candidate_budget": 3,
                      "multiplicity": "holm",
                      "mandatory_checks": ["contract_ok", "atomic_precision_reported",
                                           "disk_budget_ok", "training_time_log_valid"]},
    ),
]

# ------------------------------------------------------------------ E9
P["E9"] = [
    dict(
        pid="P0", title="OOF 汇总与提交护栏判定",
        nature="验证：不出新模型",
        deps=["E8/P2"],
        goal="汇总全部候选的 80 井 OOF、逐目标 Acc、bootstrap CI，运行 `choose_submission.py` 护栏，"
             "输出可提交短名单与决策记录。",
        why=["护栏防止\"本地漂亮但明显弱于历史锚点\"的候选被提交；",
             "本地 80 井 OOF 是**主判据**，必须与历史锚点同折可比（同一 `folds.json`）；",
             "所有本地分数必须标 `selection_score_only=true`——因为它参与了折内选择。"],
        inputs=["各候选的 `oof.npz` 与 `cv.json`", "`constants.B0_LOCAL_OOF/B0_A_BOARD/GUARDRAIL_*`"],
        outputs=["`$V4_REPORTS_DIR/E9_validation_report.json`（全部候选对比表）",
                 "`$V4_REPORTS_DIR/E9_submission_decision.json`",
                 "`versions/candidates.json`（status 更新）"],
        steps=["汇总每个候选的 OOF：总分、逐目标、连续切片、占位行、逐折 delta、bootstrap CI",
               "计算护栏下限 `guardrail_floor = max(75.0, B0_LOCAL_OOF − 1.0, B0_A_BOARD − 0.5)`",
               "对每个候选判定：OOF ≥ guardrail_floor 且（若 A 榜已知）A ≥ B0_A_BOARD − 0.5",
               "输出短名单（≤3）与每个候选的 `choice/reason/candidate_oof/guardrail_floor`",
               "把 `selection_score_only=true` 与口径 `missing_mode=drop` 写入报告头部"],
        params=[("`guardrail_floor`", "max(75, 80.382479−1.0, 82.2757−0.5)", "冻结", "常量在 `constants.py`"),
                ("`protocol_matched`", "false", "冻结", "v1 同折 parity 不可复算，仅外部参照"),
                ("bootstrap", "按井行数加权 cluster，1000 次", "冻结", "与各 Gate 口径一致")],
        done=["每个候选的护栏判定可复算（脚本 + 输入 sha256 + 输出）",
              "短名单 ≤3 且每个都有明确进入理由",
              "报告显式标注 `selection_score_only` 与 `missing_mode`",
              "未过护栏的候选被标记 `rejected` 且保留在注册表（负资产）"],
        forbid=["用未标口径的分数做比较",
                "把 `selection_score_only` 数字伪装成独立确认",
                "静默丢弃未过护栏的候选"],
        risk=[("候选间口径不一致", "对比失真", "统一用同一评分器与同一折，报告中标注口径字段"),
              ("护栏把所有候选挡掉", "只能回退 B0", "这正是护栏的目的；按 §9.5 走回退协议")],
        stop=["短名单为空 → 直接进入 E10 的回退协议分支"],
        code=["E9/code/aggregate_oof.py", "E9/code/choose_submission.py"],
        evidence=["`reports/E9_submission_decision.json`"],
        prereg_extra={"primary_metric": "guardrail_pass",
                      "mandatory_checks": ["contract_ok", "guardrail_evaluated"]},
    ),
    dict(
        pid="P1", title="16 井次级体检与泄漏终审",
        nature="验证：泄漏是红线",
        deps=["E9/P0"],
        goal="对 top-2 候选在 v2 冻结的 16 井 `folds_confirm` 上各跑一次；完成四类泄漏审计"
             "（折维度/特征来源/标准化 fit 范围/伪标签来源）。",
        why=["16 井来自 v1 训练集，是 **v1-exposed** 的次级体检，只能防崩坏、不能宣称独立确认；",
             "`资料库/12` §3.5 与 §8.5 列出完整泄漏清单；本地 OOF 虚高的主要来源就是"
             "同井相邻点跨折与折外 fit；",
             "泄漏审计是提交前的最后一道关卡，失败即放弃该候选。"],
        inputs=["top-2 候选权重", "`../v2/artifacts/E0/`（folds_confirm）", "`资料库/12` §3.5"],
        outputs=["`$V4_REPORTS_DIR/E9_confirm.json`（16 井体检分数）",
                 "`$V4_REPORTS_DIR/E9_leakage_audit.json`"],
        steps=["在 16 井上推理 top-2 候选（权重不重训），评分并与 80 井 OOF 对比",
               "判定崩坏：16 井分数较 80 井 OOF 下降 > 1.5 分即触发复核",
               "泄漏审计 1：折维度——确认所有折都是井维度，无同井跨折",
               "泄漏审计 2：特征来源——provenance CSV 中无目标派生列",
               "泄漏审计 3：标准化 fit 范围——所有 scaler/分位数参数只在训练折 fit",
               "泄漏审计 4：伪标签来源——transductive/伪标签只用输入分布或训练折模型输出",
               "label-shuffle 阴性对照：随机打乱标签后 OOF 应接近常数基线"],
        params=[("confirm 井数", "16", "冻结", "v1-exposed，标注清楚"),
                ("崩坏阈值", "1.5 分", "冻结", "超过则复核"),
                ("审计项", "4 类", "冻结", "缺一不可")],
        done=["16 井体检完成并标注 `v1_exposed=true`、`not_independent_confirmation=true`",
              "四类泄漏审计全部有结论（通过或标注残余风险）",
              "label-shuffle 对照分数接近常数基线（证明无标签泄漏）",
              "无候选出现 > 1.5 分崩坏（若有则排除该候选）"],
        forbid=["把 16 井体检当独立确认宣称显著增益",
                "用 confirm 结果回头调阈值/权重/结构",
                "跳过任何一类泄漏审计"],
        risk=[("16 井噪声大", "分数波动被误读", "只做崩坏判定，不做增益判定"),
              ("审计疏漏", "泄漏进入提交", "审计清单固定 4 类，逐项写结论与证据")],
        stop=["发现高风险泄漏 → 该候选立即作废，E10 走回退协议"],
        code=["E9/code/confirm_check.py", "E9/code/leakage_audit.py"],
        evidence=["`reports/E9_confirm.json`、`reports/E9_leakage_audit.json`"],
        prereg_extra={"primary_metric": "no_high_risk_leak",
                      "mandatory_checks": ["leakage_audit_complete", "confirm_no_breakdown"]},
    ),
    dict(
        pid="P2", title="A 榜短名单仲裁（预算制 ≤3 次）",
        nature="外部仲裁：只看崩坏，不调参",
        deps=["E9/P1"],
        goal="按预算制提交 ≤3 次（每日上限 5 次，留 1 次余量给最终提交），"
             "登记 `E9_a_board_log.json`，只做短名单仲裁与崩坏体检。",
        why=["A 榜只有 5 口井，噪声带约 ±0.02 分（`v3/PLAN.md` 继承资产表），"
             "**不能**用于细粒度调参，否则线上会掉 3–10 分（`资料库/12` §3.5）；",
             "A 榜的价值是筛掉\"本地好、线上崩\"的候选（例如 SW 被裁剪、原子门误判）；",
             "额度是共享硬瓶颈（5 次/日），必须预算制。"],
        inputs=["E9/P0 的短名单 result.zip", "历史 A 榜锚点（B0=82.2757、E10=81.8576、E13=82.3035）"],
        outputs=["`$V4_REPORTS_DIR/E9_a_board_log.json`（候选/时间/分数/用途）",
                 "短名单的 A 榜排序结果"],
        steps=["挑选最多 3 个最多样化的短名单候选（不是分数最高的 3 个）",
               "逐个提交并记录 `candidate_id/zip_sha256/submit_time/a_board_score`",
               "对比锚点：任一次低于 B0 锚点 0.10 以上 → 立即回退该路线",
               "汇总 A 榜排序与本地 OOF 排序的一致性（三口径一致性分析的第一部分）",
               "**记录但不据反馈修改任何模型**"],
        params=[("预算", "≤3 次/日（留 1 次）", "冻结", "每日上限 5 次"),
                ("仲裁判据", "不崩坏（≥ 锚点 − 0.10）", "冻结", "不做细粒度比较"),
                ("候选多样性", "不同主干/不同原子门策略", "冻结", "避免 3 个几乎相同的候选")],
        done=["每次提交都有完整记录（含 zip sha256 与用途）",
              "无候选低于锚点 0.10 以上（若有则记录回退）",
              "明确声明\"A 榜只做仲裁，不用于调参\"",
              "A 榜与本地 OOF 的一致性判断写入报告"],
        forbid=["用 A 榜反馈调整模型/阈值/权重",
                "单日超过 5 次",
                "把 A 榜分数当作优于本地 OOF 的证据"],
        risk=[("A 榜波动误判", "±0.02 噪声被当成真实差异", "只做 ±0.10 级别的崩坏判定"),
              ("额度浪费", "提交了 3 个几乎相同的候选", "先做候选多样性检查")],
        stop=["当日额度用尽 → 候选进 pending 队列，次日再提交"],
        code=["E9/code/submit_batch.py"],
        evidence=["`reports/E9_a_board_log.json`"],
        prereg_extra={"primary_metric": "a_board_no_breakdown",
                      "thresholds": {"max_degradation": 0.10},
                      "mandatory_checks": ["a_board_log_complete"]},
    ),
]

# ------------------------------------------------------------------ E10
P["E10"] = [
    dict(
        pid="P0", title="全量重训与 CPU 推理导出",
        nature="交付：训练+推理双入口",
        deps=["E9/P2"],
        goal="用 E9 通过候选的配置在全部 80 井上重训（或改用 5 折权重集成），"
             "导出 CPU 可加载权重与 ONNX 兜底；验证两次前向完全一致。",
        why=["`rules.md` §8.3 要求\"训练 + 推理\"可独立运行、结果一致；"
             "因此训练入口必须真实可跑，即使评测时不需要；",
             "评测机不保证有 GPU，推理必须在 CPU 上稳定运行（fp32、确定性）；",
             "全量重训 vs 折集成的选择必须在看到 E9 结果前预注册，避免事后择优。"],
        inputs=["E9 通过的候选配置与权重", "80 井全量数据"],
        outputs=["`models/v4/final/*.pt`（fp32，CPU 可加载）",
                 "`models/v4/final/*.onnx`（可选兜底）",
                 "`$V4_REPORTS_DIR/E10_final_train.json`（训练日志/耗时/磁盘）"],
        steps=["按预注册选择聚合方式（全量重训 或 5 折权重集成）",
               "全量重训时固定 seed、固定 epoch 数（不做早停，因为无验证折）",
               "导出：`state_dict` → fp32 `.pt`；同时尝试 `torch.onnx.export`（失败不阻塞）",
               "CPU 冒烟：加载 → 前向 → 与训练期预测逐点比对（容差 ≤1e-6）",
               "两次独立前向的 sha256 必须一致（确定性验证）",
               "记录峰值内存与单次推理耗时（目标 < 30 min、< 8 GiB）"],
        params=[("聚合方式", "5 折权重集成（默认）", "全量重训/折集成", "**预注册**后不改"),
                ("精度", "fp32", "fp32/fp16", "提交侧必须 fp32"),
                ("确定性", "固定 seed + 确定性算子", "—", "两次运行 sha256 一致"),
                ("ONNX", "尝试导出", "导出/跳过", "失败不阻塞主路径")],
        done=["CPU 加载并前向成功，两次结果 sha256 一致",
              "单次推理 < 30 min 且峰值内存 < 8 GiB",
              "训练入口 `train.py` 在 A100 上可跑通（不要求评测时执行）",
              "导出的权重与配置哈希写入 manifest"],
        forbid=["导出依赖 GPU 的权重",
                "在推理阶段做任何训练",
                "把训练日志/中间产物写进提交包"],
        risk=[("全量重训无验证折", "无法早停，可能过拟合", "固定 epoch 数（用折内平均最优 epoch）+ 强正则"),
              ("ONNX 导出失败", "兜底路径不可用", "不阻塞：主路径仍是 torch CPU"),
              ("CPU 推理超时", "> 30 min", "减成员数或减 chunk；必要时用单折权重")],
        stop=["CPU 推理 > 30 min 或内存 > 8 GiB → 简化模型（减宽度/减成员）后重试"],
        code=["E10/code/final_train.py", "E10/code/export_cpu.py", "train.py"],
        evidence=["`reports/E10_final_train.json`"],
        prereg_extra={"primary_metric": "cpu_inference_ok",
                      "thresholds": {"max_minutes": 30, "max_memory_gb": 8},
                      "mandatory_checks": ["cpu_inference_ok", "deterministic_output"]},
    ),
    dict(
        pid="P1", title="打包、干净目录复现与 B0 fallback",
        nature="复现门禁 + 保底策略",
        deps=["E10/P0"],
        goal="组装 `submission_code_v4.zip`；在**只含 zip 与 data** 的干净目录用官方命令一次性复现；"
             "同时预构建 B0 fallback 保险包并在同一干净目录验证。",
        why=["`rules.md` §8.4：复现失败即取消晋级资格——这是最高优先级风险；",
             "v4 是纯 DL 管线，没有 B0 patch 隔离兜底，因此必须**预先**构建 B0 fallback；"
             "等到失败时再建已经来不及（且情绪与时间压力下易出错）；",
             "干净目录验证必须用官方 `--data_dir`（评测命令），不能用开发时的自定义参数。"],
        inputs=["`models/v4/final/`、`v4/src/`、`v4/predict.py`、`v4/train.py`、锁文件",
                "`../v1/submission_e7/`（fallback 源）"],
        outputs=["`submission/submission_code_v4.zip`、`submission/result.zip`",
                 "`$V4_REPORTS_DIR/E10_reproduce_report.json`",
                 "`submission/submission_code_b0_fallback.zip`、`$V4_REPORTS_DIR/E10_B0_fallback_manifest.json`"],
        steps=["组装 zip：README/predict.py/train.py/requirements.txt/configs/src/models（含权重）",
               "在干净目录解压，执行 `python predict.py --data_dir ./data --output result.json`",
               "比对：与提交的 result.json 逐点差 ≤1e-6；记录最大差与超差行数",
               "两次独立运行，sha256 必须一致",
               "构建 B0 fallback：从冻结 v1 源码生成官方 `--data_dir` 兼容的自包含包",
               "在**同一干净目录**验证 fallback：结果与冻结 B0 逐点差 ≤1e-9（或字节一致）",
               "写 `inference_verified=true` 与全部指纹"],
        params=[("逐点容差", "≤1e-6", "冻结", "浮点级微差可接受"),
                ("fallback 容差", "≤1e-9 或字节一致", "冻结", "v1 包用同版本依赖"),
                ("干净目录环境", "只含 zip + data", "冻结", "不得有 v4/v1/v2 目录"),
                ("两次运行", "sha256 一致", "冻结", "确定性要求")],
        done=["干净目录复现成功且逐点差 ≤1e-6",
              "两次运行 sha256 一致",
              "B0 fallback 包在同一干净目录验证通过（逐点差 ≤1e-9 或字节一致）",
              "`E10_reproduce_report.json` 含目录/命令/stdout 摘要/版本/耗时/内存/指纹"],
        forbid=["在提交包里依赖 v1/v2/v3 目录或外部数据",
                "复现时联网或安装包",
                "跳过 fallback 构建",
                "用开发时的自定义 CLI 参数代替官方 `--data_dir`"],
        risk=[("复现失败", "干净目录报错或结果不一致", "门禁前置到每个候选；失败立即回退 B0 fallback"),
              ("fallback 也失败", "保险失效", "fallback 现在就建并验证，不等到 E10 当天"),
              ("zip 体积超限", "上传失败", "权重 < 50 MB、代码 < 5 MB，剔训练日志")],
        stop=["复现不通过 → 不提交 v4，改用 fallback 包，并在报告写明原因"],
        code=["E10/code/build_submission.py", "E10/code/verify_inference.py",
              "E10/code/build_b0_fallback.py"],
        evidence=["`reports/E10_reproduce_report.json`、`reports/E10_B0_fallback_manifest.json`"],
        prereg_extra={"primary_metric": "clean_dir_reproduce",
                      "thresholds": {"max_point_diff": 1e-6},
                      "mandatory_checks": ["clean_dir_reproduce", "b0_fallback_verified",
                                           "contract_ok"]},
    ),
    dict(
        pid="P2", title="提交执行与归档登记",
        nature="唯一外部动作",
        deps=["E10/P1"],
        goal="执行提交（预算内），登记平台返回；把提交指纹与结果写入候选注册表用于赛后复盘。",
        why=["提交是唯一不可逆的外部动作，必须完全可追溯（zip sha256、时间、返回分数）；",
             "赛后 B 榜出来后需要把\"本地 OOF / A 榜 / B 榜\"三口径对齐分析，"
             "因此提交当时的配置指纹必须被记录；",
             "提交后不得再改动候选文件（否则破坏可追溯性）。"],
        inputs=["E10/P1 的 submission zip", "平台提交入口（人工操作）"],
        outputs=["`$V4_REPORTS_DIR/E10_submission_log.json`",
                 "`versions/candidates.json`（`status=submitted/frozen_best` + a_board_score）"],
        steps=["确认当日额度与提交包 sha256",
               "人工/脚本提交 `result.zip` 与 `submission_code.zip`",
               "记录返回（分数/时间/错误信息）到 submission_log",
               "更新候选注册表状态；**冻结该候选文件**（不再修改）",
               "若提交的是 fallback 包，显式标注 `choice=b0_fallback` 与原因"],
        params=[("每日额度", "≤5 次", "硬约束", "—"),
                ("提交后", "候选冻结", "冻结", "修改必须新建 candidate_id")],
        done=["提交记录含 zip sha256、时间、返回分数或错误",
              "候选注册表状态一致（`submitted`/`frozen_best`）",
              "提交的包文件被标记为不可变（记录 sha256 并停止修改）"],
        forbid=["提交后修改候选文件",
                "根据 A 榜返回立刻改模型再提交（每日额度有限，且属于 A 榜过拟合）"],
        risk=[("提交格式错", "平台报错", "契约校验 + 干净目录复现已前置"),
              ("额度用尽", "无法提交", "预算制：E9 最多 3 次，留额度给 E10")],
        stop=["提交失败 → 立即用 fallback 包重试（若额度允许），否则次日"],
        code=["E10/code/submit.py"],
        evidence=["`reports/E10_submission_log.json`"],
        prereg_extra={"primary_metric": "submission_recorded",
                      "mandatory_checks": ["submission_log_complete"]},
    ),
]

# ------------------------------------------------------------------ E11
P["E11"] = [
    dict(
        pid="P0", title="资产归档（含 sha256）",
        nature="可独立理解的知识包",
        deps=["E10/P2"],
        goal="生成代码/模型/候选/报告的清单与 sha256，确保归档可在**不含 v1/v2/v3** 的目录中独立理解。",
        why=["归档是下一代的输入；无哈希的归档无法验证，也无法判断\"哪个数字对应哪份权重\"；",
             "git 仓库只放代码，权重与产物在 `/data`，因此归档清单必须把两侧关联起来；",
             "失败候选（rejected）与成功候选同等重要——它们是负知识资产。"],
        inputs=["`v4/` 全部报告与代码、`/data/v4/**` 的产物", "`versions/candidates.json`"],
        outputs=["`$V4_REPORTS_DIR/E11_archive_manifest.json`",
                 "`v4/docs/PROJECT_FILES.md`（目录树 + 文件用途 + 数量统计）"],
        steps=["遍历 `v4/` 与关键 `/data/v4` 产物，逐项记录路径/大小/sha256/用途",
               "生成目录树文档（层级/文件数/职责），标注\"提交必需\"与\"仅开发\"",
               "核对候选注册表里每个候选都能在归档中找到对应权重与结果",
               "确认归档不依赖 v1/v2/v3 即可理解（除 E0 冻结引用件外）",
               "删除任何重复/临时产物（保留有分析价值的）"],
        params=[("哈希算法", "sha256", "冻结", "—"),
                ("保留策略", "保留全部候选（含 rejected）", "冻结", "负资产")],
        done=["清单完整且每项有 sha256",
              "每个候选都能追溯到权重 + result.json + cv.json",
              "`PROJECT_FILES.md` 目录树与实际一致（文件数一致）"],
        forbid=["删除任何候选或报告（含 rejected）",
                "在归档中丢失\"哪个数字来自哪份权重\"的关联"],
        risk=[("清单与实物漂移", "哈希对不上", "归档脚本从文件系统实时计算，不手写")],
        stop=["—（收尾阶段无停止条件）"],
        code=["E11/code/archive.py"],
        evidence=["`reports/E11_archive_manifest.json`、`docs/PROJECT_FILES.md`"],
        prereg_extra={"primary_metric": "archive_complete",
                      "mandatory_checks": ["archive_hashes_present"]},
    ),
    dict(
        pid="P1", title="复盘与下一代方向",
        nature="知识沉淀",
        deps=["E11/P0"],
        goal="写三口径（本地 CV / A 榜 / B 榜）一致性分析、路线有效性表、资源统计与下一代方向储备。",
        why=["v1 的 `v1.md` 与 v2 的复盘是后续版本最有价值的输入（v3/v4 都直接引用了它们）；",
             "资源统计（训练时长/磁盘/A 榜配额）决定下一代预算分配；",
             "必须区分**确定性结论**（只依赖规则/数据/评分）与**经验性结论**（依赖当前模型家族），"
             "否则下一代会把经验当定律。"],
        inputs=["全部 `reports/E*.json`、A/B 榜历史、`training_time_log.json`"],
        outputs=["`$V4_REPORTS_DIR/E11_retrospective.md`",
                 "`$V4_REPORTS_DIR/E11_next_directions.md`"],
        steps=["三口径对照表：每个候选的本地 OOF / A 榜 / B 榜（若有）与排序一致性",
               "路线有效性表：每个方向的结论（采纳/NO-GO）+ 证据 + 是否可复算",
               "资源统计：单任务训练时长分布、磁盘峰值、A 榜配额使用、镜像构建次数",
               "下一代方向：未触发的余量、未验证的假设、需要什么**新信息源**才允许重开 NO-GO 方向",
               "明确标注确定性 vs 经验性结论"],
        params=[("结论分类", "确定性/经验性", "冻结", "必须逐条标注"),
                ("B 榜", "赛后才有", "冻结", "未回收则明确标缺失")],
        done=["含三口径对照表、路线有效性表、资源统计表",
              "下一代方向清单每条都有触发条件",
              "所有结论标注确定性/经验性",
              "B 榜缺失时明确标注而非猜测"],
        forbid=["把经验性结论写成确定性结论",
                "用 B 榜成绩做赛后调参并写入复盘之外的地方"],
        risk=[("复盘流于形式", "无具体数字", "每张表都必须含可核验数字与来源")],
        stop=["—"],
        code=["E11/code/retrospective.py"],
        evidence=["`reports/E11_retrospective.md`、`reports/E11_next_directions.md`"],
        prereg_extra={"primary_metric": "retrospective_complete",
                      "mandatory_checks": ["three_way_table", "route_table", "resource_table"]},
    ),
]

# =============================================================================
# 渲染
# =============================================================================

def render(e: str, p: dict) -> str:
    L: list[str] = []
    L.append(f"# {e}/{p['pid']} {p['title']}\n")
    L.append(f"> 所属阶段：[{e}](../PLAN.md)　|　总计划：[v4/PLAN.md](../../PLAN.md)　|　"
             f"索引：[资料引用索引](../../资料引用索引.md)\n")
    L.append(f"> **性质**：{p['nature']}　|　**依赖**：{'、'.join(p['deps'])}")
    L.append(">")
    status = p.get("status", "pending")
    mark = {"done": "✅ 已完成", "pending": "⏸ 待执行", "blocked": "⛔ 阻塞",
            "in_progress": "▶ 进行中", "no_go": "❌ NO-GO"}.get(status, status)
    L.append(f"> **状态**：{mark}　{('　证据：' + '、'.join(p['evidence'])) if status == 'done' else ''}")
    L.append("")
    L.append("> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的结论与证据。\n")
    L.append("---\n")

    L.append("## 1. 目标\n")
    L.append(p["goal"] + "\n")

    L.append("## 2. 为什么需要这一步\n")
    for i, x in enumerate(p["why"], 1):
        L.append(f"{i}. {x}")
    L.append("")

    L.append("## 3. 输入契约\n")
    for x in p["inputs"]:
        L.append(f"- {x}")
    L.append("")

    L.append("## 4. 输出契约\n")
    for x in p["outputs"]:
        L.append(f"- `{x}`" if not x.startswith("`") else f"- {x}")
    L.append("")

    L.append("## 5. 执行步骤\n")
    for i, x in enumerate(p["steps"], 1):
        L.append(f"{i}. {x}")
    L.append("")

    L.append("## 6. 参数与配置\n")
    L.append("| 参数 | 默认值 | 搜索范围/说明 | 选择位置 |")
    L.append("|---|---|---|---|")
    for name, default, rng, where in p["params"]:
        L.append(f"| {name} | {default} | {rng} | {where} |")
    L.append("")

    L.append("## 7. 完成判据\n")
    for x in p["done"]:
        L.append(f"- {x}")
    L.append("")

    L.append("## 8. 禁止事项\n")
    for x in p["forbid"]:
        L.append(f"- {x}")
    L.append("")

    L.append("## 9. 风险与对策\n")
    L.append("| 风险信号 | 早期表现 | 对策 |")
    L.append("|---|---|---|")
    for a, b, c in p["risk"]:
        L.append(f"| {a} | {b} | {c} |")
    L.append("")

    L.append("## 10. 停止规则\n")
    for x in p["stop"]:
        L.append(f"- {x}")
    L.append("")

    L.append("## 11. 代码归属\n")
    for x in p["code"]:
        code_path = x if x.endswith(".py") or x.endswith(".sh") else x
        L.append(f"- `{code_path}`" if not code_path.startswith("`") else f"- {code_path}")
    L.append("")

    L.append("## 12. 复算与证据\n")
    for x in p["evidence"]:
        L.append(f"- {x}")
    L.append("")
    L.append("```bash")
    L.append("# 云端（平台训练任务）")
    L.append(f"bash /code/workspace/v4/run_train.sh --mode stage --stage {e}")
    L.append("# 本机（口径层，无 torch）")
    L.append(f"python3 v4/E0/code/run_all.py && python3 v4/tools/verify_reference.py")
    L.append("```")
    L.append("")

    L.append("## 12.5 选择协议（H1：inner-OOF only）\n")
    L.append("**所有超参/阈值/早停/结构选择只允许用 inner 折**（`$V4_REPORTS_DIR/E0_folds.json::inner`）。")
    L.append("")
    L.append("| 用途 | 允许的数据 | 禁止 |")
    L.append("|---|---|---|")
    L.append("| 超参/阈值/λ/τ/集成权重选择 | 该 outer 折的 inner-OOF | outer 验证折标签 |")
    L.append("| 早停 | inner-OOF 的真实 `score.py` 分数 | outer 折分数、loss 值 |")
    L.append("| 结构/特征筛查（省机时） | 可先用 fold0 做**资源预检** | 预检结论不得进入 Gate 数值 |")
    L.append("")
    L.append("> 若某步骤确实只能看 outer 折（例如最终 OOF 汇总），该步骤**不得**反过来影响任何选择；")
    L.append("预检性质的 fold0 结果必须在报告中标 `exploratory=true`、`selection_score_only=true`。")
    L.append("")
    L.append("## 13. Gate 预注册要点\n")
    L.append(f"预注册文件：`v4/reports/{e}_{p['pid']}_gate_prereg.json`（实验**前**写入，"
             "之后不得改阈值，只能新建修订号）。"
             "完整模板见 [`docs/gate_template.md`](../../docs/gate_template.md)。\n")
    L.append("```json")
    import json as _json
    base = {
        "gate_id": f"{e}_{p['pid']}_gate",
        "stage": e,
        "p_stage": p["pid"],
        "created_at": "<ISO8601，写盘时填写>",
        "primary_metric": "oof_total",
        "primary_threshold_key": "min_delta",
        "baseline_version": "<已冻结候选或 CONST>",
        "baseline_artifact": "<基线 OOF 路径>",
        "baseline_manifest_sha256": "<sha256>",
        "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
        "alpha": 0.05,
        "multiplicity": "none",
        "candidate_budget": 1,
        "bootstrap_iters": 1000,
        "bootstrap_unit": "well_row_weighted_cluster",
        "pilot_std": None,
        "mde_units": 80,
        "min_detectable_effect": None,
        "planned_task_training_h": 1.0,
        "mandatory_checks": [
            "contract_ok", "atomic_precision_reported", "disk_budget_ok",
            "training_time_log_valid", "checkpoint_resumable", "no_label_leak",
        ],
        "decisions_locked": [],
        "notes": "",
    }
    extra = dict(p["prereg_extra"])
    # mandatory_checks 取并集：模板 6 项核心 + 本 P 追加项
    core = list(base["mandatory_checks"])
    for c in extra.pop("mandatory_checks", []):
        if c not in core:
            core.append(c)
    # thresholds 必须**深合并**（extra 里的阈值不得把 primary_threshold_key 覆盖掉）
    th = dict(base["thresholds"])
    extra_th = dict(extra.pop("thresholds", {}) or {})
    if "primary_threshold_key" in extra:
        # 本 P 指定了主阈值键名 -> 必须在合并后的表里存在
        key = extra["primary_threshold_key"]
        if key not in extra_th and key not in th:
            extra_th[key] = 0.0
    th.update(extra_th)
    base.update(extra)
    base["thresholds"] = th
    L.append(_json.dumps(base, ensure_ascii=False, indent=2))
    L.append("```")
    L.append("")
    L.append("> 模板已内置 6 项核心 `mandatory_checks`；写入实际预注册文件时：")
    L.append("> `created_at` 填当前时间；`baseline_version`/`baseline_artifact`/"
             "`baseline_manifest_sha256` 指向**已冻结**的基线；"
             "`planned_task_training_h` 必须 >0（软预算，单任务建议 ≤100h）。")
    L.append("> 校验器：`python3 v4/src/validation/gates.py --prereg <file>`（缺字段即失败）。")
    return "\n".join(L) + "\n"


def main() -> None:
    n = 0
    for e, plist in P.items():
        for p in plist:
            out = ROOT / e / p["pid"] / "PLAN.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render(e, p), encoding="utf-8")
            n += 1
    print(f"wrote {n} detailed P-level PLAN.md")


if __name__ == "__main__":
    main()
