#!/usr/bin/env python3
"""生成 v4 的 E*/PLAN.md（阶段层，12 份）与 E*/P*/PLAN.md 的**骨架**（33 份）。

注意（R2-M8 修正）：本文件确实会写 P 级骨架（`main()` 遍历 `P_STAGES` 并调用 `render_p`），
因此**必须在它之后运行 `gen_p_details.py`**，由后者把 33 份 P 级覆盖为详细版；
`P_STAGES` 里的 `(pid, title)` 同时是 E 级 §6 子计划清单的事实源。推荐的生成顺序：
    python3 docs/gen_plans.py          # 阶段层 12 份 + P 级骨架 33 份
    python3 docs/gen_p_details.py      # P 级（详细，覆盖上一步的骨架）
    python3 E0/code/mark_status.py     # 回写 E0 执行状态
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # v4/

HEADER = "> 所属总计划：[v4/PLAN.md]({up}PLAN.md)　|　本层代码 `{code}`　|　产物 `{art}`\n"

# ---------------------------------------------------------------- E 层
E_STAGES = [
    dict(
        e="E0", name="数据、评测与提交契约",
        nature="**契约冻结阶段。不训练任何模型。**",
        goal="把数据解析、缺失哨兵、标签三状态、官方评分器、按井 5 折、提交格式、版本路由与云端环境全部冻结为可复算的契约，使后续一切结论都有唯一口径。",
        why=[
            "评分口径与本地评分器实现若与官方有偏差，所有后续迭代都是在优化错误目标；v1 冻结的全常量基线 OOF = **70.4907** 是校验锚点。",
            "标签里 66.719% 是联合常量占位；**SW 是单一标签尺度（百分数，实测有效 8.305–99.9，`SW<1` 为 0 行）**——E0-R2 已证伪“99.9 与 [0,1] 双尺度”假设，`SW_SMALL_BRANCH` 作为遗留对照路径**永久关闭**；把 SW 误当小数尺度是最大的静默失分点，必须在写模型之前用单元测试锁死。",
            "云端是预装镜像（CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64，无 conda）+ 仅 64 GiB 磁盘；环境与磁盘必须在训练开始前实测确认。",
        ],
        inputs=["`../data/train`（80 井）、`../data/test`（10 井）", "`rules.md` §5–§8", "`../v1/src/well_folds.json`（按井 5 折）", "`资料库/12` §1–§5（口径与工程化落地）"],
        outputs=["`reports/E0_data_card.json`", "`reports/E0_env.json`、`reports/E0_disk_budget.json`", "`reports/E0_score_check.json`（常数基线 70.4907）", "`reports/E0_gate.json`、`reports/E0_gate_prereg.json`", "`artifacts/E0/folds.json`、`versions/folds_sha256.json`"],
        code=["`src/data/parse.py`（自写解析，只依赖标准库+numpy）", "`src/data/labels.py`（三状态判据 + SW 尺度常量）", "`src/score.py`（官方评分复算）", "`src/inference/contract.py`（提交契约校验）", "`E0/code/check_env.py`、`E0/code/setup_deps.sh`", "`E0/code/run_all.py`（一键复算）"],
        done=[
            "`E0/code/run_all.py` 一条命令复算出：80/10 井、730,268/95,948 行、三状态计数、折指纹、常数基线 70.4907（±1e-4）。",
            "`check_env.py` 的 hard 检查全过（torch 2.8.0 + torch_npu 2.8.0 / Ascend 910B / CANN 8.3rc2 / bf16 / 磁盘可用 ≥ 8 GB），结果写入 `E0_env.json`。",
            "提交契约单测通过：构造的假 `result.json` 能被 `validate_payload` 正确接受/拒绝。",
            "**全程不需要 torch**（数据与提交侧只依赖标准库+numpy/pandas）。",
        ],
        forbid=["训练任何模型", "修改折分配（必须与 `v1_well_folds.json` 逐字节一致）", "为省事裁剪 SW 或删除占位行"],
    ),
    dict(
        e="E1", name="纯 DL 行级基线",
        nature="**分母建立阶段。允许弱，必须正确。**",
        goal="用 24 维行级输入 + MLP（**无任何序列上下文**）对接三段式对齐损失，建立纯深度学习的绝对下限、容量标定与训练曲线基线；同时把**连续头参数化与损失修正**（POR 可到 0、SW 训练折仿射归一化、`L_aux` 逐目标尺度归一化、`masked_mean` 防 NaN、PERM log 空间官方截断）在这一层就冻结；硬 Gate OOF Total ≥ 78.0。",
        why=[
            "在引入序列主干之前，必须先知道\"只看当前深度点\"能拿到多少分，否则无法证明序列上下文的价值（`资料库/08` §0.3 第 1 层）。",
            "行级基线训练极快（分钟级），是验证损失实现、数据管线、OOF 流程是否正确的最高性价比手段。",
            "v2 E4/P3 的 CPU MLP 是 NO-GO，但那是逐点+无 NPU/GPU+小容量；E1 要在 Ascend 910B 上给出\"正确实现下的行级上限\"，作为 E3 的对照。",
            "改进 proposal（§3/§4）：若连续头仍用 `0.1+softplus(g)` 并被 SW 的 99.9 主导 `L_aux`，E3/E4 只会更快地优化一个错误目标；因此 POR/SW 参数化、`L_aux` 归一化、`masked_mean` 防 NaN 与 PERM 截断必须在 E1 就定稿，并配 `POR=0`/`POR<0.1` 切片单测。",
        ],
        inputs=["E0 冻结的数据卡、折、评分器", "`资料库/12` §2.4 的对齐损失参考实现"],
        outputs=["`models/E1/pd0.pt`、`experiments/E1/P1/pd0/oof.npz`", "`reports/E1_loss_curve.csv`、`reports/E1_gate.json`", "`versions/candidates.json::E1_PD0`"],
        code=["`src/features/basic.py`（F_raw + F_miss + F_depth）", "`src/losses/score_aligned.py`", "`src/models/row_mlp.py`", "`E1/code/train_row.py`、`E1/code/eval_oof.py`"],
        done=["80 井 5 折 OOF Total **≥ 78.0** 且 5 折方向一致。", "占位行上逐目标 Acc **≥ 0.98**。", "loss 曲线无 NaN；训练可 `--resume`。", "`atomic_precision_reported` 与 `contract_ok` 均写入 Gate。", "连续头参数化与损失修正落地：POR 能输出 0、SW 训练折仿射归一化可反变换、`L_aux` 用训练折稳健尺度归一化并写入 manifest、`masked_mean` NaN 单测与 `POR=0`/`POR<0.1` 切片单测通过。"],
        forbid=["加入任何窗口/序列特征（那属于 E2/E3）", "用 outer 折标签做早停或选阈值", "POR 连续头使用 `0.1 + softplus(g)`（锁死下界，无法输出实测的 0.0）", "把 SW 当 `[0,1]` 小数尺度或做任何 ×100 换算", "在整表上 fit `s_por`/`s_sw`/`sw_mu`/`sw_sigma`（只能训练折 fit）", "在 `masked_mean` 里让缺测位置的 NaN 污染均值"],
    ),
    dict(
        e="E2", name="特征工程与数据管线",
        nature="**特征与吞吐阶段。每组特征必须独立消融。**",
        goal="建立 `F_phys`/`F_win`/`F_well` 三组特征、数据增强与按井分片缓存，并把 16 GiB 内存与 64 GiB 磁盘的工程约束固化为可复用的数据管线。",
        why=[
            "序列主干需要稠密数值输入；物理交会特征（`资料库/01`/`02`）与窗口统计（`资料库/07` §6）在 v1 已被证明有效（C1→C1W +1.0562）。",
            "16 GiB 系统内存是真正的瓶颈：必须把\"按井分片 + 按需读取 + 即时增强\"写成管线，否则 E3 一开始就会 OOM。",
            "特征一旦进入训练就必须冻结版本；先定义再实验是防止\"看 OOF 后加列\"的唯一办法。",
        ],
        inputs=["E0 数据卡与分片", "`资料库/01`、`02`（物理公式）、`资料库/07` §6/§10（窗口与增强）"],
        outputs=["`src/features/physics.py`、`src/features/window.py`、`src/features/well.py`", "`cache/feat/F2/**`、`reports/E2_feature_provenance.csv`", "`reports/E2_ablation.json`、`reports/E2_mem_profile.json`"],
        code=["`E2/code/build_features.py`、`E2/code/ablate_groups.py`、`E2/code/mem_profile.py`"],
        done=["三组特征各自在行级 MLP 上有消融结果（增量或 NO-GO 均须登记）。", "`reports/E2_feature_provenance.csv` 登记每个派生列的公式与来源，无标签派生列。", "峰值常驻内存 < 12 GiB、on-disk 缓存 < 2 GB。"],
        forbid=["任何使用目标值的派生特征", "跨井 fit 的标准化参数（必须只在训练折 fit）"],
    ),
    dict(
        e="E3", name="深度序列主干",
        nature="**主线上限阶段（本计划的核心赌注）。**",
        goal="实现 1D U-Net 与 TCN 两种深度序列主干（窗口→窗口，**全段 seq2seq、无滑窗中心点**），与行级模型同头对比；硬 Gate：OOF ≥ 81.0 **且**序列主干必须显著优于同头行级模型。",
        why=[
            "`资料库/08` §0.3 把 1D U-Net / TCN 列为\"最可能冲高分的结构\"，而前代因 CPU 限制从未真正训练过。",
            "0.1 m 采样意味着 10 m 储层段 = 100 点；没有数百点的感受野，模型看不到层段结构，只能逐点外推。",
            "必须先做**感受野消融**：若缩小感受野不降分，说明上下文没被用上，此时扩容是浪费算力，应先修数据/结构。",
            "改进 proposal §6：U-Net 的 depthwise-separable + 空洞卷积会带来 **padding 边界伪影**，chunk 拼接会带来**接缝跳变**；TCN 在离线任务里必须**非因果**，dilation_max=512（≈50 m 感受野）要先验证有效而不是盲目继续加。",
        ],
        inputs=["E2 特征管线", "`资料库/08` §4/§6（1D-CNN 与 TCN 细节）"],
        outputs=["`src/models/unet1d.py`、`src/models/tcn.py`、`src/data/seq_dataset.py`", "`models/E3/{unet,tcn}_fold*.pt`、`experiments/E3/P2/*/oof.npz`", "`reports/E3_gate.json`、`reports/E3_receptive_field_ablation.json`、`reports/E3_boundary_report.json`"],
        code=["`E3/code/train_seq.py`、`E3/code/rf_ablation.py`、`E3/code/compare_row_vs_seq.py`"],
        done=["OOF Total **≥ 81.0**；相对同头行级模型的加权配对井级 bootstrap 95% CI 下界 > 0。", "感受野消融表：depth∈{3,5}、dilation_max∈{64,512} 的对照结果齐备，并给出 512 是否已达饱和的结论。", "**padding 边界伪影检查**：逐目标比较井首/井尾 10 m 与井中段的 Acc，差值超过阈值必须给出解释或修正。", "**全段 seq2seq** 输出（逐行同长，不用滑窗中心点）+ **chunk 重叠推理与加权拼接**（接缝处无跳变）。", "TCN 全程**非因果**；5 折全部同向；checkpoint 可 `--resume`；`disk_guard` 全程未触发 abort。"],
        forbid=["使用 ImageNet/自然图像预训练权重", "把验证井的数据用于训练折统计", "跳过感受野消融直接堆容量", "把 TCN 做成因果卷积（本任务是离线整井推理，因果化会白白丢失未来上下文）", "用滑窗中心点预测代替全段 seq2seq 输出", "在 chunk 拼接处不做重叠加权（会造成接缝跳变）"],
    ),
    dict(
        e="E4", name="Patch Transformer 与多尺度融合",
        nature="**第二主干与融合阶段。**",
        goal="实现 PatchTST 式**通道独立 + 相对位置编码**的 Patch Transformer，并与 CNN 主干做多尺度融合（**优先门控 / FiLM**，其次 concat），评估是否超过单一 CNN 主干。",
        why=[
            "`资料库/08` §0.3 第 3 层与 §7.5 指出 Patch 化 + 通道独立是长序列建模的低成本高效方案。",
            "CNN 主干擅长局部形态，Transformer 擅长长程依赖；两者并联是常见且有效的互补结构。",
            "必须先有 E3 的 CNN 结果做对照，否则无法判断 Transformer 是否值得其算力成本。",
            "改进 proposal §6：80 井小数据下 PatchTF 极易过拟合，必须优先小 `d`/小 `layers`；patch size/stride/overlap 只能在 inner-OOF 上搜；重叠 chunk 推理是避免边界跳变的前提；融合头必须小且结构/权重只在 inner-OOF 选。",
        ],
        inputs=["E3 主干与 OOF", "`资料库/08` §7（注意力细节）"],
        outputs=["`src/models/patchtf.py`、`src/models/multiscale.py`", "`models/E4/**`、`experiments/E4/**/oof.npz`", "`reports/E4_gate.json`、`reports/E4_param_budget.json`"],
        code=["`E4/code/train_patchtf.py`、`E4/code/fuse_multiscale.py`"],
        done=["多尺度或 PatchTF 至少一个配置 ≥ E3 最佳（CI 下界 > 0），否则明确 NO-GO 并保留 E3 结构。", "参数量/显存/单折耗时报告齐备。", "通道独立与相对位置编码保留（消融证明二者各自有贡献或记录 NO-GO）。", "patch size/stride/overlap 与融合结构/权重全部只在 inner-OOF 上选定，并记录搜索表。", "重叠 chunk 推理完成，接缝处无跳变（与单段整井推理逐点比对）。"],
        forbid=["用 2.5+ 的注意力 API", "引入需要编译的注意力扩展（flash-attn/xformers）", "用 outer 折或整井拼接结果选 patch/融合超参", "在 80 井上直接开大 `d`/`layers` 而不先做容量消融"],
    ),
    dict(
        e="E5", name="逐目标精修",
        nature="**分目标攻坚阶段（三个目标互相独立）。**",
        goal="针对 POR（±0.008 窄带）、PERM（log 域长尾）、SW（**单尺度**：占位尖峰 99.9 + 有效分布 8.3–99.9）分别设计专用连续头与解码，逐目标独立判定增益。",
        why=[
            "总分按目标可加（`rules.md` §7.4），边际收益排序为 PERM > SW ≈ POR（`资料库/12` §3.3）。",
            "POR 的容差带只有 ±0.008，精度要求与另外两个目标完全不同；SW 的占位尖峰 99.9 与有效分布（同一标签尺度）形状差异极大，单头线性回归会被尖峰拉扯。",
            "目标独立化让失败可隔离：某一个目标退步不会污染其他两个。",
            "改进 proposal §3：POR 连续头必须能输出 ~0（实测 576 行 <1、186 行 <0.1），SW 连续头必须用**训练折仿射归一化**再反变换回标签尺度，PERM 必须用官方 `max(ẑ−z, log10(eps))` 截断。",
        ],
        inputs=["E3/E4 冻结主干", "`资料库/12` §3.4（占位尖峰与有效分布）"],
        outputs=["`src/models/heads.py`、`experiments/E5/{por,perm,sw}/oof.npz`", "`reports/E5_gate.json`、`reports/E5_per_target.json`"],
        code=["`E5/code/head_por.py`、`E5/code/head_perm.py`、`E5/code/head_sw.py`"],
        done=["至少一个目标的连续切片 Acc 显著提升（CI 下界 > 0），且其他目标不退步超过 0.01。", "SW 单尺度守卫通过：连续头用训练折 `sw_mu/sw_sigma` 仿射归一化训练、输出反变换回标签尺度（百分数），`SW_SMALL_BRANCH` 永久关闭，无任何 ×100 换算。", "POR 参数化消融表完整（`por_max·sigmoid` 推荐 / `softplus(g)−softplus(g0)` 备选），POR 连续头能输出 0。", "PERM 使用官方 log 空间截断，并给出「低估尾部 vs 官方评分器」一致性报告。"],
        forbid=["把 SW 全局裁剪到 [0,1]（禁止；实测会掉约 23 分）", "用某一目标的增益掩盖另一目标的退化", "POR 连续头使用 `0.1 + softplus(g)`", "对 SW 做任何 ×100 尺度换算"],
    ),
    dict(
        e="E6", name="联合常量状态与原子门",
        nature="**完整管线打通阶段（PD1 候选诞生）。**",
        goal="把 H0 联合占位头升级为**逐目标原子头 `q_por/q_perm/q_sw`（主保护）+ 辅助 `q_joint`（可选高置信硬门禁，默认关）**，逐目标独立硬切换，用 inner-OOF **官方总分**选择门限 τ_t；完整 PD1 管线 OOF ≥ 82.0。",
        why=[
            "66.719% 的行是联合常量占位，占约 66.72 分的白送分；任何软融合都会把 POR 推出 ±0.008 容差带。",
            "占位状态可从输入预测（v1 的原子门已验证），因此应把\"是否输出常量\"做成显式可学习决策，而不是让回归头勉强逼近。",
            "硬切换保证了原子点的精确性，是纯 DL 管线（无 B0 patch 隔离）下唯一的保护屏障。",
            "改进 proposal §1/§2：单目标原子行并不少（SW 31,030 / PERM 7,373 / POR 157），一个 joint 概率同时决定三目标会漏保护非 joint 原子行、又误伤 joint 行里的非原子目标；因此必须逐目标原子头，且 τ_t 要按官方总分（而非 F1/Acc）选。",
        ],
        inputs=["E3/E4/E5 的逐行表示与预测", "`资料库/12` §3.4"],
        outputs=["`src/models/row_mlp.py`（`RowMLP` 的 `q_joint + q_por/q_perm/q_sw` 五个头）、`src/inference/atomic_gate.py`", "`models/E6/pd1_*.pt`、`experiments/E6/P2/pd1/{result.json,result.zip,cv.json}`", "`reports/E6_gate.json`、`reports/E6_atomic_report.json`"],
        code=["`E6/code/train_state.py`、`E6/code/search_tau.py`、`E6/code/build_pd1.py`"],
        done=["OOF Total **≥ 82.0**（超过 B0 本地锚点 80.382479）。", "**逐目标**占位行 Acc **≥ 0.99**、recall **≥ 0.98**，并上报 precision/F1；τ_t 的选择过程可复算（只用 inner-OOF 官方总分）。", "提交契约通过（10 井 / 95,948 行）且 `predict.py --use-version PD1` 可在本机 CPU 跑通。", "两阶段训练记录完整；`joint_guard` 默认关闭且启用与否有 inner-OOF 证据。", "**原子/连续之间无任何插值**，`no_atom_continuous_interpolation` 为 true。"],
        forbid=["在占位与连续分支之间做线性插值", "用 outer 折或 A 榜选 τ", "用 F1/准确率而非官方总分选 τ_t", "默认启用 `joint_guard`", "把 joint 头当作三目标的唯一保护（丢失逐目标原子行）"],
    ),
    dict(
        e="E7", name="评分对齐损失与解码后处理",
        nature="**损失与解码的收尾优化阶段。**",
        goal="对三段式损失（align / aux / ph）做完整权重组消融与退火策略调参；实现**归一化 `L_aux`、容差边界聚焦权重、PERM 官方截断消融**与逐目标期望分解码，全部在 inner-OOF 上选择。",
        why=[
            "`资料库/12` §2 明确指出：评分截断使超过容差阈值的点不再产生梯度收益，把容量让给\"临界点\"才是最优策略——这只能通过损失权重与解码调节实现。",
            "对齐损失在训练早期梯度稀疏，退火策略直接决定能否收敛到好的局部解。",
            "解码阶段是\"零模型改动换分\"的手段，成本最低、风险最小。",
            "改进 proposal §4/§7：`L_aux` 若仍用绝对 Smooth L1，SW 的 99.9 会主导梯度；边界聚焦（κ/σ）与 PERM 截断必须用**同结构对照**消融，解码要按逐目标期望分而不是拍脑袋偏置。",
        ],
        inputs=["E6 完整管线", "`资料库/12` §2.2–2.4"],
        outputs=["`src/losses/*` 的最终配置、`reports/E7_loss_ablation.json`", "`reports/E7_decode_search.json`、`reports/E7_gate.json`"],
        code=["`E7/code/ablate_loss.py`、`E7/code/decode_search.py`"],
        done=["对齐损失优于纯 aux 损失（**同结构对照**，CI 下界 > 0）。", "三段式权重（含 λ₁ 退火曲线）的消融表完整。", "归一化 `L_aux`（训练折 `s_por`/`s_sw`）与 PERM 官方截断的消融完成。", "边界聚焦权重消融表完整：`κ∈{0.5,1.0,2.0}` × `σ∈{0.15,0.25,0.35}`（仅作可选增强，默认关）。", "逐目标期望分解码在 inner-OOF 上选定并冻结，附敏感性曲线。", "所有对照除被消融项外结构完全相同，且从不只报整体 Total。"],
        forbid=["用 outer 折或 A 榜选择损失权重/解码参数", "同时改多个损失项导致无法归因", "只报整体 Total 而隐藏单目标或切片的退化", "把边界聚焦权重设为默认开启（它只是 E7 的可选消融项）"],
    ),
    dict(
        e="E8", name="多任务、井级分支与集成",
        nature="**增益放大阶段。**",
        goal="实现 MMoE 式任务平衡、井级分支（H4，小容量强正则、只作消融）、EMA/SWA 与同折 top-k 快照集成、多 seed/多结构集成，以及 transductive 伪标签（只作消融）。",
        why=[
            "`资料库/08` §1.4：多任务硬共享通常优于三个独立模型，但必须解决权重失衡（MMoE 是标准解法）。",
            "井内近似常数的工程曲线（CAL/DEVI/AZIM/BIT/CASE）只提供井间区分度（`资料库/08` §0.1-4），井级分支是唯一合法的井间信号通路。",
            "v1 E8–E11 与 v2 E6/E7 的井级/序列 NO-GO 是在**无 GPU、无序列主干**的条件下得出的；E8 在已有序列主干的前提下重试一次，但必须用消融说话。",
            "改进 proposal §5：EMA 要用真实 `score.py` 在 inner-OOF 上逐 epoch 评估并保存 `ema.pt`，SWA 只作对照（BN 统计需谨慎）；同折 top-k 快照融合权重只能在 inner-OOF 选；**增益 CI 含 0 即 NO-GO**，不得用同源平均冒充增益。",
        ],
        inputs=["E6/E7 冻结管线", "`资料库/08` §1.4、§8（生成模型/伪标签）"],
        outputs=["`src/models/mmoe.py`、`src/models/well_head.py`、`src/ensemble/`", "`models/E8/**`（含 `ema.pt`/`last.pt`/`best.pt`）、`experiments/E8/**`", "`reports/E8_gate.json`、`reports/E8_ensemble_report.json`（逐折 delta + CI）"],
        code=["`E8/code/train_mmoe.py`、`E8/code/well_branch.py`、`E8/code/ensemble.py`、`E8/code/pseudo_label.py`"],
        done=["集成 ≥ 最佳单成员且加权配对 bootstrap 95% CI 下界 > 0。", "同源性/成员相关性报告齐备，同源平均不得计为增益；增益 CI 含 0 的策略一律 NO-GO。", "井级分支与伪标签各自给出采纳/NO-GO 结论；井级分支保持小容量强正则且校准参数只在 inner-OOF 选。", "EMA（decay∈{0.99,0.999,0.9995}）逐 epoch 用真实 `score.py` 在 inner-OOF 评估，`ema.pt` 与 `last.pt`/`best.pt` 并存；SWA 仅作对照并报告 BN 统计处理。"],
        forbid=["用测试集标签（不存在）", "把同源模型平均包装成\"集成增益\"", "把 CI 含 0 的增益打包成\"有效\"", "让井级校准参数在 inner-OOF 之外拟合"],
    ),
    dict(
        e="E9", name="诚实验证与提交护栏",
        nature="**验证与仲裁阶段。不出新模型。**",
        goal="汇总 80 井 OOF、做 16 井次级体检、泄漏终审、A 榜短名单仲裁与 `choose_submission.py` 提交护栏判定。",
        why=[
            "本地 80 井 OOF 是主判据；16 井 confirm 只是 v1-exposed 的次级体检，不能当独立确认。",
            "A 榜只有 5 口井、噪声带约 ±0.02，只能用于短名单仲裁与崩坏体检（`资料库/12` §3.5）。",
            "提交前必须有护栏，防止\"本地漂亮但明显弱于历史锚点\"的候选被提交。",
        ],
        inputs=["全部候选的 OOF 与 result.zip", "`reports/E0_a_board_log.json`（历史 A 榜锚点）"],
        outputs=["`reports/E9_validation_report.json`、`reports/E9_leakage_audit.json`", "`reports/E9_a_board_log.json`、`reports/E9_submission_decision.json`", "`reports/E9_gate.json`"],
        code=["`E9/code/aggregate_oof.py`、`E9/code/confirm_check.py`、`E9/code/leakage_audit.py`、`E9/code/choose_submission.py`"],
        done=["候选 OOF ≥ `max(75, 80.382479 − 1.0, A 榜锚点 82.2757 − 0.5 若已知)`，否则判定回退。", "泄漏审计覆盖：井级折、特征来源表、标准化 fit 范围、伪标签来源。", "16 井体检未出现 > 1.5 分的崩坏。"],
        forbid=["用 A 榜做细粒度调参", "在看到 confirm 结果后更换候选或阈值"],
    ),
    dict(
        e="E10", name="全量重训、打包与提交",
        nature="**交付阶段。**",
        goal="产出最终候选权重与自包含提交包；在干净目录用官方命令一次性复现；执行提交；同时预构建 B0 fallback 保险包。",
        why=[
            "`rules.md` §6.2/§8.3 要求\"训练+推理\"可独立复现、结果一致、环境可复现，复现失败直接取消资格。",
            "评测机不保证有加速器，因此推理必须 CPU 可跑、确定性、< 30 min。",
            "v4 是纯 DL 管线，没有 B0 patch 隔离兜底，因此必须**预先**构建 B0 fallback 保险包（从冻结 v1 E7 源码构建、官方 `--data_dir` 兼容）。",
        ],
        inputs=["E9 通过的候选", "`../v1/submission_e7`（fallback 源）"],
        outputs=["`submission/submission_code_v4.zip`、`submission/result.zip`", "`reports/E10_reproduce_report.json`、`reports/E10_bench.json`", "`submission/submission_code_b0_fallback.zip`、`reports/E10_B0_fallback_manifest.json`"],
        code=["`E10/code/final_train.py`、`E10/code/export_cpu.py`、`E10/code/verify_inference.py`", "`E10/code/build_submission.py`、`E10/code/build_b0_fallback.py`"],
        done=["干净目录（只含提交 zip + data）官方命令一次运行成功，与提交结果逐点差 ≤ 1e-6。", "两次独立运行 sha256 一致（确定性）。", "CPU 单次推理 < 30 min、峰值内存 < 8 GiB。", "B0 fallback 包在同一干净目录验证通过（byte-identical 或逐点差 ≤ 1e-9）。"],
        forbid=["在提交包里依赖 v1/v2/v3 目录", "依赖 GPU 或联网", "现场训练"],
    ),
    dict(
        e="E11", name="归档与复盘",
        nature="**收尾阶段。不出候选、不提交。**",
        goal="把 v4 的代码/模型/候选/报告/A 榜历史归档为可独立理解的知识包，并写出三口径一致性复盘与下一代方向储备。",
        why=[
            "v1 的 `v1.md` 与 v2 的复盘是后续版本最有价值的输入；v4 必须留下同等质量的复盘。",
            "失败路线（NO-GO）与成功路线同等重要——它们决定下一代是否重复踩坑。",
        ],
        inputs=["v4 全部 reports 与 experiments", "A 榜日志、B 榜结果（若可得）"],
        outputs=["`reports/E11_archive_manifest.json`（含 sha256）", "`reports/E11_retrospective.md`、`reports/E11_next_directions.md`"],
        code=["`E11/code/archive.py`、`E11/code/retrospective.py`"],
        done=["归档清单完整且每项有 sha256；可在不含 v1/v2/v3 的目录中独立理解。", "复盘含三口径对照表、路线有效性表、资源统计表（训练时长/磁盘/A 榜配额）。", "下一代方向清单含明确触发条件。"],
        forbid=["删除任何候选或报告（含 rejected）", "把 B 榜成绩用于事后调参并写入复盘之外的地方"],
    ),
]

# ---------------------------------------------------------------- P 层
P_STAGES = {
"E0": [
 ("P0","环境与磁盘实测（云端第一次运行）",
  "在云端跑 `check_env.py` / `disk_guard.py` / `setup_deps.sh`，把 torch 2.8.0 + torch_npu 2.8.0、Ascend 910B(64G)、bf16、Python 3.11、可用磁盘与 pip freeze 全部落盘为事实。",
  "未实测的环境假设会在 E3 训练数小时后才暴露（OOM / 版本不兼容 / 磁盘写满），代价极高。",
  ["`E0/code/setup_deps.sh`","`E0/code/check_env.py`","`src/data/disk_guard.py`"],
  ["`reports/E0_env.json`","`reports/E0_disk_budget.json`","`versions/locks/cloud_frozen.txt`"],
  ["所有 hard 级检查全过（判据不是固定项数，也不因 `--allow-non-target-device` 放宽依赖检查）；磁盘预算按挂载点报告且 Gate 的 `disk_budget_ok` 读 `--data-root`（`/data`）级别"],
  ["装任何会触碰 torch/nvidia-* 的包","在磁盘未知的情况下开始 E3"]),
 ("P1","数据卡、哨兵规则与标签三状态",
  "自写解析器读取 90 口井（按表头名对齐），冻结缺失哨兵规则、三状态判据（缺测/联合常量/有效）与目标值域统计，产出数据卡与折指纹。",
  "口径是唯一事实源；SW 是单一标签尺度（百分数，实测有效 8.305–99.9，`SW<1` 为 0 行），双尺度假设已在 E0-R2 证伪。",
  ["`src/data/parse.py`","`src/data/labels.py`","`E0/code/build_data_card.py`"],
  ["`reports/E0_data_card.json`","`artifacts/E0/folds.json`","`versions/folds_sha256.json`"],
  ["80/10 井、730,268/95,948 行、三状态计数与 `资料库/12` §3.1 一致；折与 `v1_well_folds.json` 逐字节一致"],
  ["解析时保留单位行","把 -99999 之外的小负值当有效值"]),
 ("P2","官方评分器复算与分母口径冻结",
  "按 `rules.md` §7.3 实现评分器，用全常量 (0.1, 0.01, 99.9) 复算 OOF，必须命中 70.4907。",
  "评分器是所有 Gate 的度量工具；它错了，后面全部结论作废。",
  ["`src/score.py`","`E0/code/score_check.py`"],
  ["`reports/E0_score_check.json`"],
  ["常数基线 = 70.4907 ± 1e-4；逐目标 Acc 与 `资料库/12` §3.3 预算表自洽"],
  ["用第三方库的 mean_squared_error 代替官方公式","把缺失行计入分母"]),
 ("P3","提交契约、版本路由与干净目录冒烟",
  "实现 `predict.py`（官方 `--data_dir`/`--output`）、`validate_payload`、候选注册表与 manifest，并在干净目录冒烟。",
  "提交格式错误是零分风险；契约必须早于模型存在。",
  ["`predict.py`","`src/inference/contract.py`","`src/versioning/registry.py`","`E0/code/run_all.py`"],
  ["`reports/E0_contract_tests.json`","`versions/candidates.json`","`reports/E0_gate.json`"],
  ["正/负样例都被正确判定；`python predict.py --list-versions` 在干净目录可用；E0 本地契约 Gate（含 cache）全部 mandatory 通过（项数见 `reports/E0_local_contract_gate.json`，禁止手写）"],
  ["契约校验依赖 torch","允许 logId 缺失或行数不符通过"]),
],
"E1": [
 ("P0","行级输入管线与分片缓存",
  "构建 `F_raw(14) + F_miss(14+1) + F_depth(3)` 逐行张量，按井分片落盘，检查无泄漏。",
  "输入管线的正确性决定了后面所有对比是否有意义。",
  ["`src/features/basic.py`","`E1/code/build_row_features.py`"],
  ["`cache/raw/*.npz`","`reports/E1_row_features.json`"],
  ["行数与 E0 数据卡一致；缺失指示位与掩码一致；缓存 < 100 MB"],
  ["在整表上 fit 标准化参数","把井身份作为特征"]),
 ("P1","行级 MLP + 评分对齐损失 + 5 折 OOF（硬 Gate ≥ 78.0）",
  "训练多任务 MLP（共享主干 + POR/PERM/SW 连续头 + 逐目标原子头），用三段式对齐损失（含训练折尺度归一化的 `L_aux`），5 折 OOF，硬 Gate ≥ 78.0。",
  "这是纯 DL 的分母与损失实现的验证器；连续头参数化与损失修正必须在这一层定稿。",
  ["`src/losses/score_aligned.py`","`src/models/row_mlp.py`","`E1/code/train_row.py`"],
  ["`models/E1/pd0.pt`","`experiments/E1/P1/pd0/oof.npz`","`reports/E1_gate.json`"],
  ["OOF ≥ 78.0、5 折同向、占位行 Acc ≥ 0.98、loss 无 NaN、可 `--resume`；POR 能输出 0；SW 训练折仿射归一化可反变换；`masked_mean` 与 `POR=0`/`POR<0.1` 切片单测通过"],
  ["用 MSE 当主损失","用 loss 值而非真实评分做早停","POR 连续头用 `0.1+softplus(g)`","把 SW 当 `[0,1]` 尺度或做 ×100 换算"]),
],
"E2": [
 ("P0","物理与交会特征（F_phys）",
  "实现 Wyllie/密度/中子孔隙度、GR/SP 泥质指数、RT/RXO 比值、PE 骨架、DEN-CNL 差等物理派生列。",
  "物理先验是约束与归纳偏置，不是万能解；必须消融验证（`资料库/03` §8.2）。",
  ["`src/features/physics.py`","`E2/code/ablate_groups.py`"],
  ["`reports/E2_ablation.json`（组 `F_phys`）"],
  ["组消融给出增量或 NO-GO；派生列公式全部登记"],
  ["使用目标值构造派生列"]),
 ("P1","窗口与井级特征（F_win / F_well）+ 内存纪律",
  "实现居中多尺度窗口统计（窗长 11/51/201 点）与井级聚合特征，处理 16 GiB 内存纪律。",
  "v1 已证明窗口特征是稳定增益；井级聚合是唯一的井间信号通路。",
  ["`src/features/window.py`","`src/features/well.py`","`E2/code/build_features.py`"],
  ["`cache/feat/F2/**`","`reports/E2_mem_profile.json`","`reports/E2_feature_provenance.csv`"],
  ["缓存 < 2 GB；峰值内存 < 12 GiB；窗口特征消融为正"],
  ["使用非居中（因果）窗口","在验证井上 fit 分位数"]),
 ("P2","数据增强与吞吐标定",
  "实现曲线随机掩码、深度抖动、段置换等增强，并用单折小规模实验标定吞吐（点/秒）与 batch 上限。",
  "80 井太少，正则化与增强是防过拟合的主要手段；吞吐标定决定 E3 的预算可行性。",
  ["`src/data/augment.py`","`E2/code/throughput.py`"],
  ["`reports/E2_throughput.json`"],
  ["给出每折耗时估计与显存/内存峰值；增强不改变标签语义"],
  ["增强破坏占位行的标签一致性"]),
],
"E3": [
 ("P0","序列数据集与分块（chunk）策略",
  "实现按井分块（chunk）数据集：变长井切 chunk、边界 overlap、按需读取分片、worker 内存自检。",
  "整井 4k–13k 点无法一次性进模型；分块策略决定感受野与吞吐的平衡。",
  ["`src/data/seq_dataset.py`"],
  ["`reports/E3_seq_dataset.json`"],
  ["worker 常驻 < 300 MB；chunk 边界无标签错位；可复现的采样顺序"],
  ["把整井常驻内存","打乱时破坏井内深度顺序"]),
 ("P1","1D U-Net 与 TCN 主干实现",
  "实现两种主干：U-Net 用 depthwise 可分离 + 空洞卷积（全段 seq2seq 输出），TCN 用**非因果**空洞卷积（dilation_max=512 ≈ 50 m 感受野），多任务头，bf16 训练。",
  "`资料库/08` §0.3 的核心推荐；两者头对头才能知道哪种归纳偏置更适合本数据。",
  ["`src/models/unet1d.py`","`src/models/tcn.py`","`E3/code/train_seq.py`"],
  ["`models/E3/{unet,tcn}_fold*.pt`"],
  ["参数量/显存/单折耗时记录；前向输出长度与输入严格一致（不用滑窗中心点）；padding 边界伪影检查（井首/尾 10 m vs 井中段）；chunk 重叠推理 + 加权拼接无接缝跳变"],
  ["使用 2.5+ API","引入需编译的 CUDA 扩展","把 TCN 做成因果卷积","用滑窗中心点代替全段输出"]),
 ("P2","行级对照 + 感受野消融 + 硬 Gate（≥81.0）",
  "同折同头对比序列主干与 E1 行级模型；做感受野消融（depth 3/5、dilation_max 64/512）；判定 Gate ≥ 81.0。",
  "这是\"上下文是否被利用\"的判据，也是 E4/E5 是否值得继续的前提。",
  ["`E3/code/compare_row_vs_seq.py`","`E3/code/rf_ablation.py`","`E3/code/gate.py`"],
  ["`experiments/E3/P2/*/oof.npz`","`reports/E3_gate.json`","`reports/E3_receptive_field_ablation.json`"],
  ["OOF ≥ 81.0；相对行级模型 CI 下界 > 0；消融表完整；padding 边界与拼缝报告齐备"],
  ["跳过消融直接堆容量","用 outer 折选超参"]),
],
"E4": [
 ("P0","Patch Transformer 主干（通道独立）",
  "实现 PatchTST 式通道独立 Transformer（保留相对位置编码，patch/stride/overlap 在 inner-OOF 搜索），与 E3 的 CNN 主干在同数据同折下可比。",
  "长程依赖与通道独立建模是 CNN 的天然补充；80 井小数据下必须小 `d`/小 `layers`。",
  ["`src/models/patchtf.py`","`E4/code/train_patchtf.py`"],
  ["`models/E4/patchtf_*.pt`"],
  ["与 E3 同数据同折可比；注意力走 `F.scaled_dot_product_attention`；重叠 chunk 推理；patch/stride/overlap 搜索表完整"],
  ["把曲线轴当图像轴做 2D 卷积","依赖 flash-attn"]),
 ("P1","多尺度融合（CNN × Transformer）与 Gate",
  "把 CNN 主干与 PatchTF 的输出按**门控 / FiLM**（其次 concat）融合送头，比较融合与单主干。",
  "多尺度是最常见的稳定增益来源，但必须证明超过最好单主干。",
  ["`src/models/multiscale.py`","`E4/code/fuse_multiscale.py`"],
  ["`experiments/E4/P1/*/oof.npz`","`reports/E4_gate.json`"],
  ["融合 ≥ 单主干最佳（CI 下界 > 0），否则 NO-GO 并保留 E3；融合头参数量小且结构与权重只在 inner-OOF 选"],
  ["用两个高度同源的分支冒充多尺度"]),
],
"E5": [
 ("P0","POR 窄带精修（±0.008）",
  "针对 ±0.008 容差带设计 POR 连续头：推荐 `por_cont = por_max·sigmoid(g)`（`por_max = 1.2×训练折有效 POR 最大值` ≈39.8），备选 `softplus(g)−softplus(g0)`，初始化自有效 POR 中位数（≈11.34）而非 0.1。",
  "POR 是容差最窄的目标，且实测有 576 行 <1（含 186 行 <0.1、真值 0.0），连续头必须能输出 ~0。",
  ["`E5/code/head_por.py`"],
  ["`experiments/E5/por/oof.npz`","`reports/E5_per_target.json`"],
  ["POR 连续切片 Acc 提升且原子带内占比上升；不牺牲 PERM/SW；POR 参数化消融表完整，`POR=0`/`POR<0.1` 切片单测通过"],
  ["用 `0.1 + softplus(g)` 锁死下界","用全局回归头直接压 POR 到 0.1"]),
 ("P1","PERM log 域精修（长尾与数量级）",
  "PERM 在 log10 域建模，按官方评分器实现 `max(ẑ−z, log10(eps))` 截断，初始化 `perm_z` ≈ −0.08，输出裁剪到 [-6,6] 并保证 >0。",
  "PERM 权重 35%、历史探索最少、边际收益最高（`资料库/12` §3.3）。",
  ["`E5/code/head_perm.py`"],
  ["`experiments/E5/perm/oof.npz`"],
  ["PERM Acc 提升（CI 下界 > 0）；无 ≤0 或非有限输出；给出「低估尾部 vs 官方评分器」一致性报告"],
  ["线性域建模 PERM","用 ReLU 输出导致零梯度"]),
 ("P2","SW 单尺度精修（训练折仿射归一化 + q_sw 硬切换）",
  "SW 连续头用训练折仿射归一化（`sw_norm=(sw−sw_mu)/sw_sigma`，`sw_cont=sw_mu+sw_sigma·head_out`，`sw_mu=median(valid sw)`、`sw_sigma=IQR(valid sw)/1.349`），输出反变换回标签尺度；逐目标原子头 `q_sw` 精确决定 `SW=99.9`。",
  "SW 是单一标签尺度但占位尖峰与有效分布形状差异极大；单头线性回归会被尖峰拉扯。",
  ["`E5/code/head_sw.py`"],
  ["`experiments/E5/sw/oof.npz`"],
  ["SW Acc 提升且占位峰无泄漏；单尺度守卫与仿射归一化反变换测试通过（`SW_SMALL_BRANCH` 永久关闭，无 ×100，无插值）"],
  ["把 SW 全局裁剪到 [0,1]（禁止；会掉约 23 分）","对连续头做 ×100 换算","在原子与连续之间插值"]),
],
"E6": [
 ("P0","逐目标原子头 + 辅助 joint 头（两阶段训练）",
  "训练逐目标原子头 `q_por/q_perm/q_sw`（主保护）+ 辅助 `q_joint`；两阶段：stage 1 训主干+原子头（`L_atom+λ_joint·L_joint`+0.05 连续 fallback），stage 2 冻结原子头（或 `q_head_lr_mult=0.05–0.1`）训连续头（按切片加权，权重永不为 0）。",
  "占位是 66.7% 的行，但单目标原子行（SW 31,030 / PERM 7,373 / POR 157）不是 joint；用一个 joint 概率同时决定三目标会漏保护或误伤。",
  ["`src/models/row_mlp.py`","`E6/code/train_state.py`"],
  ["`models/E6/state_*.pt`","`reports/E6_atomic_report.json`"],
  ["AUC ≥ 0.97 且逐目标原子 Acc/Precision/Recall/F1 双报；`min_atom_acc≥0.99`、`min_atom_recall≥0.98`；两阶段训练记录完整"],
  ["用单个 joint 头覆盖三目标","把 joint 头当唯一保护屏障"]),
 ("P1","逐目标原子门 τ_t 搜索（inner-OOF 官方总分）",
  "在 inner-OOF 上逐目标搜索门限 τ_t，目标函数是**官方总分** `argmax 100·w_t·Acc_t(τ)`，取最宽平台中点，实现硬切换并报告误判代价分解。",
  "τ 决定走常量还是连续，是纯 DL 管线唯一保护屏障的开关；F1/准确率与官方总分不等价。",
  ["`src/inference/atomic_gate.py`","`E6/code/search_tau.py`"],
  ["`reports/E6_tau_search.json`"],
  ["τ 只在 inner-OOF 选；报告 τ-分数曲线、平台、逐目标 atomic P/R/F1/acc、连续切片 acc 与误判代价；`joint_guard` 默认关；无任何插值"],
  ["用 outer 折或 A 榜选 τ","在常量与连续值之间插值","用 F1 而非官方总分选 τ"]),
 ("P2","PD1 完整管线组装与硬 Gate（≥82.0）",
  "组装 数据→主干→逐目标原子头→原子门→连续后处理→契约 的完整 PD1，产出 OOF、result.zip 与 manifest，判定 Gate ≥ 82.0。",
  "这是 v4 第一个可作为提交候选的完整管线。",
  ["`E6/code/build_pd1.py`","`E6/code/gate.py`"],
  ["`experiments/E6/P2/pd1/{result.json,result.zip,cv.json,manifest.json}`","`reports/E6_gate.json`"],
  ["OOF ≥ 82.0（`oof_total_min`）、`min_atom_acc≥0.99`、`min_atom_recall≥0.98`、`min_joint_atom_auc≥0.90`；契约通过；本机 CPU 可跑 `--use-version PD1`"],
  ["在管线中混入未冻结的特征版本","在原子/连续之间插值"]),
],
"E7": [
 ("P0","三段式损失消融（L_aux 归一化 / 边界聚焦 / PERM 截断）",
  "对 align/aux/ph 三组权重、λ₁ 退火曲线、归一化 `L_aux`、容差边界聚焦（`κ∈{0.5,1.0,2.0}`、`σ∈{0.15,0.25,0.35}`）与 PERM 官方截断做同结构消融。",
  "`资料库/12` §2.3 指出纯对齐损失早期梯度稀疏；SW 的 99.9 若不作尺度归一化会主导 `L_aux`。",
  ["`E7/code/ablate_loss.py`"],
  ["`reports/E7_loss_ablation.json`"],
  ["对齐损失 ≥ 纯 aux；`L_aux` 归一化与 PERM 截断消融完成；边界聚焦表完整（默认关）"],
  ["同时改多个损失项导致无法归因","把边界聚焦设为默认开启"]),
 ("P1","解码与后处理（逐目标期望分 / 敏感性）",
  "实现逐目标期望分/偏置/收缩解码，并在 inner-OOF 上选择；对 q 灰区用期望总分判断动作。",
  "零模型改动的换分手段，成本最低、风险最小。",
  ["`E7/code/decode_search.py`","`src/inference/decode.py`"],
  ["`reports/E7_decode_search.json`"],
  ["解码策略在 inner-OOF 选定并冻结；附敏感性曲线；对契约无副作用；不改变原子输出"],
  ["用 A 榜选解码参数"]),
],
"E8": [
 ("P0","MMoE 任务平衡",
  "用 MMoE 替换硬共享主干，比较逐目标 Acc 与梯度冲突指标。",
  "`资料库/08` §1.4：任务相关性弱时软共享更稳。",
  ["`src/models/mmoe.py`","`E8/code/train_mmoe.py`"],
  ["`experiments/E8/mmoe/*/oof.npz`"],
  ["至少一个目标提升且无目标退化；梯度冲突指标下降"],
  ["只报告总分而隐藏单目标退化"]),
 ("P1","井级分支与 transductive 消融（各一次）",
  "实现 H4 井级 attention-pool 偏置分支（小容量、强正则、校准参数只在 inner-OOF 选）；测试推理期 transductive 适配（伪标签/井级统计对齐），两者都只作消融。",
  "工程曲线只提供井间信号；80 井上井级校准极易过拟合，v1/v2 的井级路线 NO-GO 需在序列主干条件下重验一次。",
  ["`src/models/well_head.py`","`E8/code/well_branch.py`","`E8/code/pseudo_label.py`"],
  ["`reports/E8_well_branch.json`、`reports/E8_transductive.json`"],
  ["给出明确的采纳/NO-GO 结论与 CI；强制井级消融；伪标签不得使用测试标签"],
  ["把 transductive 适配伪装成\"训练时改进\""]),
 ("P2","集成（EMA/SWA / 快照 / 多结构）与 Gate",
  "EMA（decay∈{0.99,0.999,0.9995}，逐 epoch 用真实 `score.py` 在 inner-OOF 评估，保存 `ema.pt`）、SWA 对照、同折 top-k 快照与多结构集成，报告成员同源性与加权配对 bootstrap。",
  "集成的收益必须扣除同源性；增益 CI 含 0 即 NO-GO，不得用同源平均包装。",
  ["`src/ensemble/blend.py`","`E8/code/ensemble.py`"],
  ["`experiments/E8/ensemble/**`","`reports/E8_ensemble_report.json`、`reports/E8_gate.json`"],
  ["集成 ≥ 最佳单成员且 CI 下界 > 0；EMA/SWA/快照至少一个策略有逐折 delta 与 CI；同源报告齐备"],
  ["用同源模型平均制造假增益","把 CI 含 0 的增益包装成有效"]),
],
"E9": [
 ("P0","OOF 汇总与提交护栏判定",
  "汇总全部候选的 80 井 OOF、逐目标 Acc、bootstrap CI，运行 `choose_submission.py` 护栏。",
  "护栏防止\"本地漂亮但弱于历史锚点\"的候选被提交。",
  ["`E9/code/aggregate_oof.py`","`E9/code/choose_submission.py`"],
  ["`reports/E9_validation_report.json`、`reports/E9_submission_decision.json`"],
  ["守卫下限 = max(75, 80.382479−1.0, A 榜 82.2757−0.5 若已知)；判定可复算"],
  ["用 selection_score_only 数字伪装独立确认"]),
 ("P1","16 井次级体检与泄漏终审",
  "对 top-2 候选在 `folds_confirm` 上各跑一次；完成特征来源、折、标准化、伪标签四类泄漏审计。",
  "16 井是 v1-exposed 的次级体检；泄漏审计是防\"本地虚高\"的最后一道关卡。",
  ["`E9/code/confirm_check.py`","`E9/code/leakage_audit.py`"],
  ["`reports/E9_confirm.json`、`reports/E9_leakage_audit.json`"],
  ["无 > 1.5 分崩坏；四类泄漏审计全部通过或标出残余风险"],
  ["把 confirm 当独立确认宣称显著增益"]),
 ("P2","A 榜短名单仲裁（预算制 ≤3 次）",
  "按预算制提交 ≤3 次（留 1 次余量），登记 `E9_a_board_log.json`，只做仲裁与崩坏体检。",
  "A 榜 5 口井噪声 ±0.02；它只能筛掉崩坏候选，不能用于细粒度调优。",
  ["`E9/code/submit_batch.py`"],
  ["`reports/E9_a_board_log.json`"],
  ["每次提交记录 candidate_id/score/用途；无候选低于锚点 0.10 以上"],
  ["每日超过 5 次","根据 A 榜反馈改模型"]),
],
"E10": [
 ("P0","全量重训与 CPU 推理导出",
  "用 E9 通过候选的配置在全部 80 井上重训（或折集成），导出 CPU 可加载权重与 ONNX 兜底。",
  "复现要求\"训练+推理\"可独立运行；CPU 推理保证评测机兼容。",
  ["`E10/code/final_train.py`","`E10/code/export_cpu.py`"],
  ["`models/v4/final/*.pt`、`models/v4/final/*.onnx`"],
  ["CPU 加载并前向成功；两次前向 sha256 一致"],
  ["导出依赖 GPU","把训练日志写进模型目录导致包体膨胀"]),
 ("P1","打包、干净目录复现与 B0 fallback",
  "组装 `submission_code_v4.zip`、在只含 zip+data 的干净目录跑官方命令、比对结果；同时预构建 B0 fallback 包并验证。",
  "`rules.md` §8.4：复现失败即取消资格；纯 DL 管线必须准备保底包。",
  ["`E10/code/build_submission.py`","`E10/code/verify_inference.py`","`E10/code/build_b0_fallback.py`"],
  ["`submission/*.zip`、`reports/E10_reproduce_report.json`、`reports/E10_B0_fallback_manifest.json`"],
  ["逐点差 ≤ 1e-6；两次运行 sha256 一致；CPU < 30 min；fallback 包验证通过"],
  ["干净目录里依赖 v1/v2/v3 文件","跳过 fallback 构建"]),
 ("P2","提交执行与归档登记",
  "执行提交（预算内 1 次），登记 A 榜返回；把提交指纹写入候选注册表用于赛后复盘。",
  "提交是唯一外部动作，必须可追溯。",
  ["`E10/code/submit.py`"],
  ["`reports/E10_submission_log.json`"],
  ["记录 zip sha256、提交时间、返回分数；不据反馈改动模型"],
  ["提交后修改候选文件（破坏可追溯性）"]),
],
"E11": [
 ("P0","资产归档（含 sha256）",
  "生成代码/模型/候选/报告的清单与 sha256，确保可在不含 v1/v2/v3 的目录独立理解。",
  "归档是下一代的输入；无哈希的归档无法验证。",
  ["`E11/code/archive.py`"],
  ["`reports/E11_archive_manifest.json`"],
  ["清单完整、每项有 sha256；关键结论有单一事实源指针"],
  ["删除 rejected 候选"]),
 ("P1","复盘与下一代方向",
  "写三口径（本地 CV / A 榜 / B 榜）一致性分析、路线有效性表、资源统计与下一代方向储备。",
  "失败与成功的路线都必须留证；资源统计决定下一代预算分配。",
  ["`E11/code/retrospective.py`"],
  ["`reports/E11_retrospective.md`、`reports/E11_next_directions.md`"],
  ["含三口径对照表、有效性表、资源表；方向清单含触发条件"],
  ["把经验性结论写成确定性结论"]),
],
}


def w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_e(st: dict) -> str:
    code = f"v4/{st['e']}/src/、v4/{st['e']}/code/"
    art = f"v4/experiments/{st['e']}/、v4/models/{st['e']}/"
    L: list[str] = []
    L.append(f"# {st['e']} {st['name']}\n")
    L.append(HEADER.format(up="../", code=code, art=art))
    L.append(f"> 阶段性质：{st['nature']}")
    L.append("")
    L.append("> **执行状态**以 [`versions/status.json`](../versions/status.json) 为唯一事实源；"
             "P 级计划的详细内容由 [`docs/gen_p_details.py`](../docs/gen_p_details.py) 生成。")
    L.append("")
    L.append("## 1. 目标\n")
    L.append(st["goal"] + "\n")
    L.append("## 2. 为什么这么设计\n")
    for i, x in enumerate(st["why"], 1):
        L.append(f"{i}. {x}")
    L.append("")
    L.append("## 3. 输入\n")
    for x in st["inputs"]:
        L.append(f"- {x}")
    L.append("")
    L.append("## 4. 产物\n")
    for x in st["outputs"]:
        L.append(f"- {x}")
    L.append("")
    L.append("## 5. 代码归属\n")
    for x in st["code"]:
        L.append(f"- {x}")
    L.append("")
    L.append("## 6. P 级子计划\n")
    for pid, pname, *_ in P_STAGES[st["e"]]:
        L.append(f"- [{pid} {pname}]({pid}/PLAN.md)")
    L.append("")
    L.append("## 7. 完成判据（Gate）\n")
    for x in st["done"]:
        L.append(f"- {x}")
    L.append("")
    L.append("## 8. 禁止事项\n")
    for x in st["forbid"]:
        L.append(f"- {x}")
    L.append("")
    L.append("## 9. 通用约束（继承总计划）\n")
    L.append("- 训练/推理分离：本机（无 NPU/GPU）负责代码与契约，云端（1×Ascend 910B 64GB，CANN 8.3rc2 / torch 2.8.0 + torch_npu 2.8.0 / py3.11 / arm64，**64 GiB 磁盘**）负责训练。")
    L.append("- 禁止 `pip install torch` 或变更镜像基础栈；额外轻量包须 `--no-cache-dir` 并清缓存。")
    L.append("- 一切阈值/权重/早停只在 **inner-OOF** 上选；outer 折只推理一次。")
    L.append("- 训练脚本必须支持 `--resume`、`--time-budget-h`、每 epoch checkpoint 与 `assert_disk_headroom(8.0)`。")
    L.append("- 所有 Gate 的 `mandatory_checks` 必须含 `contract_ok`、`atomic_precision_reported`、`disk_budget_ok`、`training_time_log_valid`。")
    return "\n".join(L) + "\n"


def render_p(e: str, p: tuple) -> str:
    pid, name, goal, why, code, outs, done, forbid = p
    L: list[str] = []
    L.append(f"# {e}/{pid} {name}\n")
    L.append(f"> 所属阶段：[{e}](../PLAN.md)　|　总计划：[v4/PLAN.md](../../../PLAN.md)\n")
    L.append("> 本目录是最小可执行单元：`code/` 放本 P 专属脚本，`docs/` 放本 P 的说明与结论。\n")
    L.append("## 1. 目标\n")
    L.append(goal + "\n")
    L.append("## 2. 为什么需要这一步\n")
    L.append(why + "\n")
    L.append("## 3. 代码\n")
    for x in code:
        L.append(f"- {x}")
    L.append("")
    L.append("## 4. 产物\n")
    for x in outs:
        L.append(f"- {x}")
    L.append("")
    L.append("## 5. 完成判据\n")
    for x in done:
        L.append(f"- {x}")
    L.append("")
    L.append("## 6. 禁止事项\n")
    for x in forbid:
        L.append(f"- {x}")
    L.append("")
    L.append("## 7. 执行提示\n")
    L.append("- 先跑最小可复算版本（单折 / 小样本），确认口径正确后再全量。")
    L.append("- 结果写入本 P 的 `docs/` 与 `v4/reports/{}_{}_*.json`。".format(e, pid))
    L.append("- 预注册 Gate 后才允许看结果；预注册文件为 `v4/reports/{}_{}_gate_prereg.json`。".format(e, pid))
    L.append("- 任何结论必须附「复算命令」与「产物 sha256」。")
    return "\n".join(L) + "\n"


def main() -> None:
    n = 0
    for st in E_STAGES:
        e = st["e"]
        w(ROOT / e / "PLAN.md", render_e(st))
        n += 1
        for p in P_STAGES[e]:
            w(ROOT / e / p[0] / "PLAN.md", render_p(e, p))
            n += 1
    print(f"wrote {n} PLAN.md files")


if __name__ == "__main__":
    main()
