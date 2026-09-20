"""v4 冻结常量（**唯一事实源**）。

任何模块需要数据列名、哨兵值、占位常量、评分权重或折路径时，都必须从这里取，
不得在别处硬编码字面量。修改本文件等于升 E0 契约版本。

只依赖标准库，便于在无 numpy/torch 的环境导入。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 数据列
COLUMNS: tuple[str, ...] = (
    "DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT",
    "DEVI", "AZIM", "BIT", "CASE",          # 14 个输入字段 = DEPTH + 13 条曲线
    "POR", "PERM", "SW",                     # 3 目标
)
# E0-R2 修正：DEPTH 是深度基准、单独使用；**13 条曲线**才是模型输入通道。
# 此前写 COLUMNS[:14]（含 DEPTH，共 14 列）与解析器的 arr[:,1:15] 叠加，
# 会让第 14 个"输入"落在 POR 标签上（80 口井全部泄漏），并让测试井只有 13 列。
DEPTH_COLUMN: str = "DEPTH"
INPUT_COLUMNS: tuple[str, ...] = COLUMNS[1:14]        # GR..CASE，共 13
TARGET_COLUMNS: tuple[str, ...] = ("POR", "PERM", "SW")
N_INPUT = len(INPUT_COLUMNS)                          # 13
N_TARGET = len(TARGET_COLUMNS)                        # 3

# ---------------------------------------------------------------- 哨兵
# 项目冻结工作假设（沿用 v2/reports/E0_data_card.json）：
#   -99999 / -9999 / NaN / 任何 < -1000 的原始值都视为缺测。
SENTINELS: tuple[float, ...] = (-99999.0, -9999.0)
MISSING_LT: float = -1000.0

# ---------------------------------------------------------------- 标签状态
# 联合常量占位（rules.md §5.3 项目决策：不剔除，参与全量评分）
PLACEHOLDER: dict[str, float] = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
PLACEHOLDER_ABS_TOL: float = 1e-9

# ---------------------------------------------------------------- 原子哨兵表（唯一事实源）
# 三目标的常量哨兵值；这些行是**真实监督目标**（既不删除也不掩码），
# 由模型侧"原子头 q_t"负责把它们从连续头手里抢回来。
# 顺序全局固定为 (POR, PERM, SW)，q_atom 的列序即此顺序。
ATOM_VALUES: dict[str, float] = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
TARGETS: tuple[str, ...] = ("POR", "PERM", "SW")
TARGET_WEIGHTS: tuple[float, ...] = (0.30, 0.35, 0.35)   # 官方 Total 权重（同 SCORE_WEIGHTS）

# 评分权重与容差（rules.md §7.3/§7.4）
SCORE_WEIGHTS: dict[str, float] = {"POR": 0.30, "PERM": 0.35, "SW": 0.35}
DELTA_POR: float = 0.08
DELTA_SW: float = 0.05
EPS: float = 1e-3          # 资料库/12 §2.4 提示 2：取 1e-3，不用 1e-6
PERM_LOG_MIN: float = -6.0
PERM_LOG_MAX: float = 6.0

# ---------------------------------------------------------------- 连续头参数化（E1-R3 冻结）
# POR : por = por_max * sigmoid(g)，por_max = POR_MAX_BUFFER * valid_por_max。
#       por_max 是 **buffer（不可学习）**：输出天然非负、可精确趋近 0（真实 0.0 存在），
#       且上界留有 20% 缓冲。**严禁 `0.1 + softplus(g)`** —— 那会把下界锁死在 0.1。
POR_MAX_BUFFER: float = 1.2
POR_VALID_MAX: float = 33.177     # 训练折有效 POR 最大值（por_max = 1.2 * 该值 ≈ 39.8）
# PERM: perm_z = 6 * tanh(g)，log10 空间夹到 [-6, 6]，严格保证 PERM > 0。
# SW  : sw = sw_mu + sw_sigma * g（训练折仿射反归一化），sw 为标签尺度（百分数）。

# ---------------------------------------------------------------- SW 尺度（E0-R2 冻结结论）
# **唯一权威结论**：SW 是**单一标签尺度（百分数）**。
#   80 口训练井实测有效 SW：min=8.305, median=82.805, max=99.9；
#   有效行中 SW<1 的行数为 **0**。
# "SW 双尺度（占位 99.9 = 百分数、有效值 [0,1] = 小数）"是**已被数据证伪的假设**，
# 任何文档/代码若仍这样断言都是错的，必须删除；**永远不得把 SW 裁剪到 [0,1]**。
# SW_SMALL_BRANCH 是这条被证伪路径的**遗留对照开关，永久关闭（False）**；
# SW_SMALL_BRANCH_SCALE 仅当该开关被显式打开时才有意义（保留仅为兼容旧接口）。
SW_PLACEHOLDER: float = 99.9
SW_LABEL_RANGE: tuple[float, float] = (0.0, 100.0)   # 标签尺度软上界（实测有效 8.305–99.9）
SW_VALID_MIN: float = 8.305                          # 实测有效行最小值（E0 复算）
SW_VALID_MEDIAN: float = 82.805                      # 实测有效行中位数（E0 复算）
SW_MIN_OBSERVED: float = SW_VALID_MIN                # 兼容旧名（= SW_VALID_MIN）
SW_SMALL_BRANCH: bool = False                        # 遗留对照开关，永久 False
SW_SMALL_BRANCH_SCALE: float = 100.0                  # 仅当 SW_SMALL_BRANCH=True 时用于换算

# R4-B3：提交契约的 SW **低值守卫**判据（"中位数守卫"可被 66.7% 原子行绕过）。
#   实测事实：训练标签里 SW < 1 的行数为 **0**，有效最小 = 8.305。
#   契约不能只看全体中位数：当 2/3 行是原子 99.9 时，中位数被拉到 99.9，
#   剩余连续分支即使被错误归一化到 [0,1] 也照样"通过"。
SW_LOW_GUARD_ABS: float = 1.0            # 低于此值的单行预测即视为**明确**量纲错误
SW_LOW_GUARD_FRAC_MAX: float = 1e-3      # 允许的极少数离群比例（95,948 行 ≈ 95 行）
SW_SUSPECT_FRAC_MAX: float = 0.01        # 低于 SW_VALID_MIN 的行数占比上限（允许 <1% 边界外推）
SW_LOW_GUARD_NONATOM_P05_MIN: float = SW_VALID_MIN   # 非原子行 p05 不得低于有效最小值
SW_LOW_GUARD_MIN_NONATOM: int = 20       # 非原子行少于此数时跳过 p05 守卫（小样例过度敏感）


def sw_to_norm(sw: float, mu: float = SW_VALID_MEDIAN, sigma: float = 20.0) -> float:
    """训练折仿射归一化：`z = (sw − mu) / sigma`（纯 python）。

    `mu`/`sigma` **必须**来自训练折的有效 SW 统计（见
    `features.basic.fit_target_scalers`），并写入 checkpoint manifest / scaler JSON。
    """
    if float(sigma) == 0.0:
        raise ZeroDivisionError("sw_to_norm: sigma must be non-zero")
    return (float(sw) - float(mu)) / float(sigma)


def sw_from_norm(z: float, mu: float = SW_VALID_MEDIAN, sigma: float = 20.0) -> float:
    """`sw_to_norm` 的逆：`sw = mu + sigma * z`（纯 python，与前者严格互逆）。"""
    return float(z) * float(sigma) + float(mu)

# ---------------------------------------------------------------- 提交契约
EXPECTED_N_TEST_WELLS: int = 10
EXPECTED_N_TEST_ROWS: int = 95_948
# 训练集契约值（E0/P1 复算：`wc -l data/train/*.txt` = 730,428 − 80×2 行表头）。
# 由 `tools/bootstrap_data.sh`（数据部署硬校验）与 `tools/check_data_leak.py`
# （泄漏回归的覆盖性断言）共同引用 —— 二者都**不得**再硬编码字面量。
EXPECTED_N_TRAIN_WELLS: int = 80
EXPECTED_N_TRAIN_ROWS: int = 730_268
RESULT_TOP_KEYS: tuple[str, ...] = ("modelId", "modelName", "version", "resultData")
PREDICTION_KEYS: tuple[str, ...] = ("depth", "POR", "PERM", "SW")
DEPTH_DECIMALS: int = 1

# ---------------------------------------------------------------- 已知锚点（外部参照，非本地已复算值）
# E0/P2 本机复算结论（2026-09-19）：
#   常数基线 (0.1, 0.01, 99.9) 在 missing_mode="drop" 下 = 70.490735（命中锚点 ±1e-4）；
#   在 missing_mode="mask" 下 = 69.843218（低 0.65 分）。
#   => 官方分母口径为"逐目标排除缺测行"，即 SCORE_MISSING_MODE 必须为 "drop"。
CONSTANT_BASELINE_OOF: float = 70.4907
SCORE_MISSING_MODE: str = "drop"
CONSTANT_BASELINE_MASK_MODE: float = 69.843218
B0_LOCAL_OOF: float = 80.382479            # v1 E7，protocol_matched=false
B0_A_BOARD: float = 82.2757
V2_BEST_DEV64_OOF: float = 80.305222
V2_BEST_A_BOARD: float = 82.3035
GUARDRAIL_TOLERANCE: float = 1.0
GUARDRAIL_ABOARD_MARGIN: float = 0.5
PASS_LINE: float = 75.0

# ---------------------------------------------------------------- 磁盘/内存预算（云端）
# 2026-09-20 平台规格变更：Ascend910B-1-64G → 磁盘 64 GiB（原 A100 规格为 30 GB）。
# 与 `src/hardware.py::PLATFORM["disk_gb"]` 同源，由 tests/test_hardware.py 锁定不得漂移。
DISK_BUDGET_GB: float = 64.0
DISK_MIN_FREE_GB: float = 8.0
DISK_CLEANUP_GB: float = 5.0
DISK_ABORT_GB: float = 3.0
RAM_BUDGET_GB: float = 16.0
DATALOADER_WORKERS: int = 4

# ---------------------------------------------------------------- 折与路径
WELL_FOLDS_SOURCE: str = "v1_well_folds.json"
N_FOLDS: int = 5
N_INNER_FOLDS: int = 3
