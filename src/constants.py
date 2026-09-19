"""v4 冻结常量（**唯一事实源**）。

任何模块需要数据列名、哨兵值、占位常量、评分权重或折路径时，都必须从这里取，
不得在别处硬编码字面量。修改本文件等于升 E0 契约版本。

只依赖标准库，便于在无 numpy/torch 的环境导入。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 数据列
COLUMNS: tuple[str, ...] = (
    "DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT",
    "DEVI", "AZIM", "BIT", "CASE",          # 14 输入曲线（DEPTH 为深度基准）
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

# 评分权重与容差（rules.md §7.3/§7.4）
SCORE_WEIGHTS: dict[str, float] = {"POR": 0.30, "PERM": 0.35, "SW": 0.35}
DELTA_POR: float = 0.08
DELTA_SW: float = 0.05
EPS: float = 1e-3          # 资料库/12 §2.4 提示 2：取 1e-3，不用 1e-6
PERM_LOG_MIN: float = -6.0
PERM_LOG_MAX: float = 6.0

# ---------------------------------------------------------------- SW 尺度（E0-R2 修正）
# **此前假设错误**：文档曾写"SW 占位 99.9（百分数）、有效值 [0,1]（小数）"，属双尺度。
# 80 口训练井实测（非缺测且非联合占位行）：SW min=8.305, median=82.804, max=99.9，
# SW<1 的行数为 **11**（占有效行 6.8e-5）——即 SW 与 POR/PERM 一样是**单一标签尺度（百分数）**。
# 因此取消"×100 双尺度换算"这一前提，H3/SW 头直接输出标签尺度。
SW_PLACEHOLDER: float = 99.9
SW_LABEL_RANGE: tuple[float, float] = (0.0, 100.0)   # 标签尺度上界（实测有效值 8.3–99.9）
SW_MIN_OBSERVED: float = 8.305                        # 实测有效行最小值（E0 复算）
SW_SMALL_BRANCH: bool = False                         # 是否启用 [0,1] 小值分支（默认关闭）
SW_SMALL_BRANCH_SCALE: float = 100.0                  # 仅当 SW_SMALL_BRANCH=True 时用于换算

# ---------------------------------------------------------------- 提交契约
EXPECTED_N_TEST_WELLS: int = 10
EXPECTED_N_TEST_ROWS: int = 95_948
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
DISK_BUDGET_GB: float = 30.0
DISK_MIN_FREE_GB: float = 8.0
DISK_CLEANUP_GB: float = 5.0
DISK_ABORT_GB: float = 3.0
RAM_BUDGET_GB: float = 16.0
DATALOADER_WORKERS: int = 4

# ---------------------------------------------------------------- 折与路径
WELL_FOLDS_SOURCE: str = "v1_well_folds.json"
N_FOLDS: int = 5
N_INNER_FOLDS: int = 3
