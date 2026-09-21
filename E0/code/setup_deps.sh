#!/usr/bin/env bash
# v4 云端一次性依赖安装（30 GB 云盘纪律版；Ascend 910B / CANN 8.3rc2；64 GiB 是显存）
#
# 用法（**本机**：在项目父目录执行，`v4/` 是本机真实的目录名；
#       云端由 run_train.sh 以 $HERE 调用，不要照抄下面的 `v4/` 前缀）：
#   bash v4/E0/code/setup_deps.sh                 # 按 versions/locks/cloud.txt（**不钉版本**）
#   bash v4/E0/code/setup_deps.sh --dry-run       # 只预检并打印将要安装的包
#   bash v4/E0/code/setup_deps.sh --from-frozen   # 按 cloud_frozen.txt 的**精确版本**安装
#
# 为什么默认不钉版本、以及"什么时候才钉"：
#   py3.11 + 平台镜像下 pip 会解析出哪个小版本，只有实机才知道；猜错会让镜像构建失败。
#   所以第一次不钉（让 pip 解析），装完立刻 `pip freeze` 写进
#   `versions/locks/cloud_frozen.txt`（本脚本第 3 步自动做）；需要**重建镜像 / 复现提交**时
#   用 `--from-frozen` 按那份实测版本精确安装（torch/torch_npu/npu/ascend/cann/nvidia/cuda 系列由白名单天然排除）。
set -euo pipefail

DRY=0
FROM_FROZEN=0
for _a in "$@"; do
  case "$_a" in
    --dry-run)     DRY=1 ;;
    --from-frozen) FROM_FROZEN=1 ;;
    *) echo "unknown arg: $_a"; exit 2 ;;
  esac
done

cd "$(dirname "$0")/../.."   # -> v4/（review R7：此前重复 cd 了两次）
# 可用 `V4_FROZEN_LOCK` 指向别处的 freeze 文件（默认 repo 内那份；相对路径按 v4/ 解析）
FROZEN="${V4_FROZEN_LOCK:-versions/locks/cloud_frozen.txt}"
DATA_ROOT="${V4_DATA_ROOT:-/data}"
REPORTS_DIR="${V4_REPORTS_DIR:-/data/v4/reports}"
CACHE_ROOT="${V4_CACHE_ROOT:-/data/v4/cache}"
mkdir -p "$REPORTS_DIR" 2>/dev/null || REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR"

# R3 修复：磁盘体检的权威路径必须是 data root（云端 /data 的 30 GB 云盘配额），
# 不能默认检查 `/`。本机开发环境若 /data 与 $V4_DATA_ROOT 都不存在，则降级到
# repo 目录，并明确打印**实际使用**的路径（而不是崩溃）。
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "WARN: DATA_ROOT='$DATA_ROOT' 不存在（本机开发环境？）——磁盘体检降级到 repo 目录。"
  DATA_ROOT="$PWD"
fi
echo "data root:  $DATA_ROOT    (E0_disk_budget.json::level 的口径)"
echo "reports dir: $REPORTS_DIR"

echo "== v4 setup_deps =="
echo "python: $(python3 -V 2>&1)  ($(command -v python3))"

# 0) 环境与磁盘先体检（缺 torch 时 check_env 会以非 0 退出，此处只做信息采集）
#    R5 修复：`--dry-run` 是**预检**，不得写任何证据文件 —— 否则在开发机上会把
#    "本机无 torch / python 3.12" 的结论写进 repo 的 reports/ 快照，污染 review 证据。
if [[ "$DRY" == "1" ]]; then
  python3 E0/code/check_env.py --profile base || true
  python3 src/data/disk_guard.py --min-free-gb 8 \
          --data-root "$DATA_ROOT" --path "$DATA_ROOT" --path "$CACHE_ROOT" \
          --report "$CACHE_ROOT" || true
else
  python3 E0/code/check_env.py --profile base --json "$REPORTS_DIR/E0_env.json" || true
  python3 src/data/disk_guard.py --min-free-gb 8 \
          --data-root "$DATA_ROOT" --path "$DATA_ROOT" --path "$CACHE_ROOT" \
          --report "$CACHE_ROOT" \
          --json "$REPORTS_DIR/E0_disk_budget.json" || true
fi

FREE_GB=$(DATA_ROOT="$DATA_ROOT" python3 - <<'PY'
import os, shutil
p = os.environ.get("DATA_ROOT") or "/"
try:
    print(round(shutil.disk_usage(p).free / 1024**3, 2))
except OSError:
    print("nan")   # 不阻塞：磁盘信息采集失败时按"未知"处理
PY
)
echo "free disk on $DATA_ROOT: ${FREE_GB} GiB"

# review R7：原写法 `sys.exit(0 if float(x) < 8 else 1)` 语义反读，且 `nan < 8` 为 False
# 会把"测不出余量"当成"余量充足"直接放行。现在写成"足够才继续"，并把 nan 视为不足：
# 拿不到磁盘事实时按 PLAN.md §3.4.1「先测后用，不许假设」处理 —— 装包前必须能证明有空间。
if python3 -c "
import sys
try:
    free = float('${FREE_GB}')
except ValueError:
    free = float('nan')          # 无法测量 -> 视同不足
sys.exit(0 if free >= 8 else 1)
"; then
  echo "磁盘余量 ${FREE_GB} GiB >= 8 GiB，继续安装。"
else
  echo "!! 可用余量 ${FREE_GB} GiB < 8 GiB（或无法测量）—— 按 PLAN.md §3.4.1 的安全规则："
  echo "   不安装任何额外包，仅使用镜像自带 torch + torch_npu（numpy 是口径层硬前提，缺它也必须先补上）。"
  echo "   请先清理 $CACHE_ROOT 或旧 checkpoint 后重跑本脚本。"
  exit 2
fi

# 1) 安装纪律：绝不触碰 torch / torch_npu / nvidia-* / cuda-* / ascend* / cann*
export PIP_NO_CACHE_DIR=1

# 包清单三选一：
#   a) --from-frozen：`pip freeze` 的**实测精确版本**（重建镜像/复现提交时用；
#      E0/code/frozen_pins.py 用白名单挑直接依赖，构造上不含 torch/torch_npu/
#      nvidia/cuda/triton/ascend/cann —— 目标平台是 Ascend + CANN）
#   b) versions/locks/cloud.txt：默认，**只列包名、不钉版本**（让 pip 解析当前镜像的兼容版本）
#   c) 兜底硬编码列表：与 check_env.py::REQUIRED_PY_DEPS 保持一致（lock 缺失时）
LOCK="versions/locks/cloud.txt"
if [[ "$FROM_FROZEN" == "1" ]]; then
  PINS="$(python3 E0/code/frozen_pins.py --frozen "$FROZEN")" || {
    echo "!! --from-frozen 需要可用的 frozen lock：$FROZEN"
    echo "   先跑一次不带参数的本脚本（第 3 步会自动写 cloud_frozen.txt），"
    echo "   或用 V4_FROZEN_LOCK=<path> 指向那份 pip freeze 输出。"
    exit 2
  }
  mapfile -t PKGS <<< "$PINS"
  echo "使用 frozen lock 的精确版本: ${#PKGS[@]} 个包（$FROZEN）"
  printf '  %s\n' "${PKGS[@]}"
elif [[ -f "$LOCK" ]]; then
  # 只跳过**加速栈**（torch/torch_npu/npu/ascend/cann 系列），其余照装
  mapfile -t PKGS < <(grep -vE '^\s*#|^\s*$' "$LOCK" \
    | sed -E 's/[[:space:]]*#.*$//' \
    | grep -vE '^\s*$' \
    | grep -viE '^(torch|torch[_-]npu|npu|ascend|cann|nvidia-|cuda-|triton|apex|deepspeed|flash-attn|xformers)')
  echo "使用 lock 清单: ${#PKGS[@]} 个包（不钉版本；加速栈 torch/torch_npu/npu/ascend/cann 由镜像提供，跳过）"
else
  # 与 check_env.py::REQUIRED_PY_DEPS 保持一致（四审 R5-M1：pyarrow 已移出 required）
  PKGS=( "numpy" "pandas" "scipy" "scikit-learn" "einops" )
fi

if [[ "$DRY" == "1" ]]; then
  printf 'would run: python3 -m pip install --no-cache-dir %s\n' "${PKGS[*]}"
  exit 0
fi

# 2) 预检：确认 pip 不会顺带替换加速栈（torch / torch_npu / ascend / cann / nvidia / cuda）
if python3 -m pip install --no-cache-dir --dry-run "${PKGS[@]}" 2>/dev/null \
     | grep -Ei 'torch|npu|ascend|cann|nvidia-|cuda-|triton' ; then
  echo "!! pip 计划触碰加速栈（torch/torch_npu/npu/ascend/cann/nvidia/cuda）—— 违反安装纪律，已中止。"
  echo "   请改用镜像自带版本，不要安装这些包。"
  exit 3
fi

python3 -m pip install --no-cache-dir "${PKGS[@]}"
python3 -m pip cache purge 2>/dev/null || true

# 3) 冻结事实（写**持久**目录，任务结束不丢）
python3 -m pip freeze > "$REPORTS_DIR/cloud_frozen.txt"
cp -f "$REPORTS_DIR/cloud_frozen.txt" versions/locks/cloud_frozen.txt 2>/dev/null || true
python3 E0/code/check_env.py --profile base --json "$REPORTS_DIR/E0_env.json" || true
python3 src/data/disk_guard.py --min-free-gb 8 \
        --data-root "$DATA_ROOT" --path "$DATA_ROOT" --path "$CACHE_ROOT" \
        --json "$REPORTS_DIR/E0_disk_budget.json" || true

echo "== done =="
df -h "$DATA_ROOT" / 2>/dev/null | tail -n +1 || true
