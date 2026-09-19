#!/usr/bin/env bash
# v4 云端一次性依赖安装（30 GB 磁盘纪律版）
# 用法： bash v4/E0/code/setup_deps.sh [--dry-run]
set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

cd "$(dirname "$0")/../.."   # -> v4/
REPORTS_DIR="${V4_REPORTS_DIR:-/data/v4/reports}"
CACHE_ROOT="${V4_CACHE_ROOT:-/data/v4/cache}"
mkdir -p "$REPORTS_DIR" 2>/dev/null || REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR"
echo "reports dir: $REPORTS_DIR"

echo "== v4 setup_deps =="
echo "python: $(python3 -V 2>&1)  ($(command -v python3))"

# 0) 环境与磁盘先体检（缺 torch 时 check_env 会以非 0 退出，此处只做信息采集）
python3 E0/code/check_env.py --profile base --json "$REPORTS_DIR/E0_env.json" || true
python3 src/data/disk_guard.py --min-free-gb 8 \
        --report "$CACHE_ROOT" \
        --json "$REPORTS_DIR/E0_disk_budget.json" || true

FREE_GB=$(python3 - <<'PY'
import shutil
print(round(shutil.disk_usage('/').free / 1024**3, 2))
PY
)
echo "free disk: ${FREE_GB} GiB"

if python3 -c "import sys; sys.exit(0 if float('${FREE_GB}') < 8 else 1)"; then
  echo "!! free < 8 GiB —— 按 PLAN.md §3.4.1 的安全规则：不安装任何额外包，仅使用镜像自带 torch+numpy。"
  echo "   请先清理后重跑本脚本。"
  exit 2
fi

# 1) 安装纪律：绝不触碰 torch / nvidia-* / cuda-*
export PIP_NO_CACHE_DIR=1

# 优先使用 lock 文件里的**精确版本**（R2-M6：范围约束与 lock 不一定一致）
LOCK="versions/locks/cloud.txt"
if [[ -f "$LOCK" ]]; then
  mapfile -t PKGS < <(grep -vE '^\s*#|^\s*$|^torch|^numpy' "$LOCK")
  echo "使用 lock 精确版本: ${#PKGS[@]} 个包（torch/numpy 由镜像提供，跳过）"
else
  PKGS=(
    "pandas==2.2.3" "pyarrow==17.0.0" "scipy==1.13.1"
    "scikit-learn==1.5.2" "einops==0.8.0"
    "onnx==1.16.2" "onnxruntime==1.18.1"
  )
fi

if [[ "$DRY" == "1" ]]; then
  printf 'would run: python3 -m pip install --no-cache-dir %s\n' "${PKGS[*]}"
  exit 0
fi

# 2) 预检：确认 pip 不会顺带替换 torch
if python3 -m pip install --no-cache-dir --dry-run "${PKGS[@]}" 2>/dev/null \
     | grep -Ei 'torch|nvidia-|cuda-' ; then
  echo "!! pip 计划触碰 torch/nvidia/cuda 系列 —— 违反安装纪律，已中止。"
  echo "   请改用镜像自带版本，不要安装这些包。"
  exit 3
fi

python3 -m pip install --no-cache-dir "${PKGS[@]}"
python3 -m pip cache purge || true

# 3) 冻结事实（写**持久**目录，任务结束不丢）
python3 -m pip freeze > "$REPORTS_DIR/cloud_frozen.txt"
cp -f "$REPORTS_DIR/cloud_frozen.txt" versions/locks/cloud_frozen.txt 2>/dev/null || true
python3 E0/code/check_env.py --profile base --json "$REPORTS_DIR/E0_env.json" || true
python3 src/data/disk_guard.py --min-free-gb 8 --json "$REPORTS_DIR/E0_disk_budget.json" || true

echo "== done =="
df -h / | tail -1
