#!/usr/bin/env bash
# v4 云端一次性依赖安装（30 GB 磁盘纪律版）
# 用法： bash v4/E0/code/setup_deps.sh [--dry-run]
set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

cd "$(dirname "$0")/../.."   # -> v4/

echo "== v4 setup_deps =="
echo "python: $(python3 -V 2>&1)  ($(command -v python3))"

# 0) 环境与磁盘先体检（缺 torch 时 check_env 会以非 0 退出，此处只做信息采集）
python3 E0/code/check_env.py --json reports/E0_env.json || true
python3 src/data/disk_guard.py --min-free-gb 8 \
        --report "${HOME:-/root}" \
        --json reports/E0_disk_budget.json || true

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

PKGS=(
  "pandas>=2.0,<3"
  "pyarrow>=15"
  "scipy>=1.11"
  "scikit-learn>=1.4"
  "einops>=0.7"
  "onnx>=1.16"
  "onnxruntime>=1.17"
)

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

# 3) 冻结事实
python3 -m pip freeze > versions/locks/cloud_frozen.txt
python3 E0/code/check_env.py --json reports/E0_env.json
python3 src/data/disk_guard.py --min-free-gb 8 --json reports/E0_disk_budget.json

echo "== done =="
df -h / | tail -1
