#!/usr/bin/env bash
# v4 云端一次性依赖安装（30 GB 磁盘纪律版）
# 用法： bash v4/E0/code/setup_deps.sh [--dry-run]
set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

cd "$(dirname "$0")/../.."   # -> v4/
DATA_ROOT="${V4_DATA_ROOT:-/data}"
REPORTS_DIR="${V4_REPORTS_DIR:-/data/v4/reports}"
CACHE_ROOT="${V4_CACHE_ROOT:-/data/v4/cache}"
mkdir -p "$REPORTS_DIR" 2>/dev/null || REPORTS_DIR="reports"
mkdir -p "$REPORTS_DIR"

# R3 修复：磁盘体检的权威路径必须是 data root（云端 /data 的 30 GB 配额），
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

if python3 -c "import sys; sys.exit(0 if float('${FREE_GB}') < 8 else 1)"; then
  echo "!! free < 8 GiB —— 按 PLAN.md §3.4.1 的安全规则：不安装任何额外包，仅使用镜像自带 torch+numpy。"
  echo "   请先清理后重跑本脚本。"
  exit 2
fi

# 1) 安装纪律：绝不触碰 torch / nvidia-* / cuda-*
export PIP_NO_CACHE_DIR=1

# 优先使用 lock 文件里的**精确版本**（R2-M6：范围约束与 lock 不一定一致）
# R5-M1/M2：lock 现在只列包名（版本不钉死）；**只排除 torch**（镜像提供，禁止 pip 触碰）。
#   numpy 不再被排除：它虽随 torch 提供，但也在 REQUIRED_PY_DEPS / requirements.txt /
#   用户 pip 清单里。若镜像哪天不带 numpy，pip 必须能补上（存在时 pip 只会打印
#   "Requirement already satisfied"，不会改动 torch 的 ABI）。
LOCK="versions/locks/cloud.txt"
if [[ -f "$LOCK" ]]; then
  mapfile -t PKGS < <(grep -vE '^\s*#|^\s*$|^torch' "$LOCK" | sed -E 's/[[:space:]]*#.*$//' | grep -vE '^\s*$')
  echo "使用 lock 清单: ${#PKGS[@]} 个包（仅 torch 由镜像提供，跳过）"
else
  # 与 check_env.py::REQUIRED_PY_DEPS 保持一致（四审 R5-M1：pyarrow 已移出 required）
  PKGS=( "numpy" "pandas" "scipy" "scikit-learn" "einops" )
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
python3 src/data/disk_guard.py --min-free-gb 8 \
        --data-root "$DATA_ROOT" --path "$DATA_ROOT" --path "$CACHE_ROOT" \
        --json "$REPORTS_DIR/E0_disk_budget.json" || true

echo "== done =="
df -h "$DATA_ROOT" / 2>/dev/null | tail -n +1 || true
