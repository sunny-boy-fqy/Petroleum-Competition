#!/usr/bin/env bash
# =============================================================================
# v4 平台训练任务统一入口（Git 仓库代码来源）
# =============================================================================
# 平台约定（见 docs/platform_setup.md）：
#   - Git 仓库代码被克隆到**临时**目录 /code/workspace（任务结束即丢）
#   - 云盘挂载在**持久**目录 /data（只有这里的内容会保留）
#   - 启动命令最长 500 字符，因此本脚本承担全部编排逻辑
#
# 平台【启动命令】填：
#     bash /code/workspace/v4/run_train.sh --mode all
#
# 常用模式：
#   --mode env      环境自检 + 安装额外轻量依赖（不碰 torch）
#   --mode data     把 repo 内 dist/v4_data.tar.gz 部署到 /data/v4/data（只需一次）
#   --mode e0       E0 口径复算（数据卡 / 评分锚点 / 折指纹 / 契约自检）
#   --mode data-health  数据健康硬校验（profile=full）
#   --mode smoke    5 分钟极小规模冒烟（1 折 / 少量 epoch），验证全链路可跑
#   --mode stage --stage E1   训练指定阶段
#   --mode all      依次执行 env -> data -> e0 -> E1 ...
#
# 平台集成（已按官方提示落实）：
#   * TensorBoard：导出 TENSORBOARD_LOGDIR=$V4_DATA_ROOT/v4/tb，
#     训练脚本用 src/training/tb_logger.py::RunLogger 写指标，平台任务详情页可见曲线。
#   * 云盘持久化：所有产物（cache/runs/reports/logs/tb）都在 $V4_DATA_ROOT(=/data)/v4 下，
#     任务结束或资源释放后仍保留。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # /code/workspace/v4
DATA_ROOT="${V4_DATA_ROOT:-/data}"
RUN_ROOT="${V4_RUN_ROOT:-$DATA_ROOT/v4/runs}"
CACHE_ROOT="${V4_CACHE_ROOT:-$DATA_ROOT/v4/cache}"
LOG_DIR="${V4_LOG_DIR:-$DATA_ROOT/v4/logs}"
REPORTS_DIR="$DATA_ROOT/v4/reports"

MODE="all"
STAGE="E1"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)   MODE="$2"; shift 2 ;;
    --stage)  STAGE="$2"; shift 2 ;;
    *)        EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$LOG_DIR" "$REPORTS_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/train_${MODE}_${TS}.log"

export V4_DATA_ROOT="$DATA_ROOT"
export V4_RUN_ROOT="$RUN_ROOT"
export V4_CACHE_ROOT="$CACHE_ROOT"
export V4_REPORTS_DIR="$REPORTS_DIR"
export V4_REPO_ROOT="$HERE"
# 平台集成 TensorBoard：日志写到 TENSORBOARD_LOGDIR（若有）
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$DATA_ROOT/v4/tb}"
mkdir -p "$TENSORBOARD_LOGDIR"
# 避免任何东西写到临时目录导致丢失
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=============================================================="
log "v4 training task  mode=$MODE stage=$STAGE"
log "repo(HERE)   = $HERE            <- /code/workspace/v4 (临时)"
log "DATA_ROOT    = $DATA_ROOT       <- 云盘（持久）"
log "RUN_ROOT     = $RUN_ROOT"
log "CACHE_ROOT   = $CACHE_ROOT"
log "REPORTS_DIR  = $REPORTS_DIR"
log "LOG          = $LOG"
log "=============================================================="
log "host: $(hostname)  python: $(python3 -V 2>&1)  pwd: $(pwd)"
df -h "$DATA_ROOT" | tee -a "$LOG" || true

# ---------------------------------------------------------------------------
# 首次任务把"数据未就绪"降级为 warn：check_env --profile base
# 训练前 / 数据部署后用 --profile full 做硬校验
# ---------------------------------------------------------------------------
_ENV_STATUS=0

check_env_profile() {
  local profile="$1"
  python3 "$HERE/E0/code/check_env.py" --profile "$profile" \
    --json "$REPORTS_DIR/E0_env.json" \
    --disk-path "$DATA_ROOT" --disk-path "$HERE" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_env() {
  # R2-B1 修复：先装依赖，再做硬校验（此前顺序相反导致首次必失败）
  log "--- [env] 1/3 额外轻量依赖（--no-cache-dir，绝不触碰 torch）"
  bash "$HERE/E0/code/setup_deps.sh" 2>&1 | tee -a "$LOG" \
    || log "!! setup_deps 失败（可选依赖缺失时管线有降级路径）"

  log "--- [env] 2/3 基础环境自检（profile=base：数据未就绪只告警）"
  if check_env_profile base; then
    log "[env] 基础环境检查通过（hard failures: 0）"
  else
    log "!! [env] 基础环境有 HARD FAILURE（python/torch/cuda/gpu/磁盘）。"
    log "!! 如需在此环境继续，显式设置 V4_ALLOW_ENV_FAILURE=1。"
    if [[ "${V4_ALLOW_ENV_FAILURE:-0}" != "1" ]]; then
      exit 11
    fi
  fi

  log "--- [env] 3/3 磁盘余量（$DATA_ROOT 与代码目录分别检查）"
  # R3 修复：必须显式 --data-root，否则 E0_disk_budget.json::level 描述的是 ROOT 文件系统，
  # 而云端 30 GB 配额在 $DATA_ROOT；E0_cloud_gate 的 disk_budget_ok 就读这个 level。
  if python3 "$HERE/src/data/disk_guard.py" --min-free-gb 8 \
       --data-root "$DATA_ROOT" --path "$DATA_ROOT" --path "$HERE" \
       --report "$HERE,$DATA_ROOT" --json "$REPORTS_DIR/E0_disk_budget.json" 2>&1 | tee -a "$LOG"; then
    log "[env] 磁盘余量 ok（>= 8 GiB）"
  else
    log "!! [env] 磁盘余量不足 8 GiB。请清理 $DATA_ROOT/v4/cache 或旧 checkpoint。"
    [[ "${V4_ALLOW_ENV_FAILURE:-0}" == "1" ]] || exit 12
  fi

  # 数据若已部署，顺带做一次 full 校验（失败不阻塞 env 模式）
  if [[ -d "$DATA_ROOT/v4/data/train" ]]; then
    log "--- [env] 附加：数据已部署，做 profile=full 校验"
    check_env_profile full || log "!! [env] full 校验未通过（数据健康有问题）"
  else
    log "--- [env] 数据尚未部署：请随后执行 --mode data（届时会做 full 校验）"
  fi
}

run_data() {
  log "--- [data] 部署数据集到 $DATA_ROOT/v4/data"
  bash "$HERE/tools/bootstrap_data.sh" 2>&1 | tee -a "$LOG"

  log "--- [data] 数据健康校验（profile=full：数据缺失为 hard）"
  if check_env_profile full; then
    log "[data] 数据健康校验通过"
  else
    log "!! [data] 数据健康校验失败：请检查 $DATA_ROOT/v4/data/{train,test} 与折文件"
    exit 13
  fi
}

run_e0() {
  log "--- [e0] 口径复算 + 分片缓存（不需要 torch）"
  # R2-B3 修复：默认建缓存，否则 E1/P0 无输入
  python3 "$HERE/E0/code/run_all.py" --train-dir "$DATA_ROOT/v4/data/train" \
    --test-dir "$DATA_ROOT/v4/data/test" --cache-root "$CACHE_ROOT" --with-cache \
    --out "$REPORTS_DIR/E0_data_card.json" 2>&1 | tee -a "$LOG"
  # R4-H2：计划行数证据 JSON 也在云端重生成（纯标准库，不需要 torch），
  # 使 $REPORTS_DIR 的 E0_*.json 集合自洽，而不是只在开发机上存在。
  python3 "$HERE/tools/plan_stats.py" --json "$REPORTS_DIR/E0_plan_stats.json" \
    2>&1 | tee -a "$LOG" || log "!! [e0] plan_stats 生成失败（不阻塞口径复算）"
  # 证据权威性：$REPORTS_DIR（云端 /data/v4/reports，云盘持久）= 权威来源，供 Gate / 复算引用；
  # repo 内 reports/ = 仅供 review / git diff 的快照，必须整体同步以免 data card 与
  # score-check / prereg 等互相矛盾（R3 修复：此前只 copy 3 个文件）。
  local f base dest_dir
  dest_dir="$HERE/reports"
  for f in "$REPORTS_DIR"/E0_*.json; do
    [[ -e "$f" ]] || continue                       # 未产出该文件时静默跳过
    base="$(basename "$f")"
    if [[ -f "$dest_dir/$base" && "$f" -ef "$dest_dir/$base" ]]; then
      continue                                      # 同一文件（REPORTS_DIR 指向 repo 内）不自我覆盖
    fi
    if cp -f "$f" "$dest_dir/" 2>/dev/null; then
      log "[e0] copy $base -> reports/ (review snapshot)"
    else
      log "!! [e0] copy $base 失败（忽略，权威副本仍在 $REPORTS_DIR）"
    fi
  done
}

run_smoke() {
  log "--- [smoke] 极小规模冒烟（1 折 / 2 epoch / 前 8 井）"
  if [[ ! -f "$HERE/E1/code/train_row.py" ]]; then
    log "!! E1/code/train_row.py 尚未实现（当前阶段）；smoke 跳过并返回非零"
    return 1
  fi
  python3 "$HERE/E1/code/train_row.py" \
    --train-dir "$DATA_ROOT/v4/data/train" --cache-root "$CACHE_ROOT" \
    --out-dir "$RUN_ROOT/smoke" --folds 0 --max-wells 8 --epochs 2 --smoke \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG"
}

run_stage() {
  log "--- [stage] $STAGE"
  case "$STAGE" in
    E0) run_e0 ;;
    E1) python3 "$HERE/E1/code/train_row.py" \
          --train-dir "$DATA_ROOT/v4/data/train" --cache-root "$CACHE_ROOT" \
          --out-dir "$RUN_ROOT/E1" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG" ;;
    E3) python3 "$HERE/E3/code/train_seq.py" \
          --train-dir "$DATA_ROOT/v4/data/train" --cache-root "$CACHE_ROOT" \
          --out-dir "$RUN_ROOT/E3" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG" ;;
    *)  log "!! 阶段 $STAGE 尚未实现（见 v4/PLAN.md §七 与各 E*/PLAN.md）"; return 1 ;;
  esac
}

case "$MODE" in
  env)   run_env ;;
  data)  run_data ;;
  e0)    run_e0 ;;
  smoke) run_smoke ;;
  data-health) check_env_profile full ;;
  stage) run_stage ;;
  all)
    run_env
    run_data
    run_e0
    log "--- [all] env + data + e0 完成"
    if [[ -f "$HERE/E1/code/train_row.py" ]]; then
      log "--- [all] 检测到 E1 训练脚本，继续执行 stage"
      run_stage || { log "!! [all] stage $STAGE 失败（退出码 $?）"; exit 21; }
    else
      log "--- [all] E1 训练脚本尚未实现，前置检查已完成（不算失败）"
    fi
    ;;
  *) log "unknown mode: $MODE"; exit 2 ;;
esac

log "--- 产物落盘情况（$DATA_ROOT 持久保留）"
du -sh "$RUN_ROOT" "$CACHE_ROOT" "$REPORTS_DIR" 2>/dev/null | tee -a "$LOG" || true
log "完成。"
