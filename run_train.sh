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
#   --mode smoke    5 分钟极小规模冒烟（1 折 / 少量 epoch），验证全链路可跑
#   --mode stage --stage E1   训练指定阶段
#   --mode all      依次执行 env -> data -> e0 -> E1 ...
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

run_env() {
  log "--- [env] 环境与磁盘自检"
  if python3 "$HERE/E0/code/check_env.py" --json "$REPORTS_DIR/E0_env.json" \
       | tee -a "$LOG"; then
    log "[env] check_env 通过（hard failures: 0）"
  else
    log "!! [env] check_env 有 HARD FAILURE（见上）。按 E0/P0 契约，训练不得继续。"
    log "!! 若你确认要在此环境继续（例如临时缺 GPU），请显式设置 V4_ALLOW_ENV_FAILURE=1。"
    if [[ "${V4_ALLOW_ENV_FAILURE:-0}" != "1" ]]; then
      exit 11
    fi
  fi
  log "--- [env] 额外轻量依赖（--no-cache-dir，绝不触碰 torch）"
  bash "$HERE/E0/code/setup_deps.sh" 2>&1 | tee -a "$LOG" || log "!! setup_deps 失败（可继续，代码有降级路径）"
  if python3 "$HERE/src/data/disk_guard.py" --min-free-gb 8 \
       --report "$HERE,$DATA_ROOT" --json "$REPORTS_DIR/E0_disk_budget.json" 2>&1 | tee -a "$LOG"; then
    log "[env] 磁盘余量 ok（>= 8 GiB）"
  else
    log "!! [env] 磁盘余量不足 8 GiB（见上）。请先清理 /data/v4/cache 或旧 checkpoint。"
    [[ "${V4_ALLOW_ENV_FAILURE:-0}" == "1" ]] || exit 12
  fi
}

run_data() {
  log "--- [data] 部署数据集到 $DATA_ROOT/v4/data"
  bash "$HERE/tools/bootstrap_data.sh" 2>&1 | tee -a "$LOG"
}

run_e0() {
  log "--- [e0] 口径复算（不需要 torch）"
  python3 "$HERE/E0/code/run_all.py" --train-dir "$DATA_ROOT/v4/data/train" \
    --test-dir "$DATA_ROOT/v4/data/test" \
    --out "$REPORTS_DIR/E0_data_card.json" 2>&1 | tee -a "$LOG"
  # 折导出与契约自检已在 run_all 内完成；同时把 gate 报告复制到 repo 便于 git 提交
  cp -f "$REPORTS_DIR/E0_gate.json" "$HERE/reports/E0_gate.json" 2>/dev/null || true
}

run_smoke() {
  log "--- [smoke] 极小规模冒烟（1 折 / 2 epoch / 前 8 井）"
  python3 "$HERE/E1/code/train_row.py" \
    --train-dir "$DATA_ROOT/v4/data/train" --test-dir "$DATA_ROOT/v4/data/test" \
    --out-dir "$RUN_ROOT/smoke" --folds 0 --max-wells 8 --epochs 2 --smoke \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG" || {
      log "!! smoke 失败：请先完成 E1/code/train_row.py 的实现（当前阶段尚未实现）"; }
}

run_stage() {
  log "--- [stage] $STAGE"
  case "$STAGE" in
    E1) python3 "$HERE/E1/code/train_row.py" --train-dir "$DATA_ROOT/v4/data/train" \
          --out-dir "$RUN_ROOT/E1" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG" ;;
    E3) python3 "$HERE/E3/code/train_seq.py" --train-dir "$DATA_ROOT/v4/data/train" \
          --out-dir "$RUN_ROOT/E3" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG" ;;
    *)  log "!! 阶段 $STAGE 尚未实现（见 v4/PLAN.md §七 与各 E*/PLAN.md）"; return 1 ;;
  esac
}

case "$MODE" in
  env)   run_env ;;
  data)  run_data ;;
  e0)    run_e0 ;;
  smoke) run_smoke ;;
  stage) run_stage ;;
  all)
    run_env
    run_data
    run_e0
    log "--- [all] E0 完成；后续阶段按 PLAN.md §七 顺序实现并训练"
    log "已就绪的训练入口：E1/code/train_row.py（若尚未实现会明确报错，不会静默）"
    run_stage || true
    ;;
  *) log "unknown mode: $MODE"; exit 2 ;;
esac

log "--- 产物落盘情况（$DATA_ROOT 持久保留）"
du -sh "$RUN_ROOT" "$CACHE_ROOT" "$REPORTS_DIR" 2>/dev/null | tee -a "$LOG" || true
log "完成。"
