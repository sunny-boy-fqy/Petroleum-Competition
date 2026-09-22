#!/usr/bin/env bash
# =============================================================================
# v4 云端统一启动入口（start.sh）
# =============================================================================
# 用途：把 run_train.sh 的底层模式包装成“按阶段/到某一步/单实验”都可用的一条命令。
#
# 平台【启动命令】推荐：
#     bash "$(find /code/workspace -name start.sh | head -1)" --to all
#     bash "$(find /code/workspace -name start.sh | head -1)" --to E3
#     bash "$(find /code/workspace -name start.sh | head -1)" --stage E3 --phase main
#     bash "$(find /code/workspace -name start.sh | head -1)" --stage E8 --target all
#     bash "$(find /code/workspace -name start.sh | head -1)" --wp atom-decision
#
# 主要能力：
#   --to STAGE      自动补全前置，跑到 STAGE 为止；STAGE 可为 env/data/e0/E1..E10/all/数字。
#   --through N     直接透传 run_train.sh 的任务号 1..14。
#   --stage STAGE   只跑某个阶段（自动先补全前置），可配 --phase/--target/额外参数。
#   --wp NAME       跑 WP 实验：type-well/atom-row/atom-decision/loss-full/perm-asym/
#                   gbdt/chained/stacking。
#   --fresh         清空进度台账，从 env 重新开始（不删 checkpoint/OOF 本身）。
#   --e1-rerun      清理旧 E1 预注册/产物后重跑 E1（WP0 新 Gate 专用）。
#   --dry-run       只打印将执行的命令，不真正执行。
#   --list          列出可用的 --to/--stage/--wp 取值。
#   --help          显示帮助。
#   其余未知参数会原样追加到对应的 run_train.sh / 实验脚本命令行。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$HERE/run_train.sh"

if [[ ! -f "$RUN" ]]; then
  echo "FATAL: 找不到 $RUN（start.sh 必须和 run_train.sh 在同一目录）" >&2
  exit 2
fi

# ---------------- 路径（与 run_train.sh 保持同源；因为 run_train.sh 是子进程，不能把它的 export 带回来）
NETWORK_ROOT="${V4_NETWORK_ROOT:-/data}"
LOCAL_ROOT="${V4_LOCAL_ROOT:-}"
if [[ -z "$LOCAL_ROOT" && -n "${V4_DATA_ROOT:-}" && "${V4_DATA_ROOT}" != "/data" ]]; then
  LOCAL_ROOT="${V4_DATA_ROOT}"
fi
if [[ -z "$LOCAL_ROOT" ]]; then
  if [[ -d /code/workspace && -w /code/workspace ]]; then
    LOCAL_ROOT="/code/workspace"
  elif [[ -d /workspace && -w /workspace ]]; then
    LOCAL_ROOT="/workspace"
  else
    LOCAL_ROOT="$HERE/.v4_runtime"
  fi
fi
mkdir -p "$LOCAL_ROOT" 2>/dev/null || LOCAL_ROOT="${TMPDIR:-/tmp}/v4_local"
mkdir -p "$LOCAL_ROOT"
DATA_ROOT="$LOCAL_ROOT"
RUN_ROOT="${V4_RUN_ROOT:-$DATA_ROOT/v4/runs}"
CACHE_ROOT="${V4_CACHE_ROOT:-$DATA_ROOT/v4/cache}"
REPORTS_DIR="${V4_REPORTS_DIR:-$DATA_ROOT/v4/reports}"
STATE_DIR="${V4_STATE_DIR:-$DATA_ROOT/v4/state}"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$REPORTS_DIR" "$STATE_DIR"
export V4_LOCAL_ROOT="$LOCAL_ROOT" V4_NETWORK_ROOT="$NETWORK_ROOT" V4_DATA_ROOT="$DATA_ROOT"
export V4_RUN_ROOT="$RUN_ROOT" V4_CACHE_ROOT="$CACHE_ROOT"
export V4_REPORTS_DIR="$REPORTS_DIR" V4_STATE_DIR="$STATE_DIR"
export V4_PAUSE_FLAG="${V4_PAUSE_FLAG:-$STATE_DIR/pause.flag}"

# ---------------- 参数
TO=""
THROUGH=""
STAGE=""
PHASE=""
TARGET=""
WP=""
FRESH=0
E1_RERUN=0
DRY=0
EXTRA=()

usage() {
  cat <<'EOF'
用法: start.sh [选项] [-- 额外参数...]

选项:
  --to STAGE       自动补全前置，跑到 STAGE 为止。
                   STAGE: env/data/e0/E1/E2/E3/E3-main/E3-all/E4/E5/E6/E7/E8/E9/E10/all 或 1..14
  --through N      直接指定 run_train.sh 任务号上限（1..14）
  --stage STAGE    只运行一个阶段（自动补全前置）；STAGE: E1..E10 / env/data/e0/smoke/data-health
  --phase PHASE    传给对应阶段的 --phase
  --target TARGET  传给对应阶段的 --target
  --wp NAME        运行 WP 实验；--list 可看全部
  --fresh          清空进度台账，从 env 重新开始
  --e1-rerun       清理旧 E1 预注册/产物后重跑 E1（WP0 新 Gate）
  --dry-run        只打印命令，不执行
  --list           列出可用的 --to/--stage/--wp
  --help, -h       显示本帮助
  --               后续参数原样追加到对应脚本

示例:
  start.sh --to E2                     # env -> data -> e0 -> E1 -> E2
  start.sh --to E3-main                # 跑到 E3-main（任务 6）
  start.sh --stage E3 --phase all      # 先补 E1/E2，再跑完整 E3
  start.sh --stage E8 --target mmoe --folds all
  start.sh --wp atom-decision          # 需要 E6 inner_oof；会自动补前置
  start.sh --e1-rerun --to E2          # 清旧 E1，重跑到 E2
EOF
}

list_choices() {
  cat <<'EOF'
--to / --stage 可用值:
  env, data, e0, E1, E2, E3, E3-main, E3-all, E4, E5, E6, E7, E8, E9, E10, all
  数字 1..14（run_train.sh 原生任务号）

--wp 可用值:
  type-well       E8 类型井选择报告（KL/DTW）
  data-quality    WP9 数据质量报告（缺失/插补/异常/KS）
  petro           WP10 扩展岩石物理特征报告
  atom-row        WP2 行级原子分类器（E3/code/train_atom.py）
  atom-decision   WP1 原子校准 + 期望分数动作表（E7/code/fit_atom_decision.py）
  loss-full       E7 全损失消融 exp1..exp8
  perm-asym       E7 exp8 PERM 不对称损失
  stacking        E8 集成使用 stacking 策略
  gbdt            WP11 GBDT 一阶成员（E8/code/tabular_member.py）
  chained         WP11 链式目标成员（E8/code/tabular_member.py）
  all            依次跑 data-quality/petro/type-well/atom-row/gbdt/chained/atom-decision/loss-full/stacking
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to) TO="${2:-}"; shift 2 ;;
    --through|--all-to) THROUGH="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --phase) PHASE="${2:-}"; shift 2 ;;
    --target) TARGET="${2:-}"; shift 2 ;;
    --wp) WP="${2:-}"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --e1-rerun) E1_RERUN=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --list) list_choices; exit 0 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY" == "0" ]]; then
    "$@"
  fi
}

run_all_through() {
  local n="$1"; shift || true
  local cmd=(bash "$RUN" --mode all --through "$n")
  [[ "$FRESH" == "1" ]] && cmd+=(--fresh)
  run_cmd "${cmd[@]}"
}

# ---------------- 名称 → 任务号
task_of() {
  local s="$1" n
  n="$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
  case "$n" in
    env) echo 1 ;;
    data) echo 2 ;;
    e0) echo 3 ;;
    e1) echo 4 ;;
    e2) echo 5 ;;
    e3|e3-full) echo 7 ;;
    e3-main|e3main) echo 6 ;;
    e3-all|e3all) echo 7 ;;
    e4) echo 8 ;;
    e5) echo 9 ;;
    e6) echo 10 ;;
    e7) echo 11 ;;
    e8) echo 12 ;;
    e9) echo 13 ;;
    e10) echo 14 ;;
    all) echo 14 ;;
    ''|*) echo "" ;;
  esac
}

# ---------------- E1 清理（WP0 新 Gate）
clean_e1() {
  if [[ "$DRY" == "1" ]]; then
    echo "[dry-run] 将清理旧 E1 产物/预注册/进度（不实际删除）"
    return 0
  fi
  echo "[start] 清理旧 E1 产物/预注册/进度，准备用新 Gate 重跑 E1"
  rm -rf "$RUN_ROOT/E1"
  rm -f "$REPORTS_DIR"/E1_* 2>/dev/null || true
  rm -rf "$NETWORK_ROOT/v4/mirror/run_root/E1"
  rm -f "$NETWORK_ROOT"/v4/mirror/reports/E1_* 2>/dev/null || true
  rm -f "$STATE_DIR/all_pipeline_progress.json" "$NETWORK_ROOT/v4/state/all_pipeline_progress.json"
  rm -f "$V4_PAUSE_FLAG"
}

# ---------------- 动作分发
MODE_SPECIAL=""
STAGE_NORM=""
if [[ -n "$STAGE" ]]; then
  STAGE_NORM="$(printf '%s' "$STAGE" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
fi

if [[ "$E1_RERUN" == "1" ]]; then
  clean_e1
  if [[ -z "$TO" && -z "$THROUGH" && -z "$STAGE" && -z "$WP" ]]; then
    TO="E1"
  fi
fi

# 纯 --through
if [[ -n "$THROUGH" ]]; then
  [[ "$THROUGH" =~ ^[0-9]+$ ]] || { echo "FATAL: --through 必须是 1..14 的整数" >&2; exit 2; }
  cmd=(bash "$RUN" --mode all --through "$THROUGH")
  [[ "$FRESH" == "1" ]] && cmd+=(--fresh)
  run_cmd "${cmd[@]}"
  exit 0
fi

# 纯 --to
if [[ -n "$TO" ]]; then
  N="$(task_of "$TO")"
  if [[ -z "$N" ]]; then
    [[ "$TO" =~ ^[0-9]+$ ]] && N="$TO" || { echo "FATAL: 未知 --to 取值: $TO" >&2; exit 2; }
  fi
  cmd=(bash "$RUN" --mode all --through "$N")
  [[ "$FRESH" == "1" ]] && cmd+=(--fresh)
  run_cmd "${cmd[@]}"
  exit 0
fi

# 单阶段
if [[ -n "$STAGE" ]]; then
  case "$STAGE_NORM" in
    env)          run_cmd bash "$RUN" --mode env; exit 0 ;;
    data)         run_cmd bash "$RUN" --mode data; exit 0 ;;
    e0)
      # e0 需要 data 已部署
      run_all_through 2
      run_cmd bash "$RUN" --mode e0
      exit 0 ;;
    smoke)
      run_all_through 2
      run_cmd bash "$RUN" --mode smoke "${EXTRA[@]}"
      exit 0 ;;
    data-health)
      run_all_through 2
      run_cmd bash "$RUN" --mode data-health
      exit 0 ;;
    e1|e2|e3|e3-main|e3main|e3-all|e3all|e4|e5|e6|e7|e8|e9|e10) ;;
    *) echo "FATAL: 未知 --stage 取值: $STAGE" >&2; exit 2 ;;
  esac

  # 前置任务号
  case "$STAGE_NORM" in
    e1) PREREQ=3 ;;
    e2) PREREQ=4 ;;
    e3|e3-main|e3main|e3-all|e3all) PREREQ=5 ;;
    e4) PREREQ=5 ;;
    e5) PREREQ=8 ;;
    e6) PREREQ=9 ;;
    e7) PREREQ=10 ;;
    e8) PREREQ=11 ;;
    e9) PREREQ=12 ;;
    e10) PREREQ=13 ;;
  esac

  # 先补前置
  pre=(bash "$RUN" --mode all --through "$PREREQ")
  [[ "$FRESH" == "1" ]] && pre+=(--fresh)
  run_cmd "${pre[@]}"

  # 阶段名归一化
  case "$STAGE_NORM" in
    e3main) stage_arg=E3 ;;
    e3all)  stage_arg=E3 ;;
    *)      stage_arg="$(printf '%s' "$STAGE_NORM" | tr '[:lower:]' '[:upper:]')" ;;
  esac

  # 默认 phase/target
  if [[ -z "$PHASE" ]]; then
    case "$STAGE_NORM" in
      e3|e3-all|e3all) PHASE="all" ;;
      e6|e7|e9|e10)    PHASE="all" ;;
    esac
  fi
  if [[ -z "$TARGET" ]]; then
    case "$STAGE_NORM" in
      e5) TARGET="all" ;;
      e8) TARGET="all" ;;
    esac
  fi

  cmd=(bash "$RUN" --mode stage --stage "$stage_arg")
  [[ -n "$PHASE" ]] && cmd+=(--phase "$PHASE")
  [[ -n "$TARGET" ]] && cmd+=(--target "$TARGET")
  cmd+=("${EXTRA[@]}")
  run_cmd "${cmd[@]}"
  exit 0
fi

# WP 实验
if [[ -n "$WP" ]]; then
  wp_norm="$(printf '%s' "$WP" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
  case "$wp_norm" in
    type-well)
      run_all_through 3
      run_cmd python3 "$HERE/E8/code/type_well_report.py" \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" \
        --topk 3 --method kl "${EXTRA[@]}"
      ;;
    data-quality)
      run_all_through 3
      run_cmd python3 "$HERE/E2/code/report_data_quality.py" \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" "${EXTRA[@]}"
      ;;
    petro)
      run_all_through 3
      run_cmd python3 "$HERE/E2/code/report_petro.py" \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" \
        --run-root "$RUN_ROOT" "${EXTRA[@]}"
      ;;
    atom-row)
      run_all_through 5
      run_cmd python3 "$HERE/E3/code/train_atom.py" \
        --mode row --spec F1 --folds all --device auto \
        --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT" "${EXTRA[@]}"
      ;;
    atom-decision)
      run_all_through 10
      run_cmd python3 "$HERE/E7/code/fit_atom_decision.py" \
        --oof "$RUN_ROOT/E6/state/inner_oof.npz" \
        --reports-dir "$REPORTS_DIR" \
        --out-config "$HERE/versions/configs/decode_v1.json" "${EXTRA[@]}"
      ;;
    loss-full)
      run_all_through 10
      run_cmd bash "$RUN" --mode stage --stage E7 --phase loss --arm-set all "${EXTRA[@]}"
      ;;
    perm-asym)
      run_all_through 10
      run_cmd bash "$RUN" --mode stage --stage E7 --phase loss --arm-set exp8 "${EXTRA[@]}"
      ;;
    stacking)
      run_all_through 11
      run_cmd bash "$RUN" --mode stage --stage E8 --target ensemble --strategy stacking "${EXTRA[@]}"
      ;;
    gbdt|chained)
      run_all_through 5
      run_cmd python3 "$HERE/E8/code/tabular_member.py" \
        --kind "$wp_norm" --folds all --spec F1 \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" \
        --run-root "$RUN_ROOT" "${EXTRA[@]}"
      ;;
    all)
      # 分层补前置，每层完成后再跑对应实验；run_train.sh 会自动跳过已完成任务。
      run_all_through 3
      run_cmd python3 "$HERE/E2/code/report_data_quality.py" \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR"
      run_cmd python3 "$HERE/E2/code/report_petro.py" \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT"
      run_cmd python3 "$HERE/E8/code/type_well_report.py" \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" --topk 3 --method kl

      run_all_through 5
      run_cmd python3 "$HERE/E3/code/train_atom.py" \
        --mode row --spec F1 --folds all --device auto \
        --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT"
      run_cmd python3 "$HERE/E8/code/tabular_member.py" \
        --kind gbdt --gbdt-kind histgb --folds all --spec F1 \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT" --tag gbdt
      run_cmd python3 "$HERE/E8/code/tabular_member.py" \
        --kind chained --estimator-kind histgb --folds all --spec F1 \
        --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT" --tag chained

      run_all_through 10
      run_cmd python3 "$HERE/E7/code/fit_atom_decision.py" \
        --oof "$RUN_ROOT/E6/state/inner_oof.npz" \
        --reports-dir "$REPORTS_DIR" \
        --out-config "$HERE/versions/configs/decode_v1.json"
      run_cmd bash "$RUN" --mode stage --stage E7 --phase loss --arm-set all

      run_all_through 11
      run_cmd bash "$RUN" --mode stage --stage E8 --target ensemble --strategy stacking
      ;;
    *) echo "FATAL: 未知 --wp 取值: $WP（用 --list 查看）" >&2; exit 2 ;;
  esac
  exit 0
fi

# 没有指定任何动作：默认跑到 all（和旧习惯一致，且不影响已有进度）
run_all_through 14
