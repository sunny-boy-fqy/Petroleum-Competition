#!/usr/bin/env bash
# =============================================================================
# v4 平台训练任务统一入口（Git 仓库代码来源）
# =============================================================================
# 平台约定（见 docs/platform_setup.md）：
#   - Git 仓库代码被克隆到**临时**目录 /code/workspace 下（任务结束即丢）；
#     平台表单显示的"当前工作目录"**恒为 /code/workspace**（代码在其下哪个子目录未定，
#     所以本仓库一律用 find 自定位，不硬编码路径）
#   - 云盘挂载在**持久**目录 /data（只有这里的内容会保留）
#   - 启动命令最长 500 字符，因此本脚本承担全部编排逻辑
#
# 平台【代码来源 Git 仓库】与"本机 git remote"要区分开（2026-09-20 读平台前端 bundle）：
#   1) 仓库地址必须 **HTTPS**：https://github.com/sunny-boy-fqy/Petroleum-Competition.git
#      平台侧没有本机 SSH key；且表单正则 /^(https?:\/\/|git@)[\w\-.~/]+(\.git)?$/i
#      会直接拒绝 scp 形式 git@github.com:owner/repo.git（":" 不在字符类里）。
#   2) 分支填 main（平台默认）或 master 均可：远端两个分支同指一个 commit。
#
# 任务"失败且日志为空"= 容器从未启动。**不要预设原因**：拉镜像/拉代码/调度三者都可能，
# 判定法与 2x2 实验见 docs/platform_setup.md §六-8。
#
# 平台【启动命令】填：
#     bash "$(find /code/workspace -name run_train.sh | head -1)" --mode all
#
# 常用模式：
#   --mode env      环境自检 + 安装额外轻量依赖（不碰 torch）
#   --mode data     部署数据集到 /data/v4/data（只需一次）
#                   自动搜索：--tarball / $V4_DATA_TARBALL -> repo dist/ -> /data/v4_data.tar.gz
#                   -> /data/dist/v4_data.tar.gz（云端推荐直接把 tarball 传到 /data 根）
#   --mode e0       E0 口径复算（数据卡 / 评分锚点 / 折指纹 / 契约自检）
#   --mode data-health  数据健康硬校验（profile=full）
#   --mode smoke    5 分钟极小规模冒烟（1 折 / 少量 epoch），验证全链路可跑
#   --mode stage --stage E1   训练指定阶段
#   --mode all      env -> data -> e0 -> E1 -> E2 -> ... -> E10 **全链路串行**；
#                   每个阶段自动带 all 子路由（E3 main+ablation+compare、
#                   E4 三个消融、E5 三目标、E6 P0/P1/P2、E8 四路、E9/E10 全子阶段）；
#                   `--through N` 只跑到第 N 个任务（1~14；默认 14）；
#                   1 env, 2 data, 3 e0, 4 E1, 5 E2, 6 E3-main, 7 E3-all,
#                   8 E4, 9 E5, 10 E6, 11 E7, 12 E8, 13 E9, 14 E10。
#                   已完成任务会自动跳过（断点续跑）；`--fresh` 可清空进度并重跑。
#                   任一步失败立即退出；进度写入 $STATE_DIR/all_pipeline_progress.json。
#
# 平台集成（已按官方提示落实）：
#   * TensorBoard：导出 TENSORBOARD_LOGDIR=$V4_DATA_ROOT/v4/tb，
#     训练脚本用 src/training/tb_logger.py::RunLogger 写指标，平台任务详情页可见曲线。
#   * 本地高速盘：训练期数据/缓存/checkpoint/报告/日志全部写到 `$V4_LOCAL_ROOT/v4/*`
#     （默认优先 /code/workspace，再回退 /workspace 或 $HERE/.v4_runtime，**不写网络盘 /data**）。
#   * 网络盘 `/data` 用于：读取上传的数据分发包；每 5 分钟把 checkpoint/OOF/报告
#     增量 mirror 到 `/data/v4/mirror/`（新任务恢复本地点）；训练结束后 publish 最终模型。
#   * 大 cache/原始数据不写 /data；新任务从 /data tarball 重新解压，并在缺 cache 时自动重跑 E2。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # 自定位：仓库根可能直接是 /code/workspace
NETWORK_ROOT="${V4_NETWORK_ROOT:-/data}"
LOCAL_ROOT="${V4_LOCAL_ROOT:-}"
# 兼容旧调用：显式 V4_DATA_ROOT 指向非 /data 时，仍视为本地运行时根。
if [[ -z "$LOCAL_ROOT" && -n "${V4_DATA_ROOT:-}" && "${V4_DATA_ROOT}" != "/data" ]]; then
  LOCAL_ROOT="${V4_DATA_ROOT}"
fi
if [[ -z "$LOCAL_ROOT" ]]; then
  # 优先本地大容量目录；不写 /data 网络盘。
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
DATA_ROOT="$LOCAL_ROOT"                 # 训练期运行时数据根（本地高速盘）
RUN_ROOT="${V4_RUN_ROOT:-$DATA_ROOT/v4/runs}"
CACHE_ROOT="${V4_CACHE_ROOT:-$DATA_ROOT/v4/cache}"
LOG_DIR="${V4_LOG_DIR:-$DATA_ROOT/v4/logs}"
REPORTS_DIR="$DATA_ROOT/v4/reports"
STATE_DIR="${V4_STATE_DIR:-$DATA_ROOT/v4/state}"
CANDIDATES="${V4_CANDIDATES:-$STATE_DIR/candidates.json}"
REGISTRY="${V4_REGISTRY:-$STATE_DIR/registry.json}"

MODE="all"
STAGE="E1"
ALL_THROUGH=14
ALL_FRESH=0
ALL_RESUME_FLAG=()
EXTRA_ARGS=()
# R5-B1：git 仓库里**没有** dist/*.tar.gz（.gitignore 忽略），云端必须能从云盘找到它。
# 因此 tarball/manifest 是一等参数（不会被塞进 EXTRA_ARGS 污染 smoke/stage 的命令行）。
DATA_TARBALL="${V4_DATA_TARBALL:-}"
DATA_MANIFEST="${V4_DATA_MANIFEST:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)     MODE="$2"; shift 2 ;;
    --stage)    STAGE="$2"; shift 2 ;;
    --through|--all-to) ALL_THROUGH="$2"; shift 2 ;;
    --fresh)    ALL_FRESH=1; shift ;;
    --tarball)  DATA_TARBALL="$2"; shift 2 ;;
    --manifest) DATA_MANIFEST="$2"; shift 2 ;;
    *)          EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$LOG_DIR" "$REPORTS_DIR" "$STATE_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/train_${MODE}_${TS}.log"

export V4_LOCAL_ROOT="$LOCAL_ROOT"
export V4_NETWORK_ROOT="$NETWORK_ROOT"
export V4_DATA_ROOT="$DATA_ROOT"
export V4_RUN_ROOT="$RUN_ROOT"
export V4_CACHE_ROOT="$CACHE_ROOT"
export V4_REPORTS_DIR="$REPORTS_DIR"
export V4_REPO_ROOT="$HERE"
# 候选表与版本注册表是运行时可变状态：放本地 runtime，最终 publish 再复制到 /data。
export V4_STATE_DIR="$STATE_DIR"
export V4_CANDIDATES="$CANDIDATES"
export V4_REGISTRY="$REGISTRY"
# 优雅暂停：touch "$V4_PAUSE_FLAG" 后，训练循环在下一个 epoch 边界保存 last.pt 并退出。
export V4_PAUSE_FLAG="${V4_PAUSE_FLAG:-$STATE_DIR/pause.flag}"
ALL_PROGRESS="$STATE_DIR/all_pipeline_progress.json"
NET_PROGRESS="$NETWORK_ROOT/v4/state/all_pipeline_progress.json"
# 让 bootstrap_data.sh 也能看到 tarball 位置（同一份事实，不重复解析参数）
if [[ -n "$DATA_TARBALL" ]]; then export V4_DATA_TARBALL="$DATA_TARBALL"; fi
if [[ -n "$DATA_MANIFEST" ]]; then export V4_DATA_MANIFEST="$DATA_MANIFEST"; fi
# 平台集成 TensorBoard：日志写到 TENSORBOARD_LOGDIR（若有）
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$DATA_ROOT/v4/tb}"
mkdir -p "$TENSORBOARD_LOGDIR"
# torch/XDG 缓存也放本地运行时目录，避免写满网络盘
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# all 模式进度台账（本地 STATE_DIR，最终 publish 时随模型一起复制到 /data）
mark_progress() {
  python3 - "$ALL_PROGRESS" "$1" "$2" "${3:-running}" <<'PY'
import json, os, sys, time
path, task, name, status = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
obj = {}
if os.path.exists(path):
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = {}
obj.setdefault("pipeline", "all")
obj.setdefault("tasks", {})
now = time.strftime("%Y-%m-%dT%H:%M:%S")
obj["updated_at"] = now
obj["current_task"] = task
obj["current_name"] = name
obj["status"] = status
obj["tasks"][str(task)] = {"name": name, "status": status, "updated_at": now}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
print(f"[progress] task {task} ({name}): {status} -> {path}")
PY
}

task_status() {
  python3 - "$ALL_PROGRESS" "$1" <<'PY'
import json, os, sys
path, task = sys.argv[1], str(sys.argv[2])
obj = {}
try:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
except Exception:
    obj = {}
print(obj.get("tasks", {}).get(task, {}).get("status", ""))
PY
}

task_local_ready() {
  # 进度说 done 还不够：新任务本地盘可能是空的，关键本地产物存在才允许跳过。
  local n="$1"
  case "$n" in
    1) [[ -f "$REPORTS_DIR/E0_env.json" && -f "$REPORTS_DIR/E0_disk_budget.json" ]] ;;
    2)
      [[ -d "$DATA_ROOT/v4/data/train" && -d "$DATA_ROOT/v4/data/test" ]] || return 1
      local nt ne
      nt="$(find "$DATA_ROOT/v4/data/train" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)"
      ne="$(find "$DATA_ROOT/v4/data/test"  -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)"
      [[ "$nt" -eq 80 && "$ne" -eq 10 ]] ;;
    3) [[ -f "$REPORTS_DIR/E0_data_card.json" && -d "$CACHE_ROOT/raw/train" ]] ;;
    4) [[ -f "$RUN_ROOT/E1/oof.npz" ]] ;;
    5) [[ -d "$CACHE_ROOT/feat" ]] && find "$CACHE_ROOT/feat" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -q . ;;
    6) [[ -f "$RUN_ROOT/E3/oof_unet.npz" && -f "$REPORTS_DIR/E3_metrics_unet.json" ]] ;;
    7) [[ -f "$REPORTS_DIR/E3_receptive_field_ablation.json" ]] ;;
    8) [[ -f "$RUN_ROOT/E4/oof_patchtf.npz" && -f "$REPORTS_DIR/E4_patchtf.json" ]] ;;
    9) [[ -f "$REPORTS_DIR/E5_per_target.json" ]] ;;
    10) [[ -f "$REPORTS_DIR/E6_tau_search.json" ]] ;;
    11) [[ -f "$REPORTS_DIR/E7_loss_ablation.json" ]] ;;
    12) [[ -f "$REPORTS_DIR/E8_ensemble_report.json" ]] ;;
    13) [[ -f "$REPORTS_DIR/E9_submission_decision.json" ]] ;;
    14) [[ -f "$REPORTS_DIR/E10_final_train.json" && -f "$RUN_ROOT/v4/final/final_manifest.json" ]] ;;
    *) return 0 ;;
  esac
}

log "=============================================================="
log "v4 training task  mode=$MODE stage=$STAGE"
log "repo(HERE)   = $HERE            <- 平台 zip/git 解压后的仓库根（本场景直接是 /code/workspace）"
log "LOCAL_ROOT   = $LOCAL_ROOT       <- 训练期本地高速盘（数据/cache/runs/reports/logs/state）"
log "NETWORK_ROOT = $NETWORK_ROOT     <- 网络盘：只读 tarball + 收最终模型"
log "DATA_ROOT    = $DATA_ROOT       <- 本地运行时数据根（$V4_DATA_ROOT）"
log "RUN_ROOT     = $RUN_ROOT"
log "CACHE_ROOT   = $CACHE_ROOT"
log "REPORTS_DIR  = $REPORTS_DIR"
log "STATE_DIR    = $STATE_DIR    <- 本地候选/注册表/进度"
log "CANDIDATES   = $CANDIDATES"
log "REGISTRY     = $REGISTRY"
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
    || { log "!! setup_deps 失败：依赖/磁盘预检未通过，停止"; return 1; }

  log "--- [env] 2/3 基础环境自检（profile=base：数据未就绪只告警）"
  if check_env_profile base; then
    log "[env] 基础环境检查通过（hard failures: 0）"
  else
    log "!! [env] 基础环境有 HARD FAILURE（python/torch/torch_npu/CANN/NPU 设备/架构/磁盘）。"
    log "!! 如需在此环境继续，显式设置 V4_ALLOW_ENV_FAILURE=1。"
    if [[ "${V4_ALLOW_ENV_FAILURE:-0}" != "1" ]]; then
      exit 11
    fi
  fi

  log "--- [env] 3/3 磁盘余量（$DATA_ROOT 与代码目录分别检查）"
  # R3 修复：必须显式 --data-root，否则 E0_disk_budget.json::level 描述的是 ROOT 文件系统，
  # 而云端云盘配额（30 GB）在 $DATA_ROOT；E0_cloud_gate 的 disk_budget_ok 就读这个 level。
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
    # 首次 all 流程中 env 先于 data，空数据目录/旧目录可能导致 full 校验未过；
    # 这是预期状态，只告警不阻塞，后续 run_data 会重新部署并以 full 硬校验为准。
    check_env_profile full || log "!! [env] full 校验未通过（数据健康有问题，稍后 run_data 会重部署并硬校验）"
  else
    log "--- [env] 数据尚未部署：请随后执行 --mode data（届时会做 full 校验）"
  fi
}

run_data() {
  log "--- [data] 部署数据集到 $DATA_ROOT/v4/data"
  # R5-B1：tarball 可能只在云盘（repo 内的 dist/ 被 .gitignore 忽略），
  # 所以显式把 --tarball/--manifest 透传给 bootstrap_data.sh，其余参数不进这里。
  local -a bargs=()
  if [[ -n "$DATA_TARBALL" ]]; then bargs+=(--tarball "$DATA_TARBALL"); fi
  if [[ -n "$DATA_MANIFEST" ]]; then bargs+=(--manifest "$DATA_MANIFEST"); fi
  log "[data] bootstrap_data.sh ${bargs[*]:-（自动搜索 repo dist/ 与 $DATA_ROOT）}"
  bash "$HERE/tools/bootstrap_data.sh" ${bargs[@]+"${bargs[@]}"} 2>&1 | tee -a "$LOG" || return 1

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
    --out "$REPORTS_DIR/E0_data_card.json" 2>&1 | tee -a "$LOG" || return 1
  # R4-H2：计划行数证据 JSON 也在云端重生成（纯标准库，不需要 torch），
  # 使 $REPORTS_DIR 的 E0_*.json 集合自洽，而不是只在开发机上存在。
  python3 "$HERE/tools/plan_stats.py" --json "$REPORTS_DIR/E0_plan_stats.json" \
    2>&1 | tee -a "$LOG" || return 1
  # 流水线完整性：阶段脚本/入口/接线/报告命名一次性核对（失败即停）
  python3 "$HERE/tools/check_pipeline.py" --json "$REPORTS_DIR/E0_pipeline_check.json" \
    2>&1 | tee -a "$LOG" || return 1
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

publish_progress_to_network() {
  # 只同步极小体积的进度/状态文件：大 cache/runs 永不写 /data。
  if [[ ! -f "$ALL_PROGRESS" ]]; then return 0; fi
  local dst="$NETWORK_ROOT/v4/state"
  mkdir -p "$dst" || return 1
  cp -a "$ALL_PROGRESS" "$dst/" || return 1
  for f in "$CANDIDATES" "$REGISTRY"; do
    if [[ -f "$f" ]]; then cp -a "$f" "$dst/" || return 1; fi
  done
  log "[publish] progress/state -> $dst"
}

publish_final_to_network() {
  # 训练结束后只把最终模型（+ 可选 E10 提交包）复制到网络盘 /data。
  local src_final="$RUN_ROOT/v4/final"
  local dst_final="$NETWORK_ROOT/v4/final"
  local copied=0
  if [[ -d "$src_final" ]]; then
    mkdir -p "$dst_final" || return 1
    cp -a "$src_final/." "$dst_final/" || return 1
    copied=1
    log "[publish] final model -> $dst_final"
  else
    log "[publish] 未找到最终模型目录 $src_final（跳过）"
  fi
  local src_sub="$RUN_ROOT/E10/submission"
  if [[ -d "$src_sub" ]]; then
    local dst_sub="$NETWORK_ROOT/v4/submission"
    mkdir -p "$dst_sub" || return 1
    cp -a "$src_sub/." "$dst_sub/" || return 1
    copied=1
    log "[publish] submission -> $dst_sub"
  fi
  publish_progress_to_network || log "[publish] progress 同步失败（不阻塞最终模型已发布）"
  if [[ "$copied" == "0" ]]; then
    log "[publish] 无模型可发布（可能还没跑到 E10）"
  fi
}

MIRROR_ROOT="$NETWORK_ROOT/v4/mirror"
REMOTE_RUN_MIRROR="$MIRROR_ROOT/run_root"
REMOTE_SCALER_MIRROR="$MIRROR_ROOT/scalers"
REMOTE_REPORTS_MIRROR="$MIRROR_ROOT/reports"
LOCAL_SCALER_ROOT="$DATA_ROOT/v4/scalers"
MIRROR_PIDS=()

restore_state_from_network() {
  # 新任务本地盘为空时，从 /data/v4/mirror 恢复 checkpoint / OOF / scaler / 报告。
  python3 "$HERE/tools/sync_state.py" --src "$REMOTE_RUN_MIRROR" --dst "$RUN_ROOT" --once >/dev/null 2>&1 || true
  python3 "$HERE/tools/sync_state.py" --src "$REMOTE_SCALER_MIRROR" --dst "$LOCAL_SCALER_ROOT" --once >/dev/null 2>&1 || true
  python3 "$HERE/tools/sync_state.py" --src "$REMOTE_REPORTS_MIRROR" --dst "$REPORTS_DIR" --once >/dev/null 2>&1 || true
}

sync_state_to_network() {
  python3 "$HERE/tools/sync_state.py" --src "$RUN_ROOT" --dst "$REMOTE_RUN_MIRROR" --once >/dev/null 2>&1 || true
  python3 "$HERE/tools/sync_state.py" --src "$LOCAL_SCALER_ROOT" --dst "$REMOTE_SCALER_MIRROR" --once >/dev/null 2>&1 || true
  python3 "$HERE/tools/sync_state.py" --src "$REPORTS_DIR" --dst "$REMOTE_REPORTS_MIRROR" --once >/dev/null 2>&1 || true
}

start_state_mirror() {
  [[ "${V4_DISABLE_STATE_MIRROR:-0}" == "1" ]] && return 0
  mkdir -p "$REMOTE_RUN_MIRROR" "$REMOTE_SCALER_MIRROR" "$REMOTE_REPORTS_MIRROR" "$LOCAL_SCALER_ROOT" 2>/dev/null || true
  # 每 5 分钟把本地小状态文件增量同步到 /data；大 cache/原始数据不同步。
  python3 "$HERE/tools/sync_state.py" --src "$RUN_ROOT" --dst "$REMOTE_RUN_MIRROR" --interval 300 >/dev/null 2>&1 &
  MIRROR_PIDS+=($!)
  python3 "$HERE/tools/sync_state.py" --src "$LOCAL_SCALER_ROOT" --dst "$REMOTE_SCALER_MIRROR" --interval 300 >/dev/null 2>&1 &
  MIRROR_PIDS+=($!)
  python3 "$HERE/tools/sync_state.py" --src "$REPORTS_DIR" --dst "$REMOTE_REPORTS_MIRROR" --interval 300 >/dev/null 2>&1 &
  MIRROR_PIDS+=($!)
}

cleanup_state_mirror() {
  if [[ ${#MIRROR_PIDS[@]} -gt 0 ]]; then
    kill "${MIRROR_PIDS[@]}" 2>/dev/null || true
  fi
  sync_state_to_network || true
}

trap cleanup_state_mirror EXIT

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
    E2)
        # `--phase build|ablate|all`（build=F2 特征缓存/溯源；ablate=单组消融+吞吐）
        e2_phase="all"
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          case "${a,,}" in build|ablate|all) e2_phase="${a,,}" ;; esac
        done
        if [[ "$e2_phase" == "all" || "$e2_phase" == "build" ]]; then
          log "--- [E2] 构建 F2 特征缓存 + 溯源/审计"
          python3 "$HERE/E2/code/build_features.py" --cache-root "$CACHE_ROOT" \
            --reports-dir "$REPORTS_DIR" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e2_phase" == "all" || "$e2_phase" == "ablate" ]]; then
          log "--- [E2] 单组消融 + 吞吐画像"
          python3 "$HERE/E2/code/ablate_groups.py" --cache-root "$CACHE_ROOT" \
            --reports-dir "$REPORTS_DIR" --work-dir "$RUN_ROOT/E2" 2>&1 | tee -a "$LOG" || return 1
        fi ;;
    E3)
        # `--phase main|ablation|compare|all`；缺省 main 保持旧行为。
        # all = TCN 主配置 + U-Net 主配置 + 感受野消融 + 行级/序列受控对照。
        e3_phase="main"
        e3_args=()
        e3_expect=""
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          if [[ "$e3_expect" == "phase" ]]; then
            case "${a,,}" in
              main|ablation|compare|all) e3_phase="${a,,}" ;;
              *) log "!! E3 --phase 只支持 main/ablation/compare/all，got $a"; return 1 ;;
            esac
            e3_expect=""; continue
          fi
          if [[ "$a" == "--phase" ]]; then e3_expect="phase"; continue; fi
          e3_args+=("$a")
        done
        if [[ "$e3_expect" == "phase" ]]; then log "!! E3 --phase 缺少取值"; return 1; fi
        e3_common=(--cache-root "$CACHE_ROOT" --run-root "$RUN_ROOT" \
                   --reports-dir "$REPORTS_DIR")
        if [[ "$e3_phase" == "main" || "$e3_phase" == "all" ]]; then
          if [[ "$e3_phase" == "all" ]]; then
            # 最后跑 U-Net，使 E3_gate.json 对应默认主干。
            log "--- [E3] train_seq arch=tcn"
            python3 "$HERE/E3/code/train_seq.py" "${e3_common[@]}" --arch tcn --tag tcn \
              "${e3_args[@]+"${e3_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
          fi
          log "--- [E3] train_seq arch=unet"
          python3 "$HERE/E3/code/train_seq.py" "${e3_common[@]}" --arch unet \
            "${e3_args[@]+"${e3_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e3_phase" == "ablation" || "$e3_phase" == "all" ]]; then
          log "--- [E3] receptive-field ablation"
          python3 "$HERE/E3/code/rf_ablation.py" "${e3_common[@]}" \
            "${e3_args[@]+"${e3_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e3_phase" == "compare" || "$e3_phase" == "all" ]]; then
          log "--- [E3] row vs seq controlled comparison"
          e3_seq_oof="$(python3 - "$REPORTS_DIR" "$RUN_ROOT/E3" <<'PY'
import json, sys
from pathlib import Path
reports, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
best = None
for tag in ("unet", "tcn"):
    m = reports / f"E3_metrics_{tag}.json"
    if not m.is_file():
        continue
    try:
        total = json.loads(m.read_text(encoding="utf-8")).get("oof_total")
    except Exception:
        total = None
    if total is not None and (best is None or float(total) > best[0]):
        best = (float(total), tag)
print(run_dir / f"oof_{best[1]}.npz" if best else "")
PY
)"
          e3_row_oof="$RUN_ROOT/E1/oof.npz"
          if [[ -z "$e3_seq_oof" || ! -f "$e3_seq_oof" ]]; then
            log "!! [E3] 找不到序列 OOF（$e3_seq_oof）；先跑 --phase main"
            return 1
          fi
          if [[ ! -f "$e3_row_oof" ]]; then
            log "!! [E3] 找不到行级 OOF（$e3_row_oof）；先跑 E1"
            return 1
          fi
          python3 "$HERE/E3/code/compare_row_vs_seq.py" \
            --seq-oof "$e3_seq_oof" --row-oof "$e3_row_oof" \
            --reports-dir "$REPORTS_DIR" 2>&1 | tee -a "$LOG" || return 1
        fi ;;
    E4) python3 "$HERE/E4/code/train_patchtf.py" \
          --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" \
          --run-root "$RUN_ROOT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG" ;;
    E5)
        # 目标路由：`--target por|perm|sw` 决定跑哪个逐目标头（其余参数原样透传）
        e5_script="$HERE/E5/code/head_por.py"
        e5_args=()
        e5_expect=""
        e5_all=0
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          if [[ "$e5_expect" == "target" ]]; then
            case "${a,,}" in
              por)  e5_script="$HERE/E5/code/head_por.py" ;;
              perm) e5_script="$HERE/E5/code/head_perm.py" ;;
              sw)   e5_script="$HERE/E5/code/head_sw.py" ;;
              all)  e5_all=1 ;;
              *) log "!! E5 --target 只支持 por/perm/sw/all，got $a"; return 1 ;;
            esac
            e5_expect=""
            continue
          fi
          if [[ "$a" == "--target" ]]; then e5_expect="target"; continue; fi
          e5_args+=("$a")
        done
        if [[ "$e5_expect" == "target" ]]; then log "!! --target 缺少取值"; return 1; fi
        if [[ "$e5_all" == "1" ]]; then
          # 三目标顺序执行（任一失败即停），最后做联合汇总（per-target + E5_gate）
          for e5_name in head_por head_perm head_sw; do
            e5_script="$HERE/E5/code/$e5_name.py"
            if [[ ! -f "$e5_script" ]]; then
              log "!! $e5_script 尚未实现（见 E5/*/PLAN.md）"; return 1
            fi
            log "--- [E5] $e5_name"
            python3 "$e5_script" \
              --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" \
              --run-root "$RUN_ROOT" "${e5_args[@]+"${e5_args[@]}"}" 2>&1 | tee -a "$LOG" \
              || { log "!! [E5] $e5_name 失败"; return 1; }
          done
          python3 "$HERE/E5/code/evaluate_targets.py" \
            --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT" 2>&1 | tee -a "$LOG" || return 1
        else
          if [[ ! -f "$e5_script" ]]; then
            log "!! $e5_script 尚未实现（见 E5/*/PLAN.md）"; return 1
          fi
          python3 "$e5_script" \
            --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" \
            --run-root "$RUN_ROOT" "${e5_args[@]+"${e5_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi ;;
    E6)
        # 子阶段路由：`--phase p0|p1|all`（p0=原子两阶段训练；p1=τ 搜索；all=两者依次）
        e6_phase="p0"
        e6_args=()
        e6_expect=""
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          if [[ "$e6_expect" == "phase" ]]; then
            case "${a,,}" in
              p0|p1|p2|all) e6_phase="${a,,}" ;;
              *) log "!! E6 --phase 只支持 p0/p1/p2/all，got $a"; return 1 ;;
            esac
            e6_expect=""
            continue
          fi
          if [[ "$a" == "--phase" ]]; then e6_expect="phase"; continue; fi
          e6_args+=("$a")
        done
        if [[ "$e6_expect" == "phase" ]]; then log "!! --phase 缺少取值"; return 1; fi
        if [[ "$e6_phase" == "p0" || "$e6_phase" == "all" ]]; then
          log "--- [E6] P0 原子状态头两阶段训练"
          python3 "$HERE/E6/code/train_state.py" \
            --cache-root "$CACHE_ROOT" --reports-dir "$REPORTS_DIR" \
            --run-root "$RUN_ROOT" "${e6_args[@]+"${e6_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e6_phase" == "p1" || "$e6_phase" == "all" ]]; then
          log "--- [E6] P1 τ 搜索（内折 OOF）"
          python3 "$HERE/E6/code/search_tau.py" \
            --run-root "$RUN_ROOT" --reports-dir "$REPORTS_DIR" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e6_phase" == "p2" || "$e6_phase" == "all" ]]; then
          log "--- [E6] P2 折平均 PD1（OOF + 注册 + CPU 冒烟）"
          python3 "$HERE/E6/code/build_pd1.py" \
            --run-root "$RUN_ROOT" --reports-dir "$REPORTS_DIR" \
            --cache-root "$CACHE_ROOT" \
            --test-dir "$DATA_ROOT/v4/data/test" \
            --out-dir "$RUN_ROOT/E6/P2/pd1" \
            --models-dir "$DATA_ROOT/v4/models/E6" \
            "${e6_args[@]+"${e6_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi ;;
    E7)
        # `--phase loss|decode|all`（loss=七组损失消融；decode=解码搜索）
        e7_phase="all"; e7_args=(); e7_expect=""
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          if [[ "$e7_expect" == "phase" ]]; then
            case "${a,,}" in
              loss|decode|all) e7_phase="${a,,}" ;;
              *) log "!! E7 --phase 只支持 loss/decode/all，got $a"; return 1 ;;
            esac
            e7_expect=""; continue
          fi
          if [[ "$a" == "--phase" ]]; then e7_expect="phase"; continue; fi
          e7_args+=("$a")
        done
        if [[ "$e7_expect" == "phase" ]]; then log "!! --phase 缺少取值"; return 1; fi
        if [[ "$e7_phase" == "loss" || "$e7_phase" == "all" ]]; then
          log "--- [E7] P0 损失消融"
          python3 "$HERE/E7/code/ablate_loss.py" --reports-dir "$REPORTS_DIR" \
            --cache-root "$CACHE_ROOT" "${e7_args[@]+"${e7_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e7_phase" == "decode" || "$e7_phase" == "all" ]]; then
          log "--- [E7] P1 解码搜索"
          python3 "$HERE/E7/code/decode_search.py" --reports-dir "$REPORTS_DIR" \
            --run-root "$RUN_ROOT" "${e7_args[@]+"${e7_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi ;;
    E8)
        # `--target mmoe|well|transductive|ensemble|all`
        e8_target="mmoe"; e8_args=(); e8_expect=""
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          if [[ "$e8_expect" == "target" ]]; then
            case "${a,,}" in
              mmoe|well|transductive|ensemble|all) e8_target="${a,,}" ;;
              *) log "!! E8 --target 只支持 mmoe/well/transductive/ensemble/all，got $a"; return 1 ;;
            esac
            e8_expect=""; continue
          fi
          if [[ "$a" == "--target" ]]; then e8_expect="target"; continue; fi
          e8_args+=("$a")
        done
        if [[ "$e8_expect" == "target" ]]; then log "!! --target 缺少取值"; return 1; fi
        run_e8_one() {
          local name="$1"; shift
          log "--- [E8] $name"
          python3 "$HERE/E8/code/$name.py" --reports-dir "$REPORTS_DIR" \
            --run-root "$RUN_ROOT" --cache-root "$CACHE_ROOT" \
            "${e8_args[@]+"${e8_args[@]}"}" 2>&1 | tee -a "$LOG"
        }
        if [[ "$e8_target" == "all" ]]; then
          for s in train_mmoe well_branch pseudo_label ensemble; do run_e8_one "$s" || return 1; done
        elif [[ "$e8_target" == "mmoe" ]]; then run_e8_one train_mmoe
        elif [[ "$e8_target" == "well" ]]; then run_e8_one well_branch
        elif [[ "$e8_target" == "transductive" ]]; then run_e8_one pseudo_label
        else run_e8_one ensemble; fi ;;
    E9)
        # `--phase leakage|aggregate|choose|confirm|submit|all`
        e9_phase="all"; e9_args=(); e9_expect=""
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          if [[ "$e9_expect" == "phase" ]]; then
            case "${a,,}" in
              leakage|aggregate|choose|confirm|submit|all) e9_phase="${a,,}" ;;
              *) log "!! E9 --phase 只支持 leakage/aggregate/choose/confirm/submit/all，got $a"; return 1 ;;
            esac
            e9_expect=""; continue
          fi
          if [[ "$a" == "--phase" ]]; then e9_expect="phase"; continue; fi
          e9_args+=("$a")
        done
        if [[ "$e9_expect" == "phase" ]]; then log "!! E9 --phase 缺少取值"; return 1; fi
        run_e9() {
          local stage="$1"; shift
          log "--- [E9] $stage"
          # 只传各脚本共同支持的 --reports-dir；V4_RUN_ROOT 已由 run_train.sh 导出，
          # choose_submission/confirm_check/submit_batch 的 argparse 不接受 --run-root。
          python3 "$HERE/E9/code/$stage.py" --reports-dir "$REPORTS_DIR" \
            "${e9_args[@]+"${e9_args[@]}"}" 2>&1 | tee -a "$LOG"
        }
        # H5：leakage_audit 先产出真实证据，后面的 aggregate/choose/... 才能
        # 从文件读取 no_label_leak，而不是硬编码 True。
        if [[ "$e9_phase" == "all" || "$e9_phase" == "leakage" ]]; then run_e9 leakage_audit || return 1; fi
        if [[ "$e9_phase" == "all" || "$e9_phase" == "aggregate" ]]; then run_e9 aggregate_oof || return 1; fi
        if [[ "$e9_phase" == "all" || "$e9_phase" == "choose" ]]; then run_e9 choose_submission || return 1; fi
        if [[ "$e9_phase" == "all" || "$e9_phase" == "confirm" ]]; then run_e9 confirm_check || return 1; fi
        if [[ "$e9_phase" == "all" || "$e9_phase" == "submit" ]]; then run_e9 submit_batch || return 1; fi ;;
    E10)
        # `--phase final|export|verify|build|b0|submit|all`
        e10_phase="all"; e10_args=(); e10_expect=""
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          if [[ "$e10_expect" == "phase" ]]; then
            case "${a,,}" in
              final|export|verify|build|b0|submit|all) e10_phase="${a,,}" ;;
              *) log "!! E10 --phase 只支持 final/export/verify/build/b0/submit/all，got $a"; return 1 ;;
            esac
            e10_expect=""; continue
          fi
          if [[ "$a" == "--phase" ]]; then e10_expect="phase"; continue; fi
          e10_args+=("$a")
        done
        if [[ "$e10_expect" == "phase" ]]; then log "!! --phase 缺少取值"; return 1; fi
        e10_run() {
          local stage="$1"; shift
          log "--- [E10] $stage"
          python3 "$HERE/E10/code/$stage.py" --reports-dir "$REPORTS_DIR" \
            "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG"
        }
        e10_sub="$RUN_ROOT/E10/submission"
        if [[ "$e10_phase" == "all" || "$e10_phase" == "final" ]]; then e10_run final_train --out-dir "$RUN_ROOT/v4/final" || return 1; fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "export" ]]; then
          e10_ckpt="$(ls -1 "$RUN_ROOT/v4/final"/*.fp32.pt 2>/dev/null | head -1)"
          if [[ -z "$e10_ckpt" ]]; then e10_ckpt="$(ls -1 "$RUN_ROOT/v4/final"/*.pt 2>/dev/null | head -1)"; fi
          if [[ -z "$e10_ckpt" ]]; then log "!! [E10] 找不到可用权重（先跑 --phase final）"; return 1; fi
          python3 "$HERE/E10/code/export_cpu.py" --ckpt "$e10_ckpt" \
            --out "$RUN_ROOT/v4/final" --reports-dir "$REPORTS_DIR" "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "build" ]]; then
          python3 "$HERE/E10/code/build_submission.py" --weights "$RUN_ROOT/v4/final" \
            --out "$e10_sub/submission_code_v4.zip" \
            --result-zip "$e10_sub/result.zip" \
            --manifest "$e10_sub/submission_manifest.json" \
            --reports-dir "$REPORTS_DIR" "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "b0" ]]; then
          python3 "$HERE/E10/code/build_b0_fallback.py" --reports-dir "$REPORTS_DIR" \
            --out "$e10_sub/submission_code_b0_fallback.zip" \
            "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "verify" ]]; then
          python3 "$HERE/E10/code/verify_inference.py" --reports-dir "$REPORTS_DIR" \
            --data-dir "$DATA_ROOT/v4/data" \
            --zip "$e10_sub/submission_code_v4.zip" \
            "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "submit" ]]; then
          python3 "$HERE/E10/code/submit.py" --reports-dir "$REPORTS_DIR" \
            --zip "$e10_sub/result.zip" \
            --code-zip "$e10_sub/submission_code_v4.zip" \
            --log "$REPORTS_DIR/E10_submission_log.json" \
            "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG" || return 1
        fi
        publish_final_to_network || return 1 ;;
    *)  log "!! 阶段 $STAGE 尚未实现（见 v4/PLAN.md §七 与各 E*/PLAN.md）"; return 1 ;;
  esac
}

ALL_TASK_NAMES=("" "env" "data" "e0" "E1" "E2" "E3-main" "E3-all" \
                "E4" "E5" "E6" "E7" "E8" "E9" "E10")

run_all_task() {
  local n="$1"
  local -a r=()
  if [[ "${ALL_RESUME_FLAG[0]:-}" == "--resume" ]]; then r=(--resume); fi
  case "$n" in
    1) run_env ;;
    2) run_data ;;
    3) run_e0 ;;
    4) STAGE="E1"; EXTRA_ARGS=("${r[@]+"${r[@]}"}"); run_stage ;;
    5) STAGE="E2"; EXTRA_ARGS=(); run_stage ;;
    6) STAGE="E3"; EXTRA_ARGS=(--phase main "${r[@]+"${r[@]}"}"); run_stage ;;
    # E3 消融的多个组合可能共用 checkpoint 目录，--resume 有串配置风险，故不自动续训。
    7) STAGE="E3"; EXTRA_ARGS=(--phase all "${r[@]+"${r[@]}"}"); run_stage ;;
    8) STAGE="E4"; EXTRA_ARGS=(--channel-independence-ablation --rel-pos-ablation \
                               --capacity-ablation "${r[@]+"${r[@]}"}"); run_stage ;;
    9) STAGE="E5"; EXTRA_ARGS=(--target all); run_stage ;;
    10) STAGE="E6"; EXTRA_ARGS=(--phase all); run_stage ;;
    11) STAGE="E7"; EXTRA_ARGS=(); run_stage ;;
    12) STAGE="E8"; EXTRA_ARGS=(--target all); run_stage ;;
    13) STAGE="E9"; EXTRA_ARGS=(); run_stage ;;
    14) STAGE="E10"; EXTRA_ARGS=(); run_stage ;;
    *)  log "!! unknown all task: $n"; return 1 ;;
  esac
}

start_state_mirror

case "$MODE" in
  env)   run_env ;;
  data)  run_data ;;
  e0)    run_e0 ;;
  smoke) run_smoke ;;
  data-health) check_env_profile full ;;
  stage) run_stage ;;
  all)
    if ! [[ "$ALL_THROUGH" =~ ^[0-9]+$ ]] || (( ALL_THROUGH < 1 || ALL_THROUGH > 14 )); then
      log "!! --through/--all-to 只支持 1..14，got $ALL_THROUGH"
      exit 2
    fi
    if [[ "$ALL_FRESH" == "1" ]]; then
      ALL_RESUME_FLAG=()
      rm -f "$ALL_PROGRESS" "$ALL_PROGRESS.tmp" "$NET_PROGRESS" "$V4_PAUSE_FLAG"
      log "[all] --fresh：清空本地+网络进度台账；已有 checkpoint 仍可由各训练脚本自行续训"
    else
      ALL_RESUME_FLAG=(--resume)
      if [[ ! -f "$ALL_PROGRESS" && -f "$NET_PROGRESS" ]]; then
        mkdir -p "$(dirname "$ALL_PROGRESS")"
        cp -a "$NET_PROGRESS" "$ALL_PROGRESS"
        log "[all] 本地无进度，已从 $NET_PROGRESS 恢复"
      fi
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      log "!! [all] 全链路模式忽略额外参数，请用 --through N 控制范围：${EXTRA_ARGS[*]}"
    fi
    restore_state_from_network || log "[all] 从 /data 恢复 checkpoint/OOF/report 失败（将按本地现有文件继续）"
    log "--- [all] 运行任务 1..$ALL_THROUGH（共 14 个）；进度 done 且本地产物齐备才跳过"
    for ((_n=1; _n<=ALL_THROUGH; _n++)); do
      _name="${ALL_TASK_NAMES[$_n]}"
      _status=""
      if [[ "$ALL_FRESH" != "1" ]]; then
        _status="$(task_status "$_n")"
      fi
      if [[ "$ALL_FRESH" != "1" && "$_status" == "done" ]] && task_local_ready "$_n"; then
        log "[all] task $_n/$_name already done 且本地产物齐备，跳过"
        continue
      fi
      if [[ "$ALL_FRESH" != "1" && "$_status" == "done" ]]; then
        log "[all] task $_n/$_name 进度 done 但本地产物缺失，需重跑"
      fi
      mark_progress "$_n" "$_name" running
      log "=== [all] task $_n/$_name 开始 ==="
      run_all_task "$_n" || {
        mark_progress "$_n" "$_name" failed
        publish_progress_to_network || true
        log "!! [all] task $_n/$_name 失败，链路停止（exit 21）"
        exit 21
      }
      mark_progress "$_n" "$_name" done
      publish_progress_to_network || log "[all] 进度同步 /data 失败（不阻塞当前任务）"
      log "=== [all] task $_n/$_name 完成 ==="
    done
    publish_final_to_network || { log "!! [all] 最终模型复制到 /data 失败（exit 22）"; exit 22; }
    mark_progress "DONE" "all" done
    publish_progress_to_network || true
    log "--- [all] 已完成到 task $ALL_THROUGH/${ALL_TASK_NAMES[$ALL_THROUGH]}"
    ;;
  *) log "unknown mode: $MODE"; exit 2 ;;
esac

log "--- 产物落盘情况（$DATA_ROOT 持久保留）"
du -sh "$RUN_ROOT" "$CACHE_ROOT" "$REPORTS_DIR" 2>/dev/null | tee -a "$LOG" || true
log "完成。"
