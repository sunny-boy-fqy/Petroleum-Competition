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
#   --mode all      依次执行 env -> data -> e0 -> E1 ...
#
# 平台集成（已按官方提示落实）：
#   * TensorBoard：导出 TENSORBOARD_LOGDIR=$V4_DATA_ROOT/v4/tb，
#     训练脚本用 src/training/tb_logger.py::RunLogger 写指标，平台任务详情页可见曲线。
#   * 云盘持久化：所有产物（cache/runs/reports/logs/tb）都在 $V4_DATA_ROOT(=/data)/v4 下，
#     任务结束或资源释放后仍保留。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # 自定位：/code/workspace/<仓库名>
DATA_ROOT="${V4_DATA_ROOT:-/data}"
RUN_ROOT="${V4_RUN_ROOT:-$DATA_ROOT/v4/runs}"
CACHE_ROOT="${V4_CACHE_ROOT:-$DATA_ROOT/v4/cache}"
LOG_DIR="${V4_LOG_DIR:-$DATA_ROOT/v4/logs}"
REPORTS_DIR="$DATA_ROOT/v4/reports"

MODE="all"
STAGE="E1"
EXTRA_ARGS=()
# R5-B1：git 仓库里**没有** dist/*.tar.gz（.gitignore 忽略），云端必须能从云盘找到它。
# 因此 tarball/manifest 是一等参数（不会被塞进 EXTRA_ARGS 污染 smoke/stage 的命令行）。
DATA_TARBALL="${V4_DATA_TARBALL:-}"
DATA_MANIFEST="${V4_DATA_MANIFEST:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)     MODE="$2"; shift 2 ;;
    --stage)    STAGE="$2"; shift 2 ;;
    --tarball)  DATA_TARBALL="$2"; shift 2 ;;
    --manifest) DATA_MANIFEST="$2"; shift 2 ;;
    *)          EXTRA_ARGS+=("$1"); shift ;;
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
# 让 bootstrap_data.sh 也能看到 tarball 位置（同一份事实，不重复解析参数）
if [[ -n "$DATA_TARBALL" ]]; then export V4_DATA_TARBALL="$DATA_TARBALL"; fi
if [[ -n "$DATA_MANIFEST" ]]; then export V4_DATA_MANIFEST="$DATA_MANIFEST"; fi
# 平台集成 TensorBoard：日志写到 TENSORBOARD_LOGDIR（若有）
export TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$DATA_ROOT/v4/tb}"
mkdir -p "$TENSORBOARD_LOGDIR"
# 避免任何东西写到临时目录导致丢失
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=============================================================="
log "v4 training task  mode=$MODE stage=$STAGE"
log "repo(HERE)   = $HERE            <- /code/workspace/<仓库名> (临时，实测值即本行)"
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
    check_env_profile full || log "!! [env] full 校验未通过（数据健康有问题）"
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
  bash "$HERE/tools/bootstrap_data.sh" ${bargs[@]+"${bargs[@]}"} 2>&1 | tee -a "$LOG"

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
  # 流水线完整性：阶段脚本/入口/接线/报告命名一次性核对（失败不阻塞 E0，但会留下证据）
  python3 "$HERE/tools/check_pipeline.py" --json "$REPORTS_DIR/E0_pipeline_check.json" \
    2>&1 | tee -a "$LOG" || log "!! [e0] 流水线完整性检查未通过（见 E0_pipeline_check.json）"
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
    E2)
        # `--phase build|ablate|all`（build=F2 特征缓存/溯源；ablate=单组消融+吞吐）
        e2_phase="all"
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          case "${a,,}" in build|ablate|all) e2_phase="${a,,}" ;; esac
        done
        if [[ "$e2_phase" == "all" || "$e2_phase" == "build" ]]; then
          log "--- [E2] 构建 F2 特征缓存 + 溯源/审计"
          python3 "$HERE/E2/code/build_features.py" --cache-root "$CACHE_ROOT" \
            --reports-dir "$REPORTS_DIR" 2>&1 | tee -a "$LOG"
        fi
        if [[ "$e2_phase" == "all" || "$e2_phase" == "ablate" ]]; then
          log "--- [E2] 单组消融 + 吞吐画像"
          python3 "$HERE/E2/code/ablate_groups.py" --cache-root "$CACHE_ROOT" \
            --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT" 2>&1 | tee -a "$LOG"
        fi ;;
    E3) python3 "$HERE/E3/code/train_seq.py" \
          --train-dir "$DATA_ROOT/v4/data/train" --cache-root "$CACHE_ROOT" \
          --out-dir "$RUN_ROOT/E3" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "$LOG" ;;
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
            --reports-dir "$REPORTS_DIR" --run-root "$RUN_ROOT" 2>&1 | tee -a "$LOG"
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
              p0|p1|all) e6_phase="${a,,}" ;;
              *) log "!! E6 --phase 只支持 p0/p1/all，got $a"; return 1 ;;
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
            --run-root "$RUN_ROOT" "${e6_args[@]+"${e6_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi
        if [[ "$e6_phase" == "p1" || "$e6_phase" == "all" ]]; then
          log "--- [E6] P1 τ 搜索（内折 OOF）"
          python3 "$HERE/E6/code/search_tau.py" \
            --run-root "$RUN_ROOT" --reports-dir "$REPORTS_DIR" 2>&1 | tee -a "$LOG"
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
            --cache-root "$CACHE_ROOT" "${e7_args[@]+"${e7_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi
        if [[ "$e7_phase" == "decode" || "$e7_phase" == "all" ]]; then
          log "--- [E7] P1 解码搜索"
          python3 "$HERE/E7/code/decode_search.py" --reports-dir "$REPORTS_DIR" \
            --run-root "$RUN_ROOT" "${e7_args[@]+"${e7_args[@]}"}" 2>&1 | tee -a "$LOG"
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
        # `--phase aggregate|choose|leakage|confirm|submit|all`
        e9_phase="all"
        for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
          case "${a,,}" in
            aggregate|choose|leakage|confirm|submit|all) e9_phase="${a,,}" ;;
          esac
        done
        run_e9() {
          local stage="$1"; shift
          log "--- [E9] $stage"
          python3 "$HERE/E9/code/$stage.py" --reports-dir "$REPORTS_DIR" \
            --run-root "$RUN_ROOT" 2>&1 | tee -a "$LOG"
        }
        if [[ "$e9_phase" == "all" || "$e9_phase" == "aggregate" ]]; then run_e9 aggregate_oof || return 1; fi
        if [[ "$e9_phase" == "all" || "$e9_phase" == "choose" ]]; then run_e9 choose_submission || return 1; fi
        if [[ "$e9_phase" == "all" || "$e9_phase" == "leakage" ]]; then run_e9 leakage_audit || return 1; fi
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
        if [[ "$e10_phase" == "all" || "$e10_phase" == "final" ]]; then e10_run final_train --out-dir "$RUN_ROOT/v4/final" || return 1; fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "export" ]]; then
          e10_ckpt="$(ls -1 "$RUN_ROOT/v4/final"/*.fp32.pt 2>/dev/null | head -1)"
          if [[ -z "$e10_ckpt" ]]; then e10_ckpt="$(ls -1 "$RUN_ROOT/v4/final"/*.pt 2>/dev/null | head -1)"; fi
          if [[ -z "$e10_ckpt" ]]; then log "!! [E10] 找不到可用权重（先跑 --phase final）"; return 1; fi
          python3 "$HERE/E10/code/export_cpu.py" --ckpt "$e10_ckpt" \
            --out "$RUN_ROOT/v4/final" --reports-dir "$REPORTS_DIR" "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "build" ]]; then
          python3 "$HERE/E10/code/build_submission.py" --weights "$RUN_ROOT/v4/final" \
            --reports-dir "$REPORTS_DIR" "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "b0" ]]; then
          python3 "$HERE/E10/code/build_b0_fallback.py" --reports-dir "$REPORTS_DIR" \
            "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "verify" ]]; then
          python3 "$HERE/E10/code/verify_inference.py" --reports-dir "$REPORTS_DIR" \
            --data-dir "$DATA_ROOT/v4/data" "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi
        if [[ "$e10_phase" == "all" || "$e10_phase" == "submit" ]]; then
          python3 "$HERE/E10/code/submit.py" --reports-dir "$REPORTS_DIR" \
            "${e10_args[@]+"${e10_args[@]}"}" 2>&1 | tee -a "$LOG"
        fi ;;
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
