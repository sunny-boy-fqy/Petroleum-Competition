#!/usr/bin/env bash
# 把数据部署到平台**持久化**目录 /data（云盘挂载点）。
#
# 为什么必须放在 /data：
#   平台把 Git 仓库代码解压到**临时**目录 /code/workspace，任务结束即丢失；
#   只有 /data（云盘）的内容会保留。因此：
#     代码   -> /code/workspace/...           （zip 上传时仓库根直接是 /code/workspace）
#     数据   -> /data/v4/data/...             （只部署一次，永久保留）
#     缓存   -> /data/v4/cache/...
#     运行产物 -> /data/v4/runs/...
#
# tarball 在哪里（R5-B1：云端 repo 内**不可能**有 dist/*.tar.gz，它被 .gitignore 忽略）
# ---------------------------------------------------------------------------
# 定位顺序（先显式、后约定；找不到就报错并列出全部搜索位置）：
#   1) `--tarball PATH` 或 `$V4_DATA_TARBALL`（最高优先，指向任何位置）
#   2) `$V4/dist/v4_data.tar.gz`            本机开发：pack_dataset.py 的产物
#   3) `$V4_NETWORK_ROOT/v4_data.tar.gz`    云端推荐：tarball 传到网络盘 /data 根
#   4) `$V4_NETWORK_ROOT/dist/v4_data.tar.gz` 云端：网络盘保留 dist/ 目录结构
#   5) `$DATA_ROOT/v4_data.tar.gz`          本地运行时备份位置（兼容/可选）
#   6) `$DATA_ROOT/dist/v4_data.tar.gz`     本地运行时 dist/（兼容/可选）
# manifest（可选但推荐）同法搜索：`--manifest` / `$V4_DATA_MANIFEST`，然后与
# tarball 同目录、再上述 dist 位置。manifest **存在**时额外校验 tarball sha256
# 与逐项计数；**不存在**时仍无条件校验 80/10 井与 730,268/95,948 行（R5-H2）。
# R6-L1：显式给出的 `--manifest` 路径不存在 -> exit 4；manifest 缺 `tarball.sha256`
#         -> exit 3（不得打印 "tarball sha256 OK" 假装校验通过）。
#
# 用法（**本机**在项目父目录执行可用 `v4/` 前缀；**云端**请用 run_train.sh
#      --mode data，或先 V4="$(dirname "$(find /code/workspace -name run_train.sh | head -1)")"）：
#   bash v4/tools/bootstrap_data.sh                      # 自动搜索（含 $DATA_ROOT）
#   bash v4/tools/bootstrap_data.sh --tarball /data/v4_data.tar.gz
#   V4_DATA_TARBALL=/data/v4_data.tar.gz bash v4/tools/bootstrap_data.sh
#   bash v4/tools/bootstrap_data.sh --from-dir /path/to/raw   # 直接从原始目录复制
#   bash v4/tools/bootstrap_data.sh --verify-only            # 只读校验：不建目录、不写文件
#   bash v4/tools/bootstrap_data.sh --tarball X --dest Y      # 解压后搬运到 Y（R6-L2）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V4="$(cd "$HERE/.." && pwd)"
DATA_ROOT="${V4_DATA_ROOT:-/data}"                 # 运行时数据根（run_train 传本地盘）
NETWORK_ROOT="${V4_NETWORK_ROOT:-/data}"         # 网络盘：只读 tarball / 最终模型回写
DEST="$DATA_ROOT/v4/data"
FROM_DIR=""
VERIFY_ONLY=0
DEST_EXPLICIT=0
# 显式指定的路径优先（命令行 > 环境变量 > 约定位置）
TARBALL="${V4_DATA_TARBALL:-}"
MANIFEST="${V4_DATA_MANIFEST:-}"
MANIFEST_EXPLICIT=0
if [[ -n "$MANIFEST" ]]; then MANIFEST_EXPLICIT=1; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-dir) FROM_DIR="$2"; shift 2 ;;
    --tarball) TARBALL="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; MANIFEST_EXPLICIT=1; shift 2 ;;
    --dest) DEST="$2"; DEST_EXPLICIT=1; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

echo "== v4 bootstrap_data =="
echo "repo       : $V4"
echo "data root  : $DATA_ROOT"
echo "dest       : $DEST"

# ---------------- 0) 显式路径必须先存在（R6-L1）
# 用户显式写下 --manifest 却把路径写错，语义上**不等于**"没有 manifest"，
# 静默降级会让一次真实的拼写错误伪装成 RESULT: OK。
if [[ "$MANIFEST_EXPLICIT" == "1" && ! -f "$MANIFEST" ]]; then
  echo "!! --manifest / \$V4_DATA_MANIFEST 指向的文件不存在：$MANIFEST"
  echo "   自动搜索到的 manifest 才允许缺席；显式路径写错必须显式报错。"
  exit 4
fi

# ---------------- 1) 定位 tarball（R5-B1）
TARBALL_DIRS=("$V4/dist" "$NETWORK_ROOT" "$NETWORK_ROOT/dist" "$DATA_ROOT" "$DATA_ROOT/dist")
if [[ "$VERIFY_ONLY" != "1" && -z "$FROM_DIR" ]]; then
  if [[ -n "$TARBALL" ]]; then
    if [[ ! -f "$TARBALL" ]]; then
      echo "!! --tarball / \$V4_DATA_TARBALL 指向的文件不存在：$TARBALL"
      exit 4
    fi
  else
    for cand in "${TARBALL_DIRS[@]}"; do
      if [[ -f "$cand/v4_data.tar.gz" ]]; then TARBALL="$cand/v4_data.tar.gz"; break; fi
    done
  fi
  if [[ -z "$TARBALL" ]]; then
    echo "!! 找不到数据分发包 v4_data.tar.gz。已搜索以下位置："
    echo "     --tarball <path> / \$V4_DATA_TARBALL   （未设置）"
    for cand in "${TARBALL_DIRS[@]}"; do echo "     $cand/v4_data.tar.gz"; done
    echo "   本机生成： python3 v4/tools/pack_dataset.py"
    echo "   云端部署： 把 v4_data.tar.gz 上传到 $DATA_ROOT/（或 $DATA_ROOT/dist/），"
    echo "              或显式传 --tarball <云盘上的路径> / 设 V4_DATA_TARBALL=<路径>"
    exit 4
  fi
  echo "tarball    : $TARBALL"
fi

# ---------------- 2) 定位 manifest（自动搜索；未找到 -> 无 manifest 分支）
if [[ "$MANIFEST_EXPLICIT" != "1" ]]; then
  MANIFEST_DIRS=()
  if [[ -n "$TARBALL" ]]; then MANIFEST_DIRS+=("$(dirname "$TARBALL")"); fi
  MANIFEST_DIRS+=("${TARBALL_DIRS[@]}")
  for cand in "${MANIFEST_DIRS[@]}"; do
    if [[ -f "$cand/v4_data_manifest.json" ]]; then
      MANIFEST="$cand/v4_data_manifest.json"; break
    fi
  done
fi

if [[ "$VERIFY_ONLY" == "1" ]]; then
  # R6-L2：`--verify-only` 只读，**不创建任何目录、不复制任何文件**
  :
elif [[ -n "$FROM_DIR" ]]; then
  echo "copy from  : $FROM_DIR"
  mkdir -p "$DEST/train" "$DEST/test"
  cp -f "$FROM_DIR"/train/*.txt "$DEST/train/" 2>/dev/null || true
  cp -f "$FROM_DIR"/test/*.txt  "$DEST/test/"  2>/dev/null || true
else
  if [[ -f "$MANIFEST" ]]; then
    echo "manifest   : $MANIFEST"
    WANT=$(python3 -c "import json,sys;print(json.load(open('$MANIFEST'))['tarball']['sha256'])" 2>/dev/null || echo "")
    GOT=$(python3 - "$TARBALL" <<'PY' 2>/dev/null || echo ""
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for b in iter(lambda:f.read(1<<20),b''): h.update(b)
print(h.hexdigest())
PY
)
    if [[ -z "$WANT" ]]; then
      # R6-L1：manifest 存在但缺 tarball.sha256 —— 不得假装"校验通过"
      echo "!! manifest 缺少 tarball.sha256 字段：$MANIFEST"
      echo "   无法校验分发包完整性（请重新生成 manifest：python3 v4/tools/pack_dataset.py）"
      exit 3
    fi
    if [[ "$WANT" != "$GOT" ]]; then
      echo "!! tarball sha256 mismatch"; echo "   want $WANT"; echo "   got  $GOT"; exit 3
    fi
    echo "tarball sha256 OK"
  else
    echo "manifest   : (自动搜索未找到，跳过 sha256 校验；行数/井数仍会硬校验)"
  fi
  # tar 内路径为 v4/data/train/... 与 v4/data/test/...（R6-L3），因此
  # 默认解压目标是 $DATA_ROOT（$DEST = $DATA_ROOT/v4/data）。
  if [[ "$DEST_EXPLICIT" == "1" && "$DEST" != "$DATA_ROOT/v4/data" ]]; then
    # R6-L2：显式 --dest 时先解到临时目录再搬运，使 `--dest` 真正生效
    # （此前 tar 永远写 $DATA_ROOT，校验却读 $DEST，语义不自洽）。
    echo "dest mode  : --dest 与默认不同 -> 解到临时目录后搬运到 $DEST"
    TMPX="$(mktemp -d "${TMPDIR:-/tmp}/v4_bootstrap_XXXXXX")"
    trap 'rm -rf "$TMPX"' EXIT
    tar xzf "$TARBALL" -C "$TMPX"
    for split in train test; do
      mkdir -p "$DEST/$split"
      if [[ -d "$TMPX/v4/data/$split" ]]; then
        cp -f "$TMPX/v4/data/$split"/*.txt "$DEST/$split/" 2>/dev/null || true
      fi
    done
    rm -rf "$TMPX"; trap - EXIT
  else
    mkdir -p "$DATA_ROOT"          # tar -C 需要目标存在（裸磁盘/首次部署时）
    tar xzf "$TARBALL" -C "$DATA_ROOT"
  fi
fi

# 折文件也放进 dest，使运行时完全不依赖 repo 外的路径（--verify-only 不写）
if [[ "$VERIFY_ONLY" != "1" ]]; then
  mkdir -p "$DEST/folds"
  cp -f "$V4/versions/reference/v1_well_folds.json" "$DEST/folds/" 2>/dev/null || true
fi

# ---------------- 3) 校验（R5-H2：井数 + 行数**无条件**硬校验；manifest 存在时再校 sha 计数）
python3 - "$DEST" "$MANIFEST" "$V4" <<'PY'
import json, sys
from pathlib import Path
dest = Path(sys.argv[1]); man_path = Path(sys.argv[2]); v4 = Path(sys.argv[3])
sys.path.insert(0, str(v4))
from src import constants as C            # 唯一事实源（stdlib-only，云端无 numpy 也能导入）

def n_wells(split):
    d = dest / split
    return len(list(d.glob("*.txt"))) if d.is_dir() else 0

def n_rows(split):
    """原始 txt 行数减去 2 行表头（列名行 + 单位行）。

    review R7：这个口径与 `parse.py`（跳过空行）存在潜在差异，因此先**显式检查**
    数据区没有空行 —— 有空行时总数会漂移，必须让部署**响亮地失败**而不是静默对不上。
    """
    d = dest / split
    if not d.is_dir():
        return 0
    total = 0
    for f in d.glob("*.txt"):
        lines = open(f, encoding="utf-8-sig").read().splitlines()
        blank = sum(1 for ln in lines[2:] if not ln.strip())
        if blank:
            raise SystemExit(f"!! {f.name} 数据区含 {blank} 个空行："
                             "行数口径（总行数-2 表头）会与 parse.py 不一致，请先确认数据")
        total += len(lines) - 2
    return total

n_tr, n_te = n_wells("train"), n_wells("test")
rows_tr, rows_te = n_rows("train"), n_rows("test")
print(f"train wells={n_tr} rows={rows_tr}  (expect {C.EXPECTED_N_TRAIN_WELLS} / {C.EXPECTED_N_TRAIN_ROWS})")
print(f"test  wells={n_te} rows={rows_te}  (expect {C.EXPECTED_N_TEST_WELLS} / {C.EXPECTED_N_TEST_ROWS})")

# 无条件硬校验（R5-H2）：此前只在 manifest 存在时才查 rows_tr，导致截断的 tarball 只要
# 保住 80/10 个文件与测试行数就能通过。
ok = (n_tr == C.EXPECTED_N_TRAIN_WELLS and n_te == C.EXPECTED_N_TEST_WELLS
      and rows_tr == C.EXPECTED_N_TRAIN_ROWS and rows_te == C.EXPECTED_N_TEST_ROWS)
for label, got, want in (("train wells", n_tr, C.EXPECTED_N_TRAIN_WELLS),
                         ("test wells", n_te, C.EXPECTED_N_TEST_WELLS),
                         ("train rows", rows_tr, C.EXPECTED_N_TRAIN_ROWS),
                         ("test rows", rows_te, C.EXPECTED_N_TEST_ROWS)):
    if got != want:
        print(f"  !! {label}: {got} != {want}")

if man_path.is_file():
    man = json.loads(man_path.read_text(encoding="utf-8"))
    c = man["counts"]
    keys = ("n_train_wells", "n_test_wells", "n_train_rows", "n_test_rows")
    print("manifest expects:", {k: c[k] for k in keys})
    pairs = (("n_train_wells", n_tr), ("n_test_wells", n_te),
             ("n_train_rows", rows_tr), ("n_test_rows", rows_te))
    for k, got in pairs:
        if got != c[k]:
            print(f"  !! manifest {k}: got {got}, manifest {c[k]}")
        ok = ok and got == c[k]

# 抽查 3 口畸形井是否解压完整
for w in ("42f2870b6ea743518d4ff77acca0462a", "b7eb1274305446c499a7b03848fb5bd3",
          "c7611b0148bb4b878c00bc6d1367a136"):
    f = dest/"train"/f"{w}.txt"
    print(f"  odd well {w[:8]}: {'OK' if f.is_file() else 'MISSING'}")
    ok = ok and f.is_file()
print("RESULT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo
echo "数据就绪。训练任务中请设置："
echo "  export V4_DATA_ROOT=$DATA_ROOT"
echo "  python3 \"$V4/E0/code/check_env.py\" --json $DATA_ROOT/v4/reports/E0_env.json"
if [[ "$DEST_EXPLICIT" == "1" && "$DEST" != "$DATA_ROOT/v4/data" ]]; then
  echo "注意：本次使用了自定义 --dest=$DEST；check_env/训练脚本默认读"
  echo "      \$V4_DATA_ROOT/v4/data，所以自定义 dest 只适合手工验证，正式部署请用默认 dest。"
fi
