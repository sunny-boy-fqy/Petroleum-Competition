#!/usr/bin/env bash
# 把数据部署到平台**持久化**目录 /data（云盘挂载点）。
#
# 为什么必须放在 /data：
#   平台把 Git 仓库代码解压到**临时**目录 /code/workspace，任务结束即丢失；
#   只有 /data（云盘）的内容会保留。因此：
#     代码   -> /code/workspace/v4/...        （每次任务重新 clone，可丢）
#     数据   -> /data/v4/data/...             （只部署一次，永久保留）
#     缓存   -> /data/v4/cache/...
#     运行产物 -> /data/v4/runs/...
#
# 用法（在本机或云端均可执行）：
#   bash v4/tools/bootstrap_data.sh                 # 解压 dist/v4_data.tar.gz 到 /data/v4/data
#   V4_DATA_ROOT=/somewhere bash v4/tools/bootstrap_data.sh
#   bash v4/tools/bootstrap_data.sh --from-dir /path/to/raw   # 直接从原始目录复制
#   bash v4/tools/bootstrap_data.sh --verify-only            # 只校验，不写入
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V4="$(cd "$HERE/.." && pwd)"
DATA_ROOT="${V4_DATA_ROOT:-/data}"
DEST="$DATA_ROOT/v4/data"
TARBALL="$V4/dist/v4_data.tar.gz"
MANIFEST="$V4/dist/v4_data_manifest.json"
FROM_DIR=""
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-dir) FROM_DIR="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

echo "== v4 bootstrap_data =="
echo "repo       : $V4"
echo "data root  : $DATA_ROOT"
echo "dest       : $DEST"

mkdir -p "$DEST"

if [[ "$VERIFY_ONLY" == "1" ]]; then
  :
elif [[ -n "$FROM_DIR" ]]; then
  echo "copy from  : $FROM_DIR"
  mkdir -p "$DEST/train" "$DEST/test"
  cp -f "$FROM_DIR"/train/*.txt "$DEST/train/" 2>/dev/null || true
  cp -f "$FROM_DIR"/test/*.txt  "$DEST/test/"  2>/dev/null || true
elif [[ -f "$TARBALL" ]]; then
  if [[ -f "$MANIFEST" ]]; then
    WANT=$(python3 -c "import json,sys;print(json.load(open('$MANIFEST'))['tarball']['sha256'])" 2>/dev/null || echo "")
    GOT=$(python3 - "$TARBALL" <<'PY' 2>/dev/null || echo ""
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for b in iter(lambda:f.read(1<<20),b''): h.update(b)
print(h.hexdigest())
PY
)
    if [[ -n "$WANT" && "$WANT" != "$GOT" ]]; then
      echo "!! tarball sha256 mismatch"; echo "   want $WANT"; echo "   got  $GOT"; exit 3
    fi
    echo "tarball sha256 OK"
  fi
  # 解压到 DATA_ROOT，tar 内路径为 data/train/... 与 data/test/...
  tar xzf "$TARBALL" -C "$DATA_ROOT"
else
  echo "!! 既没有 --from-dir 也没有 $TARBALL"
  echo "   请先在本机运行: python3 v4/tools/pack_dataset.py"
  exit 4
fi

# 折文件也放进 /data，使运行时完全不依赖 repo 外的路径
mkdir -p "$DEST/folds"
cp -f "$V4/versions/reference/v1_well_folds.json" "$DEST/folds/" 2>/dev/null || true

# ---------------- 校验
python3 - "$DEST" "$MANIFEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
dest = Path(sys.argv[1]); man_path = Path(sys.argv[2])
n_tr = len(list((dest/"train").glob("*.txt"))) if (dest/"train").is_dir() else 0
n_te = len(list((dest/"test").glob("*.txt"))) if (dest/"test").is_dir() else 0
rows_tr = sum(sum(1 for _ in open(f, encoding="utf-8-sig")) - 2 for f in (dest/"train").glob("*.txt"))
rows_te = sum(sum(1 for _ in open(f, encoding="utf-8-sig")) - 2 for f in (dest/"test").glob("*.txt"))
print(f"train wells={n_tr} rows={rows_tr}")
print(f"test  wells={n_te} rows={rows_te}")
ok = (n_tr == 80 and n_te == 10 and rows_te == 95948)
if man_path.is_file():
    man = json.loads(man_path.read_text(encoding="utf-8"))
    c = man["counts"]
    print("manifest expects:", {k: c[k] for k in ("n_train_wells","n_test_wells","n_train_rows","n_test_rows")})
    ok = ok and (rows_tr == c["n_train_rows"])
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
echo "  python3 /code/workspace/v4/E0/code/check_env.py --json $DATA_ROOT/v4/reports/E0_env.json"
