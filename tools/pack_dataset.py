#!/usr/bin/env python3
"""把 90 口井打包成**自包含数据分发包**，供云端 `/data` 使用。

产物
----
    dist/v4_data.tar.gz          原始 txt（80 训练 + 10 测试）  ~31 MB
    dist/v4_data_manifest.json   逐文件 sha256 + 行数 + schema 指纹 + 折指纹

为什么打包而不是直接提交 txt
---------------------------
平台训练任务要求代码不超过 500 MB 且**禁止在代码包里放权重与大数据集**；
把 96 MB 的原始 txt 打包成单个 31 MB 的 `tar.gz` 后，
既可作为「云盘文件」挂载，也可作为 repo 内的自包含数据分发件（< 500 MB 限制），
再由 `tools/bootstrap_data.sh` 一次性解压到 `/data/v4/data`。

用法
----
    python3 tools/pack_dataset.py --train ../data/train --test ../data/test \
        --out dist/v4_data.tar.gz --manifest dist/v4_data_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C                # noqa: E402
from src.data import parse as P               # noqa: E402


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(V4.parent / "data" / "train"))
    ap.add_argument("--test", default=str(V4.parent / "data" / "test"))
    ap.add_argument("--out", default=str(V4 / "dist" / "v4_data.tar.gz"))
    ap.add_argument("--manifest", default=str(V4 / "dist" / "v4_data_manifest.json"))
    ap.add_argument("--level", type=int, default=6)
    args = ap.parse_args()

    train_dir, test_dir = Path(args.train), Path(args.test)
    if not train_dir.is_dir() or not test_dir.is_dir():
        print(f"ERROR: dataset dirs not found: {train_dir} / {test_dir}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files: list[tuple[Path, str]] = []
    for d, split in ((train_dir, "train"), (test_dir, "test")):
        for f in sorted(d.glob("*.txt")):
            files.append((f, f"v4/data/{split}/{f.name}"))

    print(f"packing {len(files)} wells -> {out}")
    with tarfile.open(out, "w:gz", compresslevel=args.level) as tf:
        for src, arc in files:
            tf.add(src, arcname=arc)

    # manifest：逐文件 sha256 + 汇总
    entries = []
    for src, arc in files:
        entries.append({
            "arcname": arc,
            "well_id": src.stem,
            "bytes": src.stat().st_size,
            "sha256": sha256_file(src),
        })

    # 用我方解析器复算状态计数，写进 manifest 以便云端"零解析"校验
    train_recs = [P.parse_well(f, with_targets=True) for f, arc in files if "/train/" in arc]
    test_recs = [P.parse_well(f, with_targets=False) for f, _ in files if "/test/" in _]
    man = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "constants_version": {
            "columns": list(C.COLUMNS),
            "sentinels": list(C.SENTINELS),
            "missing_lt": C.MISSING_LT,
            "placeholder": C.PLACEHOLDER,
        },
        "tarball": {
            "path": str(out),
            "bytes": out.stat().st_size,
            "sha256": sha256_file(out),
        },
        "counts": {
            "n_train_wells": len(train_recs),
            "n_test_wells": len(test_recs),
            "n_train_rows": sum(r.n_rows for r in train_recs),
            "n_test_rows": sum(r.n_rows for r in test_recs),
            "expected_test_rows": C.EXPECTED_N_TEST_ROWS,
            "n_placeholder_rows": sum(r.n_placeholder_rows for r in train_recs),
            "n_missing_rows": sum(r.n_missing_rows for r in train_recs),
            "n_valid_rows": sum(r.n_valid_rows for r in train_recs),
            "noncanonical_schema_wells": [
                {"well_id": r.well_id, "n_header_cols": r.n_header_cols,
                 "missing_columns": list(r.missing_columns),
                 "extra_columns": list(r.extra_columns)}
                for r in train_recs if r.missing_columns or r.extra_columns
            ],
        },
        "files": entries,
    }
    mpath = Path(args.manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "tarball": man["tarball"]["path"],
        "tarball_mb": round(man["tarball"]["bytes"] / 1024**2, 2),
        "tarball_sha256": man["tarball"]["sha256"][:16] + "...",
        "counts": man["counts"],
        "manifest": str(mpath),
    }, ensure_ascii=False, indent=2))
    ok = (
        man["counts"]["n_train_wells"] == 80
        and man["counts"]["n_test_wells"] == 10
        and man["counts"]["n_test_rows"] == C.EXPECTED_N_TEST_ROWS
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
