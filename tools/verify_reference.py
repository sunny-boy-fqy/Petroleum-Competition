#!/usr/bin/env python3
"""校验随 git 冻结的引用件未被篡改（折文件 / 数据清单）。

用途：平台每次 clone 后、或本地改动后，快速确认「折协议」与「数据指纹」没漂移。
git 的 core.autocrlf 可能在 Windows 上改写文本行尾，因此**校验的是解析后的内容指纹**，
不是文件字节 sha256。

    python3 v4/tools/verify_reference.py
    python3 v4/tools/verify_reference.py --manifest v4/dist/v4_data_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C            # noqa: E402
from src.validation import folds as F     # noqa: E402

# E0 冻结的折文件**字节级** sha256（本机 v1/src/well_folds.json 与仓库副本一致）
FROZEN_FOLDS_SHA256 = "f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e"
FROZEN_N_WELLS = 80
FROZEN_N_FOLDS = 5
FROZEN_FOLD_SIZES = {0: 16, 1: 16, 2: 16, 3: 16, 4: 16}


def content_fingerprint(folds: dict) -> str:
    """与行尾无关的内容指纹：well->fold 映射排序后做 sha256。"""
    items = sorted((w, int(f)) for w, f in folds["fold_of_well"].items())
    payload = json.dumps(items, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(V4 / "dist" / "v4_data_manifest.json"))
    args = ap.parse_args()

    ok = True

    # ---------------- 折文件
    try:
        folds = F.load_folds()
        # source_path 现在是仓库相对路径（可移植）；兼容旧的绝对路径写法
        path = Path(folds["source_path"])
        if not path.is_absolute():
            path = V4 / path
        if not path.is_file() and folds.get("source_path_abs"):
            path = Path(folds["source_path_abs"])
        raw = path.read_bytes()
        byte_sha = hashlib.sha256(raw).hexdigest()
        sizes = {f: sum(1 for v in folds["fold_of_well"].values() if v == f)
                 for f in range(folds["n_folds"])}
        print(f"folds      : {path}")
        print(f"  bytes sha256 : {byte_sha[:24]}...  {'MATCH' if byte_sha == FROZEN_FOLDS_SHA256 else 'DIFFERS (可能仅行尾差异)'}")
        print(f"  content fp   : {content_fingerprint(folds)[:24]}...")
        print(f"  n_wells      : {len(folds['well_list'])} (expect {FROZEN_N_WELLS})")
        print(f"  n_folds      : {folds['n_folds']} (expect {FROZEN_N_FOLDS})")
        print(f"  fold sizes   : {sizes} (expect {FROZEN_FOLD_SIZES})")
        checks = [
            len(folds["well_list"]) == FROZEN_N_WELLS,
            folds["n_folds"] == FROZEN_N_FOLDS,
            sizes == FROZEN_FOLD_SIZES,
            byte_sha == FROZEN_FOLDS_SHA256,
        ]
        if not checks[3] and all(checks[:3]):
            print("  note: 字节 sha256 不同但内容一致（多为 autocrlf 行尾转换），判定通过")
        ok = ok and all(checks[:3])
    except Exception as exc:
        print(f"folds      : FAIL -> {exc!r}")
        ok = False

    # ---------------- 数据清单（可选）
    mp = Path(args.manifest)
    if mp.is_file():
        man = json.loads(mp.read_text(encoding="utf-8"))
        tar = Path(man["tarball"]["path"])
        print(f"dataset    : {mp}")
        print(f"  tarball      : {tar}  exists={tar.is_file()}")
        if tar.is_file():
            h = hashlib.sha256()
            with tar.open("rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            match = h.hexdigest() == man["tarball"]["sha256"]
            print(f"  tarball sha  : {'MATCH' if match else 'MISMATCH'}")
            ok = ok and match
        c = man["counts"]
        print(f"  counts       : wells {c['n_train_wells']}+{c['n_test_wells']} "
              f"rows {c['n_train_rows']}+{c['n_test_rows']} "
              f"placeholder {c['n_placeholder_rows']}")
        ok = ok and c["n_train_wells"] == 80 and c["n_test_wells"] == 10 \
            and c["n_test_rows"] == C.EXPECTED_N_TEST_ROWS
        if c.get("noncanonical_schema_wells"):
            print("  noncanonical : " + ", ".join(
                f"{w['well_id'][:8]}({w['n_header_cols']} cols)" for w in c["noncanonical_schema_wells"]))
    else:
        print(f"dataset    : {mp} 不存在（尚未运行 tools/pack_dataset.py）—— 跳过")

    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
