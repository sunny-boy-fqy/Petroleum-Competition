#!/usr/bin/env python3
"""WP8：类型井选择报告（只读输入曲线，不使用任何标签）。

用法::

    python3 E8/code/type_well_report.py --cache-root $V4_CACHE_ROOT \
        --reports-dir $V4_REPORTS_DIR --topk 3 --method kl

产出 `$REPORTS/E8_type_well.json`：每口测试井的 top-k 训练类型井与距离。
后续训练可用该报告选类型井做 scaler 参考/分布匹配/适配成员。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src.data import dataset as D  # noqa: E402
from src.data import type_well as TW  # noqa: E402


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WP8 类型井选择报告")
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--run-root", default=None, help="兼容 run_train.sh 统一传参；本脚本不使用")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--method", default="kl", choices=("kl", "dtw", "signature"))
    ap.add_argument("--curves", default="GR,AC,DEN,CNL,RT,RXO")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--tag", default="")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cache = Path(args.cache_root)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    train_dir = cache / "raw" / "train"
    test_dir = cache / "raw" / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        print(f"[E8/type-well] FATAL: 缺少 raw 分片：{train_dir} / {test_dir}",
              file=sys.stderr)
        return 4
    train_wells = sorted(p.stem for p in train_dir.glob("*.npz"))
    test_wells = sorted(p.stem for p in test_dir.glob("*.npz"))
    if args.max_wells:
        train_wells = train_wells[:args.max_wells]
        test_wells = test_wells[:args.max_wells]
    curves = tuple(c.strip().upper() for c in str(args.curves).split(",") if c.strip())
    train_shards = {w: D.read_well_shard(cache, w, "train") for w in train_wells}
    test_shards = {w: D.read_well_shard(cache, w, "test") for w in test_wells}
    t0 = time.time()
    out = TW.select_type_wells_batch(test_shards, train_shards, topk=args.topk,
                                     method=args.method, curves=curves)
    report = {"stage": "E8", "p_stage": "type-well", "tag": args.tag,
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "method": args.method, "topk": int(args.topk), "curves": list(curves),
              "n_train_wells": len(train_wells), "n_test_wells": len(test_wells),
              "seconds": round(time.time() - t0, 2), "results": out,
              "note": ("只使用输入曲线；类型井选择结果供 scaler 参考/分布匹配/"
                       "适配成员使用，不参与标签选择。")}
    path = reports / f"E8_type_well{('_' + args.tag) if args.tag else ''}.json"
    write_json(path, report)
    print(json.dumps({"n_test_wells": len(test_wells), "method": args.method,
                      "report": str(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
