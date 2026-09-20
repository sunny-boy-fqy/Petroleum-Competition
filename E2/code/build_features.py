#!/usr/bin/env python3
"""E2/P0+P1：构建 F2 特征缓存 + 溯源表 + 内存/磁盘画像。

用法（云端）：:

    python3 E2/code/build_features.py --train-dir /data/v4/data/train \\
        --test-dir /data/v4/data/test --cache-root /data/v4/cache

产出::

    $CACHE/feat/<spec_key>/{train,test}/<well>.npz     特征分片（原子写、幂等）
    $REPORTS/E2_feature_provenance.csv                 每列公式与依据（E2/P0 §4）
    $REPORTS/E2_mem_profile.json                       峰值内存/缓存体积/耗时
    $REPORTS/E2_feature_spec.json                      本次用的 FeatureSpec（列序指纹）

纪律
----
- `phys` 组的分位数基线（GR/SP 的 P5/P95）**只由训练井拟合**，并写进
  `E2_mem_profile.json::physics_params`；测试井用同一份参数（推理期一致）。
- 缓存上限 2.0 GB（E2/P1 §6）；超限时用 `--squeeze`（窗口统计减到 mean/std、窗长减到
  `{11,51}`）或先删旧版本目录（`--clean`）。
- 幂等：同 `spec_key` 已存在的分片默认跳过（`--force` 重写）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

from src.data import dataset as D  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.features import physics as PH  # noqa: E402


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def default_cache_root() -> Path:
    return _env_path("V4_CACHE_ROOT", str(_env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))


def default_reports_dir() -> Path:
    return _env_path("V4_REPORTS_DIR", str(_env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))


def peak_rss_gib() -> float:
    """进程峰值常驻内存（Linux ru_maxrss 单位是 KiB）。"""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3)


def write_json(path: str | Path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def build_spec(args) -> G.FeatureSpec:
    if args.squeeze:
        return G.FeatureSpec(groups=G.F2_GROUPS, windows=(11, 51),
                             win_stats=("mean", "std"), well_stats=("mean", "std", "p50"),
                             well_scalars=("n_rows", "depth_span", "miss_frac_mean"))
    return G.spec_from_name(args.spec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 E2 feature builder (F_phys/F_win/F_well)")
    ap.add_argument("--train-dir", default=os.environ.get("V4_TRAIN_DIR"))
    ap.add_argument("--test-dir", default=os.environ.get("V4_TEST_DIR"))
    ap.add_argument("--cache-root", default=str(default_cache_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--spec", default="F2", help="F1 / F2 / F1+phys / F1+win / F1+well / ...")
    ap.add_argument("--squeeze", action="store_true", help="收缩模式（E2/P1 §6 的降级配置）")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="重写已存在的分片")
    ap.add_argument("--clean", action="store_true", help="先删该 spec 的旧缓存目录")
    ap.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = ap.parse_args(argv)

    spec = build_spec(args)
    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not (cache / "raw" / "train").is_dir():
        if not args.train_dir:
            print("[E2] FATAL: 缺少 raw 分片且未提供 --train-dir", file=sys.stderr)
            return 4
        D.build_cache(args.train_dir, args.test_dir or args.train_dir, cache)
    train_wells = sorted(p.stem for p in (cache / "raw" / "train").glob("*.npz"))
    test_wells = sorted(p.stem for p in (cache / "raw" / "test").glob("*.npz"))
    if args.max_wells:
        train_wells, test_wells = train_wells[:args.max_wells], test_wells[:args.max_wells]

    # 物理分位数基线：**只用训练井**
    phys = G.fit_physics_params(D.read_well_shard(cache, w, "train") for w in train_wells)

    if args.clean:
        import shutil
        d = G.feature_dir(cache, spec, "train").parent
        if d.is_dir():
            shutil.rmtree(d)
            print(f"[E2] cleaned {d}")

    if args.force:
        for split, wells in (("train", train_wells), ("test", test_wells)):
            for w in wells:
                p = G.feature_path(cache, spec, w, split)
                if p.is_file():
                    p.unlink()

    prof: dict = {"spec": spec.as_dict(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for split, wells in (("train", train_wells), ("test", test_wells)):
        info = G.build_feature_cache(
            cache, spec, wells, phys, split=split, verbose=True,
            from_raw=lambda w, _s=split: D.read_well_shard(cache, w, _s))
        prof[f"cache_{split}"] = info
        print(f"[E2] {spec.key} {split}: built={info['built']} skipped={info['skipped']} "
              f"rows={info['rows']} bytes={info['bytes'] / 1e6:.1f} MB "
              f"({info['seconds']}s)", flush=True)

    total_bytes = prof["cache_train"]["bytes"] + prof["cache_test"]["bytes"]
    # 逐列缺失率与列数（用第一口井抽检；完整统计在 ablate_groups.py 的 OOF 里）
    sample = G.read_feature_cache(cache, spec, train_wells[0], "train")[0] if train_wells else None
    prov = G.provenance_rows(spec)
    prov_path = reports / "E2_feature_provenance.csv"
    with prov_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["column", "group", "formula", "source"])
        wr.writeheader()
        for r in prov:
            wr.writerow(r)

    prof.update({
        "n_features": spec.n_features(), "group_sizes": spec.group_sizes(),
        "provenance_groups": G.provenance_group_counts(spec),
        "provenance_csv": str(prov_path), "provenance_rows": len(prov),
        "n_train_wells": len(train_wells), "n_test_wells": len(test_wells),
        "cache_bytes_total": total_bytes, "cache_gb_total": round(total_bytes / 1e9, 4),
        "cache_limit_gb": args.cache_limit_gb,
        "within_cache_limit": bool(total_bytes / 1e9 <= args.cache_limit_gb),
        "physics_params": phys.as_dict(),
        "physics_params_fitted_on": "train_wells_only",
        "peak_rss_gib": peak_rss_gib(),
        "seconds_total": round(time.time() - t0, 2),
        "sample_nan_frac": (None if sample is None
                            else float((~__import__("numpy").isfinite(sample)).mean())),
        "label_free": G.audit_no_target_derivation(spec),
        "squeeze": bool(args.squeeze),
    })
    write_json(reports / "E2_mem_profile.json", prof)
    write_json(reports / "E2_feature_spec.json",
               {"spec": spec.as_dict(), "names": spec.names(),
                "provenance_groups": G.provenance_group_counts(spec)})

    print(json.dumps({"spec_key": spec.key, "n_features": spec.n_features(),
                      "cache_gb": prof["cache_gb_total"],
                      "within_limit": prof["within_cache_limit"],
                      "peak_rss_gib": prof["peak_rss_gib"],
                      "provenance_rows": len(prov),
                      "physics_params": phys.as_dict()}, ensure_ascii=False, indent=2))
    # 内存预算：E2/P1 §7 要求峰值常驻 < 12 GiB
    if prof["peak_rss_gib"] >= 12.0:
        print(f"!! [E2] 峰值常驻 {prof['peak_rss_gib']} GiB ≥ 12 GiB（E2/P1 §7）", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
