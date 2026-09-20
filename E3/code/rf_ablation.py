#!/usr/bin/env python3
"""E3/P2：**感受野消融**（depth {3,5} × dilation_max {64,512}）。

纪律（E3/P2 §5 步 4）：先在**每个 outer 折的 inner-OOF** 上筛查（本脚本用 `--folds`
指定折），胜者才跑全 5 折；fold0 单独做资源预检时结论必须标 `exploratory=true`
且**不得进入 Gate 数值**。若 512 与 64 无差异 -> 有效感受野已饱和，**禁止**继续加到 1024。

用法::

    python3 E3/code/rf_ablation.py --folds 0 --max-wells 8 --epochs 3 --exploratory
    python3 E3/code/rf_ablation.py --folds all            # 正式（每组合全折）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_seq as TS  # noqa: E402
from src.training import metrics as M  # noqa: E402

# 4 个组合：U-Net 用 depth，TCN 用 dilation_max；两组都跑（计划要求"感受野变量"齐备）
COMBOS = [
    ("unet", {"base_ch": 64, "depth": 3, "k": 5}),
    ("unet", {"base_ch": 64, "depth": 5, "k": 5}),
    ("tcn", {"channels": 128, "n_blocks": 9, "k": 3, "dilation_max": 64}),
    ("tcn", {"channels": 128, "n_blocks": 9, "k": 3, "dilation_max": 512}),
]


def main(argv: list[str] | None = None) -> int:
    ap = TS.build_parser()
    args = ap.parse_args(argv)
    reports = Path(args.reports_dir)
    rows = []
    t0 = time.time()
    for arch, kw in COMBOS:
        tag = f"rf_{arch}_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())
                                       if k in ("depth", "dilation_max"))
        sub = [a for a in sys.argv[1:] if a not in ("--tag",)]
        rc = TS.run_arch(TS.build_parser().parse_args(
            sub + ["--arch", arch, "--arch-kwargs", json.dumps(kw), "--tag", tag]))
        met_path = reports / f"E3_metrics_{tag}.json"
        met = json.loads(met_path.read_text(encoding="utf-8")) if met_path.is_file() else {}
        rows.append({"tag": tag, "arch": arch, "kwargs": kw, "rc": int(rc),
                     "oof_total": met.get("oof_total"),
                     "n_params": met.get("model", {}).get("n_params"),
                     "seconds": met.get("seconds_total"),
                     "exploratory": bool(args.exploratory)})
        print(f"[rf] {tag}: oof={rows[-1]['oof_total']} params={rows[-1]['n_params']} "
              f"rc={rc}", flush=True)

    def _best(rows_, key_pred):
        cand = [r for r in rows_ if key_pred(r) and r["oof_total"] is not None]
        return max(cand, key=lambda r: r["oof_total"]) if cand else None

    unet_rows = [r for r in rows if r["arch"] == "unet"]
    tcn_rows = [r for r in rows if r["arch"] == "tcn"]
    depth_trend = [(r["kwargs"].get("depth"), r["oof_total"]) for r in unet_rows]
    dil_trend = [(r["kwargs"].get("dilation_max"), r["oof_total"]) for r in tcn_rows]
    saturated = None
    d64 = [r for r in tcn_rows if r["kwargs"].get("dilation_max") == 64]
    d512 = [r for r in tcn_rows if r["kwargs"].get("dilation_max") == 512]
    if d64 and d512 and d64[0]["oof_total"] is not None and d512[0]["oof_total"] is not None:
        saturated = bool(abs(d512[0]["oof_total"] - d64[0]["oof_total"]) < 0.05)
    out = {
        "stage": "E3", "kind": "receptive_field_ablation", "exploratory": bool(args.exploratory),
        "folds": args.folds, "combos": rows,
        "depth_trend": depth_trend, "dilation_trend": dil_trend,
        "best_unet": _best(rows, lambda r: r["arch"] == "unet"),
        "best_tcn": _best(rows, lambda r: r["arch"] == "tcn"),
        "dilation_saturated_at_512": saturated,
        "decision": ("512 与 64 无差异 -> 有效感受野已饱和，禁止继续加到 1024"
                     if saturated else "512 仍有增量（或数据不足）-> 保留 512，暂不上 1024"),
        "seconds_total": round(time.time() - t0, 2),
        "note": ("exploratory 运行只做资源预检，结论不进 Gate 数值"
                 if args.exploratory else "正式运行"),
    }
    TS.write_json(reports / "E3_receptive_field_ablation.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "combos"}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
