#!/usr/bin/env python3
"""根据 E2 消融报告自动选择下游最优特征版本。

规则
----
- 候选来自 `$V4_REPORTS_DIR/E2_ablation.json`：
  - `baseline`（默认 F1）
  - `groups[]` 中每个特征组消融（decision / oof_total / paired_ci）
- 只在 `decision == "adopted"` 且配对 CI 下界 > 0 的候选中选 `oof_total` 最大者；
- 若没有 adopted 候选，回退 F1；
- 输出 `$V4_REPORTS_DIR/E2_best_spec.json`，并可用 `--print` / `--print-key`
  给 `run_train.sh` 直接消费。

用法::

    python3 tools/select_feature_spec.py --reports-dir /data/v4/reports
    python3 tools/select_feature_spec.py --reports-dir /data/v4/reports --print
    python3 tools/select_feature_spec.py --reports-dir /data/v4/reports --print-key
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ORDER = ("F1", "phys", "win", "well")


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


def _spec_name(groups) -> str:
    gs = [str(g) for g in (groups or ["F1"])]
    if not gs or gs == ["F1"]:
        return "F1"
    if set(gs) == set(ORDER):
        return "F2"
    ordered = [g for g in ORDER if g in gs]
    return "F1+" + "+".join(g for g in ordered if g != "F1")


def _num(x):
    return float(x) if isinstance(x, (int, float)) else None


def choose(ablation: dict) -> dict:
    baseline = ablation.get("baseline") or {}
    base = {
        "spec_name": _spec_name(["F1"]),
        "spec_key": str(baseline.get("spec_key") or "F1"),
        "groups": ["F1"],
        "oof_total": _num(baseline.get("oof_total")),
        "delta_vs_f1": 0.0,
        "paired_ci": None,
        "decision": "baseline",
        "source": "baseline",
    }
    candidates = [base]
    for cand in (ablation.get("groups") or []):
        groups = cand.get("groups") or ["F1"]
        ci = cand.get("paired_ci")
        candidates.append({
            "spec_name": _spec_name(groups),
            "spec_key": str(cand.get("spec_key") or ""),
            "groups": [str(g) for g in groups],
            "oof_total": _num(cand.get("oof_total")),
            "delta_vs_f1": _num(cand.get("delta_vs_f1")),
            "paired_ci": ([_num(ci[0]), _num(ci[1])] if isinstance(ci, (list, tuple)) and len(ci) >= 2 else None),
            "decision": str(cand.get("decision") or "no_go"),
            "source": "groups",
        })

    adopted = [
        c for c in candidates
        if c["decision"] == "adopted"
        and c["oof_total"] is not None
        and c["paired_ci"] is not None
        and c["paired_ci"][0] is not None
        and float(c["paired_ci"][0]) > 0.0
    ]
    if adopted:
        selected = max(adopted, key=lambda c: float(c["oof_total"]))
        reason = "E2 adopted 中 OOF 最高且配对 CI 下界 > 0"
    else:
        selected = base
        reason = "无 adopted 候选，回退 F1"
    return {
        "selected_spec": selected["spec_name"],
        "selected_key": selected["spec_key"],
        "selected_groups": selected["groups"],
        "selected_oof_total": selected["oof_total"],
        "selected_delta_vs_f1": selected.get("delta_vs_f1"),
        "selected_paired_ci": selected.get("paired_ci"),
        "baseline_spec": base["spec_name"],
        "baseline_key": base["spec_key"],
        "baseline_oof_total": base["oof_total"],
        "reason": reason,
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 E2_ablation.json 自动选择下游特征版本")
    ap.add_argument("--reports-dir", default=str(_env_path("V4_REPORTS_DIR", "/data/v4/reports")))
    ap.add_argument("--out", default=None, help="输出 JSON；缺省 <reports>/E2_best_spec.json")
    ap.add_argument("--print", action="store_true", help="只向 stdout 打印 selected_spec")
    ap.add_argument("--print-key", action="store_true", help="只向 stdout 打印 selected_key")
    ap.add_argument("--no-write", action="store_true", help="不写 E2_best_spec.json")
    args = ap.parse_args(argv)

    reports = Path(args.reports_dir).expanduser().resolve()
    src = reports / "E2_ablation.json"
    if src.is_file():
        try:
            ablation = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: E2_ablation.json 读取失败: {exc!r}", file=sys.stderr)
            ablation = {}
    else:
        print(f"WARN: 缺少 {src}，回退 F1", file=sys.stderr)
        ablation = {}

    out = choose(ablation)
    out.update({
        "source": str(src),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
    })
    out_path = Path(args.out).expanduser().resolve() if args.out else reports / "E2_best_spec.json"
    if not args.no_write:
        # --print/--print-key 也顺手写盘，保证 E2_best_spec.json 可被 artifact_store 持久化。
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out_path)

    if args.print:
        print(out["selected_spec"])
        return 0
    if args.print_key:
        print(out["selected_key"])
        return 0

    print(json.dumps({k: out[k] for k in (
        "selected_spec", "selected_key", "selected_oof_total",
        "selected_delta_vs_f1", "selected_paired_ci", "reason", "source")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
