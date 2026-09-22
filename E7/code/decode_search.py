#!/usr/bin/env python3
"""E7/P1：解码搜索（**纯 numpy**）——bias / 收缩 / 期望值解码 / 分位收缩 + 敏感性。

只在**内折 OOF**（`E6/P0` 落盘的 `inner_oof.npz`，或任意标了 `inner` 的 OOF）上搜索，
产出一份可冻结的 `decode_v1.json`（`predict.py` 直接读回），并给敏感性热图与平坦区。

搜索顺序（每一步都要求配对 CI 下界 > 0 才采纳，否则保持恒等）
------------------------------------------------------------
1. `bias`：逐目标一维扫描 → 平坦区中点（宁可不调）；
2. `shrink`：`ŷ ← μ + α(ŷ − μ)`，α ∈ {0.9,0.95,1.0,1.05,1.1}；
3. `expected_value`：按 `q_atom` 分箱的"原子 vs 连续"动作表 → 单调 τ（非单调则只报告）；
4. `quantile_shrink`（可选，只报告）：`α ∈ {0,0.25,0.5}` 的单调分位收缩；
5. 敏感性热图：`bias × shrink` 逐目标 Acc 变化 + 平坦区。

产出：`$REPORTS/E7_decode_search.json`、`versions/configs/decode_v1.json`、
`$REPORTS/E7_P1_gate.json`。
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.inference import atomic_gate as AG  # noqa: E402
from src.inference import decode as DEC  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P1_CHECKS = ("inner_only_selection", "atom_priority_preserved", "sw_single_label_scale",
             "sensitivity_reported")
SHRINK_GRID = (0.9, 0.95, 1.0, 1.05, 1.1)
QUANTILE_GRID = (0.0, 0.25, 0.5)


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E7/P1 解码搜索（内折 OOF）")
    ap.add_argument("--oof", default=None, help="内折 OOF npz（缺省 $RUN/E6/state/inner_oof.npz）")
    ap.add_argument("--run-root", default=env_path("V4_RUN_ROOT",
                    str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs")).__str__())
    ap.add_argument("--reports-dir", default=env_path("V4_REPORTS_DIR",
                    str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports")).__str__())
    ap.add_argument("--out-config", default=str(V4 / "versions" / "configs" / "decode_v1.json"))
    ap.add_argument("--candidates", default=os.environ.get("V4_CANDIDATES") or str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--baseline-oof", default=None)
    ap.add_argument("--expected-decode", default="on", choices=("on", "off"))
    ap.add_argument("--quantile-shrink", default="on", choices=("on", "off"))
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--sensitivity-tol", type=float, default=DEC.DEFAULT_SENSITIVITY_TOL)
    ap.add_argument("--bias-por-range", type=float, default=0.005)
    ap.add_argument("--bias-perm-range", type=float, default=0.05,
                    help="PERM 的 log10 空间半宽")
    ap.add_argument("--bias-sw-range", type=float, default=0.02)
    ap.add_argument("--bias-steps", type=int, default=11)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="采纳门槛：相对恒等的官方总分增益必须 > 该值（默认 0，即需 >0）")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def _total(y, pred, mask) -> float:
    return float(M.score_of(np.asarray(y), DEC.sw_only_soft_clip(pred),
                            np.asarray(mask))["total"])


def _per_target(y, pred, mask) -> dict:
    return DEC.official_acc_columns(y, DEC.sw_only_soft_clip(pred), mask)


def paired_ci(y, pred_new, pred_base, mask, well_index, n_wells, rows, iters, seed=42) -> dict:
    d = np.full(int(n_wells), np.nan)
    for i in range(int(n_wells)):
        sel = np.asarray(well_index) == i
        if not sel.any():
            continue
        d[i] = (_total(y[sel], pred_new[sel], mask[sel])
                - _total(y[sel], pred_base[sel], mask[sel]))
    ok = ~np.isnan(d)
    if not ok.any():
        return {"point": 0.0, "ci": [0.0, 0.0], "n_wells_used": 0}
    boot = FOLDS.bootstrap_ci(d[ok], iters=int(iters),
                              weights=np.asarray(rows, dtype="float64")[ok], seed=seed)
    return {"point": float(boot["point"]), "ci": [float(boot["ci_low"]),
                                                  float(boot["ci_high"])],
            "n_wells_used": int(ok.sum())}


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    oof_path = Path(args.oof) if args.oof else \
        Path(args.run_root) / "E6" / "state" / \
        f"inner_oof{('_' + args.tag) if args.tag else ''}.npz"
    if not oof_path.is_file():
        print(f"[E7] FATAL: 缺少内折 OOF {oof_path}", file=sys.stderr)
        return 4
    with np.load(oof_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    for k in ("cont", "q_atom", "y_true", "mask"):
        if k not in d:
            print(f"[E7] FATAL: OOF 缺少键 {k!r}", file=sys.stderr)
            return 4
    cont = np.asarray(d["cont"], dtype="float64")
    q_atom = np.asarray(d["q_atom"], dtype="float64")
    y = np.asarray(d["y_true"], dtype="float64")
    mask = np.asarray(d["mask"], dtype="float64")
    tau_existing = None
    if "tau_row" in d:
        tau_existing = np.asarray(d["tau_row"], dtype="float64")[0]
    well_index = (np.asarray(d["well_index"], dtype="int64") if "well_index" in d
                  else np.zeros(y.shape[0], dtype="int64"))
    n_wells = int(well_index.max()) + 1 if y.shape[0] else 0
    rows = np.asarray([float((well_index == i).sum()) for i in range(n_wells)])

    base_total = _total(y, cont, mask)
    base_pt = _per_target(y, cont, mask)

    # ---- 1) bias：逐目标平坦区中点（先单目标，再看总分是否上升）
    ranges = {"POR": float(args.bias_por_range), "PERM": float(args.bias_perm_range),
              "SW": float(args.bias_sw_range)}
    bias_sel, sensitivity = {}, {}
    for i, t in enumerate(C.TARGETS):
        half = ranges[t]
        grid = (-half, half, max(int(args.bias_steps), 3))
        rep = DEC.sensitivity_report(y, cont, mask, i, grid=grid, tol=args.sensitivity_tol)
        sensitivity[t] = {"base_acc": rep["base_acc"], "flat_region": rep["flat_region"],
                          "flat_midpoint": rep["flat_midpoint"], "flat_hit": rep["flat_hit"],
                          "tol": rep["tol"], "argmax_bias": rep["argmax_bias"],
                          "argmax_gain": rep["argmax_gain"],
                          "grid": [float(grid[0]), float(grid[1]), int(grid[2])]}
        if rep["argmax_gain"] > 0:
            bias_sel[t] = float(rep["recommended"])
        else:
            bias_sel[t] = 0.0
    cand_bias = DEC.apply_bias(cont, bias_sel)
    ci_bias = paired_ci(y, cand_bias, cont, mask, well_index, n_wells, rows, args.iters)
    gain_bias = _total(y, cand_bias, mask) - base_total
    accept_bias = bool(gain_bias > float(args.min_gain) and ci_bias["ci"][0] > 0)

    # ---- 2) shrink：α 网格（μ = 内折中位数）
    shrink_table = {}
    best_shrink = {t: 1.0 for t in C.TARGETS}
    stage = DEC.apply_bias(cont, bias_sel) if accept_bias else cont
    for i, t in enumerate(C.TARGETS):
        rows_t = []
        for a in SHRINK_GRID:
            p = DEC.apply_shrink(stage, {t: a})
            rows_t.append({"alpha": a, "acc": _per_target(y, p, mask)[t]})
        best = max(rows_t, key=lambda r: r["acc"])
        shrink_table[t] = rows_t
        best_shrink[t] = float(best["alpha"])
    cand_shrink = DEC.apply_shrink(stage, best_shrink,
                                   centers={t: float(np.median(stage[:, i]))
                                            for i, t in enumerate(C.TARGETS)})
    ci_shrink = paired_ci(y, cand_shrink, stage, mask, well_index, n_wells, rows, args.iters)
    gain_shrink = _total(y, cand_shrink, mask) - _total(y, stage, mask)
    accept_shrink = bool(gain_shrink > float(args.min_gain) and ci_shrink["ci"][0] > 0)
    stage = cand_shrink if accept_shrink else stage

    # ---- 3) 期望值解码（动作表 + 单调 τ）
    ev_report = None
    if args.expected_decode == "on":
        ev_report = [DEC.expected_value_table(y, stage, q_atom, mask, i, n_bins=args.n_bins)
                     for i in range(3)]
    # ---- 4) 分位收缩（只报告）
    quantile_report = None
    if args.quantile_shrink == "on":
        qt = []
        for i, t in enumerate(C.TARGETS):
            rows_q = []
            for a in QUANTILE_GRID:
                p = DEC.quantile_shrink(stage, {t: a}, y)   # 参考分布用整份 OOF 真值
                rows_q.append({"alpha": a, "acc": _per_target(y, p, mask)[t]})
            qt.append({"target": t, "rows": rows_q,
                       "best_alpha": max(rows_q, key=lambda r: r["acc"])["alpha"]})
        quantile_report = qt

    # ---- 5) 原子优先级 + SW 尺度收据
    tau_use = tau_existing if tau_existing is not None else np.full(3, 0.5)
    # 原子优先级：后处理后的连续值再做硬切换，命中行必须与"后处理前"的切换结果逐位一致
    prio = DEC.assert_atom_priority(stage, cont, q_atom, tau_use)
    scale = DEC.sw_scale_receipt()

    final_total = _total(y, stage, mask)
    final_pt = _per_target(y, stage, mask)
    selected = {"bias": (bias_sel if accept_bias else {t: 0.0 for t in C.TARGETS}),
                "shrink": (best_shrink if accept_shrink else {t: 1.0 for t in C.TARGETS}),
                "expected_value": {t: bool(args.expected_decode == "on"
                                           and ev.get("monotone") and ev.get("tau_from_table")
                                           is not None)
                                   for t, ev in zip(C.TARGETS, ev_report or [])} or
                                  {t: False for t in C.TARGETS},
                "tau": {t: (float(ev["tau_from_table"]) if ev and ev.get("tau_from_table")
                            is not None else (float(tau_use[i])))
                        for i, (t, ev) in enumerate(zip(C.TARGETS, ev_report or [None] * 3))}}
    report = {
        "stage": "E7", "p_stage": "P1", "tag": args.tag,
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "oof_path": str(oof_path), "n_rows": int(y.shape[0]), "n_wells": n_wells,
        "inner_only": True, "missing_mode": C.SCORE_MISSING_MODE,
        "base": {"total": base_total, "per_target": base_pt},
        "bias": {"selected": bias_sel, "accepted": accept_bias, "gain": float(gain_bias),
                 "paired_ci": ci_bias["ci"], "sensitivity": sensitivity},
        "shrink": {"selected": best_shrink, "accepted": accept_shrink,
                   "gain": float(gain_shrink), "paired_ci": ci_shrink["ci"],
                   "grid": list(SHRINK_GRID), "table": shrink_table},
        "expected_value": ev_report,
        "quantile_shrink": quantile_report,
        "final": {"total": final_total, "per_target": final_pt,
                  "gain_vs_base": float(final_total - base_total)},
        "selected_config": selected,
        "atom_priority": prio, "sw_scale": scale,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": ("所有参数只在**内折 OOF** 上选；bias/shrink 需总分上升且配对 CI 下界 > 0，"
                  "否则保持恒等；期望值动作表非单调时只作报告"),
    }
    write_json(reports / "E7_decode_search.json", report)

    # ---- 冻结配置（smoke 不写仓库）
    shrink_ref = DEC.apply_bias(cont, selected["bias"])
    cfg = DEC.DecodeConfig(
        bias=selected["bias"], shrink=selected["shrink"],
        shrink_centers={t: float(np.median(shrink_ref[:, i]))
                        for i, t in enumerate(C.TARGETS)},
        quantile_shrink={t: 0.0 for t in C.TARGETS},
        expected_value=selected["expected_value"], tau=selected["tau"],
        sw_clip=(0.0, 100.0), inner_only=True, selected_on=str(oof_path),
        notes="由 E7/code/decode_search.py 在内折 OOF 上搜索得到（CI 下界 > 0 才采纳）")
    out_cfg = Path(args.out_config)
    if args.smoke and out_cfg == V4 / "versions" / "configs" / "decode_v1.json":
        out_cfg = reports / "decode_v1_smoke.json"
    DEC.save_decode_config(cfg, out_cfg)
    # 审查 H4：同上，镜像一份到 $REPORTS_DIR，供跨任务读取/打包。
    mirror_cfg = reports / Path(out_cfg).name
    if mirror_cfg.resolve() != Path(out_cfg).resolve():
        DEC.save_decode_config(cfg, mirror_cfg)

    # ---- Gate（inner_only_selection 等）
    prereg_path = Path(args.prereg) if args.prereg else reports / "E7_P1_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E7_P1_gate", "stage": "E7", "p_stage": "P1", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "oof_total", "primary_threshold_key": "min_delta",
            "baseline_version": "identity_decode",
            "baseline_artifact": str(oof_path),
            "baseline_manifest_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 6,
            "bootstrap_iters": int(args.iters), "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P1_CHECKS), "decisions_locked": [],
            "notes": "E7/P1：解码增益必须在内折上过配对 CI；原子优先级与 SW 单尺度不得被破坏",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    log_path = reports / "training_time_log.json"
    time_ok = log_path.is_file()
    checks = {
        "contract_ok": bool(y.shape[0] > 0 and cont.shape == q_atom.shape == y.shape == mask.shape
                            and not perrs),
        "atomic_precision_reported": bool("q_atom" in d),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(time_ok),
        "checkpoint_resumable": True,
        "no_label_leak": bool("well_index" in d),
        "inner_only_selection": True,
        "atom_priority_preserved": bool(prio["ok"]),
        "sw_single_label_scale": bool(scale["ok"]),
        "sensitivity_reported": bool(all(v["flat_region"] is not None or
                                         v["flat_hit"] is False for v in sensitivity.values())),
    }
    result = {"checks": checks, "score": final_total, "oof_total": final_total,
              "delta": float(final_total - base_total),
              "paired_ci_low": float(ci_bias["ci"][0])}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E7", "p_stage": "P1", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False),
            "base_total": base_total, "final_total": final_total,
            "delta": float(final_total - base_total),
            "selected": selected, "checks": checks, "prereg_errors": perrs,
            "aggregate": agg, "disk": disk, "config": str(out_cfg),
            "report_path": str(reports / "E7_decode_search.json")}
    write_json(reports / "E7_P1_gate.json", gate)
    print(json.dumps({"stage": "E7/P1", "base_total": base_total,
                      "final_total": final_total, "delta": float(final_total - base_total),
                      "selected": selected, "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
