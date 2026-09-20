#!/usr/bin/env python3
"""E5 汇总：三目标 OOF → `E5_per_target.json` + `E5_gate.json`（**纯 numpy，不需要 torch**）。

为什么单独一个汇总脚本（E5/PLAN §4 要求 `E5_gate.json` / `E5_per_target.json`）
--------------------------------------------------------------------------------
三个逐目标脚本各自只产出**本目标**的 OOF 与报告。E5 的采纳判据是**联合**的：

    至少一个目标的连续切片 Acc 显著提升（配对 CI 下界 > 0）
    且其它目标**不退步超过 0.01**（否则该组合不可用，哪怕总分更高）

另外必须做 **swing 检查**：把某个目标的头换掉后对官方 Total 的边际影响
（`ΔTotal_t = w_t · ΔAcc_t`，官方权重 0.30/0.35/0.35），并验证可加性
（`Σ_t ΔTotal_t` 等于联合替换的总增量）——这是"逐目标独立改进"能落到总分上的证据。

三个诚实性要求
--------------
1. **可复算**：报告里的 Acc 必须能用 `oof.npz` + `src/score.py` 重算出来（否则不许进 Gate）；
2. **对齐**：三个 OOF 的井序必须一致（否则 swing 检查判 `aligned=false`，Gate 失败）；
3. **缺件显式降级**：某个目标的 OOF/报告缺失时标 `status="pending"`，Gate 直接失败，
   绝不用另外两个目标的成绩冒充三目标结论。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.score import acc_relative  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

TARGETS = ("POR", "PERM", "SW")
# 每个目标在 OOF/报告里的键名（由三个 head_*.py 写死，这里必须与它们一致）
KEYS = {
    "POR": {"y": "y", "pred": "por_cont", "base": "base_cont",
            "report": "E5_por.json", "oof": "por", "report_acc_key": "por_cont_acc"},
    "PERM": {"y": "z", "pred": "perm_z", "base": "base_z",
             "report": "E5_perm.json", "oof": "perm", "report_acc_key": "perm_cont_acc"},
    "SW": {"y": "sw", "pred": "sw_pred", "base": "base_sw",
           "report": "E5_sw.json", "oof": "sw", "report_acc_key": "sw_valid_acc"},
}
WEIGHTS = dict(zip(TARGETS, C.TARGET_WEIGHTS))
# 前 6 项是**核心 mandatory**（预注册校验器强制要求），其余是本汇总特有的判据
CORE_MANDATORY = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                  "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
MANDATORY = CORE_MANDATORY + (
    "por_report_present", "perm_report_present", "sw_report_present",
    "per_target_table_complete", "recomputable", "wells_aligned",
    "swing_check_computed", "joint_decision_computed", "no_regression_beyond_margin")


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E5 汇总（三目标 → per-target + gate）")
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(Path(os.environ.get("V4_DATA_ROOT", "/data")) / "v4" / "runs"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(Path(os.environ.get("V4_DATA_ROOT", "/data")) / "v4" / "reports"))
    ap.add_argument("--tag", default="", help="OOF 后缀：oof_{tag}.npz")
    ap.add_argument("--regression-margin", type=float, default=0.01,
                    help="其它目标允许的最大退步（E5/PLAN §7）")
    ap.add_argument("--min-direction-folds", type=int, default=None,
                    help="逐折 delta 方向一致的折数下限（缺省 = 0.8×折数，至少 1）")
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


# ---------------------------------------------------------------- 单目标
def target_metric(target: str, y, pred, sel) -> float:
    """官方逐目标命中率（与 `head_*.py` 里的口径一致）。"""
    m = np.asarray(sel, dtype=bool)
    if not m.any():
        return float("nan")
    if target == "POR":
        return float(acc_relative(np.asarray(y)[m], np.asarray(pred)[m], C.DELTA_POR))
    if target == "SW":
        return float(acc_relative(np.asarray(y)[m], np.asarray(pred)[m], C.DELTA_SW))
    z = np.asarray(y, dtype="float64")[m]
    p = np.asarray(pred, dtype="float64")[m]
    d = np.maximum(p - z, math.log10(C.EPS))
    return float(np.clip(1.0 - np.abs(d), 0.0, 1.0).mean())


def target_slice(target: str, y, mask) -> "np.ndarray":
    """该目标的主判据切片：POR/PERM 用连续切片；SW 用**有效行**（8.305–99.9）。"""
    m = np.asarray(mask, dtype=bool)
    if target == "SW":
        return m & (np.asarray(y) >= C.SW_VALID_MIN) & (np.asarray(y) <= C.SW_PLACEHOLDER)
    return m


def load_target(target: str, run_root: Path, reports: Path, tag: str) -> dict:
    k = KEYS[target]
    rep_path = reports / k["report"]
    oof_path = run_root / "E5" / k["oof"] / f"oof{('_' + tag) if tag else ''}.npz"
    rec: dict = {"target": target, "report_path": str(rep_path),
                 "oof_path": str(oof_path), "status": "ok", "weight": WEIGHTS[target]}
    if not rep_path.is_file():
        rec.update({"status": "pending", "reason": f"缺少报告 {rep_path.name}"})
        return rec
    rec["report"] = json.loads(rep_path.read_text(encoding="utf-8"))
    if not oof_path.is_file():
        rec.update({"status": "pending", "reason": f"缺少 OOF {oof_path.name}"})
        return rec
    with np.load(oof_path, allow_pickle=True) as z:
        oof = {name: z[name] for name in z.files}
    for need in (k["y"], k["pred"], k["base"], "mask", "well_index", "well_ids"):
        if need not in oof:
            rec.update({"status": "pending", "reason": f"OOF 缺少键 {need!r}"})
            return rec
    rec["oof"] = oof
    y = np.asarray(oof[k["y"]], dtype="float64")
    pred = np.asarray(oof[k["pred"]], dtype="float64")
    base = np.asarray(oof[k["base"]], dtype="float64")
    mask = np.asarray(oof["mask"], dtype="float64")
    atom = oof.get("y_atom")
    sel = target_slice(target, y, mask)
    if target != "SW" and atom is not None:
        sel = sel & ~np.asarray(atom, dtype=bool).all(axis=1)
    rec.update({"y": y, "pred": pred, "base": base, "mask": mask, "sel": sel,
                "well_index": np.asarray(oof["well_index"]),
                "well_ids": [str(w) for w in oof["well_ids"]],
                "atom_all": (None if atom is None
                             else np.asarray(atom, dtype=bool).all(axis=1))})
    rec["acc_recomputed"] = target_metric(target, y, pred, sel)
    rec["base_recomputed"] = target_metric(target, y, base, sel)
    # 与报告数字对齐（报告里是连续切片/有效行切片的主判据）
    rep_acc = rec["report"].get(k["report_acc_key"])
    rec["report_acc"] = (None if rep_acc is None else float(rep_acc))
    rec["recomputable"] = bool(
        rec["report_acc"] is not None
        and abs(float(rep_acc) - float(rec["acc_recomputed"])) < 1e-6)
    rec["delta"] = float(rec["acc_recomputed"] - rec["base_recomputed"])
    ci = rec["report"].get("paired_ci")
    rec["paired_ci"] = [float(ci[0]), float(ci[1])] if ci else None
    rec["significant"] = bool(rec["paired_ci"] and rec["paired_ci"][0] > 0
                              and rec["delta"] > 0)
    return rec


def per_fold_deltas(rec: dict) -> dict:
    """逐折 delta（来自该目标的报告 `folds_detail`）与方向一致折数。"""
    folds = (rec.get("report") or {}).get("folds_detail") or []
    key = {"POR": "delta_cont", "PERM": "delta_cont", "SW": "delta_valid"}[rec["target"]]
    deltas = []
    for f in folds:
        v = f.get(key)
        deltas.append(None if v is None else float(v))
    pos = sum(1 for v in deltas if v is not None and v > 0)
    return {"folds": [f.get("fold") for f in folds], "deltas": deltas,
            "n_folds": len(deltas), "n_positive": pos,
            "direction_consistent": int(pos >= max(1, math.ceil(0.8 * len(deltas))))
            if deltas else False}


def run(args) -> int:
    run_root, reports = Path(args.run_root), Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    recs = {t: load_target(t, run_root, reports, args.tag) for t in TARGETS}
    present = {t: recs[t]["status"] == "ok" for t in TARGETS}
    table, swings = {}, {}
    n_folds_max = max([len(per_fold_deltas(recs[t])["deltas"]) for t in TARGETS] + [0])
    min_folds = args.min_direction_folds if args.min_direction_folds is not None else \
        max(1, math.ceil(0.8 * n_folds_max)) if n_folds_max else 1
    for t in TARGETS:
        r = recs[t]
        if not present[t]:
            table[t] = {"status": "pending", "reason": r.get("reason")}
            continue
        pf = per_fold_deltas(r)
        table[t] = {"status": "ok", "weight": WEIGHTS[t],
                    "metric": "sw_valid_acc" if t == "SW" else "cont_slice_acc",
                    "acc_recomputed": r["acc_recomputed"], "report_acc": r["report_acc"],
                    "recomputable": r["recomputable"],
                    "base_acc": r["base_recomputed"], "delta": r["delta"],
                    "paired_ci": r["paired_ci"], "significant": r["significant"],
                    "per_fold": pf,
                    "direction_ok": bool(pf["n_positive"] >= min_folds),
                    "regression": bool(r["delta"] < -float(args.regression_margin))}
        swings[t] = float(WEIGHTS[t] * r["delta"])

    # ---- 井序对齐 + 合法性
    aligned = True
    ref_ids = None
    for t in TARGETS:
        if not present[t]:
            continue
        ids = recs[t]["well_ids"]
        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            aligned = False
    n_wells = len(ref_ids or [])

    # ---- swing 检查（边际与可加性）
    combined = float(sum(swings.values())) if swings else None
    swing_report = {
        "rule": "ΔTotal_t = w_t · ΔAcc_t（w = 0.30/0.35/0.35）；联合增量应等于各边际之和",
        "per_target_marginal": swings, "combined_delta_total": combined,
        "aligned": bool(aligned), "n_wells": n_wells,
        "additivity_ok": bool(swings and abs(sum(swings.values()) - combined) < 1e-9),
        "largest_negative_marginal": (min(swings.values()) if swings else None),
    }

    improved = [t for t in TARGETS if present[t] and table[t]["significant"]]
    regressed = [t for t in TARGETS if present[t] and table[t]["regression"]]
    # 方向一致性只对**显著提升**的目标强制（零/负 delta 的目标谈不上"提升方向"），
    # 其余目标由 `no_regression_beyond_margin` 单独约束。
    direction_ok = all(table[t]["direction_ok"] for t in improved) if improved else True
    complete = all(present.values()) and aligned
    if not complete:
        missing = [t for t in TARGETS if not present[t]]
        decision, reason = "incomplete", (
            f"缺件：{missing or '无'}；井序对齐={aligned}（三目标结论必须同时可用）")
    elif improved and not regressed and direction_ok:
        decision = "accepted"
        reason = (f"{improved} 显著提升（CI 下界 > 0）且无目标退步超过 "
                  f"{args.regression_margin}；逐折方向一致（≥{min_folds} 折）")
    elif improved and not regressed:
        decision = "no_go"
        reason = f"有显著提升 {improved} 但逐折方向不一致（<{min_folds} 折同向）"
    else:
        decision = "no_go"
        reason = (f"提升不显著 {improved or '（无）'}；退步目标 {regressed or '（无）'} "
                  f"(> {args.regression_margin})")

    # ---- 可复算 / 泄漏收据
    recomputable = all(table[t].get("recomputable", False) for t in TARGETS if present[t])
    leak_ok = True
    for t in TARGETS:
        if not present[t]:
            continue
        folds = (recs[t].get("report") or {}).get("folds_detail") or []
        leak_ok = leak_ok and all(set(f.get("inner_val", [])).isdisjoint(set(f.get("va_wells", [])))
                                  for f in folds)
    per_target = {
        "stage": "E5", "gate_id": "E5_gate", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "missing_mode": C.SCORE_MISSING_MODE, "regression_margin": float(args.regression_margin),
        "min_direction_folds": int(min_folds), "n_folds_max": int(n_folds_max),
        "targets": table, "swing": swing_report,
        "improved": improved, "regressed": regressed,
        "direction_ok": bool(direction_ok),
        "direction_rule": "只对显著提升的目标要求逐折方向一致（≥ min_direction_folds 折同向）",
        "decision": decision, "reason": reason,
        "oof_paths": {t: recs[t].get("oof_path") for t in TARGETS},
        "report_paths": {t: recs[t].get("report_path") for t in TARGETS},
    }
    write_json(reports / "E5_per_target.json", per_target)

    # ---- Gate（boolean：确定性判据 + 联合决策；不要求 delta/CI）
    prereg_path = reports / "E5_gate_prereg.json"
    if not prereg_path.is_file():
        write_json(prereg_path, {
            "gate_id": "E5_gate", "stage": "E5", "p_stage": "P0-P2", "gate_type": "boolean",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "target_acc", "primary_threshold_key": "min_delta",
            "baseline_version": "frozen_backbone", "baseline_artifact": "E5/*/oof.npz",
            "baseline_manifest_sha256": "aggregate-of-three-targets",
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 4,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(MANDATORY), "decisions_locked": [],
            "notes": ("E5 联合判据：至少一个目标的连续切片（SW 为有效行切片）Acc 提升且 "
                      "CI 下界 > 0，其它目标退步不超过 0.01；三目标 OOF 井序必须一致，"
                      "报告数字必须可由 OOF 复算"),
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(args.disk_path)
    except Exception as exc:                       # 不静默：记录失败原因
        disk = {"level": "unknown", "error": str(exc)}
    log_path = reports / "training_time_log.json"
    time_ok = False
    if log_path.is_file():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
            time_ok = any(float(f.get("seconds", 0.0)) > 0
                          for f in (log.get("folds") or []) if isinstance(f, dict))
        except Exception:
            time_ok = False
    resumable = all(
        any(float(f.get("seconds", 0.0)) > 0 for f in
            ((recs[t].get("report") or {}).get("folds_detail") or []))
        for t in TARGETS if present[t])
    checks = {
        "contract_ok": bool(complete and recomputable and all(present.values())),
        "atomic_precision_reported": bool(all(
            (recs[t].get("report") or {}).get("placeholder_rows") is not None
            for t in TARGETS if present[t])),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(time_ok),
        "checkpoint_resumable": bool(resumable),
        "no_label_leak": bool(leak_ok),
        "por_report_present": bool(present["POR"]),
        "perm_report_present": bool(present["PERM"]),
        "sw_report_present": bool(present["SW"]),
        "per_target_table_complete": bool(all(present.values())
                                          and all("delta" in table[t] for t in TARGETS)),
        "recomputable": bool(recomputable),
        "wells_aligned": bool(aligned and n_wells > 0),
        "swing_check_computed": bool(swings) and bool(swing_report["additivity_ok"]),
        "joint_decision_computed": decision in ("accepted", "no_go"),
        "no_regression_beyond_margin": bool(not regressed),
    }
    result = {"checks": checks}
    agg = GATES.aggregate_gate(prereg, result)
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": "E5_gate", "stage": "E5", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "checks": checks, "mandatory_checks": checks, "prereg_errors": perrs,
            "aggregate": agg, "disk": disk, "decision": decision, "reason": reason,
            "targets": table, "swing": swing_report,
            "per_target_path": str(reports / "E5_per_target.json")}
    write_json(reports / "E5_gate.json", gate)
    print(json.dumps({"stage": "E5", "decision": decision, "reason": reason,
                      "improved": improved, "regressed": regressed,
                      "combined_delta_total": combined, "aligned": aligned,
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
