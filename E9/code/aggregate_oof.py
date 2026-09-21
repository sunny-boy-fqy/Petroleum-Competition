#!/usr/bin/env python3
"""E9/P0：候选 OOF 汇总 + 护栏（guardrail）评估（**纯 numpy**）。

护栏下限（E9/P0 §9.4，用的是 `constants.py` 里的**冻结锚点**）
------------------------------------------------------------
    floor = max(PASS_LINE,                       # 75.0   官方及格线
                B0_LOCAL_OOF  − GUARDRAIL_TOLERANCE,        # 80.382479 − 1.0
                B0_A_BOARD    − GUARDRAIL_ABOARD_MARGIN)    # 82.2757   − 0.5  = 81.7757

候选必须 `oof_total ≥ floor`（且若已知 A 榜分，还要 `a_board ≥ B0_A_BOARD − margin`）才算
通过护栏；**A 榜分未知时不假装通过**，只评估本地 OOF 那一半并显式标注。

本脚本只**读**候选表：`status` 的更新（shortlisted/rejected）由
`E9/code/choose_submission.py` 负责（单点写入，避免两处互相覆盖）。

产出：`$REPORTS/E9_validation_report.json`、`$REPORTS/E9_P0_gate.json`。
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

from src import constants as C  # noqa: E402
from src.validation import evidence as EVID  # noqa: E402
from src.score import score_arrays  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P0_CHECKS = ("guardrail_evaluated", "oof_recomputable", "protocol_matched_reported",
             "candidates_have_provenance")
PROTOCOL_NOTE = ("B0 的本地 OOF（80.382479）与 v4 的折协议**不完全一致**"
                 "（protocol_matched=false），因此只能当护栏锚点，不能当同协议对照")


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


def guardrail_floor() -> dict[str, float]:
    """护栏下限与它的三个来源（全部来自 `constants.py`，禁止在脚本里硬编码数字）。"""
    parts = {"pass_line": float(C.PASS_LINE),
             "b0_local_oof_minus_tol": float(C.B0_LOCAL_OOF - C.GUARDRAIL_TOLERANCE),
             "b0_a_board_minus_margin": float(C.B0_A_BOARD - C.GUARDRAIL_ABOARD_MARGIN)}
    return {**parts, "floor": float(max(parts.values()))}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E9/P0 候选汇总 + 护栏评估")
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--candidates", default=os.environ.get("V4_CANDIDATES") or str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--folds", default=None, help="折文件（缺省用冻结折）")
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-shortlist", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def load_oof(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    for key in ("y_true", "mask"):
        if key not in d:
            return None
    return d


def candidate_oof_path(cand: dict, run_root: Path) -> Path | None:
    """候选的 OOF 位置：显式 `oof_path` 优先，其次按 stage 猜 `$RUN/<stage>/oof.npz`。"""
    for key in ("oof_path", "oof"):
        if cand.get(key):
            p = Path(cand[key])
            return p if p.is_absolute() else V4 / p
    stage = str(cand.get("stage", "")).upper()
    if stage:
        return run_root / stage.capitalize() / "oof.npz"
    return None


def metrics_of(oof: dict) -> dict:
    y = np.asarray(oof["y_true"], dtype="float64")
    mask = np.asarray(oof["mask"], dtype="float64")
    # M1 审查修复：E1/E3 的 OOF 键是 y_pred（已门控）与 cont（未门控），
    # 不存在 gated 键；取列顺序必须是 gated → y_pred → cont → pred，否则复核分会
    # 丢掉原子门结果、系统性低估。
    pred = np.asarray(oof.get("gated", oof.get("y_pred", oof.get("cont",
                               oof.get("pred")))), dtype="float64")
    if pred is None or pred.shape != y.shape:
        raise ValueError("OOF 里没有可用的预测（gated/cont/pred）")
    s = score_arrays(y, pred, missing=~mask.astype(bool),
                     missing_mode=C.SCORE_MISSING_MODE)
    out = {"total": float(s["total"]), "por": float(s["acc_por"]),
           "perm": float(s["acc_perm"]), "sw": float(s["acc_sw"]),
           "missing_mode": C.SCORE_MISSING_MODE, "n_rows": int(y.shape[0])}
    if "well_index" in oof and "fold_of_row" in oof:
        widx = np.asarray(oof["well_index"], dtype="int64")
        folds = np.asarray(oof["fold_of_row"], dtype="int64")
        n_wells = int(widx.max()) + 1 if y.shape[0] else 0
        per_fold, per_well = {}, np.full(n_wells, np.nan)
        for f in sorted(set(int(v) for v in folds)):
            sel = folds == f
            per_fold[str(f)] = float(score_arrays(y[sel], pred[sel],
                                                  missing=~mask[sel].astype(bool),
                                                  missing_mode=C.SCORE_MISSING_MODE)["total"])
        const = np.tile([C.ATOM_VALUES[t] for t in C.TARGETS], (y.shape[0], 1))
        c_tot = np.full(n_wells, np.nan)
        for w in range(n_wells):
            sel = widx == w
            if not sel.any():
                continue
            per_well[w] = float(score_arrays(y[sel], pred[sel], missing=~mask[sel].astype(bool),
                                             missing_mode=C.SCORE_MISSING_MODE)["total"])
            c_tot[w] = float(score_arrays(y[sel], const[sel],
                                          missing=~mask[sel].astype(bool),
                                          missing_mode=C.SCORE_MISSING_MODE)["total"])
        rows = np.asarray([float((widx == w).sum()) for w in range(n_wells)])
        ok = ~np.isnan(per_well) & ~np.isnan(c_tot)
        out["per_fold"] = per_fold
        out["const_total"] = float(score_arrays(y, const, missing=~mask.astype(bool),
                                                missing_mode=C.SCORE_MISSING_MODE)["total"])
        if ok.any():
            boot = FOLDS.bootstrap_ci((per_well - c_tot)[ok], iters=1000,
                                      weights=rows[ok], seed=42)
            out["paired_ci_vs_const"] = [float(boot["ci_low"]), float(boot["ci_high"])]
            out["delta_vs_const"] = float(boot["point"])
    return out


def ranking_consistency(cands: list[dict]) -> dict | None:
    """本地 OOF 排名 vs A 榜排名的 Spearman（都已知时才有意义）。"""
    pairs = [(c["candidate_id"], c["oof_total"], c["a_board_score"])
             for c in cands if c.get("oof_total") is not None
             and c.get("a_board_score") is not None]
    if len(pairs) < 3:
        return None
    a = np.asarray([p[1] for p in pairs], dtype="float64")
    b = np.asarray([p[2] for p in pairs], dtype="float64")
    ra = np.argsort(np.argsort(a)).astype("float64")
    rb = np.argsort(np.argsort(b)).astype("float64")
    return {"n": len(pairs), "spearman": float(np.corrcoef(ra, rb)[0, 1]),
            "candidates": [p[0] for p in pairs]}


def run(args) -> int:
    reports = Path(args.reports_dir)
    run_root = Path(args.run_root)
    reports.mkdir(parents=True, exist_ok=True)
    raw = REG.load_candidates(args.candidates)
    cands = list(raw.get("candidates", []))
    floor = guardrail_floor()
    a_board_floor = float(C.B0_A_BOARD - C.GUARDRAIL_ABOARD_MARGIN)

    out_rows, missing = [], []
    for c in cands:
        cid = str(c.get("candidate_id"))
        p = candidate_oof_path(c, run_root)
        oof = load_oof(p) if p is not None else None
        row = {"candidate_id": cid, "stage": c.get("stage"), "status": c.get("status"),
               "oof_path": (None if p is None else str(p)),
               "source": c.get("checkpoint") or c.get("checkpoints"),
               "protocol_matched": bool(c.get("protocol_matched", False)),
               "exploratory": bool(c.get("exploratory", False))}
        if oof is None:
            row.update({"oof_total": None, "status_detail": "oof_missing",
                        "guardrail_pass": False,
                        "guardrail_reason": f"缺少可复算的 OOF（{p}）"})
            missing.append(cid)
        else:
            m = metrics_of(oof)
            row.update(m)
            row["oof_total"] = float(m["total"])       # 报告口径键名（排名/短名单都用它）
            a_board = c.get("a_board_score")
            ok_local = bool(m["total"] >= floor["floor"])
            ok_aboard = (None if a_board is None
                         else bool(float(a_board) >= a_board_floor))
            row.update({"a_board_score": a_board, "a_board_pass": ok_aboard,
                        "guardrail_pass": bool(ok_local and ok_aboard is not False),
                        "guardrail_reason": (
                            f"OOF {m['total']:.4f} {'≥' if ok_local else '<'} floor "
                            f"{floor['floor']:.4f}"
                            + ("" if a_board is None else
                               f"；A 榜 {float(a_board):.4f} "
                               f"{'≥' if ok_aboard else '<'} {a_board_floor:.4f}")
                            + ("" if a_board is not None else
                               "；A 榜未知：只评估本地 OOF 那一半（如实标注）"))})
        out_rows.append(row)

    ranked = sorted([r for r in out_rows if r.get("oof_total") is not None],
                    key=lambda r: float(r["oof_total"]), reverse=True)
    passing = [r for r in ranked if r["guardrail_pass"]]
    shortlist = [r["candidate_id"] for r in passing[:max(int(args.max_shortlist), 0)]]
    report = {"stage": "E9", "p_stage": "P0", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "missing_mode": C.SCORE_MISSING_MODE, "protocol_matched": False,
              "protocol_note": PROTOCOL_NOTE,
              "guardrail": {**floor, "a_board_floor": a_board_floor,
                            "anchors": {"B0_LOCAL_OOF": C.B0_LOCAL_OOF,
                                        "B0_A_BOARD": C.B0_A_BOARD,
                                        "PASS_LINE": C.PASS_LINE,
                                        "GUARDRAIL_TOLERANCE": C.GUARDRAIL_TOLERANCE,
                                        "GUARDRAIL_ABOARD_MARGIN": C.GUARDRAIL_ABOARD_MARGIN}},
              "candidates": out_rows, "n_candidates": len(out_rows),
              "ranked_by_oof": [r["candidate_id"] for r in ranked],
              "shortlist": shortlist,
              "candidates_missing_oof": missing,
              "ranking_consistency_local_vs_aboard": ranking_consistency(out_rows)}
    out_path = Path(args.json) if args.json else reports / "E9_validation_report.json"
    write_json(out_path, report)

    prereg_path = Path(args.prereg) if args.prereg else reports / "E9_P0_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E9_P0_gate", "stage": "E9", "p_stage": "P0", "gate_type": "boolean",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "guardrail_pass", "primary_threshold_key": "min_delta",
            "baseline_version": "B0", "baseline_artifact": str(out_path),
            "baseline_manifest_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": int(args.max_shortlist),
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P0_CHECKS), "decisions_locked": [],
            "notes": ("E9/P0：候选必须 oof_total ≥ max(75, B0_local−1, B0_aboard−0.5)="
                      f"{floor['floor']:.4f}；A 榜未知只评估本地那一半"),
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    atomic_ok, atomic_ev = EVID.atomic_precision_reported(reports)
    resume_ok, resume_ev = EVID.checkpoint_resumable(reports)
    time_ok, time_ev = EVID.training_time_log_valid(reports)
    leak_ok, leak_ev = EVID.leakage_audit_ok(reports)
    checks = {
        "contract_ok": bool(out_rows and not perrs),
        "atomic_precision_reported": bool(atomic_ok),
        "disk_budget_ok": EVID.disk_budget_ok(disk.get("level")),
        "training_time_log_valid": bool(time_ok),
        "checkpoint_resumable": bool(resume_ok),
        "no_label_leak": bool(leak_ok),
        "guardrail_evaluated": bool(out_rows) and all("guardrail_pass" in r for r in out_rows),
        "oof_recomputable": bool(ranked) and all(r.get("total") is not None for r in ranked),
        "protocol_matched_reported": bool(out_rows) and all(
            isinstance(r.get("protocol_matched"), bool) for r in out_rows),
        "candidates_have_provenance": bool(all(r.get("stage") for r in out_rows)) if out_rows
        else False,
    }
    result = {"checks": checks, "guardrail_pass": bool(passing)}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E9", "p_stage": "P0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "floor": floor["floor"],
            "n_candidates": len(out_rows), "n_passing": len(passing),
            "shortlist": shortlist, "missing_oof": missing,
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "evidence": {"atomic": atomic_ev, "resumable": resume_ev,
                         "time_log": time_ev, "leakage": leak_ev},
            "report_path": str(out_path)}
    write_json(reports / "E9_P0_gate.json", gate)
    print(json.dumps({"stage": "E9/P0", "floor": floor["floor"],
                      "n_candidates": len(out_rows), "n_passing": len(passing),
                      "shortlist": shortlist, "missing_oof": missing,
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
