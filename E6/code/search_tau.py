#!/usr/bin/env python3
"""E6/P1：τ 搜索（**纯 numpy，不需要 torch**）——逐目标硬切换阈值的确定与证据。

为什么单独一步
-------------
τ 是"连续头 ↔ 原子值"的开关，直接决定 POR/PERM/SW 三个目标的命中率（尤其占位行）。
本脚本只做四件事，且**全部只用内折 OOF**（`E6/code/train_state.py` 落盘的
`inner_oof.npz`）：

1. `AG.select_tau_per_target`：91 点网格 `[0.05, 0.95]` 步长 0.01，官方逐目标口径，
   **最长平台中点**规则（抗平台抖动）；
2. 逐目标原子指标：Acc / Precision / Recall / F1（τ* 与 τ=0.5 两档）+ 连续切片 Acc；
3. 误判代价分解：非原子行被误切 / 原子行被漏切 + 分数增量；
4. `joint_guard`（默认关）：打开后必须**总分上升且配对 CI 下界 > 0** 才采纳，否则 NO-GO。

产出：`$REPORTS/E6_tau_search.json`、`$REPORTS/E6_P1_gate.json`，
并把 τ* 写回 `versions/candidates.json::PD1.atomic`（`--smoke` 不登记，绝不污染仓库）。
"""
from __future__ import annotations

import argparse
import hashlib
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
from src.features import basic as FB  # noqa: E402
from src.inference import atomic_gate as AG  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import state_train as ST  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
E6_CHECKS = ("per_target_atom_acc_reported", "per_target_atom_precision_recall_f1_reported",
             "joint_atom_auc_reported", "tau_t_inner_oof_only",
             "no_atom_continuous_interpolation", "input_no_label_leak_full")


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def default_run_root() -> Path:
    return env_path("V4_RUN_ROOT", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))


def default_reports_dir() -> Path:
    return env_path("V4_REPORTS_DIR", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E6/P1 τ 搜索（内折 OOF）")
    ap.add_argument("--oof", default=None,
                    help="内折 OOF npz（缺省：$RUN/E6/state/inner_oof.npz）")
    ap.add_argument("--run-root", default=str(default_run_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--atomic-report", default=None,
                    help="E6/P0 报告（缺省 $REPORTS/E6_atomic_report.json，用于 P1 Gate 的共享 checks）")
    ap.add_argument("--candidates", default=str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--baseline-oof", default=None, help="对照 OOF（可选，给配对 CI）")
    ap.add_argument("--tol", type=float, default=AG.DEFAULT_PLATEAU_TOL)
    ap.add_argument("--joint-guard", action="store_true", help="启用联合守卫（默认关）")
    ap.add_argument("--tau-joint-high", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


def _tau_curve(sel: dict) -> dict:
    out = {}
    for t, pairs in (sel.get("curve") or {}).items():
        out[t] = [[float(a), float(b)] for a, b in pairs]
    return out


def _well_weights(well_index, n_wells) -> "np.ndarray":
    return np.asarray([float((well_index == i).sum()) for i in range(int(n_wells))],
                      dtype="float64")


def _total_of(cont, q_atom, tau, y, mask) -> float:
    gated = AG.per_target_hard_switch(np.asarray(cont, dtype="float64"),
                                      np.asarray(q_atom, dtype="float64"), tau)
    return float(M.score_of(np.asarray(y), gated, np.asarray(mask))["total"])


def run(args) -> int:
    reports = Path(args.reports_dir)
    run_root = Path(args.run_root)
    reports.mkdir(parents=True, exist_ok=True)
    oof_path = Path(args.oof) if args.oof else \
        run_root / "E6" / "state" / f"inner_oof{('_' + args.tag) if args.tag else ''}.npz"
    if not oof_path.is_file():
        print(f"[E6] FATAL: 缺少内折 OOF {oof_path}（先跑 E6/code/train_state.py）",
              file=sys.stderr)
        return 4
    with np.load(oof_path, allow_pickle=True) as z:
        z = {k: z[k] for k in z.files}
    for key in ("cont", "q_atom", "y_true", "mask"):
        if key not in z:
            print(f"[E6] FATAL: OOF 缺少键 {key!r}", file=sys.stderr)
            return 4
    cont = np.asarray(z["cont"], dtype="float64")
    q_atom = np.asarray(z["q_atom"], dtype="float64")
    q_joint = (np.asarray(z["q_joint"], dtype="float64").reshape(-1)
               if "q_joint" in z else None)
    y = np.asarray(z["y_true"], dtype="float64")
    mask = np.asarray(z["mask"], dtype="float64")
    well_index = (np.asarray(z["well_index"], dtype="int64") if "well_index" in z
                  else np.zeros(y.shape[0], dtype="int64"))
    n_wells = int(well_index.max()) + 1 if y.shape[0] else 0

    sel = AG.select_tau_per_target(cont=cont, q_atom=q_atom, y=y, mask=mask,
                                   tol=float(args.tol))
    tau = np.asarray(sel["tau"], dtype="float64")
    pr_star = M.atomic_precision_recall(y >= 0.5, q_atom, tau)
    pr_half = M.atomic_precision_recall(y >= 0.5, q_atom, np.full(3, 0.5))
    acc_star, acc_half = {}, {}
    for i, t in enumerate(C.TARGETS):
        obs = mask[:, i] > 0
        truth = (y[:, i] >= 0.5) & obs
        if obs.any():
            acc_star[t] = float(((q_atom[:, i] >= tau[i]) == truth)[obs].mean())
            acc_half[t] = float(((q_atom[:, i] >= 0.5) == truth)[obs].mean())
    cost = AG.misclassification_cost_report(cont=cont, q_atom=q_atom, y=y, mask=mask, tau=tau)
    # 连续切片 = 非「三目标同时等于各自原子值」的行（与 E1/E3/E5 同口径）
    cs = ~np.column_stack([y[:, i] == C.ATOM_VALUES[t]
                           for i, t in enumerate(C.TARGETS)]).all(axis=1)
    cont_slice = M.score_of(y[cs], cont[cs], mask[cs])
    gated_slice = M.score_of(y[cs], AG.per_target_hard_switch(cont, q_atom, tau)[cs],
                             mask[cs])

    # ---- joint_guard（默认关）：必须"总分上升 + 配对 CI 下界 > 0"才采纳
    guard = {"enabled": bool(args.joint_guard), "tau_high": float(args.tau_joint_high),
             "adopted": False, "decision": "off"}
    if args.joint_guard and q_joint is not None:
        base = AG.per_target_hard_switch(cont, q_atom, tau)
        guarded = AG.joint_guard(base, q_joint, float(args.tau_joint_high))
        tot_base = float(M.score_of(y, base, mask)["total"])
        tot_guard = float(M.score_of(y, guarded, mask)["total"])
        d = np.full(n_wells, np.nan)
        for i in range(n_wells):
            selw = well_index == i
            if selw.any():
                d[i] = (M.score_of(y[selw], guarded[selw], mask[selw])["total"]
                        - M.score_of(y[selw], base[selw], mask[selw])["total"])
        ok = ~np.isnan(d)
        boot = FOLDS.bootstrap_ci(d[ok], iters=int(args.iters),
                                  weights=_well_weights(well_index, n_wells)[ok],
                                  seed=42) if ok.any() else {"ci_low": 0.0, "ci_high": 0.0,
                                                             "point": 0.0}
        adopted = bool(tot_guard > tot_base and float(boot["ci_low"]) > 0)
        guard.update({"base_total": tot_base, "guarded_total": tot_guard,
                      "delta": float(tot_guard - tot_base),
                      "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
                      "adopted": adopted,
                      "decision": "adopted" if adopted else "no_go",
                      "reason": ("总分上升且 CI 下界 > 0" if adopted
                                 else "总分未上升或 CI 含 0 → NO-GO（默认关更安全）")})

    # ---- 与基线对照（可选）
    baseline = None
    if args.baseline_oof and Path(args.baseline_oof).is_file():
        with np.load(args.baseline_oof, allow_pickle=True) as bz:
            b = {k: bz[k] for k in bz.files}
        if "cont" in b and np.asarray(b["cont"]).shape == cont.shape:
            tot_new = _total_of(cont, q_atom, tau, y, mask)
            tot_base = float(M.score_of(y, np.asarray(b["cont"], dtype="float64"),
                                        mask)["total"])
            baseline = {"path": str(args.baseline_oof), "baseline_total": tot_base,
                        "tau_star_total": tot_new, "delta": float(tot_new - tot_base)}
        else:
            baseline = {"path": str(args.baseline_oof),
                        "reason": "形状不匹配或缺少 cont 键，未做对照"}

    audit = ST.input_no_label_leak_full(FB.FEATURE_NAMES)
    atomic_report_path = Path(args.atomic_report) if args.atomic_report else \
        reports / "E6_atomic_report.json"
    atomic = (json.loads(atomic_report_path.read_text(encoding="utf-8"))
              if atomic_report_path.is_file() else None)

    payload = {
        "stage": "E6", "p_stage": "P1", "tag": args.tag,
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "oof_path": str(oof_path),
        "oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
        "n_rows": int(y.shape[0]), "n_wells": n_wells,
        "tau_source": "inner_oof_only",
        "grid": {"lo": AG.DEFAULT_TAU_GRID_LO, "hi": AG.DEFAULT_TAU_GRID_HI,
                 "step": AG.DEFAULT_TAU_GRID_STEP,
                 "n": int(len(AG.default_tau_grid())), "plateau_tol": float(args.tol)},
        "tau": [float(v) for v in tau],
        "plateau": {k: [float(a), float(b)] for k, (a, b) in
                    (sel.get("plateau") or {}).items()},
        "curve": _tau_curve(sel),
        "objective": float(sel.get("objective", float("nan"))),
        "score_fn": sel.get("score_fn"),
        "per_target_atom_metrics_tau_star": pr_star,
        "per_target_atom_metrics_tau_half": pr_half,
        "per_target_atom_acc_tau_star": acc_star,
        "per_target_atom_acc_tau_half": acc_half,
        "min_atom_acc": min(acc_star.values()) if acc_star else None,
        "min_atom_recall": min(float(v["recall"]) for v in pr_star.values()) if pr_star else None,
        "misclassification_cost": cost,
        "cont_slice": cont_slice, "gated_slice": gated_slice,
        "joint_guard": guard,
        "baseline": baseline,
        "input_no_label_leak_full": audit,
        "shared": {
            "atomic_report": str(atomic_report_path) if atomic else None,
            "joint_atom_auc": (None if not atomic else (
                float(np.mean(atomic["joint_atom_auc"])) if atomic.get("joint_atom_auc")
                else None)),
            "no_interpolation": (None if not atomic else bool(atomic.get("no_interpolation"))),
            "resumable": (None if not atomic else bool(all(
                r.get("resumable", {}).get("ok") for r in atomic.get("folds_detail", [])))),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": ("τ 只在内折 OOF 上选（grid [0.05,0.95] 步长 0.01，最长平台中点）；"
                  "joint_guard 默认关，开启需总分上升且配对 CI 下界 > 0"),
    }
    write_json(reports / "E6_tau_search.json", payload)

    # ---- 候选登记（smoke 不写；写仓库外路径时可安全调用）
    if not args.smoke:
        try:
            REG.upsert_candidate({
                "candidate_id": "PD1", "stage": "E6",
                "status": "local_only",
                "atomic": {"tau": payload["tau"],
                           "tau_source": payload["tau_source"],
                           "joint_guard": guard["decision"],
                           "min_atom_acc": payload["min_atom_acc"],
                           "min_atom_recall": payload["min_atom_recall"],
                           "per_target_atom_metrics_tau_star": pr_star,
                           "oof_path": payload["oof_path"],
                           "oof_sha256": payload["oof_sha256"]},
                "note": "由 E6/code/search_tau.py 写入（仅 τ/原子指标，权重在 P2 登记）",
            }, path=Path(args.candidates))
        except PermissionError as exc:
            print(f"[E6] 警告：候选登记被拒（{exc}）", file=sys.stderr)

    # ---- P1 Gate（原子 F1 主判据 + 12 项 mandatory）
    prereg_path = reports / "E6_P1_gate_prereg.json"
    if not prereg_path.is_file():
        write_json(prereg_path, {
            "gate_id": "E6_P1_gate", "stage": "E6", "p_stage": "P1", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "atomic_f1", "primary_threshold_key": "min_delta",
            "baseline_version": "PD1_pre_tau" if args.baseline_oof else "CONST",
            "baseline_artifact": str(args.baseline_oof or oof_path),
            "baseline_manifest_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0,
                           "min_atomic_acc": 0.99, "min_atom_acc": 0.99,
                           "min_atom_recall": 0.98, "oof_total_min": 81.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 3,
            "bootstrap_iters": int(args.iters), "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(REQUIRED_CHECKS) + list(E6_CHECKS),
            "decisions_locked": [],
            "notes": "E6/P1：τ 只用内折 OOF；原子 Acc/F1 与连续切片 Acc 同时上报",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    time_log = reports / "training_time_log.json"
    time_ok = False
    if time_log.is_file():
        try:
            log = json.loads(time_log.read_text(encoding="utf-8"))
            time_ok = any(float(f.get("seconds", 0.0)) > 0
                          for f in (log.get("folds") or []) if isinstance(f, dict))
        except Exception:
            time_ok = False
    atom_acc_mean = float(np.mean(list(acc_star.values()))) if acc_star else None
    atom_f1_mean = float(np.mean([float(v["f1"]) for v in pr_star.values()])) if pr_star else None
    checks = {
        "contract_ok": bool(y.shape[0] > 0 and not perrs
                            and cont.shape == q_atom.shape == y.shape == mask.shape),
        "atomic_precision_reported": bool(pr_star),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(time_ok),
        "checkpoint_resumable": bool(payload["shared"]["resumable"] is not False),
        "no_label_leak": bool(audit["ok"] and set(
            (atomic or {}).get("folds_detail", [{}])[0].get("inner_val", []))
            .isdisjoint(set((atomic or {}).get("folds_detail", [{}])[0].get("val_wells", [])))),
        "per_target_atom_acc_reported": bool(len(acc_star) == 3),
        "per_target_atom_precision_recall_f1_reported": bool(
            pr_star and all(all(k in v for k in ("precision", "recall", "f1"))
                            for v in pr_star.values())),
        "joint_atom_auc_reported": bool(payload["shared"]["joint_atom_auc"] is not None),
        "tau_t_inner_oof_only": bool(payload["tau_source"] == "inner_oof_only"),
        "no_atom_continuous_interpolation": bool(payload["shared"]["no_interpolation"]
                                                 is not False),
        "input_no_label_leak_full": bool(audit["ok"]),
    }
    result = {"checks": checks, "score": gated_slice["total"],
              "oof_total": gated_slice["total"], "atomic_f1": atom_f1_mean,
              "atomic_acc": atom_acc_mean, "atom_acc": payload["min_atom_acc"],
              "atom_recall": payload["min_atom_recall"],
              "joint_atom_auc": payload["shared"]["joint_atom_auc"],
              "delta": (baseline or {}).get("delta", 0.0),
              "paired_ci_low": 0.0}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E6", "p_stage": "P1", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "tau": payload["tau"],
            "min_atom_acc": payload["min_atom_acc"],
            "min_atom_recall": payload["min_atom_recall"],
            "atomic_f1": atom_f1_mean, "gated_total": gated_slice["total"],
            "joint_guard": guard["decision"], "checks": checks, "prereg_errors": perrs,
            "aggregate": agg, "disk": disk,
            "report_path": str(reports / "E6_tau_search.json")}
    write_json(reports / "E6_P1_gate.json", gate)
    print(json.dumps({"stage": "E6/P1", "tau": payload["tau"],
                      "min_atom_acc": payload["min_atom_acc"],
                      "min_atom_recall": payload["min_atom_recall"],
                      "atomic_f1": atom_f1_mean, "gated_total": gated_slice["total"],
                      "joint_guard": guard["decision"], "gate_passed": passed,
                      "checks": checks}, ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
