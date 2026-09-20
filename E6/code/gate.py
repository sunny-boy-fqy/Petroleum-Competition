#!/usr/bin/env python3
"""E6/P2 Gate：把 P0（原子状态头）/ P1（τ 搜索）/ P2（OOF+提交契约）的证据聚合成 `E6_gate.json`。

设计
----
* **纯 numpy**（不 import torch）：Gate 只读证据文件，任何"跑得起来才判得过"的情况都应当
  在更早的阶段报错，而不是在这里；
* 缺证据就判 **False**（不可复算的 Gate 不能算过），绝不"字段缺失即跳过"；
* `build_gate()` 可被 `build_pd1.py` 直接调用，也可命令行独立复算（同一实现，两处调用）。

判据（E6/P2 §7）
--------------
`oof_total ≥ 82.0`、`min_joint_atom_auc ≥ 0.9`、`min_atom_acc ≥ 0.99`、
`min_atom_recall ≥ 0.98`，外加 12 项 mandatory 与 `cpu_inference_ok`。
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

from src import constants as C  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
E6_CHECKS = ("per_target_atom_acc_reported", "per_target_atom_precision_recall_f1_reported",
             "joint_atom_auc_reported", "tau_t_inner_oof_only",
             "no_atom_continuous_interpolation", "input_no_label_leak_full")
P2_EXTRA = ("cpu_inference_ok",)
THRESHOLDS = {"min_delta": 0.0, "min_effect_floor": 0.0, "oof_total_min": 82.0,
              "min_atom_acc": 0.99, "min_atom_recall": 0.98, "min_joint_atom_auc": 0.9}


def _load(path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def build_gate(reports: Path, *, cv_path=None, oof_path=None, contract_path=None,
               cpu_inference_ok: bool | None = None, prereg_path=None,
               atomic_report=None, tau_report=None, gate_threshold: float = 82.0,
               exploratory: bool = False) -> dict:
    """聚合 P0/P1/P2 证据 → Gate 结果 dict（同时落盘 `E6_gate.json`）。"""
    reports = Path(reports)
    atomic = _load(atomic_report or reports / "E6_atomic_report.json")
    tau = _load(tau_report or reports / "E6_tau_search.json")
    cv = _load(cv_path) if cv_path else _load(reports / "E6_cv.json")
    contract = _load(contract_path) if contract_path else _load(reports / "E6_contract.json")
    oof_file = Path(oof_path) if oof_path else None
    oof_sha = None
    if oof_file is not None and oof_file.is_file():
        oof_sha = hashlib.sha256(oof_file.read_bytes()).hexdigest()

    # ---- 与 CONST 基线的**真实** delta + 按井行数加权配对 CI（缺 OOF 就保持 None）
    delta, ci_low, ci_high = None, None, None
    if oof_file is not None and oof_file.is_file():
        try:
            import numpy as np                                          # noqa: PLC0415
            from src.validation import folds as FOLDS                   # noqa: PLC0415
            with np.load(oof_file, allow_pickle=True) as z:
                o = {k: z[k] for k in z.files}
            need = ("y_true", "gated", "mask", "well_index")
            if all(k in o for k in need):
                y = np.asarray(o["y_true"], dtype="float64")
                pred = np.asarray(o["gated"], dtype="float64")
                mask = np.asarray(o["mask"], dtype="float64")
                widx = np.asarray(o["well_index"], dtype="int64")
                n_wells = int(widx.max()) + 1 if y.shape[0] else 0
                const = np.tile([C.ATOM_VALUES[t] for t in C.TARGETS], (y.shape[0], 1))
                d = np.full(n_wells, np.nan)
                rows = np.zeros(n_wells)
                for i in range(n_wells):
                    selw = widx == i
                    rows[i] = float(selw.sum())
                    if selw.any():
                        d[i] = (M.score_of(y[selw], pred[selw], mask[selw])["total"]
                                - M.score_of(y[selw], const[selw], mask[selw])["total"])
                ok = ~np.isnan(d)
                if ok.any():
                    boot = FOLDS.bootstrap_ci(d[ok], iters=1000, weights=rows[ok], seed=42)
                    delta, ci_low, ci_high = (float(boot["point"]), float(boot["ci_low"]),
                                              float(boot["ci_high"]))
        except Exception as exc:                       # 不静默：把原因写进 Gate
            checks_note = f"delta 计算失败：{type(exc).__name__}: {exc}"
        else:
            checks_note = None

    checks_note = locals().get("checks_note")
    per_target_auc = (atomic or {}).get("per_target_auc") or {}
    joint_auc = _mean((atomic or {}).get("joint_atom_auc") or [])
    min_atom_acc = (atomic or {}).get("min_atom_acc")
    min_atom_recall = (atomic or {}).get("min_atom_recall")
    oof_total = None
    if cv:
        oof_total = (cv.get("total") if isinstance(cv, dict) and "total" in cv
                     else (cv.get("oof_total") if isinstance(cv, dict) else None))
    cont_slice_acc = ((tau or {}).get("cont_slice") or {}).get("total")
    gated_total = ((tau or {}).get("gated_slice") or {}).get("total")

    log_path = reports / "training_time_log.json"
    time_ok = False
    if log_path.is_file():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
            time_ok = any(float(f.get("seconds", 0.0)) > 0
                          for f in (log.get("folds") or []) if isinstance(f, dict))
        except Exception:
            time_ok = False

    checks: dict = {
        "contract_ok": bool(contract is not None and contract.get("ok")),
        "atomic_precision_reported": bool(atomic is not None
                                          and atomic.get("atom_metrics_tau_half")),
        "disk_budget_ok": bool((contract or {}).get("disk", {}).get("level") == "ok"),
        "training_time_log_valid": bool(time_ok),
        "checkpoint_resumable": bool(atomic is not None and all(
            r.get("resumable", {}).get("ok") for r in atomic.get("folds_detail", []))),
        "no_label_leak": bool(atomic is not None and all(
            set(r.get("inner_val", [])).isdisjoint(set(r.get("va_wells", [])))
            for r in atomic.get("folds_detail", []))),
        "per_target_atom_acc_reported": bool(all(
            t in ((atomic or {}).get("atom_metrics_tau_half", [{}])[0].get("per_target_acc", {})
                  or {}) for t in C.TARGETS)),
        "per_target_atom_precision_recall_f1_reported": bool(
            atomic is not None and all(
                all(k in v for k in ("precision", "recall", "f1"))
                for r in atomic.get("atom_metrics_tau_half", [])
                for v in r.get("per_target", {}).values())),
        "joint_atom_auc_reported": bool(joint_auc is not None),
        "tau_t_inner_oof_only": bool((tau or {}).get("tau_source") == "inner_oof_only"),
        "no_atom_continuous_interpolation": bool(
            (tau or {}).get("shared", {}).get("no_interpolation") is True
            or (atomic or {}).get("no_interpolation") is True),
        "input_no_label_leak_full": bool((tau or {}).get("input_no_label_leak_full", {})
                                         .get("ok") is True),
        "cpu_inference_ok": bool(cpu_inference_ok is True),
    }
    prereg = _load(prereg_path or reports / "E6_P2_gate_prereg.json")
    if prereg is None:
        prereg = {
            "gate_id": "E6_gate", "stage": "E6", "p_stage": "P2", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "oof_total", "primary_threshold_key": "min_delta",
            "baseline_version": "CONST", "baseline_artifact": "E6/oof.npz",
            "baseline_manifest_sha256": oof_sha or hashlib.sha256(b"e6-oof").hexdigest(),
            "thresholds": dict(THRESHOLDS, oof_total_min=float(gate_threshold)),
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 3,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 2.0,
            "mandatory_checks": list(CORE_CHECKS) + list(E6_CHECKS) + list(P2_EXTRA),
            "decisions_locked": [],
            "notes": ("E6/P2：PD1 全管线（原子两阶段 + τ + 折平均）OOF Total ≥ 82.0、"
                      "联合原子 AUC ≥ 0.9；CPU 推理契约必须通过"),
        }
    perrs = GATES.validate_prereg(prereg)
    result = {"checks": checks, "score": oof_total, "oof_total": oof_total,
              "delta": delta, "paired_ci_low": ci_low,
              "atom_acc": min_atom_acc, "atom_recall": min_atom_recall,
              "joint_atom_auc": joint_auc}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if exploratory else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E6", "p_stage": "P2",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(exploratory), "passed": passed,
            "nogo": bool(passed is False),
            "oof_total": oof_total, "gate_threshold": float(gate_threshold),
            "delta_vs_const": delta, "paired_ci_vs_const": [ci_low, ci_high],
            "delta_note": checks_note,
            "joint_atom_auc": joint_auc, "min_atom_acc": min_atom_acc,
            "min_atom_recall": min_atom_recall, "gated_total": gated_total,
            "cont_slice_total": cont_slice_acc, "oof_sha256": oof_sha,
            "checks": checks, "prereg_errors": perrs, "aggregate": agg,
            "evidence": {"atomic_report": str(atomic_report or reports / "E6_atomic_report.json"),
                         "tau_report": str(tau_report or reports / "E6_tau_search.json"),
                         "cv": (str(cv_path) if cv_path else None),
                         "contract": (str(contract_path) if contract_path else None),
                         "oof": (str(oof_path) if oof_path else None)}}
    M.jsonable(gate)                                   # 早失败：Gate 必须可 JSON 序列化
    return gate


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E6/P2 Gate（只读证据）")
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(Path(os.environ.get("V4_DATA_ROOT", "/data")) / "v4" / "reports"))
    ap.add_argument("--atomic-report", default=None)
    ap.add_argument("--tau-report", default=None)
    ap.add_argument("--cv", default=None)
    ap.add_argument("--oof", default=None)
    ap.add_argument("--contract-json", default=None)
    ap.add_argument("--cpu-inference-ok", action="store_true",
                    help="CPU 推理冒烟已通过（由 build_pd1.py 传入）")
    ap.add_argument("--gate-threshold", type=float, default=82.0)
    ap.add_argument("--json", default=None, help="输出路径（缺省 <reports>/E6_gate.json）")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    gate = build_gate(reports, cv_path=args.cv, oof_path=args.oof,
                      contract_path=args.contract_json,
                      cpu_inference_ok=(True if args.cpu_inference_ok else None),
                      atomic_report=args.atomic_report, tau_report=args.tau_report,
                      gate_threshold=args.gate_threshold, exploratory=args.exploratory)
    out = Path(args.json) if args.json else reports / "E6_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(gate), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({"gate_id": gate["gate_id"], "oof_total": gate["oof_total"],
                      "joint_atom_auc": gate["joint_atom_auc"],
                      "passed": gate["passed"], "checks": gate["checks"]},
                     ensure_ascii=False, indent=2))
    if args.exploratory or gate["passed"] is None:
        return 0
    return 0 if gate["passed"] else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
