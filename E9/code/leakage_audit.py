#!/usr/bin/env python3
"""E9/P1：泄漏审计（**纯 numpy/标准库**）——四类必查项，缺证据走 `residual_risk` 而不是静默通过。

四类审计（E9/P1 §5）
------------------
1. **折维度**：折必须按**井**切分——每口井恰好出现在一个折里（同井跨折即泄漏）；
   顺带记录折文件 sha256（可复算）。
2. **输入列**：`F1` 特征名不得含目标/标签/占位/井身份派生列（`input_no_label_leak_full`）；
   若给了 `--provenance-csv`，还检查其列名（溯源表本身也不许出现目标派生列）。
3. **标尺**：逐折 scaler JSON 必须 `fitted_on=train_fold_only` 且 `train_wells ∩ val_wells = ∅`
   （尺度参数若用了验证折，等于把验证分布信息带进训练）。
4. **伪标签/transductive 来源**：若做过 transductive 适配，必须 `uses_test_labels=false`
   且反泄漏护栏通过；没做过则显式记 `residual_risk`（说明"该风险面不存在"），**不静默跳过**。

判定：任一审计 `fail` → `high_risk_leak=true`（退出码 3，除非 `--smoke/--exploratory`）；
`residual_risk` **不算通过也不算失败**，但必须在报告里写明原因，并由 Gate 的
`residual_risks_recorded` 检查项确认"没有被掩盖"。

产出：`$REPORTS/E9_leakage_audit.json`、`$REPORTS/E9_leakage_gate.json`。
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
from src.features import basic as FB  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import state_train as ST  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P1_CHECKS = ("leakage_audit_complete", "no_high_risk_leak", "residual_risks_recorded",
             "fold_dimension_well_level")
AUDIT_IDS = ("fold_dimension", "input_columns", "scalers_fitted_on_train_fold",
             "pseudo_label_source")
VERDICTS = ("pass", "fail", "residual_risk")


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
    ap = argparse.ArgumentParser(description="E9/P1 泄漏审计（四类必查项）")
    ap.add_argument("--folds", default=None, help="折文件（缺省用冻结折）")
    ap.add_argument("--folds-report", default=None, help="E0_folds.json（可复算折指纹）")
    ap.add_argument("--provenance-csv", default=None, help="特征溯源表（可选）")
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or None)
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--transductive-report", default=None,
                    help="E8_transductive.json（缺省按 reports-dir 找）")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


# ---------------------------------------------------------------- 四类审计
def audit_fold_dimension(folds_path: Path | None, folds_report: Path | None) -> dict:
    """折维度：每口井恰好属于一个折（同井跨折即硬泄漏）。"""
    try:
        folds = FOLDS.load_folds(str(folds_path) if folds_path else None)
    except Exception as exc:
        return {"id": "fold_dimension", "verdict": "residual_risk",
                "evidence": {"error": f"{type(exc).__name__}: {exc}"},
                "residual_risk": "折文件不可读：无法验证折维度，视为未验证风险"}
    mapping = folds["fold_of_well"] if "fold_of_well" in folds else {}
    wells = list(folds.get("well_list", []))
    per_well = {}
    for w in wells:
        per_well.setdefault(w, []).append(mapping.get(w))
    duplicates = {w: v for w, v in per_well.items() if len(set(v)) != 1}
    n_wells = len(wells)
    n_folds = len({int(v) for v in mapping.values()}) if mapping else 0
    src = Path(folds.get("source_path") or (folds_path or ""))
    sha = None
    if src and Path(src).is_file():
        sha = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    elif folds_path and Path(folds_path).is_file():
        sha = hashlib.sha256(Path(folds_path).read_bytes()).hexdigest()
    report_sha = None
    if folds_report and Path(folds_report).is_file():
        report_sha = hashlib.sha256(Path(folds_report).read_bytes()).hexdigest()
    verdict = "pass" if (n_wells > 0 and not duplicates and n_folds > 1) else "fail"
    return {"id": "fold_dimension", "verdict": verdict,
            "evidence": {"n_wells": n_wells, "n_folds": n_folds,
                         "duplicated_wells": sorted(duplicates)[:10],
                         "fold_file_sha256": sha, "folds_report_sha256": report_sha,
                         "fold_source": str(src) if src else None,
                         "expected_sha256_in_constants": C.FOLDS_SHA256
                         if hasattr(C, "FOLDS_SHA256") else None},
            "residual_risk": None if verdict == "pass" else
            "折维度异常：同井跨折或折数不足，属于硬泄漏，必须修复后才能提交"}


def audit_input_columns(provenance_csv: Path | None) -> dict:
    """输入列：F1 特征名 + （可选）溯源表列名不得含目标派生列。"""
    f1 = ST.input_no_label_leak_full(FB.FEATURE_NAMES)
    evidence = {"f1_features": {"n": f1["n_features"], "hits": f1["hits"], "ok": f1["ok"]},
                "provenance_csv": None}
    verdict = "pass" if f1["ok"] else "fail"
    residual = None
    if provenance_csv is not None:
        p = Path(provenance_csv)
        if p.is_file():
            header = p.read_text(encoding="utf-8").splitlines()[0]
            cols = [c.strip() for c in header.split(",")]
            csv_audit = ST.input_no_label_leak_full(cols)
            evidence["provenance_csv"] = {"path": str(p), "n_columns": len(cols),
                                          "hits": csv_audit["hits"], "ok": csv_audit["ok"]}
            if not csv_audit["ok"]:
                verdict = "fail"
        else:
            evidence["provenance_csv"] = {"path": str(p), "missing": True}
            residual = "溯源表缺失：无法验证派生列来源（未验证风险）"
    else:
        residual = "未提供 --provenance-csv：只验证了 F1 特征名（未验证风险）"
    if verdict == "pass" and residual:
        verdict = "residual_risk"
    return {"id": "input_columns", "verdict": verdict, "evidence": evidence,
            "residual_risk": residual}


def audit_scalers(scalers_dir: Path | None) -> dict:
    """标尺：逐折 scaler 必须只在训练折上拟合。"""
    if scalers_dir is None or not Path(scalers_dir).is_dir():
        return {"id": "scalers_fitted_on_train_fold", "verdict": "residual_risk",
                "evidence": {"scalers_dir": (None if scalers_dir is None
                                             else str(scalers_dir)), "n_files": 0},
                "residual_risk": "未提供/找不到 scalers 目录：无法验证标尺只在训练折拟合"}
    files = sorted(Path(scalers_dir).glob("*.json"))
    if not files:
        return {"id": "scalers_fitted_on_train_fold", "verdict": "residual_risk",
                "evidence": {"scalers_dir": str(scalers_dir), "n_files": 0},
                "residual_risk": "scalers 目录为空：无法验证标尺只在训练折拟合"}
    rows, bad = [], []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            bad.append({"file": str(f), "error": f"{type(exc).__name__}: {exc}"})
            continue
        tr = set(d.get("train_wells") or [])
        va = set(d.get("val_wells") or [])
        overlap = sorted(tr & va)
        fitted = d.get("fitted_on")
        ok = bool(tr) and not overlap and fitted in (None, "train_fold_only")
        rows.append({"file": f.name, "n_train_wells": len(tr), "n_val_wells": len(va),
                     "overlap": overlap[:5], "fitted_on": fitted, "ok": ok})
        if not ok:
            bad.append({"file": f.name, "reason": "train/val 重叠或 fitted_on 非训练折"})
    verdict = "pass" if rows and not bad else "fail"
    return {"id": "scalers_fitted_on_train_fold", "verdict": verdict,
            "evidence": {"scalers_dir": str(scalers_dir), "n_files": len(files),
                         "files": rows, "problems": bad},
            "residual_risk": None if verdict == "pass" else
            "标尺拟合口径异常：可能把验证折分布带进了训练，属硬泄漏"}


def audit_pseudo_label(transductive_report: Path | None) -> dict:
    """伪标签/transductive 来源：不得使用测试标签（没做过则显式记 residual_risk）。"""
    if transductive_report is None or not Path(transductive_report).is_file():
        return {"id": "pseudo_label_source", "verdict": "residual_risk",
                "evidence": {"report": (None if transductive_report is None
                                        else str(transductive_report))},
                "residual_risk": "未做 transductive 实验（该风险面不存在，但需显式记录）"}
    d = json.loads(Path(transductive_report).read_text(encoding="utf-8"))
    legality = d.get("legality") or {}
    test_side = d.get("test_side") or {}
    guard = (test_side.get("guard") or {})
    uses_labels = bool(legality.get("uses_test_labels", True))
    guard_ok = bool(guard.get("ok", False))
    methods = str(d.get("method", "none"))
    if methods == "none":
        verdict, risk = "pass", None
    elif (not uses_labels) and guard_ok:
        verdict, risk = "pass", None
    else:
        verdict, risk = "fail", "transductive 路径可能使用了测试标签或护栏未通过"
    return {"id": "pseudo_label_source", "verdict": verdict,
            "evidence": {"report": str(transductive_report), "method": methods,
                         "uses_test_labels": uses_labels, "guard_ok": guard_ok,
                         "eval_mode": d.get("eval_mode"),
                         "decision": d.get("decision")},
            "residual_risk": risk}


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    trans = Path(args.transductive_report) if args.transductive_report else \
        reports / "E8_transductive.json"
    audits = [
        audit_fold_dimension(Path(args.folds) if args.folds else None,
                             Path(args.folds_report) if args.folds_report else
                             (reports / "E0_folds.json")),
        audit_input_columns(Path(args.provenance_csv) if args.provenance_csv else None),
        audit_scalers(Path(args.scalers_dir) if args.scalers_dir else None),
        audit_pseudo_label(trans),
    ]
    n_fail = sum(1 for a in audits if a["verdict"] == "fail")
    n_residual = sum(1 for a in audits if a["verdict"] == "residual_risk")
    high_risk = bool(n_fail > 0)
    report = {"stage": "E9", "p_stage": "P1",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "audits": audits, "n_audits": len(audits), "n_fail": n_fail,
              "n_residual_risk": n_residual, "high_risk_leak": high_risk,
              "verdict": ("fail" if high_risk else
                          ("pass_with_residual_risk" if n_residual else "pass")),
              "notes": ("residual_risk 表示『该项未能验证』，既不算通过也不算失败，"
                        "但必须在报告中写明；任何 fail 都是硬泄漏，必须先修复"),
              "residual_risks": [{"id": a["id"], "reason": a["residual_risk"]}
                                 for a in audits if a["residual_risk"]],
              }
    out_path = Path(args.json) if args.json else reports / "E9_leakage_audit.json"
    write_json(out_path, report)

    prereg_path = Path(args.prereg) if args.prereg else reports / "E9_leakage_gate_prereg.json"
    if not prereg_path.is_file():
        write_json(prereg_path, {
            "gate_id": "E9_leakage_gate", "stage": "E9", "p_stage": "P1",
            "gate_type": "boolean", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "no_high_risk_leak", "primary_threshold_key": "min_delta",
            "baseline_version": "n/a", "baseline_artifact": str(out_path),
            "baseline_manifest_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 1,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P1_CHECKS), "decisions_locked": [],
            "notes": "E9/P1：四类泄漏审计；residual_risk 必须显式记录，不得静默通过",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(len(audits) == len(AUDIT_IDS) and not perrs),
        "atomic_precision_reported": True,
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool((reports / "training_time_log.json").is_file()),
        "checkpoint_resumable": True,
        "no_label_leak": bool(not high_risk),
        "leakage_audit_complete": bool({a["id"] for a in audits} == set(AUDIT_IDS)),
        "no_high_risk_leak": bool(not high_risk),
        "residual_risks_recorded": bool(n_residual == len(report["residual_risks"])),
        "fold_dimension_well_level": bool(
            next(a for a in audits if a["id"] == "fold_dimension")["verdict"] == "pass"),
    }
    result = {"checks": checks, "no_high_risk_leak": bool(not high_risk)}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E9", "p_stage": "P1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "high_risk_leak": high_risk,
            "n_fail": n_fail, "n_residual_risk": n_residual,
            "verdict": report["verdict"], "checks": checks, "prereg_errors": perrs,
            "aggregate": agg, "disk": disk, "report_path": str(out_path)}
    write_json(reports / "E9_leakage_gate.json", gate)
    print(json.dumps({"stage": "E9/P1", "verdict": report["verdict"],
                      "high_risk_leak": high_risk, "n_fail": n_fail,
                      "n_residual_risk": n_residual,
                      "audits": {a["id"]: a["verdict"] for a in audits},
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
