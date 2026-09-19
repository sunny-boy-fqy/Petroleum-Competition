"""Gate 预注册校验与聚合判定（H2：docs/PROJECT_FILES.md 曾声明本文件存在但实际缺失）。

职责
----
1. `validate_prereg(prereg)`：校验必填字段齐全、类型正确、`mandatory_checks` 含 6 项核心、
   `multiplicity` 与 `candidate_budget` 自洽、`min_detectable_effect` 与 `pilot_std/mde_units` 一致。
2. `mde_two_sided(pilot_std, n_units, alpha)`：功效下限估算。
3. `effective_threshold(prereg)`：`max(thresholds[primary_threshold_key], thresholds.min_effect_floor)`。
4. `aggregate_gate(prereg, result)`：判定 `passed`，并强制 mandatory 全绿。

只依赖标准库（外加可选 numpy 用于 MDE 的正态分位）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

V4 = Path(__file__).resolve().parents[2]

CORE_MANDATORY = (
    "contract_ok",
    "atomic_precision_reported",
    "disk_budget_ok",
    "training_time_log_valid",
    "checkpoint_resumable",
    "no_label_leak",
)

REQUIRED_FIELDS = (
    "gate_id", "stage", "p_stage", "created_at",
    "primary_metric", "primary_threshold_key",
    "baseline_version", "baseline_artifact", "baseline_manifest_sha256",
    "thresholds", "alpha", "multiplicity", "candidate_budget",
    "bootstrap_iters", "bootstrap_unit",
    "pilot_std", "mde_units", "min_detectable_effect",
    "planned_task_training_h", "mandatory_checks", "decisions_locked",
)

VALID_MULTIPLICITY = {"none", "holm", "bonferroni", "fdr_bh"}
VALID_PRIMARY = {"oof_total", "confirm_non_inferiority", "env_hard_checks_passed",
                 "data_card_recomputable", "constant_baseline_anchor",
                 "contract_selftest_passed", "row_pipeline_ok", "seq_pipeline_ok",
                 "seq_train_ok", "target_acc", "por_acc", "perm_acc", "sw_acc",
                 "state_auc", "atomic_f1", "guardrail_pass", "no_high_risk_leak",
                 "a_board_no_breakdown", "cpu_inference_ok", "clean_dir_reproduce",
                 "submission_recorded", "archive_complete", "retrospective_complete"}


def mde_two_sided(pilot_std: float, n_units: int, alpha: float = 0.05) -> float:
    """双侧 MDE ≈ (z_{1-α/2} + z_{1-β})·σ/√n，取 power=0.8（z_β=0.8416）。

    无 scipy 时用 Acklam 近似的正态分位（精度 1e-9，足够）。
    """
    if n_units <= 0:
        raise ValueError("n_units must be > 0")
    z_a = _norm_ppf(1 - alpha / 2.0)
    z_b = 0.8416212335729143
    return (z_a + z_b) * float(pilot_std) / math.sqrt(n_units)


def _norm_ppf(p: float) -> float:
    """标准正态分位（Acklam 近似）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def validate_prereg(prereg: dict[str, Any], strict: bool = True) -> list[str]:
    """返回错误列表（空表示通过）。"""
    errs: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in prereg:
            errs.append(f"missing field: {f}")
    if errs:
        return errs

    if prereg["primary_metric"] not in VALID_PRIMARY:
        errs.append(f"primary_metric {prereg['primary_metric']!r} not in {sorted(VALID_PRIMARY)}")
    th = prereg["thresholds"]
    if not isinstance(th, dict) or prereg["primary_threshold_key"] not in th:
        errs.append("thresholds must be a dict containing primary_threshold_key")
    if prereg["multiplicity"] not in VALID_MULTIPLICITY:
        errs.append(f"multiplicity {prereg['multiplicity']!r} invalid")
    if prereg["multiplicity"] == "none" and int(prereg["candidate_budget"]) != 1:
        errs.append("multiplicity=none requires candidate_budget=1")
    if int(prereg["candidate_budget"]) < 1:
        errs.append("candidate_budget must be >= 1")
    if float(prereg["planned_task_training_h"]) <= 0:
        errs.append("planned_task_training_h must be > 0")
    if not isinstance(prereg["mandatory_checks"], list):
        errs.append("mandatory_checks must be a list")
    else:
        miss = [c for c in CORE_MANDATORY if c not in prereg["mandatory_checks"]]
        if miss and strict:
            errs.append(f"mandatory_checks missing core: {miss}")
    # MDE 自洽
    ps, n, mde = prereg["pilot_std"], prereg["mde_units"], prereg["min_detectable_effect"]
    if ps is not None and mde is not None:
        want = mde_two_sided(float(ps), int(n), float(prereg["alpha"]))
        if abs(want - float(mde)) > max(1e-6, 0.05 * want):
            errs.append(f"min_detectable_effect {mde} != mde_two_sided({ps},{n})={want:.6f}")
    return errs


def effective_threshold(prereg: dict[str, Any]) -> float:
    th = prereg["thresholds"]
    base = float(th[prereg["primary_threshold_key"]])
    floor = float(th.get("min_effect_floor", 0.0))
    return max(base, floor)


def aggregate_gate(prereg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """聚合判定。result 需含 `delta`、`paired_ci_low`（阶段 Gate）与 `checks`。

    E9 例外：`primary_metric=confirm_non_inferiority` 时用 `> -margin`。
    """
    errs = validate_prereg(prereg)
    checks = result.get("checks", {})
    mandatory_fail = [c for c in prereg["mandatory_checks"] if not checks.get(c, False)]
    delta = float(result.get("delta", float("nan")))
    ci_low = result.get("paired_ci_low")
    ci_low = float("nan") if ci_low is None else float(ci_low)

    if prereg["primary_metric"] == "confirm_non_inferiority":
        margin = float(prereg["thresholds"]["non_inferiority_margin"])
        metric_pass = (delta > -margin) and (ci_low > -margin)
    else:
        eff = effective_threshold(prereg)
        metric_pass = (delta >= eff) and (ci_low > 0.0)

    passed = (not errs) and (not mandatory_fail) and metric_pass
    return {
        "gate_id": prereg["gate_id"],
        "passed": bool(passed),
        "prereg_errors": errs,
        "mandatory_failures": mandatory_fail,
        "effective_threshold": effective_threshold(prereg),
        "delta": delta,
        "paired_ci_low": ci_low,
        "metric_pass": bool(metric_pass),
    }


def _main() -> int:
    ap = argparse.ArgumentParser(description="v4 gate prereg validation")
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--result", default=None, help="可选：聚合判定用的结果 JSON")
    args = ap.parse_args()
    prereg = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    errs = validate_prereg(prereg)
    if errs:
        print("PREREG INVALID:")
        for e in errs:
            print("  -", e)
        return 1
    print(f"PREREG OK: {prereg['gate_id']}  effective_threshold={effective_threshold(prereg):.6f}")
    if args.result:
        res = json.loads(Path(args.result).read_text(encoding="utf-8"))
        out = aggregate_gate(prereg, res)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
