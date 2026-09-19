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
# Gate 类型语义（R2-B2 / R2-H4）
#   delta        : 相对基线增量；要求 delta >= effective_threshold 且 paired_ci_low > 0
#   absolute     : 绝对分数门槛；要求 score >= thresholds[primary_threshold_key]
#   boolean      : 确定性检查（env/契约/复现等）；只要求 mandatory_checks 全绿
#   non_inferior : E9 专用；delta > -margin 且 ci_low > -margin
GATE_TYPES = ("delta", "absolute", "boolean", "non_inferior")
# 除主键外还要同时满足的绝对门槛键（R2-B2：oof_total_min 等此前被忽略）
ABSOLUTE_KEYS = ("oof_total_min", "min_auc", "min_atomic_acc", "max_hard_failures",
                 "abs_tolerance", "max_point_diff", "max_minutes", "max_memory_gb",
                 "max_degradation")

BOOLEAN_METRICS = ("env_hard_checks_passed", "guardrail_pass", "no_high_risk_leak",
                   "clean_dir_reproduce", "submission_recorded", "archive_complete",
                   "retrospective_complete", "constant_baseline_anchor",
                   "data_card_recomputable", "contract_selftest_passed")
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


def is_template(prereg: dict[str, Any]) -> bool:
    """是否是"未填写的模板"（占位符仍在）。模板允许占位，生效预注册不允许。"""
    blob = " ".join(str(prereg.get(k, "")) for k in
                    ("created_at", "baseline_version", "baseline_artifact",
                     "baseline_manifest_sha256"))
    return "<" in blob


def validate_prereg(prereg: dict[str, Any], strict: bool = True,
                    template: bool | None = None) -> list[str]:
    """返回错误列表（空表示通过）。

    strict=True 时对**内容**也做校验；但 `is_template()` 为真时自动放宽占位符检查
    （模板需要能进仓库被复用，生效的预注册不能有占位符）。
    """
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
        exempt = set(prereg.get("mandatory_exempt", []) or [])
        miss = [c for c in CORE_MANDATORY
                if c not in prereg["mandatory_checks"] and c not in exempt]
        if miss and strict:
            errs.append(f"mandatory_checks missing core: {miss}")
    # 非占位内容检查（R2-B2：此前只查字段存在）
    _tpl = is_template(prereg) if template is None else bool(template)
    if strict and not _tpl:
        gtype = gate_type(prereg)
        placeholders = ("<", ">", "TODO", "TBD")
        required_nonempty = ["baseline_version", "baseline_artifact"]
        # boolean 类 Gate（env/契约/复现等）没有上游 manifest，允许留空
        if gtype != "boolean":
            required_nonempty.append("baseline_manifest_sha256")
        for f in required_nonempty:
            v = str(prereg.get(f, ""))
            if not v or any(t in v for t in placeholders):
                errs.append(f"{f} looks like a placeholder: {v!r}")
        if str(prereg.get("created_at", "")).startswith("<"):
            errs.append(f"created_at looks like a placeholder: {prereg.get('created_at')!r}")
        gid = str(prereg.get("gate_id", ""))
        stage, pstage = str(prereg.get("stage", "")), str(prereg.get("p_stage", ""))
        # gate_id 允许 {stage}_{p_stage} 或阶段级 {stage}_... 两种命名
        if stage and pstage and not (gid.startswith(f"{stage}_{pstage}")
                                     or gid.startswith(f"{stage}_")):
            errs.append(f"gate_id {gid!r} inconsistent with stage/p_stage {stage}/{pstage}")
        # 非 delta Gate 不应声明 oof_total_min 之类的增量分数门槛
        if gtype != "delta":
            bad = [k for k in ("oof_total_min", "min_auc") if k in prereg["thresholds"]]
            if bad and gtype == "boolean":
                errs.append(f"boolean Gate must not declare absolute score thresholds {bad}")

    # MDE 自洽
    ps, n, mde = prereg["pilot_std"], prereg["mde_units"], prereg["min_detectable_effect"]
    if ps is not None and mde is not None:
        want = mde_two_sided(float(ps), int(n), float(prereg["alpha"]))
        if abs(want - float(mde)) > max(1e-6, 0.05 * want):
            errs.append(f"min_detectable_effect {mde} != mde_two_sided({ps},{n})={want:.6f}")
    return errs


def gate_type(prereg: dict[str, Any]) -> str:
    """显式 `gate_type` 优先；否则按 `primary_metric` 推断（兼容旧模板）。"""
    if prereg.get("gate_type") in GATE_TYPES:
        return prereg["gate_type"]
    pm = prereg.get("primary_metric", "")
    if pm == "confirm_non_inferiority":
        return "non_inferior"
    if pm in BOOLEAN_METRICS or pm.endswith(("_ok", "_passed", "_reported")):
        return "boolean"
    return "delta"


def required_absolute_keys(prereg: dict[str, Any]) -> list[str]:
    """模板里声明、且必须同时满足的绝对门槛键（排除 delta 类键）。"""
    th = prereg["thresholds"]
    return [k for k in ABSOLUTE_KEYS if k in th]


def effective_threshold(prereg: dict[str, Any]) -> float:
    """delta 类 Gate 的生效阈值 = max(主键阈值, min_effect_floor)。"""
    th = prereg["thresholds"]
    base = float(th[prereg["primary_threshold_key"]])
    floor = float(th.get("min_effect_floor", 0.0))
    return max(base, floor)


def aggregate_gate(prereg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """按 Gate 类型做聚合判定（R2-B2 / R2-H4 修复）。

    result 字段（按需）：
      - checks        : dict[str,bool]  mandatory check 结果
      - delta         : 相对基线的增量
      - paired_ci_low : 加权配对 bootstrap 下界
      - score         : absolute 类 Gate 的绝对分数
      - value         : boolean 类 Gate 的布尔结果（可省略，改用 checks）
    """
    errs = validate_prereg(prereg)
    gtype = gate_type(prereg)
    checks = result.get("checks", {}) or {}
    mandatory_fail = [c for c in prereg["mandatory_checks"] if not checks.get(c, False)]
    delta = result.get("delta")
    delta = float("nan") if delta is None else float(delta)
    ci_low = result.get("paired_ci_low")
    ci_low = None if ci_low is None else float(ci_low)
    score = result.get("score")
    score = None if score is None else float(score)
    th = prereg["thresholds"]

    details: dict[str, Any] = {"gate_type": gtype}
    if gtype == "boolean":
        # 确定性 Gate：只看 mandatory checks（不要求 delta/CI）
        metric_pass = not mandatory_fail
        details["note"] = "boolean Gate 只要求 mandatory_checks 全绿"
    elif gtype == "absolute":
        key = prereg["primary_threshold_key"]
        if key not in th or score is None:
            metric_pass = False
            details["error"] = f"absolute Gate 需要 score 与 thresholds[{key!r}]"
        else:
            want = float(th[key])
            metric_pass = score >= want
            details["required"] = want
            details["score"] = score
    elif gtype == "non_inferior":
        margin = float(th["non_inferiority_margin"])
        metric_pass = (delta > -margin) and (ci_low is not None and ci_low > -margin)
        details["non_inferiority_margin"] = margin
    else:  # delta
        eff = effective_threshold(prereg)
        primary_pass = (delta >= eff) and (ci_low is not None and ci_low > 0.0)
        # 附加绝对门槛（如 oof_total_min / min_auc）必须同时满足
        abs_checks: dict[str, Any] = {}
        for k in required_absolute_keys(prereg):
            want = float(th[k])
            have = score
            if have is None:
                # 允许把绝对分数放进 result["score"] 或 result["oof_total"]
                have = result.get("oof_total")
                have = None if have is None else float(have)
            ok = have is not None and have >= want
            abs_checks[k] = {"required": want, "have": have, "ok": ok}
        metric_pass = primary_pass and all(v["ok"] for v in abs_checks.values())
        details["effective_threshold"] = eff
        details["primary_pass"] = primary_pass
        details["absolute_checks"] = abs_checks

    passed = (not errs) and (not mandatory_fail) and metric_pass
    return {
        "gate_id": prereg["gate_id"],
        "gate_type": gtype,
        "passed": bool(passed),
        "prereg_errors": errs,
        "mandatory_failures": mandatory_fail,
        "effective_threshold": (effective_threshold(prereg)
                                if gtype in ("delta",) else None),
        "delta": None if delta != delta else delta,
        "paired_ci_low": ci_low,
        "metric_pass": bool(metric_pass),
        "details": details,
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
