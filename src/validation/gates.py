"""Gate 预注册校验与聚合判定（H2：docs/PROJECT_FILES.md 曾声明本文件存在但实际缺失）。

职责
----
1. `validate_prereg(prereg)`：校验必填字段齐全、类型正确、`mandatory_checks` 含 6 项核心、
   `multiplicity` 与 `candidate_budget` 自洽、`min_detectable_effect` 与 `pilot_std/mde_units` 一致、
   绝对门槛键**方向已知**。
2. `mde_two_sided(pilot_std, n_units, alpha)`：功效下限估算。
3. `effective_threshold(prereg)`：`max(thresholds[primary_threshold_key], thresholds.min_effect_floor)`。
4. `required_absolute_keys` / `check_absolute_thresholds` / `absolute_direction`：
   `min_*` 走 `>=`、`max_*` 走 `<=`，并把指标名映射到 result 字段（R3-H1）。
5. `aggregate_gate(prereg, result)`：判定 `passed`，并强制 mandatory 全绿 + 绝对门槛全绿。

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

# ---------------------------------------------------------------------------
# R3-H1：绝对门槛的**方向**与**指标字段映射**
# ---------------------------------------------------------------------------
# 旧实现把所有 ABSOLUTE_KEYS 一律按 `have >= want` 判定，导致
#   max_hard_failures / max_point_diff / max_minutes / max_memory_gb / max_degradation / abs_tolerance
# 被当成下界（例如 max_degradation=0.1 要求 degradation>=0.1），方向完全反了；
# 且 result 里只认 score/oof_total，auc/atomic_acc/minutes/memory_gb/point_diff 等字段取不到值，
# 于是这些阈值**静默失效**。下面按方向拆成两组，并为每个键声明 result 字段名候选。
MIN_ABSOLUTE_KEYS = (
    "oof_total_min", "min_same_direction_folds", "min_placeholder_acc",
    "min_auc", "min_atomic_acc", "min_atomic_f1",
    "min_atomic_precision", "min_atomic_recall",
    "min_atom_acc", "min_atom_precision", "min_atom_recall", "min_atom_f1",
    "min_joint_atom_auc", "min_por_acc", "min_perm_acc", "min_sw_acc",
    "min_align_score", "min_hard_pass", "min_effect_abs",
)
MAX_ABSOLUTE_KEYS = (
    "max_hard_failures", "max_point_diff", "max_minutes", "max_memory_gb",
    "max_degradation", "max_regression", "abs_tolerance",
)
ABSOLUTE_KEYS = MIN_ABSOLUTE_KEYS + MAX_ABSOLUTE_KEYS

# 这些 `min_*` 键是 delta / non_inferior 类 Gate 的主阈值，不是绝对门槛
DELTA_THRESHOLD_KEYS = ("min_delta", "min_effect_floor", "non_inferiority_margin")

# 阈值键 -> 允许的 result 字段名（按顺序取第一个**存在**的）
METRIC_RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    # --- 分数 ---
    "oof_total_min": ("oof_total", "score"),
    # E1 硬 Gate 的“5 折同向”判据（审查 M2：此前只写在 notes 里，未真正判定）
    "min_same_direction_folds": ("same_direction_folds",),
    # E1 硬 Gate 的“占位行逐目标 Acc”判据（审查：此前只写在报告里，未真正判定）
    "min_placeholder_acc": ("placeholder_min_acc", "min_placeholder_acc"),
    # --- E6/P0 状态分类 ---
    "min_auc": ("auc", "state_auc"),
    "min_atomic_acc": ("atomic_acc",),
    "min_atomic_f1": ("atomic_f1",),
    "min_atomic_precision": ("atomic_precision",),
    "min_atomic_recall": ("atomic_recall",),
    # --- 逐目标原子头（改进 proposal §2/§11） ---
    "min_atom_acc": ("atom_acc",),
    "min_atom_precision": ("atom_precision",),
    "min_atom_recall": ("atom_recall",),
    "min_atom_f1": ("atom_f1",),
    "min_joint_atom_auc": ("joint_atom_auc",),
    "min_por_acc": ("por_acc",),
    "min_perm_acc": ("perm_acc",),
    "min_sw_acc": ("sw_acc",),
    "min_align_score": ("align_score",),
    "min_effect_abs": ("effect_abs", "min_effect_abs"),
    # --- E0/P0 环境 ---
    "min_hard_pass": ("hard_pass", "n_hard_pass", "hard_passed"),
    # --- 资源 / 退化解 ---
    "max_hard_failures": ("hard_failures",),
    "max_point_diff": ("point_diff", "max_point_diff"),
    "max_minutes": ("minutes", "cpu_minutes"),
    "max_memory_gb": ("memory_gb", "cpu_memory_gb"),
    "max_degradation": ("degradation", "max_degradation"),
    "max_regression": ("regression", "max_regression"),
    # abs_tolerance 比较的是**偏差绝对值**，因此 result 要提供 abs_diff/score_diff
    "abs_tolerance": ("abs_diff", "score_diff", "abs_error"),
}


def absolute_direction(key: str) -> str:
    """`min` -> 要求 `have >= want`；`max` -> 要求 `have <= want`。未知键抛错。"""
    if key in MIN_ABSOLUTE_KEYS:
        return "min"
    if key in MAX_ABSOLUTE_KEYS:
        return "max"
    raise KeyError(f"unknown absolute threshold key: {key!r}")

BOOLEAN_METRICS = ("env_hard_checks_passed", "guardrail_pass", "no_high_risk_leak",
                   "clean_dir_reproduce", "submission_recorded", "archive_complete",
                   "retrospective_complete", "constant_baseline_anchor",
                   "data_card_recomputable", "contract_selftest_passed",
                   # 管线/契约/交付类的确定性判据（R3-H1：此前只按 _ok/_passed/_reported
                   # 后缀推断，导致 a_board_no_breakdown 被误判为 delta，其 max_degradation
                   # 阈值与 mandatory_checks 的语义都对不上）
                   "a_board_no_breakdown", "cpu_inference_ok", "row_pipeline_ok",
                   "seq_pipeline_ok", "seq_train_ok",
                   # 改进 proposal §11：E6 的确定性 mandatory 判据
                   "tau_t_inner_oof_only", "no_atom_continuous_interpolation",
                   "joint_guard_inner_oof_only", "sw_single_label_scale")
VALID_PRIMARY = {"oof_total", "confirm_non_inferiority", "env_hard_checks_passed",
                 "data_card_recomputable", "constant_baseline_anchor",
                 "contract_selftest_passed", "row_pipeline_ok", "seq_pipeline_ok",
                 "seq_train_ok", "target_acc", "por_acc", "perm_acc", "sw_acc",
                 "state_auc", "atomic_f1", "guardrail_pass", "no_high_risk_leak",
                 "a_board_no_breakdown", "cpu_inference_ok", "clean_dir_reproduce",
                 "submission_recorded", "archive_complete", "retrospective_complete",
                 # 改进 proposal（R3-H3）：逐目标原子头与损失调试用指标
                 "joint_atom_auc", "per_target_atom_acc", "align_score"}


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

    # R3-H1：任何形如 min_*/max_*/abs_* 的阈值键都必须有已知方向，否则 aggregate_gate
    # 会静默放过（正是三审发现的缺陷）。**模板也要查**，否则坏模板会一直在仓库里。
    unknown = [k for k in th
               if k.startswith(("min_", "max_", "abs_"))
               and k not in ABSOLUTE_KEYS and k not in DELTA_THRESHOLD_KEYS]
    if unknown:
        errs.append(f"thresholds 含方向未知的绝对门槛键 {unknown}；"
                    f"请加入 gates.MIN_ABSOLUTE_KEYS / MAX_ABSOLUTE_KEYS 并给出 result 字段映射")

    # MDE 自洽
    ps, n, mde = prereg["pilot_std"], prereg["mde_units"], prereg["min_detectable_effect"]
    if ps is not None and mde is not None:
        want = mde_two_sided(float(ps), int(n), float(prereg["alpha"]))
        if abs(want - float(mde)) > max(1e-6, 0.05 * want):
            errs.append(f"min_detectable_effect {mde} != mde_two_sided({ps},{n})={want:.6f}")
    return errs


def infer_gate_type(primary_metric: str, gate_type_field: Any = None) -> str:
    """Gate 类型的**唯一事实源**（生成器 `docs/gen_p_details.py` 直接 import 本函数）。

    R3-H1：此前生成器里另抄了一份 boolean 指标清单，与 `BOOLEAN_METRICS` 不一致，
    导致同一条目「显式 gate_type」与「推断 gate_type」不相等（E9/P2 的
    `a_board_no_breakdown` 就是实例）。现在只有这一处定义。
    """
    if gate_type_field in GATE_TYPES:
        return gate_type_field
    pm = str(primary_metric or "")
    if pm == "confirm_non_inferiority":
        return "non_inferior"
    if pm in BOOLEAN_METRICS or pm.endswith(("_ok", "_passed", "_reported")):
        return "boolean"
    return "delta"


def gate_type(prereg: dict[str, Any]) -> str:
    """显式 `gate_type` 优先；否则按 `primary_metric` 推断（兼容旧模板）。"""
    return infer_gate_type(prereg.get("primary_metric", ""), prereg.get("gate_type"))


def required_absolute_keys(prereg: dict[str, Any],
                           gtype: str | None = None) -> list[str]:
    """模板里声明、且必须同时满足的绝对门槛键（排除 delta 类键）。

    R3-H1：`primary_threshold_key` 若本身就是一个绝对键，则由该 Gate 类型的主判定消费
    （delta 比增量、absolute 比分数、non_inferior 比 margin），此处不再重复计一次。
    """
    th = prereg["thresholds"]
    prim = prereg.get("primary_threshold_key")
    consumed = gtype in ("delta", "absolute", "non_inferior")
    return [k for k in ABSOLUTE_KEYS
            if k in th and not (consumed and k == prim)]


def resolve_metric_value(key: str, result: dict[str, Any]) -> tuple[Any, str | None]:
    """按 `METRIC_RESULT_FIELDS` 从 result 里取值；返回 `(value, field_name)`。

    取不到时返回 `(None, None)` —— 由调用方判为 **未通过**（宁严勿松：
    声明了阈值却拿不到指标，等价于该项没被验证）。
    """
    for field in METRIC_RESULT_FIELDS.get(key, (key,)):
        if field in result and result[field] is not None:
            return result[field], field
    return None, None


def check_absolute_thresholds(prereg: dict[str, Any],
                              result: dict[str, Any],
                              gtype: str | None = None) -> dict[str, dict[str, Any]]:
    """逐项校验绝对门槛（带方向）。返回 `{key: {...}}`，全部 `ok` 才算通过。"""
    th = prereg["thresholds"]
    out: dict[str, dict[str, Any]] = {}
    for k in required_absolute_keys(prereg, gtype=gtype):
        try:
            direction = absolute_direction(k)
        except KeyError as exc:                      # 未知键 -> 明确失败而非静默放过
            out[k] = {"required": th[k], "have": None, "ok": False,
                      "direction": "?", "field": None, "error": str(exc)}
            continue
        want = float(th[k])
        raw, field = resolve_metric_value(k, result)
        have = None if raw is None else float(raw)
        if have is None:
            ok = False
        elif direction == "min":
            ok = have >= want
        else:
            ok = have <= want
        out[k] = {"required": want, "have": have, "ok": bool(ok),
                  "direction": direction, "field": field}
    return out


def effective_threshold(prereg: dict[str, Any]) -> float:
    """delta 类 Gate 的生效阈值 = max(主键阈值, min_effect_floor)。"""
    th = prereg["thresholds"]
    base = float(th[prereg["primary_threshold_key"]])
    floor = float(th.get("min_effect_floor", 0.0))
    return max(base, floor)


def aggregate_gate(prereg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """按 Gate 类型做聚合判定（R2-B2 / R2-H4 / R3-H1 修复）。

    result 字段（按需）：
      - checks          : dict[str,bool]  mandatory check 结果
      - delta           : 相对基线的增量
      - paired_ci_low   : 加权配对 bootstrap 下界
      - score/oof_total : 绝对分数
      - auc/state_auc, atomic_acc, atomic_f1, minutes, memory_gb,
        point_diff, degradation, hard_failures, abs_diff ... 见 `METRIC_RESULT_FIELDS`

    R3-H1 关键语义：
      * `min_*` 键要求 `have >= want`；`max_*` 键要求 `have <= want`（此前一律按 >= 判定）；
      * 绝对门槛在**所有** Gate 类型下都参与判定（此前 boolean Gate 直接忽略它们，
        导致 E10/P0 的 `max_minutes`/`max_memory_gb` 与 E10/P1 的 `max_point_diff` 静默失效）；
      * 声明了阈值但 result 未提供对应指标 -> 判为**未通过**（不可复算的 Gate 不能算过）。
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
        # 确定性 Gate：不要求 delta/CI；但仍须满足声明的资源/计数类绝对门槛
        metric_pass = True
        details["note"] = "boolean Gate 不要求 delta/CI，但仍强制绝对门槛与 mandatory_checks"
    elif gtype == "absolute":
        # R4-M3：此前永远读 `result["score"]`，即使 primary_threshold_key 是
        # min_auc / max_minutes 这类应映射到 auc / minutes 的键 —— 会取错指标（或
        # 取不到值而静默判失败）。现在统一走 `resolve_metric_value`。
        key = prereg["primary_threshold_key"]
        if key not in th:
            metric_pass = False
            details["error"] = (f"absolute Gate 的 primary_threshold_key {key!r} 不在 "
                                f"thresholds 中")
        else:
            want = float(th[key])
            raw, field = resolve_metric_value(key, result)
            if raw is None and key not in METRIC_RESULT_FIELDS and score is not None:
                # 兼容旧口径：未登记映射的键（如 "score"）且 result 提供 score
                raw, field = score, "score"
            try:
                direction = absolute_direction(key)
            except KeyError:
                direction = "min"
            have = None if raw is None else float(raw)
            if have is None:
                metric_pass = False
                candidates = METRIC_RESULT_FIELDS.get(key, (key,))
                details["error"] = (f"absolute Gate 声明 thresholds[{key!r}]={want}，但 result "
                                    f"缺少 {list(candidates)} 中任一字段（宁严勿松：拿不到 "
                                    "指标等价于未验证）")
            else:
                metric_pass = (have >= want) if direction == "min" else (have <= want)
            details.update({"primary_threshold_key": key, "required": want,
                            "have": have, "field": field, "direction": direction})
    elif gtype == "non_inferior":
        margin = float(th["non_inferiority_margin"])
        metric_pass = (delta > -margin) and (ci_low is not None and ci_low > -margin)
        details["non_inferiority_margin"] = margin
    else:  # delta
        eff = effective_threshold(prereg)
        metric_pass = (delta >= eff) and (ci_low is not None and ci_low > 0.0)
        details["effective_threshold"] = eff
        details["primary_pass"] = metric_pass

    # 绝对门槛：所有 Gate 类型统一强制（R3-H1）
    abs_checks = check_absolute_thresholds(prereg, result, gtype=gtype)
    if abs_checks:
        details["absolute_checks"] = abs_checks
        metric_pass = metric_pass and all(v["ok"] for v in abs_checks.values())

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
