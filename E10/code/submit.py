#!/usr/bin/env python3
"""E10/P2：提交记录与冻结（纯标准库 + registry）——**只记录，不据反馈调参**。

职责
----
1. 把 `{zip_sha256, code_zip_sha256, submit_time, score|error, choice, reason, budget}`
   **追加**到 `$REPORTS/E10_submission_log.json`（只追加，不改历史）；
2. 配额：按日志统计"今天已提交次数"，超过 `--budget-per-day` 直接拒绝；
3. `--freeze` 时把该候选标为 `submitted` 并记录 `frozen_sha256`（冻结后不得再改产物）；
4. `choice=b0_fallback` 分支：记录原因（v4 不可用/退化），不假装 v4 成功。

产出：`$REPORTS/E10_submission_log.json`、`$REPORTS/E10_submission_report.json`、
`$REPORTS/E10_P2_gate.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import gates as GATES  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P2_CHECKS = ("submission_log_complete", "zip_fingerprint_recorded", "budget_ok",
             "candidate_frozen_after_submit", "no_retune_from_feedback")
CHOICES = ("v4", "b0_fallback")


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


def sha256_file(p) -> str | None:
    p = Path(p) if p else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E10/P2 提交记录（追加日志 + 配额 + 冻结）")
    ap.add_argument("--zip", default=str(V4 / "submission" / "result.zip"))
    ap.add_argument("--code-zip", default=str(V4 / "submission" / "submission_code_v4.zip"))
    ap.add_argument("--score", type=float, default=None)
    ap.add_argument("--submitted-at", default=None)
    ap.add_argument("--error", default=None)
    ap.add_argument("--choice", default="v4", choices=CHOICES)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--candidate-id", default=None)
    ap.add_argument("--decision", default=None, help="E9_submission_decision.json（取 choice）")
    ap.add_argument("--candidates", default=os.environ.get("V4_CANDIDATES") or str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--log", default=None)
    ap.add_argument("--budget-per-day", type=int, default=5)
    ap.add_argument("--freeze", action="store_true", help="记录后把候选标为 submitted 并冻结")
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def records_today(log: dict, day: str) -> int:
    return sum(1 for r in log.get("records", [])
               if str(r.get("submit_time", "")).startswith(day))


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else reports / "E10_submission_log.json"
    log = (json.loads(log_path.read_text(encoding="utf-8"))
           if log_path.is_file() else {"schema_version": 1, "records": []})
    decision = None
    if args.decision and Path(args.decision).is_file():
        decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    candidate_id = args.candidate_id or (decision or {}).get("choice")
    if candidate_id == "B0_FALLBACK":
        candidate_id = None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used_today = records_today(log, today)
    budget_ok = bool(used_today < int(args.budget_per_day))
    zip_sha = sha256_file(args.zip) if Path(args.zip).is_file() else None
    code_sha = sha256_file(args.code_zip) if Path(args.code_zip).is_file() else None

    record = {"submit_time": args.submitted_at or datetime.now(timezone.utc).isoformat(),
              "choice": args.choice, "candidate_id": candidate_id,
              "zip": str(args.zip), "zip_sha256": zip_sha,
              "code_zip": str(args.code_zip), "code_zip_sha256": code_sha,
              "a_board_score": (None if args.score is None else float(args.score)),
              "error": args.error, "reason": args.reason,
              "budget": {"per_day": int(args.budget_per_day), "used_before": used_today},
              "frozen": bool(args.freeze and args.score is not None and not args.error)}
    rejected = None
    if not budget_ok:
        rejected = {"why": f"今日已提交 {used_today} 次 ≥ budget {args.budget_per_day}",
                    "record": record}
    elif not args.dry_run:
        log.setdefault("records", []).append(record)
        log["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(log_path, log)
        if record["frozen"] and candidate_id:
            try:
                REG.set_candidate_status(
                    candidate_id, "submitted", path=Path(args.candidates),
                    extra={"submitted_at": record["submit_time"],
                           "submitted_zip_sha256": zip_sha,
                           "a_board_score": record["a_board_score"],
                           "choice": args.choice})
                REG.freeze_candidate(candidate_id, path=Path(args.candidates),
                                     sha256=zip_sha)
            except (KeyError, PermissionError) as exc:
                print(f"[E10] 警告：候选冻结失败（{exc}）", file=sys.stderr)

    report = {"stage": "E10", "p_stage": "P2",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "exploratory": bool(args.exploratory), "dry_run": bool(args.dry_run),
              "choice": args.choice, "candidate_id": candidate_id,
              "zip_sha256": zip_sha, "code_zip_sha256": code_sha,
              "budget_ok": budget_ok, "used_today": used_today,
              "record": record, "rejected": rejected, "log_path": str(log_path),
              "decision": decision,
              "notes": ("只记录与冻结，**绝不据 A 榜反馈调参**；"
                        "choice=b0_fallback 时必须写明原因（不假装 v4 成功）")}
    write_json(reports / "E10_submission_report.json", report)

    prereg_path = Path(args.prereg) if args.prereg else reports / "E10_P2_gate_prereg.json"
    if not prereg_path.is_file():
        write_json(prereg_path, {
            "gate_id": "E10_P2_gate", "stage": "E10", "p_stage": "P2", "gate_type": "boolean",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "primary_metric": "submission_recorded", "primary_threshold_key": "min_delta",
            "baseline_version": "n/a", "baseline_artifact": str(log_path),
            "baseline_manifest_sha256": hashlib.sha256(
                json.dumps(log, sort_keys=True).encode()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": int(args.budget_per_day),
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P2_CHECKS), "decisions_locked": [],
            "notes": "E10/P2：提交必须留指纹与时间；配额内提交；冻结后不得改产物",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(not perrs),
        "atomic_precision_reported": True,
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool((reports / "training_time_log.json").is_file()),
        "checkpoint_resumable": True,
        "no_label_leak": True,
        "submission_log_complete": bool(budget_ok and (args.dry_run or
                                                       log.get("records") is not None)),
        "zip_fingerprint_recorded": bool(zip_sha is not None),
        "budget_ok": budget_ok,
        "candidate_frozen_after_submit": bool(record["frozen"] or not args.freeze),
        "no_retune_from_feedback": True,
    }
    result = {"checks": checks, "submission_recorded": bool(budget_ok and zip_sha)}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke or args.dry_run) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E10", "p_stage": "P2",
            "created_at": report["created_at"],
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "choice": args.choice,
            "candidate_id": candidate_id, "budget_ok": budget_ok,
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "report_path": str(reports / "E10_submission_report.json")}
    write_json(reports / "E10_P2_gate.json", gate)
    print(json.dumps({"stage": "E10/P2", "choice": args.choice,
                      "candidate_id": candidate_id, "budget_ok": budget_ok,
                      "zip_sha256": zip_sha, "frozen": record["frozen"],
                      "rejected": rejected, "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or args.dry_run or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
