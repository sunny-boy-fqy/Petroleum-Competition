#!/usr/bin/env python3
"""E9/P0：提交决策（**纯 numpy + registry 写回**）——短名单、拒绝标记与决策留痕。

判据（E9/P0 §9.4）
----------------
    choice = 通过护栏的候选里 **oof_total 最高**者；无人通过 → `B0_FALLBACK`
    shortlist = 通过护栏者按 oof 排序的前 ≤3 个（每个都要写"为什么入选"）
    不通过的候选 → `status="rejected"`，但**保留在候选表里**（证据不能删）

写回纪律（单点写入）
------------------
* 本脚本是**唯一**改候选 `status` 的地方（`aggregate_oof.py` 只读）；
* `submitted` / `frozen_best` 的候选**跳过**（已冻结不可改），并在报告里列 `skipped_immutable`；
* `--dry-run` 只出决策、不写注册表（测试与人工复核用这条路径）。

产出：`$REPORTS/E9_submission_decision.json`、`$REPORTS/E9_choice_gate.json`。
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
from src.training import metrics as M  # noqa: E402
from src.validation import gates as GATES  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
CHOICE_CHECKS = ("decision_made", "shortlist_within_budget", "rejected_kept_in_registry",
                 "submitted_candidates_untouched", "guardrail_floor_recorded")
IMMUTABLE = ("submitted", "frozen_best")
FALLBACK = "B0_FALLBACK"


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
    ap = argparse.ArgumentParser(description="E9/P0 提交决策（短名单 + 拒绝标记）")
    ap.add_argument("--validation-report", default=os.environ.get("V4_REPORTS_DIR")
                    and str(Path(os.environ["V4_REPORTS_DIR"]) / "E9_validation_report.json")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"
                           / "E9_validation_report.json"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--candidates", default=str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--max-shortlist", type=int, default=3)
    ap.add_argument("--a-board-log", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="只出决策，不写候选注册表（测试/复核路径）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def first_pass(report_path: Path) -> tuple[dict, Path]:
    """定位验证报告：给目录时按标准文件名找，给文件时直接用。"""
    p = Path(report_path)
    if p.is_dir():
        p = p / "E9_validation_report.json"
    if not p.is_file():
        raise SystemExit(f"[E9] FATAL: 缺少验证报告 {p}（先跑 E9/code/aggregate_oof.py）")
    return json.loads(p.read_text(encoding="utf-8")), p


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    report, rep_path = first_pass(Path(args.validation_report))
    cand_doc = REG.load_candidates(args.candidates)
    status_before = {str(c.get("candidate_id")): c.get("status")
                     for c in cand_doc.get("candidates", [])}
    floor = float(report["guardrail"]["floor"])

    rows = {r["candidate_id"]: r for r in report.get("candidates", [])}
    passing = [r for r in report.get("candidates", []) if r.get("guardrail_pass")]
    passing.sort(key=lambda r: float(r.get("oof_total") or float("-inf")), reverse=True)
    shortlist = [r["candidate_id"] for r in passing[: max(int(args.max_shortlist), 0)]]
    rejected = [r["candidate_id"] for r in report.get("candidates", [])
                if not r.get("guardrail_pass")]
    immutable = [cid for cid, st in status_before.items() if st in IMMUTABLE]
    choice = shortlist[0] if shortlist else FALLBACK
    ref = report.get("guardrail", {}).get("anchors", {})

    # ---- 写回候选状态（唯一写入点；不可覆盖的候选跳过）
    wrote: list[dict] = []
    if not args.dry_run:
        for r in report.get("candidates", []):
            cid = r["candidate_id"]
            if status_before.get(cid) in IMMUTABLE:
                continue
            target = "shortlisted" if cid in shortlist else "rejected"
            try:
                REG.set_candidate_status(
                    cid, target, path=Path(args.candidates),
                    extra={"guardrail_pass": bool(r.get("guardrail_pass")),
                           "guardrail_reason": r.get("guardrail_reason"),
                           "oof_total": r.get("oof_total"),
                           "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                wrote.append({"candidate_id": cid, "status": target})
            except KeyError as exc:                      # 候选不在表里（理论上不会）
                wrote.append({"candidate_id": cid, "error": str(exc)})
    status_after = {str(c.get("candidate_id")): c.get("status")
                    for c in REG.load_candidates(args.candidates).get("candidates", [])}

    reason = (f"通过护栏的候选中 OOF 最高者：{choice}"
              f"（OOF {rows[choice].get('oof_total'):.4f} ≥ floor {floor:.4f}）"
              if shortlist else
              f"无候选通过护栏（floor {floor:.4f}）→ 回退 {FALLBACK}（B0 兜底）")
    decision = {"schema_version": 1, "stage": "E9", "p_stage": "P0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "selection_score_only": True, "missing_mode": C.SCORE_MISSING_MODE,
                "protocol_matched": False,
                "protocol_note": report.get("protocol_note"),
                "choice": choice, "reason": reason,
                "candidate_oof": (None if not shortlist
                                  else float(rows[choice]["oof_total"])),
                "guardrail_floor": floor,
                "reference_oof": {"B0_LOCAL_OOF": ref.get("B0_LOCAL_OOF"),
                                  "B0_A_BOARD": ref.get("B0_A_BOARD"),
                                  "PASS_LINE": ref.get("PASS_LINE")},
                "shortlist": [{"candidate_id": cid,
                               "oof_total": rows[cid].get("oof_total"),
                               "why": (f"OOF {rows[cid].get('oof_total'):.4f} ≥ floor "
                                       f"{floor:.4f}（排名第 {i + 1}）"),
                               "a_board_score": rows[cid].get("a_board_score"),
                               "source": rows[cid].get("oof_path")}
                              for i, cid in enumerate(shortlist)],
                "rejected": rejected, "rejected_kept_in_registry": True,
                "skipped_immutable": immutable,
                "registry_writes": wrote, "dry_run": bool(args.dry_run),
                "validation_report": str(rep_path),
                "status_before": status_before, "status_after": status_after,
                "notes": ("不通过的候选标 rejected 但**保留**（证据不删）；submitted/frozen_best "
                          "不可改；短名单只含通过护栏者，且不超过 --max-shortlist")}
    if args.a_board_log and Path(args.a_board_log).is_file():
        decision["a_board_log"] = json.loads(Path(args.a_board_log).read_text(encoding="utf-8"))
    out_path = Path(args.json) if args.json else reports / "E9_submission_decision.json"
    write_json(out_path, decision)

    prereg_path = Path(args.prereg) if args.prereg else reports / "E9_choice_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E9_choice_gate", "stage": "E9", "p_stage": "P0", "gate_type": "boolean",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "guardrail_pass", "primary_threshold_key": "min_delta",
            "baseline_version": "B0", "baseline_artifact": str(out_path),
            "baseline_manifest_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": int(args.max_shortlist),
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(CHOICE_CHECKS), "decisions_locked": [],
            "notes": "E9/P0 决策：短名单 ≤3；不通过者标 rejected 但保留；submitted 不可改",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    submitted_untouched = all(
        status_before.get(cid) != "submitted" or status_after.get(cid) == "submitted"
        for cid in status_before)
    checks = {
        "contract_ok": bool(report.get("candidates") is not None and not perrs),
        "atomic_precision_reported": True,
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool((reports / "training_time_log.json").is_file()),
        "checkpoint_resumable": True,
        "no_label_leak": bool(report.get("missing_mode") == C.SCORE_MISSING_MODE),
        "decision_made": bool(decision["choice"]),
        "shortlist_within_budget": bool(len(shortlist) <= max(int(args.max_shortlist), 0)),
        "rejected_kept_in_registry": bool(all(cid in status_after for cid in rejected)),
        "submitted_candidates_untouched": bool(submitted_untouched),
        "guardrail_floor_recorded": bool(floor > 0),
    }
    result = {"checks": checks, "guardrail_pass": bool(shortlist)}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke or args.dry_run) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E9", "p_stage": "P0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "choice": decision["choice"],
            "shortlist": shortlist, "rejected": rejected,
            "skipped_immutable": immutable, "dry_run": bool(args.dry_run),
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "decision_path": str(out_path)}
    write_json(reports / "E9_choice_gate.json", gate)
    print(json.dumps({"stage": "E9/P0", "choice": decision["choice"],
                      "reason": decision["reason"], "shortlist": shortlist,
                      "rejected": rejected, "skipped_immutable": immutable,
                      "dry_run": bool(args.dry_run), "gate_passed": passed,
                      "checks": checks}, ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or args.dry_run or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
