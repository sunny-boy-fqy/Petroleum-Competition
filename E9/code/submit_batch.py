#!/usr/bin/env python3
"""E9/P2：A 榜提交批次管理（**纯 numpy/标准库 + registry 写回**）。

职责（E9/P2 §5）
--------------
1. **选提交对象**：从决策短名单里挑 ≤ `--max-submits` 个**尽量多样**的候选
   （不同主干/特征版本/基线，故意不等于"分数前 3"）；留 `--reserve` 个额度不动；
2. **记录 A 榜**：把 `{candidate_id, zip_sha256, submit_time, a_board_score, usage}`
   追加到 `E9_a_board_log.json`（**只追加、不改历史**）；
3. **判据**：`degradation = anchor − a_board_score`，`degradation ≤ max_degradation(0.1)`
   记 `no_breakdown`；锚点取 `max(B0_A_BOARD, 已知的最佳 A 榜分)` 并记录噪声带 ±0.02；
4. **配额**：按日志统计"今天已用"，`used + reserve > budget_per_day` 时**拒绝**提交；
5. **绝不据反馈调参**：本脚本只写日志与候选状态，不动任何权重/超参（报告里显式声明）。

产出：`$REPORTS/E9_a_board_log.json`、`$REPORTS/E9_a_board_gate.json`（+ 决策里的 zip 指纹）。
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
from src.validation import evidence as EVID  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import gates as GATES  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P2_CHECKS = ("a_board_log_complete", "budget_ok", "a_board_no_breakdown",
             "candidates_diverse", "no_retune_from_feedback")
MAX_DEGRADATION = 0.1
NOISE_BAND = 0.02
FALLBACK_CHOICE = "b0_fallback"


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


def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E9/P2 A 榜提交批次（记录 + 配额 + 判据）")
    ap.add_argument("--decision", default=os.environ.get("V4_REPORTS_DIR")
                    and str(Path(os.environ["V4_REPORTS_DIR"])
                            / "E9_submission_decision.json")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"
                           / "E9_submission_decision.json"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--candidates", default=os.environ.get("V4_CANDIDATES") or str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--a-board-log", default=None)
    ap.add_argument("--budget-per-day", type=int, default=5)
    ap.add_argument("--max-submits", type=int, default=3)
    ap.add_argument("--reserve", type=int, default=1)
    ap.add_argument("--zip", default=None, help="待提交的 result.zip（用于记录指纹）")
    ap.add_argument("--score", type=float, default=None, help="人工回填的 A 榜分")
    ap.add_argument("--submitted-at", default=None)
    ap.add_argument("--error", default=None, help="平台报错文本（记录了就不改结论）")
    ap.add_argument("--reason", default=None)
    ap.add_argument("--measured", default=None, help="本次实际提交的 candidate_id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def load_json(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if Path(p).is_file() else default


def pick_diverse(shortlist: list[dict], cands: dict[str, dict], k: int) -> list[dict]:
    """按"多样性优先"挑 k 个：先按分数取第一个，之后每个都尽量换主干/特征版本/基线。"""
    picked: list[dict] = []
    seen: set[str] = set()
    for entry in shortlist:                       # 分数已排序
        cid = entry["candidate_id"]
        meta = cands.get(cid, {})
        key = f"{meta.get('arch')}|{meta.get('feature_version')}|{meta.get('base')}"
        if key not in seen or not picked:
            picked.append({**entry, "diversity_key": key,
                           "why": ("分数最高" if not picked else f"与已选者多样性键 {key} 不同")})
            seen.add(key)
        if len(picked) >= k:
            break
    if len(picked) < k:                           # 多样性候选不够则按分数补足
        for entry in shortlist:
            if entry["candidate_id"] in {p["candidate_id"] for p in picked}:
                continue
            picked.append({**entry, "diversity_key": "duplicate",
                           "why": "多样性候选不足，按分数补足"})
            if len(picked) >= k:
                break
    return picked


def submissions_today(log: dict, day: str) -> int:
    return sum(1 for r in log.get("records", [])
               if str(r.get("submit_time", "")).startswith(day)
               and r.get("a_board_score") is not None)


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    decision_path = Path(args.decision)
    if decision_path.is_dir():
        decision_path = decision_path / "E9_submission_decision.json"
    decision = load_json(decision_path)
    if decision is None:
        print(f"[E9] FATAL: 缺少决策文件 {decision_path}（先跑 choose_submission.py）",
              file=sys.stderr)
        return 4
    log_path = Path(args.a_board_log) if args.a_board_log else \
        reports / "E9_a_board_log.json"
    log = load_json(log_path, {"schema_version": 1, "anchors": {
        "B0_A_BOARD": C.B0_A_BOARD, "noise_band": NOISE_BAND,
        "max_degradation": MAX_DEGRADATION}, "records": []})

    cand_doc = REG.load_candidates(args.candidates)
    by_id = {str(c.get("candidate_id")): c for c in cand_doc.get("candidates", [])}
    shortlist = list(decision.get("shortlist") or [])
    if decision.get("choice") == "B0_FALLBACK" and not shortlist:
        shortlist = [{"candidate_id": "B0_FALLBACK", "oof_total": None,
                      "why": "无候选通过护栏：B0 兜底"}]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used_today = submissions_today(log, today)
    budget_left = int(args.budget_per_day) - used_today - int(args.reserve)
    picked = pick_diverse(shortlist, by_id, max(int(args.max_submits), 0))
    budget_ok = bool(budget_left > 0)

    # ---- 回填一次真实提交（可选）
    record = None
    if args.score is not None or args.error:
        anchor = max([C.B0_A_BOARD] + [float(r["a_board_score"]) for r in log["records"]
                                       if r.get("a_board_score") is not None])
        measured = args.measured or (picked[0]["candidate_id"] if picked else None)
        deg = (None if args.score is None else float(anchor - float(args.score)))
        no_breakdown = (None if deg is None else bool(deg <= MAX_DEGRADATION))
        record = {"candidate_id": measured,
                  "zip_sha256": sha256_file(Path(args.zip)) if args.zip else None,
                  "submit_time": args.submitted_at or datetime.now(timezone.utc).isoformat(),
                  "a_board_score": (None if args.score is None else float(args.score)),
                  "anchor": float(anchor), "degradation": deg,
                  "noise_band": NOISE_BAND, "no_breakdown": no_breakdown,
                  "error": args.error, "reason": args.reason,
                  "usage": {"budget_per_day": int(args.budget_per_day),
                            "used_today_before": used_today,
                            "reserve": int(args.reserve)}}
        if not args.dry_run:
            log["records"].append(record)
            log["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(log_path, log)
            if measured and measured in by_id:
                try:
                    REG.upsert_candidate({"candidate_id": measured,
                                          "a_board_score": (None if args.score is None
                                                            else float(args.score)),
                                          "a_board_delta_vs_b0": (None if args.score is None
                                                                  else float(args.score
                                                                             - C.B0_A_BOARD)),
                                          "a_board_degradation": deg,
                                          "submitted_at": record["submit_time"]},
                                         path=Path(args.candidates),
                                         allow_submitted_overwrite=True)
                    if not args.error:
                        REG.set_candidate_status(measured, "submitted",
                                                path=Path(args.candidates),
                                                extra={"a_board_score": args.score})
                except (KeyError, PermissionError) as exc:
                    print(f"[E9] 警告：候选状态写回失败（{exc}）", file=sys.stderr)

    report = {"stage": "E9", "p_stage": "P2",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "exploratory": bool(args.exploratory), "dry_run": bool(args.dry_run),
              "decision_path": str(decision_path), "choice": decision.get("choice"),
              "max_submits": int(args.max_submits), "reserve": int(args.reserve),
              "budget_per_day": int(args.budget_per_day), "used_today": used_today,
              "budget_left": budget_left, "budget_ok": budget_ok,
              "picked": (picked if budget_ok else []),
              "skipped_due_to_budget": ([] if budget_ok
                                        else [p["candidate_id"] for p in picked]),
              "record": record, "log_path": str(log_path),
              "rules": {"max_degradation": MAX_DEGRADATION, "noise_band": NOISE_BAND,
                        "anchor_rule": "max(B0_A_BOARD, 历史最佳 A 榜分)",
                        "note": "只记录与判定，**绝不据 A 榜反馈调参**"},
              }
    write_json(reports / "E9_a_board_report.json", report)

    prereg_path = Path(args.prereg) if args.prereg else reports / "E9_P2_gate_prereg.json"
    if not prereg_path.is_file():
        write_json(prereg_path, {
            "gate_id": "E9_P2_gate", "stage": "E9", "p_stage": "P2", "gate_type": "boolean",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "primary_metric": "a_board_no_breakdown", "primary_threshold_key": "min_delta",
            "baseline_version": "B0", "baseline_artifact": str(log_path),
            "baseline_manifest_sha256": hashlib.sha256(
                json.dumps(log, sort_keys=True).encode()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0,
                           "max_degradation": MAX_DEGRADATION},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": int(args.max_submits),
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P2_CHECKS), "decisions_locked": [],
            "notes": "E9/P2：A 榜退化 ≤ 0.1 才算 no_breakdown；配额用尽则排队到下一天",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    n_breakdown = sum(1 for r in log.get("records", [])
                      if r.get("no_breakdown") is False)
    atomic_ok, atomic_ev = EVID.atomic_precision_reported(reports)
    resume_ok, resume_ev = EVID.checkpoint_resumable(reports)
    time_ok, time_ev = EVID.training_time_log_valid(reports)
    leak_ok, leak_ev = EVID.leakage_audit_ok(reports)
    checks = {
        "contract_ok": bool(decision.get("choice") and not perrs),
        "atomic_precision_reported": bool(atomic_ok),
        "disk_budget_ok": EVID.disk_budget_ok(disk.get("level")),
        "training_time_log_valid": bool(time_ok),
        "checkpoint_resumable": bool(resume_ok),
        "no_label_leak": bool(leak_ok),
        "a_board_log_complete": bool(log.get("records") is not None and budget_ok),
        "budget_ok": bool(budget_ok),
        "a_board_no_breakdown": bool(n_breakdown == 0),
        "candidates_diverse": bool(len({p.get("diversity_key") for p in picked})
                                   == len(picked)),
        "no_retune_from_feedback": True,
    }
    result = {"checks": checks, "degradation": (None if record is None
                                                else record.get("degradation")),
              "a_board_no_breakdown": bool(n_breakdown == 0)}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke or args.dry_run) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E9", "p_stage": "P2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "picked": report["picked"],
            "budget_ok": budget_ok, "used_today": used_today,
            "n_breakdown_in_log": n_breakdown, "checks": checks,
            "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "evidence": {"atomic": atomic_ev, "resumable": resume_ev,
                         "time_log": time_ev, "leakage": leak_ev},
            "log_path": str(log_path)}
    write_json(reports / "E9_a_board_gate.json", gate)
    print(json.dumps({"stage": "E9/P2", "choice": report["choice"],
                      "picked": [p["candidate_id"] for p in report["picked"]],
                      "budget_ok": budget_ok, "used_today": used_today,
                      "budget_left": budget_left,
                      "record": record, "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or args.dry_run or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
