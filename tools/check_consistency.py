#!/usr/bin/env python3
"""一致性校验：候选注册表与评分器口径自洽。

校验项：
  1. 每个候选 cv.total == 100*(0.30*por + 0.35*perm + 0.35*sw)（±1e-6）
  2. cv.missing_mode == "drop"（项目冻结口径）
  3. status 合法；submitted/frozen_best 必须有 result_zip_sha256
  4. 若 a_board_score 非空，必须同时有 a_board_delta_vs_b0

    python3 v4/tools/check_consistency.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))
from src import constants as C  # noqa: E402

VALID_STATUS = {"local_only", "shortlisted", "submitted", "frozen_best", "rejected"}


def main() -> int:
    d = json.loads((V4 / "versions" / "candidates.json").read_text(encoding="utf-8"))
    st = json.loads((V4 / "versions" / "status.json").read_text(encoding="utf-8"))
    errs: list[str] = []

    for cand in d.get("candidates", []):
        cid, cv = cand["candidate_id"], cand.get("cv") or {}
        if cand.get("status") not in VALID_STATUS:
            errs.append(f"{cid}: illegal status {cand.get('status')!r}")
        if "total" in cv:
            want = 100.0 * (C.SCORE_WEIGHTS["POR"] * cv["por"]
                            + C.SCORE_WEIGHTS["PERM"] * cv["perm"]
                            + C.SCORE_WEIGHTS["SW"] * cv["sw"])
            if abs(want - cv["total"]) > 1e-5:
                errs.append(f"{cid}: total {cv['total']} != weighted sum {want:.7f}")
        if cv.get("missing_mode") not in (None, "drop"):
            errs.append(f"{cid}: missing_mode must be 'drop', got {cv.get('missing_mode')!r}")
        if cand.get("status") in ("submitted", "frozen_best") and not cand.get("result_zip_sha256"):
            errs.append(f"{cid}: status={cand['status']} requires result_zip_sha256")
        if cand.get("a_board_score") is not None and cand.get("a_board_delta_vs_b0") is None:
            errs.append(f"{cid}: a_board_score set but a_board_delta_vs_b0 missing")

    # status.json 枚举
    valid_stage_status = {"pending", "in_progress", "done", "no_go", "blocked",
                          "done_local_pending_cloud_env"}
    for s_ in st.get("stages", []):
        if s_.get("status") not in valid_stage_status:
            errs.append(f"status.json {s_['stage']}: illegal status {s_.get('status')!r}")
        for sub in s_.get("p", []):
            if sub.get("status") not in {"pending", "in_progress", "done", "no_go", "blocked"}:
                errs.append(f"status.json {s_['stage']}/{sub.get('id')}: illegal status")
    if C.SCORE_MISSING_MODE != "drop":
        errs.append(f"constants.SCORE_MISSING_MODE must be 'drop', got {C.SCORE_MISSING_MODE!r}")

    for e in errs:
        print("FAIL:", e)
    print(f"候选数: {len(d.get('candidates', []))}  错误: {len(errs)}")
    print("RESULT:", "OK" if not errs else "FAIL")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
