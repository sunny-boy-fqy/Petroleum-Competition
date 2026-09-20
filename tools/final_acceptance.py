#!/usr/bin/env python3
"""最终验收批（**纯标准库**）：一次跑完本地能跑的全部证据，汇总成一份 JSON。

包含的检查
---------
1. **两套解释器全量测试**：`../v2/.venv/bin/python tests/run_all.py`（口径层，torch 用例 skip）
   与 `./.venv-torch/bin/python tests/run_all.py`（含 torch 用例）——解析"运行 N 项"与失败/错误数；
2. **四个 checker**：`check_consistency` / `check_status` / `check_pipeline` / `audit_goal`；
3. **参考数据校验**：`tools/verify_reference.py`（tarball sha / 井数 / 行数 / 非规范井）；
4. **计划行数一致性**：`tools/sync_plan_stats.py --dry-run`（若支持）或 `tools/plan_stats.py`；
5. **契约 Gate 复算**：用 `src/validation/gates.aggregate_gate` 复算已提交的
   `E0_gate_prereg.json` + `E0_local_contract_gate.json`（不可复算的 Gate 不算过）。

无法在本地完成的（数据集/Ascend）一律记入 `cloud_pending`，**不**算失败也不假装通过。

用法::

    python3 tools/final_acceptance.py --json reports/E0_final_acceptance.json
    python3 tools/final_acceptance.py --quick         # 只跑 checker + 契约复算（秒级）
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

PY_CANDIDATES = (V4.parent / "v2" / ".venv" / "bin" / "python",
                 V4 / ".venv-torch" / "bin" / "python")
CHECKERS = ("tools/check_consistency.py", "tools/check_status.py", "tools/check_pipeline.py",
            "tools/audit_goal.py")
CLOUD_PENDING = (
    "80 井 5 折正式 Gate（E1/E3/E5/E6/E7/E8）",
    "A 榜首轮成绩与配额（E9/P2）",
    "官方 10 井 / 95,948 行干净目录复现（E10/P1）",
    "B0 兜底包当场复现（E10/P1，需官方数据）",
    "全量重训 / 折集成权重导出（E10/P0，需全部井）",
)


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 3600.0) -> dict:
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, timeout=timeout)
    return {"cmd": " ".join(str(c) for c in cmd), "returncode": int(proc.returncode),
            "seconds": round(time.time() - t0, 2),
            "stdout_full": proc.stdout or "", "stderr_tail": (proc.stderr or "")[-800:]}


def parse_suite(out: str) -> dict:
    """解析 `run_all.py` 的总结行：`运行 N 项，失败 x，错误 y，跳过 s`。"""
    m = re.search(r"运行\s*(\d+)\s*项，失败\s*(\d+)，错误\s*(\d+)，跳过\s*(\d+)", out)
    if not m:
        return {"parsed": False}
    return {"parsed": True, "ran": int(m.group(1)), "failed": int(m.group(2)),
            "errors": int(m.group(3)), "skipped": int(m.group(4))}


def interpreters() -> list[tuple[str, Path]]:
    out = []
    for p in PY_CANDIDATES:
        if p.is_file():
            out.append((p.parent.parent.name, p))
    seen, uniq = set(), []
    for name, p in out:
        if str(p) in seen:
            continue
        seen.add(str(p))
        uniq.append((name or "python", p))
    return uniq or [("python3", Path(shutil.which("python3") or sys.executable))]


def contract_recompute() -> dict:
    """复算已提交的 E0 本地契约 Gate（同一校验器，两处调用）。"""
    from src.validation import gates as G
    prereg_p = V4 / "reports" / "E0_gate_prereg.json"
    report_p = V4 / "reports" / "E0_local_contract_gate.json"
    if not (prereg_p.is_file() and report_p.is_file()):
        return {"ok": False, "reason": "缺少 E0 prereg / 报告"}
    prereg = json.loads(prereg_p.read_text(encoding="utf-8"))
    rep = json.loads(report_p.read_text(encoding="utf-8"))
    checks = rep.get("mandatory_checks") or rep.get("checks")
    result = {"checks": checks, "abs_diff": rep.get("abs_diff")}
    out = G.aggregate_gate(prereg, result)
    return {"ok": bool(out.get("passed")), "prereg": str(prereg_p), "report": str(report_p),
            "aggregate": {k: out.get(k) for k in ("passed", "mandatory_failures", "details")}}


def run(quick: bool = False, runs_dir: Path | None = None) -> dict:
    checks: dict[str, dict] = {}
    for rel in CHECKERS:
        checks[Path(rel).stem] = _run([sys.executable, V4 / rel], cwd=V4)
    checks["contract_recompute"] = {"returncode": 0 if (cr := contract_recompute())["ok"] else 1,
                                    "detail": cr, "seconds": 0.0}
    checks["verify_reference"] = _run([sys.executable, V4 / "tools" / "verify_reference.py"], cwd=V4)
    checks["plan_stats"] = _run([sys.executable, V4 / "tools" / "plan_stats.py",
                                 "--json", "/tmp/v4_plan_stats_acceptance.json"], cwd=V4)
    if not quick:
        for name, py in interpreters():
            r = _run([py, V4 / "tests" / "run_all.py"], cwd=V4, timeout=5400.0)
            r["summary"] = parse_suite(r["stdout_full"])
            checks[f"tests_{name}"] = r
    local_ok = all(v["returncode"] == 0 for v in checks.values())
    suites_ok = all(v.get("summary", {}).get("failed", 0) == 0
                    and v.get("summary", {}).get("errors", 0) == 0
                    and v["returncode"] == 0
                    for k, v in checks.items() if k.startswith("tests_")) if not quick else True
    return {"ok": bool(local_ok and suites_ok), "quick": bool(quick),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "interpreters": {name: str(p) for name, p in interpreters()},
            "checks": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("stdout_full", "stderr_tail")}
                       for k, v in checks.items()},
            "cloud_pending": list(CLOUD_PENDING),
            "note": ("本地验收只覆盖契约层与静态检查；正式 Gate 数值 / A 榜 / 官方数据复现"
                     "属于云端任务（见 cloud_pending）")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="最终验收批")
    ap.add_argument("--quick", action="store_true", help="只跑 checker + 契约复算")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    res = run(quick=args.quick)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    brief = {k: v.get("returncode") for k, v in res["checks"].items()}
    print(json.dumps({"ok": res["ok"], "quick": res["quick"], "checks": brief,
                      "cloud_pending": len(res["cloud_pending"])},
                     ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
