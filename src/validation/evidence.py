"""跨阶段 Gate 证据读取的唯一入口（H5 审查修复）。

背景
----
E9/E10 的 Gate mandatory_checks 里曾出现 `x or True`、`..._reported=True`、
`disk_budget_ok=True` 等硬编码/恒真值，导致 Gate 无法发现泄漏、续训失效或磁盘不足。
本模块把"读证据文件并做真实判定"集中起来：拿不到证据 = False（不是 True），
证据文件存在但校验不过也 = False。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def _report(reports_dir: str | Path, name: str) -> tuple[Path, dict[str, Any] | None]:
    p = Path(reports_dir) / name
    return p, load_json(p)


def leakage_audit_ok(reports_dir: str | Path) -> tuple[bool, dict[str, Any]]:
    """E9 泄漏审计的真实结论：文件必须存在且 `high_risk_leak is False`。"""
    p, d = _report(reports_dir, "E9_leakage_audit.json")
    ev: dict[str, Any] = {"path": str(p), "present": d is not None}
    if d is None:
        return False, {**ev, "reason": "缺少 E9_leakage_audit.json"}
    verdict = str(d.get("verdict", ""))
    high = d.get("high_risk_leak")
    ok = high is False and verdict in ("pass", "pass_with_residual_risk")
    return bool(ok), {**ev, "verdict": verdict, "high_risk_leak": high,
                      "n_fail": d.get("n_fail"), "n_residual_risk": d.get("n_residual_risk")}


def atomic_precision_reported(reports_dir: str | Path) -> tuple[bool, dict[str, Any]]:
    """E6 原子指标报告：必须有 `atom_metrics_tau_half` 且逐目标字段齐全。"""
    p, d = _report(reports_dir, "E6_atomic_report.json")
    ev: dict[str, Any] = {"path": str(p), "present": d is not None}
    if d is None:
        return False, {**ev, "reason": "缺少 E6_atomic_report.json"}
    m = d.get("atom_metrics_tau_half")
    ok = bool(isinstance(m, dict) and m
              and all(k in m for k in ("POR", "PERM", "SW")))
    return bool(ok), {**ev, "keys": (sorted(m) if isinstance(m, dict) else None)}


def checkpoint_resumable(reports_dir: str | Path) -> tuple[bool, dict[str, Any]]:
    """E6 报告的逐折 `resumable.ok` 必须全为真；空 folds 视为未验证。"""
    p, d = _report(reports_dir, "E6_atomic_report.json")
    ev: dict[str, Any] = {"path": str(p), "present": d is not None}
    if d is None:
        return False, {**ev, "reason": "缺少 E6_atomic_report.json"}
    folds = d.get("folds_detail") or []
    if not folds:
        return False, {**ev, "reason": "E6_atomic_report.folds_detail 为空"}
    bad = [int(r.get("fold", -1)) for r in folds
           if not (isinstance(r, dict) and (r.get("resumable") or {}).get("ok") is True)]
    return bool(not bad), {**ev, "n_folds": len(folds), "bad_folds": bad}


def training_time_log_valid(reports_dir: str | Path) -> tuple[bool, dict[str, Any]]:
    """`training_time_log.json::valid` 必须显式为真，而不是仅文件存在。"""
    p, d = _report(reports_dir, "training_time_log.json")
    ev: dict[str, Any] = {"path": str(p), "present": d is not None}
    if d is None:
        return False, {**ev, "reason": "缺少 training_time_log.json"}
    ok = d.get("valid") is True
    return bool(ok), {**ev, "valid": d.get("valid"), "folds": len(d.get("folds") or [])}


def disk_budget_ok(level: str | None) -> bool:
    """磁盘证据的唯一判定：只有 `level == "ok"` 才算通过。"""
    return str(level or "").lower() == "ok"
