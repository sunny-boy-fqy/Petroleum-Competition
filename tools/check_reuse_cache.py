#!/usr/bin/env python3
"""检查 /data 上是否已经缓存了 E0/E1/E2 的可复用成果。

用途：新开发机/新任务启动前，快速确认是否会复用，而不是盲跑后才发现 cache 不在。

检查项
------
- E0: raw 分片 cache（/data/v4/artifacts/cache/raw）+ E0 报告
- E1: OOF / checkpoint（/data/v4/mirror/run_root/E1）+ E1 报告
- E2: 特征 cache（/data/v4/artifacts/cache/feat）+ E2 消融/最佳 spec 报告
- 进度: /data/v4/state/all_pipeline_progress.json 中 task 1..5 状态

用法::

    python3 tools/check_reuse_cache.py --remote-root /data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern)) if root.is_dir() else 0


def _dir_bytes(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _fmt_bytes(n: int) -> str:
    v = float(n)
    for u in ("B", "KiB", "MiB", "GiB"):
        if v < 1024 or u == "GiB":
            return f"{v:.2f} {u}"
        v /= 1024
    return f"{n} B"


def _mark(ok: bool) -> str:
    return "OK  " if ok else "MISS"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查 /data 上的 E0/E1/E2 复用缓存")
    ap.add_argument("--remote-root", default=os.environ.get("V4_NETWORK_ROOT") or "/data")
    args = ap.parse_args(argv)
    root = Path(args.remote_root).expanduser().resolve()

    artifact = root / "v4" / "artifacts"
    mirror = root / "v4" / "mirror"
    state = root / "v4" / "state"

    checks: list[tuple[str, bool, str]] = []

    # E0 raw cache
    raw_train = artifact / "cache" / "raw" / "train"
    raw_test = artifact / "cache" / "raw" / "test"
    ntr, nte = _count(raw_train, "*.npz"), _count(raw_test, "*.npz")
    checks.append(("E0 / raw cache train", ntr >= 80, f"{raw_train}  npz={ntr} (expect 80)"))
    checks.append(("E0 / raw cache test", nte >= 10, f"{raw_test}  npz={nte} (expect 10)"))
    for rel in ("reports/E0_data_card.json", "reports/E0_folds.json",
                "reports/E0_local_contract_gate.json"):
        p = mirror / rel
        checks.append((f"E0 / {Path(rel).name}", p.is_file(), str(p)))

    # E1 OOF / checkpoints / report
    e1_oof = mirror / "run_root" / "E1" / "oof.npz"
    e1_pt = _count(mirror / "run_root" / "E1", "*.pt")
    checks.append(("E1 / oof.npz", e1_oof.is_file(), str(e1_oof)))
    checks.append(("E1 / checkpoints", e1_pt > 0, f"pt={e1_pt} under {mirror/'run_root'/'E1'}"))
    e1_report = mirror / "reports" / "E1_gate.json"
    checks.append(("E1 / E1_gate.json", e1_report.is_file(), str(e1_report)))

    # E2 feature cache / reports
    feat = artifact / "cache" / "feat"
    n_specs = sum(1 for p in feat.iterdir() if p.is_dir()) if feat.is_dir() else 0
    n_feat = _count(feat, "*.npz")
    checks.append(("E2 / feature cache", n_feat > 0 and n_specs > 0,
                   f"{feat}  specs={n_specs} npz={n_feat} bytes={_fmt_bytes(_dir_bytes(feat))}"))
    for rel in ("reports/E2_ablation.json", "reports/E2_gate.json",
                "reports/E2_best_spec.json"):
        p = mirror / rel
        checks.append((f"E2 / {Path(rel).name}", p.is_file(), str(p)))
    e2_work = mirror / "reports" / "E2_work"
    oofs = sorted(e2_work.glob("oof_*.npz")) if e2_work.is_dir() else []
    checks.append(("E2 / row OOF (E2_work)", bool(oofs),
                   f"{e2_work}  oof_files={len(oofs)}"))
    if oofs:
        checks.append(("E2 / best-spec row OOF", True, ", ".join(p.name for p in oofs[:4])))

    # progress
    prog = state / "all_pipeline_progress.json"
    if prog.is_file():
        try:
            doc = json.loads(prog.read_text(encoding="utf-8"))
            tasks = doc.get("tasks", {})
            done = [n for n in ("1", "2", "3", "4", "5")
                    if tasks.get(n, {}).get("status") == "done"]
            checks.append(("progress / task 1-5", len(done) == 5,
                           f"done={done}  file={prog}"))
        except Exception as exc:  # noqa: BLE001
            checks.append(("progress / parse", False, f"{prog}: {exc!r}"))
    else:
        checks.append(("progress / file", False, str(prog)))

    all_ok = True
    print(f"remote_root: {root}")
    print("-" * 88)
    for label, ok, detail in checks:
        all_ok = all_ok and ok
        print(f"[{_mark(ok)}] {label:34s} {detail}")
    print("-" * 88)
    print("RESULT:", "OK（E0/E1/E2 可复用缓存齐备）" if all_ok else
          "INCOMPLETE（存在缺失；新机器不会完整复用）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
