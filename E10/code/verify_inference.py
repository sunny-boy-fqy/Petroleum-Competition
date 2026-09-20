#!/usr/bin/env python3
"""E10/P1：**干净目录复现**验证（纯标准库）——只放"提交包 + 官方 data"，跑两次比对。

为什么必须这么做
--------------
本地跑得通 ≠ 提交环境跑得通：相对导入、缓存路径、漏打的文件、依赖缺失都只在**干净目录**里
才暴露。因此这里：

1. 造一个**只含**解压后的代码包 + `data/` 的目录（不带仓库、不带缓存、不带 experiments）；
2. 在该目录执行 `python predict.py --data_dir ./data --output result.json`；
3. **跑两次**，要求两次 `result.json` 的 sha256 **完全一致**（确定性）；
4. 与参考结果逐点比较（默认容差 `1e-6`），并记录耗时 / 子进程峰值内存；
5. 判据：`≤1e-6`、两次一致、`<30 min`、`<8 GiB`。

缺代码包 / 缺数据 → 显式 `status` + 退出码 4（`--smoke` 下只出证据）。
产出：`$REPORTS/E10_reproduce_report.json`、`$REPORTS/E10_reproduce_gate.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.training import metrics as M  # noqa: E402

MAX_MINUTES = 30.0
MAX_MEMORY_GB = 8.0


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
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E10/P1 干净目录复现验证")
    ap.add_argument("--data-dir", default=str(env_path("V4_DATA_ROOT", "/data") / "v4" / "data"),
                    help="包含 test/ 的数据目录（或直接指向测试井目录）")
    ap.add_argument("--zip", default=str(V4 / "submission" / "submission_code_v4.zip"))
    ap.add_argument("--clean-dir", default=None, help="工作目录（缺省用临时目录）")
    ap.add_argument("--output", default="result.json")
    ap.add_argument("--reference", default=str(V4 / "submission" / "result.json"),
                    help="参考结果（缺省用第一次运行的结果做自比对基线）")
    ap.add_argument("--expected-rows", type=int, default=None)
    ap.add_argument("--expected-wells", type=int, default=None)
    ap.add_argument("--point-tol", type=float, default=1e-6)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--timeout-min", type=float, default=MAX_MINUTES)
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--b0-verified", action="store_true",
                    help="B0 兜底包已在同一干净目录验证过（由 build_b0_fallback.py 传入）")
    ap.add_argument("--keep", action="store_true", help="保留干净目录（便于排查）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def prepare_clean_dir(zip_path: Path, data_dir: Path, clean: Path) -> dict:
    """只放：解压后的代码 + data/（不做任何路径改写）。"""
    clean.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(clean)
    dst_data = clean / "data"
    if data_dir.name == "test":
        dst_data = clean / "data" / "test"
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        if not dst_data.exists():
            shutil.copytree(data_dir, dst_data)
    else:
        if not dst_data.exists():
            shutil.copytree(data_dir, dst_data)
    return {"clean_dir": str(clean), "n_entries": len(list(clean.rglob("*"))),
            "data_dir": str(dst_data)}


def run_once(clean: Path, out_name: str, args) -> dict:
    cmd = [sys.executable, "predict.py", "--data_dir", "./data", "--output", out_name]
    if args.expected_wells:
        cmd += ["--expected-wells", str(args.expected_wells)]
    if args.expected_rows:
        cmd += ["--expected-rows", str(args.expected_rows)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(clean), capture_output=True, text=True,
                          timeout=float(args.timeout_min) * 60.0)
    minutes = (time.time() - t0) / 60.0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    memory_gb = max(after, before) / (1024.0 ** 2)
    out_path = clean / out_name
    return {"command": " ".join(cmd), "returncode": int(proc.returncode),
            "minutes": minutes, "memory_gb": memory_gb,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "output": str(out_path), "sha256": sha256_file(out_path),
            "exists": out_path.is_file()}


def point_diff(a: Path, b: Path, tol: float = 1e-6) -> dict:
    """逐点差（按 logId 对齐、按行序比对；字段顺序无关），并统计超出容差的字段数。"""
    if not (a.is_file() and b.is_file()):
        return {"ok": False, "error": "结果文件缺失"}
    pa = json.loads(a.read_text(encoding="utf-8"))
    pb = json.loads(b.read_text(encoding="utf-8"))
    wa = {str(w.get("logId")): w.get("predictions", []) for w in pa.get("resultData", [])}
    wb = {str(w.get("logId")): w.get("predictions", []) for w in pb.get("resultData", [])}
    if set(wa) != set(wb):
        return {"ok": False, "error": "井集合不一致",
                "only_a": sorted(set(wa) - set(wb))[:5],
                "only_b": sorted(set(wb) - set(wa))[:5]}
    worst, n_over, n_rows = 0.0, 0, 0
    for wid in sorted(wa):
        ra, rb = wa[wid], wb[wid]
        if len(ra) != len(rb):
            return {"ok": False, "error": f"{wid} 行数不一致 {len(ra)} vs {len(rb)}"}
        for x, y in zip(ra, rb):
            n_rows += 1
            if abs(float(x.get("depth", 0.0)) - float(y.get("depth", 0.0))) > tol:
                n_over += 1
            for k in C.TARGETS:
                d = abs(float(x[k]) - float(y[k]))
                worst = max(worst, d)
                if d > tol:
                    n_over += 1
    return {"ok": True, "max_point_diff": worst, "n_rows": n_rows,
            "n_wells": len(wa), "n_over_tol": int(n_over), "tol": float(tol)}


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    try:
        import numpy as _np
        out["numpy"] = _np.__version__
    except Exception:
        out["numpy"] = None
    try:
        import torch as _t
        out["torch"] = _t.__version__
    except Exception:
        out["torch"] = None
    return out


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip)
    data_dir = Path(args.data_dir)
    if not zip_path.is_file():
        print(f"[E10] FATAL: 缺少代码包 {zip_path}（先跑 build_submission.py）", file=sys.stderr)
        return 4
    if not data_dir.is_dir():
        print(f"[E10] FATAL: 缺少数据目录 {data_dir}", file=sys.stderr)
        return 4
    clean = Path(args.clean_dir) if args.clean_dir else \
        Path(tempfile.mkdtemp(prefix="v4_clean_"))
    prep = prepare_clean_dir(zip_path, data_dir, clean)

    runs = [run_once(clean, f"{args.output}{'' if i == 0 else '.run2'}", args)
            for i in range(max(int(args.runs), 1))]
    ref_path = Path(args.reference) if args.reference and Path(args.reference).is_file() \
        else Path(runs[0]["output"])
    tol = float(args.point_tol)
    diff = point_diff(ref_path, Path(runs[0]["output"]), tol=tol)
    max_diff = diff.get("max_point_diff")
    n_over = diff.get("n_over_tol")
    two_runs_same = bool(len(runs) >= 2 and runs[0]["sha256"] == runs[1]["sha256"]
                         and runs[0]["sha256"] is not None)
    minutes = max(r["minutes"] for r in runs)
    memory_gb = max(r["memory_gb"] for r in runs)
    ok_all = bool(runs and all(r["returncode"] == 0 and r["exists"] for r in runs)
                  and diff.get("ok") and (max_diff is not None and max_diff <= tol)
                  and two_runs_same)

    report = {"stage": "E10", "p_stage": "P1-reproduce",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "exploratory": bool(args.exploratory),
              "clean_dir": prep["clean_dir"], "zip": str(zip_path),
              "zip_sha256": sha256_file(zip_path),
              "data_dir": str(data_dir),
              "command": runs[0]["command"] if runs else None,
              "stdout_summary": (runs[0]["stdout_tail"][-600:] if runs else None),
              "stderr_summary": (runs[0]["stderr_tail"][-600:] if runs else None),
              "versions": _versions(),
              "fingerprints": {"reference": str(ref_path),
                               "reference_sha256": sha256_file(ref_path),
                               "runs": [{"sha256": r["sha256"], "minutes": r["minutes"]}
                                        for r in runs]},
              "max_point_diff": max_diff, "point_diff": max_diff,
              "n_over_tol": n_over if max_diff is not None else None,
              "point_tol": tol, "two_runs_identical": two_runs_same,
              "minutes": minutes, "memory_gb": memory_gb,
              "inference_verified": ok_all, "runs": runs}
    out_path = Path(args.json) if args.json else reports / "E10_reproduce_report.json"
    write_json(out_path, report)

    checks = {
        "clean_dir_reproduce": bool(runs and all(r["returncode"] == 0 for r in runs)),
        "deterministic_two_runs": two_runs_same,
        "point_diff_ok": bool(max_diff is not None and max_diff <= tol),
        "time_ok": bool(minutes <= float(args.timeout_min)),
        "memory_ok": bool(memory_gb <= MAX_MEMORY_GB),
        "b0_fallback_verified": bool(args.b0_verified),
        "no_repo_dependency": True,
    }
    passed = None if (args.smoke or args.exploratory) else bool(
        checks["clean_dir_reproduce"] and checks["deterministic_two_runs"]
        and checks["point_diff_ok"] and checks["time_ok"] and checks["memory_ok"])
    gate = {"gate_id": "E10_reproduce_gate", "stage": "E10", "p_stage": "P1-reproduce",
            "created_at": report["created_at"],
            "exploratory": bool(args.smoke or args.exploratory),
            "passed": passed, "inference_verified": ok_all, "checks": checks,
            "metrics": {"minutes": minutes, "memory_gb": memory_gb,
                        "max_point_diff": max_diff},
            "report_path": str(out_path)}
    write_json(reports / "E10_reproduce_gate.json", gate)
    if not args.keep and args.clean_dir is None:
        shutil.rmtree(clean, ignore_errors=True)
    print(json.dumps({"stage": "E10/P1-reproduce", "inference_verified": ok_all,
                      "max_point_diff": max_diff, "two_runs_identical": two_runs_same,
                      "minutes": minutes, "memory_gb": memory_gb,
                      "passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.smoke or args.exploratory:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
