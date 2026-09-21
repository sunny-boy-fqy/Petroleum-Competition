#!/usr/bin/env python3
"""E10/P1：B0 兜底包（**从 v1 冻结源码构建的自包含提交包**）——现在就建好，别等失败时才建。

为什么要提前做（E10/P1 §9.5）
---------------------------
v4 若在确认复验或 A 榜上出问题，兜底必须是**当场可用**的：因此本脚本
1. 从冻结的 v1 提交包（`v1/submission_e7/submission_code_e7.zip`，或已解压目录）**原样**取得
   `predict.py` + 模型文件；
2. 组装成**独立**包（不依赖 v4 的 `src/`、不依赖 `v1/` 目录树）；
3. 在**干净目录**（只放包 + `data/`）当场跑一次，与冻结的 `result.json` 逐点比较
   （默认 `1e-9`，比 v4 的 `1e-6` 更严：兜底必须与冻结结果一致）；
4. 写 `E10_B0_fallback_manifest.json`（包指纹、结果指纹、验证结论）。

缺 v1 冻结产物 → 显式 `status="missing_v1_sources"`（退出码 4，`--smoke` 下标 0），
**绝不**假装"兜底已就绪"。

产出：`submission/submission_code_b0_fallback.zip`、`$REPORTS/E10_B0_fallback.json`、
`$REPORTS/E10_B0_fallback_manifest.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
V1_ROOT = V4.parent / "v1"
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src.training import metrics as M  # noqa: E402

EXCLUDE_DIRS = ("__pycache__", ".venv", "data", "cache", "logs")
EXCLUDE_SUFFIX = (".log", ".pyc")


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
    ap = argparse.ArgumentParser(description="E10/P1 构建并验证 B0 兜底包")
    ap.add_argument("--v1-dir", default=str(V1_ROOT / "submission_e7"),
                    help="v1 冻结提交目录（含 submission_code_e7.zip 与 result.json）")
    ap.add_argument("--out", default=str(V4 / "submission" / "submission_code_b0_fallback.zip"))
    ap.add_argument("--frozen-result", default=None,
                    help="冻结的 v1 result.json（缺省取 v1-dir/result.json）")
    ap.add_argument("--data-dir", default=str(env_path("V4_DATA_ROOT", "/data") / "v4" / "data"),
                    help="包含 test/ 的数据目录（用于当场复现）")
    ap.add_argument("--point-tol", type=float, default=1e-9)
    ap.add_argument("--timeout-min", type=float, default=30.0)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def _included(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    return not rel.name.endswith(EXCLUDE_SUFFIX)


def collect_tree(root: Path) -> list[Path]:
    return [p for p in sorted(Path(root).rglob("*"))
            if p.is_file() and _included(p.relative_to(root))]


def assemble(v1_dir: Path, out_zip: Path, staging: Path) -> tuple[list[dict], dict]:
    """把 v1 冻结包原样解出并重新打包（自包含：不含 v4 的 src/）。"""
    src_zip = v1_dir / "submission_code_e7.zip"
    files: list[dict] = []
    if src_zip.is_file():
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(staging)
        origin = {"kind": "zip", "path": str(src_zip), "sha256": sha256_file(src_zip)}
    else:
        # 退化路径：v1-dir 已经是一个解压好的包目录
        for f in collect_tree(v1_dir):
            rel = f.relative_to(v1_dir)
            if rel.name in ("result.json", "result.zip", "submission_code_e7.zip"):
                continue
            tgt = staging / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, tgt)
        origin = {"kind": "dir", "path": str(v1_dir), "sha256": None}
    for f in collect_tree(staging):
        rel = f.relative_to(staging)
        files.append({"path": str(rel), "bytes": int(f.stat().st_size),
                      "sha256": sha256_file(f)})
    if files and not out_zip.exists():
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for e in files:
                zf.write(staging / e["path"], arcname=e["path"])
    return files, origin


def run_predict(clean: Path, args) -> dict:
    cmd = [sys.executable, "predict.py", "--data_dir", "./data", "--output", "result.json"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(clean), capture_output=True, text=True,
                          timeout=float(args.timeout_min) * 60.0)
    minutes = (time.time() - t0) / 60.0
    out = clean / "result.json"
    return {"command": " ".join(cmd), "returncode": int(proc.returncode),
            "minutes": minutes, "stdout_tail": (proc.stdout or "")[-1500:],
            "stderr_tail": (proc.stderr or "")[-1500:],
            "output": str(out), "sha256": sha256_file(out), "exists": out.is_file()}


def point_diff(a: Path, b: Path, tol: float) -> dict:
    if not (a.is_file() and b.is_file()):
        return {"ok": False, "error": "结果文件缺失"}
    pa, pb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    wa = {str(w.get("logId")): w.get("predictions", []) for w in pa.get("resultData", [])}
    wb = {str(w.get("logId")): w.get("predictions", []) for w in pb.get("resultData", [])}
    if set(wa) != set(wb):
        return {"ok": False, "error": "井集合不一致"}
    worst, over, n_rows = 0.0, 0, 0
    for wid in sorted(wa):
        ra, rb = wa[wid], wb[wid]
        if len(ra) != len(rb):
            return {"ok": False, "error": f"{wid} 行数不一致"}
        for x, y in zip(ra, rb):
            n_rows += 1
            for k in C.TARGETS:
                d = abs(float(x[k]) - float(y[k]))
                worst = max(worst, d)
                if d > tol:
                    over += 1
    return {"ok": True, "max_point_diff": worst, "n_over_tol": int(over),
            "n_rows": n_rows, "n_wells": len(wa), "tol": float(tol)}


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    v1_dir = Path(args.v1_dir)
    frozen = Path(args.frozen_result) if args.frozen_result else v1_dir / "result.json"
    if not (v1_dir / "submission_code_e7.zip").is_file() and not v1_dir.is_dir():
        status = {"status": "missing_v1_sources", "v1_dir": str(v1_dir),
                  "reason": "找不到 v1 冻结提交目录/包；兜底包无法构建（不假装就绪）"}
        write_json(reports / "E10_B0_fallback.json",
                   {**status, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "smoke": bool(args.smoke)})
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if args.smoke else 4
    clean = Path(args.clean_dir) if args.clean_dir else Path(tempfile.mkdtemp(prefix="v4_b0_"))
    staging = clean / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    out_zip = Path(args.out)
    if args.dry_run:
        print(f"[E10] dry-run：将从 {v1_dir} 构建 {out_zip}（未执行）")
        return 0
    files, origin = assemble(v1_dir, out_zip, staging)

    # 干净目录：只放包 + data/
    run_dir = clean / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with zipfile.ZipFile(out_zip) as zf:
        zf.extractall(run_dir)
    data_dir = Path(args.data_dir)
    if data_dir.is_dir():
        target = run_dir / "data"
        if data_dir.name == "test":
            target = run_dir / "data" / "test"
            target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copytree(data_dir, target)
    exec_result = run_predict(run_dir, args) if data_dir.is_dir() else \
        {"returncode": None, "exists": False, "reason": f"缺少数据目录 {data_dir}"}
    diff = point_diff(frozen, Path(exec_result["output"]), float(args.point_tol)) \
        if exec_result.get("exists") else {"ok": False, "error": "未产出 result.json"}
    max_diff = diff.get("max_point_diff")
    verified = bool(exec_result.get("returncode") == 0 and diff.get("ok")
                    and max_diff is not None and max_diff <= float(args.point_tol))

    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "v1_origin": origin, "frozen_result": str(frozen),
                "frozen_result_sha256": sha256_file(frozen),
                "package": str(out_zip), "package_sha256": sha256_file(out_zip),
                "n_files": len(files), "files": files,
                "executed": exec_result, "point_diff": diff,
                "point_tol": float(args.point_tol), "inference_verified": verified,
                "notes": ("B0 兜底包来自 v1 冻结源码，自包含（不依赖 v4 的 src/）；"
                          "验证容差比 v4 更严（1e-9）：必须与冻结 result.json 一致")}
    manifest_path = Path(args.manifest) if args.manifest else \
        reports / "E10_B0_fallback_manifest.json"
    write_json(manifest_path, manifest)
    report = {"stage": "E10", "p_stage": "P1-b0", "status": ("ok" if verified else "failed"),
              "created_at": manifest["created_at"], "exploratory": bool(args.exploratory),
              "package": str(out_zip), "package_sha256": manifest["package_sha256"],
              "frozen_result_sha256": manifest["frozen_result_sha256"],
              "inference_verified": verified, "point_diff": diff,
              "minutes": exec_result.get("minutes"),
              "manifest": str(manifest_path),
              "reason": (exec_result.get("reason") if not exec_result.get("exists")
                         else diff.get("error")),
              "checks": {"v1_sources_present": bool(origin.get("path") and (
                                 origin.get("sha256") or Path(origin["path"]).exists())),
                         "package_built": bool(out_zip.is_file()),
                         "clean_dir_reproduce": bool(exec_result.get("returncode") == 0),
                         "point_diff_ok": bool(max_diff is not None
                                               and max_diff <= float(args.point_tol)),
                         "fingerprints_recorded": bool(manifest["package_sha256"]),
                         "data_available": bool(data_dir.is_dir())}}
    write_json(reports / "E10_B0_fallback.json", report)
    write_json(reports / "E10_B0_gate.json",
               {"gate_id": "E10_B0_gate", "stage": "E10", "p_stage": "P1-b0",
                "created_at": report["created_at"],
                "exploratory": bool(args.smoke or args.exploratory),
                "passed": (None if (args.smoke or args.exploratory) else bool(verified)),
                "checks": report["checks"], "report_path": str(reports / "E10_B0_fallback.json")})
    print(json.dumps({"stage": "E10/P1-b0", "status": report["status"],
                      "package": str(out_zip), "n_files": len(files),
                      "inference_verified": verified,
                      "max_point_diff": max_diff, "minutes": exec_result.get("minutes"),
                      "checks": report["checks"]}, ensure_ascii=False, indent=2))
    if args.smoke or args.exploratory:
        return 0
    return 0 if verified else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
