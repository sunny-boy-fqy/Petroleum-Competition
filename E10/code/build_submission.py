#!/usr/bin/env python3
"""E10/P1：提交包组装（**纯标准库**）——只放"跑提交"需要的东西，并给出体积/内容收据。

提交包结构（PLAN §9.2）
---------------------
    README.md  predict.py  train.py  requirements.txt  configs/v4.yaml
    src/**                      # 推理所需的库代码（不含 __pycache__）
    models/v4/final/**          # fp32 权重 + manifest（体积上限 --max-weight-mb）
    submission_manifest.json    # 文件清单 + 逐个 sha256 + 体积

三条硬纪律
----------
1. **绝不打包数据/日志/虚拟环境**（`data/`、`cache/`、`*.log`、`.venv`、`__pycache__`）；
2. **体积上限**：代码 ≤ `--max-code-mb`、权重 ≤ `--max-weight-mb`，超限即 Gate 失败；
3. **requirements.txt 不得声明 torch/torch_npu**（平台镜像预装；写进去会把镜像的 torch 换掉）。

产出：`submission/submission_code_v4.zip`、`submission/result.zip`、
`submission/submission_manifest.json`、`$REPORTS/E10_build_gate.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

from src.training import metrics as M  # noqa: E402

EXCLUDE_DIRS = ("__pycache__", ".venv", ".venv-torch", "data", "cache", "logs", "logs_tb",
                ".git", "experiments", "dist")
EXCLUDE_SUFFIX = (".log", ".pyc", ".tmp")
EXCLUDE_NAMES = ("candidates.json",)
FORBIDDEN_REQ = ("torch", "torch_npu", "torch-npu", "ascend", "cann")


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E10/P1 提交包组装")
    ap.add_argument("--out", default=str(V4 / "submission" / "submission_code_v4.zip"))
    ap.add_argument("--weights", default=str(V4 / "models" / "v4" / "final"))
    ap.add_argument("--src", default=str(V4 / "src"))
    ap.add_argument("--predict", default=str(V4 / "predict.py"))
    ap.add_argument("--train", default=str(V4 / "train.py"))
    ap.add_argument("--requirements", default=str(V4 / "requirements.txt"))
    ap.add_argument("--readme", default=str(V4 / "README.md"))
    ap.add_argument("--configs", default=str(V4 / "configs"))
    ap.add_argument("--result-json", default=None, help="已生成的 result.json（打成 result.zip）")
    ap.add_argument("--result-zip", default=str(V4 / "submission" / "result.zip"))
    ap.add_argument("--manifest", default=str(V4 / "submission" / "submission_manifest.json"))
    ap.add_argument("--max-weight-mb", type=float, default=50.0)
    ap.add_argument("--max-code-mb", type=float, default=5.0)
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(Path(os.environ.get("V4_DATA_ROOT", "/data")) / "v4" / "reports"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def _included(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if rel.name in EXCLUDE_NAMES:
        return False
    return not rel.name.endswith(EXCLUDE_SUFFIX)


def collect(root: Path) -> list[Path]:
    out = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and _included(p.relative_to(root)):
            out.append(p)
    return out


def requirements_audit(path: Path) -> dict:
    if not path.is_file():
        return {"present": False, "torch_pins": [], "ok": False,
                "note": "缺少 requirements.txt"}
    lines = [ln.split("#", 1)[0].strip() for ln in
             path.read_text(encoding="utf-8").splitlines()]
    pins = [ln for ln in lines if ln and any(ln.lower().startswith(f)
                                             for f in FORBIDDEN_REQ)]
    return {"present": True, "torch_pins": pins, "ok": bool(not pins),
            "note": ("不得声明 torch/torch_npu/ascend/cann：平台镜像已预装，"
                     "pip 安装会替换镜像里的版本")}


def stage_tree(args, staging: Path) -> tuple[list[dict], list[str]]:
    """把需要打包的内容拷进 staging，返回（文件清单, 错误列表）。"""
    errors: list[str] = []
    files: list[dict] = []

    def put(src: Path, dst_rel: str) -> None:
        src = Path(src)
        if not src.exists():
            errors.append(f"缺少必需文件/目录：{src}")
            return
        if src.is_dir():
            for f in collect(src):
                rel = Path(dst_rel) / f.relative_to(src)
                tgt = staging / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, tgt)
                files.append({"path": str(rel), "bytes": int(tgt.stat().st_size),
                              "sha256": sha256_file(tgt)})
        else:
            tgt = staging / dst_rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
            files.append({"path": dst_rel, "bytes": int(tgt.stat().st_size),
                          "sha256": sha256_file(tgt)})

    put(Path(args.readme), "README.md")
    put(Path(args.predict), "predict.py")
    put(Path(args.train), "train.py")
    put(Path(args.requirements), "requirements.txt")
    put(Path(args.configs), "configs")
    put(Path(args.src), "src")
    weights_dir = Path(args.weights)
    before = len(files)
    if weights_dir.is_dir():
        put(weights_dir, "models/v4/final")
    if len(files) == before:
        errors.append(f"权重目录为空或缺失：{weights_dir}")
    return files, errors


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="v4_submission_"))
    files, errors = stage_tree(args, staging)
    code_bytes = sum(e["bytes"] for e in files if not e["path"].startswith("models/"))
    weight_bytes = sum(e["bytes"] for e in files if e["path"].startswith("models/"))
    code_mb, weight_mb = code_bytes / 1024 ** 2, weight_bytes / 1024 ** 2
    req = requirements_audit(Path(args.requirements))
    leaks = [e["path"] for e in files
             if any(part in EXCLUDE_DIRS for part in Path(e["path"]).parts)
             or e["path"].endswith(EXCLUDE_SUFFIX)
             or Path(e["path"]).name in EXCLUDE_NAMES]

    out_zip = Path(args.out)
    result_zip = Path(args.result_zip)
    manifest_path = Path(args.manifest)
    if not args.dry_run and not errors:
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for e in files:
                zf.write(staging / e["path"], arcname=e["path"])
        if args.result_json and Path(args.result_json).is_file():
            result_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(args.result_json, arcname="result.json")
        write_json(manifest_path, {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "files": files, "n_files": len(files),
            "code_bytes": code_bytes, "weight_bytes": weight_bytes,
            "code_mb": round(code_mb, 4), "weight_mb": round(weight_mb, 4),
            "zip": (str(out_zip) if out_zip.is_file() else None),
            "zip_sha256": (sha256_file(out_zip) if out_zip.is_file() else None),
            "result_zip": (str(result_zip) if result_zip.is_file() else None),
            "result_zip_sha256": (sha256_file(result_zip) if result_zip.is_file() else None),
            "requirements": req,
            "note": "只打包推理所需内容：代码 + 权重 + 入口；不含数据/日志/虚拟环境"})

    checks = {
        "contract_ok": bool(not errors and files),
        "code_size_ok": bool(code_mb <= float(args.max_code_mb)),
        "weight_size_ok": bool(weight_mb <= float(args.max_weight_mb)),
        "predict_entry_present": bool((staging / "predict.py").is_file()),
        "train_entry_present": bool((staging / "train.py").is_file()),
        "config_present": bool((staging / "configs" / "v4.yaml").is_file()),
        "requirements_no_torch_pin": bool(req["ok"]),
        "no_data_or_logs_included": bool(not leaks),
        "weights_present": bool(weight_bytes > 0),
        "result_zip_present": bool(result_zip.is_file()) if args.result_json else True,
        "dry_run": bool(args.dry_run),
    }
    passed = None if (args.smoke or args.exploratory or args.dry_run) else bool(
        all(v for k, v in checks.items() if k != "dry_run"))
    report = {"stage": "E10", "p_stage": "P1-build", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "exploratory": bool(args.exploratory), "dry_run": bool(args.dry_run),
              "errors": errors, "leaks": leaks, "requirements": req,
              "code_mb": round(code_mb, 4), "weight_mb": round(weight_mb, 4),
              "n_files": len(files), "zip": (str(out_zip) if out_zip.is_file() else None),
              "zip_sha256": (sha256_file(out_zip) if out_zip.is_file() else None),
              "manifest": (str(manifest_path) if manifest_path.is_file() else None),
              "checks": checks, "passed": passed}
    write_json(reports / "E10_build_submission.json", report)
    write_json(reports / "E10_build_gate.json",
               {"gate_id": "E10_build_gate", "stage": "E10", "p_stage": "P1-build",
                "created_at": report["created_at"],
                "exploratory": bool(args.smoke or args.exploratory),
                "passed": passed, "checks": checks,
                "metrics": {"code_mb": report["code_mb"], "weight_mb": report["weight_mb"]},
                "report_path": str(reports / "E10_build_submission.json")})
    print(json.dumps({"stage": "E10/P1-build", "n_files": len(files),
                      "code_mb": report["code_mb"], "weight_mb": report["weight_mb"],
                      "zip": report["zip"], "passed": passed, "checks": checks,
                      "errors": errors, "leaks": leaks}, ensure_ascii=False, indent=2))
    if args.dry_run or args.smoke or args.exploratory:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
