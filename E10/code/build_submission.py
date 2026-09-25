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
from src.versioning import registry as REG  # noqa: E402

# 顶层运行时目录必须排除；`data/cache/logs` 只能按**第一段路径**排除，
# 否则会误伤 `src/data/`（提交包 predict.py 必需）与 `src/training/...` 等源码。
EXCLUDE_TOP_DIRS = ("data", "cache", "logs", "logs_tb", ".git", "experiments", "dist")
EXCLUDE_ANY_DIRS = ("__pycache__", ".venv", ".venv-torch")
EXCLUDE_DIRS = EXCLUDE_TOP_DIRS + EXCLUDE_ANY_DIRS
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
    ap.add_argument("--registry-source", default=str(V4 / "versions" / "registry.json"),
                    help="源版本注册表；打包时重写为包内相对 checkpoint 路径")
    ap.add_argument("--out", default=str(V4 / "submission" / "submission_code_v4.zip"))
    ap.add_argument("--weights", default=str(V4 / "models" / "v4" / "final"))
    ap.add_argument("--src", default=str(V4 / "src"))
    ap.add_argument("--predict", default=str(V4 / "predict.py"))
    ap.add_argument("--train", default=str(V4 / "train.py"))
    ap.add_argument("--requirements", default=str(V4 / "requirements.txt"))
    ap.add_argument("--readme", default=str(V4 / "README.md"))
    ap.add_argument("--configs", default=str(V4 / "configs"))
    ap.add_argument("--version-configs", default=str(V4 / "versions" / "configs"),
                    help="E7 冻结的 loss_v1.json / decode_v1.json（若存在则一并打包）")
    ap.add_argument("--stage-code-root", default=str(V4),
                    help="E*/code 训练脚本所在仓库根；打包后 train.py 才可用（M4）")
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
    parts = rel.parts
    if parts and parts[0] in EXCLUDE_TOP_DIRS:
        return False
    if any(part in EXCLUDE_ANY_DIRS for part in parts):
        return False
    if rel.name in EXCLUDE_NAMES:
        return False
    return not rel.name.endswith(EXCLUDE_SUFFIX)



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


def _record_file(staging: Path, rel: str, files: list[dict]) -> None:
    """把 staging 内某个文件加入清单（用于打包阶段动态生成的 versions/registry.json）。"""
    p = staging / rel
    files.append({"path": rel, "bytes": int(p.stat().st_size), "sha256": sha256_file(p)})


def _package_weights(staging: Path) -> list[str]:
    """返回包内权重的**相对包根路径**（优先 final_manifest.json 的 weights 列表）。

    提交包必须自包含：registry 里记录相对路径，预测端 `_resolve_path` 会相对包根解析。
    """
    wdir = staging / "models" / "v4" / "final"
    chosen: list[Path] = []
    fm = wdir / "final_manifest.json"
    if fm.is_file():
        try:
            doc = json.loads(fm.read_text(encoding="utf-8"))
            for src in doc.get("weights", []):
                cand = wdir / Path(str(src)).name
                if cand.is_file() and cand not in chosen:
                    chosen.append(cand)
        except Exception:
            chosen = []
    if not chosen:
        chosen = sorted(p for p in wdir.glob("*.pt")
                        if p.is_file() and not p.name.endswith(".tmp.pt"))
    return [str(p.relative_to(staging)).replace(os.sep, "/") for p in chosen]


def _write_packaged_registry(args, staging: Path, files: list[dict]) -> list[str]:
    """生成包内 versions/registry.json：PD1 指向包内相对权重路径。

    为什么必须重写：源注册表可能由 E6/P2 在临时仓库或 /data 下写出，checkpoint 为
    绝对路径；而提交包解压目录与训练机路径不同。包内注册表是预测端默认读取的文件，
    必须保证 `checkpoint` / `checkpoints` 是相对包根且真实存在的路径。
    """
    weights = _package_weights(staging)
    if not weights:
        return ["包内未发现 .pt 权重，无法生成 versions/registry.json"]
    src = Path(args.registry_source)
    if src.is_file():
        doc = json.loads(src.read_text(encoding="utf-8"))
    else:
        doc = json.loads(json.dumps(REG.DEFAULT_REGISTRY))
    versions = doc.setdefault("versions", {})
    pd1 = dict(versions.get("PD1", {}))
    pd1.update({
        "type": "pipeline",
        "available": True,
        "completed": True,
        "desc": pd1.get("desc") or "纯 DL 完整管线（E6/P2，折平均）",
        "entrypoint": pd1.get("entrypoint") or "src/inference/predictor.py",
        "checkpoint": weights[0],
        "checkpoints": weights,
        "packaged": True,
        "checkpoint_paths": "relative_to_package_root",
    })
    # 折集成的每个 manifest 都随权重打包；predict.py 会按每个 manifest 自己的
    # row_scaler/tau 做推理，因此这里不再要求折间 scaler 完全一致。
    versions["PD1"] = pd1
    doc["latest"] = "PD1"
    doc["notes"] = ("提交包内置注册表：checkpoint/checkpoints 均相对包根；"
                    "由 E10/code/build_submission.py 生成。")
    reg_rel = "versions/registry.json"
    write_json(staging / reg_rel, doc)
    _record_file(staging, reg_rel, files)
    return []


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
            # 注意：排除规则必须按**包内目标路径**判断，不能按源相对路径。
            # 否则 `src/data/` 会被源 rel 的第一段 `data` 误伤，而提交包 predict.py 必需它。
            for f in sorted(src.rglob("*")):
                if not f.is_file():
                    continue
                rel = Path(dst_rel) / f.relative_to(src)
                if not _included(rel):
                    continue
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
    version_cfg = Path(args.version_configs)
    if version_cfg.is_dir():
        put(version_cfg, "versions/configs")
    # 审查 H4：多任务拆分时 repo 内 versions/configs 可能不存在（临时 clone），
    # 但 E7 已把配置镜像到 $V4_REPORTS_DIR；这里补进提交包，保证 predict.py 可读回。
    reports_cfg = Path(args.reports_dir)
    for cfg_name in ("loss_v1.json", "decode_v1.json"):
        rel_cfg = Path("versions/configs") / cfg_name
        if not (staging / rel_cfg).is_file():
            src_cfg = reports_cfg / cfg_name
            if src_cfg.is_file():
                put(src_cfg, str(rel_cfg))
    put(Path(args.src), "src")
    # M4：train.py 的 STAGE_SCRIPTS 指向 E*/code/*.py；把它们打进包，
    # 训练入口才不是"有脚本却找不到阶段实现"的空壳（数据/权重仍不打包）。
    stage_root = Path(args.stage_code_root)
    for stage in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"):
        stage_code = stage_root / stage / "code"
        if stage_code.is_dir():
            put(stage_code, f"{stage}/code")
    weights_dir = Path(args.weights)
    before = len(files)
    if weights_dir.is_dir():
        put(weights_dir, "models/v4/final")
    if len(files) == before:
        errors.append(f"权重目录为空或缺失：{weights_dir}")
    if not errors:
        errors.extend(_write_packaged_registry(args, staging, files))
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
             if (Path(e["path"]).parts and Path(e["path"]).parts[0] in EXCLUDE_TOP_DIRS)
             or any(part in EXCLUDE_ANY_DIRS for part in Path(e["path"]).parts)
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

    # C1 回归：包内注册表必须存在、checkpoint 必须为**相对包根**且真实存在。
    packaged_reg = staging / "versions" / "registry.json"
    reg_doc = None
    reg_ckpts: list[str] = []
    if packaged_reg.is_file():
        try:
            reg_doc = json.loads(packaged_reg.read_text(encoding="utf-8"))
            info = ((reg_doc.get("versions") or {}).get("PD1") or {})
            reg_ckpts = [str(x) for x in (info.get("checkpoints")
                                          or ([info["checkpoint"]] if info.get("checkpoint")
                                              else []))]
        except Exception:
            reg_doc = None
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
        "bundled_registry": bool(packaged_reg.is_file()),
        "registry_checkpoints_relative": bool(reg_ckpts and all(
            not Path(x).is_absolute() and ".." not in Path(x).parts for x in reg_ckpts)),
        "registry_checkpoints_present": bool(reg_ckpts and all(
            (staging / x).is_file() for x in reg_ckpts)),
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
              "checks": checks, "passed": passed,
              "bundled_registry": ({"present": bool(packaged_reg.is_file()),
                                    "latest": (reg_doc or {}).get("latest"),
                                    "checkpoints": reg_ckpts} if reg_doc is not None or
                                   packaged_reg.is_file() else None)}
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
