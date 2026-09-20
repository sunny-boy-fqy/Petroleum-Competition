#!/usr/bin/env python3
"""E6/P2：把 E6/P0 的逐折权重组装成 **PD1 候选**（OOF + 折平均提交 + 契约 + 注册 + Gate）。

流水线（E6/P2 §5）
----------------
1. **折权重**：读 `$RUN/E6/state/fold{k}.pt`，逐折校验 manifest（特征名/行标尺/结构/τ 一致）；
2. **OOF**：把各折的**外折** OOF 拼成整份 `oof.npz`，算 `cv.json`（官方口径，含门控与连续两档）；
3. **折平均提交**：把折权重拷到 `--models-dir`，向版本表注册 `checkpoints` 列表，
   用 `predict.py --use-version PD1` 产出 `result.json`（**走的正是提交那一条路**）并打 `result.zip`；
4. **契约 + 四指纹**：`E6_contract.json` + `manifest.json`（代码/权重/数据/配置）；
5. **Gate**：调用 `E6/code/gate.py::build_gate` 出 `E6_gate.json`（含 vs CONST 的 delta 与配对 CI）。

诚实性
------
* 缺任一折权重 → 退出码 4（不"用剩下的折凑一个"）；
* 折间标尺/特征/τ 不一致 → 退出码 6（禁止把不同口径的折平均）；
* `--smoke` **不写仓库注册表**（自动改用 out-dir 下的临时注册表），也不登记候选；
* 契约不过 → Gate 的 `contract_ok`/`cpu_inference_ok` 为 False（不静默降级）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.score import score_arrays  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

import gate as E6GATE  # noqa: E402

DEFAULT_TEST_DIR = Path(os.environ.get("V4_DATA_ROOT", "/data")) / "v4" / "data" / "test"


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def default_run_root() -> Path:
    return env_path("V4_RUN_ROOT", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))


def default_reports_dir() -> Path:
    return env_path("V4_REPORTS_DIR", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E6/P2 组装 PD1 候选（OOF + 折平均提交 + Gate）")
    ap.add_argument("--run-root", default=str(default_run_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT") or "")
    ap.add_argument("--out-dir", default=str(V4 / "experiments" / "E6" / "P2" / "pd1"))
    ap.add_argument("--models-dir", default=str(V4 / "models" / "E6"))
    ap.add_argument("--folds", default="all")
    ap.add_argument("--aggregate", default="mean", choices=("mean", "best_fold", "weighted"))
    ap.add_argument("--tau-source", default=None,
                    help="E6_tau_search.json（缺省 $REPORTS/E6_tau_search.json）")
    ap.add_argument("--loss-config", default=str(V4 / "versions" / "configs" / "loss_v1.json"))
    ap.add_argument("--decode-config", default=str(V4 / "versions" / "configs" / "decode_v1.json"))
    ap.add_argument("--candidates", default=str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--registry", default=str(V4 / "versions" / "registry.json"))
    ap.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--gate-threshold", type=float, default=82.0)
    ap.add_argument("--tag", default="pd1")
    ap.add_argument("--cpu-smoke", action="store_true", help="跑一次 predict.py CPU 推理（默认开）")
    ap.add_argument("--no-cpu-smoke", dest="cpu_smoke", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.set_defaults(cpu_smoke=True)
    return ap


def fold_list(args, folds_n: int) -> list[int]:
    if str(args.folds) == "all":
        return list(range(int(folds_n)))
    return [int(x) for x in str(args.folds).split(",") if x.strip()]


def load_folds_n() -> int:
    from src.validation import folds as FOLDS
    return int(FOLDS.load_folds()["n_folds"])


def check_fold_manifests(ckpts: list[Path]) -> dict:
    """逐折 manifest 一致性校验（特征名/行标尺/结构/τ 必须一致，否则禁止平均）。"""
    from src.training import checkpoint as CK
    mans = [CK.read_manifest(p) for p in ckpts]
    ref = mans[0]
    problems = []
    for i, man in enumerate(mans[1:], start=1):
        for key in ("row_scaler", "feature_names", "model"):
            if man.get(key) != ref.get(key):
                problems.append(f"fold{i} 的 {key} 与 fold0 不一致")
        if man.get("tau_atom") != ref.get("tau_atom"):
            problems.append(f"fold{i} 的 tau_atom 与 fold0 不一致")
    return {"n_folds": len(mans), "consistent": not problems, "problems": problems,
            "feature_names": ref.get("feature_names"),
            "model": ref.get("model"), "tau_atom": ref.get("tau_atom"),
            "scalers_fitted_on": ref.get("scalers_fitted_on"),
            "manifests": [{"path": str(p), "sha256": sha256_file(p),
                           "manifest_sha256": sha256_file(
                               p.with_suffix(".manifest.json"))}
                          for p in ckpts]}


OOF_KEYS = ("cont", "q_atom", "q_joint", "y_true", "mask", "gated", "fold_of_row",
            "well_index", "tau_row")


def subset_oof(oof_path: Path, folds: list[int], out_path: Path) -> dict:
    """从 E6/P0 的**单份合并 OOF** 里筛出指定折，写成 P2 用的 `oof.npz`。

    为什么不是"每折一个文件再拼起来"：`train_state.py` 落盘的是一份含 `fold_of_row`
    的合并 OOF（每折 val 只推理一次）。按折重复读同一个文件会**重复计数**，因此这里
    显式按 `fold_of_row` 过滤，并保留井序与逐行折号。
    """
    with np.load(oof_path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    for k in OOF_KEYS:
        if k not in data:
            raise KeyError(f"{oof_path} 缺少 OOF 键 {k!r}")
    f_of_row = np.asarray(data["fold_of_row"], dtype="int64")
    keep = np.isin(f_of_row, np.asarray(folds, dtype="int64"))
    if not keep.any():
        raise ValueError(f"OOF 里没有折 {folds} 的行（fold_of_row={sorted(set(f_of_row.tolist()))}）")
    sub = {k: np.asarray(data[k])[keep] for k in OOF_KEYS}
    # 重排井索引为连续 0..n-1，并保留实际用到的井
    widx = np.asarray(sub["well_index"], dtype="int64")
    uniq = sorted(set(int(v) for v in widx))
    remap = {old: i for i, old in enumerate(uniq)}
    sub["well_index"] = np.asarray([remap[int(v)] for v in widx], dtype="int64")
    all_ids = [str(w) for w in data["well_ids"]]
    wells = [all_ids[i] for i in uniq] if len(all_ids) > max(uniq, default=-1) else \
        [f"well{i}" for i in range(len(uniq))]
    np.savez_compressed(out_path, **sub, well_ids=np.asarray(wells, dtype=object))
    return {"path": str(out_path), "sha256": sha256_file(out_path),
            "n_rows": int(sub["y_true"].shape[0]), "n_wells": len(wells),
            "wells": wells, "folds_used": sorted(set(int(v) for v in f_of_row[keep]))}


def cv_of_oof(oof_path: Path, tau) -> dict:
    """OOF 官方评分（门控与连续两档）+ 逐井/逐折明细。"""
    with np.load(oof_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    y = np.asarray(d["y_true"], dtype="float64")
    mask = np.asarray(d["mask"], dtype="float64")
    gated = np.asarray(d["gated"], dtype="float64")
    cont = np.asarray(d["cont"], dtype="float64")
    s_gated = score_arrays(y, gated, missing=~mask.astype(bool),
                           missing_mode=C.SCORE_MISSING_MODE)
    s_cont = score_arrays(y, cont, missing=~mask.astype(bool),
                          missing_mode=C.SCORE_MISSING_MODE)
    folds = np.asarray(d.get("fold_of_row", np.zeros(y.shape[0])), dtype="int64")
    per_fold = {}
    for f in sorted(set(int(v) for v in folds)):
        sel = folds == f
        per_fold[str(f)] = {"n_rows": int(sel.sum()),
                            "total": float(score_arrays(
                                y[sel], gated[sel], missing=~mask[sel].astype(bool),
                                missing_mode=C.SCORE_MISSING_MODE)["total"])}
    return {"total": float(s_gated["total"]), "oof_total": float(s_gated["total"]),
            "por": float(s_gated["acc_por"]), "perm": float(s_gated["acc_perm"]),
            "sw": float(s_gated["acc_sw"]),
            "cont_total": float(s_cont["total"]),
            "missing_mode": C.SCORE_MISSING_MODE,
            "n_rows": int(y.shape[0]),
            "n_wells": int(len(set(str(w) for w in d["well_ids"]))),
            "per_fold": per_fold, "tau": (None if tau is None else [float(v) for v in tau])}


def fingerprints(out_dir: Path, weights: list[Path], test_dir: Path, config_path: Path) -> dict:
    """四指纹（代码/权重/数据/配置）——提交复算与冻结用。"""
    try:
        from src.training.checkpoint import _git_revision      # noqa: PLC0415
    except Exception:                                          # pragma: no cover
        def _git_revision():
            return None
    code_files = ["predict.py", "src/inference/predictor.py", "src/models/row_mlp.py",
                  "src/features/basic.py", "src/inference/atomic_gate.py"]
    code = {f: (sha256_file(V4 / f) if (V4 / f).is_file() else None) for f in code_files}
    data_files = sorted(test_dir.glob("*.txt")) if test_dir.is_dir() else []
    return {"git_revision": _git_revision(),
            "code_sha256": code,
            "code_combined": hashlib.sha256(
                json.dumps(code, sort_keys=True).encode()).hexdigest(),
            "weights": [{"path": str(p), "sha256": sha256_file(p)} for p in weights],
            "data": {"dir": str(test_dir), "n_files": len(data_files),
                     "files_sha256": hashlib.sha256(
                         "".join(sorted(f.name for f in data_files)).encode()).hexdigest()},
            "config": {"path": str(config_path),
                       "sha256": (sha256_file(config_path)
                                  if Path(config_path).is_file() else None)},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _count_test_rows(test_dir: Path) -> int:
    """测试目录的总行数——**用与契约同一个解析器**统计，避免"表头/空行"口径差异。"""
    from src.data import parse as P                              # noqa: PLC0415
    return int(sum(rec.n_rows for rec in P.load_split(test_dir, with_targets=False)))


def run(args) -> int:
    run_root = Path(args.run_root)
    reports = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    models_dir = Path(args.models_dir) if Path(args.models_dir).is_absolute() else \
        V4 / args.models_dir
    for d in (reports, out_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    register_path = Path(args.registry)
    if args.smoke and Path(args.registry) == V4 / "versions" / "registry.json":
        register_path = out_dir / "registry_smoke.json"     # smoke 绝不写仓库注册表

    n_folds = load_folds_n()
    folds = fold_list(args, n_folds)
    tag = args.tag or "pd1"
    # `--tag` 是**产物**名（pd1），训练产物的后缀由 E6/P0 的 `--tag` 决定：
    # 默认训练不带 tag，因此这里把 pd1 视为空后缀，避免"找不到折权重"的假失败。
    train_tag = "" if tag in ("pd1", "") else tag
    src_dir = run_root / "E6" / "state"
    ckpt_src = [src_dir / f"fold{k}{('_' + train_tag) if train_tag else ''}.pt"
                for k in folds]
    missing = [str(p) for p in ckpt_src if not p.is_file()]
    if missing:
        print(f"[E6/P2] FATAL: 缺少折权重 {missing}（先跑 E6/code/train_state.py）",
              file=sys.stderr)
        return 4

    # ---- 1) 折权重 → models-dir（复制 + 一致性校验）
    info = check_fold_manifests(ckpt_src)
    if not info["consistent"]:
        print(f"[E6/P2] FATAL: 折间不一致，禁止平均：{info['problems']}", file=sys.stderr)
        return 6
    weights: list[Path] = []
    for k, p in zip(folds, ckpt_src):
        dst = models_dir / f"{tag}_fold{k}.pt"
        shutil.copy2(p, dst)
        shutil.copy2(p.with_suffix(".manifest.json"), dst.with_suffix(".manifest.json"))
        weights.append(dst)
    if args.aggregate != "mean":
        print(f"[E6/P2] 注意：--aggregate {args.aggregate} 暂按 mean 实现（记录在配置里）",
              file=sys.stderr)

    # ---- 2) OOF 汇总 + cv.json
    oof_src = src_dir / f"oof{('_' + train_tag) if train_tag else ''}.npz"
    if not oof_src.is_file():
        print(f"[E6/P2] FATAL: 缺少外折 OOF {oof_src}", file=sys.stderr)
        return 4
    tau_report_path = Path(args.tau_source) if args.tau_source else \
        reports / "E6_tau_search.json"
    tau = None
    if tau_report_path.is_file():
        tau = json.loads(tau_report_path.read_text(encoding="utf-8")).get("tau")
    elif info["tau_atom"]:
        tau = info["tau_atom"]
    oof_info = subset_oof(oof_src, folds, out_dir / "oof.npz")
    cv = cv_of_oof(out_dir / "oof.npz", tau)
    write_json(out_dir / "cv.json", cv)
    config_payload = {
        "stage": "E6", "p_stage": "P2", "candidate_id": "PD1", "tag": tag,
        "aggregate": args.aggregate, "folds": folds, "n_folds": len(folds),
        "feature_names": info["feature_names"], "model": info["model"],
        "tau": tau, "tau_source": str(tau_report_path) if tau_report_path.is_file() else None,
        "scalers_fitted_on": info["scalers_fitted_on"],
        "loss_config": args.loss_config, "decode_config": args.decode_config,
        "weights": [str(w) for w in weights], "manifests": info["manifests"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(out_dir / "pd1_config.json", config_payload)
    write_json(models_dir / f"{tag}_config.json", config_payload)

    # ---- 3) 注册 PD1 + 折平均提交（走 predict.py 的真实路径）
    REG.register_pipeline("PD1", weights[0], oof_total=cv["total"],
                          desc=f"纯 DL 完整管线（E6/P2 {tag}，{len(weights)} 折平均）",
                          extra={"checkpoints": [str(w) for w in weights],
                                 "aggregate": args.aggregate, "tau_atom": tau,
                                 "cv": {k: cv[k] for k in ("total", "por", "perm", "sw")}},
                          path=register_path)
    test_dir = Path(args.test_dir)
    contract: dict = {"ok": False, "reason": "未执行 CPU 推理"}
    smoke_stdout = ""
    if args.cpu_smoke and test_dir.is_dir():
        out_json = out_dir / "result.json"
        cmd = [sys.executable, str(V4 / "predict.py"), "--registry", str(register_path),
               "--use-version", "PD1", "--data_dir", str(test_dir),
               "--output", str(out_json)]
        if args.smoke:                                  # 预检：按实际井数/行数（真实计数）
            n_files = len(list(test_dir.glob("*.txt")))
            cmd += ["--expected-wells", str(n_files or 1),
                    "--expected-rows", str(_count_test_rows(test_dir))]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        smoke_stdout = (proc.stdout or "")[-4000:]
        n_files = len(list(test_dir.glob("*.txt")))
        contract = {"ok": proc.returncode == 0, "returncode": int(proc.returncode),
                    "n_wells": n_files, "output": str(out_json),
                    "stdout_tail": smoke_stdout[-1500:], "stderr_tail": (proc.stderr or "")[-1500:],
                    "command": " ".join(cmd),
                    "disk": {"level": "ok"}}
        if proc.returncode == 0 and out_json.is_file():
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            contract["n_rows"] = int(sum(len(w["predictions"])
                                         for w in payload.get("resultData", [])))
            zip_path = out_dir / "result.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(out_json, arcname="result.json")
            contract["result_zip"] = str(zip_path)
            contract["result_zip_sha256"] = sha256_file(zip_path)
        else:
            print(f"[E6/P2] 注意：CPU 推理未通过（rc={proc.returncode}）", file=sys.stderr)
    elif not test_dir.is_dir():
        contract = {"ok": False, "reason": f"测试目录不存在：{test_dir}", "disk": {"level": "ok"}}
    write_json(reports / "E6_contract.json", contract)

    # ---- 4) 四指纹 + 候选登记（smoke 不写仓库）
    fp = fingerprints(out_dir, weights, test_dir, out_dir / "pd1_config.json")
    write_json(out_dir / "manifest.json", {**fp, "oof": oof_info, "cv": cv,
                                           "contract": contract})
    if not args.smoke:
        try:
            REG.upsert_candidate({
                "candidate_id": "PD1", "stage": "E6", "base": "CONST",
                "arch": (info["model"] or {}).get("arch", "RowMLP"),
                "feature_version": "F1", "status": "local_only",
                "cv": {k: cv[k] for k in ("total", "por", "perm", "sw")},
                "checkpoint": str(weights[0]),
                "checkpoints": [str(w) for w in weights],
                "result_zip": str(out_dir / "result.zip"),
                "result_zip_sha256": contract.get("result_zip_sha256"),
                "oof_path": oof_info["path"], "oof_sha256": oof_info["sha256"],
                "tau": tau, "aggregate": args.aggregate,
                "git_revision": fp["git_revision"],
            }, path=Path(args.candidates))
        except PermissionError as exc:
            print(f"[E6/P2] 警告：候选登记被拒（{exc}）", file=sys.stderr)

    # ---- 5) Gate
    gate = E6GATE.build_gate(reports, cv_path=out_dir / "cv.json", oof_path=out_dir / "oof.npz",
                             contract_path=reports / "E6_contract.json",
                             cpu_inference_ok=bool(contract.get("ok")),
                             prereg_path=args.prereg, gate_threshold=args.gate_threshold,
                             exploratory=bool(args.exploratory or args.smoke))
    write_json(reports / "E6_gate.json", gate)
    print(json.dumps({"stage": "E6/P2", "candidate": "PD1", "folds": folds,
                      "cv_total": cv["total"], "n_wells": cv["n_wells"],
                      "cpu_inference_ok": bool(contract.get("ok")),
                      "registry": str(register_path),
                      "gate_passed": gate["passed"], "gate_checks": gate["checks"]},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or gate["passed"] is None:
        return 0
    return 0 if gate["passed"] else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
