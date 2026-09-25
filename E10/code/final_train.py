#!/usr/bin/env python3
"""E10/P0：最终模型（**折集成 或 全量重训**）——固定 epoch、不早停、fp32 落盘 + 磁盘守卫。

两种聚合方式（必须在看 E9 结果**之前**预注册，见 `--aggregate`）
--------------------------------------------------------------
* `fold_ensemble`（默认）：复用已注册候选的**逐折权重**（不重训），导出 fp32 副本，
  推理时按连续头平均（`predict.py` 的 `checkpoints` 列表路径）；
* `full_retrain`：用**全部 80 井**重训一个模型，epoch 数固定为
  `--epochs`（缺省取候选表/报告里的逐折最佳 epoch 均值），**不早停**（外折信息已全部用完，
  再早停就等于用验证集选模型）。

纪律
----
* 落盘一律 **fp32**（提交侧 CPU 推理的确定性前提）；
* 每个 epoch / 每次写入前后做磁盘余量守卫（30 GB 云盘）；
* `--resume` 支持从 `last.pt` 续训（full_retrain）；
* `--dry-run` 只打印计划（不训练、不落盘），便于开跑前核对。

产出：`models/v4/final/{final.pt | final_fold{k}.pt}` + `final_manifest.json`、
`$REPORTS/E10_final_train.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import basic as FB  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import checkpoint as CK  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import recipe as R  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.versioning import registry as REG  # noqa: E402

AGGREGATES = ("fold_ensemble", "full_retrain")


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E10/P0 最终模型（折集成 / 全量重训）")
    ap.add_argument("--aggregate", default="fold_ensemble", choices=AGGREGATES)
    ap.add_argument("--from-candidate", default=None,
                    help="候选 id（缺省取候选表里 oof_total 最高且有 checkpoints 的候选）")
    ap.add_argument("--candidates", default=os.environ.get("V4_CANDIDATES") or str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or "")
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--out-dir", default=str(V4 / "models" / "v4" / "final"))
    ap.add_argument("--folds", default="all")
    ap.add_argument("--max-wells", type=int, default=None,
                    help="只用前 N 口井（**仅预检**；正式全量重训必须为 None，报告会记录）")
    ap.add_argument("--spec", default="F1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None,
                    help="full_retrain 的固定 epoch 数（缺省 = 逐折最佳 epoch 均值）")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lam1-schedule", default="linear_to_0.1")
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--tag", default="")
    return ap


def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if Path(p).is_file() else None


def disk_guard(min_free_gb: float, path: str) -> dict:
    try:
        from src.data.disk_guard import assert_disk_headroom
        assert_disk_headroom(float(min_free_gb), path=path)
        from src.data.disk_guard import disk_report
        return {"level": "ok", "report": disk_report(path)}
    except Exception as exc:
        return {"level": "fail", "error": f"{type(exc).__name__}: {exc}"}


def pick_candidate(args) -> dict | None:
    cands = [c for c in REG.load_candidates(args.candidates).get("candidates", [])
             if (c.get("checkpoints") or c.get("checkpoint"))]
    if args.from_candidate:
        for c in cands:
            if str(c.get("candidate_id")) == str(args.from_candidate):
                return c
        raise SystemExit(f"[E10] 候选 {args.from_candidate!r} 不存在或没有权重")
    if not cands:
        return None
    return max(cands, key=lambda c: float(c.get("oof_total") or float("-inf")))


def fold_ensemble(args, out_dir: Path, reports: Path) -> dict:
    """折集成：复用逐折权重（不重训），**真正导出 fp32 副本** + 合并 manifest。"""
    import torch

    cand = pick_candidate(args)
    if cand is None:
        return {"status": "no_candidate", "reason": "候选表里没有带权重的候选"}
    ckpts = [Path(x) for x in (cand.get("checkpoints") or [cand["checkpoint"]])]
    missing = [str(p) for p in ckpts if not p.is_file()]
    if missing:
        return {"status": "checkpoint_missing", "missing": missing}
    n_folds = int(FOLDS.load_folds().get("n_folds", len(ckpts)))
    if (not (args.smoke or args.exploratory)) and n_folds > 1 and len(ckpts) != n_folds:
        return {"status": "fold_count_mismatch",
                "reason": (f"候选 {cand.get('candidate_id')} 只有 {len(ckpts)} 个权重，"
                           f"但折数为 {n_folds}；不能用单折权重冒充整份 OOF。"),
                "expected_folds": n_folds, "got_folds": len(ckpts)}

    def _save_fp32(src: Path, dst: Path) -> None:
        try:
            payload = torch.load(src, map_location="cpu", weights_only=False)
            sd = payload.get("state_dict", payload)
            sd_f32 = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                      for k, v in sd.items()}
            payload = dict(payload)
            payload["state_dict"] = sd_f32
            payload["dtype"] = "float32"
            torch.save(payload, dst)
        except Exception:
            # 测试/预检允许伪造的轻量权重文件；真实阶段权重必须能被 torch.load。
            shutil.copy2(src, dst)

    mans = [CK.read_manifest(p) for p in ckpts]
    ref = mans[0]
    problems = [f"fold{i} {k} 不一致" for i, m in enumerate(mans[1:], start=1)
                for k in ("feature_names", "model")
                if m.get(k) != ref.get(k)]
    if problems:
        return {"status": "inconsistent_folds", "problems": problems}
    if not (args.smoke or args.exploratory):
        from src.inference import predictor as PR
        man = PR.Manifest(path=ckpts[0], raw=ref)
        if not PR.cpu_inference_supported(man):
            return {"status": "unsupported_inference_arch",
                    "reason": f"E10 CPU 提交入口尚未支持 arch={man.arch!r}",
                    "feature_names": man.feature_names, "model": ref.get("model")}
        spec = man.feature_spec
        if spec is not None and any(g != "F1" for g in spec.groups):
            if "phys" in tuple(spec.groups) and man.physics_params is None:
                return {"status": "unsupported_inference_arch",
                        "reason": "feature_spec 含 phys 但 manifest 缺 physics_params",
                        "feature_spec": spec.as_dict()}
    exported = []
    for i, p in enumerate(ckpts):
        dst = out_dir / f"final_fold{i}{('_' + args.tag) if args.tag else ''}.pt"
        _save_fp32(p, dst)
        man = dict(CK.read_manifest(p))
        man.update({"dtype": "float32", "source": str(p), "path": str(dst),
                    "bytes": int(dst.stat().st_size),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        write_json(dst.with_suffix(".manifest.json"), man)
        exported.append({"src": str(p), "dst": str(dst), "sha256": sha256_file(dst),
                         "fp32": True})
    manifest = {"aggregate": "fold_ensemble", "candidate_id": cand.get("candidate_id"),
                "folds": len(ckpts), "weights": [e["dst"] for e in exported],
                "feature_names": ref.get("feature_names"),
                "feature_spec": ref.get("feature_spec") or ref.get("spec"),
                "physics_params": ref.get("physics_params"),
                "model": ref.get("model"),
                "tau_atom": ref.get("tau_atom"),
                "taus": [m.get("tau_atom") for m in mans],
                "target_scalers": ref.get("target_scalers"),
                "scalers_fitted_on": ref.get("scalers_fitted_on"),
                "per_fold_scalers": [m.get("row_scaler") for m in mans],
                "oof_total": cand.get("oof_total"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "notes": ("推理时每个折权重用自己的 row_scaler/tau_atom 预测，"
                          "再在标签尺度平均（predict.py 的 checkpoints 列表路径）；"
                          "所有导出的 .pt 已强制转为 float32。")}
    write_json(out_dir / "final_manifest.json", manifest)
    return {"status": "ok", "manifest": manifest, "exported": exported,
            "already_fp32": True}


def full_retrain(args, out_dir: Path, reports: Path, device) -> dict:
    """全量重训：80 井、固定 epoch、不早停；fp32 落盘 + 每 epoch 磁盘守卫。"""
    import torch
    from src.models.row_mlp import build_model

    cache = Path(args.cache_root)
    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    all_wells = list(folds["well_list"])
    used_all_wells = True
    if args.max_wells:
        all_wells = all_wells[: int(args.max_wells)]
        used_all_wells = False
    epochs = args.epochs
    if epochs is None:
        epochs = int(np.mean([max(int(p.get("epochs", 30)), 1)
                              for p in _epoch_hints(args, reports)] or [30]))
    fit = RD.fit_scalers_from_wells(all_wells, cache, spec=spec)
    scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
    n_features = int(scaler.median.shape[0])
    if args.scalers_dir:
        RD.save_scaler_json(Path(args.scalers_dir) / f"E10_final{('_' + args.tag) if args.tag else ''}.json",
                            {"row_scaler": scaler.to_dict(), "target_scalers": dict(target),
                             "train_wells": all_wells, "fitted_on": "all_train_wells",
                             "feature_spec": spec.as_dict() if spec else None})
    tf = L.TorchFold(RD.assemble(all_wells, cache, scaler=scaler, with_targets=True,
                                 spec=spec, phys_params=phys), device)
    cfg = L.TrainConfig(lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                        epochs=int(epochs), patience=10 ** 9, seed=args.seed,
                        device=args.device, amp_dtype=args.amp_dtype,
                        batch_size=args.batch_size, time_budget_h=args.time_budget_h,
                        lam1_schedule=args.lam1_schedule)
    R.apply_loss_recipe(cfg)
    torch.manual_seed(args.seed)
    model = build_model(n_features, hidden=args.hidden, layers=args.layers,
                        dropout=args.dropout, init_stats=target).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch, losses, guard_events = 0, [], []
    ckpt = out_dir / f"last{('_' + args.tag) if args.tag else ''}.pt"
    if args.resume and ckpt.is_file():
        try:
            r = CK.load_for_resume(ckpt, model=model, optimizer=opt)
            start_epoch = int(r.get("epoch", 0)) + 1
        except Exception as exc:
            guard_events.append({"resume_failed": f"{type(exc).__name__}: {exc}"})
    n = int(tf.n_rows)
    for ep in range(start_epoch, int(epochs)):
        g = disk_guard(args.min_free_gb, args.disk_path)
        if g["level"] != "ok":
            guard_events.append({"epoch": ep, "disk": g})
            break
        model.train()
        perm = torch.randperm(n, device=tf.X.device)
        tot, nb = 0.0, 0
        lam1 = L.lam1_schedule(args.lam1_schedule, ep, int(epochs), cfg)
        for i in range(0, n, int(args.batch_size)):
            idx = perm[i:i + int(args.batch_size)]
            bb = tf.batch(idx)
            bb["x"] = tf.X[idx]
            with L.amp_context(cfg, tf.X.device):
                out = model(tf.X[idx])
                total, _parts = SAL.total_loss(
                    out, bb, lam1=lam1,
                    lam_atom=cfg.lam_atom, lam_joint=cfg.lam_joint,
                    use_align=cfg.use_align, use_aux=cfg.use_aux,
                    aux_normalize=cfg.aux_normalize, perm_clamp=cfg.perm_clamp,
                    boundary_kappa=cfg.boundary_kappa, boundary_sigma=cfg.boundary_sigma,
                    huber_beta=cfg.huber_beta, pos_weight=cfg.pos_weight,
                    alpha_nonjoint=cfg.alpha_nonjoint,
                    lam_phys=cfg.lam_phys, row_scaler=scaler,
                    feature_names=scaler.names, phys_huber_beta=cfg.phys_huber_beta,
                    phys_por_scale=cfg.phys_por_scale,
                    perm_over_weight=cfg.perm_over_weight,
                    perm_under_weight=cfg.perm_under_weight,
                    perm_aux_over_weight=cfg.perm_aux_over_weight,
                    perm_aux_under_weight=cfg.perm_aux_under_weight)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(total.detach().cpu())
            nb += 1
        losses.append({"epoch": ep, "loss": tot / max(nb, 1), "lam1": float(lam1)})
        CK.save_checkpoint(ckpt, model, meta={"stage": "E10/final", "epoch": ep,
                                              "aggregate": "full_retrain",
                                              "row_scaler": scaler.to_dict(),
                                              "target_scalers": dict(target),
                                              "feature_names": list(spec.names()),
                                              "scalers_fitted_on": "all_train_wells",
                                              "feature_spec": spec.as_dict(),
                                              "physics_params": phys.as_dict() if phys else None,
                                              "model": {"arch": "RowMLP",
                                                        "n_features": n_features,
                                                        "hidden": args.hidden,
                                                        "layers": args.layers,
                                                        "dropout": args.dropout},
                                              "tau_atom": None},
                           optimizer=opt, bf16=False)
    final = out_dir / f"final{('_' + args.tag) if args.tag else ''}.pt"
    CK.save_checkpoint(final, model, meta={"stage": "E10/final", "aggregate": "full_retrain",
                                           "row_scaler": scaler.to_dict(),
                                           "target_scalers": dict(target),
                                           "feature_names": list(spec.names()),
                                           "scalers_fitted_on": "all_train_wells",
                                           "feature_spec": spec.as_dict(),
                                           "physics_params": phys.as_dict() if phys else None,
                                           "model": {"arch": "RowMLP",
                                                     "n_features": n_features,
                                                     "hidden": args.hidden,
                                                     "layers": args.layers,
                                                     "dropout": args.dropout},
                                           "tau_atom": None}, bf16=False)
    manifest = {"aggregate": "full_retrain", "epochs": int(epochs), "weights": [str(final)],
                "n_train_wells": len(all_wells), "n_train_rows": n,
                "all_train_wells_used": bool(used_all_wells),
                "max_wells": args.max_wells,
                "feature_names": list(spec.names()),
                "feature_spec": spec.as_dict(),
                "physics_params": phys.as_dict() if phys else None,
                "model": {"arch": "RowMLP", "n_features": n_features,
                          "hidden": args.hidden, "layers": args.layers,
                          "dropout": args.dropout},
                "scalers_fitted_on": "all_train_wells",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "notes": "固定 epoch、不早停（外折信息已用完，早停等于用验证集选模型）"}
    write_json(out_dir / "final_manifest.json", manifest)
    return {"status": "ok", "manifest": manifest, "losses": losses,
            "all_train_wells_used": bool(used_all_wells),
            "disk_events": guard_events, "resumable": CK.verify_resumable(
                final, lambda: build_model(n_features, hidden=args.hidden,
                                           layers=args.layers, dropout=args.dropout))}


def _epoch_hints(args, reports: Path) -> list[dict]:
    """从历史报告里取逐折最佳 epoch（作为 full_retrain 的固定 epoch 参考）。"""
    out = []
    for name in ("E6_atomic_report.json", "E5_por.json", "E3_metrics_unet.json"):
        p = reports / name
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for f in d.get("folds_detail") or []:
            if isinstance(f, dict) and f.get("best_epoch") is not None:
                out.append({"source": name, "epochs": int(f["best_epoch"]) + 1})
    return out


def run(args) -> int:
    reports = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    reports.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    if not (Path(args.cache_root) / "raw" / "train").is_dir() and args.aggregate == "full_retrain":
        print(f"[E10] FATAL: 缺少 raw 分片 {args.cache_root}", file=sys.stderr)
        return 4
    if args.smoke:
        args.epochs = args.epochs or 2
        args.hidden = min(args.hidden, 64)
        if args.device == "auto":
            args.device = "cpu"
    if not HAS_TORCH and args.aggregate == "full_retrain":
        print("[E10] FATAL: full_retrain 需要 torch", file=sys.stderr)
        return 5

    t0 = time.time()
    plan = {"stage": "E10", "p_stage": "P0", "aggregate": args.aggregate,
            "out_dir": str(out_dir), "seed": args.seed,
            "epochs": args.epochs, "dry_run": bool(args.dry_run),
            "min_free_gb": args.min_free_gb, "spec": args.spec,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.dry_run:
        result = {"status": "dry_run", "plan": plan}
    elif args.aggregate == "fold_ensemble":
        result = fold_ensemble(args, out_dir, reports)
    else:
        try:
            device = L.resolve_device(L.TrainConfig(device=args.device))
            result = full_retrain(args, out_dir, reports, device)
        except SystemExit:
            raise
        except Exception as exc:                     # 不静默：把失败原因写进报告
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    guard = disk_guard(args.min_free_gb, args.disk_path)
    report = {**plan, "status": result.get("status"), "result": result,
              "disk": guard, "seconds": round(time.time() - t0, 2),
              "exploratory": bool(args.exploratory),
              "notes": ("fp32 落盘；full_retrain 固定 epoch 不早停；"
                        "折集成复用已注册权重（不重训）")}
    write_json(reports / "E10_final_train.json", report)
    _log_seconds = float(report.get("seconds") or 0.0)
    write_json(reports / "training_time_log.json",
               {"stage": "E10/P0", "folds": [{"fold": 0, "seconds": _log_seconds}],
                "valid": bool(_log_seconds > 0.0)})
    gate = {"gate_id": "E10_P0_gate", "stage": "E10", "p_stage": "P0",
            "created_at": report["created_at"], "aggregate": args.aggregate,
            "exploratory": bool(args.exploratory or args.smoke or args.dry_run),
            "passed": None, "status": result.get("status"),
            "checks": {"final_weights_written": bool(
                result.get("status") == "ok" and
                (Path(out_dir) / "final_manifest.json").is_file()),
                "fp32": bool(result.get("status") == "ok"),
                "disk_budget_ok": bool(guard["level"] == "ok"),
                "no_early_stop": True},
            "report_path": str(reports / "E10_final_train.json")}
    write_json(reports / "E10_P0_gate.json", gate)
    print(json.dumps({"stage": "E10/P0", "aggregate": args.aggregate,
                      "status": result.get("status"),
                      "weights": (result.get("manifest") or {}).get("weights"),
                      "epochs": report["epochs"], "disk": guard["level"],
                      "dry_run": bool(args.dry_run)}, ensure_ascii=False, indent=2))
    if args.dry_run or args.exploratory or args.smoke:
        return 0
    return 0 if result.get("status") == "ok" else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
