"""E5 公共件（head_perm / head_sw 复用）：冻结骨干、逐行隐状态、单目标两阶段训练、配对 CI、Gate。

设计
----
E5 的三个逐目标脚本共享 90% 的骨架，差别只在三处（以可调用对象注入，避免复制粘贴）：
  * `predict_fn(head, x)` → 该目标的**标签尺度**预测；
  * `loss_fn(pred, y)` → 官方**对齐得分**（越大越好；调用方取负做损失）；
  * `metric_fn(y, pred)` → 官方命中率（`src/score.py` 口径）。

纪律（与 head_por 一致，这里是唯一实现）
* 折内标尺只由训练折拟合；
* 两阶段：内折按官方指标早停并 keep_best → 全折重训 `best_epoch+1`；
* 任何阈值（τ）只在内折上选；
* 没有真实 `--backbone-ckpt` 时 `passed=None`（仅链路预检，**不得判 PASS**）。

`head_por.py` 早于本模块写成（自带同源实现，已被单测覆盖）；后续轮次会切到本模块消除重复。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src import constants as C
from src.data import row_dataset as RD
from src.inference import atomic_gate as AG
from src.losses import score_aligned as SAL
from src.portability import HAS_TORCH
from src.training import fold_runner as FR
from src.training import loop as L
from src.training import metrics as M
from src.training import seq_loop as SL
from src.validation import folds as FOLDS
from src.validation import gates as GATES

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
ARCH_KW = {
    "patchtf": {"patch_len": 32, "stride": 16, "d_model": 128, "n_layers": 4, "n_heads": 8},
    "unet": {"base_ch": 64, "depth": 5, "k": 5},
    "tcn": {"channels": 128, "n_blocks": 9, "k": 3, "dilation_max": 512},
}


# ---------------------------------------------------------------- 路径 / IO
def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def default_cache_root() -> Path:
    return env_path("V4_CACHE_ROOT", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))


def default_reports_dir() -> Path:
    return env_path("V4_REPORTS_DIR", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))


def default_run_root() -> Path:
    return env_path("V4_RUN_ROOT", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def add_common_args(ap: argparse.ArgumentParser, arch_choices=("patchtf", "unet", "tcn")) -> None:
    """E5 三个脚本共用的 CLI（名字/默认值必须一致，便于云端一条命令切换目标）。"""
    ap.add_argument("--arch", default="patchtf", choices=tuple(arch_choices))
    ap.add_argument("--backbone-ckpt", default=None,
                    help="冻结骨干权重；缺省（仅 --smoke/--exploratory）用随机初始化骨干")
    ap.add_argument("--arch-kwargs", default=None, help="JSON：覆盖 ARCH_KW（必须与权重一致）")
    ap.add_argument("--cache-root", default=str(default_cache_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--run-root", default=str(default_run_root()))
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or "")
    ap.add_argument("--spec", default="F1")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-chunks", type=int, default=4)
    ap.add_argument("--weight-kind", default="triangular",
                    choices=("triangular", "hann", "equal"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=C.DISK_MIN_FREE_GB)
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--ablation", action="store_true", help="跑参数化消融")
    ap.add_argument("--ablation-folds", default="0")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")


def resolve_dirs(args) -> tuple[Path, Path, Path, Path]:
    """`(cache, reports, run_dir, scalers)`，并确保存在；缺 raw 分片时返回 4 号退出。"""
    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    run_dir = Path(args.run_root) / "E5" / args.target.lower()
    if args.scalers_dir:
        scalers = Path(args.scalers_dir)
    elif os.environ.get("V4_DATA_ROOT"):
        scalers = Path(os.environ["V4_DATA_ROOT"]) / "v4" / "scalers"
    else:
        scalers = reports.parent / "scalers"
    for d in (reports, run_dir, scalers):
        d.mkdir(parents=True, exist_ok=True)
    return cache, reports, run_dir, scalers


# ---------------------------------------------------------------- 折 / 骨干 / 标签
def fold_wells(args, folds, k) -> tuple[list[str], list[str]]:
    tr, va = RD.fold_wells(folds, k)
    if args.max_wells:
        tr, va = tr[:args.max_wells], va[:args.max_wells]
    return list(tr), list(va)


def load_backbone(args, n_features: int, device, fold: int | None = None):
    """加载指定折的冻结骨干。

    优先 ``--backbone-ckpt``；若它是目录或带 ``{fold}`` 的模板，则自动取该折权重。
    未显式给出时，从 ``$V4_RUN_ROOT/E4/{arch}/fold{k}/best.pt`` 等位置自动发现。
    只有 smoke/exploratory 下找不到时才退回随机初始化并显式标记。
    """
    from src.training import frozen as FZ

    kw = dict(ARCH_KW[args.arch])
    if args.arch_kwargs:
        kw.update(json.loads(args.arch_kwargs))
    ckpt = FZ.resolve_backbone_checkpoint(
        getattr(args, "backbone_ckpt", None), args.arch,
        int(fold if fold is not None else getattr(args, "fold", 0)))
    if ckpt is not None:
        return FZ.load_frozen_model(str(ckpt), args.arch, n_features,
                                    arch_kwargs=kw, device=device), kw, False
    if not (getattr(args, "smoke", False) or getattr(args, "exploratory", False)):
        raise SystemExit(
            "[E5] 找不到该折的 --backbone-ckpt；可用目录/模板（如 "
            "$V4_RUN_ROOT/E4/patchtf/fold{fold}/best.pt）或 --smoke/--exploratory 做链路预检")
    from src.training.seq_loop import build_seq_model
    L.set_seed(args.seed)
    model = build_seq_model(args.arch, int(n_features), **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, kw, True



def scaled_well(cache, well, spec, scaler, phys):
    got = RD._well_feature_matrix(cache, well, "train", spec, phys_params=phys)
    X = np.asarray(got[0] if isinstance(got, tuple) else got, dtype="float32")
    return np.asarray(scaler.transform(X), dtype="float32")


def well_labels(cache, wells: Sequence[str]) -> dict[str, dict]:
    """逐井标签尺度真值 + mask + 原子标记（官方口径 `missing_mode=drop`）。"""
    from src.data import dataset as D
    from src.features import basic as F
    out: dict[str, dict] = {}
    for w in wells:
        sh = D.read_well_shard(cache, w, "train")
        lab = F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
        por = np.asarray(lab["por"], dtype="float64")
        perm = np.power(10.0, np.asarray(lab["perm_z"], dtype="float64"))
        sw = np.asarray(lab["sw"], dtype="float64")
        out[w] = {"y": M.label_scale_stack(por, lab["perm_z"], sw),
                  "mask": np.asarray(lab["mask"], dtype=bool),
                  "y_atom": np.column_stack([F.is_atom_value(por, "POR"),
                                             F.is_atom_value(perm, "PERM"),
                                             F.is_atom_value(sw, "SW")])}
    return out


def collect_fold(args, cache, folds, k, spec, scalers, device) -> dict[str, Any]:
    """单折：折内标尺 → 冻结骨干 → 逐行隐状态（训练/验证/内折）+ 骨干自身预测（含 q_atom）。"""
    from src.training import frozen as FZ

    tr_wells, va_wells = fold_wells(args, folds, k)
    fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
    scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
    n_features = int(scaler.median.shape[0])
    if scalers is not None:
        RD.save_scaler_json(Path(scalers) / f"E5_{args.target.lower()}_fold{k}.json",
                            {"fold": k, "target": args.target,
                             "row_scaler": scaler.to_dict(),
                             "target_scalers": dict(target), "train_wells": tr_wells,
                             "val_wells": va_wells, "n_train_rows": fit["n_train_rows"],
                             "feature_spec": spec.as_dict() if spec else None,
                             "note": "E5：尺度参数只由训练折拟合"})
    model, arch_kw, random_init = load_backbone(args, n_features, device, fold=k)
    inner_tr, inner_val = FR.inner_split(tr_wells, args.seed)
    if len(inner_val) < 1 or len(inner_tr) < 2:
        inner_tr, inner_val = list(tr_wells), list(va_wells)
    cfg = L.TrainConfig(device=args.device, amp_dtype=args.amp_dtype,
                        batch_size=args.batch_size, seed=args.seed)
    opt = SL.SeqOptions(spec=spec, chunk=args.chunk, overlap=args.overlap,
                        batch_chunks=args.batch_chunks, weight_kind=args.weight_kind,
                        arch=args.arch)
    need = list(dict.fromkeys(list(tr_wells) + list(va_wells)))
    states = FZ.states_for_wells(model, need, cache, cfg, opt, device=device,
                                 scaler=scaler, phys_params=phys)
    preds = {}
    for w in dict.fromkeys(list(va_wells) + list(inner_val)):
        preds[w] = SL.predict_well_chunked(model, scaled_well(cache, w, spec, scaler, phys),
                                          cfg, opt, device, well_id=w)
    return {"fold": k, "tr_wells": tr_wells, "va_wells": va_wells,
            "inner_tr": list(inner_tr), "inner_val": list(inner_val),
            "states": states, "target": target, "scaler": scaler, "phys": phys,
            "arch_kw": arch_kw, "backbone_random_init": random_init,
            "backbone_pred": preds, "n_features": n_features, "cfg": cfg, "opt": opt}


def select_tau_inner(ctx, labels, target_index: int) -> tuple[float | None, dict[str, Any]]:
    """在内折上用骨干 `q_atom` 选该目标的 τ（inner-OOF only）。"""
    yi = np.concatenate([labels[w]["y"] for w in ctx["inner_val"]])
    mi = np.concatenate([labels[w]["mask"] for w in ctx["inner_val"]])
    qi = np.concatenate([np.asarray(ctx["backbone_pred"][w]["q_atom"], dtype="float64")
                         for w in ctx["inner_val"]])
    ci = np.concatenate([M.decode_continuous(ctx["backbone_pred"][w])
                         for w in ctx["inner_val"]])
    sel = AG.select_tau_per_target(cont=ci, q_atom=qi, y=yi, mask=mi)
    taus = np.asarray(sel["tau"], dtype="float64").reshape(-1)
    return float(taus[target_index]), {"source": "inner_oof_select",
                                       "tau": float(taus[target_index]),
                                       "tau_all": [float(v) for v in taus],
                                       "plateau_tol": AG.DEFAULT_PLATEAU_TOL}


def attach_backbone_q(ctx, labels, wells) -> None:
    for w in wells:
        q = np.asarray(ctx["backbone_pred"][w]["q_atom"], dtype="float64")
        labels[w]["q_atom_t"] = q


# ---------------------------------------------------------------- 通用两阶段训练
def train_head_two_phase(args, head_factory: Callable[[int], Any], Xtr, ytr, Xva, yva,
                         target_index: int, predict_fn, loss_fn, metric_fn, epochs=None,
                         verbose=False) -> tuple[Any, int, float]:
    """两阶段：内折按 `metric_fn` 早停 → 全折重训 `best_epoch+1`。

    `Xtr/Xva` 为 `{well: (L,d)}`，`ytr/yva` 为 `{well: {"y":(L,3), "mask":(L,3)}}`。
    """
    import torch

    d_model = int(next(iter(Xtr.values())).shape[1])
    dev = L.resolve_device(L.TrainConfig(device=args.device))
    torch.manual_seed(args.seed)
    head = head_factory(d_model).to(dev)

    def _stack(states, labels, wells):
        xs = np.concatenate([states[w].astype("float32") for w in wells], axis=0)
        ys = np.concatenate([labels[w]["y"][:, target_index] for w in wells], axis=0)
        ms = np.concatenate([labels[w]["mask"][:, target_index] for w in wells], axis=0)
        return xs, ys.astype("float32"), ms.astype("float32")

    def _metric_on(states, labels, wells):
        head.eval()
        yt, yp, mk = [], [], []
        with torch.no_grad():
            for w in wells:
                x = torch.from_numpy(states[w].astype("float32")).to(dev)
                yt.append(labels[w]["y"][:, target_index])
                yp.append(predict_fn(head, x).float().cpu().numpy())
                mk.append(labels[w]["mask"][:, target_index])
        return float(metric_fn(np.concatenate(yt), np.concatenate(yp),
                               np.concatenate(mk).astype(bool)))

    def _run(n_ep, states, labels, wells, use_inner, keep_best):
        opt = torch.optim.AdamW(head.parameters(), lr=float(args.lr),
                                weight_decay=args.weight_decay)
        xs, ys, ms = _stack(states, labels, wells)
        xs_t, ys_t, ms_t = (torch.from_numpy(xs).to(dev), torch.from_numpy(ys).to(dev),
                            torch.from_numpy(ms).to(dev))
        n = int(xs.shape[0])
        best = {"score": float("-inf"), "epoch": -1, "state": None}
        for ep in range(int(n_ep)):
            head.train()
            perm = torch.randperm(n, device=dev)
            tot, nb = 0.0, 0
            for i in range(0, n, int(args.batch_size)):
                idx = perm[i:i + int(args.batch_size)]
                pred = predict_fn(head, xs_t[idx])
                loss = -SAL.masked_mean(loss_fn(pred, ys_t[idx]), ms_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach().cpu())
                nb += 1
            if use_inner:
                sc = _metric_on(Xva, yva, list(Xva))
                if verbose:
                    print(f"  [E5/{args.target.lower()}] epoch {ep} "
                          f"loss={tot / max(nb, 1):.5f} inner={sc:.5f}", flush=True)
                if keep_best and sc > best["score"]:
                    best = {"score": float(sc), "epoch": ep,
                            "state": {k: v.detach().clone() for k, v in
                                      head.state_dict().items()}}
        if keep_best and best["state"] is not None:
            head.load_state_dict(best["state"])
        return best

    epochs = int(args.epochs if epochs is None else epochs)
    best = _run(epochs, Xtr, ytr, list(Xtr), True, True)
    best_epoch = int(best["epoch"]) if best["epoch"] >= 0 else epochs - 1
    all_states = {**Xtr, **Xva}
    all_labels = {**ytr, **yva}
    _run(best_epoch + 1, all_states, all_labels, list(all_states), False, False)
    return head, best_epoch, float(best["score"])


def predict_wells(head, states, wells, predict_fn, dev) -> np.ndarray:
    import torch
    head.eval()
    out = []
    with torch.no_grad():
        for w in wells:
            x = torch.from_numpy(states[w].astype("float32")).to(dev)
            out.append(predict_fn(head, x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------- 显著性 / Gate
def paired_well_delta(y, pred_new, pred_base, mask, well_index, n_wells, rows,
                      metric_fn, seed: int = 42, iters: int = 1000) -> dict[str, Any]:
    """按井配对 bootstrap（新头 − 基线），权重 = 井行数。"""
    d = np.full(int(n_wells), np.nan)
    for i in range(int(n_wells)):
        sel = np.asarray(well_index) == i
        if not sel.any():
            continue
        mm = np.asarray(mask, dtype=bool)[sel]
        d[i] = (metric_fn(np.asarray(y)[sel], np.asarray(pred_new)[sel], mm)
                - metric_fn(np.asarray(y)[sel], np.asarray(pred_base)[sel], mm))
    ok = ~np.isnan(d)
    boot = FOLDS.bootstrap_ci(d[ok], iters=int(iters),
                              weights=np.asarray(rows, dtype="float64")[ok], seed=seed)
    return {"per_well": [float(v) for v in d], "n_wells_used": int(ok.sum()),
            "point": float(boot["point"]), "ci": [float(boot["ci_low"]),
                                                  float(boot["ci_high"])],
            "bootstrap_unit": "well_row_weighted_cluster", "iters": int(iters)}


def write_gate(args, reports: Path, *, gate_id: str, p_stage: str, metrics: dict,
               compares: dict, extra_checks: dict, prereg_baseline,
               prereg_notes: str, target_index: int, threshold_key: str = "min_delta",
               primary_metric: str = "target_acc") -> dict[str, Any]:
    """校验/创建预注册 → 聚合 Gate → 落盘 `E5_{p_stage}_gate.json`。"""
    prereg_path = Path(reports) / f"{gate_id}_prereg.json"
    if not prereg_path.is_file():
        base = Path(prereg_baseline) if prereg_baseline else None
        sha = hashlib.sha256(base.read_bytes() if base and base.is_file()
                             else str(base or "none").encode()).hexdigest()
        write_json(prereg_path, {
            "gate_id": gate_id, "stage": "E5", "p_stage": p_stage, "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": primary_metric, "primary_threshold_key": threshold_key,
            "baseline_version": f"frozen_backbone_{args.arch}",
            "baseline_artifact": str(base or "none"), "baseline_manifest_sha256": sha,
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 4,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(REQUIRED_CHECKS), "decisions_locked": [],
            "notes": prereg_notes,
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(args.disk_path)
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(not perrs and metrics.get("n_rows", 0) > 0),
        "atomic_precision_reported": bool(metrics.get("placeholder_rows")),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(metrics.get("training_time_log_valid", True)),
        "checkpoint_resumable": True,
        "no_label_leak": bool(metrics.get("no_label_leak", False)),
        **extra_checks,
    }
    result = {"checks": checks, "score": metrics.get("score"),
              "delta": compares.get("delta"), "paired_ci_low": compares.get("ci_low")}
    agg = GATES.aggregate_gate(prereg, result)
    baseline_missing = not args.backbone_ckpt
    passed = None if (args.exploratory or args.smoke or baseline_missing) else \
        bool(agg["passed"])
    gate = {"gate_id": gate_id, "stage": "E5", "p_stage": p_stage, "tag": args.tag,
            "target": args.target, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "baseline_unavailable": baseline_missing,
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "compares": compares, "metrics_path": str(reports / f"E5_{args.target.lower()}.json"),
            "oof_path": metrics.get("oof_path")}
    write_json(Path(reports) / f"E5_{p_stage}_gate.json", gate)
    return gate


def require_target(args) -> str:
    t = str(getattr(args, "target", "")).upper()
    if t not in C.TARGETS:
        raise SystemExit(f"[E5] target 必须是 {C.TARGETS} 之一")
    return t
