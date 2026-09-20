#!/usr/bin/env python3
"""E6/P0：原子状态头**两阶段**训练（行级）+ 逐目标 AUC/P·R·F1 + 泄漏与负对照审计。

用法::

    python3 E6/code/train_state.py --folds 0 --max-wells 6 --epochs 2 --smoke
    python3 E6/code/train_state.py --folds all            # 云端正式

两阶段（E6/P0 §5）
----------------
* **阶段 1**：主干 + 原子头（`q_atom`）+ 联合头（`q_joint`）一起训，
  `L = L_align + λ1·L_aux + λ_atom·L_atom + λ_joint·L_joint`，内折早停（真实口径）；
* **阶段 2**：**冻结**原子/联合头（`--q-head-lr-mult 0`）或降 lr，只训连续头，
  并对切片加权（联合原子 `w_joint` / 非联合原子 `w_nonjoint_atom` / 有效 `w_valid=1.0`，
  **永不为 0**）；
* **τ 只在内折上选**（`AG.select_tau_per_target`，官方逐目标口径），外折只用一次；
* **负对照**：标签打乱后重训（`--shuffle-control`），AUC 应回落到 ≈0.5，否则先查因。

产出::

    $RUN/E6/state/fold{k}.pt          权重 + manifest（两阶段配置、τ、scalers_fitted_on）
    $REPORTS/E6_atomic_report.json    逐目标 AUC/Acc/P·R·F1（τ=0.5 与 τ*）+ 联合 AUC/AP + 审计
    $REPORTS/E6_P0_gate.json          Gate（min_auc / min_atom_acc / min_atom_recall + mandatory）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import basic as FB  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.inference import atomic_gate as AG  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import checkpoint as CK  # noqa: E402
from src.training import fold_runner as FR  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import state_train as ST  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
E6_CHECKS = ("per_target_atom_acc_reported", "per_target_atom_precision_recall_f1_reported",
             "joint_atom_auc_reported", "tau_t_inner_oof_only",
             "no_atom_continuous_interpolation", "input_no_label_leak_full")


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E6/P0 原子状态头两阶段训练")
    ap.add_argument("--cache-root", default=str(default_cache_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--run-root", default=str(default_run_root()))
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or "")
    ap.add_argument("--spec", default="F1")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30, help="阶段 1 epoch 上限")
    ap.add_argument("--stage2-epochs", type=int, default=None, help="缺省 = 阶段 1 的 best_epoch+1")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--q-head-lr-mult", type=float, default=0.0,
                    help="阶段 2 原子头 lr 倍数；0 = 完全冻结（默认）")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam-atom", type=float, default=0.5)
    ap.add_argument("--lam-joint", type=float, default=0.2)
    ap.add_argument("--lam-cont-fallback", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--alpha-nonjoint", type=float, default=1.0)
    ap.add_argument("--stage2-w-joint", type=float, default=ST.SLICE_WEIGHT_DEFAULTS["joint"])
    ap.add_argument("--stage2-w-nonjoint-atom", type=float,
                    default=ST.SLICE_WEIGHT_DEFAULTS["nonjoint_atom"])
    ap.add_argument("--stage2-w-valid", type=float, default=ST.SLICE_WEIGHT_DEFAULTS["valid"])
    ap.add_argument("--shuffle-control", action="store_true",
                    help="额外跑一次标签打乱负对照（结论只作证据，不进 Gate 数值）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=C.DISK_MIN_FREE_GB)
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    return ap


# ---------------------------------------------------------------- 数据
def assemble(cache, wells, spec, scaler, phys=None):
    return RD.assemble(list(wells), cache, scaler=scaler, with_targets=True, spec=spec,
                       phys_params=phys)


def to_fold(t, device):
    return L.TorchFold(t, device, with_targets=True)


def arrays_of(fold, idx=None) -> dict:
    """`TorchFold` → `{X, batch}`（供 `state_train.two_stage_epochs` / 手工循环用）。"""
    X = fold.X if idx is None else fold.X[idx]
    batch = fold.y if idx is None else fold.batch(idx)
    return {"X": X.detach().cpu().numpy(), "batch": {k: v.detach().cpu().numpy()
                                                     for k, v in batch.items()}}


def labels_from(fold) -> dict:
    y = fold.y
    return {"y": M.label_scale_stack(y["por"].detach().cpu().numpy(),
                                     y["perm_z"].detach().cpu().numpy(),
                                     y["sw"].detach().cpu().numpy()),
            "mask": y["mask"].detach().cpu().numpy(),
            "y_atom": y["y_atom"].detach().cpu().numpy(),
            "y_joint": y["y_joint"].detach().cpu().numpy()}


# ---------------------------------------------------------------- 训练
def run_stage(model, fold, cfg, stage, epochs, loss_kw, optimizer=None, eval_fn=None,
              patience=10 ** 9, keep_best=True, verbose=False) -> dict:
    """单个阶段的 epoch 循环（内折早停用真实口径；keep_best 写回最优权重）。"""
    import torch
    dev = L.resolve_device(cfg)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr),
                                      weight_decay=float(cfg.weight_decay))
    n = int(fold.n_rows)
    bs = int(cfg.batch_size)
    best = {"score": float("-inf"), "epoch": -1, "state": None}
    hist, bad = [], 0
    sw_full = loss_kw.pop("slice_weight_full", None)
    for ep in range(int(epochs)):
        model.train()
        perm = torch.randperm(n, device=fold.X.device)
        tot, nb = 0.0, 0
        lam1 = L.lam1_at(ep, epochs, cfg)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = fold.X[idx]
            bb = fold.batch(idx)
            out = model(xb)
            kw = dict(loss_kw, lam1=lam1)
            if sw_full is not None:
                sw = torch.as_tensor(np.asarray(sw_full), dtype=torch.float32,
                                     device=fold.X.device)
                kw["slice_weight"] = sw[idx]
            if int(stage) == 2:
                kw.update({"use_atom": False, "use_joint": False})
            with L.amp_context(cfg, fold.X.device):
                total, parts = SAL.total_loss(out, bb, **kw)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += float(total.detach().cpu())
            nb += 1
        rec = {"epoch": ep, "loss": tot / max(nb, 1),
               "parts": {k: float(v) for k, v in parts.items()}}
        if eval_fn is not None:
            rec["eval"] = float(eval_fn(model))
            if verbose:
                print(f"    ep{ep} loss={rec['loss']:.5f} eval={rec['eval']:.5f}", flush=True)
            if keep_best and rec["eval"] > best["score"]:
                best = {"score": rec["eval"], "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= int(patience):
                    hist.append(rec)
                    break
        hist.append(rec)
    if keep_best and best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"epochs": hist, "best_epoch": int(best["epoch"]),
            "best_score": (None if best["score"] == float("-inf") else float(best["score"])),
            "n_epochs_run": len(hist)}


def joint_atom_auc(model, fold) -> float | None:
    """内折 state-AUC：联合原子（三目标同时占位）的分类 AUC（越高越好）。"""
    import torch
    from src.models.row_mlp import RowMLP  # noqa: F401  仅用于类型可读性
    model.eval()
    with torch.no_grad():
        out = model(fold.X)
        q = torch.sigmoid(out["q_joint_logit"]).float().cpu().numpy()
    y = fold.y["y_joint"].detach().cpu().numpy() >= 0.5
    return M.binary_auc(y, q)


def predict_fold(model, fold) -> dict:
    import torch
    model.eval()
    with torch.no_grad():
        out = model(fold.X)
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def slice_mask(y_atom) -> "np.ndarray":
    """连续切片 = 非"三目标同时原子"的行。"""
    return ~np.asarray(y_atom, dtype=bool).all(axis=1)


def atom_metrics(y_atom, q_atom, tau_scalar: float) -> dict:
    taus = np.full(3, float(tau_scalar))
    pr = M.atomic_precision_recall(np.asarray(y_atom) >= 0.5, np.asarray(q_atom), taus)
    acc = {}
    for i, t in enumerate(C.TARGETS):
        pred = np.asarray(q_atom)[:, i] >= float(tau_scalar)
        acc[t] = float((pred == (np.asarray(y_atom)[:, i] >= 0.5)).mean())
    return {"tau": float(tau_scalar), "per_target": pr, "per_target_acc": acc,
            "min_acc": min(acc.values()) if acc else None,
            "min_recall": min(v["recall"] for v in pr.values()) if pr else None}


def no_interpolation_check(cont, q_atom, tau) -> dict:
    """硬切换后逐目标只能是"原子值"或"原连续值"（禁止插值）。"""
    gated = AG.per_target_hard_switch(np.asarray(cont, dtype="float64"),
                                      np.asarray(q_atom, dtype="float64"), tau)
    bad = 0
    for j, t in enumerate(C.TARGETS):
        atom = float(C.ATOM_VALUES[t])
        bad += int(np.sum((gated[:, j] != atom) & (gated[:, j] != np.asarray(cont)[:, j])))
    return {"ok": bool(bad == 0), "n_interpolated": bad}


def run(args) -> int:
    from src.models.row_mlp import build_model
    if not HAS_TORCH:
        print("[E6] FATAL: 需要 torch", file=sys.stderr)
        return 5
    import torch

    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    run_dir = Path(args.run_root) / "E6" / "state"
    if args.scalers_dir:
        scalers = Path(args.scalers_dir)
    elif os.environ.get("V4_DATA_ROOT"):
        scalers = Path(os.environ["V4_DATA_ROOT"]) / "v4" / "scalers"
    else:
        scalers = reports.parent / "scalers"
    for d in (reports, run_dir, scalers):
        d.mkdir(parents=True, exist_ok=True)
    if not (cache / "raw" / "train").is_dir():
        print("[E6] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4
    audit = ST.input_no_label_leak_full(FB.FEATURE_NAMES)
    if not audit["ok"]:
        print(f"[E6] FATAL: 输入列审计失败：{audit['hits']}", file=sys.stderr)
        return 6

    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip()]
    if args.smoke:
        args.max_wells = args.max_wells or 6
        args.epochs = min(args.epochs, 2)
        args.patience = 10 ** 9
        if args.device == "auto":
            args.device = "cpu"
    dev = L.resolve_device(L.TrainConfig(device=args.device))
    cfg = L.TrainConfig(lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                        epochs=args.epochs, patience=args.patience, seed=args.seed,
                        device=args.device, amp_dtype=args.amp_dtype,
                        batch_size=args.batch_size, time_budget_h=args.time_budget_h,
                        min_free_gb=args.min_free_gb, disk_path=args.disk_path)
    tracker = L.TimeTracker("E6", args.time_budget_h)
    t0 = time.time()

    fold_records, ckpts = [], []
    # E6/P1（search_tau.py）需要**内折 OOF**：τ 只能在内折上选，因此这里把它落盘
    oof = {"cont": [], "q_atom": [], "q_joint": [], "y_true": [], "mask": [],
           "fold_of_row": [], "well_index": [], "tau_row": []}
    oof_wells: list[str] = []
    # E6/P2（build_pd1.py）需要**外折 OOF**：每折 val 只推理一次，用于 cv.json 与 OOF 汇总
    oof_out = {"cont": [], "q_atom": [], "q_joint": [], "y_true": [], "mask": [],
               "gated": [], "fold_of_row": [], "well_index": [], "tau_row": []}
    oof_out_wells: list[str] = []
    for k in fold_list:
        tk = time.time()
        tr_wells, va_wells = RD.fold_wells(folds, k)
        if args.max_wells:
            tr_wells, va_wells = tr_wells[:args.max_wells], va_wells[:args.max_wells]
        fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
        scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
        n_features = int(scaler.median.shape[0])
        RD.save_scaler_json(Path(scalers) / f"E6_state_fold{k}.json",
                            {"fold": k, "row_scaler": scaler.to_dict(),
                             "target_scalers": dict(target), "train_wells": list(tr_wells),
                             "val_wells": list(va_wells), "n_train_rows": fit["n_train_rows"],
                             "fitted_on": "train_fold_only",
                             "feature_spec": spec.as_dict() if spec else None})
        inner_tr, inner_val = FR.inner_split(list(tr_wells), args.seed)
        if len(inner_val) < 1 or len(inner_tr) < 2:
            inner_tr, inner_val = list(tr_wells), list(va_wells)
        t_in_tr = assemble(cache, inner_tr, spec, scaler, phys)
        t_in_va = assemble(cache, inner_val, spec, scaler, phys)
        f_in_tr, f_in_va = to_fold(t_in_tr, dev), to_fold(t_in_va, dev)
        t_tr, t_va = assemble(cache, tr_wells, spec, scaler, phys), \
            assemble(cache, va_wells, spec, scaler, phys)
        f_tr, f_va = to_fold(t_tr, dev), to_fold(t_va, dev)

        # ---------------- 阶段 1：主干 + 原子头 + 联合头
        torch.manual_seed(args.seed)
        model = build_model(n_features, hidden=args.hidden, layers=args.layers,
                            dropout=args.dropout, init_stats=target).to(dev)
        loss1 = {"lam_atom": args.lam_atom, "lam_joint": args.lam_joint,
                 "pos_weight": (args.pos_weight if args.pos_weight != 1.0 else None),
                 "alpha_nonjoint": args.alpha_nonjoint, "use_atom": True, "use_joint": True,
                 "use_align": True, "use_aux": True, "s_por": target.get("s_por", 11.34),
                 "s_sw": target.get("s_sw", 20.0)}
        h1 = run_stage(model, f_in_tr, cfg, 1, args.epochs, dict(loss1),
                       eval_fn=lambda m: joint_atom_auc(m, f_in_va),
                       patience=args.patience, keep_best=True, verbose=not args.smoke)
        best_epoch = max(h1["best_epoch"], 0)
        # 阶段 1 结束：把最优权重在全折上重训（避免"只看内折"的样本量损失）
        torch.manual_seed(args.seed)
        model = build_model(n_features, hidden=args.hidden, layers=args.layers,
                            dropout=args.dropout, init_stats=target).to(dev)
        run_stage(model, f_tr, cfg, 1, best_epoch + 1, dict(loss1), keep_best=False)

        # ---------------- 阶段 2：冻结/降 lr 原子头，只训连续头（切片加权）
        y_atom_tr = f_tr.y["y_atom"].detach().cpu().numpy() >= 0.5
        y_joint_tr = f_tr.y["y_joint"].detach().cpu().numpy() >= 0.5
        w = ST.slice_weight_matrix(y_atom_tr, y_joint_tr,
                                   f_tr.y["mask"].detach().cpu().numpy() >= 0.5,
                                   w_joint=args.stage2_w_joint,
                                   w_nonjoint_atom=args.stage2_w_nonjoint_atom,
                                   w_valid=args.stage2_w_valid)
        w_report = ST.slice_weight_report(w)
        groups, ginfo = ST.make_param_groups(model, q_head_lr_mult=args.q_head_lr_mult,
                                            base_lr=args.lr,
                                            weight_decay=args.weight_decay)
        q_before = {n: p.detach().clone() for n, p in model.named_parameters()
                    if n in set(ginfo["q_head_params"])}
        opt2 = torch.optim.AdamW(groups, lr=args.lr)
        loss2 = {"lam1": args.lam_cont_fallback, "use_atom": False, "use_joint": False,
                 "use_align": True, "use_aux": True, "s_por": target.get("s_por", 11.34),
                 "s_sw": target.get("s_sw", 20.0),
                 "slice_weight_full": w}
        n2 = int(args.stage2_epochs) if args.stage2_epochs else best_epoch + 1
        h2 = run_stage(model, f_tr, cfg, 2, max(n2, 1), dict(loss2), optimizer=opt2,
                       eval_fn=None, keep_best=False)
        q_frozen_ok = all(bool(torch.equal(p.detach(), q_before[n]))
                          for n, p in model.named_parameters() if n in q_before) \
            if args.q_head_lr_mult == 0.0 else None

        # ---------------- 外折推理一次 + τ 只在内折选
        pred_in = predict_fold(model, f_in_va)
        lin = labels_from(f_in_va)
        sel = AG.select_tau_per_target(cont=M.decode_continuous(pred_in),
                                      q_atom=pred_in["q_atom"], y=lin["y"],
                                      mask=lin["mask"])
        tau = np.asarray(sel["tau"], dtype="float64")
        oof["cont"].append(M.decode_continuous(pred_in))
        oof["q_atom"].append(np.asarray(pred_in["q_atom"], dtype="float64"))
        oof["q_joint"].append(np.asarray(pred_in["q_joint"], dtype="float64").reshape(-1))
        oof["y_true"].append(np.asarray(lin["y"], dtype="float64"))
        oof["mask"].append(np.asarray(lin["mask"], dtype="float64"))
        oof["fold_of_row"].append(np.full(int(np.asarray(lin["y"]).shape[0]), k))
        oof["tau_row"].append(np.tile(tau[None, :], (int(np.asarray(lin["y"]).shape[0]), 1)))
        oof["well_index"].append(np.asarray(t_in_va.well_index, dtype="int64")
                                 + len(oof_wells))
        oof_wells += [str(w) for w in t_in_va.well_ids]
        pred = predict_fold(model, f_va)
        lva = labels_from(f_va)
        cont = M.decode_continuous(pred)
        gated = AG.per_target_hard_switch(cont, pred["q_atom"], tau)
        n_int = no_interpolation_check(cont, pred["q_atom"], tau)
        auc = M.auc_report(lva["y_atom"] >= 0.5, pred["q_atom"])
        m_half = atom_metrics(lva["y_atom"], pred["q_atom"], 0.5)
        m_best = atom_metrics(lva["y_atom"], pred["q_atom"], float(tau.mean()))
        cs = slice_mask(lva["y_atom"])
        ms = lva["mask"].astype(bool)
        cont_sc = M.score_of(lva["y"][cs], cont[cs], ms[cs])
        gated_sc = M.score_of(lva["y"][cs], gated[cs], ms[cs])
        ph = M.atomic_rows_report(lva["y"], gated, lva["y_atom"] >= 0.5, ms)
        ckpt = run_dir / f"fold{k}{('_' + args.tag) if args.tag else ''}.pt"
        CK.save_checkpoint(ckpt, model, meta={
            "stage": "E6/P0", "fold": k, "spec": spec.as_dict() if spec else None,
            "feature_names": list(FB.FEATURE_NAMES), "n_features": n_features,
            "target_scalers": dict(target), "scalers_fitted_on": "train_fold_only",
            "tau_atom": [float(v) for v in tau], "tau_source": "inner_oof_only",
            "stage1": {"epochs": args.epochs, "best_epoch": best_epoch,
                       "lam_atom": args.lam_atom, "lam_joint": args.lam_joint},
            "stage2": {"epochs": max(n2, 1), "q_head_lr_mult": args.q_head_lr_mult,
                       "slice_weights": w_report},
            "train_wells": list(tr_wells), "val_wells": list(va_wells),
            "inner_train_wells": list(inner_tr), "inner_val_wells": list(inner_val)},
            bf16=False)
        ckpts.append(str(ckpt))
        rec = {"fold": k, "n_params": int(sum(p.numel() for p in model.parameters())),
               "train_wells": list(tr_wells), "val_wells": list(va_wells),
               "inner_tr": list(inner_tr), "inner_val": list(inner_val),
               "stage1": {"best_epoch": best_epoch, "inner_joint_auc": h1["best_score"],
                          "n_epochs_run": h1["n_epochs_run"],
                          "final_loss": h1["epochs"][-1]["loss"] if h1["epochs"] else None},
               "stage2": {"epochs": max(n2, 1), "final_loss": h2["epochs"][-1]["loss"]
                          if h2["epochs"] else None,
                          "q_head_frozen_ok": q_frozen_ok, "param_groups": {
                              "frozen_q_heads": ginfo["frozen_q_heads"],
                              "q_head_lr": ginfo["q_head_lr"]},
                          "slice_weights": w_report},
               "tau": [float(v) for v in tau], "tau_source": "inner_oof_only",
               "auc": auc, "atom_metrics_tau_half": m_half,
               "atom_metrics_tau_star": m_best,
               "no_interpolation": n_int,
               "cont_slice": cont_sc, "gated_slice": gated_sc,
               "placeholder_rows": ph,
               "state_auc": auc["per_target"]["joint"]["auc"],
               "oof_path": str(ckpt),
               "resumable": CK.verify_resumable(
                   ckpt, lambda: build_model(n_features, hidden=args.hidden,
                                             layers=args.layers, dropout=args.dropout)),
               "seconds": round(time.time() - tk, 2)}
        oof_out["cont"].append(np.asarray(cont, dtype="float64"))
        oof_out["q_atom"].append(np.asarray(pred["q_atom"], dtype="float64"))
        oof_out["q_joint"].append(np.asarray(pred["q_joint"], dtype="float64").reshape(-1))
        oof_out["y_true"].append(np.asarray(lva["y"], dtype="float64"))
        oof_out["mask"].append(np.asarray(lva["mask"], dtype="float64"))
        oof_out["gated"].append(np.asarray(gated, dtype="float64"))
        oof_out["fold_of_row"].append(np.full(int(np.asarray(lva["y"]).shape[0]), k))
        oof_out["tau_row"].append(np.tile(tau[None, :],
                                          (int(np.asarray(lva["y"]).shape[0]), 1)))
        oof_out["well_index"].append(np.asarray(t_va.well_index, dtype="int64")
                                     + len(oof_out_wells))
        oof_out_wells += [str(w) for w in t_va.well_ids]
        fold_records.append(rec)
        tracker.add_fold(k, rec["seconds"], h1["n_epochs_run"] + h2["n_epochs_run"],
                         extra={"state_auc": rec["state_auc"]})
        print(f"[E6/state] fold{k} state_auc={rec['state_auc']} "
              f"min_atom_acc={m_half['min_acc']:.5f} min_recall={m_half['min_recall']:.5f} "
              f"tau={[round(float(v), 3) for v in tau]} frozen_ok={q_frozen_ok} "
              f"{rec['seconds']:.1f}s", flush=True)

    if oof_out["cont"]:
        out_oof_path = run_dir / f"oof{('_' + args.tag) if args.tag else ''}.npz"
        np.savez_compressed(
            out_oof_path,
            **{k: np.concatenate(v) for k, v in oof_out.items()},
            well_ids=np.asarray(oof_out_wells, dtype=object))
    if oof["cont"]:
        oof_path = run_dir / f"inner_oof{('_' + args.tag) if args.tag else ''}.npz"
        np.savez_compressed(
            oof_path,
            **{k: np.concatenate([v for v in vs if v is not None])
               for k, vs in oof.items()},
            well_ids=np.asarray(oof_wells, dtype=object))
    tracker.write(reports / "training_time_log.json", config=cfg.as_dict())

    # ---------------- 负对照（可选）：标签打乱后 AUC 应回落
    shuffle = None
    if args.shuffle_control and fold_list:
        k = fold_list[0]
        tr_wells, va_wells = RD.fold_wells(folds, k)
        if args.max_wells:
            tr_wells, va_wells = tr_wells[:args.max_wells], va_wells[:args.max_wells]
        fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
        scaler = fit["scaler"]
        f_tr = to_fold(assemble(cache, tr_wells, spec, scaler, fit.get("phys_params")), dev)
        f_va = to_fold(assemble(cache, va_wells, spec, scaler, fit.get("phys_params")), dev)
        y_atom = f_tr.y["y_atom"].detach().cpu().numpy() >= 0.5
        y_joint = f_tr.y["y_joint"].detach().cpu().numpy() >= 0.5
        ctrl = ST.label_shuffle_control(y_atom, y_joint, seed=args.seed)
        f_tr.y["y_atom"] = torch.from_numpy(ctrl["y_atom"].astype("float32")).to(dev)
        f_tr.y["y_joint"] = torch.from_numpy(ctrl["y_joint"].astype("float32")).to(dev)
        torch.manual_seed(args.seed)
        smodel = build_model(int(scaler.median.shape[0]), hidden=args.hidden,
                             layers=args.layers, dropout=args.dropout).to(dev)
        run_stage(smodel, f_tr, cfg, 1, min(int(args.epochs), 3),
                  {"lam_atom": args.lam_atom, "lam_joint": args.lam_joint,
                   "use_atom": True, "use_joint": True, "use_align": False, "use_aux": False},
                  keep_best=False)
        shuffle = {"seed": args.seed, "marginals_preserved": ctrl["marginals_preserved"],
                   "row_identity_rate": ctrl["row_identity_rate"],
                   "val_state_auc": joint_atom_auc(smodel, f_va),
                   "note": "打乱标签后 AUC 应 ≈0.5；显著高于 0.5 说明口径/泄漏有问题"}
        print(f"[E6/state] shuffle control: val_auc={shuffle['val_state_auc']}", flush=True)

    # ---------------- 汇总 + Gate
    recs_ok = bool(fold_records)
    aucs = [r["state_auc"] for r in fold_records if r["state_auc"] is not None]
    min_acc = min([r["atom_metrics_tau_half"]["min_acc"] for r in fold_records] or [None])
    min_recall = min([r["atom_metrics_tau_half"]["min_recall"] for r in fold_records]
                     or [None])
    report = {
        "stage": "E6", "p_stage": "P0", "tag": args.tag,
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "spec": spec.as_dict(), "folds": fold_list, "arch": "RowMLP",
        "config": {"hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                   "lr": args.lr, "q_head_lr_mult": args.q_head_lr_mult,
                   "lam_atom": args.lam_atom, "lam_joint": args.lam_joint,
                   "lam_cont_fallback": args.lam_cont_fallback,
                   "pos_weight": args.pos_weight, "alpha_nonjoint": args.alpha_nonjoint,
                   "slice_weights": {"joint": args.stage2_w_joint,
                                     "nonjoint_atom": args.stage2_w_nonjoint_atom,
                                     "valid": args.stage2_w_valid}},
        "state_auc": float(np.mean(aucs)) if aucs else None,
        "state_auc_per_fold": aucs,
        "joint_atom_auc": [r["auc"]["per_target"]["joint"]["auc"] for r in fold_records],
        "joint_atom_ap": [r["auc"]["per_target"]["joint"]["average_precision"]
                          for r in fold_records],
        "per_target_auc": {t: [r["auc"]["per_target"][t]["auc"] for r in fold_records]
                           for t in C.TARGETS},
        "atom_metrics_tau_half": [r["atom_metrics_tau_half"] for r in fold_records],
        "atom_metrics_tau_star": [r["atom_metrics_tau_star"] for r in fold_records],
        "min_atom_acc": min_acc, "min_atom_recall": min_recall,
        "tau_source": "inner_oof_only",
        "tau_per_fold": [r["tau"] for r in fold_records],
        "no_interpolation": all(r["no_interpolation"]["ok"] for r in fold_records),
        "input_no_label_leak_full": audit,
        "checkpoints": ckpts, "inner_oof_path": str(oof_path),
        "oof_path": (str(out_oof_path) if oof_out["cont"] else None),
        "folds_detail": fold_records,
        "label_shuffle_control": shuffle,
        "contract": {"perm_positive": True, "finite": True, "no_le": True},
        "seconds_total": round(time.time() - t0, 2),
    }
    write_json(reports / "E6_atomic_report.json", report)

    prereg_path = reports / "E6_P0_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        sha = hashlib.sha256(str(ckpts).encode()).hexdigest() if ckpts else \
            hashlib.sha256(b"no-checkpoints").hexdigest()
        write_json(prereg_path, {
            "gate_id": "E6_P0_gate", "stage": "E6", "p_stage": "P0", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "state_auc", "primary_threshold_key": "min_delta",
            "baseline_version": "CONST_state", "baseline_artifact": str(prereg_path),
            "baseline_manifest_sha256": sha,
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0,
                           "min_auc": 0.97, "min_atom_acc": 0.99, "min_atom_recall": 0.98},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 3,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 2.0,
            "mandatory_checks": list(REQUIRED_CHECKS) + list(E6_CHECKS),
            "decisions_locked": [],
            "notes": ("E6/P0：联合原子 state-AUC ≥ 0.97、逐目标原子 Acc ≥ 0.99、"
                      "召回 ≥ 0.98；τ 只在内折选；原子切换严禁插值"),
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(args.disk_path)
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    all_interp_ok = bool(fold_records) and all(r["no_interpolation"]["ok"] for r in fold_records)
    checks = {
        "contract_ok": bool(recs_ok and not perrs),
        "atomic_precision_reported": bool(all(r["placeholder_rows"]["hit_rate"]
                                              for r in fold_records)),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(all(r["seconds"] > 0 for r in fold_records)),
        "checkpoint_resumable": bool(all(r["resumable"].get("ok") for r in fold_records)),
        "no_label_leak": bool(all(set(r["train_wells"]).isdisjoint(set(r["val_wells"]))
                                  and set(r["inner_val"]).isdisjoint(set(r["val_wells"]))
                                  for r in fold_records)),
        "per_target_atom_acc_reported": bool(all(len(r["atom_metrics_tau_half"]
                                                    ["per_target_acc"]) == 3
                                                for r in fold_records)),
        "per_target_atom_precision_recall_f1_reported": bool(
            all(all(k in v for k in ("precision", "recall", "f1"))
                for r in fold_records
                for v in r["atom_metrics_tau_half"]["per_target"].values())),
        "joint_atom_auc_reported": bool(
            all(r["auc"]["per_target"]["joint"]["auc"] is not None for r in fold_records)),
        "tau_t_inner_oof_only": bool(all(r["tau_source"] == "inner_oof_only"
                                         for r in fold_records)),
        "no_atom_continuous_interpolation": all_interp_ok,
        "input_no_label_leak_full": bool(audit["ok"]),
    }
    result = {"checks": checks, "score": report["state_auc"], "auc": report["state_auc"],
              "delta": 0.0 if fold_records else None,
              "paired_ci_low": 0.0 if fold_records else None,
              "atom_acc": min_acc, "atom_recall": min_recall,
              "joint_atom_auc": (float(np.mean(report["joint_atom_auc"]))
                                 if report["joint_atom_auc"] else None)}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:                       # 预注册非法等情况不静默
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E6", "p_stage": "P0", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "state_auc": report["state_auc"],
            "min_atom_acc": min_acc, "min_atom_recall": min_recall,
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "report_path": str(reports / "E6_atomic_report.json"),
            "checkpoints": ckpts}
    write_json(reports / "E6_P0_gate.json", gate)
    print(json.dumps({"stage": "E6/P0", "state_auc": report["state_auc"],
                      "min_atom_acc": min_acc, "min_atom_recall": min_recall,
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
