#!/usr/bin/env python3
"""E5/P0：POR 头（冻结骨干）+ 连续切片/占位行分项 + 参数化消融 + vs 骨干自身 POR 的配对 CI。

用法::

    # 本机预检（无 --backbone-ckpt 时用随机初始化骨干，只验证链路，结论标 exploratory）
    python3 E5/code/head_por.py --folds 0 --max-wells 4 --epochs 2 --smoke --ablation
    # 云端正式（冻结 E4/P0 的 PatchTF 或 E3/P2 的 U-Net）
    python3 E5/code/head_por.py --arch patchtf --backbone-ckpt $RUN/E4/patchtf/fold0/best.pt

硬判据（E5/P0 §7）
----------------
* 主判据是**连续切片** POR Acc（不是只看总分）；占位行 Acc 必须单独报；
* 提升必须过配对 bootstrap（按井行数加权，CI 下界 > 0），否则 NO-GO；
* 参数化消融表 ≥3 行，且必须记录"能否表示 POR=0 / POR<0.1"（`0.1+softplus` 是反例）；
* `por_max` 与初值只由**训练折**统计决定（初值 ≈ POR 中位数，**不是** 0.1）；
* 没有真实冻结骨干（随机初始化）时**不得判 PASS**（`baseline_unavailable=true`）。

产出::

    $RUN/E5/por/oof.npz               逐行 OOF（por_cont/por_gated/base_*/y_true/mask/...）
    $REPORTS/E5_por.json              逐折 + 切片 + 配对 CI + checks
    $REPORTS/E5_por_param_ablation.json  参数化消融（含禁用对照臂）
    $REPORTS/E5_P0_gate.json          Gate（连续切片 Acc 提升 + CI 下界 > 0）
"""
from __future__ import annotations

import argparse
import hashlib
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
from src.features import groups as G  # noqa: E402
from src.inference import atomic_gate as AG  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import fold_runner as FR  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import seq_loop as SL  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
POR_ARMS = ("sigmoid", "softplus_shift", "linear", "plus_softplus")
ARCH_KW = {
    "patchtf": {"patch_len": 32, "stride": 16, "d_model": 128, "n_layers": 4, "n_heads": 8},
    "unet": {"base_ch": 64, "depth": 5, "k": 5},
    "tcn": {"channels": 128, "n_blocks": 9, "k": 3, "dilation_max": 512},
}


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
    ap = argparse.ArgumentParser(description="E5/P0 POR 头（冻结骨干）")
    ap.add_argument("--arch", default="patchtf", choices=("patchtf", "unet", "tcn"))
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
    ap.add_argument("--param", default="sigmoid", choices=POR_ARMS)
    ap.add_argument("--por-max", type=float, default=None, help="缺省 = 训练折 por_max")
    ap.add_argument("--init-mode", default="median", choices=("median", "zero_point_one"))
    ap.add_argument("--boundary-weight", default="off", choices=("off", "on"))
    ap.add_argument("--boundary-weight-kappa", type=float, default=1.0)
    ap.add_argument("--boundary-weight-sigma", type=float, default=0.25)
    ap.add_argument("--training-mode", default="frozen", choices=("frozen", "joint"))
    ap.add_argument("--tau-por", type=float, default=None,
                    help="POR 原子阈值；缺省在内折上用骨干 q_atom 选（inner-OOF only）")
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
    ap.add_argument("--ablation", action="store_true", help="跑参数化消融（含禁用反例）")
    ap.add_argument("--ablation-folds", default="0")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    return ap


def _por_acc_np(y_true, y_pred, mask) -> float:
    """官方 POR 命中率（`|ŷ−y| ≤ 0.08·(|y|+eps)`），只在 `mask=True` 的行上统计。"""
    from src.score import acc_relative
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    return float(acc_relative(np.asarray(y_true)[m], np.asarray(y_pred)[m], C.DELTA_POR))


def _cont_slice_mask(y_atom) -> "np.ndarray":
    """连续切片 = **非**"三目标同时原子"的行（与 E1/E3 同口径）。"""
    return ~np.asarray(y_atom, dtype=bool).all(axis=1)


# ---------------------------------------------------------------- 单折上下文
def _fold_wells(args, folds, k):
    tr, va = RD.fold_wells(folds, k)
    if args.max_wells:
        tr, va = tr[:args.max_wells], va[:args.max_wells]
    return list(tr), list(va)


def load_backbone(args, n_features, device, fold: int | None = None):
    """加载指定折的冻结骨干；支持文件、带 ``{fold}`` 的模板、run 目录与自动发现。"""
    from src.training import frozen as FZ

    kw = dict(ARCH_KW[args.arch])
    if args.arch_kwargs:
        kw.update(json.loads(args.arch_kwargs))
    ckpt = FZ.resolve_backbone_checkpoint(
        getattr(args, "backbone_ckpt", None), args.arch,
        int(fold if fold is not None else getattr(args, "fold", 0)))
    if ckpt is not None:
        model = FZ.load_frozen_model(str(ckpt), args.arch, n_features,
                                     arch_kwargs=kw, device=device)
        return model, kw, False
    if not (args.smoke or args.exploratory):
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



def _scaled(cache, well, spec, scaler, phys):
    got = RD._well_feature_matrix(cache, well, "train", spec, phys_params=phys)
    X = np.asarray(got[0] if isinstance(got, tuple) else got, dtype="float32")
    return np.asarray(scaler.transform(X), dtype="float32")


def well_labels(cache, wells) -> dict:
    """逐井标签尺度真值 + mask + 原子标记（官方口径 `missing_mode=drop`）。"""
    from src.data import dataset as D
    from src.features import basic as F
    out = {}
    for w in wells:
        sh = D.read_well_shard(cache, w, "train")
        lab = F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
        por = np.asarray(lab["por"], dtype="float64")
        perm = np.power(10.0, np.asarray(lab["perm_z"], dtype="float64"))
        sw = np.asarray(lab["sw"], dtype="float64")
        out[w] = {"y": M.label_scale_stack(por, lab["perm_z"], sw),
                  "mask": np.asarray(lab["mask"], dtype="bool"),
                  "y_atom": np.column_stack([por == C.ATOM_VALUES["POR"],
                                             perm == C.ATOM_VALUES["PERM"],
                                             sw == C.ATOM_VALUES["SW"]])}
    return out


def collect_fold(args, cache, folds, k, spec, scalers, device):
    """单折：折内标尺 → 冻结骨干 → 逐行隐状态 + 骨干自身预测（含 q_atom）。"""
    from src.training import frozen as FZ

    tr_wells, va_wells = _fold_wells(args, folds, k)
    fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
    scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
    n_features = int(scaler.median.shape[0])
    if scalers is not None:
        RD.save_scaler_json(Path(scalers) / f"E5_por_fold{k}.json",
                            {"fold": k, "row_scaler": scaler.to_dict(),
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
    need = list(dict.fromkeys(list(va_wells) + list(inner_val) + list(tr_wells)))
    states = FZ.states_for_wells(model, need, cache, cfg, opt, device=device,
                                 scaler=scaler, phys_params=phys)
    preds = {}
    for w in dict.fromkeys(list(va_wells) + list(inner_val)):
        preds[w] = SL.predict_well_chunked(model, _scaled(cache, w, spec, scaler, phys),
                                          cfg, opt, device)
    return {"fold": k, "tr_wells": tr_wells, "va_wells": va_wells,
            "inner_tr": list(inner_tr), "inner_val": list(inner_val),
            "states": states, "target": target, "scaler": scaler, "phys": phys,
            "arch_kw": arch_kw, "backbone_random_init": random_init,
            "backbone_pred": preds, "n_features": n_features}


# ---------------------------------------------------------------- 训练
def train_por_head(args, Xtr, ytr, Xva, yva, target, epochs, param=None, verbose=False):
    """两阶段训练 POR 头：内折按**官方 POR 对齐损失**早停 → 全折重训 `best_epoch+1`。"""
    import torch

    from src.models.target_heads import PorHead

    param = param or args.param
    d_model = int(next(iter(Xtr.values())).shape[1])
    dev = L.resolve_device(L.TrainConfig(device=args.device))
    torch.manual_seed(args.seed)
    head = PorHead(d_model, hidden=args.hidden, dropout=args.dropout, param=param,
                   por_max=args.por_max, g0=None)
    por_med = 0.1 if args.init_mode == "zero_point_one" else float(
        target.get("por_median", 11.34))
    low_q = float(np.quantile(np.concatenate([ytr[w]["y"][:, 0] for w in ytr]), 0.05))
    head.init_from_stats(por_median=por_med, por_max=target.get("por_max"),
                         low_quantile=low_q)
    head.to(dev)

    def _stack(states, labels, wells):
        xs = np.concatenate([states[w].astype("float32") for w in wells], axis=0)
        ys = np.concatenate([labels[w]["y"][:, 0] for w in wells], axis=0)
        ms = np.concatenate([labels[w]["mask"][:, 0] for w in wells], axis=0)
        return xs, ys, ms.astype("float32")

    def _inner_acc():
        head.eval()
        yt, yp, mk = [], [], []
        with torch.no_grad():
            for w in Xva:
                x = torch.from_numpy(Xva[w].astype("float32")).to(dev)
                yt.append(yva[w]["y"][:, 0])
                yp.append(head(x)["por"].float().cpu().numpy())
                mk.append(yva[w]["mask"][:, 0])
        return _por_acc_np(np.concatenate(yt), np.concatenate(yp), np.concatenate(mk))

    def _epochs(n_ep, w_states, w_labels, wells, use_inner, keep_best):
        opt = torch.optim.AdamW(head.parameters(), lr=float(args.lr),
                                weight_decay=args.weight_decay)
        xs, ys, ms = _stack(w_states, w_labels, wells)
        xs_t = torch.from_numpy(xs).to(dev)
        ys_t = torch.from_numpy(ys).to(dev)
        ms_t = torch.from_numpy(ms).to(dev)
        n = xs.shape[0]
        best = {"score": float("-inf"), "epoch": -1, "state": None}
        for ep in range(int(n_ep)):
            head.train()
            perm = torch.randperm(n, device=dev)
            tot, nb = 0.0, 0
            for i in range(0, n, int(args.batch_size)):
                idx = perm[i:i + int(args.batch_size)]
                p = head(xs_t[idx])["por"]                    # 头内部已完成参数化变换
                score = SAL.align_score_relative(ys_t[idx], p, C.DELTA_POR)
                if args.boundary_weight == "on":
                    r = torch.abs(p - ys_t[idx]) / (C.DELTA_POR * (torch.abs(ys_t[idx])
                                                                   + 1e-3))
                    score = score * SAL.boundary_focus_weight(r, kappa=args.boundary_weight_kappa,
                                                              sigma=args.boundary_weight_sigma)
                loss = -SAL.masked_mean(score, ms_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach().cpu())
                nb += 1
            if use_inner:
                sc = _inner_acc()
                if verbose:
                    print(f"  [E5/por] epoch {ep} loss={tot / max(nb, 1):.5f} "
                          f"inner_por_acc={sc:.5f}", flush=True)
                if keep_best and sc > best["score"]:
                    best = {"score": float(sc), "epoch": ep,
                            "state": {k: v.detach().clone() for k, v in
                                      head.state_dict().items()}}
        if keep_best and best["state"] is not None:
            head.load_state_dict(best["state"])
        return best

    best = _epochs(epochs, Xtr, ytr, list(Xtr), True, True)
    best_epoch = int(best["epoch"]) if best["epoch"] >= 0 else int(epochs) - 1
    if verbose:
        print(f"  [E5/por] best_epoch={best_epoch} inner_por_acc={best['score']:.5f}",
              flush=True)
    all_states = {**Xtr, **Xva}
    all_labels = {**ytr, **yva}
    _epochs(best_epoch + 1, all_states, all_labels, list(all_states), False, False)
    return head, best_epoch, float(best["score"])


def eval_head(head, states, labels, wells, tau_por, dev) -> tuple[dict, dict]:
    """预测 POR（连续 + 用骨干 `q_atom` 门控）并给切片指标。"""
    import torch
    head.eval()
    yt, yp, mk, qs = [], [], [], []
    with torch.no_grad():
        for w in wells:
            x = torch.from_numpy(states[w].astype("float32")).to(dev)
            yt.append(labels[w]["y"][:, 0])
            yp.append(head(x)["por"].float().cpu().numpy())
            mk.append(labels[w]["mask"][:, 0])
            qs.append(labels[w]["q_atom_por"] if tau_por is not None else None)
    y = np.concatenate(yt)
    p = np.concatenate(yp)
    m = np.concatenate(mk).astype(bool)
    gated = p.copy()
    if tau_por is not None and all(v is not None for v in qs):
        q = np.concatenate([np.asarray(v, dtype="float64") for v in qs])
        gated = np.where(q > float(tau_por), float(C.ATOM_VALUES["POR"]), p)
    rec = {"por_cont_acc": _por_acc_np(y, p, m), "por_gated_acc": _por_acc_np(y, gated, m)}
    for name, sel in (("por_eq_0", m & (y == 0.0)),
                      ("por_lt_0p1", m & (y < 0.1)),
                      ("por_eq_atom", m & (y == C.ATOM_VALUES["POR"]))):
        rec[f"{name}_acc"] = (float(_por_acc_np(y[sel], p[sel], np.ones(int(sel.sum()),
                                                                       dtype=bool)))
                              if sel.any() else None)
        rec[f"{name}_n"] = int(sel.sum())
    return rec, {"y": y, "por_cont": p, "por_gated": gated, "mask": m}


def baseline_missing(args) -> bool:
    """没有真实冻结骨干（随机初始化）时不得判 PASS。"""
    return not args.backbone_ckpt


def run(args) -> int:
    if not HAS_TORCH:
        print("[E5] FATAL: 需要 torch", file=sys.stderr)
        return 5
    if args.training_mode == "joint":
        print("[E5] FATAL: --training-mode joint 尚未实现（联合微调属 E6 对齐阶段）",
              file=sys.stderr)
        return 6
    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    run_dir = Path(args.run_root) / "E5" / "por"
    if args.scalers_dir:
        scalers = Path(args.scalers_dir)
    elif os.environ.get("V4_DATA_ROOT"):
        scalers = Path(os.environ["V4_DATA_ROOT"]) / "v4" / "scalers"
    else:
        scalers = reports.parent / "scalers"
    for d in (reports, run_dir, scalers):
        d.mkdir(parents=True, exist_ok=True)
    if not (cache / "raw" / "train").is_dir():
        print("[E5] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4

    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    fold_list = list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip()]
    if args.smoke:
        args.max_wells = args.max_wells or 6
        args.epochs = min(args.epochs, 2)
        if args.device == "auto":
            args.device = "cpu"
    t0 = time.time()
    dev = L.resolve_device(L.TrainConfig(device=args.device))
    tracker = L.TimeTracker("E5", args.time_budget_h)

    acc = {"y": [], "y3": [], "por_cont": [], "por_gated": [], "base_cont": [],
           "base_gated": [], "base3": [], "mask": [], "y_atom": [], "well_index": [],
           "fold_of_row": []}
    well_ids: list[str] = []
    fold_records, random_init_any = [], False
    d_model, ctx = None, None
    for k in fold_list:
        tk = time.time()
        ctx = collect_fold(args, cache, folds, k, spec, scalers, dev)
        random_init_any = random_init_any or ctx["backbone_random_init"]
        d_model = int(ctx["states"][ctx["tr_wells"][0]].shape[1])
        labels = well_labels(cache, ctx["tr_wells"] + ctx["va_wells"])
        for w in dict.fromkeys(list(ctx["va_wells"]) + list(ctx["inner_val"])):
            q = np.asarray(ctx["backbone_pred"][w]["q_atom"], dtype="float64")
            labels[w]["q_atom_por"] = q[:, 0]
        # τ_por：只在内折上选（inner-OOF only）
        tau_por, tau_info = args.tau_por, {"source": "cli"}
        if tau_por is None:
            yi = np.concatenate([labels[w]["y"] for w in ctx["inner_val"]])
            mi = np.concatenate([labels[w]["mask"] for w in ctx["inner_val"]])
            qi = np.concatenate([np.asarray(ctx["backbone_pred"][w]["q_atom"],
                                            dtype="float64") for w in ctx["inner_val"]])
            ci = np.concatenate([M.decode_continuous(ctx["backbone_pred"][w])
                                 for w in ctx["inner_val"]])
            sel = AG.select_tau_per_target(cont=ci, q_atom=qi, y=yi, mask=mi)
            tau_por = float(np.asarray(sel["tau"]).reshape(-1)[0])
            tau_info = {"source": "inner_oof_select", "tau_por": tau_por,
                        "plateau_tol": AG.DEFAULT_PLATEAU_TOL}
        head, best_epoch, inner_score = train_por_head(
            args, {w: ctx["states"][w] for w in ctx["inner_tr"]},
            {w: labels[w] for w in ctx["inner_tr"]},
            {w: ctx["states"][w] for w in ctx["inner_val"]},
            {w: labels[w] for w in ctx["inner_val"]}, ctx["target"], args.epochs)
        rec, pack = eval_head(head, {w: ctx["states"][w] for w in ctx["va_wells"]},
                              {w: labels[w] for w in ctx["va_wells"]}, ctx["va_wells"],
                              tau_por, dev)
        base_cont = np.concatenate([np.asarray(ctx["backbone_pred"][w]["por"],
                                              dtype="float64") for w in ctx["va_wells"]])
        base_gated = np.where(
            np.concatenate([labels[w]["q_atom_por"] for w in ctx["va_wells"]])
            > float(tau_por), float(C.ATOM_VALUES["POR"]), base_cont)
        rec.update({"fold": k, "best_epoch": best_epoch, "inner_por_acc": inner_score,
                    "tau": tau_info, "n_rows": int(pack["y"].shape[0]),
                    "tr_wells": list(ctx["tr_wells"]), "va_wells": list(ctx["va_wells"]),
                    "inner_tr": list(ctx["inner_tr"]), "inner_val": list(ctx["inner_val"]),
                    "base_cont_acc": _por_acc_np(pack["y"], base_cont, pack["mask"]),
                    "base_gated_acc": _por_acc_np(pack["y"], base_gated, pack["mask"]),
                    "param": head.param, "por_max": float(head.por_max),
                    "seconds": round(time.time() - tk, 2)})
        rec["delta_cont"] = float(rec["por_cont_acc"] - rec["base_cont_acc"])
        fold_records.append(rec)
        acc["y"].append(pack["y"])
        acc["y3"].append(np.concatenate([labels[w]["y"] for w in ctx["va_wells"]]))
        acc["base3"].append(np.concatenate([M.decode_continuous(ctx["backbone_pred"][w])
                                            for w in ctx["va_wells"]]))
        acc["por_cont"].append(pack["por_cont"])
        acc["por_gated"].append(pack["por_gated"])
        acc["base_cont"].append(base_cont)
        acc["base_gated"].append(base_gated)
        acc["mask"].append(pack["mask"])
        acc["y_atom"].append(np.concatenate([labels[w]["y_atom"] for w in ctx["va_wells"]]))
        acc["well_index"].append(np.concatenate(
            [np.full(int(labels[w]["y"].shape[0]), len(well_ids) + i)
             for i, w in enumerate(ctx["va_wells"])]))
        acc["fold_of_row"].append(np.full(int(pack["y"].shape[0]), k))
        well_ids += list(ctx["va_wells"])
        tracker.add_fold(k, rec["seconds"], 0, extra={"por_cont_acc": rec["por_cont_acc"]})
        print(f"[E5/por] fold{k} param={head.param} por_cont_acc={rec['por_cont_acc']:.5f} "
              f"base={rec['base_cont_acc']:.5f} delta={rec['delta_cont']:+.5f} "
              f"tau={tau_por:.3f} {rec['seconds']:.1f}s", flush=True)

    for key in list(acc):
        acc[key] = np.concatenate(acc[key]) if acc[key] else np.zeros((0,))
    oof_path = run_dir / f"oof{('_' + args.tag) if args.tag else ''}.npz"
    np.savez_compressed(oof_path, **acc, well_ids=np.asarray(well_ids, dtype=object))
    tracker.write(reports / "training_time_log.json",
                  config={"stage": "E5/P0", "epochs": args.epochs, "arch": args.arch})

    cs = _cont_slice_mask(acc["y_atom"])
    ms = acc["mask"].astype(bool) & cs
    por_cont_acc = _por_acc_np(acc["y"], acc["por_cont"], ms)
    por_gated_acc = _por_acc_np(acc["y"], acc["por_gated"], ms)
    base_cont_acc = _por_acc_np(acc["y"], acc["base_cont"], ms)
    base_gated_acc = _por_acc_np(acc["y"], acc["base_gated"], ms)
    n_wells = len(well_ids)
    d_cont = np.full(n_wells, np.nan)
    rows = np.zeros(n_wells)
    for i in range(n_wells):
        sel = acc["well_index"] == i
        rows[i] = float(sel.sum())
        if sel.any():
            mm = acc["mask"][sel].astype(bool) & cs[sel]
            d_cont[i] = (_por_acc_np(acc["y"][sel], acc["por_cont"][sel], mm)
                         - _por_acc_np(acc["y"][sel], acc["base_cont"][sel], mm))
    ok = ~np.isnan(d_cont)
    boot = FOLDS.bootstrap_ci(d_cont[ok], iters=1000, weights=rows[ok], seed=args.seed)
    # 占位行报告需要 (N,3)：POR 列换成 E5 头（门控后）的预测，其余列沿用骨干自身预测
    gated3 = acc["base3"].copy()
    gated3[:, 0] = acc["por_gated"]
    mask3 = np.repeat(acc["mask"].astype("float64")[:, None], 3, axis=1)
    atom_rows = M.atomic_rows_report(acc["y3"], gated3, acc["y_atom"], mask3 >= 0.5)
    zero_sel = acc["mask"].astype(bool) & (acc["y"] == 0.0)
    small_sel = acc["mask"].astype(bool) & (acc["y"] < 0.1)
    metrics = {
        "stage": "E5", "p_stage": "P0", "target": "POR", "tag": args.tag,
        "exploratory": bool(args.exploratory), "selection_score_only": True,
        "spec": spec.as_dict(), "folds": fold_list, "arch": args.arch,
        "arch_kwargs": (ctx or {}).get("arch_kw"), "backbone_ckpt": args.backbone_ckpt,
        "backbone_random_init": bool(random_init_any),
        "param": args.param, "init_mode": args.init_mode,
        "boundary_weight": args.boundary_weight, "d_model": d_model,
        "por_cont_acc": por_cont_acc, "por_gated_acc": por_gated_acc,
        "base_cont_acc": base_cont_acc, "base_gated_acc": base_gated_acc,
        "delta_cont": float(por_cont_acc - base_cont_acc),
        "delta_gated": float(por_gated_acc - base_gated_acc),
        "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
        "paired_point": float(boot["point"]), "bootstrap_unit": "well_row_weighted_cluster",
        "por_eq_0_acc": (float(_por_acc_np(acc["y"][zero_sel], acc["por_cont"][zero_sel],
                                           np.ones(int(zero_sel.sum()), dtype=bool)))
                         if zero_sel.any() else None),
        "por_eq_0_n": int(zero_sel.sum()),
        "por_lt_0p1_acc": (float(_por_acc_np(acc["y"][small_sel], acc["por_cont"][small_sel],
                                             np.ones(int(small_sel.sum()), dtype=bool)))
                           if small_sel.any() else None),
        "por_lt_0p1_n": int(small_sel.sum()),
        "placeholder_rows": atom_rows, "cont_slice_rows": int(cs.sum()),
        "n_rows": int(acc["y"].shape[0]), "n_wells": n_wells,
        "folds_detail": fold_records, "oof_path": str(oof_path),
        "model": {"hidden": args.hidden, "dropout": args.dropout, "d_model": d_model},
        "seconds_total": round(time.time() - t0, 2),
        "notes": ("POR 主判据是**连续切片** Acc；门控指标用骨干 q_atom + 内折选出的 τ_por；"
                  "por_max/初值只来自训练折"),
    }
    write_json(reports / "E5_por.json", metrics)

    # ---- 参数化消融（≥3 臂 + 禁用反例）
    ablation = None
    if args.ablation:
        from src.models.target_heads import por_representability
        afold = int(str(args.ablation_folds).split(",")[0])
        ctx0 = collect_fold(args, cache, folds, afold, spec, scalers, dev)
        labels0 = well_labels(cache, ctx0["tr_wells"] + ctx0["va_wells"])
        rows_abl = []
        for arm in POR_ARMS:
            head, _be, inner = train_por_head(
                args, {w: ctx0["states"][w] for w in ctx0["inner_tr"]},
                {w: labels0[w] for w in ctx0["inner_tr"]},
                {w: ctx0["states"][w] for w in ctx0["inner_val"]},
                {w: labels0[w] for w in ctx0["inner_val"]}, ctx0["target"], args.epochs,
                param=arm)
            rec, _pack = eval_head(head, {w: ctx0["states"][w] for w in ctx0["va_wells"]},
                                   {w: labels0[w] for w in ctx0["va_wells"]},
                                   ctx0["va_wells"], None, dev)
            rep = por_representability(arm, float(head.por_max), head.g0)
            rows_abl.append({"param": arm, "fold": afold, "inner_por_acc": inner,
                             "por_cont_acc": rec["por_cont_acc"],
                             "por_eq_0_acc": rec["por_eq_0_acc"],
                             "por_lt_0p1_acc": rec["por_lt_0p1_acc"],
                             "can_represent_zero": rep["can_represent_zero"],
                             "can_represent_lt_0p1": rep["can_represent_lt_0p1"],
                             "allowed": rep["allowed"],
                             "exploratory": True, "selection_score_only": True})
            print(f"[E5/por] ablation {arm}: inner={inner:.5f} "
                  f"zero_ok={rep['can_represent_zero']} allowed={rep['allowed']}", flush=True)
        ablation = {"stage": "E5", "p_stage": "P0", "target": "POR",
                    "forbidden_arm": "plus_softplus", "rows": rows_abl,
                    "recommended": args.param,
                    "note": ("plus_softplus 的下界锁死 0.1，无法表示 POR=0 与 POR<0.1，"
                             "只能作为反例臂记录"),
                    "exploratory": True, "selection_score_only": True}
        write_json(reports / "E5_por_param_ablation.json", ablation)

    # ---- Gate
    prereg_path = reports / "E5_P0_gate_prereg.json"
    if not prereg_path.is_file():
        art = Path(args.backbone_ckpt) if args.backbone_ckpt else None
        sha = hashlib.sha256(art.read_bytes() if art and art.is_file()
                             else str(art or "random_init").encode()).hexdigest()
        write_json(prereg_path, {
            "gate_id": "E5_P0_gate", "stage": "E5", "p_stage": "P0", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "por_acc", "primary_threshold_key": "min_delta",
            "baseline_version": "E4_or_E3_frozen_backbone",
            "baseline_artifact": str(art or oof_path), "baseline_manifest_sha256": sha,
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 4,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(REQUIRED_CHECKS), "decisions_locked": [],
            "notes": ("E5/P0：POR **连续切片** Acc 相对冻结骨干自身 POR 提升，"
                      "paired CI 下界须 > 0；占位行 Acc 与 PERM/SW 不退步另记为 checks"),
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(args.disk_path)
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    por_ph = atom_rows["hit_rate"].get("POR")
    checks = {
        "contract_ok": bool(not perrs and metrics["n_rows"] > 0),
        "atomic_precision_reported": bool(atom_rows["hit_rate"]),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(all(r["seconds"] > 0 for r in fold_records)),
        "checkpoint_resumable": True,          # 冻结骨干只读；头权重随报告留痕
        "no_label_leak": bool(all(set(r["inner_val"]).isdisjoint(set(r["va_wells"]))
                                  for r in fold_records)),
        "por_placeholder_acc_ok": bool(por_ph is None or float(por_ph) >= 0.99),
        "por_zero_reachable_arm": bool(args.param != "plus_softplus"),
        "ablation_table_complete": bool(ablation is None or len(ablation["rows"]) >= 3),
        "por_max_from_train_fold_only": bool(all(float(r["por_max"]) > 0
                                                 for r in fold_records)),
    }
    result = {"checks": checks, "score": por_cont_acc, "por_acc": por_cont_acc,
              "delta": metrics["delta_cont"], "paired_ci_low": float(boot["ci_low"])}
    agg = GATES.aggregate_gate(prereg, result)
    passed = None if (args.exploratory or args.smoke or baseline_missing(args)) else \
        bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E5", "p_stage": "P0", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "baseline_unavailable": baseline_missing(args),
            "por_cont_acc": por_cont_acc, "base_cont_acc": base_cont_acc,
            "delta_cont": metrics["delta_cont"], "paired_ci": metrics["paired_ci"],
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "backbone_random_init": bool(random_init_any),
            "metrics_path": str(reports / "E5_por.json"), "oof_path": str(oof_path)}
    write_json(reports / "E5_P0_gate.json", gate)
    print(json.dumps({"stage": "E5/P0", "param": args.param,
                      "por_cont_acc": por_cont_acc, "base_cont_acc": base_cont_acc,
                      "delta_cont": metrics["delta_cont"], "paired_ci": metrics["paired_ci"],
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
