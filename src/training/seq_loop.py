"""序列训练循环（E3）：chunk 批训练 + **分块重叠推理 + 加权拼接** + 两阶段协议。

与行级 `loop.py` 的分工
-----------------------
行级整折 730k×F 一次上卡；序列模型必须按 chunk 喂（1024 点 × batch_chunks），
但**协议完全相同**：inner-OOF 选 `best_epoch` → 选 τ → 全 outer-train 重训 → outer-val
只推理一次（`fold_runner.inner_split` / `fold_runner.pick_tau` 直接复用，不另写一份）。

推理路径（E3/P1 §5 步 5）
------------------------
`predict_well_chunked`：把整井按 `chunks_for` 切块 → 逐块前向 → `stitch_chunks`
（overlap-add + 加权归一）→ 逐行输出。因此：
  * 输出与输入**逐行同长**（全段 seq2seq，不使用滑窗中心点）；
  * 接缝处由权重归一化保证无跳变（`seam_report` 可复算）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .. import constants as C
from ..data import row_dataset as RD
from ..data import seq_dataset as SD
from ..portability import HAS_TORCH, require
from ..validation import folds as FOLDS
from . import checkpoint as CK
from . import fold_runner as FR
from . import loop as L
from . import metrics as M

OUT_KEYS = ("por", "perm_z", "sw", "q_atom", "q_joint")


@dataclass
class SeqOptions:
    spec: Any = None
    chunk: int = SD.DEFAULT_CHUNK
    overlap: int = SD.DEFAULT_OVERLAP
    batch_chunks: int = 4
    weight_kind: str = "triangular"
    max_wells: int | None = None
    smoke: bool = False
    resume: bool = False
    save_checkpoints: bool = True
    select_tau: bool = True
    scaler_prefix: str = "E3"
    arch: str = "unet"
    run_dir: Path | None = None
    scalers_dir: Path | None = None
    tb_run_name: str | None = None
    on_select_epoch: Callable[[int, dict], None] | None = None
    on_final_epoch: Callable[[int, dict], None] | None = None


@dataclass
class SeqFoldResult:
    fold: int
    tau: dict
    best_epoch: int
    inner_oof_total: float | None
    pred: dict[str, Any]          # 逐行（outer-val 整折）
    va_wells: list[str]
    tr_wells: list[str]
    inner_tr_wells: list[str]
    inner_val_wells: list[str]
    fit_wells: list[str]
    scaler: RD.RowScaler
    target: dict
    phys_params: Any
    seconds: float
    hist1: dict
    hist2: dict
    resumable: dict
    coverage: dict = field(default_factory=dict)
    model_summary: dict = field(default_factory=dict)
    spec_dict: dict = field(default_factory=dict)


def build_seq_model(arch: str, n_features: int, init_stats: dict | None = None, **kw):
    require("torch")
    if arch == "unet":
        from ..models.unet1d import build_unet
        return build_unet(n_features, init_stats=init_stats, **kw)
    if arch == "tcn":
        from ..models.tcn import build_tcn
        return build_tcn(n_features, init_stats=init_stats, **kw)
    if arch == "patchtf":
        from ..models.patchtf import build_patchtf
        return build_patchtf(n_features, init_stats=init_stats, **kw)
    raise ValueError(f"unknown seq arch: {arch!r}")


# ---------------------------------------------------------------- 推理
def predict_well_chunked(model, X: "np.ndarray", cfg: L.TrainConfig, opt: SeqOptions,
                         device) -> dict[str, "np.ndarray"]:
    """整井分块推理 + 加权拼接 → 逐行输出（长度 == `X.shape[0]`）。"""
    require("torch")
    import torch

    n = int(X.shape[0])
    chunks = SD.chunks_for(n, opt.chunk, opt.overlap)
    per_key: dict[str, list] = {k: [] for k in OUT_KEYS}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(chunks), max(int(opt.batch_chunks), 1)):
            batch = chunks[i:i + max(int(opt.batch_chunks), 1)]
            xs = np.stack([X[s:s + L] for s, L in batch])
            with L.amp_context(cfg, device):
                out = model(torch.from_numpy(xs).to(device))
            for k in OUT_KEYS:
                v = out[k].float().cpu().numpy()
                for j, (s, ln) in enumerate(batch):
                    per_key[k].append(v[j, :ln])
    stitched: dict[str, Any] = {}
    for k in OUT_KEYS:
        arr = np.concatenate(per_key[k], axis=0) if per_key[k] else np.zeros((0,))
        # 按 chunk 切分回来做 overlap-add（每个 chunk 的预测已按 L 截断）
        pieces, pos = [], 0
        for s, ln in chunks:
            pieces.append(arr[pos:pos + ln])
            pos += ln
        stitched[k] = SD.stitch_chunks(chunks, pieces, n, kind=opt.weight_kind)
    return stitched


# ---------------------------------------------------------------- 训练
def score_wells(preds: dict, wells: Sequence[str], cache, opt: SeqOptions) -> dict[str, Any]:
    """按井打分（官方口径；`missing_mode="drop"`）——直接读 labels 分片，避免整表常驻。"""
    from ..data import dataset as D
    from ..features import basic as F
    from ..score import score_arrays

    y_true, y_pred, mask = [], [], []
    for w in wells:
        sh = D.read_well_shard(cache, w, "train")
        lab = F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
        p = preds[w]
        cont = M.decode_continuous(p)
        tau = p.get("tau")
        gated = M.atom_gate(cont, p["q_atom"], tau) if tau is not None else cont
        y_true.append(M.label_scale_stack(lab["por"], lab["perm_z"], lab["sw"]))
        y_pred.append(gated)
        mask.append(np.asarray(lab["mask"], dtype="float32"))
    yt = np.concatenate(y_true) if y_true else np.zeros((0, 3))
    yp = np.concatenate(y_pred) if y_pred else np.zeros((0, 3))
    m = np.concatenate(mask) if mask else np.zeros((0, 3))
    s = score_arrays(yt, yp, missing=~m.astype(bool), missing_mode=C.SCORE_MISSING_MODE)
    return {"total": float(s["total"]), "acc_por": float(s["acc_por"]),
            "acc_perm": float(s["acc_perm"]), "acc_sw": float(s["acc_sw"]),
            "n_rows": int(yt.shape[0]), "y_true": yt, "y_pred": yp, "mask": m}


def run_two_phase_seq_fold(fold: int, folds: dict, cache, cfg: L.TrainConfig,
                           opt: SeqOptions, arch_kwargs: dict | None = None) -> SeqFoldResult:
    """两阶段序列单折（协议与行级逐字一致，只是把"整折张量"换成"按 chunk 采样"）。"""
    require("torch")
    import torch

    t0 = time.time()
    L.set_seed(cfg.seed)
    dev = L.resolve_device(cfg)
    arch_kwargs = dict(arch_kwargs or {})

    tr_wells, va_wells = RD.fold_wells(folds, fold)
    if opt.max_wells:
        tr_wells, va_wells = tr_wells[:opt.max_wells], va_wells[:opt.max_wells]
    fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=opt.spec)
    scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
    if opt.scalers_dir is not None:
        RD.save_scaler_json(Path(opt.scalers_dir) / f"{opt.scaler_prefix}_fold{fold}.json",
                            {"fold": fold, "row_scaler": scaler.to_dict(),
                             "target_scalers": dict(target), "train_wells": tr_wells,
                             "val_wells": va_wells, "n_train_rows": fit["n_train_rows"],
                             "feature_spec": opt.spec.as_dict() if opt.spec else None,
                             "physics_params": phys.as_dict() if phys else None,
                             "note": "序列主干：尺度参数同样只由训练折拟合"})
    n_features = int(scaler.median.shape[0])

    inner_tr, inner_val = FR.inner_split(tr_wells, cfg.seed)
    if len(inner_val) < 1 or len(inner_tr) < 2:
        if not opt.smoke:
            raise SystemExit(f"[seq fold{fold}] inner split too small "
                             f"({len(inner_tr)}/{len(inner_val)})")
        inner_tr, inner_val = list(tr_wells), list(va_wells)

    def _mk():
        return build_seq_model(opt.arch, n_features, init_stats=target, **arch_kwargs)

    model = _mk().to(dev)
    fold_dir = Path(opt.run_dir) / f"fold{fold}" if opt.run_dir is not None else None
    select_dir = fold_dir / "select" if fold_dir is not None else None
    if select_dir is not None:
        select_dir.mkdir(parents=True, exist_ok=True)
    ds_in = SD.SeqChunkDataset(cache, inner_tr, chunk=opt.chunk, overlap=opt.overlap,
                               split="train", spec=opt.spec, phys_params=phys,
                               seed=cfg.seed, epoch=0, scaler=scaler)

    logger = None
    try:
        from .tb_logger import RunLogger
        logger = RunLogger(opt.tb_run_name or f"{opt.scaler_prefix}_{opt.arch}_fold{fold}")
    except Exception:
        logger = None

    def eval_inner(m) -> dict:
        preds = {}
        for w in inner_val:
            X = scaler.transform(np.asarray(
                RD._well_feature_matrix(cache, w, "train", opt.spec, phys)[0], "float32"))
            preds[w] = predict_well_chunked(m, X, cfg, opt, dev)
        sc = score_wells(preds, inner_val, cache, opt)
        return {"total": sc["total"], "acc_por": sc["acc_por"], "acc_perm": sc["acc_perm"],
                "acc_sw": sc["acc_sw"]}

    # ---- 阶段 1 checkpoint：select_last.pt 每 epoch；select_best.pt 仅在 inner-OOF 提升时。
    opt1 = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr),
                             weight_decay=float(cfg.weight_decay))
    resume1_epoch, resume1_sched = -1, None
    prev_best_manifest: dict[str, Any] | None = None

    def _compatible(man: dict[str, Any]) -> bool:
        # 同一 run_dir/tag 可能在不同搜索/消融配置间复用；配置不一致绝不能 resume。
        return (man.get("arch") == opt.arch
                and man.get("arch_kwargs") == arch_kwargs)

    if select_dir is not None and opt.resume and (select_dir / "last.pt").is_file():
        try:
            last_man = CK.read_manifest(select_dir / "last.pt")
        except Exception:
            last_man = None
        if last_man is not None and _compatible(last_man):
            rr1 = CK.load_for_resume(select_dir / "last.pt", model, optimizer=opt1)
            resume1_epoch = int(rr1["epoch"])
            resume1_sched = rr1.get("scheduler_state")
    if select_dir is not None and (select_dir / "best.pt").is_file():
        try:
            man = CK.read_manifest(select_dir / "best.pt")
            prev_best_manifest = man if _compatible(man) else None
        except Exception:
            prev_best_manifest = None

    select_state: dict[str, Any] = {"best_total": float("-inf"), "best_epoch": -1}
    if prev_best_manifest is not None and prev_best_manifest.get("best_total") is not None:
        select_state["best_total"] = float(prev_best_manifest["best_total"])
        select_state["best_epoch"] = int(prev_best_manifest.get("best_epoch", -1))
    if select_dir is not None and opt.save_checkpoints:
        def _select_meta(epoch: int, total: float | None) -> dict[str, Any]:
            return {"stage": opt.scaler_prefix, "arch": opt.arch, "fold": int(fold),
                    "phase": "select", "epoch": int(epoch),
                    "inner_oof_total": (None if total is None else float(total)),
                    "best_epoch": int(select_state["best_epoch"]),
                    "best_total": (None if select_state["best_total"] == float("-inf")
                                   else float(select_state["best_total"])),
                    "tau_atom": None, "target_scalers": dict(target),
                    "config": cfg.as_dict(), "row_scaler": scaler.to_dict(),
                    "arch_kwargs": arch_kwargs,
                    "feature_spec": opt.spec.as_dict() if opt.spec else None}

        def save_select(epoch: int, rec: dict, model_, opt_, sched_) -> None:
            total = rec.get("val_total")
            if (select_dir / "last.pt").is_file():
                CK.rotate(select_dir)
            CK.save_checkpoint(select_dir / "last.pt", model_,
                               meta=_select_meta(epoch, total), optimizer=opt_, scheduler=sched_)
            if total is not None and float(total) > float(select_state["best_total"]):
                select_state["best_total"] = float(total)
                select_state["best_epoch"] = int(epoch)
                CK.save_checkpoint(select_dir / "best.pt", model_,
                                   meta=_select_meta(epoch, total), optimizer=opt_,
                                   scheduler=sched_)
    else:
        save_select = None

    hist1 = _train_loop(model, ds_in, cfg, eval_fn=eval_inner, opt=opt, dev=dev,
                        scaler_params=target, logger=logger,
                        on_epoch=opt.on_select_epoch, keep_best=True,
                        optimizer=opt1, resume_epoch=resume1_epoch,
                        resume_scheduler_state=resume1_sched,
                        save_hook=save_select)
    # 若上次中断前已有更优的 select_best，恢复它，避免 resume 后丢失旧 best。
    if prev_best_manifest is not None:
        prev_total = prev_best_manifest.get("best_total")
        if (prev_total is not None
                and (hist1["best_total"] is None or float(prev_total) > float(hist1["best_total"]))):
            CK.load_checkpoint(select_dir / "best.pt", model=model)
            hist1["best_epoch"] = int(prev_best_manifest.get("best_epoch", hist1["best_epoch"]))
            hist1["best_total"] = float(prev_total)
            hist1["best_restored"] = True
    best_epoch = hist1["best_epoch"] if hist1["best_epoch"] >= 0 \
        else max(len(hist1["epochs"]) - 1, 0)
    if hist1.get("stopped_reason") in ("time_budget", "paused"):
        # 阶段 1 到点/暂停：select_last.pt 已由 save_hook 落盘；不得继续选 τ / 阶段 2 / 写 OOF。
        if opt.save_checkpoints and select_dir is not None and save_select is not None:
            pass
        raise L.TrainingPaused(
            f"[seq fold{fold}] stage1 stopped: {hist1.get('stopped_reason')}; "
            f"checkpoint={select_dir if select_dir is not None else 'disabled'}")

    tau_info: dict[str, Any] = {"tau": [0.5, 0.5, 0.5], "objective": None, "plateau": {},
                                "score_fn": "skipped"}
    if opt.select_tau and not opt.smoke:
        ipreds = {}
        for w in inner_val:
            X = scaler.transform(np.asarray(
                RD._well_feature_matrix(cache, w, "train", opt.spec, phys)[0], "float32"))
            ipreds[w] = predict_well_chunked(model, X, cfg, opt, dev)
        sc = score_wells(ipreds, inner_val, cache, opt)
        from ..inference import atomic_gate as AG
        sel = AG.select_tau_per_target(cont=M.decode_continuous(
            {k: np.concatenate([ipreds[w][k] for w in inner_val]) for k in OUT_KEYS}),
            q_atom=np.concatenate([ipreds[w]["q_atom"] for w in inner_val]),
            y=sc["y_true"], mask=sc["mask"])
        tau_info = {"tau": [float(v) for v in sel["tau"]],
                    "objective": float(sel["objective"]),
                    "plateau": {k: [float(x) for x in v] for k, v in sel["plateau"].items()},
                    "score_fn": sel["score_fn"]}

    # ---- 阶段 2：全部 outer-train 重训 best_epoch
    cfg2 = L.TrainConfig(**{**cfg.as_dict(), "epochs": max(best_epoch + 1, 1),
                            "patience": 10 ** 9})
    model2 = _mk().to(dev)
    ds_all = SD.SeqChunkDataset(cache, tr_wells, chunk=opt.chunk, overlap=opt.overlap,
                                split="train", spec=opt.spec, phys_params=phys,
                                seed=cfg.seed, epoch=0, scaler=scaler)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=float(cfg.lr),
                             weight_decay=float(cfg.weight_decay))
    fold_dir = Path(opt.run_dir) / f"fold{fold}" if opt.run_dir is not None else None
    resume_epoch = -1
    resume_sched = None
    if fold_dir is not None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        from ..data.disk_guard import register_cache_dirs, register_cleanup_paths
        register_cleanup_paths([fold_dir / "last_prev.pt"])
        register_cache_dirs([Path(cache) / "tmp"])
        if opt.resume and (fold_dir / "last.pt").is_file():
            rr = CK.load_for_resume(fold_dir / "last.pt", model2, optimizer=opt2)
            resume_epoch = int(rr["epoch"])
            resume_sched = rr.get("scheduler_state")

    save_last_hook = capacity_hook = None
    state = {"epoch": int(resume_epoch), "sched": None}
    if fold_dir is not None and opt.save_checkpoints:

        def _meta(ep: int) -> dict[str, Any]:
            return {"stage": opt.scaler_prefix, "arch": opt.arch, "fold": fold,
                    "phase": "final", "epoch": int(ep), "n_epochs_run": int(ep) + 1,
                    "best_epoch_from_inner": int(best_epoch),
                    "inner_oof_total": hist1["best_total"], "tau_atom": tau_info["tau"],
                    "target_scalers": dict(target), "config": cfg.as_dict(),
                    "row_scaler": scaler.to_dict(), "arch_kwargs": arch_kwargs,
                    "feature_spec": opt.spec.as_dict() if opt.spec else None}

        def save_last(epoch: int, rec: dict, model, opt_, sched_) -> None:
            state["epoch"] = int(epoch)
            state["sched"] = sched_
            if (fold_dir / "last.pt").is_file():
                CK.rotate(fold_dir)
            CK.save_checkpoint(fold_dir / "last.pt", model, meta=_meta(epoch),
                               optimizer=opt_, scheduler=sched_)

        def save_on_disk_pressure() -> None:
            ep = int(state["epoch"])
            if (fold_dir / "last.pt").is_file():
                CK.rotate(fold_dir)
            CK.save_checkpoint(fold_dir / "last.pt", model2, meta=_meta(ep),
                               optimizer=opt2, scheduler=state.get("sched"))

        save_last_hook = save_last
        capacity_hook = save_on_disk_pressure

    hist2 = _train_loop(model2, ds_all, cfg2, eval_fn=None, opt=opt, dev=dev,
                        scaler_params=target, logger=None, on_epoch=opt.on_final_epoch,
                        optimizer=opt2, keep_best=False, resume_epoch=resume_epoch,
                        resume_scheduler_state=resume_sched,
                        save_hook=save_last_hook, capacity_hook=capacity_hook)

    if hist2.get("stopped_reason") in ("time_budget", "paused"):
        # 阶段 2 到点/暂停：last.pt 已由 save_hook 落盘，但不得写 best.pt / outer OOF。
        if fold_dir is not None and opt.save_checkpoints:
            ep = (int(hist2["epochs"][-1]["epoch"]) if hist2["epochs"]
                  else int(state.get("epoch", resume_epoch)))
            CK.save_checkpoint(fold_dir / "last.pt", model2, meta=_meta(ep),
                               optimizer=opt2, scheduler=state.get("sched"))
        raise L.TrainingPaused(
            f"[seq fold{fold}] stage2 stopped: {hist2.get('stopped_reason')}; "
            f"checkpoint={fold_dir if fold_dir is not None else 'disabled'}")

    resumable = {"ok": False, "skipped": True}
    if fold_dir is not None and opt.save_checkpoints:
        actual_epoch = (int(hist2["epochs"][-1]["epoch"]) if hist2["epochs"]
                        else int(resume_epoch))
        meta = _meta(actual_epoch)
        meta["n_epochs_run"] = int(hist2["n_epochs_run"])
        if not (fold_dir / "last.pt").is_file():
            CK.save_checkpoint(fold_dir / "last.pt", model2, meta=meta, optimizer=opt2)
        CK.save_checkpoint(fold_dir / "best.pt", model2, meta=meta)
        CK.prune_keep_only(fold_dir, ("best.pt", "last.pt", "last_prev.pt"))
        resumable = CK.verify_resumable(fold_dir / "best.pt", _mk)

    # ---- outer-val 只推理一次（分块 + 拼接）
    pred: dict[str, Any] = {}
    y_true, mask = [], []
    for w in va_wells:
        X = scaler.transform(np.asarray(
            RD._well_feature_matrix(cache, w, "train", opt.spec, phys)[0], "float32"))
        pred[w] = predict_well_chunked(model2, X, cfg2, opt, dev)
        pred[w]["tau"] = np.asarray(tau_info["tau"], dtype="float64")
    sc = score_wells(pred, va_wells, cache, opt)
    if logger is not None:
        logger.close()

    from ..data import seq_dataset as _SD
    cov = {"chunk": opt.chunk, "overlap": opt.overlap, "weight_kind": opt.weight_kind,
           "n_chunks": int(sum(len(_SD.chunks_for(
               int(np.asarray(RD._well_feature_matrix(cache, w, "train", opt.spec,
                                                     phys)[0]).shape[0]),
               opt.chunk, opt.overlap)) for w in va_wells)),
           "wells": len(va_wells), "rows": int(sc["n_rows"])}
    return SeqFoldResult(
        fold=int(fold), tau=tau_info, best_epoch=int(best_epoch),
        inner_oof_total=hist1["best_total"], pred=pred, va_wells=list(va_wells),
        tr_wells=list(tr_wells), inner_tr_wells=list(inner_tr),
        inner_val_wells=list(inner_val), fit_wells=list(scaler.fit_wells), scaler=scaler,
        target=dict(target), phys_params=phys, seconds=time.time() - t0, hist1=hist1,
        hist2=hist2, resumable=resumable, coverage=cov,
        model_summary={"arch": opt.arch, "n_params": int(sum(
            p.numel() for p in model2.parameters())), "chunk": opt.chunk,
            "overlap": opt.overlap}, spec_dict=opt.spec.as_dict() if opt.spec else {"key": "F1"})


def _train_loop(model, ds, cfg: L.TrainConfig, eval_fn, opt: SeqOptions, dev,
                scaler_params: dict, logger=None, on_epoch=None, optimizer=None,
                keep_best: bool = True, resume_epoch: int = -1,
                resume_scheduler_state: dict | None = None,
                save_hook=None, capacity_hook=None) -> dict[str, Any]:
    """序列版训练循环（与 `loop.run_training` 同语义：真实评分早停 + best 权重写回）。"""
    require("torch")
    import torch

    L.install_pause_handlers()
    opt_ = opt
    o = optimizer or torch.optim.AdamW(model.parameters(), lr=float(cfg.lr),
                                       weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(int(cfg.epochs), 1),
                                                       eta_min=float(cfg.lr_min))
    if resume_scheduler_state:
        try:
            sched.load_state_dict(resume_scheduler_state)
        except Exception:
            pass
    t0 = time.time()
    hist: list[dict[str, Any]] = []
    best = {"epoch": -1, "total": float("-inf")}
    best_state = None
    n_bad = 0
    stopped = "completed"
    for epoch in range(int(resume_epoch) + 1, int(cfg.epochs)):
        if L.pause_requested():
            stopped = "paused"
            break
        if cfg.time_budget_h is not None and (time.time() - t0) > cfg.time_budget_h * 3600:
            stopped = "time_budget"
            break
        parts = _train_epoch_with_opt(model, o, ds, cfg, epoch, scaler_params, dev, opt_)
        sched.step()
        rec: dict[str, Any] = {"epoch": epoch, "lr": float(o.param_groups[0]["lr"]),
                               "seconds": round(time.time() - t0, 2), **parts}
        if eval_fn is not None:
            ev = eval_fn(model)
            rec["val"] = ev
            rec["val_total"] = float(ev.get("total", float("nan")))
            if rec["val_total"] > best["total"]:
                best = {"epoch": epoch, "total": rec["val_total"]}
                if keep_best:
                    best_state = {k: v.detach().to("cpu").clone()
                                  for k, v in model.state_dict().items()}
                n_bad = 0
            else:
                n_bad += 1
        hist.append(rec)
        if on_epoch is not None:
            on_epoch(epoch, rec)
        if save_hook is not None:
            save_hook(epoch, rec, model, o, sched)
        if logger is not None:
            vals = {"loss/total": rec.get("total", float("nan")), "lr": rec["lr"]}
            for k in ("align", "aux", "joint", "atom"):
                if rec.get(k) is not None:
                    vals[f"loss/{k}"] = rec[k]
            if eval_fn is not None:
                vals["score/inner_oof_total"] = rec.get("val_total", 0.0)
            logger.scalars(vals, step=epoch)
        L.check_disk(cfg, capacity_hook=capacity_hook)
        if eval_fn is not None and n_bad >= int(cfg.patience):
            stopped = "early_stop"
            break
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs": hist, "best_epoch": int(best["epoch"]),
            "best_total": (None if best["total"] == float("-inf") else float(best["total"])),
            "best_restored": bool(keep_best and best_state is not None),
            "stopped_reason": stopped, "seconds": round(time.time() - t0, 2),
            "n_epochs_run": len(hist)}


def _train_epoch_with_opt(model, optimizer, ds, cfg: L.TrainConfig, epoch: int,
                          scaler_params: dict, dev, opt: SeqOptions) -> dict[str, float]:
    """与 `train_seq_epoch` 相同，但显式接收 optimizer（避免闭包/全局状态）。"""
    require("torch")
    import torch
    from ..losses.score_aligned import total_loss

    model.train()
    # 每个 epoch 重新生成 SeqChunkDataset.order；否则训练循环永远用 epoch=0 的洗牌顺序。
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)
    lam1 = L.lam1_at(epoch, cfg.epochs, cfg)
    agg: dict[str, float] = {}
    nb = 0
    bs = max(int(opt.batch_chunks), 1)
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        if not items:
            continue
        xb = torch.stack([it["X"] for it in items]).to(dev)
        batch = {k: torch.stack([it[k] for it in items]).to(dev)
                 for k in ("por", "perm_z", "sw", "mask", "y_atom", "y_joint")
                 if k in items[0]}
        optimizer.zero_grad(set_to_none=True)
        with L.amp_context(cfg, dev):
            out = model(xb)
            loss, parts = total_loss(out, batch, lam1=lam1, lam_atom=cfg.lam_atom,
                                     lam_joint=cfg.lam_joint,
                                     s_por=float(scaler_params.get("s_por", 11.34)),
                                     s_sw=float(scaler_params.get("s_sw", 20.0)))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"seq: non-finite loss at epoch {epoch}")
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()
        parts["total"] = float(loss.detach())
        parts["lam1"] = lam1
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()} if nb else {"total": float("nan")}


def assemble_oof_seq(results: Sequence["SeqFoldResult"], cache) -> dict[str, Any]:
    """逐折序列结果 → 整表 OOF（well_index 全局偏移；逐行不改动、不重采样）。"""
    from ..data import dataset as D
    from ..features import basic as F
    keys = ("y_true", "y_pred", "cont", "q_atom", "q_joint", "mask", "y_atom",
            "depth", "fold_of_row", "tau_per_row", "well_index")
    acc: dict[str, list] = {k: [] for k in keys}
    well_ids: list[str] = []
    offset = 0
    for r in results:
        yt, yp, mask, ya, dep = [], [], [], [], []
        for w in r.va_wells:
            sh = D.read_well_shard(cache, w, "train")
            lab = F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
            p = r.pred[w]
            cont = M.decode_continuous(p)
            tau = np.asarray(p.get("tau", r.tau["tau"]), dtype="float64")
            gated = M.atom_gate(cont, p["q_atom"], tau)
            yt.append(M.label_scale_stack(lab["por"], lab["perm_z"], lab["sw"]))
            yp.append(gated)
            mask.append(np.asarray(lab["mask"], dtype="float32"))
            ya.append(np.asarray(lab["y_atom"], dtype="float32"))
            dep.append(np.asarray(sh["depth"], dtype="float64"))
            acc["cont"].append(cont)
            acc["q_atom"].append(np.asarray(p["q_atom"]))
            acc["q_joint"].append(np.asarray(p["q_joint"]))
            acc["tau_per_row"].append(np.tile(tau[None, :], (len(lab["por"]), 1)))
        n = sum(len(x) for x in yt)
        well_ids += list(r.va_wells)
        acc["y_true"].append(np.concatenate(yt))
        acc["y_pred"].append(np.concatenate(yp))
        acc["mask"].append(np.concatenate(mask))
        acc["y_atom"].append(np.concatenate(ya))
        acc["depth"].append(np.concatenate(dep))
        acc["fold_of_row"].append(np.full(n, r.fold, dtype="int32"))
        wlens = [len(x) for x in yt]
        idx = np.concatenate([np.full(L, i, dtype="int64") for i, L in enumerate(wlens)])
        acc["well_index"].append(idx + offset)
        offset += len(r.va_wells)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["well_ids"] = np.array(well_ids, dtype=object)
    out["n_wells"] = len(well_ids)
    out["spec_key"] = results[0].spec_dict.get("key", "F1") if results else "F1"
    return out


def boundary_report(preds_by_well: dict, cache, edge_m: float = 10.0,
                    step_m: float = 0.1) -> dict[str, Any]:
    """边界体检（E3/P2 §5 步 5）：井首/井尾 `edge_m` 与井中段的逐目标 Acc 差。

    `edge_m` 按**实际深度**切（不是行数），因此井采样间隔不同也公平。
    """
    from ..data import dataset as D
    from ..features import basic as F
    from ..score import score_arrays

    rows = {"head": [], "tail": [], "middle": []}
    per_well: dict[str, Any] = {}
    for w, p in preds_by_well.items():
        sh = D.read_well_shard(cache, w, "train")
        lab = F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
        yt = M.label_scale_stack(lab["por"], lab["perm_z"], lab["sw"])
        cont = M.decode_continuous(p)
        tau = np.asarray(p.get("tau", [0.5, 0.5, 0.5]), dtype="float64")
        yp = M.atom_gate(cont, p["q_atom"], tau)
        m = np.asarray(lab["mask"], dtype=bool)
        d = np.asarray(sh["depth"], dtype="float64")
        if d.size == 0:
            continue
        lo = d.min() + float(edge_m)
        hi = d.max() - float(edge_m)
        sel = {"head": d <= lo, "tail": d >= hi, "middle": (d > lo) & (d < hi)}
        entry: dict[str, Any] = {"n_rows": int(d.size), "edge_m": float(edge_m)}
        for name, s in sel.items():
            if int(s.sum()) == 0:
                entry[name] = None
                continue
            sc = score_arrays(yt[s], yp[s], missing=~m[s],
                              missing_mode=C.SCORE_MISSING_MODE)
            rows[name].append((int(s.sum()), sc))
            entry[name] = {k: float(v) for k, v in sc.items() if k != "missing_mode"}
        per_well[w] = entry

    def _agg(name: str) -> dict[str, Any] | None:
        if not rows[name]:
            return None
        tot = sum(n for n, _ in rows[name])
        out: dict[str, Any] = {"n_rows": int(tot)}
        for key in ("total", "acc_por", "acc_perm", "acc_sw"):
            out[key] = float(sum(n * sc[key] for n, sc in rows[name]) / max(tot, 1))
        return out

    agg = {k: _agg(k) for k in ("head", "tail", "middle")}
    worst = 0.0
    if agg["middle"] and agg["head"] and agg["tail"]:
        for key in ("acc_por", "acc_perm", "acc_sw"):
            worst = max(worst, abs(agg["middle"][key] - agg["head"][key]),
                        abs(agg["middle"][key] - agg["tail"][key]))
    return {"edge_m": float(edge_m), "aggregate": agg, "per_well": per_well,
            "max_edge_gap": float(worst), "threshold": 0.02,
            "within_threshold": bool(worst < 0.02),
            "note": "井首/尾 10 m 与中段的逐目标 Acc 差；≥0.02 必须给出修正计划（E3/P2 §6）"}


def fold_metrics_seq(res: "SeqFoldResult", cache, opt: SeqOptions) -> dict[str, Any]:
    """单折指标（序列版）：与行级 `fold_runner.fold_metrics` 同字段，便于并列比较。"""
    sc = score_wells(res.pred, res.va_wells, cache, opt)
    const = {w: {"por": np.full_like(np.asarray(res.pred[w]["por"], dtype="float64"),
                                     C.ATOM_VALUES["POR"]),
                 "perm_z": np.zeros_like(np.asarray(res.pred[w]["por"], dtype="float64")),
                 "sw": np.full_like(np.asarray(res.pred[w]["por"], dtype="float64"),
                                    C.ATOM_VALUES["SW"]),
                 "q_atom": np.zeros((np.asarray(res.pred[w]["por"]).size, 3)),
                 "q_joint": np.zeros(np.asarray(res.pred[w]["por"]).size),
                 "tau": np.asarray([2.0, 2.0, 2.0])}      # τ>1 -> 不切换，保持常数
             for w in res.va_wells}
    sconst = score_wells(const, res.va_wells, cache, opt)
    return {"fold": res.fold, "n_rows": int(sc["n_rows"]), "total": float(sc["total"]),
            "por": float(sc["acc_por"]), "perm": float(sc["acc_perm"]),
            "sw": float(sc["acc_sw"]), "const_total": float(sconst["total"]),
            "delta_vs_const": float(sc["total"] - sconst["total"]),
            "tau": [float(x) for x in res.tau["tau"]], "best_epoch": int(res.best_epoch),
            "inner_oof_total": res.inner_oof_total, "seconds": round(float(res.seconds), 2),
            "n_features": int(res.scaler.median.shape[0]), "arch": res.model_summary.get("arch"),
            "n_params": res.model_summary.get("n_params")}
