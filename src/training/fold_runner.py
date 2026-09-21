"""两阶段单折训练协议（E1 起用，E2/E3 复用）：**outer 折不参与任何选择**。

协议（E1/P0 §12.5 + E1/P1 §5，E2/E3 逐字继承）
----------------------------------------------
1. **选择阶段**：把 outer-train 井切成 `C.N_INNER_FOLDS` 个 inner 折（`inner0` 验证、
   其余训练）；每个 epoch 用**真实 `score.py` 分数**（不是 loss）早停；训练结束时把
   **最优 epoch 权重**写回（`loop.run_training(keep_best=True)`）；再在该模型的
   `inner0` 预测上选逐目标原子阈值 `tau`（平台中点规则、官方目标函数）。
2. **终训阶段**：用**全部** outer-train 井按同一配方训练 `best_epoch` 个 epoch；
   对 outer-val 折**只推理一次**，用阶段 1 的 `tau` 解码。

为什么独立成模块：E2 的组级消融、E3 的序列主干、E6 的对齐微调都要跑同一套协议；
此前只有 E1 的脚本里有实现，任何"消融时顺手改一下顺序"都会造成口径漂移
（正是 `s_por/s_sw` 曾经"计划里有、代码里丢"的那类问题）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .. import constants as C
from ..data import row_dataset as RD
from ..features import basic as F
from ..inference import atomic_gate as AG
from ..portability import HAS_TORCH, require
from ..validation import folds as FOLDS
from . import checkpoint as CK
from . import loop as L
from . import metrics as M


@dataclass
class FoldOptions:
    """单折协议的可调项（其余一律走 `loop.TrainConfig`）。"""
    spec: Any = None                  # features.groups.FeatureSpec
    max_wells: int | None = None
    smoke: bool = False
    resume: bool = False
    save_checkpoints: bool = True
    select_tau: bool = True
    scaler_prefix: str = "E1"
    run_dir: Path | None = None
    scalers_dir: Path | None = None
    tb_run_name: str | None = None
    on_select_epoch: Callable[[int, dict], None] | None = None
    on_final_epoch: Callable[[int, dict], None] | None = None
    # E2/P2：增强**只在训练侧**（outer-train 矩阵）生效，验证折永不增强。
    augment: Any = None
    augment_seed: int = 0


@dataclass
class FoldResult:
    fold: int
    tau: dict
    best_epoch: int
    inner_oof_total: float | None
    pred: dict
    va: RD.FoldTensors
    tr_wells: list[str]
    va_wells: list[str]
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
    spec_dict: dict = field(default_factory=dict)
    augment: dict | None = None

    @property
    def n_train_epochs(self) -> int:
        return len(self.hist1.get("epochs", [])) + len(self.hist2.get("epochs", []))


def labels_of(va) -> "np.ndarray":
    """`FoldTensors`/`TorchFold` → (N,3) 标签尺度真值。"""
    def _c(key: str, attr: str):
        if hasattr(va, "y"):
            x = va.y[key]
            return x.float().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x, "float32")
        v = getattr(va, attr)
        return v.float().cpu().numpy() if hasattr(v, "cpu") else np.asarray(v, "float32")
    return M.label_scale_stack(_c("por", "y_por"), _c("perm_z", "y_perm_z"), _c("sw", "y_sw"))


def mask_of(va) -> "np.ndarray":
    if hasattr(va, "y"):
        x = va.y["mask"]
        return x.float().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x, "float32")
    v = va.mask
    return v.float().cpu().numpy() if hasattr(v, "cpu") else np.asarray(v, "float32")


def y_atom_of(va) -> "np.ndarray":
    if hasattr(va, "y"):
        x = va.y["y_atom"]
        return x.float().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x, "float32")
    v = va.y_atom
    return v.float().cpu().numpy() if hasattr(v, "cpu") else np.asarray(v, "float32")


def inner_split(tr_wells: Sequence[str], seed: int,
                n_inner: int = C.N_INNER_FOLDS) -> tuple[list[str], list[str]]:
    """outer-train 井 → `(inner_train, inner_val)`；井太少时抛错（调用方决定是否降级）。"""
    inner = FOLDS.make_inner_folds(list(tr_wells), n_inner=n_inner, seed=seed)
    keys = sorted(inner)
    tr_set = set(tr_wells)
    val = [w for w in inner[keys[0]] if w in tr_set]
    train = [w for k in keys[1:] for w in inner[k] if w in tr_set]
    return train, val


def pick_tau(model, va_in_t, y_in: "np.ndarray", m_in: "np.ndarray",
             cfg: L.TrainConfig) -> dict[str, Any]:
    """在 inner-OOF 预测上选逐目标 τ（**官方目标函数** + 平台中点规则）。"""
    pred = L.predict_torch(model, va_in_t, cfg)
    sel = AG.select_tau_per_target(cont=M.decode_continuous(pred), q_atom=pred["q_atom"],
                                   y=y_in, mask=m_in)
    return {"tau": [float(v) for v in sel["tau"]], "objective": float(sel["objective"]),
            "plateau": {k: [float(x) for x in v] for k, v in sel["plateau"].items()},
            "score_fn": sel["score_fn"]}


def run_two_phase_fold(fold: int, folds: dict, cache: str | Path, cfg: L.TrainConfig,
                       opt: FoldOptions) -> FoldResult:
    """跑一个 outer 折的完整两阶段协议（含 checkpoint 与 scaler 落盘）。"""
    require("torch")
    import time
    import torch
    from ..models.row_mlp import build_model

    t0 = time.time()
    L.set_seed(cfg.seed)
    dev = L.resolve_device(cfg)
    spec = opt.spec

    tr_wells, va_wells = RD.fold_wells(folds, fold)
    if opt.max_wells:
        tr_wells, va_wells = tr_wells[:opt.max_wells], va_wells[:opt.max_wells]

    fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
    scaler, target = fit["scaler"], fit["target"]
    phys = fit.get("phys_params")
    if opt.scalers_dir is not None:
        payload = {"fold": fold, "row_scaler": scaler.to_dict(),
                   "target_scalers": dict(target), "train_wells": tr_wells,
                   "val_wells": va_wells, "n_train_rows": fit["n_train_rows"],
                   "note": "全部尺度参数只由训练折拟合（E1/P0 §5 步 4）"}
        if spec is not None:
            payload["feature_spec"] = spec.as_dict()
        if phys is not None:
            payload["physics_params"] = phys.as_dict()
        RD.save_scaler_json(Path(opt.scalers_dir) / f"{opt.scaler_prefix}_fold{fold}.json",
                            payload)

    tr_all = RD.assemble(tr_wells, cache, scaler=scaler, with_targets=True, spec=spec,
                         phys_params=phys)
    va = RD.assemble(va_wells, cache, scaler=scaler, with_targets=True, spec=spec,
                     phys_params=phys)
    RD.assert_alignment(tr_all)
    RD.assert_alignment(va)
    n_features = int(tr_all.X.shape[1])

    # 数据增强（E2/P2）：只动训练侧特征，标签/掩码逐行不变
    augment_info = None
    if opt.augment is not None and getattr(opt.augment, "enabled", False):
        from ..data import augment as AUG
        rng = np.random.default_rng(int(opt.augment_seed) + int(fold))
        before = tr_all.X
        tr_all.X = AUG.augment_matrix(before, opt.augment, rng)
        augment_info = AUG.label_invariance(before, tr_all.X, tr_all.y_por, tr_all.mask)
        augment_info["config"] = opt.augment.as_dict()
        augment_info["scope"] = "outer-train only（验证折未增强）"

    # ---- 阶段 1：inner 折选 best_epoch + tau
    inner_tr_wells, inner_val_wells = inner_split(tr_wells, cfg.seed)
    if len(inner_val_wells) < 1 or len(inner_tr_wells) < 2:
        if not opt.smoke:
            raise SystemExit(f"[fold{fold}] inner split too small "
                             f"({len(inner_tr_wells)}/{len(inner_val_wells)})")
        inner_tr_wells, inner_val_wells = list(tr_wells), list(va_wells)

    tr_in_t = L.TorchFold(RD.subset(tr_all, inner_tr_wells), dev)
    va_in = RD.assemble(inner_val_wells, cache, scaler=scaler, with_targets=True, spec=spec,
                        phys_params=phys)
    va_in_t = L.TorchFold(va_in, dev)
    y_in, m_in = labels_of(va_in), mask_of(va_in)
    a_in = y_atom_of(va_in)

    model = build_model(n_features=n_features, hidden=cfg.hidden, layers=cfg.layers,
                        dropout=cfg.dropout, seed=cfg.seed, init_stats=target)

    logger = None
    try:
        from .tb_logger import RunLogger
        logger = RunLogger(opt.tb_run_name or f"{opt.scaler_prefix}_fold{fold}")
    except Exception:                                  # 观测失败不得中断训练
        logger = None

    def inner_eval(m) -> dict:
        pred = L.predict_torch(m, va_in_t, cfg)
        return M.evaluate_predictions(y_in, m_in, pred, y_atom=a_in, tau=None)["cont"]

    def on_select(epoch: int, rec: dict) -> None:
        val = rec.get("val")
        if logger is not None:
            vals = {"loss/total": rec.get("total", float("nan")), "lr": rec.get("lr", 0.0)}
            for k in ("align", "aux", "joint", "atom"):
                if rec.get(k) is not None:
                    vals[f"loss/{k}"] = rec[k]
            if val:
                vals.update({"score/inner_oof_total": val.get("total", 0.0),
                             "score/acc_por": val.get("acc_por", 0.0),
                             "score/acc_perm": val.get("acc_perm", 0.0),
                             "score/acc_sw": val.get("acc_sw", 0.0)})
            logger.scalars(vals, step=epoch)
        if opt.on_select_epoch is not None:
            opt.on_select_epoch(epoch, rec)

    hist1 = L.run_training(model, tr_in_t, cfg, eval_fn=inner_eval, on_epoch=on_select,
                           scaler_params=target, keep_best=True)
    best_epoch = hist1["best_epoch"] if hist1["best_epoch"] >= 0 \
        else max(len(hist1["epochs"]) - 1, 0)

    tau_info: dict[str, Any] = {"tau": [0.5, 0.5, 0.5], "objective": None, "plateau": {},
                                "score_fn": "skipped"}
    if opt.select_tau and not opt.smoke:
        tau_info = pick_tau(model, va_in_t, y_in, m_in, cfg)
    del tr_in_t, va_in_t, va_in

    # ---- 阶段 2：全部 outer-train 重训 best_epoch，outer-val 只推理一次
    cfg2 = L.TrainConfig(**{**cfg.as_dict(), "epochs": max(best_epoch + 1, 1),
                            "patience": 10 ** 9})
    tr_t = L.TorchFold(tr_all, dev)
    model2 = build_model(n_features=n_features, hidden=cfg.hidden, layers=cfg.layers,
                         dropout=cfg.dropout, seed=cfg.seed, init_stats=target)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=float(cfg.lr),
                             weight_decay=float(cfg.weight_decay))
    fold_dir = Path(opt.run_dir) / f"fold{fold}" if opt.run_dir is not None else None
    resume_epoch = -1
    resume_sched = None
    if fold_dir is not None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        # M6：把旧 checkpoint 与临时缓存登记给 disk_guard，低磁盘时自动清理。
        from ..data.disk_guard import register_cache_dirs, register_cleanup_paths
        register_cleanup_paths([fold_dir / "last_prev.pt"])
        register_cache_dirs([Path(cache) / "tmp"])
        if opt.resume and (fold_dir / "last.pt").is_file():
            rr = CK.load_for_resume(fold_dir / "last.pt", model2, optimizer=opt2)
            resume_epoch = int(rr["epoch"])
            resume_sched = rr.get("scheduler_state")
    save_last_hook = None
    capacity_hook = None
    if fold_dir is not None and opt.save_checkpoints:
        state = {"epoch": int(resume_epoch)}

        def _meta(ep: int) -> dict[str, Any]:
            return {"stage": opt.scaler_prefix, "fold": fold, "phase": "final",
                    "epoch": int(ep), "n_epochs_run": int(ep) + 1,
                    "best_epoch_from_inner": int(best_epoch),
                    "inner_oof_total": hist1["best_total"],
                    "tau_atom": [float(t) for t in tau_info["tau"]],
                    "model": {"n_features": n_features, "hidden": cfg.hidden,
                              "layers": cfg.layers, "dropout": cfg.dropout},
                    "target_scalers": dict(target), "config": cfg.as_dict(), "seed": cfg.seed,
                    "row_scaler": scaler.to_dict(),
                    "feature_spec": spec.as_dict() if spec is not None else None,
                    "physics_params": phys.as_dict() if phys is not None else None}

        def save_last(epoch: int, rec: dict, model, opt_, sched_) -> None:
            state["epoch"] = int(epoch)
            if (fold_dir / "last.pt").is_file():
                CK.rotate(fold_dir)                # 先把上一版 last.pt 轮成 last_prev.pt
            CK.save_checkpoint(fold_dir / "last.pt", model, meta=_meta(epoch),
                               optimizer=opt_, scheduler=sched_)

        def save_on_disk_pressure() -> None:
            ep = int(state["epoch"])
            if (fold_dir / "last.pt").is_file():
                CK.rotate(fold_dir)
            CK.save_checkpoint(fold_dir / "last.pt", model2, meta=_meta(ep), optimizer=opt2)

        save_last_hook = save_last
        capacity_hook = save_on_disk_pressure

    hist2 = L.run_training(model2, tr_t, cfg2, eval_fn=None, optimizer=opt2,
                           resume_epoch=resume_epoch,
                           resume_scheduler_state=resume_sched,
                           scaler_params=target, keep_best=False,
                           on_epoch=opt.on_final_epoch,
                           save_hook=save_last_hook, capacity_hook=capacity_hook)
    del tr_t, tr_all

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
        resumable = CK.verify_resumable(
            fold_dir / "best.pt",
            lambda: build_model(n_features=n_features, hidden=cfg.hidden, layers=cfg.layers,
                                dropout=cfg.dropout))

    pred = L.predict_torch(model2, L.TorchFold(va, dev), cfg2)
    if logger is not None:
        logger.close()
    return FoldResult(fold=int(fold), tau=tau_info, best_epoch=int(best_epoch),
                      augment=augment_info,
                      inner_oof_total=hist1["best_total"], pred=pred, va=va,
                      tr_wells=list(tr_wells), va_wells=list(va_wells),
                      inner_tr_wells=list(inner_tr_wells),
                      inner_val_wells=list(inner_val_wells),
                      fit_wells=list(scaler.fit_wells), scaler=scaler, target=dict(target),
                      phys_params=phys, seconds=time.time() - t0, hist1=hist1, hist2=hist2,
                      resumable=resumable,
                      spec_dict=spec.as_dict() if spec is not None else {"key": "F1"})


def leakage_evidence(results: Sequence[FoldResult]) -> dict[str, Any]:
    """`no_label_leak` 的可复算证据：井维度互斥 + 尺度只由训练井拟合。"""
    bad: list[str] = []
    for r in results:
        if set(r.tr_wells) & set(r.va_wells):
            bad.append(f"fold{r.fold}: train/val wells overlap")
        if not set(r.fit_wells).issubset(set(r.tr_wells)):
            bad.append(f"fold{r.fold}: scaler fitted on non-train wells")
        if set(r.inner_val_wells) & set(r.va_wells):
            bad.append(f"fold{r.fold}: inner-val leaks into outer-val")
    return {"ok": not bad, "violations": bad}


def fold_metrics(res: FoldResult) -> dict[str, Any]:
    """单折的官方口径指标（连续 + 原子门 + 占位行 + 常数基线对照）。"""
    from ..score import score_arrays
    va = res.va
    yt, mt = labels_of(va), mask_of(va)
    const = np.tile([C.ATOM_VALUES[t] for t in C.TARGETS], (yt.shape[0], 1))
    gated = M.atom_gate(M.decode_continuous(res.pred), res.pred["q_atom"], res.tau["tau"])
    s = score_arrays(yt, gated, missing=~mt.astype(bool), missing_mode=C.SCORE_MISSING_MODE)
    sc = score_arrays(yt, const, missing=~mt.astype(bool), missing_mode=C.SCORE_MISSING_MODE)
    return {"fold": res.fold, "n_rows": int(va.n_rows), "total": float(s["total"]),
            "por": float(s["acc_por"]), "perm": float(s["acc_perm"]), "sw": float(s["acc_sw"]),
            "const_total": float(sc["total"]),
            "delta_vs_const": float(s["total"] - sc["total"]),
            "tau": [float(x) for x in res.tau["tau"]], "best_epoch": int(res.best_epoch),
            "inner_oof_total": res.inner_oof_total, "seconds": round(float(res.seconds), 2),
            "n_features": int(res.va.X.shape[1])}


def assemble_oof(results: Sequence[FoldResult], cache: str | Path) -> dict[str, Any]:
    """把逐折 `FoldResult` 拼成整表 OOF（**well_index 做全局偏移**，避免跨折井号撞车）。

    返回 numpy 数组字典，键与 E1 写出的 `oof.npz` 保持一致（well_ids/well_index/depth/
    depth_in/y_true/y_pred/cont/q_atom/q_joint/mask/y_atom/fold_of_row/tau_per_row）。
    """
    from ..data import dataset as D
    well_ids: list[str] = []
    keys = ("y_true", "y_pred", "cont", "q_atom", "q_joint", "mask", "y_atom",
            "depth", "depth_in", "fold_of_row", "tau_per_row", "well_index")
    acc: dict[str, list] = {k: [] for k in keys}
    offset = 0
    for r in results:
        va, pred = r.va, r.pred
        cont = M.decode_continuous(pred)
        tau = np.asarray(r.tau["tau"], dtype="float64")
        gated = M.atom_gate(cont, pred["q_atom"], tau)
        d_in = np.concatenate([D.read_well_shard(cache, w, "train")["depth"]
                               for w in va.well_ids])
        well_ids += list(va.well_ids)
        acc["y_true"].append(labels_of(va))
        acc["y_pred"].append(gated)
        acc["cont"].append(cont)
        acc["q_atom"].append(np.asarray(pred["q_atom"]))
        acc["q_joint"].append(np.asarray(pred["q_joint"]))
        acc["mask"].append(mask_of(va))
        acc["y_atom"].append(y_atom_of(va))
        acc["depth"].append(np.asarray(va.depth, dtype="float64"))
        acc["depth_in"].append(np.asarray(d_in, dtype="float64"))
        acc["fold_of_row"].append(np.full(va.n_rows, r.fold, dtype="int32"))
        acc["tau_per_row"].append(np.tile(tau[None, :], (va.n_rows, 1)))
        acc["well_index"].append(np.asarray(va.well_index, dtype="int64") + offset)
        offset += va.n_wells
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["well_ids"] = np.array(well_ids, dtype=object)
    out["n_wells"] = len(well_ids)
    out["spec_key"] = (results[0].spec_dict.get("key", "F1") if results else "F1")
    return out


def per_well_totals(oof: dict[str, Any], y_pred=None) -> tuple["np.ndarray", "np.ndarray"]:
    """逐井官方 Total（bootstrap 的 cluster 单元）与逐井行数（权重）。"""
    from ..score import score_arrays
    yt = oof["y_true"]
    yp = oof["y_pred"] if y_pred is None else y_pred
    m = oof["mask"] >= 0.5
    n_wells = int(oof["n_wells"])
    tot = np.full(n_wells, np.nan)
    rows = np.zeros(n_wells, dtype="int64")
    for w in range(n_wells):
        sel = oof["well_index"] == w
        rows[w] = int(sel.sum())
        if rows[w]:
            tot[w] = score_arrays(yt[sel], yp[sel], missing=~m[sel],
                                  missing_mode=C.SCORE_MISSING_MODE)["total"]
    return tot, rows
