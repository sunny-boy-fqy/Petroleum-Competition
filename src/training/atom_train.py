"""WP2：独立原子分类头的训练器（torch 门控）。

与 E6 的两阶段训练不同，这里专注 **q_atom 分类质量**：
  * Focal BCE + 逐目标正类权重 + 非联合原子行加权；
  * 可选边界难负例加权（需要连续值预测 ``cont``）；
  * 早停指标 = 逐目标 AUC 均值（单类时退化为 Accuracy）；
  * 训练结束后可在 inner-val 上拟合温度/等渗校准与期望分数动作表。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .. import constants as C
from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch

    from ..losses.atom_classifier import (atom_classifier_loss,
                                         boundary_hard_negative_weight)
    from ..models.atom_head import build_atom_head
    from ..training import metrics as M


def _pos_weight(y, mask):
    """逐目标 neg/pos（截断到 [0.5, 50]），返回长度 3 的 list。"""
    y = np.asarray(y, dtype="float64")
    m = np.asarray(mask, dtype=bool)
    out = []
    for j in range(3):
        pos = float(((y[:, j] > 0.5) & m[:, j]).sum())
        neg = float(((y[:, j] <= 0.5) & m[:, j]).sum())
        w = neg / max(pos, 1.0)
        out.append(float(np.clip(w, 0.5, 50.0)))
    return out


def _atom_metric(y, q, mask) -> float:
    """逐目标 AUC 均值；AUC 不可用时用逐目标 Accuracy。"""
    y = np.asarray(y, dtype=bool)
    q = np.asarray(q, dtype="float64")
    m = np.asarray(mask, dtype=bool)
    aucs, accs = [], []
    for j in range(3):
        sel = m[:, j]
        if not sel.any():
            continue
        auc = M.binary_auc(y[sel, j], q[sel, j])
        if auc is not None:
            aucs.append(float(auc))
        pred = q[sel, j] >= 0.5
        accs.append(float((pred == y[sel, j]).mean()))
    if aucs:
        return float(np.mean(aucs))
    return float(np.mean(accs)) if accs else float("nan")


def train_atom_head(X_tr, y_tr, m_tr, y_joint_tr,
                    X_va, y_va, m_va, y_joint_va,
                    cont_tr=None, cont_va=None,
                    hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                    epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-4,
                    batch_size: int = 4096, gamma: float = 1.5,
                    alpha_nonjoint: float = 1.0, max_boundary_weight: float = 3.0,
                    use_boundary: bool = True, patience: int = 5,
                    seed: int = 42, device: str = "auto", verbose: bool = False,
                    separate_heads: bool = False) -> dict[str, Any]:
    """训练原子头；返回 model/history/best_*，可选校准与动作表。"""
    require("torch")
    from . import loop as L

    dev = L.resolve_device(L.TrainConfig(device=device))
    L.set_seed(seed)
    Xtr = np.asarray(X_tr, dtype="float32")
    ytr = np.asarray(y_tr, dtype="float32")
    mtr = np.asarray(m_tr, dtype="float32")
    jtr = np.asarray(y_joint_tr, dtype="float32").reshape(-1)
    Xva = np.asarray(X_va, dtype="float32")
    yva = np.asarray(y_va, dtype="float32")
    mva = np.asarray(m_va, dtype="float32")
    jva = np.asarray(y_joint_va, dtype="float32").reshape(-1)
    if Xtr.ndim != 2:
        raise ValueError(f"X_tr 必须为 (N,d)，got {Xtr.shape}")
    n, d_in = Xtr.shape
    model = build_atom_head(d_in, hidden=hidden, layers=layers, dropout=dropout,
                            separate_heads=separate_heads, seed=seed).to(dev)
    pw = _pos_weight(ytr, mtr)
    pos_w = torch.tensor(pw, dtype=torch.float32, device=dev)

    # 边界难负例权重（可选）
    wtr = wva = None
    if use_boundary and cont_tr is not None:
        wtr = boundary_hard_negative_weight(
            cont_tr, y_tr, [C.ATOM_VALUES[t] for t in C.TARGETS],
            [C.DELTA_POR, None, C.DELTA_SW], max_weight=max_boundary_weight)
        if cont_va is not None:
            wva = boundary_hard_negative_weight(
                cont_va, y_va, [C.ATOM_VALUES[t] for t in C.TARGETS],
                [C.DELTA_POR, None, C.DELTA_SW], max_weight=max_boundary_weight)

    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr).to(dev)
    mt = torch.from_numpy(mtr).to(dev)
    jt = torch.from_numpy(jtr).to(dev)
    wt = None if wtr is None else wtr.to(dev)
    Xv = torch.from_numpy(Xva).to(dev)
    yv = torch.from_numpy(yva).to(dev)
    mv = torch.from_numpy(mva).to(dev)
    jv = torch.from_numpy(jva).to(dev)
    wv = None if wva is None else wva.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=float(weight_decay))
    hist: list[dict] = []
    best = {"metric": float("-inf"), "epoch": -1, "state": None}
    bad = 0
    bs = max(int(batch_size), 2)
    for ep in range(int(epochs)):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            out = model(Xt[idx])
            loss = atom_classifier_loss(
                out["q_atom_logit"], yt[idx], mt[idx], jt[idx],
                gamma=gamma, pos_weight=pos_w, alpha_nonjoint=alpha_nonjoint,
                boundary_weight=(None if wt is None else wt[idx]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach().cpu())
            nb += 1
        model.eval()
        with torch.no_grad():
            out_va = model(Xv)
        q_va = out_va["q_atom"].float().cpu().numpy()
        metric = _atom_metric(yva, q_va, mva)
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "metric": float(metric)}
        hist.append(rec)
        if verbose:
            print(f"[atom] ep{ep:3d} loss={rec['loss']:.5f} auc/acc={metric:.5f}",
                  flush=True)
        if metric > best["metric"]:
            best = {"metric": float(metric), "epoch": ep,
                    "state": {k: v.detach().to("cpu").clone()
                              for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        q_va = model(Xv)["q_atom"].float().cpu().numpy()
    result: dict[str, Any] = {"model": model, "history": hist,
                              "best_epoch": int(best["epoch"]),
                              "best_metric": (None if best["metric"] == float("-inf")
                                              else float(best["metric"])),
                              "q_atom_va": q_va, "pos_weight": pw,
                              "perm_over_weight": 1.0}
    # 校准 + 期望分数动作表（需要连续值 cont_va）
    if cont_va is not None:
        from ..inference import atom_decision as AD
        from ..inference import calibration as CAL
        temp = {}
        for j, t in enumerate(C.TARGETS):
            sel = np.asarray(mva, dtype=bool)[:, j] & np.isfinite(q_va[:, j])
            if sel.sum() >= 20 and len(np.unique(np.asarray(yva)[sel, j])) > 1:
                fit = CAL.fit_temperature(CAL.logit(q_va[sel, j]),
                                          np.asarray(yva)[sel, j])
                temp[t] = float(fit["temperature"])
        q_cal = q_va.copy()
        if temp:
            from ..inference import decode as DEC
            q_cal = DEC.calibrate_atom_probs(q_va, {"temperature": temp})
        result["atom_calibration"] = ({"kind": "temperature", "temperature": temp}
                                      if temp else {})
        result["action_table"] = AD.fit_decision_table(
            cont_va, q_cal, yva, mva, n_bins=20, min_bin=200, min_gain=0.0)
        result["q_atom_va_calibrated"] = q_cal
    return result


def predict_atom_logits(model, X, device: str = "auto") -> np.ndarray:
    """批量推理 q_atom（概率，N,3）。"""
    require("torch")
    from . import loop as L
    dev = L.resolve_device(L.TrainConfig(device=device))
    model.eval().to(dev)
    X = np.asarray(X, dtype="float32")
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 8192):
            xb = torch.from_numpy(X[i:i + 8192]).to(dev)
            out.append(torch.sigmoid(model(xb)["q_atom_logit"]).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype="float32")
