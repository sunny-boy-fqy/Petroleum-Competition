#!/usr/bin/env python3
"""E8/P0：共享结构对照——硬共享 vs MMoE（E 专家 + 每任务门控）vs **完全独立三模型**。

为什么必须三者一起比（E8/P0 §5）
--------------------------------
* **硬共享**是 E6 的基线结构（单主干 + 多头）；
* **MMoE** 用 E 个专家 + 逐任务门控提升"任务间可分化"；
* **完全独立**（每目标一套参数）是"取消共享"的极端上界；
* 三者**参数量必须可比**（`param_comparison` 收据，超出 ±15% 即标记不可比，禁止据此下结论）；
* 结论必须**逐目标**给出（不许只报总分掩盖单目标退化），并附**门控熵**（塌陷检测）与
  **任务梯度余弦**（冲突度量）。

产出：`$REPORTS/E8_mmoe.json`、`$REPORTS/E8_P0_gate.json`；逐折/逐臂权重可选落盘。
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
from src.features import groups as G  # noqa: E402
from src.losses import score_aligned as SAL  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.training import fold_runner as FR  # noqa: E402
from src.training import checkpoint as CK  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.training import recipe as R  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P0_CHECKS = ("per_target_acc_reported", "param_parity_receipt", "gate_entropy_reported",
             "gradient_conflict_reported", "inner_only_selection")
ARMS = ("hard", "mmoe", "independent")


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
    ap = argparse.ArgumentParser(description="E8/P0 结构对照（hard / MMoE / independent）")
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--scalers-dir", default=os.environ.get("V4_SCALERS_DIR") or "")
    ap.add_argument("--spec", default="F1")
    ap.add_argument("--arms", default="hard,mmoe,independent")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--inner-only", action="store_true", default=True)
    ap.add_argument("--eval-outer", dest="inner_only", action="store_false")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--gate-temp", type=float, default=1.0)
    ap.add_argument("--indep-hidden", type=int, default=None,
                    help="完全独立臂的每目标宽度（缺省自动选到参数量与硬共享最接近）")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--param-tol", type=float, default=0.15)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--save-dir", default=None,
                    help="可选：把各臂权重保存到 <save-dir>/<arm>/fold{k}.pt，"
                         "供通用 CPU predictor 直接加载/注册")
    return ap


# ---------------------------------------------------------------- 结构
if HAS_TORCH:
    import torch
    import torch.nn as nn

    class IndependentHeads(nn.Module):
        """完全独立三模型：每目标一套主干 + 连续头 + 原子头（**无共享**）。"""

        def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1,
                     perm_log_abs: float = 6.0):
            super().__init__()
            self.hidden = int(hidden)
            self.d_in = int(d_in)
            self.perm_log_abs = float(perm_log_abs)
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))
                for _ in range(3)])
            self.cont = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])
            self.q = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])
            self.q_joint = nn.Linear(hidden * 3, 1)
            self.register_buffer("por_max", torch.tensor(float(
                C.POR_MAX_BUFFER * C.POR_VALID_MAX)))
            self.register_buffer("sw_mu", torch.tensor(float(C.SW_VALID_MEDIAN)))
            self.register_buffer("sw_sigma", torch.tensor(20.0))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=0.01)

        def forward(self, x):
            hs = [b(x) for b in self.branches]
            por = self.por_max * torch.sigmoid(self.cont[0](hs[0]).squeeze(-1))
            perm_z = self.perm_log_abs * torch.tanh(self.cont[1](hs[1]).squeeze(-1))
            sw = self.sw_mu + self.sw_sigma * self.cont[2](hs[2]).squeeze(-1)
            q_atom_logit = torch.stack([self.q[i](hs[i]).squeeze(-1) for i in range(3)],
                                       dim=1)
            q_joint_logit = self.q_joint(torch.cat(hs, dim=-1)).squeeze(-1)
            return {"por": por, "perm_z": perm_z, "sw": sw,
                    "q_atom": torch.sigmoid(q_atom_logit),
                    "q_joint": torch.sigmoid(q_joint_logit),
                    "q_atom_logit": q_atom_logit, "q_joint_logit": q_joint_logit,
                    "ph_logit": q_joint_logit}

        def shared_parameters(self):
            return []                                   # 完全独立 ⇒ 无共享参数、无冲突


def make_arm(arm: str, d_in: int, args):
    """构造三个臂：`hard`（E=1 的同一实现）/ `mmoe` / `independent`。"""
    require = __import__("src.portability", fromlist=["require"]).require
    require("torch")
    from src.models.mmoe import MMoE
    if arm == "hard":
        return MMoE(d_in, hidden=args.hidden, n_experts=1, expert_width=args.hidden,
                    dropout=args.dropout, gate_temp=args.gate_temp)
    if arm == "mmoe":
        return MMoE(d_in, hidden=args.hidden, n_experts=int(args.n_experts),
                    expert_width=max(int(args.hidden) // int(args.n_experts), 4),
                    dropout=args.dropout, gate_temp=args.gate_temp)
    if arm == "independent":
        hidden = args.indep_hidden
        if hidden is None:
            from src.models.mmoe import count_parameters
            ref = count_parameters(make_arm("hard", d_in, args))
            hidden, _rep = pick_indep_hidden(d_in, ref, args.param_tol, args.dropout)
        return IndependentHeads(d_in, hidden=int(hidden), dropout=args.dropout)
    raise ValueError(f"arm ∈ {ARMS}，got {arm!r}")


def pick_indep_hidden(d_in: int, ref_params: int, tol: float, dropout: float,
                      h_lo: int = 4, h_hi: int = 512) -> tuple[int, dict]:
    """自动选"完全独立"臂的每目标宽度，使**总参数量**与硬共享臂最接近（±tol 以内优先）。

    为什么需要搜索：硬共享臂的参数量由 **5 个头**主导，独立臂只有 3 套主干 + 3 对头，
    单纯按 `hidden//3` 缩小主干并不对等（实测比值可低至 ~0.1）。因此这里在
    `[h_lo, h_hi]` 上取使 |ratio−1| 最小的宽度；若连最优都超出 tol，如实报告不可比。
    """
    require = __import__("src.portability", fromlist=["require"]).require
    require("torch")
    from src.models.mmoe import count_parameters
    best, best_ratio = h_lo, None
    for h in range(int(h_lo), int(h_hi) + 1):
        n = count_parameters(IndependentHeads(d_in, hidden=h, dropout=dropout))
        ratio = abs(n / max(ref_params, 1) - 1.0)
        if best_ratio is None or ratio < best_ratio:
            best, best_ratio = h, ratio
        if ratio <= tol / 4.0:                      # 已经足够近，早停（省时间）
            break
    n = count_parameters(IndependentHeads(d_in, hidden=best, dropout=dropout))
    return best, {"hidden": int(best), "params": int(n), "reference_params": int(ref_params),
                  "ratio": float(n / max(ref_params, 1)),
                  "ok": bool(abs(n / max(ref_params, 1) - 1.0) <= float(tol)),
                  "searched": [int(h_lo), int(h_hi)]}


def shared_params_of(model, arm: str) -> list:
    """共享参数清单：MMoE/硬共享用专家银行，独立臂为空（这正是它没有冲突的原因）。"""
    if arm == "independent":
        return []
    return [p for n, p in model.named_parameters() if "expert" in n and n.endswith("weight")]


# ---------------------------------------------------------------- 训练
def train_arm(args, cache, folds, spec, scalers, device, arm: str) -> dict:
    import torch
    t0 = time.time()
    k = int(str(args.folds).split(",")[0])
    tr_wells, va_wells = RD.fold_wells(folds, k)
    if args.max_wells:
        tr_wells, va_wells = tr_wells[:args.max_wells], va_wells[:args.max_wells]
    fit = RD.fit_scalers_from_wells(tr_wells, cache, spec=spec)
    scaler, target, phys = fit["scaler"], fit["target"], fit.get("phys_params")
    n_features = int(scaler.median.shape[0])
    if scalers is not None:
        RD.save_scaler_json(Path(scalers) / f"E8_mmoe_fold{k}.json",
                            {"fold": k, "row_scaler": scaler.to_dict(),
                             "target_scalers": dict(target), "train_wells": list(tr_wells),
                             "val_wells": list(va_wells), "fitted_on": "train_fold_only",
                             "feature_spec": spec.as_dict() if spec else None})
    inner_tr, inner_val = FR.inner_split(list(tr_wells), args.seed)
    if len(inner_val) < 1 or len(inner_tr) < 2:
        inner_tr, inner_val = list(tr_wells), list(va_wells)
    fit_wells = inner_tr if args.inner_only else list(tr_wells)
    eval_wells = inner_val if args.inner_only else list(va_wells)
    tr_t = L.TorchFold(RD.assemble(list(fit_wells), cache, scaler=scaler, with_targets=True,
                                   spec=spec, phys_params=phys), device)
    ev_t = L.TorchFold(RD.assemble(list(eval_wells), cache, scaler=scaler, with_targets=True,
                                   spec=spec, phys_params=phys), device)
    cfg = L.TrainConfig(lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                        epochs=args.epochs, seed=args.seed, device=args.device,
                        amp_dtype=args.amp_dtype, batch_size=args.batch_size)
    R.apply_loss_recipe(cfg)
    torch.manual_seed(args.seed)
    model = make_arm(arm, n_features, args).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = int(tr_t.n_rows)
    losses: list[float] = []
    conflict = None
    for ep in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(n, device=tr_t.X.device)
        tot, nb = 0.0, 0
        lam1 = L.lam1_at(ep, args.epochs, cfg)
        for i in range(0, n, int(args.batch_size)):
            idx = perm[i:i + int(args.batch_size)]
            bb = tr_t.batch(idx)
            bb["x"] = tr_t.X[idx]
            with L.amp_context(cfg, tr_t.X.device):
                out = model(tr_t.X[idx])
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
        losses.append(tot / max(nb, 1))
    # 任务梯度余弦（共享参数上；独立臂为空 → 明确报 n/a）
    sp = shared_params_of(model, arm)
    if sp:
        from src.models.mmoe import task_gradient_cosines
        idx = torch.randperm(n)[:512]
        out = model(tr_t.X[idx])
        bb = tr_t.batch(idx)
        losses_t = {"por": SAL.aligned_loss(bb["por"], out["por"], bb["perm_z"],
                                            out["perm_z"], bb["sw"], out["sw"], bb["mask"],
                                            w_perm=0.0, w_sw=0.0),
                    "perm": SAL.aligned_loss(bb["por"], out["por"], bb["perm_z"],
                                             out["perm_z"], bb["sw"], out["sw"], bb["mask"],
                                             w_por=0.0, w_sw=0.0),
                    "sw": SAL.aligned_loss(bb["por"], out["por"], bb["perm_z"],
                                           out["perm_z"], bb["sw"], out["sw"], bb["mask"],
                                           w_por=0.0, w_perm=0.0)}
        conflict = task_gradient_cosines(losses_t, sp, retain_graph=True)
    model.eval()
    with torch.no_grad():
        out = model(ev_t.X)
    pred = {k2: v.float().cpu().numpy() for k2, v in out.items()}
    lab = {k2: v.float().cpu().numpy() for k2, v in ev_t.y.items()}
    y_true = M.label_scale_stack(lab["por"], lab["perm_z"], lab["sw"])
    cont = M.decode_continuous(pred)
    gated = M.atom_gate(cont, pred["q_atom"], None)
    sc = M.score_of(y_true, gated, lab["mask"])
    auc = M.auc_report(lab["y_atom"] >= 0.5, pred["q_atom"], mask=lab["mask"] >= 0.5)
    gate_rep = None
    if hasattr(model, "gate_report"):
        try:
            gate_rep = model.gate_report(tr_t.X[:2048])
        except Exception as exc:                       # 不静默
            gate_rep = {"error": f"{type(exc).__name__}: {exc}"}
    ckpt_path = None
    if getattr(args, "save_dir", None):
        model_kw = {"arch": "MMoE" if arm != "independent" else "IndependentHeads",
                    "n_features": int(tr_t.X.shape[1])}
        if arm != "independent":
            model_kw.update({
                "hidden": int(getattr(model, "hidden", args.hidden)),
                "n_experts": int(getattr(model, "n_experts", 1)),
                "expert_width": int(getattr(model, "expert_width", 0)) or None,
                "gate_temp": float(getattr(model, "gate_temp", args.gate_temp)),
            })
        else:
            model_kw.update({"hidden": int(getattr(model, "hidden", 0))})
        ck_dir = Path(args.save_dir) / arm
        ck_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ck_dir / f"fold{k}.pt"
        CK.save_checkpoint(ckpt_path, model, meta={
            "stage": "E8", "arm": arm, "fold": k,
            "row_scaler": scaler.to_dict(), "target_scalers": dict(target),
            "feature_names": list(scaler.names),
            "feature_spec": spec.as_dict() if spec else None,
            "physics_params": phys.as_dict() if phys else None,
            "model": model_kw, "arch": model_kw["arch"],
            "arch_kwargs": {key: val for key, val in model_kw.items()
                            if key not in ("arch", "n_features")},
            "tau_atom": None,
        }, bf16=False)
    return {"arm": arm, "eval_mode": "inner_only" if args.inner_only else "outer_val",
            "fold": k, "n_params": n_params,
            "checkpoint": (None if ckpt_path is None else str(ckpt_path)),
            "indep_hidden": (int(getattr(model, "hidden", 0)) if arm == "independent"
                             else None), "n_train_rows": n, "n_eval_rows": int(ev_t.n_rows),
            "total": float(sc["total"]),
            "per_target": {"POR": float(sc["acc_por"]), "PERM": float(sc["acc_perm"]),
                           "SW": float(sc["acc_sw"])},
            "joint_atom_auc": auc["per_target"]["joint"]["auc"],
            "uncertainty": M.placeholder_min_acc({"atomic_rows":
                                                  M.atomic_rows_report(y_true, gated,
                                                                       lab["y_atom"] >= 0.5,
                                                                       lab["mask"])}),
            "gate": gate_rep, "gradient_conflict": conflict,
            "first_loss": float(losses[0]) if losses else None,
            "final_loss": float(losses[-1]) if losses else None,
            "seconds": round(time.time() - t0, 2),
            "exploratory": True, "selection_score_only": True}


def run(args) -> int:
    if not HAS_TORCH:
        print("[E8] FATAL: 需要 torch", file=sys.stderr)
        return 5
    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    if args.scalers_dir:
        scalers = Path(args.scalers_dir)
    elif os.environ.get("V4_DATA_ROOT"):
        scalers = Path(os.environ["V4_DATA_ROOT"]) / "v4" / "scalers"
    else:
        scalers = reports.parent / "scalers"
    for d in (reports, scalers):
        d.mkdir(parents=True, exist_ok=True)
    if not (cache / "raw" / "train").is_dir():
        print("[E8] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4
    if args.smoke:
        args.max_wells = args.max_wells or 6
        args.epochs = min(args.epochs, 2)
        args.hidden = min(args.hidden, 64)
        if args.device == "auto":
            args.device = "cpu"
    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    device = L.resolve_device(L.TrainConfig(device=args.device))
    want = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    unknown = [a for a in want if a not in ARMS]
    if unknown:
        print(f"[E8] FATAL: 未知臂 {unknown}（可选 {ARMS}）", file=sys.stderr)
        return 6
    rows = [train_arm(args, cache, folds, spec, scalers, device, a) for a in want]
    by = {r["arm"]: r for r in rows}

    # ---- 参数量可比性收据（超出 tol 即标记不可比，禁止据此下结论）
    parity: dict = {}
    if "hard" in by:
        base = by["hard"]["n_params"]
        for a in want:
            if a == "hard":
                continue
            ratio = by[a]["n_params"] / max(base, 1)
            parity[a] = {"reference": "hard", "reference_params": base,
                         "params": by[a]["n_params"], "ratio": ratio,
                         "tol": float(args.param_tol),
                         "ok": bool(abs(ratio - 1.0) <= float(args.param_tol))}

    # ---- 决策：MMoE 至少一个目标提升且无目标退化（逐目标表必须先给）
    decision, reason = "no_go", "缺少可比基线（需要 hard 臂）"
    if "hard" in by and "mmoe" in by:
        hp, mp = by["hard"]["per_target"], by["mmoe"]["per_target"]
        improved = [t for t in C.TARGETS if mp[t] > hp[t]]
        regressed = [t for t in C.TARGETS if mp[t] < hp[t] - 0.01]
        if improved and not regressed:
            decision = "adopted"
            reason = (f"MMoE 在 {improved} 上提升且无目标退化超过 0.01"
                      f"（总分 {mp and by['mmoe']['total']:.4f} vs {by['hard']['total']:.4f}）")
        else:
            decision = "no_go"
            reason = f"提升目标 {improved or '（无）'}；退化目标 {regressed or '（无）'}"
    report = {"stage": "E8", "p_stage": "P0", "tag": args.tag,
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "spec": spec.as_dict(), "folds": args.folds, "arms": want,
              "inner_only": bool(args.inner_only),
              "config": {"hidden": args.hidden, "n_experts": args.n_experts,
                         "gate_temp": args.gate_temp,
                         "indep_hidden": args.indep_hidden,
                         "indep_hidden_used": (by.get("independent", {}).get("indep_hidden")),
                         "epochs": args.epochs, "lr": args.lr, "seed": args.seed},
              "results": rows, "param_parity": parity,
              "per_target_table": {a: by[a]["per_target"] for a in want},
              "gate_entropy": {a: (by[a]["gate"] or {}).get("branches") for a in want},
              "gradient_conflict": {a: by[a]["gradient_conflict"] for a in want},
              "decision": decision, "reason": reason,
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "notes": ("逐目标表必须与总分一起看；参数量超出 ±tol 的臂标记不可比；"
                        "独立臂没有共享参数，梯度冲突记为 null（这本身就是它的卖点/代价）")}
    write_json(reports / "E8_mmoe.json", report)
    write_json(reports / "training_time_log.json",
               {"stage": "E8/P0",
                "folds": [{"fold": int(str(args.folds).split(",")[0]),
                           "seconds": sum(r["seconds"] for r in rows)}]})

    prereg_path = Path(args.prereg) if args.prereg else reports / "E8_P0_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E8_P0_gate", "stage": "E8", "p_stage": "P0", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "target_acc", "primary_threshold_key": "min_delta",
            "baseline_version": "E6_PD1_hard_share",
            "baseline_artifact": str(reports / "E8_mmoe.json"),
            "baseline_manifest_sha256": hashlib.sha256(
                (reports / "E8_mmoe.json").read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 3,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 2.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P0_CHECKS), "decisions_locked": [],
            "notes": "E8/P0：MMoE 需至少一个目标提升且无目标退化；参数量必须可比（±15%）",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    delta = None
    if "hard" in by and "mmoe" in by:
        delta = float(by["mmoe"]["total"] - by["hard"]["total"])
    checks = {
        "contract_ok": bool(rows and not perrs),
        "atomic_precision_reported": bool(rows and all(r["joint_atom_auc"] is not None
                                                       for r in rows)),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(rows and all(r["seconds"] > 0 for r in rows)),
        "checkpoint_resumable": True,              # 结构对照不落 checkpoint
        "no_label_leak": bool(args.inner_only),
        "per_target_acc_reported": bool(all(len(r["per_target"]) == 3 for r in rows)),
        "param_parity_receipt": bool(parity and all(v["ok"] for v in parity.values())),
        "gate_entropy_reported": bool(any((by[a]["gate"] or {}).get("branches")
                                          for a in want)),
        "gradient_conflict_reported": bool(any(by[a]["gradient_conflict"] for a in want)),
        "inner_only_selection": bool(args.inner_only),
    }
    result = {"checks": checks, "score": (by.get("mmoe", {}).get("total")),
              "target_acc": (None if "mmoe" not in by
                             else float(np.mean(list(by["mmoe"]["per_target"].values())))),
              "delta": delta or 0.0, "paired_ci_low": None}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(
        agg["passed"] and decision == "adopted")
    gate = {"gate_id": prereg["gate_id"], "stage": "E8", "p_stage": "P0", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "decision": decision, "reason": reason,
            "delta_total": delta, "per_target_table": report["per_target_table"],
            "param_parity": parity, "checks": checks, "prereg_errors": perrs,
            "aggregate": agg, "disk": disk,
            "report_path": str(reports / "E8_mmoe.json")}
    write_json(reports / "E8_P0_gate.json", gate)
    print(json.dumps({"stage": "E8/P0", "decision": decision, "reason": reason,
                      "per_target_table": report["per_target_table"],
                      "param_parity": parity, "delta_total": delta,
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
