#!/usr/bin/env python3
"""WP2：独立原子分类器训练入口（行级特征或预计算隐状态）。

支持两种输入模式：

* ``--mode row``（默认）：从 ``$V4_CACHE_ROOT/row`` 读取 F1/F2 行级特征，
  逐折拟合 scaler → 训练 ``AtomClassifierHead``；
* ``--mode arrays``：读取 ``--train-npz`` / ``--val-npz``，每个文件含
  ``X`` (N,d)、``y_atom`` (N,3)、``mask`` (N,3)、``y_joint`` (N,)、
  可选 ``cont`` (N,3)（用于边界权重/期望分数动作表）。适合 E5 冻结骨干隐状态。

产出（每折）：
  ``$RUN/E3/atom/fold{k}.pt`` + ``.manifest.json`` + ``E3_atom_report.json``。
纯 numpy 读数据；只有 ``--mode arrays`` 的训练部分需要 torch。
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

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def write_json(path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WP2 独立原子分类器")
    ap.add_argument("--mode", default="row", choices=("row", "arrays"))
    ap.add_argument("--train-npz", default=None)
    ap.add_argument("--val-npz", default=None)
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--spec", default="F1")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--alpha-nonjoint", type=float, default=1.0)
    ap.add_argument("--boundary-weight", type=float, default=3.0)
    ap.add_argument("--no-boundary", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--tag", default="")
    return ap


# ---------------------------------------------------------------- 数据
def _arrays_from_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    for key in ("X", "y_atom", "mask"):
        if key not in d:
            raise ValueError(f"{path} 缺少键 {key!r}")
    d["y_joint"] = (np.asarray(d.get("y_joint", np.zeros(d["X"].shape[0])))
                    if "y_joint" in d else
                    np.all(np.asarray(d["y_atom"]) > 0.5, axis=1).astype("float64"))
    d["cont"] = np.asarray(d["cont"], dtype="float64") if "cont" in d else None
    return d


def _row_fold_arrays(args, k: int):
    """行级模式：用 fold 协议装配 X/y_atom/mask/y_joint（可选 cont 为空）。"""
    from src.data import row_dataset as RD
    from src.features import groups as G
    from src.validation import folds as FOLDS

    spec = G.spec_from_name(args.spec)
    folds = FOLDS.load_folds()
    tr, va = RD.fold_wells(folds, int(k))
    if args.max_wells:
        tr, va = tr[:args.max_wells], va[:args.max_wells]
    fit = RD.fit_scalers_from_wells(tr, args.cache_root, spec=spec)
    scaler, phys = fit["scaler"], fit.get("phys_params")
    tr_t = RD.assemble(tr, args.cache_root, scaler=scaler, with_targets=True,
                       spec=spec, phys_params=phys)
    va_t = RD.assemble(va, args.cache_root, scaler=scaler, with_targets=True,
                       spec=spec, phys_params=phys)
    return {"X": tr_t.X, "y_atom": tr_t.y_atom, "mask": tr_t.mask,
            "y_joint": tr_t.y_joint, "cont": None,
            "X_val": va_t.X, "y_atom_val": va_t.y_atom,
            "mask_val": va_t.mask, "y_joint_val": va_t.y_joint,
            "cont_val": None, "well_ids": list(va_t.well_ids),
            "row_scaler": scaler.to_dict(), "feature_spec": spec.as_dict()}


# ---------------------------------------------------------------- 指标
def atom_metrics(y_atom, q, mask, y_joint=None, threshold: float = 0.5) -> dict:
    from src.training import metrics as M
    y = np.asarray(y_atom) > 0.5
    m = np.asarray(mask, dtype=bool)
    q = np.asarray(q, dtype="float64")
    out = {"per_target": {}, "threshold": float(threshold)}
    for j, t in enumerate(C.TARGETS):
        sel = m[:, j]
        yj, qj = y[sel, j], q[sel, j]
        pred = qj >= float(threshold)
        tp = float((pred & yj).sum())
        fp = float((pred & ~yj).sum())
        fn = float((~pred & yj).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        out["per_target"][t] = {
            "auc": M.binary_auc(yj, qj),
            "ap": M.average_precision(yj, qj),
            "acc": float((pred == yj).mean()) if sel.any() else float("nan"),
            "precision": prec, "recall": rec, "f1": f1,
            "n_observed": int(sel.sum()),
        }
    # 非联合原子行召回
    if y_joint is not None:
        yj = np.asarray(y_joint, dtype="float64").reshape(-1) > 0.5
        for j, t in enumerate(C.TARGETS):
            sel = m[:, j] & (~yj) & y[:, j]
            qj = q[sel, j]
            out["per_target"][t]["nonjoint_recall"] = (
                float((qj >= float(threshold)).mean()) if sel.any() else None)
            out["per_target"][t]["n_nonjoint_atom"] = int(sel.sum())
    vals = [v["auc"] for v in out["per_target"].values() if v.get("auc") is not None]
    out["min_auc"] = float(min(vals)) if vals else None
    out["mean_auc"] = float(np.mean(vals)) if vals else None
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not HAS_TORCH:
        print("[E3/atom] FATAL: 需要 torch", file=sys.stderr)
        return 5
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        if args.device == "auto":
            args.device = "cpu"
    reports, run_dir = Path(args.reports_dir), Path(args.run_root) / "E3" / "atom"
    save_dir = Path(args.save_dir) if args.save_dir else run_dir
    for d in (reports, run_dir, save_dir):
        d.mkdir(parents=True, exist_ok=True)
    fold_list = [int(x) for x in str(args.folds).split(",") if x.strip()]
    from src.training.atom_train import train_atom_head
    from src.training import checkpoint as CK

    rows, t0 = [], time.time()
    for k in fold_list:
        if args.mode == "arrays":
            if not args.train_npz or not args.val_npz:
                print("[E3/atom] FATAL: arrays 模式需要 --train-npz/--val-npz",
                      file=sys.stderr)
                return 4
            d_tr, d_va = _arrays_from_npz(Path(args.train_npz)), _arrays_from_npz(Path(args.val_npz))
            X, y, m, j = d_tr["X"], d_tr["y_atom"], d_tr["mask"], d_tr["y_joint"]
            Xv, yv, mv, jv = d_va["X"], d_va["y_atom"], d_va["mask"], d_va["y_joint"]
            cont, contv = d_tr.get("cont"), d_va.get("cont")
        else:
            d = _row_fold_arrays(args, k)
            X, y, m, j = d["X"], d["y_atom"], d["mask"], d["y_joint"]
            Xv, yv, mv, jv = d["X_val"], d["y_atom_val"], d["mask_val"], d["y_joint_val"]
            cont, contv = d.get("cont"), d.get("cont_val")
        res = train_atom_head(
            X, y, m, j, Xv, yv, mv, jv, cont_tr=cont, cont_va=contv,
            hidden=args.hidden, layers=args.layers, dropout=args.dropout,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            batch_size=args.batch_size, gamma=args.gamma,
            alpha_nonjoint=args.alpha_nonjoint, max_boundary_weight=args.boundary_weight,
            use_boundary=not args.no_boundary, patience=args.patience,
            seed=args.seed, device=args.device, verbose=not args.smoke)
        met = atom_metrics(yv, res["q_atom_va"], mv, jv)
        ckpt = save_dir / f"atom_fold{k}{('_' + args.tag) if args.tag else ''}.pt"
        CK.save_checkpoint(ckpt, res["model"], meta={
            "stage": "E3/atom", "fold": int(k), "mode": args.mode,
            "n_features": int(np.asarray(X).shape[1]),
            "model": {"arch": "AtomClassifierHead", "hidden": args.hidden,
                      "layers": args.layers, "dropout": args.dropout},
            "seed": args.seed, "gamma": args.gamma,
            "alpha_nonjoint": args.alpha_nonjoint,
            "atom_calibration": res.get("atom_calibration") or {},
            "action_table": res.get("action_table") or {},
            "spec": args.spec,
        }, bf16=False)
        rows.append({"fold": int(k), "checkpoint": str(ckpt),
                     "best_epoch": res["best_epoch"], "best_metric": res["best_metric"],
                     "pos_weight": res["pos_weight"], "metrics": met,
                     "history": res["history"]})
        print(f"[E3/atom] fold{k} best_epoch={res['best_epoch']} "
              f"mean_auc={met['mean_auc']} min_auc={met['min_auc']} "
              f"nonjoint_recall={[met['per_target'][t].get('nonjoint_recall') for t in C.TARGETS]}",
              flush=True)

    aucs = [r["metrics"]["min_auc"] for r in rows if r["metrics"].get("min_auc") is not None]
    report = {"stage": "E3", "p_stage": "P-atom", "tag": args.tag,
              "mode": args.mode, "spec": args.spec, "folds": fold_list,
              "exploratory": bool(args.exploratory or args.smoke),
              "min_auc": float(min(aucs)) if aucs else None,
              "rows": rows, "seconds": round(time.time() - t0, 2),
              "n_rows": int(sum(int(np.asarray(r.get("metrics", {}).get("per_target", {})
                                               .get(t, {}).get("n_observed", 0) or 0))
                                for r in rows for t in C.TARGETS))}
    write_json(reports / f"E3_atom_report{('_' + args.tag) if args.tag else ''}.json", report)
    print(json.dumps({"stage": "E3/atom", "min_auc": report["min_auc"],
                      "folds": len(rows), "report": str(
                          reports / f"E3_atom_report{('_' + args.tag) if args.tag else ''}.json")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
