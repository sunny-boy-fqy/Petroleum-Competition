#!/usr/bin/env python3
"""E7/P1 附加件：在 inner-OOF 上拟合「温度/等渗校准 + 期望分数动作表」。

用法::

    python3 E7/code/fit_atom_decision.py \
        --oof $RUN_ROOT/E6/state/inner_oof.npz \
        --out-config versions/configs/decode_v1.json \
        --reports-dir $REPORTS_DIR --inner-only

产出：
  * `$REPORTS/E7_atom_decision.json`：交叉拟合的诚实评估、paired CI、动作表摘要；
  * 若采纳：更新 `decode_v1.json` 的 `atom_calibration` / `action_table`。

纪律：
  * 交叉拟合：用 K-1 折拟合、留出折评估，避免"自己评自己"；
  * 只有 `delta > 0` 且 `paired_ci_low > 0` 才写入冻结配置；
  * 纯 numpy，不 import torch；不接触测试标签。
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
from src.inference import atom_decision as AD  # noqa: E402
from src.inference import calibration as CAL  # noqa: E402
from src.inference import decode as DEC  # noqa: E402
from src.inference import atomic_gate as AG  # noqa: E402
from src.score import score_arrays  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402


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


# ---------------------------------------------------------------- 校准
def _atom_labels(y_atom, mask, t: int):
    return (np.asarray(y_atom)[:, t] >= 0.5) & (np.asarray(mask)[:, t].astype(bool))


def fit_calibration(q_atom, y_atom, mask, kind: str = "temperature",
                    n_bins: int = 50) -> dict:
    """逐目标拟合校准器；返回可写入 decode_v1.json 的 dict。"""
    q = np.asarray(q_atom, dtype="float64")
    y = np.asarray(y_atom, dtype="float64")
    m = np.asarray(mask, dtype="float64") > 0
    out: dict = {"kind": str(kind)}
    if kind == "none":
        return out
    if kind == "temperature":
        temps: dict[str, float] = {}
        for j, t in enumerate(C.TARGETS):
            obs = m[:, j] & np.isfinite(q[:, j])
            if obs.sum() < 20 or len(np.unique(y[obs, j])) < 2:
                temps[t] = 1.0
                continue
            fit = CAL.fit_temperature(CAL.logit(q[obs, j]), y[obs, j])
            temps[t] = float(fit["temperature"])
        out["temperature"] = temps
        return out
    if kind == "isotonic":
        iso: dict = {}
        for j, t in enumerate(C.TARGETS):
            obs = m[:, j] & np.isfinite(q[:, j])
            if obs.sum() < 20:
                continue
            iso[t] = CAL.fit_isotonic_binned(q[obs, j], y[obs, j], n_bins=n_bins)
        out["isotonic"] = iso
        return out
    raise ValueError(f"unknown calibration kind: {kind!r}")


def apply_calibration(q_atom, calib: dict) -> np.ndarray:
    return DEC.calibrate_atom_probs(q_atom, calib)


# ---------------------------------------------------------------- 评分
def per_well_scores(pred, y, mask, well_index, n_wells) -> np.ndarray:
    out = np.full(int(n_wells), np.nan, dtype="float64")
    for i in range(int(n_wells)):
        sel = np.asarray(well_index) == i
        if sel.any():
            out[i] = score_arrays(y[sel], pred[sel], missing=~mask[sel].astype(bool),
                                  missing_mode=C.SCORE_MISSING_MODE)["total"]
    return out


def pair_ci(fuse_tot, base_tot, well_index, n_wells) -> dict:
    d = fuse_tot - base_tot
    ok = ~np.isnan(d)
    if not ok.any():
        return {"ok": False, "delta": None, "ci_low": None, "ci_high": None}
    # 井行数权重（cluster bootstrap 的单元权重）
    rows = np.asarray([(np.asarray(well_index) == i).sum() for i in range(int(n_wells))],
                      dtype="float64")[ok]
    boot = FOLDS.bootstrap_ci(d[ok], iters=1000, weights=rows, seed=42)
    return {"ok": True, "delta": float(boot["point"]),
            "ci_low": float(boot["ci_low"]), "ci_high": float(boot["ci_high"]),
            "n_wells": int(ok.sum())}


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E7/P1 原子校准 + 期望分数动作表")
    ap.add_argument("--oof", default=None)
    ap.add_argument("--run-root", default=str(env_path(
        "V4_RUN_ROOT", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))))
    ap.add_argument("--reports-dir", default=str(env_path(
        "V4_REPORTS_DIR", str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))))
    ap.add_argument("--out-config", default=str(V4 / "versions" / "configs" / "decode_v1.json"))
    ap.add_argument("--calibrate", default="temperature",
                    choices=("none", "temperature", "isotonic"))
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--min-bin", type=int, default=200)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.run_root)
    oof_path = Path(args.oof) if args.oof else (
        run_root / "E6" / "state" / f"inner_oof{('_' + args.tag) if args.tag else ''}.npz")
    if not oof_path.is_file():
        print(f"[E7/atom] FATAL: 缺少 inner-OOF {oof_path}", file=sys.stderr)
        return 4
    with np.load(oof_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    for key in ("cont", "q_atom", "y_true", "mask", "y_atom", "well_index"):
        if key not in d:
            print(f"[E7/atom] FATAL: OOF 缺少键 {key!r}", file=sys.stderr)
            return 4
    cont = np.asarray(d["cont"], dtype="float64")
    q_atom = np.asarray(d["q_atom"], dtype="float64")
    y = np.asarray(d["y_true"], dtype="float64")
    mask = np.asarray(d["mask"], dtype="float64")
    y_atom = np.asarray(d["y_atom"], dtype="float64")
    well_index = np.asarray(d["well_index"], dtype="int64")
    n_wells = int(well_index.max()) + 1 if y.shape[0] else 0
    folds = (np.asarray(d["fold_of_row"], dtype="int64")
             if "fold_of_row" in d else np.zeros(y.shape[0], dtype="int64"))
    fold_ids = sorted(set(int(v) for v in folds))

    # ---- baseline：全局 τ（全 OOF 上选，作为对照；交叉拟合版对照见报告注）
    tau_sel = AG.select_tau_per_target(cont=cont, q_atom=q_atom, y=y, mask=mask)
    tau = np.asarray(tau_sel["tau"], dtype="float64")
    base_pred = AG.per_target_hard_switch(cont, q_atom, tau)

    # ---- 交叉拟合：每折用其余折拟合校准 + 动作表
    cross_pred = base_pred.copy()
    cross_actions = np.zeros(y.shape, dtype=bool)
    for k in fold_ids:
        tr = folds != k
        va = folds == k
        calib = fit_calibration(q_atom[tr], y_atom[tr], mask[tr], kind=args.calibrate,
                                n_bins=args.n_bins)
        q_tr = apply_calibration(q_atom[tr], calib)
        table = AD.fit_decision_table(cont[tr], q_tr, y[tr], mask[tr],
                                      n_bins=args.n_bins, min_bin=args.min_bin,
                                      min_gain=args.min_gain)
        q_va = apply_calibration(q_atom[va], calib)
        pred_va, act_va = AD.apply_decision_table(cont[va], q_va, table)
        cross_pred[va] = pred_va
        cross_actions[va] = act_va

    # ---- 全量重拟合（写盘用）
    calib_full = fit_calibration(q_atom, y_atom, mask, kind=args.calibrate,
                                 n_bins=args.n_bins)
    q_full = apply_calibration(q_atom, calib_full)
    table_full = AD.fit_decision_table(cont, q_full, y, mask, n_bins=args.n_bins,
                                       min_bin=args.min_bin, min_gain=args.min_gain)

    # ---- 评估：cross-fit decision vs global tau baseline
    base_well = per_well_scores(base_pred, y, mask, well_index, n_wells)
    cross_well = per_well_scores(cross_pred, y, mask, well_index, n_wells)
    ci = pair_ci(cross_well, base_well, well_index, n_wells)
    base_sc = score_arrays(y, base_pred, missing=~mask.astype(bool),
                           missing_mode=C.SCORE_MISSING_MODE)
    cross_sc = score_arrays(y, cross_pred, missing=~mask.astype(bool),
                            missing_mode=C.SCORE_MISSING_MODE)
    adopted = bool(ci.get("ok") and ci.get("delta") is not None
                   and float(ci["delta"]) > 0 and float(ci.get("ci_low") or -1) > 0)
    if args.smoke or args.exploratory:
        adopted = bool(adopted)

    # ---- 写 decode 配置（只有采纳时才改 action_table；否则保持原样）
    out_cfg = Path(args.out_config)
    if out_cfg.is_file():
        cfg = DEC.load_decode_config(out_cfg)
    else:
        cfg = DEC.DecodeConfig()
    if adopted:
        cfg.atom_calibration = calib_full
        cfg.action_table = table_full
        cfg.inner_only = True
        cfg.selected_on = str(oof_path)
        cfg.notes = ((cfg.notes or "") +
                     "；WP1 原子校准+期望分数动作表（inner-OOF cross-fit）").strip("；")
        DEC.save_decode_config(cfg, out_cfg)
        mirror = reports / out_cfg.name
        if mirror.resolve() != out_cfg.resolve():
            DEC.save_decode_config(cfg, mirror)

    report = {
        "stage": "E7", "p_stage": "P1-atom-decision", "tag": args.tag,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "oof_path": str(oof_path), "n_rows": int(y.shape[0]), "n_wells": n_wells,
        "calibration": args.calibrate, "n_bins": int(args.n_bins),
        "min_bin": int(args.min_bin), "min_gain": float(args.min_gain),
        "baseline_tau": [float(v) for v in tau],
        "baseline": {"total": float(base_sc["total"]), "por": float(base_sc["acc_por"]),
                     "perm": float(base_sc["acc_perm"]), "sw": float(base_sc["acc_sw"])},
        "crossfit_decision": {"total": float(cross_sc["total"]),
                              "por": float(cross_sc["acc_por"]),
                              "perm": float(cross_sc["acc_perm"]),
                              "sw": float(cross_sc["acc_sw"])},
        "paired_ci": ci, "adopted": adopted,
        "n_actions": int(cross_actions.sum()),
        "action_rates": {t: float(cross_actions[:, j].mean())
                         for j, t in enumerate(C.TARGETS)},
        "atom_calibration": calib_full,
        "action_table_summary": {
            t: {"n_bins": len((table_full.get("targets") or {}).get(t, {}).get(
                    "bin_actions", [])),
                "n_atom_bins": int(sum((table_full.get("targets") or {}).get(t, {}).get(
                    "bin_actions", []) or [])),
                "edges": (table_full.get("targets") or {}).get(t, {}).get("bin_edges")}
            for t in C.TARGETS},
        "note": ("交叉拟合：每折用其余折拟合；采纳判据 delta>0 且 paired ci_low>0。"
                 "baseline τ 在全 OOF 上选，只作对照；上线只用 cross-fit 结论 + 全量重拟合表。"),
    }
    write_json(reports / f"E7_atom_decision{('_' + args.tag) if args.tag else ''}.json", report)
    print(json.dumps({"adopted": adopted,
                      "baseline_total": report["baseline"]["total"],
                      "crossfit_total": report["crossfit_decision"]["total"],
                      "paired_ci": ci, "output": str(out_cfg)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
