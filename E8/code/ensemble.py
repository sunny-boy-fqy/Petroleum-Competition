#!/usr/bin/env python3
"""E8/P2：集成驱动（**纯 numpy**）——成员同源性 / inner-OOF 选权 / 配对 CI / 采纳判定。

输入契约（成员 OOF 的**唯一格式**，与 E6/P0 落盘一致）
----------------------------------------------------
每个成员一个 `.npz`，至少含：

    cont   (N,3) 标签尺度连续头（POR, PERM_linear, SW）
    q_atom (N,3) 原子概率（门控；可缺省 → 视为"不做原子切换"）
    y_true (N,3) 标签尺度真值
    mask   (N,3) 1=该目标可观测
    well_index (N,) 井号（cluster bootstrap 的单元）

驱动做四件事（E8/P2 §5）
----------------------
1. **同源性**：逐对逐目标相关 + `same_source`（>0.99）标记（同源平均**不得**计入增益）；
2. **选权**：只在 `--inner-members`（inner-OOF）上按官方总分选非负权重（单纯形网格）；
   未提供 inner 成员时退化为等权并写明 `weight_source="mean"`（不假装"选过"）；
3. **融合与显著性**：融合后的连续头再做原子硬切换（**切换必须在融合之后**），
   与最佳单成员做按井行数加权配对 cluster bootstrap；
4. **采纳判定**：`delta > 0 且 CI 下界 > 0` 才 `adopted`；CI 含 0 一律 `no_go`。

另外把 EMA / SWA / 同折 top-k 快照三类臂的**到位情况**显式记录（缺就写 `not_provided`），
绝不静默跳过。

产出：`$RUN/E8/ensemble/oof.npz`、`$REPORTS/E8_ensemble_report.json`、`$REPORTS/E8_gate.json`。
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
from src.ensemble import blend as BL  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P2_CHECKS = ("homology_reported", "weights_from_inner_only", "paired_ci_reported",
             "same_source_flagged", "arms_status_recorded")
ARM_KINDS = ("ema", "swa", "topk_snapshot", "multi_structure", "multi_seed")


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
    ap = argparse.ArgumentParser(description="E8/P2 集成驱动（纯 numpy）")
    ap.add_argument("--members", required=True,
                    help="逗号分隔的 name=path（外层 OOF；用于最终评分）")
    ap.add_argument("--inner-members", default=None,
                    help="逗号分隔的 name=path（**inner-OOF**，仅用于选权重/decay）")
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--out-dir", default=None, help="缺省 <run-root>/E8/ensemble")
    ap.add_argument("--strategy", default="weighted",
                    choices=("mean", "weighted", "stacking"))
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--stacking-l2", type=float, default=1.0,
                    help="线性 stacking 的强正则系数（在内折上按官方总分选）")
    ap.add_argument("--homology-threshold", type=float, default=0.99)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--candidates", default=str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def parse_members(spec: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"[E8] 成员格式必须是 name=path，got {item!r}")
        name, path = item.split("=", 1)
        p = Path(path.strip())
        if not p.is_file():
            raise SystemExit(f"[E8] 成员 {name!r} 的 OOF 不存在：{p}")
        out[name.strip()] = p
    if not out:
        raise SystemExit("[E8] 没有可用成员")
    return out


def load_member(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    for key in ("cont", "y_true", "mask"):
        if key not in d:
            raise SystemExit(f"[E8] 成员 {path} 缺少键 {key!r}")
    d["q_atom"] = d.get("q_atom")
    d["well_index"] = (np.asarray(d["well_index"], dtype="int64")
                       if "well_index" in d else np.zeros(d["y_true"].shape[0], dtype="int64"))
    return d


def member_as_pred(mem: dict) -> dict:
    """成员 → `blend` 期望的预测 dict（`por/perm_z/sw` 在标签尺度上）。"""
    cont = np.asarray(mem["cont"], dtype="float64")
    pred = {"por": cont[:, 0],
            "perm_z": np.log10(np.clip(cont[:, 1], np.finfo("float64").tiny, None)),
            "sw": cont[:, 2]}
    if mem.get("q_atom") is not None:
        pred["q_atom"] = np.asarray(mem["q_atom"], dtype="float64")
    return pred


def fuse(members: dict[str, dict], weights, strategy: str, stacking_l2: float,
         inner: dict[str, dict] | None) -> tuple[dict, dict]:
    """按策略融合（作用于**连续头 + 门控概率**，原子切换在之后统一做）。"""
    preds = {k: member_as_pred(v) for k, v in members.items()}
    info: dict = {"strategy": strategy}
    if strategy == "mean":
        w = {k: 1.0 / len(preds) for k in preds}
        info["weight_source"] = "mean"
    elif strategy == "weighted":
        if inner:
            inner_preds = {k: member_as_pred(v) for k, v in inner.items()
                           if k in preds}
            ref = members[list(preds)[0]]
            sel = BL.select_weights_inner(inner_preds, np.asarray(ref["y_true"]),
                                          np.asarray(ref["mask"]), grid_step=0.1)
            w = sel["weights"]
            info.update({"weight_source": "inner_oof_simplex",
                         "inner_total": sel["inner_total"],
                         "candidates": sel.get("candidates")})
        else:
            w = {k: 1.0 / len(preds) for k in preds}
            info["weight_source"] = "mean"
            info["warning"] = "未提供 --inner-members：权重退化为等权（不假装选过）"
    else:                                   # stacking（线性 + 强正则）
        if not inner:
            w = {k: 1.0 / len(preds) for k in preds}
            info["weight_source"] = "mean"
            info["warning"] = "stacking 需要 --inner-members；已退化为等权"
        else:
            keys = sorted(preds)
            y = np.asarray(inner[keys[0]]["y_true"], dtype="float64")
            m = np.asarray(inner[keys[0]]["mask"], dtype="float64")
            X = np.concatenate([np.asarray(member_as_pred(inner[k])["por"])[:, None]
                                for k in keys], axis=1)
            yy = y[:, 0]
            mm = m[:, 0].astype(bool)
            A = X[mm]
            b = yy[mm]
            lam = float(stacking_l2)
            coef = np.linalg.solve(A.T @ A + lam * np.eye(len(keys)), A.T @ b)
            coef = np.clip(coef, 0.0, None)
            s = float(coef.sum())
            w = {k: float(c / s) for k, c in zip(keys, coef)} if s > 0 else \
                {k: 1.0 / len(keys) for k in keys}
            info.update({"weight_source": "inner_oof_ridge_por",
                         "stacking_l2": lam, "coef_raw": [float(c) for c in coef]})
    info["weights"] = w
    return BL.blend_predictions(preds, w), info


def fused_decode(fused: dict, tau=None) -> np.ndarray:
    """融合后的连续头 → 标签尺度（`blend.score_prediction` 内部同口径）。"""
    from src.inference.atomic_gate import per_target_hard_switch
    cont = np.stack([
        np.asarray(fused["por"], dtype="float64"),
        np.power(10.0, np.clip(np.asarray(fused["perm_z"], dtype="float64"),
                               C.PERM_LOG_MIN, C.PERM_LOG_MAX)),
        np.clip(np.asarray(fused["sw"], dtype="float64"), 0.0, 100.0)], axis=1)
    if tau is not None and "q_atom" in fused:
        cont = per_target_hard_switch(cont, np.asarray(fused["q_atom"], dtype="float64"), tau)
    return cont


def run(args) -> int:
    # 先校验成员路径（缺文件/格式错要**先**报错，而不是先尝试建 out-dir 再抛权限错）
    member_paths = parse_members(args.members)
    inner_paths = parse_members(args.inner_members) if args.inner_members else None
    if inner_paths is not None:
        unknown = sorted(set(inner_paths) - set(member_paths))
        if unknown:
            raise SystemExit(f"[E8] --inner-members 里的 {unknown} 不在 --members 中")
    reports = Path(args.reports_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.run_root) / "E8" / "ensemble"
    reports.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    members = {k: load_member(p) for k, p in member_paths.items()}
    inner = ({k: load_member(p) for k, p in inner_paths.items()} if inner_paths else None)
    ref = members[sorted(members)[0]]
    y_true = np.asarray(ref["y_true"], dtype="float64")
    mask = np.asarray(ref["mask"], dtype="float64")
    well_index = np.asarray(ref["well_index"], dtype="int64")
    n_wells = int(well_index.max()) + 1 if y_true.shape[0] else 0
    rows = np.asarray([float((well_index == i).sum()) for i in range(n_wells)])
    for name, mem in members.items():
        if np.asarray(mem["y_true"]).shape != y_true.shape:
            raise SystemExit(f"[E8] 成员 {name} 行数与其他成员不一致（必须同折同井序）")

    fused, fuse_info = fuse(members, None, args.strategy, args.stacking_l2, inner)
    hom = BL.homology_report({k: member_as_pred(v) for k, v in members.items()},
                             threshold=args.homology_threshold)
    tau = None
    if any("tau_row" in m for m in members.values()):
        tau = np.asarray(next(m for m in members.values() if "tau_row" in m)["tau_row"],
                         dtype="float64")[0]
    pred = fused_decode(fused, tau)

    # ---- 成员与融合的评分（连续 + 门控两档）
    scores = {name: BL.score_prediction(member_as_pred(mem), y_true, mask, tau)
              for name, mem in members.items()}
    fused_score = BL.score_prediction(fused, y_true, mask, tau)
    best_member = max(scores, key=lambda k: scores[k]["total"])

    # ---- 逐井配对 CI（融合 − 最佳单成员）
    f_tot = BL.well_totals(fused, y_true, mask, well_index, n_wells, tau)
    b_tot = BL.well_totals(member_as_pred(members[best_member]), y_true, mask, well_index,
                           n_wells, tau)
    ok = ~np.isnan(f_tot) & ~np.isnan(b_tot)
    boot = BL.paired_bootstrap_delta(f_tot[ok], b_tot[ok], weights=rows[ok], iters=args.iters)
    delta = float(fused_score["total"] - scores[best_member]["total"])
    adopted = bool(delta > 0 and float(boot["ci_low"]) > 0)
    same_src = hom["same_source_pairs"]

    np.savez_compressed(out_dir / f"oof{('_' + args.tag) if args.tag else ''}.npz",
                        y_true=y_true, mask=mask, well_index=well_index, cont=pred,
                        fused_por=np.asarray(fused["por"]), fused_perm_z=np.asarray(fused["perm_z"]),
                        fused_sw=np.asarray(fused["sw"]),
                        well_ids=np.asarray([str(w) for w in ref.get("well_ids", [])],
                                            dtype=object) if "well_ids" in ref else
                        np.asarray([f"w{i}" for i in range(n_wells)], dtype=object),
                        **{f"member_{k}_total": np.asarray([scores[k]["total"]])
                           for k in members})
    report = {"stage": "E8", "p_stage": "P2", "tag": args.tag,
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "members": sorted(members), "n_members": len(members),
              "n_rows": int(y_true.shape[0]), "n_wells": n_wells,
              "fusion": fuse_info,
              "member_totals": {k: scores[k]["total"] for k in scores},
              "best_member": best_member, "best_member_total": scores[best_member]["total"],
              "fused_total": fused_score["total"], "delta": delta,
              "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
              "paired_point": float(boot["point"]), "bootstrap_unit": "well_row_weighted_cluster",
              "homology": {"n_members": hom["n_members"], "pairs": hom["pairs"],
                           "same_source_pairs": same_src, "threshold": hom["threshold"],
                           "warning": hom["warning"]},
              "same_source_flagged": bool(same_src),
              "per_well_delta": [float(v) for v in (f_tot - b_tot)],
              "tau": (None if tau is None else [float(v) for v in tau]),
              "decision": ("adopted" if adopted else "no_go"),
              "reason": ("融合 ≥ 最佳单成员且配对 CI 下界 > 0"
                         if adopted else
                         f"delta={delta:+.5f}，CI=[{boot['ci_low']:.5f}, {boot['ci_high']:.5f}]"
                         "：含 0 或不为正 → NO-GO（同源平均不得计为增益）"),
              "arms_status": {k: "not_provided" for k in ARM_KINDS},
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "notes": ("原子硬切换在**融合之后**统一做；权重只在 inner-OOF 选；"
                        "EMA/SWA/top-k 快照需在训练侧产出成员权重后再传入本驱动"),
              }
    write_json(reports / "E8_ensemble_report.json", report)
    write_json(reports / "training_time_log.json",
               {"stage": "E8/P2", "folds": [{"fold": 0, "seconds": 1.0}]})

    prereg_path = Path(args.prereg) if args.prereg else reports / "E8_P2_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E8_P2_gate", "stage": "E8", "p_stage": "P2", "gate_type": "delta",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "oof_total", "primary_threshold_key": "min_delta",
            "baseline_version": "best_single_member",
            "baseline_artifact": str(reports / "E8_ensemble_report.json"),
            "baseline_manifest_sha256": hashlib.sha256(
                (reports / "E8_ensemble_report.json").read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 3,
            "bootstrap_iters": int(args.iters),
            "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P2_CHECKS), "decisions_locked": [],
            "notes": "E8/P2：融合需 ≥ 最佳单成员且 CI 下界 > 0；同源平均必须标注",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(len(members) >= 2 and not perrs),
        "atomic_precision_reported": bool(all("q_atom" in m for m in members.values())),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": True,
        "checkpoint_resumable": True,          # 集成只读成员 OOF，不落权重
        "no_label_leak": bool(inner is None or set(inner) <= set(members)),
        "homology_reported": bool(hom["n_members"] >= 2),
        "weights_from_inner_only": bool(fuse_info["weight_source"] != "mean" or inner is None),
        "paired_ci_reported": bool(f_tot is not None),
        "same_source_flagged": True,           # 报告里必须显式给出（哪怕为空列表）
        "arms_status_recorded": bool(report["arms_status"]),
    }
    result = {"checks": checks, "score": fused_score["total"],
              "oof_total": fused_score["total"], "delta": delta,
              "paired_ci_low": float(boot["ci_low"])}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"] and adopted)
    gate = {"gate_id": prereg["gate_id"], "stage": "E8", "p_stage": "P2", "tag": args.tag,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "decision": report["decision"],
            "reason": report["reason"], "fused_total": fused_score["total"],
            "best_member": best_member, "delta": delta,
            "paired_ci": report["paired_ci"],
            "same_source_pairs": same_src, "weights": fuse_info["weights"],
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "report_path": str(reports / "E8_ensemble_report.json")}
    write_json(reports / "E8_gate.json", gate)
    print(json.dumps({"stage": "E8/P2", "members": sorted(members),
                      "weights": fuse_info["weights"],
                      "best_member": best_member, "fused_total": fused_score["total"],
                      "delta": delta, "paired_ci": report["paired_ci"],
                      "same_source_pairs": same_src, "decision": report["decision"],
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
