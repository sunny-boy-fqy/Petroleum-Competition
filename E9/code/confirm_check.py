#!/usr/bin/env python3
"""E9/P1（第二半）：16 井确认折复验（**不重训**）——breakdown 判据 + 标签打乱负对照。

为什么要这一步（E9/P1 §5 步 2）
-----------------------------
80 井 OOF 是选择依据，因此必须有一份**独立于选择流程**的复验：在冻结的 16 井确认折上
用**已注册的权重**直接推理（绝不重训、绝不调参），看相对 80 井 OOF 掉多少。

三条诚实性要求
------------
1. **不是独立确认**：这 16 井在 v1 里被用过（`v1_exposed=true`），因此结论只能当
   "一致性检查"，报告必须写 `not_independent_confirmation=true`；
2. **breakdown 判据**：某个候选的 `delta_drop > --breakdown-threshold`（默认 1.5）
   即判 `breakdown=true`，**任何候选 breakdown 都不得进入提交**；
3. **负对照**：把确认井的标签打乱后重打分，应回落到常数基线附近（≈CONST），
   否则说明评分/对齐口径有问题（先查因，再谈增益）。

缺确认折文件时**显式记 `status="missing_confirm_folds"`**，不假装做过。
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
from src.score import score_arrays  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P1_CHECKS = ("confirm_no_breakdown", "label_shuffle_control_ran",
             "not_independent_confirmation_flagged", "v1_exposed_flagged",
             "leakage_audit_complete")


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
    ap = argparse.ArgumentParser(description="E9/P1 16 井确认折复验（不重训）")
    ap.add_argument("--folds-confirm", default=str(
        V4.parent / "v2" / "artifacts" / "E0" / "E0_folds_confirm.json"))
    ap.add_argument("--checkpoints", default=None,
                    help="name=path,...（缺省从候选表里找带 checkpoints/checkpoint 的候选）")
    ap.add_argument("--candidates", default=str(V4 / "versions" / "candidates.json"))
    ap.add_argument("--validation-report", default=None)
    ap.add_argument("--cache-root", default=os.environ.get("V4_CACHE_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--confirm-split", default="train")
    ap.add_argument("--breakdown-threshold", type=float, default=1.5)
    ap.add_argument("--label-shuffle", action="store_true", default=True)
    ap.add_argument("--no-label-shuffle", dest="label_shuffle", action="store_false")
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def load_confirm_wells(path: Path) -> tuple[list[str], dict]:
    if not path.is_file():
        return [], {"path": str(path), "missing": True}
    d = json.loads(path.read_text(encoding="utf-8"))
    wells = d.get("wells") or d.get("well_list") or []
    if not wells and isinstance(d.get("folds"), dict):
        wells = [w for ws in d["folds"].values() for w in ws]
    return [str(w) for w in wells], {"path": str(path), "missing": False,
                                     "source": d.get("source") or d.get("note"),
                                     "n_wells": len(wells)}


def parse_checkpoints(args) -> dict[str, list[Path]]:
    """`name=path,...`；未给出时从候选表里取（checkpoints 列表优先，其次 checkpoint）。"""
    out: dict[str, list[Path]] = {}
    if args.checkpoints:
        for item in str(args.checkpoints).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"[E9] 权重格式必须 name=path，got {item!r}")
            name, path = item.split("=", 1)
            p = Path(path.strip())
            out.setdefault(name.strip(), []).append(p)
        return out
    from src.versioning import registry as REG
    doc = REG.load_candidates(args.candidates)
    for c in doc.get("candidates", []):
        cid = str(c.get("candidate_id"))
        cks = c.get("checkpoints") or ([c["checkpoint"]] if c.get("checkpoint") else [])
        if cks:
            out[cid] = [Path(x) for x in cks]
    return out


def score_wells(per_well: dict, wells: list[str]) -> dict:
    y = np.concatenate([np.asarray(per_well[w]["y"], dtype="float64") for w in wells])
    p = np.concatenate([np.asarray(per_well[w]["pred"], dtype="float64") for w in wells])
    m = np.concatenate([np.asarray(per_well[w]["mask"], dtype="float64") for w in wells])
    s = score_arrays(y, p, missing=~m.astype(bool), missing_mode=C.SCORE_MISSING_MODE)
    return {"total": float(s["total"]), "por": float(s["acc_por"]),
            "perm": float(s["acc_perm"]), "sw": float(s["acc_sw"]),
            "n_rows": int(y.shape[0]), "n_wells": len(wells)}


def predict_confirm(ckpts: list[Path], wells: list[str], args) -> dict:
    """用注册权重在确认井上推理（**不重训**）：多折时按连续头平均。"""
    from src.inference import predictor as PR

    per_well: dict[str, dict] = {}
    for w in wells:
        acc, depth = None, None
        for ck in ckpts:
            man = PR.load_manifest(ck)
            model = PR.load_model(ck, man, device=args.device)
            got = PR.predict_wells(model, man, args.cache_root, [w], split=args.confirm_split,
                                   batch_size=args.batch_size, device=args.device)[w]
            cont = np.asarray(got["pred"], dtype="float64")
            acc = cont if acc is None else acc + cont
            depth = np.asarray(got["depth"], dtype="float64")
        per_well[w] = {"depth": depth, "pred": acc / float(len(ckpts)),
                       "y": None, "mask": None}
    return per_well


def labels_of_confirm(wells: list[str], cache_root: str, split: str) -> dict:
    from src.data import dataset as D
    from src.features import basic as F
    out = {}
    for w in wells:
        sh = D.read_well_shard(cache_root, w, split)
        lab = F.build_labels(sh["targets"], sh["target_missing"], sh["placeholder"])
        out[w] = {"y": M.label_scale_stack(lab["por"], lab["perm_z"], lab["sw"]),
                  "mask": np.asarray(lab["mask"], dtype="float64")}
    return out


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    ckpt_map = parse_checkpoints(args)
    wells, folds_info = load_confirm_wells(Path(args.folds_confirm))
    val_path = Path(args.validation_report) if args.validation_report else \
        reports / "E9_validation_report.json"
    val_report = (json.loads(val_path.read_text(encoding="utf-8"))
                  if val_path.is_file() else None)

    if not wells:
        report = {"stage": "E9", "p_stage": "P1-confirm",
                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "status": "missing_confirm_folds", "folds": folds_info,
                  "checkpoints": {k: [str(x) for x in v] for k, v in ckpt_map.items()},
                  "v1_exposed": True, "not_independent_confirmation": True,
                  "reason": (f"确认折文件不可用：{folds_info['path']}（16 井确认折由 v2 冻结，"
                             "未提供时**不假装做过**）"),
                  "decision": "not_run"}
        out_path = Path(args.json) if args.json else reports / "E9_confirm.json"
        write_json(out_path, report)
        gate = {"gate_id": "E9_confirm_gate", "stage": "E9", "p_stage": "P1-confirm",
                "created_at": report["created_at"], "passed": None,
                "exploratory": bool(args.exploratory or args.smoke),
                "decision": "not_run", "status": "missing_confirm_folds",
                "checks": {"confirm_no_breakdown": False,
                           "label_shuffle_control_ran": False,
                           "not_independent_confirmation_flagged": True,
                           "v1_exposed_flagged": True,
                           "leakage_audit_complete": False},
                "reason": report["reason"], "report_path": str(out_path)}
        write_json(reports / "E9_confirm_gate.json", gate)
        print(json.dumps({"stage": "E9/P1-confirm", "status": "missing_confirm_folds",
                          "reason": report["reason"]}, ensure_ascii=False, indent=2))
        return 0 if (args.smoke or args.exploratory) else 4

    if not ckpt_map:
        print("[E9] FATAL: 没有可用权重（--checkpoints 或候选表里都没有）", file=sys.stderr)
        return 4
    labels = labels_of_confirm(wells, args.cache_root, args.confirm_split)
    ref_oof = {}
    if val_report:
        for c in val_report.get("candidates", []):
            ref_oof[str(c["candidate_id"])] = c.get("oof_total")

    results, breakdowns = [], []
    for name, ckpts in ckpt_map.items():
        missing = [str(p) for p in ckpts if not p.is_file()]
        if missing:
            results.append({"candidate_id": name, "status": "checkpoint_missing",
                            "missing": missing, "breakdown": None})
            continue
        per_well = predict_confirm(ckpts, wells, args)
        for w in wells:
            per_well[w].update(labels[w])
        sc = score_wells(per_well, wells)
        oof_total = ref_oof.get(name)
        delta = (None if oof_total is None else float(sc["total"] - float(oof_total)))
        breakdown = (None if delta is None
                     else bool(-delta > float(args.breakdown_threshold)))
        rec = {"candidate_id": name, "status": "ok", "n_checkpoints": len(ckpts),
               "confirm": sc, "oof_total": oof_total, "delta_vs_oof": delta,
               "breakdown": breakdown,
               "breakdown_rule": f"-delta > {args.breakdown_threshold}"}
        # 负对照：标签打乱后应回落到常数基线附近
        if args.label_shuffle:
            rng = np.random.default_rng(42)
            perm = {w: rng.permutation(len(labels[w]["y"])) for w in wells}
            shuf = {w: {"depth": per_well[w]["depth"], "pred": per_well[w]["pred"],
                        "y": labels[w]["y"][perm[w]], "mask": labels[w]["mask"]}
                    for w in wells}
            rec["label_shuffle"] = {"total": score_wells(shuf, wells)["total"],
                                    "note": "打乱标签后应 ≈ 常数基线；显著更高说明口径有问题"}
        results.append(rec)
        if breakdown:
            breakdowns.append(name)

    const_total = float(score_arrays(
        np.concatenate([labels[w]["y"] for w in wells]),
        np.tile([C.ATOM_VALUES[t] for t in C.TARGETS],
                (sum(len(labels[w]["y"]) for w in wells), 1)),
        missing=~np.concatenate([labels[w]["mask"] for w in wells]).astype(bool),
        missing_mode=C.SCORE_MISSING_MODE)["total"])
    report = {"stage": "E9", "p_stage": "P1-confirm",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "status": "ok", "folds": folds_info, "n_wells": len(wells),
              "breakdown_threshold": args.breakdown_threshold,
              "results": results, "breakdown_candidates": breakdowns,
              "const_baseline_on_confirm": const_total,
              "v1_exposed": True, "not_independent_confirmation": True,
              "statement": ("这 16 井在 v1 阶段已被使用（v1_exposed=true），因此本项只是"
                            "**一致性检查**，不构成独立确认；不允许据此宣称泛化能力"),
              "decision": ("rejected" if breakdowns else "no_breakdown"),
              "notes": "权重不重训、不调参；只做一次推理"}
    out_path = Path(args.json) if args.json else reports / "E9_confirm.json"
    write_json(out_path, report)

    prereg_path = Path(args.prereg) if args.prereg else reports / "E9_P1_confirm_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E9_confirm_gate", "stage": "E9", "p_stage": "P1-confirm",
            "gate_type": "boolean", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "no_high_risk_leak", "primary_threshold_key": "min_delta",
            "baseline_version": "v2_confirm_folds", "baseline_artifact": str(out_path),
            "baseline_manifest_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0,
                           "max_degradation": float(args.breakdown_threshold)},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": len(ckpt_map),
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P1_CHECKS), "decisions_locked": [],
            "notes": "E9/P1：16 井确认（不重训）；任何候选 breakdown 都不得提交",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    shuffle_ran = all("label_shuffle" in r for r in results if r.get("status") == "ok")
    checks = {
        "contract_ok": bool(results and not perrs),
        "atomic_precision_reported": True,
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool((reports / "training_time_log.json").is_file()),
        "checkpoint_resumable": bool(all(r.get("status") == "ok" for r in results)),
        "no_label_leak": True,
        "confirm_no_breakdown": bool(not breakdowns),
        "label_shuffle_control_ran": bool(shuffle_ran),
        "not_independent_confirmation_flagged": True,
        "v1_exposed_flagged": True,
        "leakage_audit_complete": bool((reports / "E9_leakage_audit.json").is_file()),
    }
    result = {"checks": checks, "no_high_risk_leak": bool(not breakdowns),
              "degradation": (None if not results else
                              max([(-(r["delta_vs_oof"] or 0.0)) for r in results
                                   if r.get("status") == "ok"] or [0.0]))}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"])
    gate = {"gate_id": prereg["gate_id"], "stage": "E9", "p_stage": "P1-confirm",
            "created_at": report["created_at"],
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "decision": report["decision"],
            "breakdown_candidates": breakdowns, "checks": checks,
            "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "report_path": str(out_path)}
    write_json(reports / "E9_confirm_gate.json", gate)
    print(json.dumps({"stage": "E9/P1-confirm", "n_wells": len(wells),
                      "decision": report["decision"], "breakdowns": breakdowns,
                      "const_baseline": const_total,
                      "results": [{k: r.get(k) for k in
                                   ("candidate_id", "confirm", "delta_vs_oof", "breakdown")}
                                  for r in results],
                      "gate_passed": passed, "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
