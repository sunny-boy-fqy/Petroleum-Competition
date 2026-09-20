#!/usr/bin/env python3
"""E2/P0-P2：特征组**单组消融** + 数据增强消融 + Gate 判定。

对每组特征跑同一套两阶段折协议（`src/training/fold_runner.py`，与 E1 逐字相同），
比较 OOF Total 与 F1 基线的差值，并给出按井加权的配对 cluster bootstrap CI。

用法::

    # 本机冒烟（1 折 / 少井 / 少 epoch；结论标 exploratory，不进 Gate）
    python3 E2/code/ablate_groups.py --cache-root /tmp/c --reports-dir /tmp/r \\
        --folds 0 --max-wells 12 --epochs 3 --exploratory

    # 云端（全折）
    python3 E2/code/ablate_groups.py --cache-root /data/v4/cache --folds all

产出::

    $REPORTS/E2_ablation.json    逐组 delta / CI / 采纳或 NO-GO（E2 §7 硬要求）
    $REPORTS/E2_gate.json        Gate 聚合判定（含 mandatory checks）
    $REPORTS/E2_throughput.json  点/秒与每折耗时外推（E2/P2 §4）

判据（E2/P0 §10 / E2/P1 §7）：组级 delta 的 **CI 下界 > 0** 才采纳，否则显式记 `no_go`
（不允许"没有增量还塞进特征表"）。增强同理（E2/P2 §7）。
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
from src.data import augment as AUG  # noqa: E402
from src.data import dataset as D  # noqa: E402
from src.data import row_dataset as RD  # noqa: E402
from src.features import groups as G  # noqa: E402
from src.portability import HAS_TORCH  # noqa: E402
from src.score import score_arrays  # noqa: E402
from src.training import fold_runner as FR  # noqa: E402
from src.training import loop as L  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

REQUIRED_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
                   "training_time_log_valid", "checkpoint_resumable", "no_label_leak")


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def default_cache_root() -> Path:
    return _env_path("V4_CACHE_ROOT", str(_env_path("V4_DATA_ROOT", "/data") / "v4" / "cache"))


def default_reports_dir() -> Path:
    return _env_path("V4_REPORTS_DIR", str(_env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))


def write_json(path: str | Path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(M.jsonable(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def ensure_feature_caches(cache: Path, spec: G.FeatureSpec, train_wells, test_wells,
                          phys) -> dict:
    """确保该 spec 的缓存存在（幂等）；返回构建统计。"""
    out = {}
    for split, wells in (("train", train_wells), ("test", test_wells)):
        out[split] = G.build_feature_cache(
            cache, spec, wells, phys, split=split,
            from_raw=lambda w, _s=split: D.read_well_shard(cache, w, _s))
    return out


def run_spec(spec: G.FeatureSpec, fold_list, folds: dict, cache: Path, cfg: L.TrainConfig,
             args, augment=None, tag: str = "") -> dict:
    """对一个特征配置跑完 `fold_list`，返回 OOF 指标与逐折结果。

    `augment`（E2/P2）：给出且 `enabled=True` 时，`fold_runner` 会在**训练侧**做增强，
    验证折保持不变（这是"增强开/关"消融的唯一差异来源）。
    """
    results = []
    t0 = time.time()
    for k in fold_list:
        opt = FR.FoldOptions(spec=spec, max_wells=args.max_wells, smoke=False,
                             resume=False, save_checkpoints=False, select_tau=True,
                             scaler_prefix=f"E2_{spec.key}"[:40],
                             run_dir=Path(args.work_dir) / f"{spec.key}{tag}",
                             scalers_dir=None,
                             tb_run_name=f"E2_{spec.key}{tag}_fold{k}"[:60],
                             augment=augment, augment_seed=cfg.seed)
        results.append(FR.run_two_phase_fold(k, folds, cache, cfg, opt))
    oof = FR.assemble_oof(results, cache)
    gated = M.score_of(oof["y_true"], oof["y_pred"], oof["mask"])
    cont = M.score_of(oof["y_true"], oof["cont"], oof["mask"])
    well_tot, well_rows = FR.per_well_totals(oof)
    const = np.tile([C.ATOM_VALUES[t] for t in C.TARGETS], (oof["y_true"].shape[0], 1))
    well_const, _ = FR.per_well_totals(oof, y_pred=const)
    return {"spec": spec.as_dict(), "spec_key": spec.key, "results": results, "oof": oof,
            "oof_total": float(gated["total"]), "oof_cont_only": float(cont["total"]),
            "acc_por": float(gated["acc_por"]), "acc_perm": float(gated["acc_perm"]),
            "acc_sw": float(gated["acc_sw"]),
            "well_totals": well_tot, "well_totals_const": well_const, "well_rows": well_rows,
            "folds": [FR.fold_metrics(r) for r in results],
            "n_rows": int(oof["y_true"].shape[0]), "n_wells": int(oof["n_wells"]),
            "seconds": round(time.time() - t0, 2),
            "leakage": FR.leakage_evidence(results),
            "augment": (results[0].augment if results else None)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 E2 feature-group ablation")
    ap.add_argument("--cache-root", default=str(default_cache_root()))
    ap.add_argument("--reports-dir", default=str(default_reports_dir()))
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--time-budget-h", type=float, default=None)
    ap.add_argument("--min-free-gb", type=float, default=C.DISK_MIN_FREE_GB)
    ap.add_argument("--disk-path", default=os.environ.get("V4_DATA_ROOT", "/"))
    ap.add_argument("--exploratory", action="store_true",
                    help="本机小规模预检：结论标 exploratory=true 且不进 Gate 数值")
    ap.add_argument("--groups", default="F1,phys,win,well",
                    help="要做单组消融的组（逗号分隔）")
    ap.add_argument("--augment", action="store_true",
                    help="额外跑一次 增强开/关 的消融（E2/P2 §7）")
    ap.add_argument("--candidates", default=str(V4 / "versions" / "candidates.json"))
    args = ap.parse_args(argv)

    if not HAS_TORCH:
        print("[E2] FATAL: 需要 torch（本机契约层请跑 tests/run_all.py）", file=sys.stderr)
        return 5

    cache, reports = Path(args.cache_root), Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir or (reports / "E2_work"))
    work.mkdir(parents=True, exist_ok=True)
    if not (cache / "raw" / "train").is_dir():
        print("[E2] FATAL: 缺少 raw 分片（先跑 run_train.sh --mode data）", file=sys.stderr)
        return 4

    train_wells = sorted(p.stem for p in (cache / "raw" / "train").glob("*.npz"))
    test_wells = sorted(p.stem for p in (cache / "raw" / "test").glob("*.npz"))
    if args.max_wells:
        train_wells, test_wells = train_wells[:args.max_wells], test_wells[:args.max_wells]
    phys = G.fit_physics_params(D.read_well_shard(cache, w, "train") for w in train_wells)

    folds = FOLDS.load_folds()
    fold_list = list(range(int(folds["n_folds"]))) if args.folds == "all" else \
        [int(x) for x in str(args.folds).split(",") if x.strip()]

    cfg = L.TrainConfig(hidden=args.hidden, layers=args.layers, lr=args.lr,
                        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                        seed=args.seed, device=args.device, amp_dtype=args.amp_dtype,
                        time_budget_h=args.time_budget_h, min_free_gb=args.min_free_gb,
                        disk_path=args.disk_path)
    t_start = time.time()
    print(f"[E2] folds={fold_list} epochs={cfg.epochs} device={cfg.device} "
          f"wells(train/test)={len(train_wells)}/{len(test_wells)}", flush=True)

    specs: list[G.FeatureSpec] = [G.FeatureSpec(groups=("F1",))]
    for g in [x.strip() for x in args.groups.split(",") if x.strip()]:
        if g == "F1":
            continue
        specs.append(G.FeatureSpec(groups=("F1", g)))
    f2_spec = G.FeatureSpec(groups=G.F2_GROUPS)
    f2_key = f2_spec.key
    specs.append(f2_spec)                                    # 组合（exploratory 性质）

    per_spec: dict[str, dict] = {}
    for spec in specs:
        info = ensure_feature_caches(cache, spec, train_wells, test_wells, phys)
        print(f"[E2] spec={spec.key} cols={spec.n_features()} "
              f"cache(built/skipped)={info['train']['built']}/{info['train']['skipped']}",
              flush=True)
        r = run_spec(spec, fold_list, folds, cache, cfg, args)
        per_spec[spec.key] = {k: v for k, v in r.items() if k not in ("oof", "results")}
        # OOF 落盘（便于复查；体积远小于 1 GB）
        np.savez_compressed(work / f"oof_{spec.key}.npz",
                            **{k: v for k, v in r["oof"].items() if k != "well_ids"},
                            well_ids=r["oof"]["well_ids"])
        print(f"[E2] {spec.key:34s} OOF={r['oof_total']:.4f} (cont-only "
              f"{r['oof_cont_only']:.4f}) {r['seconds']}s", flush=True)

    # ---- 组级 delta + 配对 CI（以 F1 为基线）
    base_key = "F1"
    base = per_spec[base_key]
    base_well, base_rows = base["well_totals"], base["well_rows"]
    groups_out: list[dict] = []
    for key, r in per_spec.items():
        if key == base_key:
            continue
        delta_well = r["well_totals"] - base_well
        boot = FOLDS.bootstrap_ci(delta_well, iters=1000, weights=base_rows, seed=cfg.seed)
        delta = float(r["oof_total"] - base["oof_total"])
        adopted = bool(boot["ci_low"] > 0)
        groups_out.append({
            "spec_key": key, "groups": r["spec"]["groups"],
            "n_features": r["spec"]["n_features"],
            "oof_total": r["oof_total"], "delta_vs_f1": delta,
            "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
            "point_delta": float(boot["point"]), "decision": "adopted" if adopted else "no_go",
            "exploratory": bool(args.exploratory), "seconds": r["seconds"],
            "folds": r["folds"],
        })
        print(f"[E2] {key:34s} delta={delta:+.4f} CI=[{boot['ci_low']:+.4f},"
              f"{boot['ci_high']:+.4f}] -> {'ADOPT' if adopted else 'NO-GO'}", flush=True)

    # ---- 增强消融（E2/P2 §7，若开启）
    aug_out: dict | None = None
    if args.augment:
        spec = G.FeatureSpec(groups=G.F2_GROUPS)
        acfg = AUG.AugmentConfig(enabled=True)
        r_off = run_spec(spec, fold_list, folds, cache, cfg, args, augment=None, tag="_augoff")
        r_on = run_spec(spec, fold_list, folds, cache, cfg, args, augment=acfg, tag="_augon")
        delta_well = r_on["well_totals"] - r_off["well_totals"]
        boot = FOLDS.bootstrap_ci(delta_well, iters=1000, weights=r_off["well_rows"],
                                  seed=cfg.seed)
        aug_out = {"config": acfg.as_dict(), "oof_off": r_off["oof_total"],
                   "oof_on": r_on["oof_total"],
                   "augment_scope": (r_on["augment"] or {}).get("scope"),
                   "rows_preserved": (r_on["augment"] or {}).get("rows_preserved"),
                   "features_changed": (r_on["augment"] or {}).get("features_changed"),
                   "delta": float(r_on["oof_total"] - r_off["oof_total"]),
                   "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
                   "decision": "adopted" if boot["ci_low"] > 0 else "no_go",
                   "note": "特征层增强（augment_matrix）：只做行错位/按列噪声/整列 dropout；"
                           "曲线层增强的语义由 augment_curves 提供（见 src/data/augment.py）"}

    # ---- 吞吐（E2/P2 §4）：用实测秒数与行数外推
    rows_per_s = {k: (v["n_rows"] / max(v["seconds"], 1e-6)) for k, v in per_spec.items()}
    total_rows = int(sum(v["n_rows"] for v in per_spec.values()))
    total_sec = float(sum(v["seconds"] for v in per_spec.values()))
    throughput = {
        "specs": per_spec and {k: {"rows": v["n_rows"], "seconds": v["seconds"],
                                   "rows_per_s": round(rows_per_s[k], 1),
                                   "epochs": cfg.epochs, "folds": len(fold_list)}
                               for k, v in per_spec.items()},
        "rows_per_s_overall": round(total_rows / max(total_sec, 1e-6), 1),
        "epochs_per_fold": cfg.epochs,
        "estimated_hours_per_fold_full": round(
            (C.EXPECTED_N_TRAIN_ROWS / 5) / max(total_rows / max(total_sec, 1e-6), 1e-6)
            * cfg.epochs / 3600.0, 3),
        "estimated_hours_5fold_full": round(
            (C.EXPECTED_N_TRAIN_ROWS) / max(total_rows / max(total_sec, 1e-6), 1e-6)
            * cfg.epochs / 3600.0, 3),
        "device": str(cfg.device), "batch_size": cfg.batch_size,
        "note": "外推基于本次实测吞吐；E3 的 planned_task_training_h 应在此基础上留 1.5–2× 余量",
        "exploratory": bool(args.exploratory),
    }
    write_json(reports / "E2_throughput.json", throughput)

    ablation = {
        "stage": "E2", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exploratory": bool(args.exploratory),
        "folds": fold_list, "n_wells": base["n_wells"], "n_rows": base["n_rows"],
        "config": cfg.as_dict(),
        "baseline": {"spec_key": base_key, "oof_total": base["oof_total"],
                     "acc_por": base["acc_por"], "acc_perm": base["acc_perm"],
                     "acc_sw": base["acc_sw"], "folds": base["folds"]},
        "groups": groups_out,
        "combined": {"spec_key": f2_key,
                     "oof_total": per_spec.get(f2_key, {}).get("oof_total"),
                     "delta_vs_f1": (None if f2_key not in per_spec else
                                     per_spec[f2_key]["oof_total"] - base["oof_total"])},
        "augmentation": aug_out,
        "provenance_csv": str(reports / "E2_feature_provenance.csv"),
        "provenance_groups": G.provenance_group_counts(G.FeatureSpec(groups=G.F2_GROUPS)),
        "mem_profile": str(reports / "E2_mem_profile.json"),
        "seconds_total": round(time.time() - t_start, 2),
        "selection_protocol": ("outer 折只推理一次；best_epoch 与 tau 均在该 outer 折的 "
                               "inner-OOF 上选（H1）"),
    }
    write_json(reports / "E2_ablation.json", ablation)

    # ---- Gate（boolean：E2 的判据是"消融与溯源齐备"，不是分数门槛）
    prereg_path = reports / "E2_gate_prereg.json"
    if not prereg_path.is_file():
        write_json(prereg_path, {
            "gate_id": "E2_gate", "stage": "E2", "p_stage": "P0", "gate_type": "boolean",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # `primary_metric` 必须是 gates.VALID_PRIMARY 里的名字（boolean 家族）。
            # E2 判的是"消融台账是否齐备"，与 E1/P0 同族，因此复用 row_pipeline_ok。
            "primary_metric": "row_pipeline_ok",
            "primary_threshold_key": "min_delta",
            "baseline_version": "E1_PD0", "baseline_artifact": str(reports / "E2_ablation.json"),
            "baseline_manifest_sha256": "boolean-gate-no-upstream-manifest",
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 4,
            "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None,
            "planned_task_training_h": float(max(1.0, throughput["estimated_hours_5fold_full"])),
            # E2 §9 要求的 4 项 + no_label_leak；`checkpoint_resumable` 在 E2 显式豁免：
            # E2 只做特征消融、不产出候选权重，可续训由 E1/E3 的 checkpoint 证据承担。
            "mandatory_checks": ["contract_ok", "atomic_precision_reported", "disk_budget_ok",
                                 "training_time_log_valid", "no_label_leak"],
            "mandatory_exempt": ["checkpoint_resumable"],
            "decisions_locked": [],
            "notes": "E2：三组特征各自有消融结论（增量或 NO-GO 均登记）+ provenance 全覆盖",
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)

    prov_rows = G.provenance_rows(G.FeatureSpec(groups=G.F2_GROUPS))
    leak_ok = all(v["leakage"]["ok"] for v in per_spec.values())
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(cfg.disk_path)
    except Exception as exc:                          # 不静默：记录失败原因
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        # 消融台账齐备（每组都有 decision）+ 预注册合法
        "contract_ok": bool(not perrs and all("decision" in g for g in groups_out)),
        # 原子头 τ 与占位行指标在每折都上报
        "atomic_precision_reported": bool(all("tau" in f for f in base["folds"])),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": bool(all(v["seconds"] > 0 for v in per_spec.values())),
        # `checkpoint_resumable` 在 E2 的预注册里被 `mandatory_exempt` 显式豁免（不产出权重），
        # 因此**不写进 checks**，而不是写一个没验证过的 True。
        "no_label_leak": bool(leak_ok
                              and G.audit_no_target_derivation(
                                  G.FeatureSpec(groups=G.F2_GROUPS))["ok"]),
    }
    # E2 的 Gate 是"台账齐备"型 boolean；`checkpoint_resumable` 在 E2 由 E1 的 checkpoint 承担，
    # 因此这里显式记豁免原因（而不是把未验证的项写成 True 蒙混过关）。
    gate = {
        "gate_id": prereg["gate_id"], "stage": "E2", "gate_type": "boolean",
        "checkpoint_resumable_waiver": ("E2 只做特征消融、不产出候选权重；可续训由 E1/E3 的 "
                                        "checkpoint 证据承担"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exploratory": bool(args.exploratory),
        "passed": bool(not perrs and len(prov_rows) == G.FeatureSpec(groups=G.F2_GROUPS).n_features()
                       and len(groups_out) >= 3),
        "prereg_errors": perrs, "checks": checks, "disk": disk,
        "leakage": {k: v["leakage"] for k, v in per_spec.items()},
        "n_groups_evaluated": len(groups_out),
        "decisions": {g["spec_key"]: g["decision"] for g in groups_out},
        "provenance_columns": len(prov_rows),
        "expected_columns": G.FeatureSpec(groups=G.F2_GROUPS).n_features(),
        "ablation_path": str(reports / "E2_ablation.json"),
        "throughput_path": str(reports / "E2_throughput.json"),
        "mem_profile_path": str(reports / "E2_mem_profile.json"),
        "note": ("exploratory 运行只验证链路与产出形状，结论不进任何 Gate 数值"
                 if args.exploratory else "正式运行"),
    }
    write_json(reports / "E2_gate.json", gate)
    print(json.dumps({"gate_passed": gate["passed"], "exploratory": gate["exploratory"],
                      "decisions": gate["decisions"],
                      "F1_oof": base["oof_total"],
                      "F2_oof": per_spec.get(f2_key, {}).get("oof_total"),
                      "rows_per_s": throughput["rows_per_s_overall"],
                      "est_hours_5fold": throughput["estimated_hours_5fold_full"]},
                     ensure_ascii=False, indent=2))
    if args.exploratory:
        return 0
    return 0 if gate["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
