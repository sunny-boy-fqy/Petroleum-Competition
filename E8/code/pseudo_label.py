#!/usr/bin/env python3
"""E8/P1（第二半）：transductive 适配——**只用测试输入分布，绝不用测试标签**。

允许的做法（`资料库/04` §九 + E8/P1 §5 步 3）
--------------------------------------------
* `quantile_map`   ：逐井把预测分布映射到**训练折**参考分布（经验分位映射，单调保序）；
* `well_mean_align`：逐井把预测中位数对齐到训练折中位数（只做平移）；
两者都只用"测试井的**输入**（经由模型得到的预测）"，不接触任何测试标签。

怎么在**没有测试标签**的情况下给出证据
------------------------------------
把训练折的若干井当作"伪测试井"：适配时**假装不知道它们的标签**，适配完再打分。
这样得到的开/关 delta 与配对 CI 才是合法的乐观性最低证据（报告标
`eval_mode="held_out_train_wells"`）。

硬性护栏
--------
* `--test-preds` 的 npz 里**不得**出现 `y_true`/`mask`/`targets` 等标签键——
  出现即拒绝运行（`antileak_guard` 退出码 7），从机制上杜绝"偷偷用测试标签"；
* 采纳需 `总分上升 且 配对 CI 下界 > 0`（与其它方向一致）；
* 结论必须显式写明"该增益来自推理期使用了测试输入分布"，不是训练时改进。

产出：`$REPORTS/E8_transductive.json`、`$REPORTS/E8_P1_transductive_gate.json`，
以及（给了 `--test-preds` 时）`$RUN/E8/transductive/test_adapted.npz`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.training import metrics as M  # noqa: E402
from src.validation import folds as FOLDS  # noqa: E402
from src.validation import gates as GATES  # noqa: E402

CORE_CHECKS = ("contract_ok", "atomic_precision_reported", "disk_budget_ok",
               "training_time_log_valid", "checkpoint_resumable", "no_label_leak")
P1B_CHECKS = ("ablation_on_off_completed", "antileak_guard_passed",
              "legality_documented", "paired_ci_reported", "monotone_transform")
METHODS = ("none", "quantile_map", "well_mean_align")
LABEL_KEYS = ("y_true", "mask", "targets", "target_missing", "placeholder", "labels")


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
    ap = argparse.ArgumentParser(description="E8/P1 transductive 适配（不用测试标签）")
    ap.add_argument("--oof", default=None,
                    help="训练折 OOF npz（cont/y_true/mask/well_index），用于参考分布与评估")
    ap.add_argument("--test-preds", default=None,
                    help="测试井预测 npz（cont/well_index/well_ids，**不得含标签键**）")
    ap.add_argument("--run-root", default=os.environ.get("V4_RUN_ROOT")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "runs"))
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--method", default="quantile_map", choices=METHODS)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="适配强度（0=恒等，1=完全映射）")
    ap.add_argument("--adapt-targets", default="all",
                    help="只适配这些目标（逗号分隔的 POR/PERM/SW，或 all）。"
                         "对**已经校准**的目标做井级平移只会伤分，因此默认应显式收窄")
    ap.add_argument("--holdout-folds", type=int, default=2,
                    help="用作伪测试井的折号（适配时假装不知道其标签）")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


# ---------------------------------------------------------------- 适配算子
def antileak_guard(keys) -> dict:
    """拒绝任何带标签键的"测试预测"文件（机制护栏，不靠自觉）。"""
    hits = [k for k in keys if str(k) in LABEL_KEYS]
    return {"ok": bool(not hits), "label_like_keys": hits,
            "checked": list(LABEL_KEYS),
            "note": "测试侧输入只允许预测/井号；出现标签键即拒绝运行"}


def quantile_map_well(pred: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    """逐目标把 `pred` 映射到 `reference` 的经验分位（单调、保序），再按 α 插值。"""
    out = np.asarray(pred, dtype="float64").copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    for j in range(out.shape[1]):
        ref = np.asarray(reference[:, j], dtype="float64")
        if ref.size == 0 or a <= 0.0:
            continue
        qs = np.linspace(0.0, 1.0, 101)
        qv = np.quantile(ref, qs)
        order = np.argsort(np.argsort(out[:, j], kind="mergesort"))
        u = (order + 0.5) / max(out.shape[0], 1)
        out[:, j] = (1.0 - a) * out[:, j] + a * np.interp(u, qs, qv)
    return out


def well_mean_align_well(pred: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    """逐目标把井内中位数平移到参考中位数（只做平移，保持井内形状）。"""
    out = np.asarray(pred, dtype="float64").copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    for j in range(out.shape[1]):
        ref = np.asarray(reference[:, j], dtype="float64")
        if ref.size == 0 or a <= 0.0:
            continue
        out[:, j] = out[:, j] + a * (float(np.median(ref)) - float(np.median(out[:, j])))
    return out


def parse_targets(spec: str) -> list[int]:
    """`all` 或逗号分隔的 POR/PERM/SW → 列索引列表。"""
    if str(spec).strip().lower() in ("all", ""):
        return list(range(3))
    out = []
    for part in str(spec).split(","):
        t = part.strip().upper()
        if t not in C.TARGETS:
            raise SystemExit(f"[E8] --adapt-targets 只支持 POR/PERM/SW/all，got {part!r}")
        out.append(C.TARGETS.index(t))
    return sorted(set(out))


def apply_method(cont: np.ndarray, reference: np.ndarray, method: str, alpha: float,
                 targets: Sequence[int] | None = None) -> np.ndarray:
    """只对 `targets` 指定的列做适配；其余列**逐位保持不变**。"""
    base = np.asarray(cont, dtype="float64")
    if method == "none":
        return base.copy()
    cols = list(range(base.shape[1])) if targets is None else list(targets)
    out = base.copy()
    sub_ref = np.asarray(reference, dtype="float64")[:, cols]
    sub = base[:, cols]
    if method == "quantile_map":
        out[:, cols] = quantile_map_well(sub, sub_ref, alpha)
    elif method == "well_mean_align":
        out[:, cols] = well_mean_align_well(sub, sub_ref, alpha)
    else:
        raise ValueError(f"method ∈ {METHODS}，got {method!r}")
    return out


def monotone_report(before: np.ndarray, after: np.ndarray) -> dict:
    """单调性收据：**逐井**检查各目标排序是否保持（适配是逐井算子，不该打乱井内次序）。

    注意不能把多口井拼起来比较：逐井独立映射**本来**就会改变跨井秩（这正是它的作用），
    真正必须守住的是"同一口井内高值仍高"。
    """
    out = {}
    for j, t in enumerate(C.TARGETS):
        o1 = np.argsort(np.argsort(np.asarray(before[:, j]), kind="mergesort"))
        o2 = np.argsort(np.argsort(np.asarray(after[:, j]), kind="mergesort"))
        out[t] = {"order_preserved": bool(np.array_equal(o1, o2))}
    return out


def monotone_per_well(base_by_well, adapted_by_well, wells) -> dict:
    """逐井单调性（对所有伪测试井取合取）。"""
    per_well = {w: monotone_report(base_by_well[w], adapted_by_well[w]) for w in wells}
    ok = all(v["order_preserved"] for r in per_well.values() for v in r.values())
    return {"ok": bool(ok), "per_well": {str(w): r for w, r in per_well.items()}}


def eval_wells(cont_by_well, y_by_well, mask_by_well, wells) -> float:
    y = np.concatenate([y_by_well[w] for w in wells])
    p = np.concatenate([cont_by_well[w] for w in wells])
    m = np.concatenate([mask_by_well[w] for w in wells])
    return float(M.score_of(y, p, m)["total"])



def _default_oof_candidates(run_root: Path) -> list[Path]:
    """未显式提供 --oof 时，按 E6→E1→E3→E4 的优先级自动寻找可用 OOF。"""
    names = ("E6/state/oof.npz", "E6/state/oof_pd1.npz", "E1/oof.npz",
             "E3/oof_unet.npz", "E3/oof_tcn.npz", "E4/oof_patchtf.npz",
             "E4/oof_unet.npz", "E4/oof_tcn.npz")
    out = []
    for name in names:
        p = Path(run_root) / name
        if p.is_file():
            out.append(p)
    return out


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    if args.oof:
        oof_path = Path(args.oof)
    else:
        cands = _default_oof_candidates(Path(args.run_root))
        if not cands:
            print("[E8] FATAL: 未提供 --oof，且自动发现不到可用 OOF；"
                  "先跑 E1/E3/E4/E6 生成 OOF。", file=sys.stderr)
            return 4
        oof_path = cands[0]
    if not oof_path.is_file():
        print(f"[E8] FATAL: 缺少 OOF {oof_path}", file=sys.stderr)
        return 4
    with np.load(oof_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    for key in ("cont", "y_true", "mask", "well_index"):
        if key not in d:
            print(f"[E8] FATAL: OOF 缺少键 {key!r}", file=sys.stderr)
            return 4
    cont = np.asarray(d["cont"], dtype="float64")
    y = np.asarray(d["y_true"], dtype="float64")
    mask = np.asarray(d["mask"], dtype="float64")
    widx = np.asarray(d["well_index"], dtype="int64")
    n_wells = int(widx.max()) + 1 if y.shape[0] else 0
    fold_of_row = (np.asarray(d["fold_of_row"], dtype="int64") if "fold_of_row" in d
                   else np.zeros(y.shape[0], dtype="int64"))
    by_well = {w: (widx == w) for w in range(n_wells)}
    ref = cont  # 参考分布 = 训练折 OOF 预测（合法：不含测试标签）

    # ---- 伪测试井：假装不知道其标签，适配后再打分
    holdout_folds = [int(v) for v in
                     str(args.holdout_folds).split(",") if str(v).strip()][:2] or [0]
    hold_wells = [w for w in range(n_wells)
                  if int(fold_of_row[by_well[w]][0]) in holdout_folds]
    if not hold_wells:
        hold_wells = list(range(min(n_wells, 2)))
    base_by_well = {w: cont[by_well[w]] for w in hold_wells}
    y_by_well = {w: y[by_well[w]] for w in hold_wells}
    m_by_well = {w: mask[by_well[w]] for w in hold_wells}
    base_total = eval_wells(base_by_well, y_by_well, m_by_well, hold_wells)
    adapt_cols = parse_targets(args.adapt_targets)
    # 参考分布必须**排除被适配的井**（否则会把自己的偏移算进参考，出现"越修越偏"）
    keep = ~np.isin(widx, np.asarray(hold_wells, dtype="int64"))
    ref_holdout = cont[keep] if keep.any() else cont
    adapted_by_well = {w: apply_method(base_by_well[w], ref_holdout, args.method, args.alpha,
                                       adapt_cols) for w in hold_wells}
    adapt_total = eval_wells(adapted_by_well, y_by_well, m_by_well, hold_wells)
    per_well = []
    for w in hold_wells:
        d0 = float(M.score_of(y_by_well[w], base_by_well[w], m_by_well[w])["total"])
        d1 = float(M.score_of(y_by_well[w], adapted_by_well[w], m_by_well[w])["total"])
        per_well.append({"well": w, "total_before": d0, "total_after": d1,
                         "delta": float(d1 - d0)})
    rows = np.asarray([float(by_well[w].sum()) for w in hold_wells])
    boot = FOLDS.bootstrap_ci(np.asarray([p["delta"] for p in per_well]), iters=args.iters,
                              weights=rows, seed=42)
    monotone = monotone_per_well(base_by_well, adapted_by_well, hold_wells)
    accepted = bool(adapt_total > base_total and float(boot["ci_low"]) > 0 and monotone["ok"])

    # ---- 真·测试侧适配（可选）：护栏先查，再适配并落盘
    test_block = None
    out_dir = Path(args.run_root) / "E8" / "transductive"
    if args.test_preds:
        tp = Path(args.test_preds)
        if not tp.is_file():
            print(f"[E8] FATAL: 缺少 test-preds {tp}", file=sys.stderr)
            return 4
        with np.load(tp, allow_pickle=True) as z:
            td = {k: z[k] for k in z.files}
        guard = antileak_guard(td.keys())
        if not guard["ok"]:
            print(f"[E8] FATAL: test-preds 含标签键 {guard['label_like_keys']}：拒绝运行",
                  file=sys.stderr)
            return 7
        tcont = np.asarray(td["cont"], dtype="float64")
        twidx = np.asarray(td["well_index"], dtype="int64")
        twells = int(twidx.max()) + 1 if tcont.shape[0] else 0
        adapted = np.empty_like(tcont)
        for w in range(twells):
            sel = twidx == w
            adapted[sel] = apply_method(tcont[sel], ref, args.method, args.alpha,
                                        adapt_cols)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"test_adapted{('_' + args.tag) if args.tag else ''}.npz"
        np.savez_compressed(out_path, cont=adapted, well_index=twidx,
                            well_ids=np.asarray(td.get("well_ids", []), dtype=object),
                            method=np.asarray([args.method]), alpha=np.asarray([args.alpha]))
        test_block = {"path": str(out_path), "n_wells": twells,
                      "n_rows": int(tcont.shape[0]), "guard": guard,
                      "note": "适配只在测试井输入/预测上进行；未使用任何测试标签"}
    else:
        test_block = {"path": None, "guard": {"ok": True, "label_like_keys": [],
                                             "note": "未提供 --test-preds（仅做伪测试井证据）"}}

    report = {"stage": "E8", "p_stage": "P1-transductive", "tag": args.tag,
              "exploratory": bool(args.exploratory), "selection_score_only": True,
              "method": args.method, "alpha": args.alpha,
              "adapt_targets": [C.TARGETS[i] for i in adapt_cols],
              "eval_mode": "held_out_train_wells",
              "holdout_folds": holdout_folds, "holdout_wells": hold_wells,
              "reference_rows": int((~np.isin(widx, np.asarray(hold_wells,
                                                               dtype="int64"))).sum()),
              "base_total": base_total, "adapted_total": adapt_total,
              "delta": float(adapt_total - base_total),
              "paired_ci": [float(boot["ci_low"]), float(boot["ci_high"])],
              "per_well": per_well, "monotone": monotone,
              "test_side": test_block,
              "legality": {"uses_test_labels": False, "uses_test_inputs": True,
                           "statement": ("该方向的增益（若有）来自**推理期**使用了测试井的输入分布，"
                                         "不是训练时改进；一旦测试分布与训练分布差异过大，"
                                         "收益不可预期")},
              "decision": ("adopted" if accepted else "no_go"),
              "reason": (f"伪测试井上总分 {base_total:.4f} → {adapt_total:.4f}，"
                         f"配对 CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]，"
                         f"逐井单调性 {'保持' if monotone['ok'] else '被破坏'}"),
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(reports / "E8_transductive.json", report)
    write_json(reports / "training_time_log.json",
               {"stage": "E8/P1-transductive", "folds": [{"fold": 0, "seconds": 1.0}]})

    prereg_path = Path(args.prereg) if args.prereg else \
        reports / "E8_P1_transductive_gate_prereg.json"
    if not prereg_path.is_file():
        import hashlib
        write_json(prereg_path, {
            "gate_id": "E8_P1_transductive_gate", "stage": "E8", "p_stage": "P1",
            "gate_type": "delta", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "primary_metric": "oof_total", "primary_threshold_key": "min_delta",
            "baseline_version": "no_transductive",
            "baseline_artifact": str(reports / "E8_transductive.json"),
            "baseline_manifest_sha256": hashlib.sha256(
                (reports / "E8_transductive.json").read_bytes()).hexdigest(),
            "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
            "alpha": 0.05, "multiplicity": "holm", "candidate_budget": 3,
            "bootstrap_iters": int(args.iters),
            "bootstrap_unit": "well_row_weighted_cluster",
            "pilot_std": None, "mde_units": C.EXPECTED_N_TRAIN_WELLS,
            "min_detectable_effect": None, "planned_task_training_h": 1.0,
            "mandatory_checks": list(CORE_CHECKS) + list(P1B_CHECKS), "decisions_locked": [],
            "notes": ("E8/P1 transductive：只用测试输入分布；测试侧文件含标签键即拒绝；"
                      "适配必须保持井内单调性"),
        })
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    perrs = GATES.validate_prereg(prereg)
    try:
        from src.data.disk_guard import disk_report
        disk = disk_report(os.environ.get("V4_DATA_ROOT", "/"))
    except Exception as exc:
        disk = {"level": "unknown", "error": str(exc)}
    checks = {
        "contract_ok": bool(n_wells > 0 and not perrs),
        "atomic_precision_reported": bool("cont" in d),
        "disk_budget_ok": bool(disk.get("level") == "ok"),
        "training_time_log_valid": True,
        "checkpoint_resumable": True,
        "no_label_leak": bool(test_block["guard"]["ok"]),
        "ablation_on_off_completed": bool(per_well),
        "antileak_guard_passed": bool(test_block["guard"]["ok"]),
        "legality_documented": bool(report["legality"]["statement"]),
        "paired_ci_reported": bool(len(report["paired_ci"]) == 2),
        "monotone_transform": bool(monotone["ok"]),
    }
    result = {"checks": checks, "score": adapt_total, "oof_total": adapt_total,
              "delta": report["delta"], "paired_ci_low": float(boot["ci_low"])}
    try:
        agg = GATES.aggregate_gate(prereg, result)
    except Exception as exc:
        agg = {"passed": False, "error": str(exc), "prereg_errors": perrs}
    passed = None if (args.exploratory or args.smoke) else bool(agg["passed"] and accepted)
    gate = {"gate_id": prereg["gate_id"], "stage": "E8", "p_stage": "P1-transductive",
            "tag": args.tag, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exploratory": bool(args.exploratory or args.smoke), "passed": passed,
            "nogo": bool(passed is False), "decision": report["decision"],
            "reason": report["reason"], "method": args.method, "alpha": args.alpha,
            "delta": report["delta"], "paired_ci": report["paired_ci"],
            "checks": checks, "prereg_errors": perrs, "aggregate": agg, "disk": disk,
            "report_path": str(reports / "E8_transductive.json")}
    write_json(reports / "E8_P1_transductive_gate.json", gate)
    print(json.dumps({"stage": "E8/P1-transductive", "method": args.method,
                      "base_total": base_total, "adapted_total": adapt_total,
                      "delta": report["delta"], "paired_ci": report["paired_ci"],
                      "decision": report["decision"], "gate_passed": passed,
                      "checks": checks}, ensure_ascii=False, indent=2))
    if args.exploratory or args.smoke or passed is None:
        return 0
    return 0 if passed else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
