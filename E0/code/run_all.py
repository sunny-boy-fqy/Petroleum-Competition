#!/usr/bin/env python3
"""E0 一键复算：数据卡 + 三状态计数 + 折指纹 + 常数基线评分 + 契约自检。

    python3 v4/E0/code/run_all.py [--train-dir ../data/train] [--test-dir ../data/test]
                                  [--out reports/E0_data_card.json]

只依赖标准库 + numpy，**不需要 torch / GPU**，因此本机（开发机）即可运行。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

# 云端：产物写到云盘持久目录（V4_REPORTS_DIR）；本机：默认 <v4>/reports
REPORTS_DIR = Path(os.environ.get("V4_REPORTS_DIR", str(V4 / "reports")))
DATA_ROOT = os.environ.get("V4_DATA_ROOT")

from src import constants as C                      # noqa: E402
from src.data import labels as L                    # noqa: E402
from src.data import parse as P                     # noqa: E402
from src.inference import contract as CT            # noqa: E402
from src.portability import describe, HAS_NUMPY     # noqa: E402
from src.score import constant_prediction, score_arrays  # noqa: E402
from src.validation import folds as F               # noqa: E402


def echo(msg: str) -> None:
    print(f"[E0] {msg}", flush=True)


def contract_selftest(v4: Path) -> dict:
    """正/负样例契约自检（不需要 torch）。"""
    good = {
        "modelId": "",
        "modelName": "t",
        "version": "1.0",
        "resultData": [{
            "logId": "w1",
            "predictions": [
                {"depth": 1.0, "POR": 0.1, "PERM": 0.01, "SW": 99.9},
                {"depth": 1.1, "POR": 0.12, "PERM": 1.5, "SW": 0.4},
            ],
        }],
    }
    cases: dict[str, dict] = {}
    r = CT.validate_payload(good, expected_rows=2, strict_keys=True)
    cases["good_minimal"] = {"ok": r.ok, "errors": r.errors}

    bad = json.loads(json.dumps(good))
    bad["resultData"][0]["predictions"][1]["PERM"] = 0.0
    cases["bad_perm_nonpositive"] = {"ok": CT.validate_payload(bad, expected_rows=2).ok}

    bad = json.loads(json.dumps(good))
    bad.pop("version")
    cases["bad_missing_toplevel"] = {"ok": CT.validate_payload(bad, expected_rows=2).ok}

    cases["bad_rowcount"] = {"ok": CT.validate_payload(good, expected_rows=3).ok}

    bad = json.loads(json.dumps(good))
    bad["resultData"][0]["predictions"][1]["depth"] = 1.0
    cases["bad_depth_order"] = {"ok": CT.validate_payload(bad, expected_rows=2).ok}

    bad = json.loads(json.dumps(good))
    p0 = bad["resultData"][0]["predictions"][0]
    p0["DEPTH"] = p0.pop("depth")
    cases["bad_uppercase_depth"] = {"ok": CT.validate_payload(bad, expected_rows=2).ok}

    bad = json.loads(json.dumps(good))
    bad["resultData"][0]["predictions"][0]["SW"] = float("nan")
    cases["bad_nan"] = {"ok": CT.validate_payload(bad, expected_rows=2).ok}

    passed = cases["good_minimal"]["ok"] and all(
        not v["ok"] for k, v in cases.items() if k.startswith("bad_")
    )
    out = {"passed": passed, "cases": cases}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "E0_contract_tests.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def export_folds(v4: Path) -> dict:
    folds = F.load_folds()
    man = F.fold_manifest(folds)
    inner: dict[str, dict[str, list[str]]] = {}
    for f in range(folds["n_folds"]):
        train_wells = [w for w, k in folds["fold_of_well"].items() if k != f]
        inner[str(f)] = F.make_inner_folds(train_wells, C.N_INNER_FOLDS,
                                           seed=folds["seed"] or 42)
    art = v4 / "artifacts" / "E0"
    art.mkdir(parents=True, exist_ok=True)
    (art / "folds.json").write_text(
        json.dumps({"outer": folds, "inner": inner}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (v4 / "versions").mkdir(parents=True, exist_ok=True)
    (v4 / "versions" / "folds_sha256.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return man


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    default_train = (Path(DATA_ROOT) / "v4" / "data" / "train") if DATA_ROOT \
        else (V4.parent / "data" / "train")
    default_test = (Path(DATA_ROOT) / "v4" / "data" / "test") if DATA_ROOT \
        else (V4.parent / "data" / "test")
    ap.add_argument("--train-dir", default=str(default_train))
    ap.add_argument("--test-dir", default=str(default_test))
    ap.add_argument("--out", default=str(REPORTS_DIR / "E0_data_card.json"))
    ap.add_argument("--limit", type=int, default=None, help="只解析前 N 口井（快速自检）")
    args = ap.parse_args()

    if not HAS_NUMPY:
        print("ERROR: 本脚本需要 numpy（口径层唯一硬依赖）。", file=sys.stderr)
        return 2

    import numpy as np  # noqa: PLC0415

    t0 = datetime.now(timezone.utc)
    print("== E0 run_all ==")
    print("deps:", json.dumps(describe()["versions"], ensure_ascii=False))

    # ---------------- 训练集
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    train_files = sorted(train_dir.glob("*.txt"))
    test_files = sorted(test_dir.glob("*.txt"))
    if args.limit:
        train_files = train_files[: args.limit]

    tr_records = [P.parse_well(f, with_targets=True) for f in train_files]
    card_train = P.summarise("train", tr_records)

    y_true = np.concatenate([r.targets for r in tr_records], axis=0)
    depth = np.concatenate([r.depth for r in tr_records], axis=0)
    missing = L.missing_masks(y_true)
    state = L.state_labels(y_true)

    # ---------------- 常数基线（两种分母口径）
    yhat = constant_prediction(y_true.shape[0], "placeholder")
    sc_mask = score_arrays(y_true, yhat, missing, missing_mode="mask")
    sc_drop = score_arrays(y_true, yhat, missing, missing_mode="drop")

    anchor = C.CONSTANT_BASELINE_OOF
    hit_mask = abs(sc_mask["total"] - anchor) <= 1e-4
    hit_drop = abs(sc_drop["total"] - anchor) <= 1e-4
    hit = hit_mask or hit_drop
    mode = "mask" if hit_mask else ("drop" if hit_drop else None)

    # ---------------- 测试集（无标签，只统计行数与契约）
    te_records = [P.parse_well(f, with_targets=False) for f in test_files]
    card_test = P.summarise("test", te_records)
    card_test["expected_rows"] = C.EXPECTED_N_TEST_ROWS
    card_test["rows_match_contract"] = card_test["n_rows"] == C.EXPECTED_N_TEST_ROWS

    # ---------------- 折指纹
    folds_info = {"path": None, "exists": False}
    try:
        folds_src = F.find_folds_file(V4)
        folds_info = {"path": str(folds_src), "exists": folds_src.is_file()}
    except FileNotFoundError as exc:
        folds_info = {"path": None, "exists": False, "error": str(exc)}
        folds_src = Path("/nonexistent")
    if folds_src.is_file():
        folds_info["sha256"] = sha256_file(folds_src)
        folds_info["bytes"] = folds_src.stat().st_size
        folds_info["n_folds"] = len(json.loads(folds_src.read_text(encoding="utf-8"))
                                    .get("folds", {})) or None

    # ---------------- 汇总
    card = {
        "schema_version": 1,
        "stage": "E0",
        "created_at": t0.isoformat(),
        "constants": {
            "columns": list(C.COLUMNS),
            "placeholder": C.PLACEHOLDER,
            "sentinels": list(C.SENTINELS),
            "missing_lt": C.MISSING_LT,
            "score_weights": C.SCORE_WEIGHTS,
            "delta_por": C.DELTA_POR,
            "delta_sw": C.DELTA_SW,
            "eps": C.EPS,
            "sw_dual_scale": {
                "placeholder": C.SW_PLACEHOLDER,
                "valid_range": list(C.SW_VALID_RANGE),
                "valid_scale": C.SW_VALID_SCALE,
            },
        },
        "train": {
            **card_train,
            "state_counts": {
                "missing": int((state == -1).sum()),
                "placeholder": int((state == 1).sum()),
                "valid": int((state == 0).sum()),
            },
        },
        "test": card_test,
        "folds": folds_info,
        "constant_baseline": {
            "anchor_expected": anchor,
            "tolerance": 1e-4,
            "missing_mode_mask": sc_mask,
            "missing_mode_drop": sc_drop,
            "hit": hit,
            "hit_mode": mode,
        },
        "env": describe(),
        "depth": {
            "min": float(np.nanmin(depth)),
            "max": float(np.nanmax(depth)),
        },
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = REPORTS_DIR / out.name
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- 折导出 + 契约自检（不需要 torch）
    try:
        fold_man = export_folds(V4)
    except Exception as exc:  # 折文件缺失时不阻塞数据卡产出
        fold_man = {"error": repr(exc)}
    ct = contract_selftest(V4)
    card["folds_manifest"] = fold_man
    card["contract_selftest"] = {"passed": ct["passed"]}
    out.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    mandatory = {
        "data_card_recomputable": bool(tr_records) and not args.limit,
        "row_counts_match": card_train["n_rows"] == 730_268 and card_test["n_rows"] == C.EXPECTED_N_TEST_ROWS,
        "constant_baseline_anchor_hit": hit,
        "folds_fingerprint_present": "source_sha256" in fold_man,
        "contract_selftest": ct["passed"],
        "no_torch_required": True,
    }
    gate = {
        "gate_id": "E0_gate",
        "stage": "E0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(mandatory.values()),
        "mandatory_checks": mandatory,
        "constant_baseline": {"drop": sc_drop["total"], "mask": sc_mask["total"],
                              "anchor": anchor, "hit_mode": mode},
        "state_counts": card["train"]["state_counts"],
        "folds": {k: fold_man.get(k) for k in ("source_sha256", "n_wells", "n_folds")},
    }
    (REPORTS_DIR / "E0_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "train_wells": card_train["n_wells"],
        "train_rows": card_train["n_rows"],
        "state_counts": card["train"]["state_counts"],
        "test_wells": card_test["n_wells"],
        "test_rows": card_test["n_rows"],
        "test_rows_match": card_test["rows_match_contract"],
        "const_baseline_mask": round(sc_mask["total"], 6),
        "const_baseline_drop": round(sc_drop["total"], 6),
        "anchor_hit": hit,
        "anchor_mode": mode,
        "contract_selftest": ct["passed"],
        "folds_sha256": fold_man.get("source_sha256"),
        "gate_passed": gate["passed"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    print(f"elapsed: {(datetime.now(timezone.utc) - t0).total_seconds():.1f}s")
    return 0 if (hit and ct["passed"] and gate["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
