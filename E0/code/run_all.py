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
from src.data import dataset as DS                  # noqa: E402
from src.data import labels as L                    # noqa: E402
from src.data import parse as P                     # noqa: E402
from src.inference import contract as CT            # noqa: E402
from src.portability import describe, HAS_NUMPY     # noqa: E402
from src.score import constant_prediction, score_arrays  # noqa: E402
from src.validation import folds as F               # noqa: E402


def echo(msg: str) -> None:
    print(f"[E0] {msg}", flush=True)


def contract_selftest(v4: Path) -> dict:
    """正/负样例契约自检（不需要 torch）。

    R4-B3：正样例里的 SW 必须是**真实标签尺度**（≥ SW_VALID_MIN=8.305）。
    旧正样例用了 `SW=0.4`，在"SW 单一百分数尺度、实测有效最小 8.305、SW<1 行数=0"
    的事实下本身就不该通过——它会同时训练读者的错误直觉并掩盖低值守卫。
    """
    good = {
        "modelId": "",
        "modelName": "t",
        "version": "1.0",
        "resultData": [{
            "logId": "w1",
            "predictions": [
                {"depth": 1.0, "POR": 0.1, "PERM": 0.01, "SW": 99.9},
                {"depth": 1.1, "POR": 0.12, "PERM": 1.5, "SW": 80.0},
            ],
        }],
    }
    cases: dict[str, dict] = {}
    r = CT.validate_payload(good, expected_rows=2, strict_keys=True)
    cases["good_minimal"] = {"ok": r.ok, "errors": r.errors}

    # R4-B3 反例：7/10 原子行把中位数拉到 99.9，掩盖 3 行被归一化到 [0,1] 的连续分支
    bad = json.loads(json.dumps(good))
    preds = ([{"depth": 1.0 + 0.1 * i, "POR": 0.1, "PERM": 0.01, "SW": 99.9}
              for i in range(7)]
             + [{"depth": 2.0 + 0.1 * i, "POR": 12.0, "PERM": 1.5, "SW": 0.8}
                for i in range(3)])
    bad["resultData"][0]["predictions"] = preds
    cases["bad_sw_atom_majority_hides_unit_error"] = {
        "ok": CT.validate_payload(bad, expected_rows=10).ok,
        "sw_median": 99.9,
        "expected_reason": "低值计数守卫（中位数守卫会被 7/10 原子行绕过）",
    }

    bad = json.loads(json.dumps(good))
    bad["resultData"][0]["predictions"] = [
        {"depth": 1.0, "POR": 0.1, "PERM": 0.01, "SW": 0.4},
        {"depth": 1.1, "POR": 0.12, "PERM": 1.5, "SW": 0.62},
    ]
    cases["bad_sw_all_normalized"] = {"ok": CT.validate_payload(bad, expected_rows=2).ok}

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
    neg = sorted(k for k in cases if k.startswith("bad_"))
    rejected = sorted(k for k in neg if not cases[k]["ok"])
    out = {
        "passed": passed,
        "n_cases": len(cases),
        "n_negative": len(neg),
        "n_rejected": len(rejected),
        "negative_cases": neg,
        "unrejected": sorted(set(neg) - set(rejected)),
        "cases": cases,
    }
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
    # M4：折导出写到 REPORTS_DIR（云端在 /data，持久）；不再依赖被 gitignore 的 artifacts/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "E0_folds.json").write_text(
        json.dumps({"outer": folds, "inner": inner}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 同时保留 artifacts/E0/folds.json 供本机使用（可重算，不进 git）
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


# ---------------------------------------------------------------- 证据可移植化（R3）
EPHEMERAL_PREFIXES = ("/tmp", "/var/tmp", "/dev/shm", "/run")


def _resolved(p: Path) -> Path:
    try:
        return p.resolve()
    except Exception:  # pragma: no cover - 路径异常时退回原值
        return p


def _same_path(a: Path, b: Path) -> bool:
    return _resolved(a) == _resolved(b)


def _relative_to(path: Path, base: Path) -> str | None:
    try:
        return _resolved(path).relative_to(_resolved(base)).as_posix()
    except ValueError:
        return None


def portable_path(path: Path) -> tuple[str, bool]:
    """把绝对路径转成可复现证据用的 portable 形式。

    优先级（R3 修复）：
      1. 等于 ``$V4_CACHE_ROOT`` -> 字面量 ``"$V4_CACHE_ROOT"``
      2. 位于 ``$V4_DATA_ROOT`` 内 -> 用 ``$V4_DATA_ROOT`` 前缀替换
      3. 位于仓库根内 -> 仓库相对路径
      4. 其它 -> 绝对路径

    返回 ``(portable, cache_root_portable)``；后者的语义是"portable 形式含 $ 占位符
    或是仓库相对路径"。
    """
    cache_env = os.environ.get("V4_CACHE_ROOT")
    if cache_env and _same_path(path, Path(cache_env)):
        return "$V4_CACHE_ROOT", True
    data_env = os.environ.get("V4_DATA_ROOT")
    if data_env:
        rel = _relative_to(path, Path(data_env))
        if rel is not None:
            return ("$V4_DATA_ROOT" if rel in ("", ".") else f"$V4_DATA_ROOT/{rel}"), True
    rel = _relative_to(path, V4)
    if rel is not None:
        return (rel or "."), True
    return str(_resolved(path)), False


def resolve_cache_root(arg: str | None) -> Path:
    """解析 --cache-root：相对路径按**仓库根**解析，因此 --cache-root .v4cache 稳定可用。"""
    if arg:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = V4 / p
        return p
    return Path(os.environ.get("V4_CACHE_ROOT", str(V4 / "cache")))


def warn_if_ephemeral(cache_root: Path) -> bool:
    """缓存落在 /tmp 等易失位置时大声告警（不 hard-fail，本机开发允许）。"""
    sp = str(_resolved(cache_root))
    ephemeral = any(sp == pre or sp.startswith(pre + os.sep)
                    for pre in EPHEMERAL_PREFIXES)
    if not ephemeral or os.environ.get("V4_ALLOW_TMP_CACHE") == "1":
        return False
    bar = "!" * 78
    print(bar, file=sys.stderr)
    print(f"WARN: 分片缓存根目录位于临时/易失位置：{sp}", file=sys.stderr)
    print("WARN: 该路径在任务/重启后不保留，E0_data_card.json 的缓存证据将不可复现；",
          file=sys.stderr)
    print("WARN: 云端请用 $V4_CACHE_ROOT（/data/v4/cache），本机开发可设 "
          "V4_ALLOW_TMP_CACHE=1 关闭本告警。", file=sys.stderr)
    print(bar, file=sys.stderr)
    return True


def evaluate_disk_budget(data_root: str | None = None,
                         reports_dirs: tuple[Path, ...] | None = None
                         ) -> tuple[bool, str | None, str | None]:
    """从 E0_disk_budget.json 求 cloud gate 的 disk_budget_ok。

    R3 修复：必须读 **data root**（$V4_DATA_ROOT=/data 的 30 GB 配额）的 level，
    而不是 ROOT 文件系统的 level。

    取值顺序：
      1. `primary_path` 与 data root 相同 -> 用顶层 `level`（权威）
      2. 否则若存在 `worst_level` -> 用 `worst_level`（所有挂载点里最差的）
      3. 否则退回顶层 `level`（兼容没有 primary_path 的旧报告）

    返回 ``(ok, level, primary_path)``；文件缺失 -> ``(False, None, None)``。
    """
    if data_root is None:
        data_root = DATA_ROOT
    if reports_dirs is None:
        reports_dirs = (REPORTS_DIR, V4 / "reports")
    for cand in reports_dirs:
        f = Path(cand) / "E0_disk_budget.json"
        if not f.is_file():
            continue
        try:
            db = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        primary_path = db.get("primary_path") or db.get("path")
        if primary_path is not None and data_root and \
                _same_path(Path(primary_path), Path(data_root)):
            level = db.get("level")
        elif "worst_level" in db:
            level = db.get("worst_level")
        else:
            level = db.get("level")
        return level == "ok", level, primary_path
    return False, None, None


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
    ap.add_argument("--with-cache", action="store_true",
                    help="同时构建 cache/raw|labels 分片与 cache/manifest.json（E1 的输入，"
                         "审查 B6）")
    ap.add_argument("--cache-root", default=None,
                    help="缓存根目录；默认 $V4_CACHE_ROOT 或 <v4>/cache。相对路径按仓库根解析"
                         "（--cache-root .v4cache 会存成 repo 相对证据）")
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

    # ---------------- 目标分布统计（E0-R2：修正 SW 尺度错误的前提）
    ph_mask = L.placeholder_flags(y_true)
    miss_any = missing.any(axis=1)
    valid_mask = (~miss_any) & (~ph_mask)
    target_stats: dict[str, dict] = {}
    for j, tname in enumerate(C.TARGET_COLUMNS):
        col_all = y_true[~missing[:, j], j]
        col_val = y_true[valid_mask, j]
        target_stats[tname] = {
            "non_missing": {
                "n": int(col_all.size),
                "min": float(col_all.min()), "p01": float(np.percentile(col_all, 1)),
                "median": float(np.median(col_all)),
                "p99": float(np.percentile(col_all, 99)), "max": float(col_all.max()),
            },
            "valid_rows_only": {
                "n": int(col_val.size),
                "min": float(col_val.min()), "p01": float(np.percentile(col_val, 1)),
                "median": float(np.median(col_val)),
                "p99": float(np.percentile(col_val, 99)), "max": float(col_val.max()),
                "n_lt_1": int((col_val < 1).sum()),
            },
        }
    # 总分一致性：total 必须等于 100*Σ w_t·acc_t
    total_check = 100.0 * (
        C.SCORE_WEIGHTS["POR"] * sc_drop["acc_por"]
        + C.SCORE_WEIGHTS["PERM"] * sc_drop["acc_perm"]
        + C.SCORE_WEIGHTS["SW"] * sc_drop["acc_sw"]
    )
    total_consistent = abs(total_check - sc_drop["total"]) < 1e-6

    # ---------------- 测试集（无标签，只统计行数与契约）
    te_records = [P.parse_well(f, with_targets=False) for f in test_files]
    card_test = P.summarise("test", te_records)
    card_test["expected_rows"] = C.EXPECTED_N_TEST_ROWS
    card_test["rows_match_contract"] = card_test["n_rows"] == C.EXPECTED_N_TEST_ROWS

    # ---------------- 输入列回归：13 列输入、与任何目标不相交、90 井一致
    leak_report = {"checked_wells": 0, "violations": [], "n_curves": C.N_INPUT}
    for rec in tr_records[:12] + te_records[:6]:
        leak_report["checked_wells"] += 1
        n_in = rec.inputs.shape[1] if HAS_NUMPY else len(rec.inputs[0])
        if n_in != C.N_INPUT:
            leak_report["violations"].append({"well": rec.well_id, "reason": f"n_inputs={n_in}"})
            continue
        if rec.targets is None:
            continue
        for j, tname in enumerate(C.TARGET_COLUMNS):
            m = ~L.missing_masks(rec.targets)[:, j]
            if m.sum() < 50:
                continue
            for k in range(C.N_INPUT):
                same = float(np.isclose(rec.inputs[m, k], rec.targets[m, j], atol=1e-9).mean())
                if same > 0.999:
                    leak_report["violations"].append(
                        {"well": rec.well_id, "reason": f"{tname} == inputs[:,{k}]"})
    # R2-H1：抽样之外，追加**全量 90 井**回归（同一逻辑，避免"抽样过了 Gate 就过"）
    full_leak = {"checked": 0, "violations": []}
    for split_, wt_ in (("train", True), ("test", False)):
        d_ = train_dir if split_ == "train" else test_dir
        for f_ in sorted(Path(d_).glob("*.txt")):
            rec_ = P.parse_well(f_, with_targets=wt_)
            full_leak["checked"] += 1
            n_in_ = rec_.inputs.shape[1]
            if n_in_ != C.N_INPUT:
                full_leak["violations"].append({"well": rec_.well_id, "reason": f"n_inputs={n_in_}"})
                continue
            if not wt_ or rec_.targets is None:
                continue
            for j_, tname_ in enumerate(C.TARGET_COLUMNS):
                m_ = ~L.missing_masks(rec_.targets)[:, j_]
                if int(m_.sum()) < 20:
                    continue
                for k_ in range(C.N_INPUT):
                    if float(np.isclose(rec_.inputs[m_, k_], rec_.targets[m_, j_],
                                        atol=1e-9).mean()) > 0.999:
                        full_leak["violations"].append(
                            {"well": rec_.well_id, "reason": f"{tname_} == inputs[:,{k_}]"})
    leak_report["full_90_wells"] = {
        "checked": full_leak["checked"],
        "violations": len(full_leak["violations"]),
        "passed": not full_leak["violations"],
        "examples": full_leak["violations"][:5],
    }
    leak_report["passed"] = (not leak_report["violations"]) and leak_report["full_90_wells"]["passed"]
    leak_report["scope"] = ("全量 90 井（train+test）逐井检查 13 列输入且输入列与任何目标列不完全相等；"
                            "等价于 tools/check_data_leak.py")


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
        _raw = json.loads(folds_src.read_text(encoding="utf-8"))
        folds_info["n_folds"] = int(_raw.get("n_folds", 0)) or None
        folds_info["n_wells"] = len(_raw.get("well_list", [])) or None
        folds_info["seed"] = _raw.get("seed")

    # ---------------- 分片缓存（E1 的输入；审查 B6：此前无任何 P 负责生成）
    cache_info: dict = {"built": False}
    if args.with_cache:
        cache_root = resolve_cache_root(args.cache_root)
        cache_root_portable, cache_root_is_portable = portable_path(cache_root)
        warn_if_ephemeral(cache_root)
        echo(f"构建分片缓存 -> {cache_root}")
        man = DS.build_cache(train_dir if not args.limit else train_dir,
                             test_dir, cache_root, limit=args.limit, verbose=False)
        # 校验：输入列数（R4-M6：**两个 split 都查**，此前只查 train 的 80 个 shard）、
        # 目标不进入输入
        viol = []
        checked = {"train": 0, "test": 0}
        for split_ in ("train", "test"):
            for w in DS.iter_wells(cache_root, split_):
                sh = DS.read_well_shard(cache_root, w, split_)
                checked[split_] += 1
                if sh["inputs"].shape[1] != C.N_INPUT:
                    viol.append({"split": split_, "well": w,
                                 "reason": f"n_inputs={sh['inputs'].shape[1]}"})
        manifest_path = cache_root / "manifest.json"
        cache_info = {
            "built": True,
            # R3 修复：cache_root 必须是**可复现证据**（$V4_CACHE_ROOT / repo 相对），
            # 不能是 /tmp/v4cache_final 这类本机临时路径；绝对路径另存 cache_root_abs。
            "cache_root": cache_root_portable,
            "cache_root_abs": str(_resolved(cache_root)),
            "cache_root_portable": cache_root_is_portable,
            "counts": man["counts"],
            "manifest": str(manifest_path),
            "manifest_portable": portable_path(manifest_path)[0],
            "mb": round(DS.shard_bytes(cache_root) / 1e6, 2),
            "input_cols_ok": not viol,
            "input_cols_checked": checked,
            "violations": viol,
        }
        echo(f"缓存完成：{man['counts']['train_wells']}训练/{man['counts']['test_wells']}测试井，"
             f"{cache_info['mb']:.1f} MB")

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
            "sw_scale": {
                "placeholder": C.SW_PLACEHOLDER,
                "label_range": list(C.SW_LABEL_RANGE),
                "small_branch_enabled": C.SW_SMALL_BRANCH,
                "note": "E0-R2 修正：SW 为单一标签尺度（实测有效值 8.305–99.9），非双尺度",
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
        "target_stats": target_stats,
        "sw_scale": L.sw_scale_report(y_true[valid_mask, 2]),
        "score_consistency": {
            "total_from_weights": total_check,
            "total_reported": sc_drop["total"],
            "consistent": total_consistent,
            "missing_mode_frozen": C.SCORE_MISSING_MODE,
        },
        "input_leak_regression": leak_report,
        "shard_cache": cache_info,
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
    # ---------------- 评分口径对照（M6：产出 E0_score_check.json）
    score_check = {
        "anchors": {"expected_constant_total": anchor,
                    "abs_tolerance": 1e-4,
                    "frozen_missing_mode": C.SCORE_MISSING_MODE},
        "constant_baseline": {
            "drop": {k: sc_drop[k] for k in ("acc_por", "acc_perm", "acc_sw", "total")},
            "mask": {k: sc_mask[k] for k in ("acc_por", "acc_perm", "acc_sw", "total")},
        },
        "hit": hit, "hit_mode": mode,
        "total_consistency": {"from_weights": total_check,
                              "reported": sc_drop["total"],
                              "consistent": total_consistent},
        "formula": {
            "acc_por": "mean max(0, 1 - |yhat-y|/(0.08*(|y|+eps)))",
            "acc_perm": "mean max(0, 1 - |log10(max(yhat/y, eps))|)",
            "acc_sw": "mean max(0, 1 - |yhat-y|/(0.05*(|y|+eps)))",
            "total": "100*(0.30*acc_por + 0.35*acc_perm + 0.35*acc_sw)",
            "eps": C.EPS,
        },
        "recompute_command": "python3 v4/E0/code/run_all.py",
    }
    (REPORTS_DIR / "E0_score_check.json").write_text(
        json.dumps(score_check, ensure_ascii=False, indent=2), encoding="utf-8")

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
        # E0-R2 新增（审查 B1/B2/B5/M3）
        "missing_mode_is_drop": C.SCORE_MISSING_MODE == "drop",
        "score_total_consistent": total_consistent,
        "input_no_label_leak": leak_report["passed"],
        "target_scale_reported": bool(target_stats) and "valid_rows_only" in target_stats["SW"],
    }
    # R2-B3：E0 local Gate 必须包含缓存两项（E1/P0 依赖它）
    mandatory["shard_cache_built"] = bool(cache_info.get("built"))
    mandatory["shard_cache_input_cols_ok"] = bool(cache_info.get("input_cols_ok"))
    # R4-H1：把 prereg 里声明的 `contract_ok` 作为 `contract_selftest` 的**显式别名**
    # 写进 report。此前 prereg 有 13 项、report 只有 12 项，用同一个 `aggregate_gate`
    # 复算会 `mandatory_failures=['contract_ok']` → 预注册与实物报告对不上。
    mandatory["contract_ok"] = ct["passed"]
    # R4-H1：absolute 门槛 `abs_tolerance` 需要一个**可复算**的偏差指标。
    # 这里给两个都客观、可重算的数：常数基线锚点偏差 与 总分恒等式偏差。
    abs_diff = abs(float(sc_drop["total"]) - float(anchor))
    score_total_abs_diff = abs(float(total_check) - float(sc_drop["total"]))
    gate = {
        "gate_id": "E0_local_contract_gate",
        "stage": "E0",
        "scope": "local_contract_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(mandatory.values()),
        "mandatory_checks": mandatory,
        "constant_baseline": {"drop": sc_drop["total"], "mask": sc_mask["total"],
                              "anchor": anchor, "hit_mode": mode},
        "state_counts": card["train"]["state_counts"],
        "folds": {k: fold_man.get(k) for k in ("source_sha256", "n_wells", "n_folds")},
        # 供 aggregate_gate 复算的指标字段（R4-H1）
        "abs_diff": abs_diff,
        "score_total_abs_diff": score_total_abs_diff,
    }
    # 注意：gate JSON 的落盘推迟到 prereg 之后 —— 先把 `aggregate_gate(prereg, gate)`
    # 的复算结论写进同一份报告，避免"预注册与实物报告对不上"再次发生（R4-H1）。

    # ---------------- 云端 Gate（B4：与本地契约 Gate 分离）
    env_json = None
    for cand in (REPORTS_DIR / "E0_env.json", V4 / "reports" / "E0_env.json"):
        if cand.is_file():
            try:
                env_json = json.loads(cand.read_text(encoding="utf-8"))
                break
            except Exception:
                env_json = None
    env_ok = bool(env_json and env_json.get("passed") and env_json.get("hard_failures") == 0)
    # R3 修复：disk_budget_ok 必须读**data root**（$V4_DATA_ROOT=/data 的 30 GB 配额）的 level，
    # 而不是 ROOT 文件系统的 level。
    disk_ok, disk_level, disk_primary_path = evaluate_disk_budget(DATA_ROOT)
    cloud_checks = {
        "env_hard_checks_passed": env_ok,
        "disk_budget_ok": disk_ok,
    }
    cloud_gate = {
        "gate_id": "E0_cloud_gate",
        "stage": "E0",
        "scope": "cloud_env_and_disk",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(cloud_checks.values()),
        "status": "passed" if all(cloud_checks.values()) else "blocked_pending_cloud_run",
        "mandatory_checks": cloud_checks,
        # 证据自描述：说明 disk_budget_ok 到底测的是哪个文件系统
        "disk_budget_primary_path": disk_primary_path,
        "disk_budget_level": disk_level,
        "disk_budget_expected_data_root": DATA_ROOT,
        "how_to_satisfy": (
            "在平台训练任务执行 `bash /code/workspace/v4/run_train.sh --mode env`，"
            "产出 $V4_REPORTS_DIR/E0_env.json 与 E0_disk_budget.json，然后重跑本脚本。"
        ),
        "note": ("E0 阶段只有在本地契约 Gate 与云端 Gate **都通过**后才算完成；"
                 "仅本地通过时 status.json 记为 in_progress。"),
    }
    (REPORTS_DIR / "E0_cloud_gate.json").write_text(
        json.dumps(cloud_gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- Gate 预注册（E0 为口径层，事后补记 + supersede 说明）
    prereg = {
        "gate_id": "E0_local_contract_gate",
        "stage": "E0",
        "p_stage": "P0-P3",
        "gate_type": "boolean",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supersede_note": (
            "E0 是口径层而非模型实验，其判定阈值不是可调超参（数据计数、评分锚点、折指纹、"
            "契约自检都是客观等式），因此本文件在实现完成后补记。若未来 E0 需要修订判定项，"
            "必须新建 E0_gate_r2 预注册，不得原地修改本文件。"
        ),
        "primary_metric": "data_card_recomputable",
        "primary_threshold_key": "abs_tolerance",
        "thresholds": {"abs_tolerance": 1e-4},
        "baseline_version": "CONST",
        "baseline_artifact": "reports/E0_data_card.json",
        "baseline_manifest_sha256": "",   # E0 无上游 manifest；填占位会导致校验失败，这里留空并在 notes 说明
        "alpha": 0.05,
        "multiplicity": "none",
        "candidate_budget": 1,
        "bootstrap_iters": 1000,
        "bootstrap_unit": "well_row_weighted_cluster",
        "pilot_std": None,
        "mde_units": 80,
        "min_detectable_effect": None,
        "mandatory_checks": ["data_card_recomputable", "row_counts_match",
                             "constant_baseline_anchor_hit", "folds_fingerprint_present",
                             "contract_selftest", "contract_ok", "no_torch_required",
                             "missing_mode_is_drop", "score_total_consistent",
                             "input_no_label_leak", "target_scale_reported",
                             "shard_cache_built", "shard_cache_input_cols_ok"],
        "decisions_locked": ["data_parsing_by_header_name", "score_missing_mode=drop",
                             "folds=v1_well_folds.json", "placeholder_kept"],
        # E0 是口径层：不训练、无 checkpoint、无原子 precision 报告需求；
        # 显式声明豁免并给出理由，而不是塞入假检查。
        "mandatory_exempt": {
            "atomic_precision_reported": "E0 不训练模型，无原子门",
            "disk_budget_ok": "磁盘门禁在 E0_cloud_gate（需云端实测）",
            "training_time_log_valid": "E0 无训练任务",
            "checkpoint_resumable": "E0 无 checkpoint",
            "no_label_leak": "等价检查为 input_no_label_leak + full_90_wells",
        },
        "planned_task_training_h": 0.05,
        "notes": ("不训练模型；只冻结数据/评分/折/提交契约。E0 为确定性检查"
                  "（gate_type=boolean），不依赖 delta/CI。baseline_manifest_sha256 留空："
                  "E0 没有上游 manifest，其参照是公开锚点 70.4907。"),
    }
    (REPORTS_DIR / "E0_gate_prereg.json").write_text(
        json.dumps(prereg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- R4-H1：prereg × report 必须能用同一个校验器复算通过
    from src.validation.gates import aggregate_gate  # noqa: PLC0415

    gate_result = {
        "checks": dict(mandatory),
        "abs_diff": abs_diff,
        "score_total_abs_diff": score_total_abs_diff,
    }
    # report 里也提供 `checks` 键，使任何只给 report 的调用方都能直接复算
    gate["checks"] = dict(mandatory)
    recompute = aggregate_gate(prereg, gate_result)
    gate["prereg_recompute"] = recompute
    if not recompute["passed"]:
        echo("!! [FATAL] E0_gate_prereg × E0_local_contract_gate 复算未通过："
             f"prereg_errors={recompute['prereg_errors']} "
             f"mandatory_failures={recompute['mandatory_failures']} "
             f"absolute={recompute['details'].get('absolute_checks')}")
    # gate JSON 落盘（含复算结论）；E0_gate.json 保留为同一内容的旧别名
    for _name in ("E0_local_contract_gate.json", "E0_gate.json"):
        (REPORTS_DIR / _name).write_text(
            json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (REPORTS_DIR / "E0_gate_result.json").write_text(
        json.dumps({"gate_result_input": gate_result,
                    "aggregate_gate": recompute}, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
        "per_target_acc": {k: round(sc_drop[k], 7) for k in ("acc_por", "acc_perm", "acc_sw")},
        "score_total_consistent": total_consistent,
        "input_no_label_leak": leak_report["passed"],
        "sw_valid_min": target_stats["SW"]["valid_rows_only"]["min"],
        "sw_valid_median": target_stats["SW"]["valid_rows_only"]["median"],
        "shard_cache": ({"built": cache_info.get("built"),
                         "mb": cache_info.get("mb", 0.0),
                         "input_cols_ok": bool(cache_info.get("input_cols_ok")),
                         "violations": cache_info.get("violations", [])}
                        if args.with_cache else {"built": False}),
        "gate_passed": gate["passed"],
        "prereg_recompute_passed": bool(recompute["passed"]),
        "prereg_recompute_abs_checks": recompute["details"].get("absolute_checks"),
        "cloud_gate_passed": cloud_gate["passed"],
        "cloud_gate_status": cloud_gate["status"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    print(f"elapsed: {(datetime.now(timezone.utc) - t0).total_seconds():.1f}s")
    return 0 if (hit and ct["passed"] and gate["passed"] and recompute["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
