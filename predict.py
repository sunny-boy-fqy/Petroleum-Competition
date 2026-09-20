#!/usr/bin/env python3
"""v4 推理入口（官方口径）。

    python predict.py --data_dir ./data --output result.json
    python predict.py --list-versions
    python predict.py --use-version CONST --data_dir ./data/test --output /tmp/r.json
    python predict.py --data_dir ./data/test --output /tmp/r.json --validate-only

设计约束（PLAN.md §9.2）：
  - 只读 `--data_dir`，不联网、不训练、不写 `--output` 以外的路径；
  - 推理主路径 **CPU-only**、确定性；
  - 输出的 SW 保持训练标签尺度（**禁止裁剪到 [0,1]**）；PERM 严格为正；
  - 生成后自动跑 `validate_payload` 契约校验。

当前实现状态：模型尚未训练，因此除 `CONST`（常数基线，用于契约自检）之外的版本
都会**明确报错**，不会静默输出无意义结果。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parent
sys.path.insert(0, str(V4))

from src import constants as C                      # noqa: E402
from src.data import parse as P                     # noqa: E402
from src.inference import contract as CT            # noqa: E402
from src.portability import describe                # noqa: E402
from src.versioning import registry as REG          # noqa: E402


# ---------------------------------------------------------------- 版本表
def _versions(registry_path: str | None = None) -> dict[str, dict]:
    """可运行版本表：来自 `versions/registry.json`（审查 M5：不再硬编码）。

    `--registry` 可指向别的注册表（测试与多版本并行实验用；默认读仓库内那一份）。
    """
    return REG.versions(registry_path)


DEFAULT_VERSION = "PD1"


# ---------------------------------------------------------------- 预测器
def predict_const(test_dir: Path) -> dict:
    """常数基线预测：为每口井逐行输出占位常量。"""
    result_data = []
    for rec in P.load_split(test_dir, with_targets=False):
        preds = [
            {
                "depth": round(float(d), C.DEPTH_DECIMALS),
                "POR": C.PLACEHOLDER["POR"],
                "PERM": C.PLACEHOLDER["PERM"],
                "SW": C.PLACEHOLDER["SW"],
            }
            for d in list(rec.depth)
        ]
        result_data.append({"logId": rec.well_id, "predictions": preds})
    return result_data


PREDICTORS = {"CONST": predict_const}


def _resolve_path(ck) -> Path:
    """绝对路径直用；相对路径相对**仓库根**（`v4/`）。"""
    p = Path(ck)
    if not p.is_absolute():
        p = V4 / p
    return p


def _resolve_checkpoint(info: dict) -> Path:
    """单权重版本的 checkpoint（`checkpoint` 字段）。"""
    ck = info.get("checkpoint")
    if not ck:
        raise SystemExit(f"[predict] 版本缺少 checkpoint 字段：{info.get('desc')}")
    p = _resolve_path(ck)
    if not p.is_file():
        raise SystemExit(f"[predict] checkpoint 不存在：{p}")
    return p


def _resolve_checkpoints(info: dict) -> list[Path]:
    """**折集成**版本的权重清单（`checkpoints` 列表）：E6/P2 的 5 折平均走这条路径。"""
    cks = info.get("checkpoints")
    if not cks:
        return [_resolve_checkpoint(info)]
    paths = [_resolve_path(c) for c in cks]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"[predict] 折权重缺失：{missing}")
    if not paths:
        raise SystemExit("[predict] checkpoints 为空列表")
    return paths


def predict_pd1(test_dir: Path, info: dict, batch_size: int = 65536,
                device: str = "cpu") -> tuple[list, dict]:
    """纯 DL 管线推理（**CPU 主路径**）：manifest → 权重 → 逐井解码 → 提交载荷。

    * 特征变换与训练**逐位同源**：`build_row_features` + 折内 `RowScaler`（从 manifest 读回）；
    * `missing` 由 `~isfinite(inputs)` 现场导出（与写分片缓存时的口径完全一致）；
    * 原子硬切换用 manifest 里的 `tau_atom`（若训练时选了 τ）；SW 只做 [0,100] 软裁剪。
    """
    import numpy as np

    from src.data import parse as _P                       # noqa: PLC0415
    from src.features import basic as F                    # noqa: PLC0415
    from src.inference import predictor as PR              # noqa: PLC0415
    from src.training import metrics as M                  # noqa: PLC0415

    ckpts = _resolve_checkpoints(info)
    manifests = [PR.load_manifest(c) for c in ckpts]
    ref = manifests[0]
    # 折集成的硬前提：所有折的**标尺与特征口径必须一致**，否则"平均"没有意义
    for i, man in enumerate(manifests[1:], start=1):
        if man.raw.get("row_scaler") != ref.raw.get("row_scaler"):
            raise SystemExit(f"[predict] 折 {i} 的 row_scaler 与 fold0 不一致，禁止平均")
        if list(man.raw.get("feature_names", [])) != list(ref.raw.get("feature_names", [])):
            raise SystemExit(f"[predict] 折 {i} 的 feature_names 与 fold0 不一致，禁止平均")
    taus = [man.tau_atom for man in manifests]
    if len(ckpts) > 1 and any(t != taus[0] for t in taus[1:]):
        raise SystemExit(f"[predict] 折间 tau_atom 不一致：{taus}（不允许静默取第一折）")
    models = [PR.load_model(c, man, device=device) for c, man in zip(ckpts, manifests)]
    tau = taus[0]
    per_well: dict = {}
    n_rows = 0
    for rec in _P.load_split(test_dir, with_targets=False):
        inputs = np.asarray(rec.inputs, dtype="float32")
        depth = np.asarray(rec.depth, dtype="float32")
        missing = (~np.isfinite(inputs)).astype("int8")
        X = ref.row_scaler.transform(F.build_row_features(inputs, missing, depth))
        acc = None
        for model in models:
            out = PR.predict_x(model, X, batch_size=batch_size, device=device)
            cur = np.column_stack([out["por"], out["perm_z"], out["sw"], out["q_atom"]])
            acc = cur if acc is None else acc + cur
        acc = acc / float(len(models))                    # 折平均（连续头 + 门控概率）
        out_avg = {"por": acc[:, 0], "perm_z": acc[:, 1], "sw": acc[:, 2], "q_atom": acc[:, 3:]}
        cont = M.decode_continuous(out_avg)
        pred = M.atom_gate(cont, out_avg["q_atom"], tau) if tau is not None else cont
        per_well[rec.well_id] = {"depth": depth, "pred": pred}
        n_rows += int(X.shape[0])
    payload_data = PR.build_payload(per_well, model_name="v4-PD1")
    return payload_data["resultData"], {"checkpoints": [str(c) for c in ckpts],
                                        "n_folds": len(ckpts), "n_rows": n_rows,
                                        "n_wells": len(per_well), "tau_atom": tau,
                                        "device": device,
                                        "aggregate": ("single" if len(ckpts) == 1
                                                      else "fold_mean")}


def build_payload(version: str, data_dir: Path, model_name: str | None = None,
                  registry_path: str | None = None) -> dict:
    info = _versions(registry_path)[version]
    if not info["available"]:
        raise SystemExit(
            f"[predict] version '{version}' is registered but NOT trained yet.\n"
            f"  desc: {info['desc']}\n"
            "  先完成对应阶段（见 v4/PLAN.md §七）后再运行；"
            "如需校验提交契约，请用 --use-version CONST。"
        )
    if version == "PD1":
        rows, _summary = predict_pd1(data_dir, info)
        return {
            "modelId": "",
            "modelName": model_name or f"v4-{version}",
            "version": "1.0",
            "resultData": rows,
        }
    if version not in PREDICTORS:
        raise SystemExit(
            f"[predict] 版本 {version!r} 已注册为 available 但推理入口未接线；"
            f"已接线：{sorted(PREDICTORS)} + PD1。请补 predict.py 的 PREDICTORS 分支。")
    fn = PREDICTORS[version]
    return {
        "modelId": "",
        "modelName": model_name or f"v4-{version}",
        "version": "1.0",
        "resultData": fn(data_dir),
    }


def cmd_list_versions(registry_path: str | None = None) -> int:
    for line in REG.list_lines(registry_path):
        print(line)
    avail = [k for k, v in _versions(registry_path).items() if v.get("available")]
    print(f"default: {DEFAULT_VERSION}  (可用: {avail})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 reservoir-parameter predictor")
    ap.add_argument("--list-versions", action="store_true")
    ap.add_argument("--use-version", default=None)
    ap.add_argument("--data_dir", "--data-dir", dest="data_dir", default=None,
                    help="官方参数名 --data_dir（同时兼容 --data-dir）")
    ap.add_argument("--output", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--expected-rows", type=int, default=C.EXPECTED_N_TEST_ROWS)
    ap.add_argument("--expected-wells", type=int, default=C.EXPECTED_N_TEST_WELLS)
    ap.add_argument("--validate-only", action="store_true",
                    help="只校验 --data_dir 是否可用，不写结果")
    ap.add_argument("--print-deps", action="store_true")
    ap.add_argument("--registry", default=None,
                    help="版本注册表路径（默认 versions/registry.json）")
    args = ap.parse_args(argv)

    if args.print_deps:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0

    if args.list_versions:
        return cmd_list_versions(args.registry)

    if args.data_dir is None:
        ap.error("--data_dir is required (official CLI, rules.md §6.3)")

    data_dir = Path(args.data_dir)
    # 官方 --data_dir 可能指向 data/（内含 test/）或直接指向测试井目录
    if (data_dir / "test").is_dir():
        test_dir = data_dir / "test"
    elif any(data_dir.glob("*.txt")):
        test_dir = data_dir
    else:
        ap.error(f"no *.txt wells found under {data_dir}")

    version = args.use_version or DEFAULT_VERSION
    if version not in _versions(args.registry):
        ap.error(f"unknown version '{version}'; run --list-versions")

    n_test = len(list(test_dir.glob("*.txt")))
    if n_test != args.expected_wells:
        print(f"[predict] WARNING: found {n_test} wells, contract expects "
              f"{args.expected_wells}", file=sys.stderr)

    if args.validate_only:
        print(json.dumps({"data_dir": str(data_dir), "test_dir": str(test_dir),
                          "n_wells": n_test, "version": version,
                          "available": _versions(args.registry)[version]["available"]},
                         ensure_ascii=False, indent=2))
        return 0

    if args.output is None:
        ap.error("--output is required")

    payload = build_payload(version, test_dir, args.model_name, args.registry)

    res = CT.validate_payload(payload, test_dir=test_dir,
                              expected_rows=args.expected_rows,
                              expected_wells=args.expected_wells)
    if not res.ok:
        print("[predict] CONTRACT VALIDATION FAILED:", file=sys.stderr)
        for e in res.errors[:20]:
            print("  -", e, file=sys.stderr)
        return 3

    # R2-B6：深度逐行对齐必须硬校验
    align = CT.depth_alignment_report(
        {item["logId"]: [p["depth"] for p in item["predictions"]] for item in payload["resultData"]},
        test_dir)
    bad_align = {k: v for k, v in align.items() if not v.get("ok")}
    if bad_align:
        print("[predict] DEPTH ALIGNMENT FAILED:", file=sys.stderr)
        for k, v in list(bad_align.items())[:10]:
            print(f"  - {k}: {v}", file=sys.stderr)
        return 4

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "version": version,
        "model_name": payload["modelName"],
        "output": str(out),
        "n_wells": res.stats["n_wells"],
        "n_rows": res.stats["n_rows"],
        "contract_ok": res.ok,
        "warnings": res.warnings,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
