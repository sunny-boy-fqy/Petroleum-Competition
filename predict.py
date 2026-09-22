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
import os
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


def _load_decode_config(info: dict):
    """从 registry/manifest、env、$V4_REPORTS_DIR 或默认路径加载 E7 冻结解码配置。

    审查 H4：E7 现在会把 `decode_v1.json` 镜像到 `$V4_REPORTS_DIR`（随 mirror 持久化）。
    多任务拆分时先查 reports，避免下一任务静默回退到“无解码配置”。
    """
    from src.inference import decode as DEC
    candidates: list[str] = []
    if info.get("decode_config"):
        candidates.append(str(info["decode_config"]))
    if os.environ.get("V4_DECODE_CONFIG"):
        candidates.append(str(os.environ["V4_DECODE_CONFIG"]))
    rep = os.environ.get("V4_REPORTS_DIR")
    if rep:
        candidates.append(str(Path(rep) / "decode_v1.json"))
    candidates.append(str(V4 / "versions" / "configs" / "decode_v1.json"))
    for raw in candidates:
        p = _resolve_path(raw)
        if p.is_file():
            return DEC.load_decode_config(p)
    return None


def _effective_tau(manifest_tau, decode_cfg):
    """合并 manifest 原子阈值与 E7 expected-value 解码阈值。

    仅当 decode_cfg.expected_value[t] 为 True 时用 decode_cfg.tau[t] 覆盖对应目标；
    其余目标仍使用 training/inner-OOF 选出的 manifest tau。
    """
    import numpy as np
    if manifest_tau is None and decode_cfg is None:
        return None
    base = None
    if manifest_tau is not None:
        arr = np.asarray(manifest_tau, dtype="float64").reshape(-1)
        base = (np.repeat(arr, 3) if arr.size == 1 else arr).astype("float64").tolist()
    if base is None:
        base = [0.5, 0.5, 0.5]
    if decode_cfg is None or not decode_cfg.tau:
        return base
    for i, t in enumerate(C.TARGETS):
        if bool(decode_cfg.expected_value.get(t, False)):
            base[i] = float(decode_cfg.tau.get(t, base[i]))
    return base


def _fused_tau(manifest_taus, decode_cfg):
    """把多折 tau 融合成一个逐目标阈值；任一折缺失语义时返回 None。"""
    import numpy as np
    eff: list = []
    for t in manifest_taus:
        v = _effective_tau(t, decode_cfg)
        if v is None:
            return None
        eff.append(np.asarray(v, dtype="float64").reshape(3))
    if not eff:
        return None
    return np.mean(np.vstack(eff), axis=0).astype("float64").tolist()


def predict_pd1(test_dir: Path, info: dict, batch_size: int = 65536,
                device: str = "cpu") -> tuple[list, dict]:
    """纯 DL 管线推理（**CPU 主路径**）：manifest → 权重 → 逐井解码 → 提交载荷。

    折集成语义（审查 H3 修复，遵循项目自身纪律）
    ------------------------------------------
    每个折权重仍用自己的 row_scaler 做标准化；但**先融合连续头与 q_atom**，
    再统一做**一次**原子硬切换。这样折间分歧不会产生“一半原子值、一半连续值”的插值，
    与 `src/ensemble/blend.py` / `E8/code/ensemble.py` 的“切换必须在融合之后”一致。
    统一 tau 取各折 `tau_atom`（经 E7 decode 配置覆盖后）的逐目标均值。
    """
    import numpy as np

    from src.data import parse as _P                       # noqa: PLC0415
    from src.inference import predictor as PR              # noqa: PLC0415
    from src.training import metrics as M                  # noqa: PLC0415

    ckpts = _resolve_checkpoints(info)
    manifests = [PR.load_manifest(c) for c in ckpts]
    ref = manifests[0]

    def _sig(man):
        return (str(man.arch).lower(), tuple(man.feature_names),
                json.dumps(man.arch_kwargs, sort_keys=True, default=str),
                json.dumps(man.feature_spec.as_dict() if man.feature_spec else None,
                           sort_keys=True, default=str))

    for i, man in enumerate(manifests):
        if not PR.cpu_inference_supported(man):
            raise SystemExit(f"[predict] 折 {i} 的架构 {man.arch!r} 尚未接线 CPU 推理")
    for i, man in enumerate(manifests[1:], start=1):
        if _sig(man) != _sig(ref):
            raise SystemExit(f"[predict] 折 {i} 的模型/特征配置与 fold0 不一致，禁止平均")
    taus = [man.tau_atom for man in manifests]
    has_tau = [t is not None for t in taus]
    if has_tau and any(h != has_tau[0] for h in has_tau):
        raise SystemExit(f"[predict] 折间 tau_atom 存在/缺失状态不一致：{taus}")
    models = [PR.load_model(c, man, device=device) for c, man in zip(ckpts, manifests)]
    decode_cfg = _load_decode_config(info)
    from src.inference import decode as DEC
    tau_fused = _fused_tau(taus, decode_cfg)

    per_well: dict = {}
    n_rows = 0
    n_folds = len(models)
    for rec in _P.load_split(test_dir, with_targets=False):
        inputs = np.asarray(rec.inputs, dtype="float32")
        depth = np.asarray(rec.depth, dtype="float32")
        missing = (~np.isfinite(inputs)).astype("int8")
        cont_sum = None
        q_sum = None
        for model, man in zip(models, manifests):
            X_raw = PR.build_inference_features(inputs, missing, depth, man)
            X = man.row_scaler.transform(X_raw)
            out = PR.predict_manifest(model, man, X, batch_size=batch_size, device=device)
            cont = np.asarray(M.decode_continuous(out), dtype="float64")
            q_atom = np.asarray(out["q_atom"], dtype="float64")
            cont_sum = cont if cont_sum is None else cont_sum + cont
            q_sum = q_atom if q_sum is None else q_sum + q_atom
        cont_fused = cont_sum / float(n_folds)
        q_fused = q_sum / float(n_folds)
        if decode_cfg is not None:
            cont_fused = DEC.apply_decode_config(cont_fused, decode_cfg, q_atom=q_fused)
        pred = (M.atom_gate(cont_fused, q_fused, tau_fused)
                if tau_fused is not None else cont_fused)
        per_well[rec.well_id] = {"depth": depth,
                                 "pred": np.asarray(pred, dtype="float64")}
        n_rows += int(inputs.shape[0])
    payload_data = PR.build_payload(per_well, model_name="v4-PD1")
    return payload_data["resultData"], {"checkpoints": [str(c) for c in ckpts],
                                        "n_folds": len(ckpts), "n_rows": n_rows,
                                        "n_wells": len(per_well),
                                        "tau_atom": taus[0], "taus": taus,
                                        "tau_fused": tau_fused,
                                        "per_fold_scaler": True,
                                        "decode_config": (None if decode_cfg is None
                                                          else decode_cfg.as_dict()),
                                        "device": device,
                                        "aggregate": ("single" if len(ckpts) == 1
                                                      else "fuse_then_gate")}


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
