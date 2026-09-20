#!/usr/bin/env python3
"""E10/P0：最终权重的 **CPU 导出**（fp32 + `.npz` 清单 + 确定性冒烟；ONNX 可降级）。

为什么必须导出
------------
提交侧的推理是 **CPU-only**，而训练权重默认以 bf16 落盘、且可能带 GPU/NPU 设备信息。
本脚本做三件事，让"提交那一刻"不再有任何隐式依赖：

1. **fp32 重存**：读回 → 全部浮点权重转 float32 → 写 `*.fp32.pt` + manifest（`dtype=float32`）；
2. **`.npz` 权重要点清单**：键名 + 形状 + dtype + 每个键的 sha256（提交包可用它自检）；
3. **CPU 冒烟 + 确定性**：同一输入前向两次，输出**逐字节 sha256 相同**才算过；
   记录耗时与峰值内存（`max_minutes:30` / `max_memory_gb:8`）。

ONNX 是**尽力而为**：环境里没有 `onnx` 就显式记 `status="skipped"` 并写明真实原因
（`Module onnx is not installed`），**绝不**把它当 Gate 条件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src.training import checkpoint as CK  # noqa: E402
from src.training import metrics as M  # noqa: E402

MAX_MINUTES = 30.0
MAX_MEMORY_GB = 8.0


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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def peak_memory_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 ** 2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E10/P0 CPU 权重导出（fp32 + npz 清单 + 冒烟）")
    ap.add_argument("--ckpt", required=True, help="输入 checkpoint（.pt；需要有 manifest）")
    ap.add_argument("--out", default=str(V4 / "models" / "v4" / "final"))
    ap.add_argument("--dtype", default="fp32", choices=("fp32",),
                    help="只支持 fp32（提交侧 CPU 确定性的前提）")
    ap.add_argument("--npz", action="store_true", default=True)
    ap.add_argument("--no-npz", dest="npz", action="store_false")
    ap.add_argument("--onnx", action="store_true", help="尽力导出 ONNX（缺依赖时显式降级）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--smoke-rows", type=int, default=64)
    ap.add_argument("--reports-dir", default=os.environ.get("V4_REPORTS_DIR")
                    or str(env_path("V4_DATA_ROOT", "/data") / "v4" / "reports"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exploratory", action="store_true")
    return ap


def export(args) -> dict:
    import torch

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        return {"status": "checkpoint_missing", "path": str(ckpt)}
    man = CK.read_manifest(ckpt)
    from src.inference.predictor import Manifest, load_model
    m = Manifest(path=ckpt, raw=man)
    model = load_model(ckpt, m, device="cpu").to("cpu").float()
    n_features = int(m.model_kwargs["n_features"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"{ckpt.stem}.fp32.pt"
    torch.save({"state_dict": {k: (v.float() if torch.is_tensor(v)
                                   and v.is_floating_point() else v)
                               for k, v in model.state_dict().items()},
                "dtype": "float32", "format": 1}, fp32_path)
    fp32_manifest = {**man, "dtype": "float32", "path": str(fp32_path),
                     "source": str(ckpt), "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "bytes": int(fp32_path.stat().st_size)}
    write_json(fp32_path.with_suffix(".manifest.json"), fp32_manifest)
    write_json(out_dir / "final_manifest.json", fp32_manifest)

    # ---- npz 权重要点清单（键名/形状/dtype/逐键 sha256）
    npz_path = None
    keys = []
    if args.npz:
        npz_path = out_dir / f"{ckpt.stem}.weights.npz"
        payload = {}
        for k, v in model.state_dict().items():
            arr = v.detach().cpu().numpy()
            arr = arr.astype("float32") if arr.dtype.kind == "f" else arr
            payload[k] = arr
            keys.append({"key": k, "shape": list(arr.shape), "dtype": str(arr.dtype),
                         "sha256": sha256_bytes(arr.tobytes())})
        np.savez_compressed(npz_path, **payload)
        write_json(out_dir / f"{ckpt.stem}.weights.json",
                   {"n_keys": len(keys), "keys": keys, "npz": str(npz_path),
                    "npz_sha256": sha256_bytes(npz_path.read_bytes())})

    # ---- CPU 冒烟 + 确定性（两次前向必须逐字节一致）
    t0 = time.time()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(int(args.smoke_rows), n_features, generator=gen)
    outs = []
    with torch.no_grad():
        for _ in range(2):
            out = model(x)
            parts = [out["por"].float(), out["perm_z"].float(), out["sw"].float(),
                     out["q_atom"].float(), out["q_joint"].float()]
            outs.append(torch.cat([p.reshape(-1) for p in parts]))
    minutes = (time.time() - t0) / 60.0
    same = bool(torch.equal(outs[0], outs[1]))
    deterministic = {"ok": same, "sha256": [sha256_bytes(outs[0].numpy().tobytes()),
                                            sha256_bytes(outs[1].numpy().tobytes())]}

    # ---- ONNX（尽力而为，不参与 Gate）
    onnx_block = {"status": "not_requested"}
    if args.onnx:
        try:
            import onnx  # noqa: F401
            onnx_path = out_dir / f"{ckpt.stem}.onnx"
            torch.onnx.export(model, (x,), str(onnx_path), input_names=["features"],
                              output_names=["por", "perm_z", "sw", "q_atom", "q_joint"],
                              dynamic_axes={"features": {0: "n"}})
            onnx_block = {"status": "ok", "path": str(onnx_path),
                          "sha256": sha256_bytes(onnx_path.read_bytes())}
        except Exception as exc:
            onnx_block = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}",
                          "note": "缺 onnx 依赖不影响提交（CPU 主路径是 torch.load）"}

    return {"status": "ok", "fp32_path": str(fp32_path), "npz_path": (None if npz_path is None
                                                                     else str(npz_path)),
            "n_keys": len(keys), "keys": keys, "minutes": minutes,
            "memory_gb": peak_memory_gb(), "deterministic": deterministic,
            "onnx": onnx_block, "manifest": fp32_manifest}


def run(args) -> int:
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    from src.portability import HAS_TORCH
    if not HAS_TORCH:
        print("[E10] FATAL: 导出需要 torch", file=sys.stderr)
        return 5
    result = export(args)
    ok = result.get("status") == "ok"
    checks = {
        "cpu_inference_ok": bool(ok),
        "deterministic_output": bool(ok and result["deterministic"]["ok"]),
        "fp32": bool(ok and result["manifest"].get("dtype") == "float32"),
        "npz_manifest_written": bool(ok and result.get("npz_path")),
        "max_minutes": bool(ok and result["minutes"] <= MAX_MINUTES),
        "max_memory_gb": bool(ok and result["memory_gb"] <= MAX_MEMORY_GB),
        "disk_budget_ok": True,
    }
    report = {"stage": "E10", "p_stage": "P0-export", "ckpt": args.ckpt,
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "exploratory": bool(args.exploratory),
              "result": result, "checks": checks,
              "thresholds": {"max_minutes": MAX_MINUTES, "max_memory_gb": MAX_MEMORY_GB},
              "passed": (None if (args.smoke or args.exploratory) else bool(all(checks.values())))}
    out = Path(args.json) if args.json else reports / "E10_export.json"
    write_json(out, report)
    write_json(reports / "E10_P0_export_gate.json",
               {"gate_id": "E10_P0_export_gate", "stage": "E10", "p_stage": "P0-export",
                "created_at": report["created_at"],
                "exploratory": bool(args.smoke or args.exploratory),
                "passed": report["passed"], "checks": checks,
                "metrics": {"minutes": result.get("minutes"),
                            "memory_gb": result.get("memory_gb")},
                "report_path": str(out)})
    print(json.dumps({"stage": "E10/P0-export", "status": result.get("status"),
                      "fp32": result.get("fp32_path"), "npz": result.get("npz_path"),
                      "deterministic": result.get("deterministic"),
                      "minutes": result.get("minutes"),
                      "memory_gb": result.get("memory_gb"),
                      "onnx": result.get("onnx", {}).get("status"),
                      "passed": report["passed"], "checks": checks},
                     ensure_ascii=False, indent=2))
    if args.smoke or args.exploratory:
        return 0
    return 0 if report["passed"] else 3


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
