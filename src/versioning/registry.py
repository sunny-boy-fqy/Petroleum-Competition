"""版本注册表（`versions/registry.json`）与 M5 要求的可运行版本清单。

`predict.py` 通过本模块解析可用版本，而不是硬编码版本表（审查 M5）。

约定：
  - `versions/registry.json` 记录**可运行 pipeline 版本**（type=pipeline）与 legacy 回退（B0）；
  - `versions/candidates.json` 记录**候选**（实验级）；两者分离；
  - 未训练完成的版本写 `available=false`，`predict.py` 会明确报错而不是静默输出常数。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

V4 = Path(__file__).resolve().parents[2]
REGISTRY = V4 / "versions" / "registry.json"
CANDIDATES = V4 / "versions" / "candidates.json"

DEFAULT_REGISTRY: dict[str, Any] = {
    "schema_version": 1,
    "latest": None,
    "versions": {
        "CONST": {
            "type": "baseline",
            "available": True,
            "completed": True,
            "desc": "常数基线 (POR=0.1, PERM=0.01, SW=99.9)，仅用于契约自检与分母参照",
            "entrypoint": None,
            "oof_total": 70.49073500477093,
        },
        "PD1": {
            "type": "pipeline",
            "available": False,
            "completed": False,
            "desc": "纯 DL 完整管线（E6 产出）",
            "entrypoint": "src/inference/predictor.py",
            "oof_total": None,
        },
    },
    "notes": "latest 只能由 E9 通过后的候选写入；未训练的版本必须 available=false。",
}


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else REGISTRY
    if not p.is_file():
        return json.loads(json.dumps(DEFAULT_REGISTRY))
    return json.loads(p.read_text(encoding="utf-8"))


def save_registry(data: dict[str, Any], path: str | Path | None = None) -> Path:
    p = Path(path) if path else REGISTRY
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def versions() -> dict[str, dict]:
    return load_registry()["versions"]


def available_versions() -> list[str]:
    return [k for k, v in versions().items() if v.get("available")]


def get(version: str) -> dict:
    vs = versions()
    if version not in vs:
        raise KeyError(f"unknown version {version!r}; available: {sorted(vs)}")
    return vs[version]


def latest() -> str | None:
    return load_registry().get("latest")


def list_lines() -> list[str]:
    out = [f"{'version':<10} {'available':<10} {'type':<10} OOF        description",
           "-" * 84]
    for k, v in versions().items():
        oof = "-" if v.get("oof_total") is None else f"{v['oof_total']:.6f}"
        out.append(f"{k:<10} {str(bool(v.get('available'))):<10} {v.get('type',''):<10} "
                   f"{oof:<10} {v.get('desc','')}")
    out.append("-" * 84)
    out.append(f"latest: {latest()}")
    return out


def candidates() -> list[dict]:
    if not CANDIDATES.is_file():
        return []
    return json.loads(CANDIDATES.read_text(encoding="utf-8")).get("candidates", [])
