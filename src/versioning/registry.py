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


def candidates(path: str | Path | None = None) -> list[dict]:
    return load_candidates(path).get("candidates", [])


# ---------------------------------------------------------------- 候选写回（E9/E10 依赖）
CANDIDATE_STATUSES: tuple[str, ...] = ("local_only", "shortlisted", "submitted",
                                      "frozen_best", "rejected")
CANDIDATES_NOTE = ("v4 候选注册表（唯一事实源）。未登记的候选不得提交。"
                   "status ∈ local_only / shortlisted / submitted / frozen_best / rejected。"
                   "一旦 submitted 不得覆盖，修改必须新建 candidate_id 并写 parent。")


def _jsonable(obj: Any) -> Any:
    """递归转 JSON-可序列化（numpy 标量/数组、dataclass 之外的容器）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:                                     # pragma: no cover
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def load_candidates(path: str | Path | None = None) -> dict[str, Any]:
    """读候选注册表；文件不存在时返回**空骨架**（不写盘）。"""
    p = Path(path) if path else CANDIDATES
    if not p.is_file():
        return {"schema_version": 1, "created_at": _now(), "note": CANDIDATES_NOTE,
                "candidates": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_candidates(doc: dict[str, Any] | list[dict], path: str | Path | None = None) -> Path:
    """原子写候选注册表（tmp → replace），并做状态合法性校验。"""
    p = Path(path) if path else CANDIDATES
    payload = doc if isinstance(doc, dict) else {"schema_version": 1, "created_at": _now(),
                                                 "note": CANDIDATES_NOTE, "candidates": doc}
    payload = _jsonable(payload)
    for e in payload.get("candidates", []):
        st = e.get("status")
        if st is not None and st not in CANDIDATE_STATUSES:
            raise ValueError(f"非法 status {st!r}；允许 {CANDIDATE_STATUSES}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def find_candidate(candidate_id: str, path: str | Path | None = None) -> dict | None:
    for e in candidates(path):
        if str(e.get("candidate_id")) == str(candidate_id):
            return e
    return None


def upsert_candidate(entry: dict[str, Any], path: str | Path | None = None,
                     allow_submitted_overwrite: bool = False) -> Path:
    """新增/更新候选；**已 submitted 的候选默认拒绝覆盖**（必须新建 candidate_id）。"""
    if not entry.get("candidate_id"):
        raise ValueError("候选必须带 candidate_id")
    doc = load_candidates(path)
    cid = str(entry["candidate_id"])
    for i, e in enumerate(doc.get("candidates", [])):
        if str(e.get("candidate_id")) == cid:
            if e.get("status") == "submitted" and not allow_submitted_overwrite:
                raise PermissionError(
                    f"候选 {cid} 已 submitted，不得覆盖（新建 candidate_id 并写 parent）")
            merged = {**e, **_jsonable(entry)}
            hist = list(e.get("status_history", []))
            if entry.get("status") and entry["status"] != e.get("status"):
                hist.append({"status": entry["status"], "at": _now(),
                             "from": e.get("status")})
            if hist:
                merged["status_history"] = hist
            doc["candidates"][i] = merged
            return save_candidates(doc, path)
    new_entry = {**_jsonable(entry), "registered_at": _now()}
    if entry.get("status"):
        new_entry.setdefault("status_history", [{"status": entry["status"], "at": _now(),
                                                 "from": None}])
    doc.setdefault("candidates", []).append(new_entry)
    return save_candidates(doc, path)


def set_candidate_status(candidate_id: str, status: str, path: str | Path | None = None,
                         extra: dict[str, Any] | None = None) -> Path:
    """改状态（记录 `status_history`），非法状态抛错，未知候选抛错。"""
    if status not in CANDIDATE_STATUSES:
        raise ValueError(f"非法 status {status!r}；允许 {CANDIDATE_STATUSES}")
    doc = load_candidates(path)
    for e in doc.get("candidates", []):
        if str(e.get("candidate_id")) == str(candidate_id):
            if extra:
                e.update(_jsonable(extra))
            hist = list(e.get("status_history", []))
            hist.append({"status": status, "at": _now(), "from": e.get("status")})
            e["status_history"] = hist
            e["status"] = status
            return save_candidates(doc, path)
    raise KeyError(f"未知候选 {candidate_id!r}")


def freeze_candidate(candidate_id: str, path: str | Path | None = None,
                     sha256: str | None = None) -> Path:
    """冻结候选（`frozen_best`）：记录 sha256 与时间，此后不得再改产物。"""
    extra = {"frozen_at": _now()}
    if sha256:
        extra["frozen_sha256"] = str(sha256)
    return set_candidate_status(candidate_id, "frozen_best", path, extra=extra)


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

