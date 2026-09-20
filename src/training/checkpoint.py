"""checkpoint 落盘/读取/滚动淘汰（E1/P1 §4 + §7「可 --resume」「disk_budget_ok」）。

要点
----
1. **bf16 存储**：Ascend 910B 上 state_dict 以 bfloat16 保存（体积减半），读取时按需转 float32
   载入模型。*只有* 权重张量转 bf16，manifest 里的标量保持 python float。
2. **manifest 与权重同源落盘**：连续头标尺（`por_max`/`sw_mu`/`sw_sigma`）与 `L_aux`
   尺度（`s_por`/`s_sw`）**必须**随 checkpoint 一起存 —— 否则推理期无法反变换（E1/P1 §7）。
3. **滚动淘汰**：每个 run 目录只保留 `best.pt`、`last.pt`、`last_prev.pt`（64 GiB 磁盘纪律）。
4. **可续训**：`load_for_resume` 同时恢复 optimizer 状态与 epoch 计数；`verify_resumable`
   在 Gate 里给出 `checkpoint_resumable` 的**可复算**证据（真的读回来一次）。

本模块需要 torch；无 torch 时导入不报错，调用会抛出明确错误（便于本机无 torch 跑契约层单测）。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch


def manifest_path(ckpt: str | Path) -> Path:
    """`x.pt` → `x.manifest.json`（与权重同目录、同名不同后缀）。"""
    p = Path(ckpt)
    return p.with_suffix(".manifest.json")


def _tensor_to_bf16(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state.items():
        out[k] = v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v
    return out


def save_checkpoint(path: str | Path, model, meta: dict[str, Any] | None = None,
                    optimizer=None, bf16: bool = True, extra: dict[str, Any] | None = None) -> Path:
    """原子写 checkpoint（tmp → rename）+ manifest。"""
    require("torch")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    payload: dict[str, Any] = {
        "state_dict": _tensor_to_bf16(state) if bf16 else state,
        "dtype": "bfloat16" if bf16 else "float32",
        "format": 1,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = p.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(p)

    man = {
        "path": str(p),
        "bytes": int(p.stat().st_size),
        "dtype": payload["dtype"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": getattr(torch, "__version__", "unknown"),
        "git_revision": _git_revision(),
        "n_params": int(sum(v.numel() for v in state.values() if torch.is_tensor(v))),
    }
    if meta:
        man.update(meta)
    if extra:
        man.update(extra)
    mp = manifest_path(p)
    mtmp = mp.with_suffix(".tmp.json")
    mtmp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    mtmp.replace(mp)
    return p


def _git_revision() -> str | None:
    """尽力取仓库 revision（无 git 时返回 None，不阻塞训练）。"""
    try:
        import subprocess
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def read_manifest(path: str | Path) -> dict[str, Any]:
    mp = manifest_path(path)
    if not mp.is_file():
        raise FileNotFoundError(f"checkpoint manifest missing: {mp}")
    return json.loads(mp.read_text(encoding="utf-8"))


def load_checkpoint(path: str | Path, model=None, map_location: str = "cpu",
                    strict: bool = True) -> dict[str, Any]:
    """读回权重（默认 CPU，便于本机做"可读性"验证与 CPU 推理）。"""
    require("torch")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint missing: {p}")
    payload = torch.load(p, map_location=map_location, weights_only=False)
    state = payload["state_dict"]
    if model is not None:
        state_f32 = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                     for k, v in state.items()}
        model.load_state_dict(state_f32, strict=strict)
    return {"state_dict": state, "dtype": payload.get("dtype"),
            "optimizer": payload.get("optimizer")}


def load_for_resume(path: str | Path, model, optimizer=None,
                    map_location: str = "cpu") -> dict[str, Any]:
    """续训：恢复权重（+optimizer）。返回 `{"epoch": int, "manifest": {...}}`。"""
    out = load_checkpoint(path, model=model, map_location=map_location)
    if optimizer is not None and out.get("optimizer"):
        optimizer.load_state_dict(out["optimizer"])
    man = read_manifest(path)
    return {"epoch": int(man.get("epoch", -1)), "manifest": man,
            "optimizer_restored": bool(optimizer is not None and out.get("optimizer"))}


def rotate(run_dir: str | Path, keep: tuple[str, ...] = ("best.pt", "last.pt", "last_prev.pt")
           ) -> list[str]:
    """把 `last.pt` 轮转为 `last_prev.pt`，并删除 keep 之外的 *.pt（滚动淘汰）。

    返回被删除的文件名列表（供日志/审计）。
    """
    d = Path(run_dir)
    if (d / "last.pt").is_file():
        shutil.copy2(d / "last.pt", d / "last_prev.pt")
    removed: list[str] = []
    for f in sorted(d.glob("*.pt")):
        if f.name not in keep:
            f.unlink()
            removed.append(f.name)
    return removed


def verify_resumable(path: str | Path, model_factory, map_location: str = "cpu") -> dict[str, Any]:
    """Gate 证据：真的把 checkpoint 读回一个新建模型（`checkpoint_resumable`）。

    `model_factory()` 必须返回一个**结构一致**的模型。返回 dict 可直接写进 Gate 的
    `checks` 说明里（含 `ok`、参数张量数、dtype、字节数）。
    """
    require("torch")
    m = model_factory()
    out = load_checkpoint(path, model=m, map_location=map_location, strict=True)
    n = len(out["state_dict"])
    return {"ok": True, "path": str(path), "tensors": int(n), "dtype": out.get("dtype"),
            "bytes": int(Path(path).stat().st_size),
            "manifest": read_manifest(path)}


def prune_keep_only(run_dir: str | Path, names: tuple[str, ...]) -> None:
    """只保留给定文件名（更强的淘汰，用于每次 epoch 落盘后调用）。"""
    d = Path(run_dir)
    keep = set(names)
    for f in d.glob("*.pt"):
        if f.name not in keep:
            f.unlink()


def total_bytes(paths) -> int:
    return int(sum(Path(p).stat().st_size for p in paths if Path(p).is_file()))


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")
