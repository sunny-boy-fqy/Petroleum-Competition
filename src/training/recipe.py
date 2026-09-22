"""E7 冻结配置的读取与应用（loss_v1 / decode_v1）。

E7 的消融搜索会写出 ``versions/configs/loss_v1.json``；本模块提供唯一的读取入口，
并由各训练脚本显式应用，避免“报告生成了配方但训练路径没消费”。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

V4 = Path(__file__).resolve().parents[2]
DEFAULT_LOSS_CONFIG = V4 / "versions" / "configs" / "loss_v1.json"
DEFAULT_DECODE_CONFIG = V4 / "versions" / "configs" / "decode_v1.json"


def _reports_dir() -> Path | None:
    rep = os.environ.get("V4_REPORTS_DIR")
    return Path(rep) if rep else None


def default_loss_recipe_candidates() -> list[Path]:
    """损失配方的解析顺序（审查 H4）：显式 env -> $V4_REPORTS_DIR -> 仓库默认。

    E7 现在会把 `loss_v1.json` 同时写到 `$V4_REPORTS_DIR`（随 5 分钟 mirror 持久化），
    因此多任务拆分时下一条任务也能读到，不会再静默回退到默认配置。
    """
    out: list[Path] = []
    rep = _reports_dir()
    if rep is not None:
        out.append(rep / "loss_v1.json")
    out.append(DEFAULT_LOSS_CONFIG)
    return out


def _read_recipe_file(p: Path) -> dict[str, Any]:
    """读取并校验一个配置 JSON；**存在但损坏必须报错**（审查 M1）。"""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"loss recipe 文件存在但无法解析：{p}（{type(exc).__name__}: {exc}）") from exc
    if not isinstance(d, dict):
        raise ValueError(f"loss recipe 必须是 JSON object：{p}（got {type(d).__name__}）")
    return d


def load_loss_recipe(path: str | Path | None = None) -> dict[str, Any] | None:
    """加载损失配方。

    * 显式传入 `path` 或设置 `V4_LOSS_CONFIG`：路径不存在/损坏都是**硬错误**；
    * 否则按 `$V4_REPORTS_DIR/loss_v1.json` -> 仓库默认顺序查找；都不存在才返回 None。
    """
    explicit = path if path is not None else os.environ.get("V4_LOSS_CONFIG")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"loss recipe 显式路径不存在：{p}")
        return _read_recipe_file(p)
    for p in default_loss_recipe_candidates():
        if p.is_file():
            return _read_recipe_file(p)
    return None


_LOSS_MAPPING: dict[str, str] = {
    "lam1": "lam1",
    "lam1_schedule": "lam1_schedule",
    "lam2": "lam_joint",
    "lam3": "lam_phys",
    "use_align": "use_align",
    "use_aux": "use_aux",
    "aux_normalize": "aux_normalize",
    "perm_clamp": "perm_clamp",
    "boundary_kappa": "boundary_kappa",
    "boundary_sigma": "boundary_sigma",
    "huber_beta": "huber_beta",
    "pos_weight": "pos_weight",
    "alpha_nonjoint": "alpha_nonjoint",
}


def loss_params(recipe: dict[str, Any] | None) -> dict[str, Any]:
    """返回可直接用于 ``total_loss`` 的 kwargs（只含已实现字段）。"""
    out: dict[str, Any] = {}
    if not recipe:
        return out
    src = recipe.get("loss") if isinstance(recipe.get("loss"), dict) else recipe
    for src_key, dst_key in _LOSS_MAPPING.items():
        if src_key in src and src[src_key] is not None:
            out[dst_key] = src[src_key]
    return out


_LOSS_META_KEYS = frozenset({
    "schema_version", "version", "stage", "selected_arm", "selected_on",
    "inner_only", "not_implemented", "created_at",
})


def ignored_loss_keys(recipe: dict[str, Any] | None) -> list[str]:
    """返回配方里**本实现不消费**的键，供调用方显式告警（审查 M1）。"""
    if not recipe:
        return []
    src = recipe.get("loss") if isinstance(recipe.get("loss"), dict) else recipe
    if not isinstance(src, dict):
        return []
    known = set(_LOSS_MAPPING) | set(_LOSS_META_KEYS) | {"loss"}
    return sorted(k for k in src if k not in known)


def apply_loss_recipe(cfg: Any, recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    """把 E7 配方写入一个 ``loop.TrainConfig`` 实例（仅覆盖配方明确给出的字段）。"""
    recipe = recipe if recipe is not None else load_loss_recipe()
    params = loss_params(recipe)
    ignored = ignored_loss_keys(recipe)
    if ignored:
        import sys as _sys
        print(f"[recipe] WARNING: loss recipe 中以下键未被消费：{ignored}",
              file=_sys.stderr, flush=True)
    if not params:
        return {}
    # TrainConfig 字段映射
    if "lam1" in params:
        cfg.lam1_start = float(params["lam1"])
    if "lam1_schedule" in params:
        cfg.lam1_schedule = str(params["lam1_schedule"])
    if "lam_joint" in params:
        cfg.lam_joint = float(params["lam_joint"])
    for k in ("use_align", "use_aux", "aux_normalize", "perm_clamp"):
        if k in params and hasattr(cfg, k):
            setattr(cfg, k, bool(params[k]))
    for k in ("boundary_kappa", "boundary_sigma", "huber_beta", "alpha_nonjoint",
              "lam_phys", "phys_huber_beta", "phys_por_scale"):
        if k in params and hasattr(cfg, k):
            setattr(cfg, k, float(params[k]))
    if "pos_weight" in params and hasattr(cfg, "pos_weight"):
        cfg.pos_weight = (None if params["pos_weight"] is None
                          else float(params["pos_weight"]))
    return params
