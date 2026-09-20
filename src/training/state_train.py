"""E6/P0 原子状态头的**两阶段训练件**（纯逻辑层，torch 门控）。

E6 的核心不是"多一个头"，而是**两阶段 + 切片权重 + 负对照**三件事（E6/P0 §5）：

1. **阶段 1**：主干 + 原子头（`q_atom`）+ 联合头（`q_joint`）一起训，
   损失 = `L_align + λ1·L_aux + λ_atom·L_atom + λ_joint·L_joint`；
2. **阶段 2**：**冻结/降 lr** 原子头，改训连续头，并对不同切片加权
   （联合原子行 `w_joint`、非联合原子行 `w_nonjoint_atom`、有效行 `w_valid=1.0`），
   权重**永不为 0**（连续头是原子头误判时的 fallback）；
3. **负对照**：标签打乱（`label_shuffle_control`）——保持逐目标/联合边际分布不变，
   只破坏"输入↔标签"的对应关系；若打乱后指标不崩，说明指标口径或泄漏有问题。

另外提供两项**泄漏审计**：输入特征名不得含目标派生列（`input_no_label_leak_full`），
以及"原子标签只能来自标签列"的结构性检查（`atom_label_provenance`）。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .. import constants as C
from ..portability import HAS_TORCH, require

# 阶段 2 的切片权重默认值（E6/P0 §6：joint/non-joint-atom ∈ [0.1, 0.3]，valid 恒为 1.0）
SLICE_WEIGHT_DEFAULTS: dict[str, float] = {"joint": 0.2, "nonjoint_atom": 0.2, "valid": 1.0}
SLICE_WEIGHT_MIN: float = 1e-3
BANNED_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "por", "perm", "sw_", "target", "label", "y_true", "placeholder", "atom", "logid",
)


def slice_weight_matrix(y_atom: Any, y_joint: Any, mask: Any,
                        w_joint: float = SLICE_WEIGHT_DEFAULTS["joint"],
                        w_nonjoint_atom: float = SLICE_WEIGHT_DEFAULTS["nonjoint_atom"],
                        w_valid: float = SLICE_WEIGHT_DEFAULTS["valid"]) -> "np.ndarray":
    """按切片给逐元素权重 `(N,3)`（**永不为 0**，统一夹到 `SLICE_WEIGHT_MIN`）。

    * 联合原子行（三目标同时原子）→ `w_joint`
    * 非联合原子行（该目标原子、联合不为真）→ `w_nonjoint_atom`
    * 其它（有效/连续行）→ `w_valid`
    * 缺测行的权重置为 `w_valid`（是否参与完全由 `mask` 决定，避免"权重 0 × mask 0"二义）
    """
    y = np.asarray(y_atom, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"y_atom 必须为 (N,3)，got {y.shape}")
    if m.shape != y.shape:
        raise ValueError(f"mask 必须与 y_atom 同形，got {m.shape} vs {y.shape}")
    j = np.asarray(y_joint, dtype=bool).reshape(-1)
    if j.shape[0] != y.shape[0]:
        raise ValueError(f"y_joint 必须为 (N,)，got {j.shape}")
    w = np.full(y.shape, float(w_valid), dtype="float64")
    nonjoint = (~j)[:, None] & y
    w[nonjoint] = float(w_nonjoint_atom)
    w[j[:, None] & y] = float(w_joint)
    w = np.where(m, w, float(w_valid))
    return np.maximum(w, float(SLICE_WEIGHT_MIN))


def slice_weight_report(w: Any) -> dict[str, Any]:
    """权重收据（Gate 要求"切片权重非 0 且分档可查"）。"""
    a = np.asarray(w, dtype="float64")
    return {"shape": list(a.shape), "min": float(a.min()), "max": float(a.max()),
            "n_zero": int((a <= 0).sum()), "unique_rounded": sorted(
                {round(float(v), 6) for v in np.unique(a)}),
            "never_zero": bool((a > 0).all())}


# ---------------------------------------------------------------- 阶段 2 参数分组
def head_param_names(model) -> dict[str, list[str]]:
    """按名字前缀把参数分成 `q_*`（原子/联合头）与 `cont_*`（连续头）。"""
    require("torch")
    out: dict[str, list[str]] = {"q": [], "cont": [], "other": []}
    for name, _p in model.named_parameters():
        low = name.lower()
        if "q_" in low or low.endswith("q_atom") or low.endswith("q_joint"):
            out["q"].append(name)
        elif "cont_" in low or low.endswith("_cont"):
            out["cont"].append(name)
        else:
            out["other"].append(name)
    return out


def make_param_groups(model, q_head_lr_mult: float = 0.0, base_lr: float = 1e-3,
                      weight_decay: float = 1e-4) -> tuple[list[dict], dict[str, Any]]:
    """阶段 2 的优化器参数组：`q_head_lr_mult=0` → **冻结**原子头，否则 lr 按倍数缩放。

    联合头与原子头在阶段 2 **不参与训练**（E6/P0 §5 步 2）：它们的职责已在阶段 1 完成，
    阶段 2 只让连续头适配"原子行 fallback"的切片。
    """
    require("torch")
    names = head_param_names(model)
    q_params = [p for n, p in model.named_parameters() if n in set(names["q"])]
    rest = [p for n, p in model.named_parameters() if n not in set(names["q"])]
    mult = float(q_head_lr_mult)
    if mult < 0:
        raise ValueError(f"q_head_lr_mult 必须 ≥ 0，got {q_head_lr_mult}")
    frozen = mult == 0.0
    for p in q_params:
        p.requires_grad_(not frozen)
    groups: list[dict] = [{"params": rest, "lr": float(base_lr),
                           "weight_decay": float(weight_decay)}]
    if q_params and not frozen:
        groups.append({"params": q_params, "lr": float(base_lr) * mult,
                       "weight_decay": float(weight_decay)})
    return groups, {"q_head_params": names["q"], "cont_or_other_params": names["cont"]
                    + names["other"], "frozen_q_heads": bool(frozen),
                    "q_head_lr": (0.0 if frozen else float(base_lr) * mult),
                    "n_params": int(sum(p.numel() for p in rest)
                                    + (0 if frozen else sum(p.numel() for p in q_params)))}


# ---------------------------------------------------------------- 负对照 / 审计
def label_shuffle_control(y_atom: Any, y_joint: Any, seed: int = 42) -> dict[str, Any]:
    """标签打乱负对照：**保持边际分布**、只破坏配对（行级置换）。"""
    y = np.asarray(y_atom, dtype=bool)
    j = np.asarray(y_joint, dtype=bool).reshape(-1)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"y_atom 必须为 (N,3)，got {y.shape}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(y.shape[0])
    y_s, j_s = y[perm], j[perm]
    same_pair = float(np.mean((y_s == y).all(axis=1))) if y.shape[0] else 0.0
    return {"y_atom": y_s, "y_joint": j_s, "seed": int(seed),
            "marginals_preserved": bool(
                np.array_equal(y_s.mean(axis=0), y.mean(axis=0))
                and float(j_s.mean()) == float(j.mean())),
            "per_target_rates": {t: [float(y[:, i].mean()), float(y_s[:, i].mean())]
                                 for i, t in enumerate(C.TARGETS)},
            "joint_rate": [float(j.mean()), float(j_s.mean())],
            "row_identity_rate": same_pair,
            "note": ("打乱后指标应显著下降（≈随机）；若不降，说明指标口径或输入含泄漏，"
                     "必须先查因再谈提升")}


def input_no_label_leak_full(feature_names: Sequence[str],
                             banned: Sequence[str] = BANNED_FEATURE_SUBSTRINGS
                            ) -> dict[str, Any]:
    """输入特征名审计：不得含目标/标签/占位/井身份派生列（E6/P0 mandatory）。"""
    names = [str(n) for n in feature_names]
    hits: dict[str, list[str]] = {}
    for b in banned:
        low = str(b).lower()
        found = [n for n in names if low in n.lower()]
        # `sw_` 这类后缀片段要能匹配 `sw`，因此再补一次下划线归一化比较
        if not found and low.endswith("_"):
            stem = low[:-1]
            found = [n for n in names if stem == n.lower().strip("_")]
        if found:
            hits[str(b)] = found
    return {"n_features": len(names), "banned": list(banned), "hits": hits,
            "ok": bool(not hits),
            "note": "命中即为**违规**：目标派生列只允许来自标签表，不得进入输入矩阵"}


def atom_label_provenance(y_atom: Any, y_por: Any, y_perm: Any, y_sw: Any, mask: Any,
                          atol: float = 1e-9) -> dict[str, Any]:
    """结构性审计：原子标记必须**严格等价于**标签等于该目标的原子值。"""
    y = np.asarray(y_atom, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    cols = {"POR": np.asarray(y_por, dtype="float64"),
            "PERM": np.asarray(y_perm, dtype="float64"),
            "SW": np.asarray(y_sw, dtype="float64")}
    out: dict[str, Any] = {"per_target": {}}
    ok = True
    for i, t in enumerate(C.TARGETS):
        expect = np.isclose(cols[t], float(C.ATOM_VALUES[t]), atol=atol, rtol=0.0)
        mismatch = int(np.sum((y[:, i] != expect) & m[:, i]))
        out["per_target"][t] = {"n_mismatch": mismatch, "atom_rate": float(y[m[:, i], i].mean())
                                if m[:, i].any() else None}
        ok = ok and mismatch == 0
    out["ok"] = bool(ok)
    out["note"] = "原子标记只能由标签列表导出；与标签不一致即视为口径错误"
    return out


if HAS_TORCH:
    import torch

    def two_stage_epochs(model, arrays: Mapping[str, Any], cfg, stage: int = 1,
                         epochs: int = 1, optimizer=None, loss_kw: Mapping[str, Any] | None = None,
                         seed: int = 42) -> dict[str, Any]:
        """在**行级数组**上跑若干 epoch（`stage=1` 原子头 + 联合头；`stage=2` 连续头）。

        `arrays` 需含 `X` 与 `batch`（`por/perm_z/sw/mask/y_atom/y_joint`）。
        返回 `{epochs: [{epoch, loss, parts}], history: {...}}`；不做早停（早停由调用方
        用**官方分数**在内折上做，见 E6/code/train_state.py）。
        """
        require("torch")
        from . import loop as L                                       # noqa: PLC0415
        from ..losses import score_aligned as SAL                     # noqa: PLC0415

        loss_kw = dict(loss_kw or {})
        sw_full = loss_kw.pop("slice_weight", None)     # 全量 (N,3) 权重，按 minibatch 切片
        X = np.asarray(arrays["X"], dtype="float32")
        batch = {k: np.asarray(v) for k, v in arrays["batch"].items()}
        n = int(X.shape[0])
        dev = L.resolve_device(cfg)
        torch.manual_seed(int(seed))
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr),
                                          weight_decay=float(cfg.weight_decay))
        bs = int(getattr(cfg, "batch_size", 4096) or 4096)
        hist: list[dict[str, Any]] = []
        model.to(dev)
        for ep in range(int(epochs)):
            model.train()
            perm = torch.randperm(n)
            tot, nb = 0.0, 0
            for i in range(0, n, bs):
                idx = perm[i:i + bs].numpy()
                xb = torch.from_numpy(X[idx]).to(dev)
                bb = {k: torch.from_numpy(v[idx]).to(dev) for k, v in batch.items()}
                out = model(xb)
                kw = dict(loss_kw)
                if sw_full is not None:
                    sw = torch.as_tensor(np.asarray(sw_full), dtype=torch.float32)
                    if sw.shape[0] != n:
                        raise ValueError(f"slice_weight 行数 {sw.shape[0]} != 数据行数 {n}")
                    kw["slice_weight"] = sw[torch.from_numpy(idx)].to(dev)
                if int(stage) == 2:                     # 阶段 2：只训连续头
                    kw.update({"use_atom": False, "use_joint": False})
                with L.amp_context(cfg, dev):
                    total, parts = SAL.total_loss(out, bb, **kw)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(getattr(cfg, "grad_clip", 1.0) or 1.0))
                optimizer.step()
                tot += float(total.detach().cpu())
                nb += 1
            hist.append({"epoch": ep, "loss": tot / max(nb, 1),
                         "parts": {k: float(v) for k, v in parts.items()}})
        return {"epochs": hist, "stage": int(stage), "n_rows": n,
                "final_loss": hist[-1]["loss"] if hist else None}
