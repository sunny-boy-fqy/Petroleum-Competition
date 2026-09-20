"""E8/P2 EMA 权重影子（torch 侧）：`decay ∈ {0.99, 0.999, 0.9995}`，逐 **step** 更新。

与 `ensemble.blend.ema_update`（numpy，单测用）同语义，这里面向真实的 `nn.Module`：

* `init_shadow(model)`：复制一份 `float32` 影子（**不建计算图**）；
* `ema_step(shadow, model, decay)`：原地更新；
* `context_ema(model, shadow)`：临时把影子权重换进模型（`ema.pt` 评估用），退出时**精确还原**；
* `shadow_state_dict/load_shadow`：与训练 checkpoint 同生命周期，便于 resume。

注意：EMA 的评估必须用**真实 `score.py`**（E8/P2 §5 步 1），因此调用方在 `context_ema`
内跑一次内折推理即可——本模块不涉及任何 loss 选择。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Sequence

from ..portability import HAS_TORCH, require

DEFAULT_DECAYS: tuple[float, ...] = (0.99, 0.999, 0.9995)


def check_decays(decays: Sequence[float]) -> tuple[float, ...]:
    out = tuple(float(d) for d in decays)
    if not out:
        raise ValueError("decays 不能为空")
    for d in out:
        if not (0.0 <= d < 1.0):
            raise ValueError(f"decay 必须在 [0,1)，got {d}")
    return out


if HAS_TORCH:
    import torch

    def init_shadow(model) -> dict[str, Any]:
        """`{name: tensor}` 影子（detach + clone，不带梯度）。"""
        require("torch")
        return {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def ema_step(shadow: Mapping[str, Any], model, decay: float) -> dict[str, Any]:
        """`θ_ema ← decay·θ_ema + (1−decay)·θ`（要求键集合一致）。"""
        require("torch")
        d = check_decays([decay])[0]
        cur = model.state_dict()
        missing = [k for k in cur if k not in shadow]
        if missing:
            raise KeyError(f"影子缺少键：{missing[:5]}（resume 时键不匹配？）")
        for k, v in cur.items():
            s = shadow[k]
            if not torch.is_floating_point(v):
                s.copy_(v)                       # 整数缓冲（如 num_batches_tracked）直接同步
                continue
            s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
        return dict(shadow)

    @contextmanager
    def context_ema(model, shadow: Mapping[str, Any]):
        """临时把 `shadow` 换进 `model`（评估 EMA 权重），退出时**精确还原**。"""
        require("torch")
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict({k: shadow[k] for k in backup}, strict=True)
            yield model
        finally:
            model.load_state_dict(backup, strict=True)

    def shadow_state_dict(shadow: Mapping[str, Any], decay: float | None = None) -> dict[str, Any]:
        return {"decay": (None if decay is None else float(decay)),
                "shadow": {k: v.detach().cpu().clone() for k, v in shadow.items()}}

    def load_shadow(payload: Mapping[str, Any], model) -> tuple[dict[str, Any], float | None]:
        sh = {k: v.detach().clone().float() for k, v in payload["shadow"].items()}
        keys = set(model.state_dict())
        if set(sh) != keys:
            raise KeyError(f"EMA 影子键与模型不一致：多 {sorted(set(sh) - keys)[:3]} / "
                           f"缺 {sorted(keys - set(sh))[:3]}")
        return sh, payload.get("decay")

    def decay_report(model, shadows: Mapping[float, Mapping[str, Any]]) -> list[dict[str, Any]]:
        """影子与当前权重的 L2 偏差（诊断：EMA 是否真的在动）。"""
        cur = model.state_dict()
        out = []
        for d, sh in sorted(shadows.items()):
            num = den = 0.0
            for k, v in cur.items():
                if not torch.is_floating_point(v):
                    continue
                diff = (sh[k] - v.detach().float()).norm().item()
                num += diff * diff
                den += float(v.detach().float().norm().item()) ** 2
            out.append({"decay": float(d), "rel_l2": (num ** 0.5) / max(den ** 0.5, 1e-12)})
        return out
