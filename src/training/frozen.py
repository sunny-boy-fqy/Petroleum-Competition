"""E5 冻结骨干：加载已训练权重并从**逐行隐状态**取特征（不重训、不改折）。

为什么需要它
------------
E5 的三个逐目标头都建立在**冻结的序列主干**上（E3/P2 的 U-Net 或 E4/P1 的 PatchTF），
而 `seq_loop.predict_well_chunked` 只回传 `OUT_KEYS`（头输出），不暴露倒数第二层特征。
本模块补上这一层，并且**沿用同一条分块推理 + 加权拼接路径**（避免"训练用分块、取特征用整井"
这种口径漂移）。

两种取特征方式
--------------
1. **原生**：主干自带 `forward_states`（`models.patchtf.PatchTF`）→ 直接逐 chunk 调用；
2. **钩子**：其它主干（U-Net/TCN）给定 `hook_module`（例如 `model.blocks[-1]`），
   用 forward hook 捕获其**逐行**输出 `(B,L,d)`。
两者都不接受"用整井一次性前向"（内存与训练口径都不一致）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..portability import HAS_TORCH, require




def resolve_backbone_checkpoint(backbone_ckpt: str | Path | None, arch: str | None,
                                fold: int, run_root: str | Path | None = None
                                ) -> Path | None:
    """定位某个折的冻结骨干 checkpoint。

    支持：
      * 直接文件路径；
      * 含 ``{fold}`` / ``{k}`` 的模板，例如
        ``$V4_RUN_ROOT/E4/patchtf/fold{fold}/best.pt``；
      * 一个 run 目录：自动尝试 ``fold{k}/best.pt`` 与
        ``fold{k}/select/best.pt``；
      * 未显式给路径时，按 ``E4/{arch}/fold{k}/best.pt`` 与
        ``E3/{arch}/fold{k}/best.pt`` 自动发现。

    返回 None 表示未找到；调用方在 smoke/exploratory 下可退回随机初始化。
    """
    fmt = {"fold": int(fold), "k": int(fold), "i": int(fold)}
    if backbone_ckpt:
        raw = str(backbone_ckpt)
        try:
            raw = raw.format(**fmt)
        except Exception:
            pass
        p = Path(raw).expanduser()
        if p.is_file():
            return p
        if p.is_dir():
            for cand in (p / f"fold{fold}" / "best.pt",
                         p / f"fold{fold}" / "select" / "best.pt",
                         p / f"select" / f"fold{fold}" / "best.pt"):
                if cand.is_file():
                    return cand
        # 未展开的 glob 模板（例如 fold*）交给调用方报错更清晰。
        matches: list[Path] = []
        if any(ch in raw for ch in "*?["):
            pp = Path(raw)
            try:
                if pp.is_absolute():
                    # Path().glob() 不支持绝对 pattern；拆成 parent + name。
                    matches = sorted(pp.parent.glob(pp.name))
                else:
                    matches = sorted(Path().glob(raw))
            except (NotImplementedError, ValueError, OSError):
                matches = []
        for cand in matches:
            if cand.is_file() and ("fold" not in cand.name or str(fold) in cand.name):
                return cand
        return None

    root = Path(run_root or os.environ.get("V4_RUN_ROOT")
                or (Path(os.environ.get("V4_DATA_ROOT", "/data")) / "v4" / "runs"))
    arch_candidates: list[str] = []
    for a in (arch, "patchtf", "unet", "tcn"):
        if a and str(a) not in arch_candidates:
            arch_candidates.append(str(a))
    for a in arch_candidates:
        for cand in (root / "E4" / a / f"fold{fold}" / "best.pt",
                     root / "E3" / a / f"fold{fold}" / "best.pt",
                     root / "E4" / a / "select" / f"fold{fold}" / "best.pt"):
            if cand.is_file():
                return cand
    return None

def states_supported(model) -> str:
    """返回取特征方式：`"native"`（有 `forward_states`）或 `"hook"`（需传 hook_module）。"""
    return "native" if hasattr(model, "forward_states") else "hook"


def load_frozen_model(ckpt: str | Path, arch: str, n_features: int,
                      arch_kwargs: dict | None = None, device: str | None = None,
                      init_stats: dict | None = None):
    """读回权重并置 `eval()`（**冻结**：所有参数 `requires_grad_(False)`）。"""
    require("torch")
    from ..training import checkpoint as CK
    from ..training import loop as L
    from ..training.seq_loop import build_seq_model

    model = build_seq_model(str(arch), int(n_features), init_stats=init_stats,
                            **(dict(arch_kwargs or {})))
    CK.load_checkpoint(ckpt, model=model, map_location="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device:
        import torch as _t
        model.to(_t.device(device) if isinstance(device, str) else device)
    return model


if HAS_TORCH:
    import torch

    def _chunk_states(model, X: np.ndarray, cfg, opt, device, hook_module=None):
        """逐 chunk 取 `(L, d)` 隐状态（原生优先，其次 hook）。"""
        from ..data import seq_dataset as SD
        from ..training import loop as L

        if isinstance(device, str):
            device = torch.device(device)          # `amp_context` 需要 torch.device
        n = int(X.shape[0])
        chunks = SD.chunks_for(n, opt.chunk, opt.overlap)
        pieces: list[np.ndarray] = []
        native = hasattr(model, "forward_states")
        captured: dict[str, Any] = {}

        handle = None
        if not native:
            if hook_module is None:
                raise ValueError("该主干没有 forward_states；请传入 hook_module（逐行输出的子模块）")

            def _hook(_m, _inp, out):
                captured["out"] = out

            handle = hook_module.register_forward_hook(_hook)
        try:
            step = max(int(getattr(opt, "batch_chunks", 1) or 1), 1)
            for i in range(0, len(chunks), step):
                batch = chunks[i:i + step]
                xs = np.stack([X[s:s + ln] for s, ln in batch])
                with torch.no_grad(), L.amp_context(cfg, device):
                    t = torch.from_numpy(xs).to(device)
                    if native:
                        st = model.forward_states(t)
                    else:
                        captured.clear()
                        model(t)
                        st = captured["out"]
                st = st.float().cpu().numpy()
                for j, (_s, ln) in enumerate(batch):
                    pieces.append(st[j, :ln])
        finally:
            if handle is not None:
                handle.remove()
        return SD.stitch_chunks(chunks, pieces, n, kind=opt.weight_kind)

    def well_states(model, X: np.ndarray, cfg, opt, device, hook_module=None) -> np.ndarray:
        """单井逐行隐状态 `(L, d)`（分块 + 加权拼接；长度 == `X.shape[0]`）。"""
        require("torch")
        if X.ndim != 2:
            raise ValueError(f"X 必须为 (L,F)，got {tuple(X.shape)}")
        return _chunk_states(model, np.asarray(X, dtype="float32"), cfg, opt, device,
                             hook_module=hook_module)

    def states_for_wells(model, wells: Sequence[str], cache, cfg, opt, device,
                         split: str = "train", scaler=None, phys_params=None,
                         hook_module=None) -> dict[str, np.ndarray]:
        """逐井取隐状态（特征矩阵与训练完全同源：同一 spec / 同一折内标尺）。"""
        require("torch")
        from ..data import row_dataset as RD

        out: dict[str, np.ndarray] = {}
        for w in wells:
            got = RD._well_feature_matrix(cache, w, split, opt.spec, phys_params=phys_params)
            X = np.asarray(got[0] if isinstance(got, tuple) else got, dtype="float32")
            if scaler is not None:
                X = np.asarray(scaler.transform(X), dtype="float32")
            out[str(w)] = well_states(model, X, cfg, opt, device, hook_module=hook_module)
        return out

    def state_report(states: dict[str, np.ndarray], limit_mb: float = 300.0) -> dict[str, Any]:
        """内存/形状收据（E5 §风险：整井特征常驻会撑爆 16 GB 内存）。"""
        n_rows = int(sum(v.shape[0] for v in states.values()))
        dim = int(next(iter(states.values())).shape[1]) if states else 0
        mb = n_rows * dim * 4 / 1024 ** 2
        return {"n_wells": len(states), "n_rows": n_rows, "d_model": dim,
                "float32_mb": round(mb, 3), "limit_mb": float(limit_mb),
                "ok": bool(mb <= float(limit_mb))}


def pooled_from_states(states: np.ndarray, how: str = "mean") -> np.ndarray:
    """井级向量（`mean`/`max`）——供 E8/P1 井级分支对照用（纯 numpy，便于单测）。"""
    x = np.asarray(states, dtype="float64")
    if x.ndim != 2:
        raise ValueError(f"states 必须为 (L,d)，got {tuple(x.shape)}")
    if how == "mean":
        return x.mean(axis=0)
    if how == "max":
        return x.max(axis=0)
    raise ValueError(f"how ∈ mean/max，got {how!r}")
