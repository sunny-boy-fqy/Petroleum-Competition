"""训练循环（E1/P1 §5）：bf16 前向、λ1 退火、梯度裁剪、真实评分早停、时间/磁盘纪律。

设计要点
--------
1. **早停依据必须是真实 `score.py` 分数**（`资料库/12` §2.3）：`run_training` 只负责
   "跑 epoch + 调回调"，评分由调用方传入的 `eval_fn` 完成（E1 用 inner-OOF 折）；
   loss 值只进日志，**不参与任何选择**。
2. **整折张量常驻**：E1 的行级数据 730k×32 f32 ≈ 93 MB，直接放到 GPU 上分批切片，
   比 DataLoader + worker 更省内存也更快（16 GiB 内存约束下 worker 反而危险）。
   E2/E3 的按井分片读取不走这里。
3. **时间预算**：每个 epoch 结束检查 `time_budget_h`，超时**优雅收尾**（返回
   `stopped_reason="time_budget"`），已训练的部分照样可用于早停决策与提交。
4. **磁盘纪律**：每个 epoch 调 `assert_disk_headroom(min_free_gb, path=...)`；
   checkpoint 由调用方（`checkpoint.rotate`）滚动淘汰。
5. 所有随机性在 `set_seed` 里固定（python/numpy/torch/cuda），保证同一 `--seed`
   的 OOF 可复算。

无 torch 时本模块可被导入（用于本机契约层单测），但任何训练调用都会抛出明确错误。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import constants as C
from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch

from . import metrics as M


@dataclass
class TrainConfig:
    """E1/P1 §6 的默认配方（超参只在 inner-OOF 上选；此处只是**起点**）。"""
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.1
    lr: float = 2e-3
    lr_min: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4096
    epochs: int = 40
    patience: int = 5
    seed: int = 42
    lam1_start: float = 1.0
    lam1_schedule: str = "linear_to_0.1"   # constant / linear_to_0.1 / cosine
    lam1_end: float = 0.1
    lam1_frac: float = 0.6
    lam_atom: float = 0.5
    lam_joint: float = 0.2
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"          # "bf16" | "fp32"
    device: str = "auto"             # "auto" | "cpu" | "cuda" | "cuda:0"
    time_budget_h: float | None = None
    min_free_gb: float = C.DISK_MIN_FREE_GB
    disk_path: str = "/"
    log_every: int = 5
    max_wells: int | None = None     # 仅 smoke 用；None = 全部

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(cfg: TrainConfig):
    """`auto` = 按 `src/hardware.py` 的优先级挑加速器（npu → cuda → cpu）。

    NPU 路径依赖 `torch_npu` 把 `torch.npu` 注册进来（`hardware.detect_accelerator`
    负责这次导入）；显式传 `--device npu:0` / `cuda:0` / `cpu` 时原样使用。
    """
    require("torch")
    import torch
    from .. import hardware as HW
    if cfg.device != "auto":
        return torch.device(cfg.device)
    kind = HW.detect_accelerator(torch)
    return torch.device(HW.device_string(kind))


def set_seed(seed: int) -> None:
    require("torch")
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    for backend in ("cuda", "npu"):
        mod = getattr(torch, backend, None)
        if mod is not None and hasattr(mod, "manual_seed_all"):
            try:
                mod.manual_seed_all(seed)
            except Exception:
                pass


def amp_context(cfg: TrainConfig, device):
    """bf16 autocast 上下文（CPU 上自动退化，便于本机做链路测试）。

    `device_type` 直接取实际设备（`npu`/`cuda`），因此 Ascend 上走的是
    `torch.autocast("npu", dtype=torch.bfloat16)`（由 torch_npu 实现）。
    """
    require("torch")
    if cfg.amp_dtype != "bf16" or device.type == "cpu":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


class TorchFold:
    """整折张量 → GPU 上的 torch 张量（一次搬运，之后按索引切片）。"""

    def __init__(self, t, device, with_targets: bool = True, to_device: bool = True):
        require("torch")
        self.device = device
        self.n_rows = t.n_rows
        self.X = torch.from_numpy(t.X)
        if to_device:
            self.X = self.X.to(device)
        self.y: dict[str, Any] = {}
        if with_targets:
            for k, v in (("por", t.y_por), ("perm_z", t.y_perm_z), ("sw", t.y_sw),
                         ("mask", t.mask), ("y_atom", t.y_atom), ("y_joint", t.y_joint)):
                if v is None:
                    continue
                ten = torch.from_numpy(v)
                self.y[k] = ten.to(device) if to_device else ten

    def batch(self, idx) -> dict[str, Any]:
        return {k: v[idx] for k, v in self.y.items()}


def lam1_schedule(kind: str, epoch: int, epochs: int, cfg: TrainConfig) -> float:
    """λ1 退火曲线（E7/P0 消融臂）：`constant` / `linear_to_0.1` / `cosine`。

    * `constant`      : 全程 `lam1_start`（**不退化**，作为"退火到底有没有用"的对照）；
    * `linear_to_0.1` : 前 `lam1_frac` 比例 epoch 内从 `lam1_start` 线性降到 `lam1_end`；
    * `cosine`        : 同一区间内余弦退火（两端导数为 0，中段更快）。

    三者都保证 `epoch >= span` 后恒为 `lam1_end`（可复算的终值）。
    """
    k = str(kind or "linear_to_0.1").lower()
    span = max(int(round(int(epochs) * float(cfg.lam1_frac))), 1)
    if k in ("constant", "const", "off"):
        return float(cfg.lam1_start)
    if epoch >= span:
        return float(cfg.lam1_end)
    r = float(epoch) / float(span)
    if k in ("cosine", "cos"):
        import math as _m
        frac = 0.5 * (1.0 - _m.cos(_m.pi * r))
    elif k in ("linear", "linear_to_0.1", "linear_to_end"):
        frac = r
    else:
        raise ValueError(f"未知 λ1 退火曲线：{kind!r}（constant/linear_to_0.1/cosine）")
    return float(cfg.lam1_start + (cfg.lam1_end - cfg.lam1_start) * frac)


def lam1_at(epoch: int, epochs: int, cfg: TrainConfig) -> float:
    """兼容旧调用点：按 `cfg.lam1_schedule` 取曲线（默认线性到 `lam1_end`）。"""
    return lam1_schedule(getattr(cfg, "lam1_schedule", "linear_to_0.1"), epoch, epochs, cfg)


def train_epoch(model, opt, data: TorchFold, cfg: TrainConfig, epoch: int,
                scaler_params: dict[str, Any], gen=None) -> dict[str, float]:
    """一个 epoch 的优化（返回逐项 loss 的均值）。

    `scaler_params` 必须携带**训练折**的 `s_por`/`s_sw`（`L_aux` 归一化尺度）。
    """
    require("torch")
    from ..losses.score_aligned import total_loss
    s_por = float(scaler_params.get("s_por", 11.34))
    s_sw = float(scaler_params.get("s_sw", 20.0))
    model.train()
    n = data.n_rows
    bs = int(cfg.batch_size)
    perm = torch.randperm(n, device=data.device, generator=gen)
    agg: dict[str, float] = {}
    nb = 0
    lam1 = lam1_at(epoch, cfg.epochs, cfg)
    for a in range(0, n, bs):
        idx = perm[a:a + bs]
        if idx.numel() < 2:      # BatchNorm 需要 >=2 个样本
            continue
        xb = data.X[idx]
        batch = data.batch(idx)
        opt.zero_grad(set_to_none=True)
        with amp_context(cfg, data.device):
            out = model(xb)
            loss, parts = total_loss(out, batch, lam1=lam1, lam_atom=cfg.lam_atom,
                                     lam_joint=cfg.lam_joint, s_por=s_por, s_sw=s_sw)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch} (batch {nb})")
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        opt.step()
        parts["total"] = float(loss.detach())
        parts["lam1"] = lam1
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()} if nb else {"total": float("nan")}


@torch.no_grad() if HAS_TORCH else (lambda f: f)
def predict_torch(model, data: TorchFold, cfg: TrainConfig) -> dict[str, Any]:
    """整折推理（返回 numpy，标签尺度解码交给 `metrics`）。"""
    require("torch")
    model.eval()
    n = data.n_rows
    bs = max(int(cfg.batch_size), 8192)
    chunks: dict[str, list] = {}
    for a in range(0, n, bs):
        xb = data.X[a:a + bs]
        with amp_context(cfg, data.device):
            out = model(xb)
        for k in ("por", "perm_z", "sw", "q_atom", "q_joint"):
            chunks.setdefault(k, []).append(out[k].float().cpu().numpy())
    import numpy as np
    pred = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    return pred


def check_disk(cfg: TrainConfig, verbose: bool = False) -> dict[str, Any]:
    """每个 epoch 的磁盘余量检查（E1/P1 §5 步 2）。"""
    from ..data.disk_guard import assert_disk_headroom, disk_report
    assert_disk_headroom(float(cfg.min_free_gb), path=cfg.disk_path, verbose=verbose)
    return disk_report(cfg.disk_path)


def run_training(model, data: TorchFold, cfg: TrainConfig,
                 eval_fn: Callable[[Any], dict[str, Any]] | None = None,
                 on_epoch: Callable[[int, dict[str, Any]], None] | None = None,
                 optimizer=None, resume_epoch: int = -1,
                 scaler_params: dict[str, Any] | None = None,
                 keep_best: bool = True) -> dict[str, Any]:
    """训练 `cfg.epochs` 个 epoch；每个 epoch 用 `eval_fn` 拿**真实分数**做早停。

    eval_fn(model) -> dict，至少含 `total`（越高越好）。

    `keep_best=True`（默认）时把**最优 epoch 的权重**留到最后（训练结束时写回模型），
    这样"用 inner-OOF 早停"选出来的模型与随后"选 τ / 推理"用的模型是同一个 ——
    否则 τ 会在**最后一个** epoch 的权重上选，与 `best_epoch` 报告的语义不符。
    返回 history：`epochs` 列表、`best_epoch`、`best_total`、`stopped_reason`、`seconds`。
    """
    require("torch")
    device = data.device
    sp = dict(scaler_params or {})

    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=float(cfg.lr),
                                         weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(int(cfg.epochs), 1), eta_min=float(cfg.lr_min))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed))

    t0 = time.time()
    hist: list[dict[str, Any]] = []
    best = {"epoch": -1, "total": -math.inf}
    best_state: dict[str, Any] | None = None
    stopped = "completed"
    n_bad = 0
    for epoch in range(int(resume_epoch) + 1, int(cfg.epochs)):
        if cfg.time_budget_h is not None and (time.time() - t0) > float(cfg.time_budget_h) * 3600:
            stopped = "time_budget"
            break
        parts = train_epoch(model, opt, data, cfg, epoch, sp, gen=gen)
        sched.step()
        rec: dict[str, Any] = {"epoch": epoch, "lr": float(opt.param_groups[0]["lr"]),
                               "seconds": round(time.time() - t0, 2), **parts}
        if eval_fn is not None:
            ev = eval_fn(model)
            rec["val_total"] = float(ev.get("total", float("nan")))
            rec["val"] = ev
            if rec["val_total"] > best["total"]:
                best = {"epoch": epoch, "total": rec["val_total"]}
                if keep_best:
                    best_state = {k: v.detach().to("cpu").clone()
                                  for k, v in model.state_dict().items()}
                n_bad = 0
            else:
                n_bad += 1
        hist.append(rec)
        if on_epoch is not None:
            on_epoch(epoch, rec)
        try:
            rec["disk_free_gb"] = round(float(check_disk(cfg)["free_gb"]), 3)
        except Exception as exc:                      # 磁盘检查失败不静默：记录原因
            rec["disk_error"] = str(exc)
            raise
        if eval_fn is not None and n_bad >= int(cfg.patience):
            stopped = "early_stop"
            break
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs": hist, "best_epoch": int(best["epoch"]),
            "best_total": (None if best["total"] == -math.inf else float(best["total"])),
            "best_restored": bool(keep_best and best_state is not None),
            "stopped_reason": stopped, "seconds": round(time.time() - t0, 2),
            "n_epochs_run": len(hist)}


class TimeTracker:
    """写 `training_time_log.json`（Gate 判据 `training_time_log_valid`）。"""

    def __init__(self, stage: str, time_budget_h: float | None = None):
        self.stage = stage
        self.time_budget_h = time_budget_h
        self.t0 = time.time()
        self.folds: list[dict[str, Any]] = []

    def add_fold(self, fold: int, seconds: float, epochs_run: int,
                 stopped_reason: str = "completed", extra: dict[str, Any] | None = None) -> None:
        rec = {"fold": int(fold), "seconds": round(float(seconds), 3),
               "epochs_run": int(epochs_run), "stopped_reason": stopped_reason}
        if extra:
            rec.update(extra)
        self.folds.append(rec)

    def payload(self, manifest_sha256: str | None = None,
                config: dict[str, Any] | None = None) -> dict[str, Any]:
        total = round(time.time() - self.t0, 3)
        valid = bool(self.folds) and total > 0 and all(f["seconds"] > 0 for f in self.folds)
        if self.time_budget_h is not None:
            valid = valid and total <= float(self.time_budget_h) * 3600 * 1.5
        out: dict[str, Any] = {
            "stage": self.stage,
            "started_at_unix": self.t0,
            "total_seconds": total,
            "time_budget_h": self.time_budget_h,
            "folds": self.folds,
            "valid": valid,
        }
        if config:
            out["config"] = config
        if manifest_sha256:
            out["manifest_sha256"] = manifest_sha256
        return out

    def write(self, path: str | Path, **kw) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(M.jsonable(self.payload(**kw)), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)
        return p
