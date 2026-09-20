"""已训练模型的推理（CPU-only 主路径）：manifest → 权重 → 逐井预测 → 提交载荷。

为什么 CPU-only
---------------
平台评分环境不保证 GPU，且 95,948 行 × 32 维的行级/序列前向在 CPU 上是秒级到分钟级；
`PLAN.md` 的部署约束是"权重随包 + CPU 可推理"。因此本模块**默认 `map_location="cpu"`**，
显式传 `device="cuda"` 才用 GPU。

与 `predict.py` 的分工
----------------------
- `predict.py` 是**官方入口**（`--data_dir/--output/--use-version`），版本表在
  `versions/registry.json`；
- 本模块提供"trained 版本"真正需要的加载/解码/组包逻辑，供 `predict.py` 在
  E1/E10 之后挂上（`PREDICTORS` 表加一项即可）。

解码纪律
--------
1. 连续头：`PERM = 10**clip(z, -6, 6)`（严格 > 0）、SW 只做 `[0, 100]` 软保护裁剪
   （**永不裁到 [0,1]**）、POR 直出；
2. 原子门：若 manifest 带 `tau_atom`（逐目标 τ，来自 inner-OOF 选择），执行
   `per_target_hard_switch`（**无插值**）；
3. 深度保留 1 位小数（`C.DEPTH_DECIMALS`），逐井逐行与输入深度对齐。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import constants as C
from ..data import dataset as D
from ..data import row_dataset as RD
from ..features import basic as F
from ..portability import HAS_TORCH, require
from ..training import checkpoint as CK
from ..training import metrics as M


@dataclass
class Manifest:
    """checkpoint manifest 的**唯一读取口径**（含反变换所需的一切标量）。"""
    path: Path
    raw: dict[str, Any]

    @property
    def row_scaler(self) -> RD.RowScaler:
        return RD.RowScaler.from_dict(self.raw["row_scaler"])

    @property
    def target_scalers(self) -> dict[str, Any]:
        return dict(self.raw.get("target_scalers", {}))

    @property
    def model_kwargs(self) -> dict[str, Any]:
        m = dict(self.raw.get("model", {}))
        return {"n_features": int(m.get("n_features", F.N_FEATURES)),
                "hidden": int(m.get("hidden", 256)),
                "layers": int(m.get("layers", 2)),
                "dropout": float(m.get("dropout", 0.0))}

    @property
    def tau_atom(self):
        return self.raw.get("tau_atom")

    @property
    def n_features(self) -> int:
        return int(self.raw.get("model", {}).get("n_features", F.N_FEATURES))


def load_manifest(ckpt: str | Path) -> Manifest:
    p = Path(ckpt)
    return Manifest(path=p, raw=CK.read_manifest(p))


def load_model(ckpt: str | Path, manifest: Manifest | None = None, device: str = "cpu"):
    """按 manifest 里的结构重建模型并载入权重（bf16 → float32）。"""
    require("torch")
    man = manifest or load_manifest(ckpt)
    from ..models.row_mlp import build_model

    scalers = man.target_scalers
    model = build_model(init_stats=None, **man.model_kwargs)
    # 连续头标尺必须与训练折一致（否则 SW/POR 反变换会整体偏移）
    import torch
    if scalers:
        with torch.no_grad():
            if "por_max" in scalers:
                model.por_max.fill_(float(scalers["por_max"]))
            if "sw_mu" in scalers:
                model.sw_mu.fill_(float(scalers["sw_mu"]))
            if "sw_sigma" in scalers:
                model.sw_sigma.fill_(max(float(scalers["sw_sigma"]), 1e-6))
    CK.load_checkpoint(ckpt, model=model, map_location=device)
    model.eval()
    if device != "cpu":
        model.to(device)
    return model


def predict_x(model, X, batch_size: int = 65536, device: str = "cpu") -> dict[str, Any]:
    """对 (N,32) 原始 F1 特征做推理（标准化由调用方先做）。"""
    require("torch")
    import numpy as np
    import torch

    out_chunks: dict[str, list] = {}
    with torch.no_grad():
        for a in range(0, int(X.shape[0]), int(batch_size)):
            xb = torch.from_numpy(np.ascontiguousarray(X[a:a + batch_size])).to(device)
            out = model(xb)
            for k in ("por", "perm_z", "sw", "q_atom", "q_joint"):
                out_chunks.setdefault(k, []).append(out[k].float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out_chunks.items()}


def predict_wells(model, manifest: Manifest, cache_root: str | Path,
                  wells: Sequence[str], split: str = "test",
                  batch_size: int = 65536, device: str = "cpu") -> dict[str, Any]:
    """逐井推理并解码成标签尺度。

    返回 `{well_id: {"depth": (n,), "pred": (n,3)}}`（按井保存，避免全测试集常驻内存）。
    """
    require("torch")
    import numpy as np

    scaler = manifest.row_scaler
    tau = manifest.tau_atom
    per_well: dict[str, Any] = {}
    for w in wells:
        sh = D.read_well_shard(cache_root, w, split)
        X_raw = F.build_row_features(sh["inputs"], sh["missing"], sh["depth"])
        X = scaler.transform(X_raw)
        out = predict_x(model, X, batch_size=batch_size, device=device)
        cont = M.decode_continuous(out)
        pred = M.atom_gate(cont, out["q_atom"], tau) if tau is not None else cont
        per_well[w] = {"depth": np.asarray(sh["depth"], dtype="float64"), "pred": pred}
    return per_well


def build_payload(per_well: dict[str, Any], model_name: str = "v4-E1-pd0",
                  version: str = "1.0", model_id: str = "") -> dict[str, Any]:
    """组装官方提交载荷（`resultData` 逐井逐行，深度 1 位小数）。"""
    result_data = []
    for well_id in sorted(per_well):
        r = per_well[well_id]
        preds = []
        for d, p in zip(list(r["depth"]), list(r["pred"])):
            preds.append({
                "depth": round(float(d), C.DEPTH_DECIMALS),
                "POR": float(p[0]),
                "PERM": float(p[1]),
                "SW": float(p[2]),
            })
        result_data.append({"logId": well_id, "predictions": preds})
    return {"modelId": model_id, "modelName": model_name, "version": version,
            "resultData": result_data}


def write_payload(payload: dict[str, Any], out: str | Path) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def validate(payload: dict[str, Any], test_dir: str | Path,
             expected_rows: int | None = C.EXPECTED_N_TEST_ROWS,
             expected_wells: int | None = C.EXPECTED_N_TEST_WELLS) -> dict[str, Any]:
    """提交契约 + 深度对齐（与 `predict.py` 同一套校验，返回可 JSON 化的摘要）。"""
    from . import contract as CT
    res = CT.validate_payload(payload, test_dir=test_dir, expected_rows=expected_rows,
                              expected_wells=expected_wells)
    align = CT.depth_alignment_report(
        {it["logId"]: [p["depth"] for p in it["predictions"]] for it in payload["resultData"]},
        test_dir)
    bad = {k: v for k, v in align.items() if not v.get("ok")}
    return {"contract_ok": bool(res.ok), "errors": list(res.errors),
            "warnings": list(res.warnings), "stats": dict(res.stats),
            "depth_alignment_ok": not bad, "bad_wells": list(bad)}


def summary_rows(per_well: dict[str, Any]) -> dict[str, Any]:
    """提交前的粗检（行数/井数/取值域），供日志与 Gate 引用。"""
    import numpy as np
    preds = np.concatenate([np.asarray(v["pred"], dtype="float64") for v in per_well.values()],
                           axis=0) if per_well else np.zeros((0, 3))
    if preds.size == 0:
        return {"n_wells": 0, "n_rows": 0}
    return {
        "n_wells": len(per_well),
        "n_rows": int(preds.shape[0]),
        "por": {"min": float(preds[:, 0].min()), "max": float(preds[:, 0].max())},
        "perm": {"min": float(preds[:, 1].min()), "max": float(preds[:, 1].max()),
                 "nonpositive": int((preds[:, 1] <= 0).sum())},
        "sw": {"min": float(preds[:, 2].min()), "max": float(preds[:, 2].max()),
               "below_1": int((preds[:, 2] < 1.0).sum())},
    }
