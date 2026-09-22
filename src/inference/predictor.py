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
from ..features import groups as GRP
from ..features import physics as PH
from ..portability import HAS_TORCH, require
from ..training import checkpoint as CK
from ..training import loop as L
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
    def arch(self) -> str:
        m = dict(self.raw.get("model", {}))
        return str(self.raw.get("arch") or m.get("arch") or "RowMLP")

    @property
    def model_kwargs(self) -> dict[str, Any]:
        m = dict(self.raw.get("model", {}))
        return {"n_features": int(self.n_features),
                "hidden": int(m.get("hidden", 256)),
                "layers": int(m.get("layers", 2)),
                "dropout": float(m.get("dropout", 0.0))}

    @property
    def arch_kwargs(self) -> dict[str, Any]:
        """序列/MoE 结构参数；从 `arch_kwargs` 与 `model` 两个来源合并。"""
        kw = dict(self.raw.get("arch_kwargs") or {})
        m = dict(self.raw.get("model") or {})
        for k, v in m.items():
            if k in ("arch", "n_features"):
                continue
            kw.setdefault(k, v)
        return kw

    @property
    def feature_names(self) -> list[str]:
        names = self.raw.get("feature_names")
        if names is None:
            try:
                names = self.row_scaler.names
            except Exception:
                names = None
        return [str(x) for x in (names or [])]

    @property
    def feature_spec(self):
        d = self.raw.get("feature_spec") or self.raw.get("spec")
        if not isinstance(d, dict):
            return None
        try:
            return GRP.FeatureSpec.from_dict(d)
        except Exception:
            return None

    @property
    def physics_params(self):
        d = self.raw.get("physics_params") or self.raw.get("phys_params")
        if not isinstance(d, dict):
            return None
        try:
            return PH.PhysicsParams.from_dict(d)
        except Exception:
            return None

    @property
    def tau_atom(self):
        return self.raw.get("tau_atom")

    @property
    def n_features(self) -> int:
        m = dict(self.raw.get("model", {}))
        n = m.get("n_features") or self.raw.get("n_features")
        if n is not None:
            return int(n)
        try:
            return int(len(self.row_scaler.names))
        except Exception:
            return int(F.N_FEATURES)


def load_manifest(ckpt: str | Path) -> Manifest:
    p = Path(ckpt)
    return Manifest(path=p, raw=CK.read_manifest(p))


def _target_scaler_attrs(model):
    """返回模型上需要按 manifest 覆盖的连续头标尺 tensor。"""
    out = []
    for attr in ("por_max", "sw_mu", "sw_sigma"):
        v = getattr(model, attr, None)
        if v is not None and hasattr(v, "fill_"):
            out.append((attr, v))
    head = getattr(model, "head", None)
    if head is not None:
        for attr in ("por_max", "sw_mu", "sw_sigma"):
            v = getattr(head, attr, None)
            if v is not None and hasattr(v, "fill_"):
                out.append((attr, v))
    return out


def _apply_target_scalers(model, scalers: dict) -> None:
    if not scalers:
        return
    import torch
    with torch.no_grad():
        for attr, buf in _target_scaler_attrs(model):
            if attr not in scalers:
                continue
            val = float(scalers[attr])
            if attr == "sw_sigma":
                val = max(val, 1e-6)
            buf.fill_(val)


def _build_model_from_manifest(man: Manifest):
    """按 manifest 的 arch/arch_kwargs 构建推理模型（RowMLP / 序列 / MMoE / 独立三模型）。"""
    require("torch")
    arch = str(man.arch).lower()
    n_features = int(man.n_features)
    if arch in ("rowmlp", "row_mlp", "row-mlp", "mlp"):
        from ..models.row_mlp import build_model
        return build_model(init_stats=None, **man.model_kwargs)
    if arch in ("unet", "tcn", "patchtf"):
        from ..training.seq_loop import build_seq_model
        kw = {k: v for k, v in man.arch_kwargs.items() if k != "n_features"}
        return build_seq_model(arch, n_features, **kw)
    if arch in ("mmoe", "moe"):
        from ..models.mmoe import MMoE
        kw = man.arch_kwargs
        return MMoE(
            n_features,
            hidden=int(kw.get("hidden", 128)),
            n_experts=int(kw.get("n_experts", 4)),
            dropout=float(kw.get("dropout", 0.1)),
            gate_temp=float(kw.get("gate_temp", 1.0)),
            expert_width=(None if kw.get("expert_width") is None
                          else int(kw.get("expert_width"))),
            perm_log_abs=float(kw.get("perm_log_abs", 6.0)),
        )
    if arch in ("independent", "independentheads", "independent_heads"):
        from ..models.mmoe import IndependentHeads
        kw = man.arch_kwargs
        return IndependentHeads(
            n_features,
            hidden=int(kw.get("hidden", 256)),
            dropout=float(kw.get("dropout", 0.1)),
            perm_log_abs=float(kw.get("perm_log_abs", 6.0)),
        )
    raise ValueError(f"[predictor] 不支持的模型 arch={man.arch!r}")


def load_model(ckpt: str | Path, manifest: Manifest | None = None, device: str = "cpu"):
    """按 manifest 重建任意已支持结构并载入权重（bf16 → float32）。"""
    require("torch")
    man = manifest or load_manifest(ckpt)
    model = _build_model_from_manifest(man)
    CK.load_checkpoint(ckpt, model=model, map_location=device)
    _apply_target_scalers(model, man.target_scalers)
    model.eval()
    if str(device) != "cpu":
        model.to(device)
    return model


def cpu_inference_supported(manifest: Manifest) -> bool:
    """当前 CPU 提交入口是否支持该 manifest 的结构。"""
    arch = str(manifest.arch).lower()
    return arch in ("rowmlp", "row_mlp", "row-mlp", "mlp",
                    "unet", "tcn", "patchtf",
                    "mmoe", "moe",
                    "independent", "independentheads", "independent_heads")



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


def build_inference_features(inputs, missing, depth, manifest: Manifest):
    """按 manifest 的 FeatureSpec 构建原始特征矩阵（未标准化）。"""
    spec = manifest.feature_spec
    if spec is not None and tuple(spec.groups) != ("F1",):
        if "phys" in tuple(spec.groups):
            phys = manifest.physics_params
            if phys is None:
                raise ValueError("[predictor] manifest 的 feature_spec 含 phys，"
                                 "但缺少 physics_params，无法安全重建 F2 输入。")
        else:
            phys = None
        shard = {"inputs": inputs, "missing": missing, "depth": depth}
        X, _names = GRP.build_matrix(shard, spec, phys_params=phys)
        return X
    return F.build_row_features(inputs, missing, depth)


def predict_manifest(model, manifest: Manifest, X,
                     batch_size: int = 65536, device: str = "cpu",
                     chunk: int | None = None, overlap: int | None = None
                     ) -> dict[str, Any]:
    """统一推理入口：RowMLP/MMoE 走行级 batch；UNet/TCN/PatchTF 走分块 seq2seq。"""
    require("torch")
    import torch
    device = device if isinstance(device, torch.device) else torch.device(device)
    arch = str(manifest.arch).lower()
    if arch in ("unet", "tcn", "patchtf"):
        require("torch")
        import numpy as np
        from ..training import seq_loop as SL
        n = int(np.asarray(X).shape[0])
        # PatchTF 至少需要覆盖一个 patch；其余默认沿用 E3/E4 的 chunk/overlap。
        patch_len = int(manifest.arch_kwargs.get("patch_len", 1) or 1)
        c = int(chunk if chunk is not None else max(1024, patch_len))
        c = min(c, max(n, 1))
        ov = int(overlap if overlap is not None else min(128, max(c // 8, 0)))
        if ov >= c:
            ov = max(c // 8, 0)
        cfg = L.TrainConfig(device=device, amp_dtype="fp32")
        opt = SL.SeqOptions(spec=manifest.feature_spec, chunk=c, overlap=ov,
                           batch_chunks=1, weight_kind="triangular", arch=arch)
        return SL.predict_well_chunked(model, np.asarray(X, dtype="float32"),
                                       cfg, opt, device)
    return predict_x(model, X, batch_size=batch_size, device=device)


def predict_well_components(model, manifest: Manifest, shard: dict,
                            device: str = "cpu", decode_cfg=None,
                            batch_size: int = 65536) -> tuple:
    """单口井推理的**组件输出**：`(depth, cont, q_atom)`。

    供折集成使用：调用方先融合多折的 `cont`/`q_atom`，再做一次原子硬切换。
    这与项目“原子切换必须在融合之后”的纪律一致（审查 H3）。
    """
    import numpy as np
    X_raw = build_inference_features(shard["inputs"], shard["missing"], shard["depth"],
                                     manifest)
    X = manifest.row_scaler.transform(X_raw)
    out = predict_manifest(model, manifest, X, batch_size=batch_size, device=device)
    cont = M.decode_continuous(out)
    if decode_cfg is not None:
        from . import decode as DEC
        cont = DEC.apply_decode_config(cont, decode_cfg, q_atom=out["q_atom"])
    return (np.asarray(shard["depth"], dtype="float64"),
            np.asarray(cont, dtype="float64"),
            np.asarray(out["q_atom"], dtype="float64"))


def predict_well(model, manifest: Manifest, shard: dict,
                 device: str = "cpu", decode_cfg=None,
                 batch_size: int = 65536) -> tuple:
    """对单口井的 raw shard 做推理并解码为标签尺度 `(depth, pred)`。"""
    import numpy as np
    depth, cont, q_atom = predict_well_components(
        model, manifest, shard, device=device, decode_cfg=decode_cfg,
        batch_size=batch_size)
    tau = manifest.tau_atom
    pred = M.atom_gate(cont, q_atom, tau) if tau is not None else cont
    return depth, np.asarray(pred, dtype="float64")


def predict_wells(model, manifest: Manifest, cache_root: str | Path,
                  wells: Sequence[str], split: str = "test",
                  batch_size: int = 65536, device: str = "cpu",
                  decode_cfg=None) -> dict[str, Any]:
    """逐井推理并解码成标签尺度。

    返回 `{well_id: {"depth": (n,), "pred": (n,3)}}`（按井保存，避免全测试集常驻内存）。
    """
    per_well: dict[str, Any] = {}
    for w in wells:
        sh = D.read_well_shard(cache_root, w, split)
        depth, pred = predict_well(model, manifest, sh, device=device,
                                   decode_cfg=decode_cfg, batch_size=batch_size)
        per_well[w] = {"depth": depth, "pred": pred}
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
