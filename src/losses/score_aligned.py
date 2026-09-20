"""评分对齐损失 + 原子门辅助损失（与 `rules.md` §7.3 同构，可微）。

依据 `资料库/12` §2.2–2.4：
  1. Charbonnier 平滑绝对值：`|x| ≈ sqrt(x²+α²) − α`（消除 |·| 在 0 处的尖峰，梯度有界于 [-1,1]）；
  2. softplus 平滑截断：`min(x,1) ≈ x − softplus(x−1; β)`；
  于是 `max(0, 1−ℓ) ≈ 1 − ℓ + softplus(ℓ−1)`，β→∞ 时逐点等于官方得分，且处处可微；
  3. 四段式（R3）：`L = L_align + λ1·L_aux + λ_joint·L_joint + λ_atom·L_atom`
     （`L_joint` = 旧的占位/联合项，`L_atom` = 新增的三目标原子 BCE）。

**重要纪律**（`资料库/12` §2.4 提示、§2.3 末）：
  - 损失只用于反向传播；**模型选择/早停一律用真实 `score.py` 分数**；
  - PERM 一律在 log10 空间，网络输出 z，`PERM = 10^z`；用 tanh 夹到 [-6,6]，**不用 ReLU**；
  - `eps` 取 1e-3（不是 1e-6），显著稳定梯度；
  - **SW 是单一标签尺度（百分数）**，`aux_loss` 用训练折稳健尺度 `s_sw` 归一化，
    以抵消 99.9 量级对 POR/PERM 的支配；**绝不把 SW 裁剪到 [0,1]**。

掩码语义：`mask=0` 的行（缺测）不参与任何损失；**占位行 mask=1 且照常参与回归监督**
（占位常量本身就是要预测的正确目标）。

尺度归一化的纪律（R3）
----------------------
`aux_loss` 的 `s_por`/`s_sw` 是**训练折有效标签**的稳健尺度（IQR/1.349，IQR=0 时退化为 std），
只允许用训练折 fit（见 `features.basic.fit_target_scalers`），并**必须写入 checkpoint manifest /
scaler JSON**，推理期解码需要同一对 (mu, sigma) 反归一化。
"""
from __future__ import annotations

from .. import constants as C
from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


# 兼容旧名：占位/联合 BCE
def smooth_abs(x, alpha: float = 1e-3):
    """Charbonnier 平滑绝对值，α 控制平滑半径（对标准化量纲取 1e-3~1e-2）。"""
    require("torch")
    return torch.sqrt(x * x + alpha * alpha) - alpha


def soft_min1(x, beta: float = 20.0):
    """平滑 min(x, 1)：x − softplus(x−1; β)。β→+∞ 时等于 min(x,1)。"""
    require("torch")
    return x - F.softplus(x - 1.0, beta=beta)


def align_score_relative(y, yhat, delta: float, eps: float = 1e-3,
                         alpha: float = 1e-3, beta: float = 20.0):
    """POR / SW 的对齐**得分**（越大越好）：s ≈ max(0, 1 − |ŷ−y|/(δ(|y|+ε)))。"""
    require("torch")
    denom = delta * (y.abs() + eps)
    ell = smooth_abs(yhat - y, alpha) / denom
    return 1.0 - ell + F.softplus(ell - 1.0, beta=beta)


def align_score_log(z, zhat, alpha: float = 1e-3, beta: float = 20.0,
                    eps: float = 1e-3):
    """PERM 的对齐**得分**（log10 空间，与官方严格同构）。

    官方：`s = max(0, 1 − |log10(max(ŷ/y, ε))|)`
      - 当 `ŷ/y ≥ ε` 时，`log10(max(ŷ/y,ε)) = ẑ − z`；
      - 当 `ŷ/y < ε`（严重低估）时，官方把比值**截断在 ε**，
        误差恒为 `log10(1/ε)`（ε=1e-3 → 3.0），不再随低估程度增长。
    **R2-H3 修复**（保留）：此前直接用 `|ẑ − z|`，在 `ŷ/y < ε` 区域比官方惩罚更重
    （例如 `ẑ−z=−5` 时官方误差 3.0、旧实现 5.0），梯度方向与官方评分不一致。
    这里对 **log 空间的差值**做同样的下截断：`d = max(ẑ − z, log10(ε))`。
    """
    require("torch")
    import math as _math
    d = zhat - z
    d = torch.maximum(d, torch.full_like(d, _math.log10(eps)))
    ell = smooth_abs(d, alpha)
    return 1.0 - ell + F.softplus(ell - 1.0, beta=beta)


def masked_mean(x, mask=None):
    """掩码均值。**R2-H2 修复**：先把被屏蔽位置的 NaN/Inf 清零再乘掩码，
    否则 `NaN * 0 = NaN` 会让整个 batch 的 loss 变成 NaN。"""
    require("torch")
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return (x_safe * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------- 切片权重工具
def _as_slice_weight(slice_weight, ref, n_targets: int = 3, min_weight: float = 1e-3):
    """把 `slice_weight` 规范为 `(B, n_targets)` 张量，并夹到 `>= min_weight`。

    接受 `None` / `(B,3)` 张量 / 三个 `(B,)` 张量组成的序列。
    **权重永远不得为 0**：连续头是原子头误判时的 fallback，权重为 0 会让该切片
    完全失去连续监督；因此统一夹到 `min_weight`（默认 1e-3）。
    """
    if slice_weight is None:
        return None
    if isinstance(slice_weight, (tuple, list)):
        ws = [torch.as_tensor(w, dtype=ref.dtype, device=ref.device).reshape(-1)
              for w in slice_weight]
        if len(ws) != n_targets:
            raise ValueError(f"slice_weight must have {n_targets} entries, got {len(ws)}")
        W = torch.stack(ws, dim=1)
    else:
        W = torch.as_tensor(slice_weight, dtype=ref.dtype, device=ref.device)
        if W.dim() == 1:
            W = W.unsqueeze(1).expand(-1, n_targets)
    if W.dim() != 2 or W.shape[1] != n_targets:
        raise ValueError(f"slice_weight must be (B,{n_targets}) or {n_targets}×(B,), got {tuple(W.shape)}")
    return W.clamp_min(float(min_weight))


# ---------------------------------------------------------------- 边界聚焦（E7 消融）
def boundary_focus_weight(r, kappa: float = 0.0, sigma: float = 0.25):
    """边界聚焦权重 `w = 1 + κ·exp(−(r−1)²/(2σ²))`。

    `r` 是到"容差边界"的归一化距离：
      - POR/SW : `r = |pred − y| / (δ·(|y| + eps))`（δ=0.08/0.05），`r≈1` 即刚好在容差边界；
      - PERM   : `r = |ẑ − z|`（log10 空间，官方容差就是 1 个数量级）。
    `κ == 0` 时恒返回全 1（默认关闭；仅在 E7 消融里显式打开）。
    """
    require("torch")
    if kappa == 0.0:
        return torch.ones_like(r)
    return 1.0 + kappa * torch.exp(-((r - 1.0) ** 2) / (2.0 * sigma * sigma))


# ---------------------------------------------------------------- 对齐损失
def aligned_loss(y_por, p_por, z_perm, zhat_perm, y_sw, p_sw, mask=None,
                 w_por: float = 0.30, w_perm: float = 0.35, w_sw: float = 0.35,
                 eps: float = 1e-3, alpha: float = 1e-3, beta: float = 20.0,
                 boundary_kappa: float = 0.0, boundary_sigma: float = 0.25,
                 y_atom=None, include_atom_mask: bool = True):
    """三目标加权对齐损失（返回标量，越小越好）。

    mask : (B, 3) float，1=该目标参与监督（缺测为 0）

    `boundary_kappa > 0` 时启用边界聚焦（E7 消融，默认 OFF）：对逐元素得分布乘
    `boundary_focus_weight(r)`，使恰好落在容差边界附近的行获得更大权重。
    最简诚实的原子行处理：给定 `y_atom`(B,3) 且 `include_atom_mask=True` 时，
    原子行的聚焦权重回落到 1.0（它们由原子头负责，连续头只是 fallback）；
    `include_atom_mask=False` 则忽略 `y_atom`，原子行与普通行一视同仁。
    缺测行始终由 `mask` 排除。
    """
    require("torch")
    m_por = None if mask is None else mask[:, 0]
    m_perm = None if mask is None else mask[:, 1]
    m_sw = None if mask is None else mask[:, 2]

    s_por = align_score_relative(y_por, p_por, 0.08, eps, alpha, beta)
    s_perm = align_score_log(z_perm, zhat_perm, alpha, beta)
    s_sw = align_score_relative(y_sw, p_sw, 0.05, eps, alpha, beta)

    atom = None
    if boundary_kappa and y_atom is not None and include_atom_mask:
        atom = torch.as_tensor(y_atom, dtype=s_por.dtype, device=s_por.device)

    if boundary_kappa:

        def _focus(r, t: int):
            wf = boundary_focus_weight(r, boundary_kappa, boundary_sigma)
            if atom is not None:
                non_atom = (atom[:, t] == 0).to(wf.dtype)
                wf = 1.0 + (wf - 1.0) * non_atom
            return wf

        r_por = (p_por - y_por).abs() / (0.08 * (y_por.abs() + eps))
        r_sw = (p_sw - y_sw).abs() / (0.05 * (y_sw.abs() + eps))
        r_perm = (zhat_perm - z_perm).abs()
        s_por = s_por * _focus(r_por, 0)
        s_perm = s_perm * _focus(r_perm, 1)
        s_sw = s_sw * _focus(r_sw, 2)

    return -(
        w_por * masked_mean(s_por, m_por)
        + w_perm * masked_mean(s_perm, m_perm)
        + w_sw * masked_mean(s_sw, m_sw)
    )


def aux_loss(y_por, p_por, z_perm, zhat_perm, y_sw, p_sw, mask=None,
             s_por: float = 11.34, s_sw: float = 20.0, huber_beta: float = 1.0,
             slice_weight=None):
    """变换空间稠密损失（早期梯度来源）——**逐目标尺度归一化**（R3）。

        aux_por  = smooth_l1((p_por − y_por) / s_por)
        aux_perm = smooth_l1(zhat_perm − z_perm)          # 已在 log10 空间
        aux_sw   = smooth_l1((p_sw − y_sw) / s_sw)
        L_aux    = 0.30·aux_por + 0.35·aux_perm + 0.35·aux_sw

    归一化的目的：SW 的量级是 99.9，若不除 `s_sw` 会直接支配 POR/PERM 的梯度。
    `s_por`/`s_sw` 是**训练折有效标签**的稳健尺度（IQR/1.349，退化时用 std），
    必须与 (sw_mu, sw_sigma) 一起写入 checkpoint manifest / scaler JSON。

    `slice_weight`（可选，proposal §4 C3）：`(B,3)` 或三个 `(B,)`，逐元素乘在
    归一化后的连续损失上；`None` 等价于全 1。预期用法：
      - 联合占位行在连续头损失上取 0.1–0.3 权重（硬切换后连续头不再服务于它们）；
      - 非联合的原子行保留中等权重（原子头误判时连续头是 fallback）。
    **权重永远不得为 0**（统一夹到 `min_weight=1e-3`，见 `_as_slice_weight`）。
    """
    require("torch")

    def sl1(a, b, m, sw_col: int, scale: float = 1.0):
        # 归一化必须在 smooth_l1 **内部**（否则 Huber 的 β 边界会破坏尺度不变性）：
        #   smooth_l1((a−b)/s)  ≠  smooth_l1(a−b)/s
        s = max(float(scale), 1e-9)
        l = F.smooth_l1_loss(a / s, b / s, beta=huber_beta, reduction="none")
        if slice_weight is not None:
            W = _as_slice_weight(slice_weight, l, 3)
            l = l * W[:, sw_col]
        return masked_mean(l, m)

    m_por = None if mask is None else mask[:, 0]
    m_perm = None if mask is None else mask[:, 1]
    m_sw = None if mask is None else mask[:, 2]
    return (
        0.30 * sl1(p_por, y_por, m_por, 0, s_por)
        + 0.35 * sl1(zhat_perm, z_perm, m_perm, 1, 1.0)
        + 0.35 * sl1(p_sw, y_sw, m_sw, 2, s_sw)
    )


# ---------------------------------------------------------------- 占位/联合 + 原子 BCE
def placeholder_bce(y_ph, q_logit, mask=None):
    """联合常量占位状态的 BCE（mask 取"任一行非缺测"即可）。"""
    require("torch")
    l = F.binary_cross_entropy_with_logits(q_logit, y_ph, reduction="none")
    return masked_mean(l, mask)


def joint_bce(q_joint_logit, y_joint, mask=None):
    """联合占位状态 BCE（`joint_bce` 为 `placeholder_bce` 的语义化新名）。"""
    require("torch")
    l = F.binary_cross_entropy_with_logits(q_joint_logit, y_joint, reduction="none")
    return masked_mean(l, mask)


def atom_bce(q_atom_logit, y_atom, mask, pos_weight=None, alpha_nonjoint: float = 1.0,
             y_joint=None, return_parts: bool = False):
    """三目标原子 BCE（R3）。

        w_t = 1.0 + alpha_nonjoint · y_atom[:,t] · (1 − y_joint)
        L_t = Σ(BCE(q_t, y_atom_t) · w_t · mask_t) / clamp_min(Σ(w_t · mask_t), 1.0)

    即：**非联合的原子行**获得 `1+alpha_nonjoint` 倍权重（它们是最容易被漏掉的
    "单目标占位"行）；联合原子行保持 1.0 倍；非原子行权重恒为 1.0。
    `y_joint is None` 时不做该区分（全部按非联合处理）。

    `pos_weight`：`(3,)` 或三个标量，逐目标正类权重（传给 `BCEWithLogits`）。
    `return_parts=True` 时返回 `{"por","perm","sw","atom"}` 各目标项的 dict（便于日志）。
    """
    require("torch")
    q = torch.as_tensor(q_atom_logit)
    y = torch.as_tensor(y_atom, dtype=q.dtype, device=q.device)
    if y.dim() != 2 or y.shape[1] != 3:
        raise ValueError(f"y_atom must be (B,3), got {tuple(y.shape)}")
    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=q.dtype, device=q.device).reshape(-1)
        if pw.numel() == 1:
            pw = pw.expand(3)

    bce = F.binary_cross_entropy_with_logits(q, y, reduction="none", pos_weight=pw)
    if y_joint is None:
        nonjoint = torch.ones_like(y)
    else:
        yj = torch.as_tensor(y_joint, dtype=q.dtype, device=q.device).reshape(-1, 1)
        nonjoint = (1.0 - yj).expand_as(y)
    w = 1.0 + float(alpha_nonjoint) * y * nonjoint

    m = None if mask is None else torch.as_tensor(mask, dtype=q.dtype, device=q.device)
    parts = {}
    for t, name in enumerate(C.TARGETS):
        wt = w[:, t] if m is None else w[:, t] * m[:, t]
        bce_t = torch.nan_to_num(bce[:, t], nan=0.0, posinf=0.0, neginf=0.0)
        parts[name] = (bce_t * wt).sum() / wt.sum().clamp_min(1.0)
    mean = (parts["POR"] + parts["PERM"] + parts["SW"]) / 3.0
    if return_parts:
        parts["atom"] = mean
        return parts
    return mean


def _require_logit(out: dict, logit_key: str, prob_key: str):
    """取 logits，**拒绝**把概率当 logits 用（R4-B2）。

    `RowMLP.forward` 同时给出 `q_*`（概率，门控用）与 `q_*_logit`（logits，损失用）。
    只给概率却要算 BCE 时，若静默接受会让 BCEWithLogits 在 [0,1] 输入上错训，
    所以这里显式报错并提示正确的键名。
    """
    if logit_key in out and out[logit_key] is not None:
        return out[logit_key]
    if prob_key in out and out[prob_key] is not None:
        raise ValueError(
            f"total_loss: 只找到概率键 out[{prob_key!r}]，缺少 logit 键 out[{logit_key!r}]。"
            "BCEWithLogits 需要 logits（概率会静默错训）；请改用模型 forward 返回的 "
            f"{logit_key!r}（RowMLP 同时给出两者）。"
        )
    raise KeyError(f"total_loss: out 既没有 {logit_key!r} 也没有 {prob_key!r}")


# ---------------------------------------------------------------- 组合
def total_loss(out: dict, batch: dict, lam1: float = 1.0, lam2: float | None = None,
               lam_atom: float = 0.5, lam_joint: float = 0.2,
               use_align: bool = True, use_aux: bool = True, use_ph: bool = True,
               use_atom: bool | None = None, use_joint: bool | None = None,
               slice_weight=None, **kw) -> tuple:
    """组合损失（R3 四段式）。

        L = L_align + λ1·L_aux + λ_joint·L_joint + λ_atom·L_atom

    out  : 模型输出 `{'por','perm_z','sw','q_atom','q_joint',...}`。
           **R4-B2**：BCE 项一律取 `q_joint_logit` / `q_atom_logit`（logits）；
           只给概率键时会抛出清晰错误，而不是把概率当 logits 静默错训。
    batch: {'por','perm_z','sw','mask'} + 新键 {'y_atom'(B,3), 'y_joint'(B,)}
           旧键 'y_ph'(B,) 仍作为联合标签的 fallback。
    返回 (total, parts_dict_of_floats)

    `s_por` / `s_sw`（可选 kw）：`L_aux` 的逐目标稳健尺度，**必须来自训练折**
    （`features.basic.fit_target_scalers`，见 E1/P1 §5 步 4）；未给出时回落到
    `(11.34, 20.0)`，仅用于兼容旧调用点。`huber_beta` 同理由这里透传。

    兼容性：
      - `use_ph` 映射到联合项 `L_joint`（`use_joint` 显式优先）；
      - `lam2` 为旧名，显式传入时覆盖 `lam_joint`；
      - `use_atom=None` 时**自动**按 batch 是否含 `y_atom` 决定（旧 batch 不报错）；
        显式 `use_atom=True` 而 batch 缺 `y_atom` 时才抛清晰错误。
    """
    require("torch")
    mask = batch.get("mask")
    joint_on = use_joint if use_joint is not None else use_ph
    if use_atom is None:
        atom_on = "y_atom" in batch
    else:
        atom_on = bool(use_atom)
    w_joint = float(lam_joint) if lam2 is None else float(lam2)
    # 原子项专属 kwargs 先取出，避免被透传给 aligned_loss 造成 TypeError
    pos_weight = kw.pop("pos_weight", None)
    alpha_nonjoint = kw.pop("alpha_nonjoint", 1.0)
    # E1/P1 步骤 4（E1-R3 缺口修复）：`L_aux` 的**逐目标稳健尺度**必须可传入。
    # 此前 `total_loss` 直接丢弃这两个量，`aux_loss` 永远用默认 (11.34, 20.0)，
    # 于是"按训练折尺度归一化"在组合损失里静默失效（SW 的 99.9 重新支配梯度）。
    s_por = kw.pop("s_por", 11.34)
    s_sw = kw.pop("s_sw", 20.0)
    huber_beta = kw.pop("huber_beta", 1.0)

    if mask is None:
        mask = torch.ones_like(torch.as_tensor(out["por"]))[:, None].expand(-1, 3)

    parts: dict[str, "torch.Tensor"] = {}
    total = None

    def _add(key: str, value, weight: float):
        nonlocal total
        parts[key] = value
        total = weight * value if total is None else total + weight * value

    if use_align:
        _add("align", aligned_loss(
            batch["por"], out["por"], batch["perm_z"], out["perm_z"],
            batch["sw"], out["sw"], mask, y_atom=batch.get("y_atom"), **kw), 1.0)
    if use_aux:
        _add("aux", aux_loss(
            batch["por"], out["por"], batch["perm_z"], out["perm_z"],
            batch["sw"], out["sw"], mask, s_por=s_por, s_sw=s_sw,
            huber_beta=huber_beta, slice_weight=slice_weight), lam1)
    if joint_on:
        y_joint = batch.get("y_joint")
        if y_joint is None:
            y_joint = batch.get("y_ph")
        if y_joint is None:
            raise ValueError("total_loss: joint term enabled but batch has neither 'y_joint' nor 'y_ph'")
        joint_mask = (mask.sum(dim=1) > 0).to(mask.dtype)
        jl = joint_bce(_require_logit(out, "q_joint_logit", "q_joint"), y_joint, joint_mask)
        _add("joint", jl, w_joint)
        parts["ph"] = jl            # 旧键别名（日志兼容）
    if atom_on:
        y_atom = batch.get("y_atom")
        if y_atom is None:
            raise ValueError("total_loss: atom term enabled but batch is missing 'y_atom' (B,3)")
        ap = atom_bce(_require_logit(out, "q_atom_logit", "q_atom"), y_atom, mask,
                      pos_weight=pos_weight,
                      alpha_nonjoint=alpha_nonjoint,
                      y_joint=batch.get("y_joint"), return_parts=True)
        _add("atom", ap["atom"], lam_atom)
        parts["atom_por"] = ap["POR"]
        parts["atom_perm"] = ap["PERM"]
        parts["atom_sw"] = ap["SW"]

    if total is None:
        raise ValueError("at least one loss term must be enabled")
    return total, {
        k: float(v.detach()) for k, v in parts.items() if hasattr(v, "detach")
    }
