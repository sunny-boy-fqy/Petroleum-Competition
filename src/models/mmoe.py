"""E8/P0 MMoE（E 个专家 + **每任务独立门控** softmax 加权）：与 `SeqHead` **同键**输出。

要点
----
1. **同键**：输出键与 `models.heads.SeqHead` 完全一致（`por/perm_z/sw/q_atom/q_joint/
   *_logit/ph_logit`），因此损失、解码、τ 选择、指标、契约**一行不改**。
2. **参数量可比**（E8/P0 §6/§9 的硬要求）：专家银行总宽 ≈ 硬共享主干宽，
   即 `expert_width = hidden // n_experts`；`param_comparison()` 给出比值与告警，
   禁止"参数量差异巨大还比结构"。
3. **门控塌陷必须被看见**：`gate_report()` 报告每个分支的门控熵 / 最大使用率 / 是否塌陷，
   并提供可选负载均衡损失（`load_balance_loss`）。
4. 梯度冲突（§5 步 3）：`task_gradient_cosines()` 对共享参数逐任务反传，给出任务间余弦。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..portability import HAS_TORCH, require

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

# 分支定义：3 个连续目标 + 原子头(3 输出) + 联合头(1 输出)
BRANCHES: tuple[str, ...] = ("por", "perm", "sw", "atom", "joint")
BRANCH_OUT: dict[str, int] = {"por": 1, "perm": 1, "sw": 1, "atom": 3, "joint": 1}


if HAS_TORCH:

    class MMoE(nn.Module):
        """逐行 MMoE：`x (·, d_in)` → 同 `SeqHead` 的多任务输出。

        `x` 支持 `(N, d_in)` 与 `(B, L, d_in)`（逐点作用，输出保持前缀形状）。
        """

        def __init__(self, d_in: int, hidden: int = 128, n_experts: int = 4,
                     dropout: float = 0.1, gate_temp: float = 1.0,
                     expert_width: int | None = None, perm_log_abs: float = 6.0,
                     por_max: float | None = None, sw_mu: float | None = None,
                     sw_sigma: float | None = None):
            super().__init__()
            from .. import constants as C

            if int(n_experts) < 1:
                raise ValueError(f"n_experts 必须 ≥1，got {n_experts}")
            self.d_in = int(d_in)
            self.hidden = int(hidden)
            self.n_experts = int(n_experts)
            self.gate_temp = float(gate_temp)
            if self.gate_temp <= 0:
                raise ValueError(f"gate_temp 必须 >0，got {gate_temp}")
            # 专家瓶颈窄化：ew=hidden//E ⇒ 专家银行总参数量 ≈ 单个宽专家（参数量可比）
            self.expert_width = int(expert_width) if expert_width is not None \
                else max(self.hidden // self.n_experts, 4)
            self.perm_log_abs = float(perm_log_abs)

            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(self.d_in, self.expert_width), nn.GELU(),
                              nn.Linear(self.expert_width, self.hidden), nn.GELU(),
                              nn.Dropout(dropout)) for _ in range(self.n_experts)])
            self.gates = nn.ModuleDict({
                b: nn.Linear(self.d_in, self.n_experts) for b in BRANCHES})
            self.towers = nn.ModuleDict({
                b: nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(self.hidden, BRANCH_OUT[b]))
                for b in BRANCHES})

            self.register_buffer("por_max", torch.tensor(float(
                por_max if por_max is not None else C.POR_MAX_BUFFER * C.POR_VALID_MAX)))
            self.register_buffer("sw_mu", torch.tensor(float(
                sw_mu if sw_mu is not None else C.SW_VALID_MEDIAN)))
            self.register_buffer("sw_sigma", torch.tensor(float(sw_sigma if sw_sigma is not None
                                                                else 20.0)))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=0.01)

        # ------------------------------------------------------------ 前向
        def _gates(self, x):
            """返回 `{branch: (gate_weights, expert_mix)}`；gate_weights 在最后一维。"""
            stacked = torch.stack([e(x) for e in self.experts], dim=-2)   # (·, E, w)
            out = {}
            for b in BRANCHES:
                logits = self.gates[b](x) / self.gate_temp
                g = torch.softmax(logits, dim=-1)                        # (·, E)
                out[b] = (g, torch.einsum("...e,...ew->...w", g, stacked))
            return out

        def forward(self, x):
            require("torch")
            if x.dim() not in (2, 3):
                raise ValueError(f"x 必须为 (N,d) 或 (B,L,d)，got {tuple(x.shape)}")
            mix = self._gates(x)
            por = self.por_max * torch.sigmoid(self.towers["por"](mix["por"][1]).squeeze(-1))
            perm_z = self.perm_log_abs * torch.tanh(
                self.towers["perm"](mix["perm"][1]).squeeze(-1))
            sw = self.sw_mu + self.sw_sigma * self.towers["sw"](mix["sw"][1]).squeeze(-1)
            q_atom_logit = self.towers["atom"](mix["atom"][1])
            q_joint_logit = self.towers["joint"](mix["joint"][1]).squeeze(-1)
            return {
                "por": por, "perm_z": perm_z, "sw": sw,
                "q_atom": torch.sigmoid(q_atom_logit),
                "q_joint": torch.sigmoid(q_joint_logit),
                "q_atom_logit": q_atom_logit,
                "q_joint_logit": q_joint_logit,
                "ph_logit": q_joint_logit,
            }

        # ------------------------------------------------------------ 诊断
        @torch.no_grad()
        def gate_report(self, x, collapse_entropy: float = 0.1) -> dict[str, Any]:
            """门控熵（nats，除以 log E 归一）/ 最大使用率 / 塌陷标记（§9 风险表）。"""
            require("torch")
            import math
            self.eval()
            g = self._gates(x)
            e = float(self.n_experts)
            norm = math.log(e) if e > 1 else 1.0
            rep: dict[str, Any] = {"n_experts": self.n_experts, "gate_temp": self.gate_temp,
                                   "collapse_entropy": float(collapse_entropy), "branches": {}}
            collapsed = []
            for b in BRANCHES:
                gw = g[b][0]
                ent = float(-(gw * torch.log(gw.clamp_min(1e-12))).sum(-1).mean()) / norm
                usage = [float(v) for v in gw.mean(dim=tuple(range(gw.dim() - 1)))]
                rep["branches"][b] = {"entropy_norm": ent, "usage": usage,
                                      "max_usage": max(usage) if usage else 0.0,
                                      "collapsed": bool(ent < collapse_entropy)}
                if rep["branches"][b]["collapsed"]:
                    collapsed.append(b)
            rep["collapsed_branches"] = collapsed
            rep["ok"] = not collapsed
            return rep

        def load_balance_loss(self, x):
            """可微负载均衡：`E·Σ_e f_e·P_e`（Switch/MoE 常用形式），越小越均衡。"""
            require("torch")
            g = self._gates(x)
            loss = x.new_zeros(())
            for b in BRANCHES:
                gw = g[b][0]
                flat = gw.reshape(-1, self.n_experts)
                f = (flat > (1.0 / self.n_experts)).to(flat.dtype).mean(0)
                p = flat.mean(0)
                loss = loss + self.n_experts * (f * p).sum()
            return loss / len(BRANCHES)

        @torch.no_grad()
        def init_from_stats(self, por_median: float = 11.34, por_max: float = 39.8,
                            sw_mu: float = 82.805, sw_sigma: float = 20.0,
                            perm_z_median: float = -0.08, joint_atom_rate: float = 0.667,
                            atom_rates: tuple = (0.6674, 0.6773, 0.7097)) -> None:
            """与 `SeqHead.init_from_stats` 同语义（只改塔的末层偏置 + 尺度 buffer）。"""
            import math as _m
            self.por_max.fill_(float(por_max))
            self.sw_mu.fill_(float(sw_mu))
            self.sw_sigma.fill_(max(float(sw_sigma), 1e-6))

            def _logit(p: float) -> float:
                q = min(max(float(p), 1e-6), 1.0 - 1e-6)
                return _m.log(q / (1.0 - q))

            self.towers["por"][-1].bias.fill_(_logit(float(por_median) / max(float(por_max), 1e-9)))
            z_ratio = float(perm_z_median) / max(self.perm_log_abs, 1e-9)
            self.towers["perm"][-1].bias.fill_(
                _m.atanh(min(max(z_ratio, -0.999999), 0.999999)))
            self.towers["sw"][-1].bias.fill_(0.0)
            self.towers["joint"][-1].bias.fill_(_logit(joint_atom_rate))
            self.towers["atom"][-1].bias.copy_(
                torch.tensor([_logit(r) for r in tuple(atom_rates)]))


def count_parameters(model) -> int:
    require("torch")
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def moe_pair(d_in: int, hidden: int = 128, n_experts: int = 4, **kwargs):
    """返回 `(hard_share, mmoe)`：硬共享 = `n_experts=1` 的同一实现（只改共享结构、总参数量对等）。

    同一类实现保证"结构之外的一切"（towers/尺度 buffer/初始化）完全一致，
    对照结论才可归因于**共享方式**本身。
    """
    require("torch")
    hard = MMoE(d_in, hidden=hidden, n_experts=1, expert_width=hidden, **kwargs)
    mmoe = MMoE(d_in, hidden=hidden, n_experts=n_experts,
                expert_width=max(int(hidden) // int(n_experts), 4), **kwargs)
    return hard, mmoe


def param_comparison(reference, candidate, tol: float = 0.15) -> dict[str, Any]:
    """参数量可比性收据；`ok=False` 时**禁止**据此比较结构（E8/P0 §8）。"""
    require("torch")
    a, b = count_parameters(reference), count_parameters(candidate)
    ratio = float(b) / float(max(a, 1))
    return {"reference_params": a, "candidate_params": b, "ratio": ratio,
            "tol": float(tol), "ok": bool(abs(ratio - 1.0) <= float(tol)),
            "note": ("瓶颈窄化 expert_width=hidden//E：专家银行总参数量 ≈ 单宽专家"
                     "（E 个 d_in×ew + ew×hidden ≈ 1 个 d_in×hidden + hidden×hidden）")}


def task_gradient_cosines(losses: Mapping[str, Any], shared_parameters: Sequence[Any],
                          retain_graph: bool = True) -> dict[str, Any]:
    """任务间**梯度余弦相似度**（共享参数上）。

    `losses` 为 `{task: scalar tensor}`；逐任务反传并收集共享参数梯度（不更新参数）。
    返回逐对余弦 + 均值；任一梯度全零则该对记 `None`。
    """
    require("torch")
    grads: dict[str, list[Any]] = {}
    for task, loss in losses.items():
        for p in shared_parameters:
            if p.grad is not None:
                p.grad = None
        loss.backward(retain_graph=retain_graph)
        grads[task] = [(p.grad.detach().clone() if p.grad is not None
                        else torch.zeros_like(p)) for p in shared_parameters]

    def flat(vs):
        return torch.cat([v.reshape(-1) for v in vs]) if vs else torch.zeros(0)

    tasks = list(losses)
    pairs: dict[str, float | None] = {}
    vals: list[float] = []
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            va, vb = flat(grads[a]), flat(grads[b])
            na, nb = float(va.norm()), float(vb.norm())
            if na <= 0 or nb <= 0:
                pairs[f"{a}|{b}"] = None
                continue
            c = float(torch.dot(va, vb) / (va.norm() * vb.norm()))
            pairs[f"{a}|{b}"] = c
            vals.append(c)
    for p in shared_parameters:
        if p.grad is not None:
            p.grad = None
    return {"cosines": pairs, "mean_cosine": (sum(vals) / len(vals)) if vals else None,
            "n_tasks": len(tasks)}
