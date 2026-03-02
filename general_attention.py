# /mnt/data/general_attention.py
"""
General (subset) attention via Gibbs sampling over subsets.

Implements:
  - X -> Q,K,V projections
  - Gibbs distribution over subsets S: p(S) ∝ exp(β F2(S))
  - Output x' = E_p[F1(S)]
  - Approximate E using m independent runs of Algorithm 1 (random-scan Gibbs sampler)
    from Gotovos et al., "Sampling from Probabilistic Submodular Models" (NeurIPS 2015).

We provide efficient F2 instantiations with O(d) incremental Δ updates:
  1) modular_dot:   F(S) = Σ_{i∈S} a_i, a_i = <q,k_i>/√d
  2) logsumexp:     F(S) = log(ε + Σ_{i∈S} exp(a_i))
  3) dot_repulsion: F(S) = Σ_{i∈S} a_i - λ Σ_{i<j∈S} <k_i,k_j>/√d
  4) neural_mlp:    F(S) = MLP([q, mean_k(S), log(1+|S|)])

and F1 instantiations operating on sufficient stats:
  - mean
  - mlp_mean
  - mlp_concat (explicit subset members, padded/truncated)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import math
import torch
import torch.nn as nn

F2Type = Literal["modular_dot", "logsumexp", "dot_repulsion", "neural_mlp"]
F1Type = Literal["mean", "mlp_mean", "mlp_concat"]
InitType = Literal["empty", "random"]

class SetAggregator(nn.Module):
    """Base class for F1(S).

    Most implementations consume sufficient stats (sum_v, count); some optional
    implementations can consume explicit subset members.
    """
    def forward(self, sum_v: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MeanAggregator(SetAggregator):
    def __init__(self, eps: float = 1e-8, empty_value: float = 0.0):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_empty", torch.tensor(float(empty_value)))

    def forward(self, sum_v: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        denom = count.to(sum_v.dtype).clamp_min(self.eps).unsqueeze(-1)
        out = sum_v / denom
        if out.numel() == 0:
            return out
        if self._empty.item() != 0.0:
            empty_vec = self._empty.to(out.dtype).expand_as(out)
            out = torch.where(count.unsqueeze(-1) > 0, out, empty_vec)
        else:
            out = torch.where(count.unsqueeze(-1) > 0, out, torch.zeros_like(out))
        return out

class MLPMeanAggregator(SetAggregator):
    """Learnable F1 that still only needs (sum_v, count):
       features = [mean_v, log(1+count)]
    """
    def __init__(self, d_v: int, hidden: int = 128, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_v + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_v),
        )

    def forward(self, sum_v: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        denom = count.to(sum_v.dtype).clamp_min(self.eps).unsqueeze(-1)
        mean_v = sum_v / denom
        log_count = torch.log1p(count.to(sum_v.dtype)).unsqueeze(-1)
        x = torch.cat([mean_v, log_count], dim=-1)
        return self.mlp(x)

class ConcatMLPAggregator(SetAggregator):
    """Learnable F1 over explicit subset members.

    Selects up to `max_set_size` members from S in index order, zero-pads if needed,
    concatenates them, appends log(1+|S|), and applies an MLP.
    """
    def __init__(self, d_v: int, max_set_size: int = 8, hidden: int = 128):
        super().__init__()
        self.d_v = int(d_v)
        self.max_set_size = int(max_set_size)
        if self.max_set_size <= 0:
            raise ValueError("f1 concat max_set_size must be > 0")
        self.mlp = nn.Sequential(
            nn.Linear(self.max_set_size * self.d_v + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_v),
        )

    def forward(self, sum_v: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("ConcatMLPAggregator requires subset masks; call forward_subset")

    def forward_subset(
        self,
        v: torch.Tensor,          # (B, L, d_v)
        batch_idx: torch.Tensor,  # (n_chains,)
        mask: torch.Tensor,       # (n_chains, L) bool
        count: torch.Tensor,      # (n_chains,)
    ) -> torch.Tensor:
        n_chains, L = mask.shape
        chain_v = v[batch_idx]  # (n_chains, L, d_v)
        d_v = chain_v.shape[-1]
        k_keep = min(self.max_set_size, L)

        if k_keep > 0:
            pos = torch.arange(L, device=mask.device, dtype=torch.float32)
            # Selected positions get positive scores; top-k prefers lower indices.
            scores = mask.to(torch.float32) * (float(L) - pos).unsqueeze(0)
            top_scores, top_idx = torch.topk(scores, k=k_keep, dim=1)
            picked = top_scores > 0

            sentinel = torch.full_like(top_idx, L)
            sortable = torch.where(picked, top_idx, sentinel)
            sortable, order = torch.sort(sortable, dim=1)
            picked = torch.gather(picked, 1, order)

            safe_idx = torch.where(picked, sortable, torch.zeros_like(sortable))
            gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, d_v)
            packed = torch.gather(chain_v, 1, gather_idx)
            packed = packed * picked.unsqueeze(-1).to(packed.dtype)
        else:
            packed = chain_v.new_zeros((n_chains, 0, d_v))

        if k_keep < self.max_set_size:
            pad = chain_v.new_zeros((n_chains, self.max_set_size - k_keep, d_v))
            packed = torch.cat([packed, pad], dim=1)

        flat = packed.reshape(n_chains, self.max_set_size * d_v)
        log_count = torch.log1p(count.to(chain_v.dtype)).unsqueeze(-1)
        x = torch.cat([flat, log_count], dim=-1)
        return self.mlp(x)

def build_f1(
    f1_type: F1Type,
    d_v: int,
    *,
    concat_max_set_size: int = 8,
    concat_hidden: int = 128,
) -> SetAggregator:
    if f1_type == "mean":
        return MeanAggregator()
    if f1_type == "mlp_mean":
        return MLPMeanAggregator(d_v=d_v)
    if f1_type == "mlp_concat":
        return ConcatMLPAggregator(
            d_v=d_v,
            max_set_size=concat_max_set_size,
            hidden=concat_hidden,
        )
    raise ValueError(f"Unknown f1_type: {f1_type}")

class NeuralMLPF2(nn.Module):
    """Learnable F2(S) conditioned on query q and subset key summary."""
    def __init__(self, d_qk: int, hidden: int = 128, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_qk + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        q: torch.Tensor,      # (n_chains, d_qk)
        sum_k: torch.Tensor,  # (n_chains, d_qk)
        count: torch.Tensor,  # (n_chains,)
    ) -> torch.Tensor:
        denom = count.to(sum_k.dtype).clamp_min(self.eps).unsqueeze(-1)
        mean_k = sum_k / denom
        log_count = torch.log1p(count.to(sum_k.dtype)).unsqueeze(-1)
        feat = torch.cat([q, mean_k, log_count], dim=-1)
        return self.mlp(feat).squeeze(-1)

@dataclass
class GibbsConfig:
    beta: float = 1.0
    gibbs_steps: int = 64
    runs: int = 4
    init: InitType = "empty"
    init_p: float = 0.5
    logsumexp_eps: float = 1e-6
    repulsion_lambda: float = 0.1

def _gibbs_sample_subsets(
    q: torch.Tensor,     # (B, Lq, d_qk)
    k: torch.Tensor,     # (B, L,  d_qk)
    v: torch.Tensor,     # (B, L,  d_v)
    cfg: GibbsConfig,
    f2_type: F2Type,
    f1: SetAggregator,
    f2_neural: Optional[NeuralMLPF2] = None,
) -> torch.Tensor:
    if cfg.beta <= 0: raise ValueError("beta must be > 0")
    if cfg.gibbs_steps <= 0: raise ValueError("gibbs_steps must be > 0")
    if cfg.runs <= 0: raise ValueError("runs must be > 0")

    B, Lq, d_qk = q.shape
    _, L, d_qk2 = k.shape
    _, Lv, d_v = v.shape
    if L != Lv: raise ValueError("k/v length mismatch")
    if d_qk != d_qk2: raise ValueError("q/k dim mismatch")

    device = q.device
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d_qk)
    needs_sum_k = f2_type in ("dot_repulsion", "neural_mlp")
    needs_sum_w = f2_type == "logsumexp"

    n_queries = B * Lq
    n_chains = n_queries * cfg.runs

    q_flat = q_f.reshape(n_queries, d_qk)
    q_rep = q_flat.repeat_interleave(cfg.runs, dim=0)  # (n_chains, d_qk)

    batch_for_query = torch.arange(B, device=device).repeat_interleave(Lq)
    batch_idx = batch_for_query.repeat_interleave(cfg.runs)

    # --- init state ---
    if cfg.init == "empty":
        mask = torch.zeros((n_chains, L), device=device, dtype=torch.bool)
        count = torch.zeros((n_chains,), device=device, dtype=torch.int32)
        sum_v = torch.zeros((n_chains, d_v), device=device, dtype=torch.float32)
        sum_k = torch.zeros((n_chains, d_qk), device=device, dtype=torch.float32) if needs_sum_k else None
        sum_w = torch.zeros((n_chains,), device=device, dtype=torch.float32) if needs_sum_w else None
    elif cfg.init == "random":
        if f2_type == "logsumexp":
            raise ValueError("init='random' not supported for logsumexp (needs per-query weights)")
        mask = (torch.rand((n_chains, L), device=device) < float(cfg.init_p))
        count = mask.sum(dim=1, dtype=torch.int32)
        sum_v = torch.zeros((n_chains, d_v), device=device, dtype=torch.float32)
        sum_k = torch.zeros((n_chains, d_qk), device=device, dtype=torch.float32) if needs_sum_k else None
        sum_w = None

        chains_per_batch = Lq * cfg.runs
        for b in range(B):
            start = b * chains_per_batch
            end = (b + 1) * chains_per_batch
            m_b = mask[start:end].to(torch.float32)
            sum_v[start:end] = m_b @ v_f[b]
            if sum_k is not None:
                sum_k[start:end] = m_b @ k_f[b]
    else:
        raise ValueError(f"Unknown init: {cfg.init}")

    # --- Algorithm 1 Gibbs loop ---
    chain_ids = torch.arange(n_chains, device=device)
    beta = float(cfg.beta)

    for _ in range(int(cfg.gibbs_steps)):
        vidx = torch.randint(0, L, (n_chains,), device=device)
        old_in = mask[chain_ids, vidx]

        kv = k_f[batch_idx, vidx]  # (n_chains, d_qk)
        vv = v_f[batch_idx, vidx]  # (n_chains, d_v)

        a_v = (q_rep * kv).sum(dim=-1) * scale

        if f2_type == "modular_dot":
            delta = a_v
        elif f2_type == "logsumexp":
            assert sum_w is not None
            wv = torch.exp(a_v)
            sum_excl = sum_w - old_in.to(sum_w.dtype) * wv
            eps = float(cfg.logsumexp_eps)
            delta = torch.log(sum_excl + wv + eps) - torch.log(sum_excl + eps)
        elif f2_type == "dot_repulsion":
            assert sum_k is not None
            lam = float(cfg.repulsion_lambda)
            sum_k_excl = sum_k - old_in.to(sum_k.dtype).unsqueeze(-1) * kv
            interaction = (kv * sum_k_excl).sum(dim=-1) * scale
            delta = a_v - lam * interaction
        elif f2_type == "neural_mlp":
            if f2_neural is None:
                raise ValueError("f2_type='neural_mlp' requires a neural F2 module")
            assert sum_k is not None
            old_f = old_in.to(sum_k.dtype).unsqueeze(-1)
            sum_k_excl = sum_k - old_f * kv
            count_excl = count - old_in.to(count.dtype)
            sum_k_add = sum_k_excl + kv
            count_add = count_excl + 1
            e_excl = f2_neural(q_rep, sum_k_excl, count_excl.to(sum_k_excl.dtype))
            e_add = f2_neural(q_rep, sum_k_add, count_add.to(sum_k_add.dtype))
            delta = e_add - e_excl
        else:
            raise ValueError(f"Unknown f2_type: {f2_type}")

        p_add = torch.sigmoid(beta * delta)
        z = torch.rand((n_chains,), device=device)
        new_in = z <= p_add

        if torch.any(new_in != old_in):
            sign = (new_in.to(torch.float32) - old_in.to(torch.float32))
            mask[chain_ids, vidx] = new_in
            count = count + sign.to(torch.int32)
            sum_v = sum_v + sign.unsqueeze(-1) * vv
            if f2_type in ("dot_repulsion", "neural_mlp"):
                sum_k = sum_k + sign.unsqueeze(-1) * kv
            elif f2_type == "logsumexp":
                sum_w = sum_w + sign * torch.exp(a_v)

    if isinstance(f1, ConcatMLPAggregator):
        out_chain = f1.forward_subset(v=v_f, batch_idx=batch_idx, mask=mask, count=count)  # (n_chains, d_v)
    else:
        out_chain = f1(sum_v, count.to(sum_v.dtype))  # (n_chains, d_v)
    out = out_chain.view(B, Lq, cfg.runs, d_v).mean(dim=2)
    return out

class GeneralAttention(nn.Module):
    """General subset attention.
       Input:  (B,L,d) or (L,d)
       Output: (B,L,d_v) or (L,d_v)
    """
    def __init__(
        self,
        d_model: int,
        d_qk: Optional[int] = None,
        d_v: Optional[int] = None,
        *,
        f2_type: F2Type = "logsumexp",
        f1_type: F1Type = "mean",
        cfg: Optional[GibbsConfig] = None,
        query_chunk_size: int = 128,
        bias: bool = False,
        f1_concat_max_set_size: int = 8,
        f1_concat_hidden: int = 128,
        f2_neural_hidden: int = 128,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_qk = int(d_qk if d_qk is not None else d_model)
        self.d_v = int(d_v if d_v is not None else d_model)

        self.W_Q = nn.Linear(self.d_model, self.d_qk, bias=bias)
        self.W_K = nn.Linear(self.d_model, self.d_qk, bias=bias)
        self.W_V = nn.Linear(self.d_model, self.d_v, bias=bias)

        self.f2_type = f2_type
        self.f1 = build_f1(
            f1_type,
            d_v=self.d_v,
            concat_max_set_size=f1_concat_max_set_size,
            concat_hidden=f1_concat_hidden,
        )
        self.f2_neural = NeuralMLPF2(self.d_qk, hidden=f2_neural_hidden) if self.f2_type == "neural_mlp" else None
        self.cfg = cfg if cfg is not None else GibbsConfig()

        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be > 0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        if x.dim() != 3:
            raise ValueError(f"Expected (B,L,d) or (L,d), got {tuple(x.shape)}")

        B, L, d = x.shape
        if d != self.d_model:
            raise ValueError("d_model mismatch")

        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        outs = []
        for start in range(0, L, self.query_chunk_size):
            end = min(L, start + self.query_chunk_size)
            q_chunk = q[:, start:end, :]
            out_chunk = _gibbs_sample_subsets(
                q=q_chunk, k=k, v=v,
                cfg=self.cfg,
                f2_type=self.f2_type,
                f1=self.f1,
                f2_neural=self.f2_neural,
            )
            outs.append(out_chunk)
        y = torch.cat(outs, dim=1)
        return y.squeeze(0) if squeeze_batch else y
