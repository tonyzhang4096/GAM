# /mnt/data/general_attention.py
"""
General (subset) attention via Gibbs sampling over subsets.

Implements:
  - X -> Q,K,V projections
  - Gibbs distribution over subsets S: p(S) ∝ exp(β F2(S))
  - Output x' = E_p[F1(S)]
  - Approximate E using m independent runs of Algorithm 1 (random-scan Gibbs sampler)
    from Gotovos et al., "Sampling from Probabilistic Submodular Models" (NeurIPS 2015).
  - restricted_softmax + full_set recovers ordinary scaled dot-product attention exactly.

We provide F2 instantiations:
  1) full_set:      S = {1, ..., L} deterministically
  2) modular_dot:   F(S) = Σ_{i∈S} a_i, a_i = <q,k_i>/√d
  3) modular_dot_hard_singleton:
                     F(S) = Σ_{i∈S} a_i - τ(q) 1[|S| > 1]
  4) modular_dot_first_free:
                     F(S) = Σ_{i∈S} a_i - τ(q) (|S|-1)_+
  5) logsumexp:     F(S) = log(ε + Σ_{i∈S} exp(a_i))
  6) dot_repulsion: F(S) = Σ_{i∈S} a_i - λ Σ_{i<j∈S} <k_i,k_j>/√d
  7) neural_mlp:    F(S) = MLP([q, k_{i1}, ..., k_{ir} (padded), log(1+|S|)])

and F1 instantiations:
  - mean
  - mlp_mean
  - mlp_concat (explicit subset members, padded/truncated)
  - transformer (tiny Transformer over explicit subset members)
  - restricted_softmax (ordinary attention weights over the selected support)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import math
import torch
import torch.nn as nn

F2Type = Literal[
    "full_set",
    "modular_dot",
    "modular_dot_hard_singleton",
    "modular_dot_first_free",
    "logsumexp",
    "dot_repulsion",
    "neural_mlp",
]
F1Type = Literal["mean", "mlp_mean", "mlp_concat", "transformer", "restricted_softmax"]
InitType = Literal["empty", "random"]
STGradientMode = Literal["partial", "consistent"]
GradientMethod = Literal["ste", "gumbel"]
F1QueryMode = Literal["none", "replace", "add"]

class SetAggregator(nn.Module):
    """Base class for F1(S).

    Most implementations consume sufficient stats (sum_v, count); some optional
    implementations can consume explicit subset members.
    """
    def forward(self, sum_v: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _pack_selected_members(
    chain_x: torch.Tensor,       # (n_chains, L, d)
    mask: torch.Tensor,          # (n_chains, L) bool
    max_set_size: int,
    rank_scores: Optional[torch.Tensor] = None,  # (n_chains, L)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack up to max_set_size selected members, padded with zeros.

    Members are chosen by top rank_scores among selected positions; if scores are
    omitted, selected positions receive equal rank and truncation follows index
    order after tie-breaking.
    """

    n_chains, _, d = chain_x.shape
    safe_idx, picked = _select_subset_indices(
        mask=mask, max_set_size=max_set_size, rank_scores=rank_scores
    )
    keep = safe_idx.shape[1]
    if keep <= 0:
        packed = chain_x.new_zeros((n_chains, 0, d))
        return packed, picked

    gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, d)
    packed = torch.gather(chain_x, 1, gather_idx)
    packed = packed * picked.unsqueeze(-1).to(packed.dtype)
    return packed, picked


def _select_subset_indices(
    mask: torch.Tensor,          # (n_chains, L) bool
    max_set_size: int,
    rank_scores: Optional[torch.Tensor] = None,  # (n_chains, L)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select up to max_set_size subset indices by rank_scores among mask=True."""

    n_chains, L = mask.shape
    keep = min(int(max_set_size), int(L))
    if keep <= 0:
        safe_idx = torch.zeros((n_chains, 0), device=mask.device, dtype=torch.long)
        picked = torch.zeros((n_chains, 0), device=mask.device, dtype=torch.bool)
        return safe_idx, picked

    if rank_scores is None:
        base_scores = mask.to(torch.float32)
    else:
        base_scores = rank_scores.to(torch.float32)
    neg_inf = torch.finfo(base_scores.dtype).min
    masked_scores = torch.where(mask, base_scores, torch.full_like(base_scores, neg_inf))
    _, top_idx = torch.topk(masked_scores, k=keep, dim=1)
    picked = torch.gather(mask, 1, top_idx)

    # Keep canonical order of selected positions for downstream sequence models.
    sentinel = torch.full_like(top_idx, L)
    sortable = torch.where(picked, top_idx, sentinel)
    sortable, order = torch.sort(sortable, dim=1)
    picked = torch.gather(picked, 1, order)
    safe_idx = torch.where(picked, sortable, torch.zeros_like(sortable))
    return safe_idx, picked


def _query_conditioned_subset_pool(
    rank_scores: torch.Tensor,   # (n_chains, L)
    v: torch.Tensor,             # (B, L, d_v)
    batch_idx: torch.Tensor,     # (n_chains,)
    mask: torch.Tensor,          # (n_chains, L)
    max_set_size: int,
) -> torch.Tensor:
    """Restricted query-conditioned pooling over selected subset members."""

    safe_idx, picked = _select_subset_indices(
        mask=mask, max_set_size=max_set_size, rank_scores=rank_scores
    )
    n_chains, keep = safe_idx.shape
    d_v = v.shape[-1]
    if keep <= 0:
        return v.new_zeros((n_chains, d_v))

    gather_batch = batch_idx.unsqueeze(1).expand(-1, keep)
    sel_v = v[gather_batch, safe_idx]  # (n_chains, keep, d_v)
    sel_scores = torch.gather(rank_scores, 1, safe_idx)
    neg_inf = torch.finfo(sel_scores.dtype).min
    sel_scores = torch.where(picked, sel_scores, torch.full_like(sel_scores, neg_inf))
    attn = torch.softmax(sel_scores, dim=1)
    attn = torch.where(picked, attn, torch.zeros_like(attn))
    non_empty = picked.any(dim=1, keepdim=True)
    attn = torch.where(non_empty, attn, torch.zeros_like(attn))
    pooled = (attn.unsqueeze(-1).to(sel_v.dtype) * sel_v).sum(dim=1)
    return pooled


def _restricted_softmax_subset_pool(
    rank_scores: torch.Tensor,   # (n_chains, L)
    v: torch.Tensor,             # (B, L, d_v)
    batch_idx: torch.Tensor,     # (n_chains,)
    mask: torch.Tensor,          # (n_chains, L)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Softmax attention restricted to mask=True positions.

    When the support is empty, this returns a zero vector for that chain.
    """

    chain_v = v[batch_idx]  # (n_chains, L, d_v)
    mask_f = mask.to(rank_scores.dtype)
    non_empty = mask.any(dim=1, keepdim=True)
    neg_inf = torch.finfo(rank_scores.dtype).min
    masked_scores = torch.where(mask, rank_scores, torch.full_like(rank_scores, neg_inf))
    row_max = masked_scores.max(dim=1, keepdim=True).values
    row_max = torch.where(non_empty, row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(rank_scores - row_max) * mask_f
    denom = exp_scores.sum(dim=1, keepdim=True).clamp_min(float(eps))
    weights = exp_scores / denom
    weights = torch.where(non_empty, weights, torch.zeros_like(weights))
    return (weights.unsqueeze(-1).to(chain_v.dtype) * chain_v).sum(dim=1)


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

    Selects up to `max_set_size` members from S by relevance, zero-pads if needed,
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
        rank_scores: Optional[torch.Tensor] = None,  # (n_chains, L)
    ) -> torch.Tensor:
        chain_v = v[batch_idx]  # (n_chains, L, d_v)
        d_v = chain_v.shape[-1]
        packed, _ = _pack_selected_members(
            chain_x=chain_v,
            mask=mask,
            max_set_size=self.max_set_size,
            rank_scores=rank_scores,
        )
        n_chains = packed.shape[0]
        k_keep = packed.shape[1]
        if k_keep < self.max_set_size:
            pad = chain_v.new_zeros((n_chains, self.max_set_size - k_keep, d_v))
            packed = torch.cat([packed, pad], dim=1)

        flat = packed.reshape(n_chains, self.max_set_size * d_v)
        log_count = torch.log1p(count.to(chain_v.dtype)).unsqueeze(-1)
        x = torch.cat([flat, log_count], dim=-1)
        return self.mlp(x)


class RestrictedSoftmaxAggregator(SetAggregator):
    """Exact softmax attention weights over the selected support."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, sum_v: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "RestrictedSoftmaxAggregator requires q/k-aware subset masks; call forward_subset"
        )

    def forward_subset(
        self,
        v: torch.Tensor,          # (B, L, d_v)
        batch_idx: torch.Tensor,  # (n_chains,)
        mask: torch.Tensor,       # (n_chains, L) bool
        count: torch.Tensor,      # (n_chains,)
        rank_scores: Optional[torch.Tensor] = None,  # (n_chains, L)
    ) -> torch.Tensor:
        del count
        if rank_scores is None:
            raise ValueError("f1_type='restricted_softmax' requires rank_scores")
        return _restricted_softmax_subset_pool(
            rank_scores=rank_scores,
            v=v,
            batch_idx=batch_idx,
            mask=mask,
            eps=self.eps,
        )


class TransformerSubsetAggregator(SetAggregator):
    """Lightweight Transformer F1 over explicit subset members.

    Selects up to `max_set_size` members from S by relevance, runs a tiny
    Transformer encoder over the packed sequence, masked-mean pools valid
    positions, appends log(1+|S|), and projects back to d_v.
    """

    def __init__(
        self,
        d_v: int,
        max_set_size: int = 8,
        hidden: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_v = int(d_v)
        self.max_set_size = int(max_set_size)
        if self.max_set_size <= 0:
            raise ValueError("f1 transformer max_set_size must be > 0")
        if int(num_layers) <= 0:
            raise ValueError("f1 transformer num_layers must be > 0")

        n_heads = max(1, min(int(num_heads), self.d_v))
        while self.d_v % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.num_heads = n_heads
        self.eps = float(eps)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_set_size, self.d_v))
        ff_dim = max(int(hidden), self.d_v)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_v,
            nhead=self.num_heads,
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.out = nn.Sequential(
            nn.Linear(self.d_v + 1, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, self.d_v),
        )

    def forward(self, sum_v: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("TransformerSubsetAggregator requires subset masks; call forward_subset")

    def forward_subset(
        self,
        v: torch.Tensor,          # (B, L, d_v)
        batch_idx: torch.Tensor,  # (n_chains,)
        mask: torch.Tensor,       # (n_chains, L) bool
        count: torch.Tensor,      # (n_chains,)
        rank_scores: Optional[torch.Tensor] = None,  # (n_chains, L)
    ) -> torch.Tensor:
        chain_v = v[batch_idx]  # (n_chains, L, d_v)
        d_v = chain_v.shape[-1]
        if d_v != self.d_v:
            raise ValueError("TransformerSubsetAggregator d_v mismatch")

        packed, picked = _pack_selected_members(
            chain_x=chain_v,
            mask=mask,
            max_set_size=self.max_set_size,
            rank_scores=rank_scores,
        )
        n_chains, k_keep, _ = packed.shape
        if k_keep < self.max_set_size:
            pad = chain_v.new_zeros((n_chains, self.max_set_size - k_keep, d_v))
            packed = torch.cat([packed, pad], dim=1)
            pad_mask = torch.zeros(
                (n_chains, self.max_set_size - k_keep), device=mask.device, dtype=torch.bool
            )
            picked = torch.cat([picked, pad_mask], dim=1)

        x = packed + self.pos_embed[:, : self.max_set_size, :].to(packed.dtype)
        key_padding_mask = ~picked

        # Avoid fully-masked rows: unmask one token, then zero it out in pooling.
        empty = key_padding_mask.all(dim=1)
        if empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty, 0] = False

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        picked_f = picked.to(encoded.dtype)
        denom = picked_f.sum(dim=1, keepdim=True).clamp_min(self.eps)
        pooled = (encoded * picked_f.unsqueeze(-1)).sum(dim=1) / denom
        pooled = torch.where(
            picked.any(dim=1, keepdim=True),
            pooled,
            torch.zeros_like(pooled),
        )
        log_count = torch.log1p(count.to(chain_v.dtype)).unsqueeze(-1)
        feat = torch.cat([pooled, log_count], dim=-1)
        return self.out(feat)


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
    if f1_type == "transformer":
        return TransformerSubsetAggregator(
            d_v=d_v,
            max_set_size=concat_max_set_size,
            hidden=concat_hidden,
        )
    if f1_type == "restricted_softmax":
        return RestrictedSoftmaxAggregator()
    raise ValueError(f"Unknown f1_type: {f1_type}")

class NeuralMLPF2(nn.Module):
    """Learnable F2(S) conditioned on query q and explicit subset keys."""

    def __init__(
        self,
        d_qk: int,
        hidden: int = 128,
        eps: float = 1e-8,
        max_set_size: int = 8,
    ):
        super().__init__()
        self.d_qk = int(d_qk)
        self.eps = float(eps)
        self.max_set_size = int(max_set_size)
        if self.max_set_size <= 0:
            raise ValueError("f2 neural max_set_size must be > 0")
        in_dim = self.d_qk + self.max_set_size * self.d_qk + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _pack_subset_keys(
        self,
        k: torch.Tensor,          # (B, L, d_qk)
        batch_idx: torch.Tensor,  # (n_chains,)
        mask: torch.Tensor,       # (n_chains, L) bool
        rank_scores: Optional[torch.Tensor] = None,  # (n_chains, L)
    ) -> torch.Tensor:
        d_qk = k.shape[-1]
        if d_qk != self.d_qk:
            raise ValueError("NeuralMLPF2 d_qk mismatch")

        safe_idx, picked = _select_subset_indices(
            mask=mask,
            max_set_size=self.max_set_size,
            rank_scores=rank_scores,
        )
        n_chains, keep = safe_idx.shape
        if keep > 0:
            gather_batch = batch_idx.unsqueeze(1).expand(-1, keep)
            packed = k[gather_batch, safe_idx]  # (n_chains, keep, d_qk)
            packed = packed * picked.unsqueeze(-1).to(packed.dtype)
        else:
            packed = k.new_zeros((n_chains, 0, d_qk))

        if keep < self.max_set_size:
            pad = k.new_zeros((n_chains, self.max_set_size - keep, d_qk))
            packed = torch.cat([packed, pad], dim=1)

        return packed.reshape(n_chains, -1)

    def forward_subset(
        self,
        q: torch.Tensor,          # (n_chains, d_qk)
        k: torch.Tensor,          # (B, L, d_qk)
        batch_idx: torch.Tensor,  # (n_chains,)
        mask: torch.Tensor,       # (n_chains, L) bool
        count: torch.Tensor,      # (n_chains,)
        rank_scores: Optional[torch.Tensor] = None,  # (n_chains, L)
    ) -> torch.Tensor:
        packed = self._pack_subset_keys(
            k=k,
            batch_idx=batch_idx,
            mask=mask,
            rank_scores=rank_scores,
        )
        log_count = torch.log1p(count.to(q.dtype)).unsqueeze(-1)
        feat = torch.cat([q, packed.to(q.dtype), log_count], dim=-1)
        return self.mlp(feat).squeeze(-1)

class QueryTau(nn.Module):
    """Learnable per-query threshold tau(q) used in Gibbs logits."""
    def __init__(self, d_qk: int, init_value: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_qk, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, float(init_value))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.proj(q).squeeze(-1)

@dataclass
class GibbsConfig:
    beta: float = 1.0
    gibbs_steps: int = 64
    runs: int = 4
    init: InitType = "empty"
    init_p: float = 0.5
    logsumexp_eps: float = 1e-6
    repulsion_lambda: float = 0.1
    # Use straight-through gradients through Bernoulli decisions so Q/K can train.
    straight_through: bool = True
    # partial: update sampler side-state with hard sign; consistent: use ST sign.
    st_gradient_mode: STGradientMode = "partial"
    gradient_method: GradientMethod = "ste"
    gumbel_tau: float = 1.0
    gumbel_tau_min: float = 0.1

def _gibbs_sample_subsets(
    q: torch.Tensor,     # (B, Lq, d_qk)
    k: torch.Tensor,     # (B, L,  d_qk)
    v: torch.Tensor,     # (B, L,  d_v)
    cfg: GibbsConfig,
    f2_type: F2Type,
    f1: SetAggregator,
    f2_neural: Optional[NeuralMLPF2] = None,
    tau_fn: Optional[QueryTau] = None,
    f1_query_mode: F1QueryMode = "none",
    f1_query_max_set_size: int = 8,
) -> torch.Tensor:
    if cfg.beta <= 0: raise ValueError("beta must be > 0")
    if cfg.gibbs_steps <= 0: raise ValueError("gibbs_steps must be > 0")
    if cfg.runs <= 0: raise ValueError("runs must be > 0")
    if f2_type == "full_set":
        raise ValueError("f2_type='full_set' must use the deterministic support path")

    B, Lq, d_qk = q.shape
    _, L, d_qk2 = k.shape
    _, Lv, d_v = v.shape
    if L != Lv: raise ValueError("k/v length mismatch")
    if d_qk != d_qk2: raise ValueError("q/k dim mismatch")
    if f1_query_mode not in ("none", "replace", "add"):
        raise ValueError("f1_query_mode must be one of: none, replace, add")
    if int(f1_query_max_set_size) <= 0:
        raise ValueError("f1_query_max_set_size must be > 0")

    device = q.device
    work_dtype = q.dtype if q.is_floating_point() else torch.float32
    q_f, k_f, v_f = q.to(work_dtype), k.to(work_dtype), v.to(work_dtype)
    scale = 1.0 / math.sqrt(d_qk)
    needs_sum_k = f2_type == "dot_repulsion"
    needs_sum_w = f2_type == "logsumexp"

    n_queries = B * Lq
    n_chains = n_queries * cfg.runs

    q_flat = q_f.reshape(n_queries, d_qk)
    q_rep = q_flat.repeat_interleave(cfg.runs, dim=0)  # (n_chains, d_qk)
    tau_q = tau_fn(q_rep) if tau_fn is not None else None

    batch_for_query = torch.arange(B, device=device).repeat_interleave(Lq)
    batch_idx = batch_for_query.repeat_interleave(cfg.runs)
    rank_scores_rep = torch.matmul(q_f, k_f.transpose(1, 2)) * scale  # (B, Lq, L)
    rank_scores_rep = rank_scores_rep.reshape(n_queries, L)
    rank_scores_rep = rank_scores_rep.repeat_interleave(cfg.runs, dim=0)

    # --- init state ---
    if cfg.init == "empty":
        mask = torch.zeros((n_chains, L), device=device, dtype=torch.bool)
        count = torch.zeros((n_chains,), device=device, dtype=torch.int32)
        count_f = torch.zeros((n_chains,), device=device, dtype=work_dtype)
        sum_v = torch.zeros((n_chains, d_v), device=device, dtype=work_dtype)
        sum_k = torch.zeros((n_chains, d_qk), device=device, dtype=work_dtype) if needs_sum_k else None
        sum_w = torch.zeros((n_chains,), device=device, dtype=work_dtype) if needs_sum_w else None
    elif cfg.init == "random":
        if f2_type == "logsumexp":
            raise ValueError("init='random' not supported for logsumexp (needs per-query weights)")
        mask = (torch.rand((n_chains, L), device=device) < float(cfg.init_p))
        count = mask.sum(dim=1, dtype=torch.int32)
        count_f = count.to(work_dtype)
        sum_v = torch.zeros((n_chains, d_v), device=device, dtype=work_dtype)
        sum_k = torch.zeros((n_chains, d_qk), device=device, dtype=work_dtype) if needs_sum_k else None
        sum_w = None

        chains_per_batch = Lq * cfg.runs
        for b in range(B):
            start = b * chains_per_batch
            end = (b + 1) * chains_per_batch
            m_b = mask[start:end].to(work_dtype)
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

        tau_logits = tau_q

        if f2_type == "modular_dot":
            delta = a_v
        elif f2_type == "modular_dot_hard_singleton":
            count_excl = count - old_in.to(count.dtype)
            if tau_q is not None:
                second_token_penalty = tau_q * (count_excl == 1).to(tau_q.dtype)
                delta = a_v - second_token_penalty
                tau_logits = None
            else:
                delta = a_v
                tau_logits = None
        elif f2_type == "modular_dot_first_free":
            count_excl = count - old_in.to(count.dtype)
            if tau_q is not None:
                extra_token_penalty = tau_q * (count_excl > 0).to(tau_q.dtype)
                delta = a_v - extra_token_penalty
                tau_logits = None
            else:
                delta = a_v
                tau_logits = None
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

            mask_excl = mask.clone()
            mask_excl[chain_ids, vidx] = False
            mask_add = mask_excl.clone()
            mask_add[chain_ids, vidx] = True

            count_excl = count - old_in.to(count.dtype)
            count_add = count_excl + 1
            e_excl = f2_neural.forward_subset(
                q=q_rep,
                k=k_f,
                batch_idx=batch_idx,
                mask=mask_excl,
                count=count_excl,
                rank_scores=rank_scores_rep,
            )
            e_add = f2_neural.forward_subset(
                q=q_rep,
                k=k_f,
                batch_idx=batch_idx,
                mask=mask_add,
                count=count_add,
                rank_scores=rank_scores_rep,
            )
            delta = e_add - e_excl
        else:
            raise ValueError(f"Unknown f2_type: {f2_type}")

        if tau_logits is not None:
            logits = beta * (delta - tau_logits)
        else:
            logits = beta * delta
        p_add = torch.sigmoid(logits)

        old_f = old_in.to(work_dtype)

        if cfg.gradient_method == "gumbel":
            # Gumbel-Softmax: differentiable relaxation of the binary decision.
            # Work directly with logits (numerically stable) instead of log-probs.
            tau = max(cfg.gumbel_tau, 1e-6)
            binary_logits = torch.stack([-logits, logits], dim=-1)  # (n_chains, 2): [exclude, include]
            soft_samples = torch.nn.functional.gumbel_softmax(
                binary_logits, tau=tau, hard=False, dim=-1,
            )
            new_f_soft = soft_samples[:, 1]  # soft "include" probability
            new_in_hard = (new_f_soft > 0.5)
            new_f_hard = new_in_hard.to(work_dtype)
            sign_hard = new_f_hard - old_f
            # Straight-through: hard forward, soft backward
            new_f = new_f_hard.detach() - new_f_soft.detach() + new_f_soft
            sign = new_f - old_f
        else:
            # STE (original behavior)
            z = torch.rand((n_chains,), device=device)
            new_in_hard = z <= p_add
            new_f_hard = new_in_hard.to(work_dtype)
            sign_hard = new_f_hard - old_f
            if cfg.straight_through:
                new_f = new_f_hard.detach() - p_add.detach() + p_add
                sign = new_f - old_f
            else:
                sign = sign_hard

        mask[chain_ids, vidx] = new_in_hard
        count = count + sign_hard.to(torch.int32)
        count_f = count_f + sign
        sum_v = sum_v + sign.unsqueeze(-1) * vv
        if cfg.st_gradient_mode not in ("partial", "consistent"):
            raise ValueError("st_gradient_mode must be one of: partial, consistent")
        state_sign = sign_hard
        if cfg.straight_through and cfg.st_gradient_mode == "consistent":
            state_sign = sign
        if f2_type == "dot_repulsion":
            sum_k = sum_k + state_sign.unsqueeze(-1) * kv
        elif f2_type == "logsumexp":
            sum_w = sum_w + state_sign * torch.exp(a_v)

    out_chain = _apply_f1_to_support(
        f1=f1,
        v=v_f,
        batch_idx=batch_idx,
        mask=mask,
        count=count_f,
        rank_scores=rank_scores_rep,
        sum_v=sum_v,
        f1_query_mode=f1_query_mode,
        f1_query_max_set_size=f1_query_max_set_size,
    )
    out = out_chain.view(B, Lq, cfg.runs, d_v).mean(dim=2)
    return out


def _apply_f1_to_support(
    *,
    f1: SetAggregator,
    v: torch.Tensor,             # (B, L, d_v)
    batch_idx: torch.Tensor,     # (n_supports,)
    mask: torch.Tensor,          # (n_supports, L)
    count: torch.Tensor,         # (n_supports,)
    rank_scores: torch.Tensor,   # (n_supports, L)
    sum_v: Optional[torch.Tensor],
    f1_query_mode: F1QueryMode,
    f1_query_max_set_size: int,
) -> torch.Tensor:
    if isinstance(f1, RestrictedSoftmaxAggregator):
        if f1_query_mode != "none":
            raise ValueError(
                "f1_type='restricted_softmax' is already query-conditioned; "
                "use f1_query_mode='none'"
            )
        return f1.forward_subset(
            v=v,
            batch_idx=batch_idx,
            mask=mask,
            count=count,
            rank_scores=rank_scores,
        )

    out_chain: Optional[torch.Tensor]
    if f1_query_mode == "replace":
        out_chain = None
    elif isinstance(f1, (ConcatMLPAggregator, TransformerSubsetAggregator)):
        out_chain = f1.forward_subset(
            v=v,
            batch_idx=batch_idx,
            mask=mask,
            count=count,
            rank_scores=rank_scores,
        )
    else:
        if sum_v is None:
            raise ValueError("sum_v is required for summary-stat F1 aggregators")
        out_chain = f1(sum_v, count)

    if f1_query_mode != "none":
        q_ctx = _query_conditioned_subset_pool(
            rank_scores=rank_scores,
            v=v,
            batch_idx=batch_idx,
            mask=mask,
            max_set_size=f1_query_max_set_size,
        )
        if f1_query_mode == "replace":
            out_chain = q_ctx
        else:
            assert out_chain is not None
            out_chain = out_chain + q_ctx

    assert out_chain is not None
    return out_chain


def _deterministic_full_set_output(
    q: torch.Tensor,     # (B, Lq, d_qk)
    k: torch.Tensor,     # (B, L,  d_qk)
    v: torch.Tensor,     # (B, L,  d_v)
    f1: SetAggregator,
    *,
    f1_query_mode: F1QueryMode,
    f1_query_max_set_size: int,
) -> torch.Tensor:
    """Deterministic support path for full_set; skips Gibbs and MC entirely."""

    B, Lq, d_qk = q.shape
    _, L, _ = k.shape
    _, Lv, d_v = v.shape
    if L != Lv:
        raise ValueError("k/v length mismatch")
    scale = 1.0 / math.sqrt(d_qk)
    rank_scores = torch.matmul(q, k.transpose(1, 2)) * scale  # (B, Lq, L)

    if isinstance(f1, RestrictedSoftmaxAggregator):
        if f1_query_mode != "none":
            raise ValueError(
                "f1_type='restricted_softmax' is already query-conditioned; "
                "use f1_query_mode='none'"
            )
        # Exact-special-case path: ordinary scaled dot-product attention.
        weights = torch.softmax(rank_scores, dim=-1)
        return weights @ v

    n_supports = B * Lq
    batch_idx = torch.arange(B, device=q.device).repeat_interleave(Lq)
    mask = torch.ones((n_supports, L), device=q.device, dtype=torch.bool)
    count = torch.full((n_supports,), L, device=q.device, dtype=q.dtype)
    rank_scores_flat = rank_scores.reshape(n_supports, L)
    sum_v = v[batch_idx].sum(dim=1)

    out = _apply_f1_to_support(
        f1=f1,
        v=v,
        batch_idx=batch_idx,
        mask=mask,
        count=count,
        rank_scores=rank_scores_flat,
        sum_v=sum_v,
        f1_query_mode=f1_query_mode,
        f1_query_max_set_size=f1_query_max_set_size,
    )
    return out.view(B, Lq, d_v)

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
        use_learned_tau: bool = True,
        tau_init: float = 0.0,
        f1_concat_max_set_size: int = 8,
        f1_concat_hidden: int = 128,
        f2_neural_hidden: int = 128,
        f1_query_mode: F1QueryMode = "none",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_qk = int(d_qk if d_qk is not None else d_model)
        self.d_v = int(d_v if d_v is not None else d_model)

        self.W_Q = nn.Linear(self.d_model, self.d_qk, bias=bias)
        self.W_K = nn.Linear(self.d_model, self.d_qk, bias=bias)
        self.W_V = nn.Linear(self.d_model, self.d_v, bias=bias)

        self.f2_type = f2_type
        self.f1_type = f1_type
        self.f1 = build_f1(
            f1_type,
            d_v=self.d_v,
            concat_max_set_size=f1_concat_max_set_size,
            concat_hidden=f1_concat_hidden,
        )
        self.f2_neural = (
            NeuralMLPF2(
                self.d_qk,
                hidden=f2_neural_hidden,
                max_set_size=f1_concat_max_set_size,
            )
            if self.f2_type == "neural_mlp"
            else None
        )
        self.tau_fn = QueryTau(self.d_qk, init_value=tau_init) if use_learned_tau else None
        self.cfg = cfg if cfg is not None else GibbsConfig()
        self.f1_concat_max_set_size = int(f1_concat_max_set_size)
        if self.f1_concat_max_set_size <= 0:
            raise ValueError("f1_concat_max_set_size must be > 0")
        if f1_query_mode not in ("none", "replace", "add"):
            raise ValueError("f1_query_mode must be one of: none, replace, add")
        if f1_type == "restricted_softmax" and f1_query_mode != "none":
            raise ValueError(
                "f1_type='restricted_softmax' is already query-conditioned; "
                "use f1_query_mode='none'"
            )
        self.f1_query_mode = f1_query_mode

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
            if self.f2_type == "full_set":
                out_chunk = _deterministic_full_set_output(
                    q=q_chunk,
                    k=k,
                    v=v,
                    f1=self.f1,
                    f1_query_mode=self.f1_query_mode,
                    f1_query_max_set_size=self.f1_concat_max_set_size,
                )
            else:
                out_chunk = _gibbs_sample_subsets(
                    q=q_chunk, k=k, v=v,
                    cfg=self.cfg,
                    f2_type=self.f2_type,
                    f1=self.f1,
                    f2_neural=self.f2_neural,
                    tau_fn=self.tau_fn,
                    f1_query_mode=self.f1_query_mode,
                    f1_query_max_set_size=self.f1_concat_max_set_size,
                )
            outs.append(out_chunk)
        y = torch.cat(outs, dim=1)
        return y.squeeze(0) if squeeze_batch else y
