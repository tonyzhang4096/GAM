#!/usr/bin/env python3
"""Verify that restricted_softmax + full_set matches exact attention."""

from __future__ import annotations

import math

import torch

from general_attention import GeneralAttention, GibbsConfig


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 7, 16, dtype=torch.float32)

    attn = GeneralAttention(
        d_model=16,
        d_qk=8,
        d_v=12,
        f1_type="restricted_softmax",
        f2_type="full_set",
        f1_query_mode="none",
        cfg=GibbsConfig(gibbs_steps=5, runs=4),
        query_chunk_size=3,
        use_learned_tau=True,
        tau_init=1.25,
    ).eval()

    with torch.no_grad():
        y, _ = attn(x)
        q = attn.W_Q(x)
        k = attn.W_K(x)
        v = attn.W_V(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(attn.d_qk)
        weights = torch.softmax(scores, dim=-1)
        y_ref = torch.matmul(weights, v)

    max_abs_err = (y - y_ref).abs().max().item()
    print(f"max_abs_err={max_abs_err:.8e}")
    assert max_abs_err < 1e-6, f"expected max_abs_err < 1e-6, got {max_abs_err:.8e}"
    print("Exact attention recovery verified.")


if __name__ == "__main__":
    main()
