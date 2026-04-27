# General Attention Mechanism (GAM) — Codebase Summary

> **Author**: Derek Long (MIT License, 2026)
> **Repo**: [dlon450/general-attention-mechanism](https://github.com/dlon450/general-attention-mechanism)
> **Status**: Research prototype — minimal docs, no comments, functional but rough

---

## What GAM Does

Standard attention computes `softmax(QK^T/√d) V` over **all** tokens. GAM replaces this with a **probabilistic subset selection**: it uses Gibbs sampling to stochastically choose which tokens to attend to, weighted by a learned energy function.

**Core idea:**
1. **F2(S)** — an energy function scores how "good" a subset S of tokens is
2. **Gibbs sampling** — draws subsets S from the distribution `p(S) ∝ exp(β · F2(S))`
3. **F1(S)** — an aggregation function computes the output from the selected tokens
4. Gradients flow through the hard sampling via a **straight-through estimator**

**Key property:** When `f2=full_set` + `f1=restricted_softmax`, GAM **exactly recovers** standard scaled dot-product attention (verified by `verify_exact_attention.py`).

---

## File Structure

```
GAM/
├── general_attention.py          # Core: attention mechanism, Gibbs sampler, F1/F2 modules
├── train_vit_cifar.py            # Training script: tiny ViT on CIFAR-10/100
├── verify_exact_attention.py     # Sanity check: GAM == exact attention in special case
├── plot_training_curves.py       # Visualization: accuracy/speed plots from JSON logs
├── gibbs_nips2015_experiments.ipynb  # Replication of Gotovos et al. (NeurIPS 2015)
├── sweeps/
│   ├── sweep_restricted_softmax_cifar10.sh  # Sweep F2 variants with fixed F1=restricted_softmax
│   └── sweep_f1_f2_cifar10.sh               # Cartesian sweep of F1 × F2 combinations
├── requirements.txt
├── README.md                     # Minimal (title only)
└── LICENSE                       # MIT
```

---

## File-by-File Breakdown

### `general_attention.py` (~35 KB) — Core Module

The heart of the project. Contains all attention logic.

**Classes:**

| Class | Purpose |
|---|---|
| `GeneralAttention` (nn.Module) | Main entry point. Projects Q/K/V, dispatches to Gibbs or deterministic path |
| `SetAggregator` | Abstract base for F1 functions |
| `MeanAggregator` | F1: simple mean of selected values |
| `MLPMeanAggregator` | F1: learned MLP over (mean, log_count) |
| `ConcatMLPAggregator` | F1: MLP over explicit padded subset members (max 8) |
| `RestrictedSoftmaxAggregator` | F1: softmax weights over selected support only |
| `TransformerSubsetAggregator` | F1: tiny 1-layer Transformer encoder over subset members |
| `NeuralMLPF2` | Learnable F2 energy function (2-layer MLP) |
| `QueryTau` | Learnable per-query threshold τ(q) for singleton penalties |
| `GibbsConfig` (dataclass) | Sampling hyperparameters (beta, steps, runs, init, etc.) |

**Key functions:**

| Function | Purpose |
|---|---|
| `_gibbs_sample_subsets()` | Core Gibbs sampling loop (Algorithm 1). ~200 lines. Runs parallel MCMC chains |
| `_apply_f1_to_support()` | Routes sampled stats to the chosen F1 aggregator |
| `_deterministic_full_set_output()` | Fast path when f2=full_set (skips sampling entirely) |
| `build_f1()` | Factory that builds the appropriate F1 aggregator |
| `_pack_selected_members()` | Packs selected k/v into padded tensors for concat-style F1 |

**F2 energy functions** (7 variants):

| F2 Type | Description |
|---|---|
| `full_set` | Deterministic: always select all tokens (recovers standard attention) |
| `modular_dot` | Sum of query-key dot products for selected tokens |
| `modular_dot_hard_singleton` | + penalty if more than 1 token selected |
| `modular_dot_first_free` | + penalty for each token beyond the first |
| `dot_repulsion` | Dot products minus λ × pairwise key similarity (encourages diversity) |
| `logsumexp` | Log-sum-exp pooling of attention scores |
| `neural_mlp` | Fully learned 2-layer MLP energy |

**F1 aggregation functions** (5 variants):

| F1 Type | Description |
|---|---|
| `mean` | Simple mean of selected values |
| `mlp_mean` | MLP over (mean, log(1+count)) |
| `mlp_concat` | MLP over explicit subset members (padded, max 8) |
| `restricted_softmax` | Softmax attention restricted to selected tokens |
| `transformer` | Tiny Transformer encoder over subset members |

---

### `train_vit_cifar.py` (~33 KB) — Training Pipeline

Trains a **tiny Vision Transformer (ViT)** on CIFAR-10 or CIFAR-100 with either standard MHA or GAM attention.

**Model architecture (TinyViT):**

```
Image (B, 3, 32, 32)
  → PatchEmbed: Conv2d(3, 192, kernel=4, stride=4) → (B, 64, 192)
  → Prepend [CLS] token + positional embeddings → (B, 65, 192)
  → 6× TransformerBlock:
      → LayerNorm → Self-Attention (MHA or GAM, 3 heads) → residual
      → LayerNorm → MLP (192→768→192, GELU) → residual
  → LayerNorm → [CLS] token → Linear(192, num_classes)
  → logits (B, 10 or 100)
```

**Training details:**
- Optimizer: AdamW (lr=5e-4, weight_decay=0.05)
- Scheduler: cosine annealing with linear warmup (5 epochs)
- AMP (mixed precision) support
- Grad clipping at 1.0
- Outputs: JSON file with per-epoch metrics (loss, accuracy, throughput)

**Key CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | cifar10 | cifar10 or cifar100 |
| `--attention` | general | `mha` or `general` |
| `--f1-type` | mean | Aggregation function |
| `--f2-type` | modular_dot | Energy function |
| `--gibbs-steps` | 100 | MCMC chain length |
| `--gibbs-runs` | 30 | Independent parallel chains |
| `--gibbs-beta` | 1.0 | Inverse temperature |
| `--embed-dim` | 192 | ViT embedding dimension |
| `--depth` | 6 | Number of transformer blocks |
| `--num-heads` | 3 | Number of attention heads |
| `--epochs` | 100 | Training epochs |
| `--download` | flag | Auto-download CIFAR data |

---

### `verify_exact_attention.py` (~1 KB)

Sanity check: creates random input, runs GAM with `f2=full_set + f1=restricted_softmax`, and asserts the output matches manual `softmax(QK^T/√d)V` within 1e-6.

### `plot_training_curves.py` (~7 KB)

Reads JSON logs from training runs and generates:
- `accuracy_curves.png` — train/val accuracy over epochs
- `speed_curves.png` — throughput (images/sec) and epoch time
- `speed_summary_arch.png` — bar chart comparing MHA vs GAM speed

### `gibbs_nips2015_experiments.ipynb` (~162 KB)

Replicates 3 experiments from **Gotovos et al. (NeurIPS 2015)** on n=20 variable models:
1. **Facility-location** — influence matrix submodular model
2. **Pairwise MRF** — 2D cluster repulsion model
3. **Higher-order MRF** — cascade on Watts-Strogatz graph

Uses synthetic data as a fallback (original water-network matrix not included).

### `sweeps/` — Hyperparameter Sweep Scripts

Shell scripts that run `train_vit_cifar.py` across F1/F2 combinations:
- `sweep_restricted_softmax_cifar10.sh` — varies F2 with fixed F1=restricted_softmax
- `sweep_f1_f2_cifar10.sh` — Cartesian product of F1 × F2

---

## Gibbs Sampling Algorithm

```
Input: query q, keys K, values V, config (β, gibbs_steps, runs)

For each of 'runs' independent chains:
    Initialize S = ∅ (or random)
    For t = 1 to gibbs_steps:
        Pick random position i ∈ {1, ..., L}
        δ = F2(S ∪ {i}) - F2(S \ {i})        # marginal gain
        p = sigmoid(β · δ)                     # inclusion probability
        Sample: add i to S with prob p, remove otherwise
        Update running stats (sum_v, count)     # for F1

    output_chain = F1(running_stats)

Output = mean over all chains
```

Gradients through the hard Bernoulli sampling use a **straight-through estimator** (hard decisions forward, soft sigmoid backward).

---

## Dependencies

From `requirements.txt`:
- `torch==2.10.0`, `torchvision==0.25.0`
- `matplotlib==3.10.8`, `numpy==2.4.2`
- `sympy==1.14.0`, `mpmath==1.3.0`
- `networkx==3.6.1` (for notebook experiments)

---

## Quick Start

```bash
# Install dependencies
cd /Users/tony/Projects/Choro/GAM
uv pip install -r requirements.txt

# Verify exact attention recovery
python verify_exact_attention.py

# Train with standard attention baseline (MHA)
python train_vit_cifar.py --dataset cifar10 --attention mha --download --epochs 10

# Train with GAM (simple subset attention)
python train_vit_cifar.py --dataset cifar10 --attention general \
  --f1-type mean --f2-type modular_dot \
  --gibbs-steps 64 --gibbs-runs 4 --download --epochs 10

# Train with GAM (exact attention recovery mode)
python train_vit_cifar.py --dataset cifar10 --attention general \
  --f1-type restricted_softmax --f2-type full_set --f1-query-mode none \
  --download --epochs 10

# Run a full sweep
bash sweeps/sweep_restricted_softmax_cifar10.sh
```

---

## Known Gaps and Issues

1. **No documentation** — README is title-only, no docstrings, very few inline comments
2. **No multi-GPU support** — no DDP/FSDP, single-device only
3. **Large monolithic functions** — `_gibbs_sample_subsets()` is ~200 lines with no decomposition
4. **Incomplete feature exploration** — `f1_query_mode` ("replace"/"add") exists but is untested in sweeps
5. **Reproducibility** — no `torch.backends.cudnn.deterministic=True`, Jupyter notebook uses synthetic data fallback
6. **No pre-trained checkpoints** or example results included
7. **Learned tau(q)** — implemented but marked as optional/experimental (`--disable-learned-tau`)

---

## Architecture Diagram

```
                    ┌──────────────────────────────┐
                    │   GeneralAttention Module     │
                    │                               │
  x ──→ W_Q ──→ Q ─┤                               │
  x ──→ W_K ──→ K ─┤  ┌─────────────────────────┐  │
  x ──→ W_V ──→ V ─┤  │  f2_type == "full_set"? │  │
                    │  └──────┬──────────┬────────┘  │
                    │      YES│          │NO         │
                    │         ▼          ▼           │
                    │  ┌───────────┐ ┌────────────┐  │
                    │  │Deterministic│ │Gibbs Sampler│ │
                    │  │ full-set  │ │(Algorithm 1)│  │
                    │  │  path     │ │ β, steps,  │  │
                    │  │           │ │ runs chains │  │
                    │  └─────┬─────┘ └──────┬─────┘  │
                    │        │              │         │
                    │        ▼              ▼         │
                    │  ┌──────────────────────────┐  │
                    │  │   F1 Aggregation          │  │
                    │  │ (mean/mlp/softmax/xfmr)   │  │
                    │  └────────────┬──────────────┘  │
                    │               │                 │
                    └───────────────┼─────────────────┘
                                    ▼
                              output (B, L, d_v)
```
