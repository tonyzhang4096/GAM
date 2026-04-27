#!/usr/bin/env python3
"""
Train a lightweight ViT on CIFAR-10/100 with either:
  - Regular multi-head self-attention (MHA), or
  - General subset attention from general_attention.py

Example:
  python train_vit_cifar.py --dataset cifar10 --attention mha --download
  python train_vit_cifar.py --dataset cifar10 --attention general --download
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from general_attention import GeneralAttention, GibbsConfig


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def json_safe_config(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MHASelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float, proj_dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.mha(x, x, x, need_weights=False)
        return self.proj_drop(y)


class GeneralSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_qk: Optional[int],
        cfg: GibbsConfig,
        f1_type: str,
        f2_type: str,
        f1_query_mode: str,
        query_chunk_size: int,
        f1_concat_max_set_size: int,
        f1_concat_hidden: int,
        f2_neural_hidden: int,
        use_learned_tau: bool,
        tau_init: float,
        proj_dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if dim % self.num_heads != 0:
            raise ValueError("embed dim must be divisible by num_heads for general attention")
        self.head_dim = dim // self.num_heads

        if d_qk is None:
            head_d_qk = self.head_dim
        else:
            if d_qk % self.num_heads != 0:
                raise ValueError("general_d_qk must be divisible by num_heads")
            head_d_qk = d_qk // self.num_heads

        self.attn_heads = nn.ModuleList(
            [
                GeneralAttention(
                    d_model=dim,
                    d_qk=head_d_qk,
                    d_v=self.head_dim,
                    f2_type=f2_type,  # type: ignore[arg-type]
                    f1_type=f1_type,  # type: ignore[arg-type]
                    f1_query_mode=f1_query_mode,  # type: ignore[arg-type]
                    cfg=cfg,
                    query_chunk_size=query_chunk_size,
                    bias=bias,
                    use_learned_tau=use_learned_tau,
                    tau_init=tau_init,
                    f1_concat_max_set_size=f1_concat_max_set_size,
                    f1_concat_hidden=f1_concat_hidden,
                    f2_neural_hidden=f2_neural_hidden,
                )
                for _ in range(self.num_heads)
            ]
        )
        # MHA includes an output projection; mirror that here for fairer comparison.
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outs = [attn(x) for attn in self.attn_heads]
        y = torch.cat(head_outs, dim=-1)
        y = self.out_proj(y)
        return self.proj_drop(y)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        attention_type: str,
        general_cfg: GibbsConfig,
        general_d_qk: Optional[int],
        f1_type: str,
        f2_type: str,
        f1_query_mode: str,
        query_chunk_size: int,
        f1_concat_max_set_size: int,
        f1_concat_hidden: int,
        f2_neural_hidden: int,
        general_bias: bool,
        use_learned_tau: bool,
        tau_init: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attention_type == "mha":
            self.attn = MHASelfAttention(dim, num_heads, attn_dropout, dropout)
        elif attention_type == "general":
            self.attn = GeneralSelfAttention(
                dim=dim,
                num_heads=num_heads,
                d_qk=general_d_qk,
                cfg=general_cfg,
                f1_type=f1_type,
                f2_type=f2_type,
                f1_query_mode=f1_query_mode,
                query_chunk_size=query_chunk_size,
                f1_concat_max_set_size=f1_concat_max_set_size,
                f1_concat_hidden=f1_concat_hidden,
                f2_neural_hidden=f2_neural_hidden,
                use_learned_tau=use_learned_tau,
                tau_init=tau_init,
                proj_dropout=dropout,
                bias=general_bias,
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        attention_type: str,
        general_cfg: GibbsConfig,
        general_d_qk: Optional[int],
        f1_type: str,
        f2_type: str,
        f1_query_mode: str,
        query_chunk_size: int,
        f1_concat_max_set_size: int,
        f1_concat_hidden: int,
        f2_neural_hidden: int,
        general_bias: bool,
        use_learned_tau: bool,
        tau_init: float,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
        )
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    attention_type=attention_type,
                    general_cfg=general_cfg,
                    general_d_qk=general_d_qk,
                    f1_type=f1_type,
                    f2_type=f2_type,
                    f1_query_mode=f1_query_mode,
                    query_chunk_size=query_chunk_size,
                    f1_concat_max_set_size=f1_concat_max_set_size,
                    f1_concat_hidden=f1_concat_hidden,
                    f2_neural_hidden=f2_neural_hidden,
                    general_bias=general_bias,
                    use_learned_tau=use_learned_tau,
                    tau_init=tau_init,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # Skip lazy modules until first forward materializes their parameters.
            if isinstance(m.weight, torch.nn.parameter.UninitializedParameter):
                return
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if isinstance(m.bias, torch.nn.parameter.UninitializedParameter):
                    return
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        bsz = x.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits


def build_dataloaders(
    dataset_name: str,
    data_dir: Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    download: bool,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, int]:
    log("[setup] importing torchvision...")
    try:
        import torchvision
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required. Install with: pip install torchvision"
        ) from exc

    log(f"[setup] preparing transforms for dataset={dataset_name}")
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    if dataset_name == "cifar10":
        log(f"[setup] building CIFAR10 train dataset (download={download})")
        train_ds = torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=download,
        )
        log(f"[setup] building CIFAR10 val dataset (download={download})")
        test_ds = torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=download,
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        log(f"[setup] building CIFAR100 train dataset (download={download})")
        train_ds = torchvision.datasets.CIFAR100(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=download,
        )
        log(f"[setup] building CIFAR100 val dataset (download={download})")
        test_ds = torchvision.datasets.CIFAR100(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=download,
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    log(
        f"[setup] creating dataloaders: batch_size={batch_size}, eval_batch_size={eval_batch_size}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    log(
        f"[setup] dataloaders ready: train_samples={len(train_ds)}, val_samples={len(test_ds)}, "
        f"train_batches={len(train_loader)}, val_batches={len(test_loader)}"
    )
    return train_loader, test_loader, num_classes


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    epoch_seconds: float
    images_per_sec: float


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0


def count_trainable_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, torch.nn.parameter.UninitializedParameter):
            continue
        total += p.numel()
    return int(total)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip: float,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0
    start = time.perf_counter()
    n_batches = len(loader)
    log(
        f"[train] epoch {epoch}/{total_epochs}: start "
        f"(batches={n_batches}, amp={use_amp}, grad_clip={grad_clip})"
    )
    first_batch_wait_start = time.perf_counter()

    for step, (images, targets) in enumerate(loader, start=1):
        if step == 1:
            first_batch_wait = time.perf_counter() - first_batch_wait_start
            log(f"[train] epoch {epoch}/{total_epochs}: first batch loaded after {first_batch_wait:.2f}s")
        step_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bsz = images.shape[0]

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item() * bsz
        total_correct += (logits.argmax(dim=1) == targets).float().sum().item()
        total_seen += bsz
        step_seconds = time.perf_counter() - step_start

        should_log_step = (
            step == 1
            or step == n_batches
            or (log_every_batches > 0 and step % log_every_batches == 0)
        )
        if should_log_step:
            running_loss = total_loss / max(total_seen, 1)
            running_acc = (total_correct / max(total_seen, 1)) * 100.0
            lr = optimizer.param_groups[0]["lr"]
            pct = step / n_batches * 100.0
            elapsed_so_far = time.perf_counter() - start
            imgs_per_sec = total_seen / max(elapsed_so_far, 1e-6)
            eta = (elapsed_so_far / step) * (n_batches - step) if step > 0 else 0.0
            log(
                f"[train] epoch {epoch}/{total_epochs} batch {step}/{n_batches} ({pct:.1f}%) "
                f"loss={running_loss:.4f} acc={running_acc:.2f}% lr={lr:.3e} "
                f"speed={imgs_per_sec:.1f} img/s ETA={eta:.0f}s"
            )

    elapsed = time.perf_counter() - start
    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = (total_correct / max(total_seen, 1)) * 100.0
    log(
        f"[train] epoch {epoch}/{total_epochs}: done "
        f"loss={avg_loss:.4f} acc={avg_acc:.2f}% epoch_s={elapsed:.2f}"
    )
    return avg_loss, avg_acc, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0
    n_batches = len(loader)
    log(f"[eval] epoch {epoch}/{total_epochs}: start (batches={n_batches})")
    first_batch_wait_start = time.perf_counter()
    eval_start = time.perf_counter()

    for step, (images, targets) in enumerate(loader, start=1):
        if step == 1:
            first_batch_wait = time.perf_counter() - first_batch_wait_start
            log(f"[eval] epoch {epoch}/{total_epochs}: first batch loaded after {first_batch_wait:.2f}s")
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bsz = images.shape[0]
        total_loss += loss.item() * bsz
        total_correct += (logits.argmax(dim=1) == targets).float().sum().item()
        total_seen += bsz

        should_log_step = (
            step == 1
            or step == n_batches
            or (log_every_batches > 0 and step % log_every_batches == 0)
        )
        if should_log_step:
            running_loss = total_loss / max(total_seen, 1)
            running_acc = (total_correct / max(total_seen, 1)) * 100.0
            pct = step / n_batches * 100.0
            elapsed_so_far = time.perf_counter() - eval_start
            imgs_per_sec = total_seen / max(elapsed_so_far, 1e-6)
            log(
                f"[eval] epoch {epoch}/{total_epochs} batch {step}/{n_batches} ({pct:.1f}%) "
                f"loss={running_loss:.4f} acc={running_acc:.2f}% speed={imgs_per_sec:.1f} img/s"
            )

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = (total_correct / max(total_seen, 1)) * 100.0
    log(f"[eval] epoch {epoch}/{total_epochs}: done loss={avg_loss:.4f} acc={avg_acc:.2f}%")
    return avg_loss, avg_acc


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(0, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps == warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tiny ViT CIFAR baseline vs general attention. "
            "Exact ordinary attention is recovered with "
            "--f1-type restricted_softmax --f2-type full_set --f1-query-mode none."
        )
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--attention", choices=["mha", "general"], default="general")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=346511053)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.0)

    parser.add_argument("--general-d-qk", type=int, default=None)
    parser.add_argument(
        "--f1-type",
        choices=["mean", "mlp_mean", "mlp_concat", "transformer", "restricted_softmax"],
        default="mean",
        help=(
            "General-attention F1. "
            "Use restricted_softmax with f2=full_set and f1_query_mode=none "
            "to recover ordinary scaled dot-product attention exactly."
        ),
    )
    parser.add_argument(
        "--f2-type",
        choices=[
            "full_set",
            "modular_dot",
            "modular_dot_hard_singleton",
            "modular_dot_first_free",
            "logsumexp",
            "dot_repulsion",
            "neural_mlp",
        ],
        default="modular_dot",
        help=(
            "General-attention F2. "
            "full_set deterministically selects all tokens and, together with "
            "f1=restricted_softmax and f1_query_mode=none, recovers ordinary attention exactly."
        ),
    )
    parser.add_argument(
        "--f1-query-mode",
        choices=["none", "replace", "add"],
        default="none",
        help=(
            "Optional query-conditioned pooling over selected members: "
            "none (disabled), replace (use only query-conditioned pool), "
            "or add (base F1 output + query-conditioned pool). "
            "restricted_softmax is already query-conditioned and requires none."
        ),
    )
    parser.add_argument(
        "--gradient-method",
        choices=["ste", "gumbel"],
        default="ste",
        help="Gradient method for discrete sampling: ste (straight-through) or gumbel (Gumbel-Softmax).",
    )
    parser.add_argument("--gumbel-tau-start", type=float, default=1.0,
                        help="Gumbel-Softmax initial temperature.")
    parser.add_argument("--gumbel-tau-min", type=float, default=0.1,
                        help="Gumbel-Softmax minimum temperature after annealing.")
    parser.add_argument("--gibbs-beta", type=float, default=1.0)
    parser.add_argument("--gibbs-steps", type=int, default=100)
    parser.add_argument("--gibbs-runs", type=int, default=30)
    parser.add_argument("--gibbs-init", choices=["empty", "random"], default="empty")
    parser.add_argument("--gibbs-init-p", type=float, default=0.5)
    parser.add_argument("--gibbs-logsumexp-eps", type=float, default=1e-6)
    parser.add_argument("--gibbs-repulsion-lambda", type=float, default=0.1)
    parser.add_argument(
        "--st-gradient-mode",
        choices=["partial", "consistent"],
        default="partial",
        help=(
            "Straight-through sampler-state gradient mode: "
            "partial (default, hard side-state updates) or consistent (ST side-state updates)."
        ),
    )
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument(
        "--f1-concat-max-set-size",
        type=int,
        default=8,
        help=(
            "Max selected subset tokens for explicit subset modules "
            "(f1=mlp_concat/transformer and f2=neural_mlp key packing)."
        ),
    )
    parser.add_argument(
        "--f1-concat-hidden",
        type=int,
        default=128,
        help="Hidden width used by f1=mlp_concat or f1=transformer.",
    )
    parser.add_argument("--f2-neural-hidden", type=int, default=128)
    parser.add_argument("--general-bias", action="store_true")
    parser.add_argument(
        "--disable-learned-tau",
        action="store_true",
        help="Disable learned per-query threshold tau(q) in general attention.",
    )
    parser.add_argument(
        "--tau-init",
        type=float,
        default=0.0,
        help="Initial bias value for learned tau(q).",
    )

    parser.add_argument("--save-dir", type=Path, default=Path("./results"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save a periodic checkpoint every N epochs (<=0 disables periodic checkpointing).",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=10,
        help="Print intra-epoch train/eval progress every N batches (<=0 disables periodic logs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("[setup] parsed args")
    set_seed(args.seed)
    log(f"[setup] seed set to {args.seed}")

    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    log(f"[setup] using device={device} (requested={args.device})")

    log(f"[setup] building dataloaders from data_dir={args.data_dir}")
    train_loader, val_loader, num_classes = build_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        download=args.download,
        pin_memory=pin_memory,
    )
    log("[setup] dataloaders built")

    cfg = GibbsConfig(
        beta=args.gibbs_beta,
        gibbs_steps=args.gibbs_steps,
        runs=args.gibbs_runs,
        init=args.gibbs_init,  # type: ignore[arg-type]
        init_p=args.gibbs_init_p,
        logsumexp_eps=args.gibbs_logsumexp_eps,
        repulsion_lambda=args.gibbs_repulsion_lambda,
        st_gradient_mode=args.st_gradient_mode,  # type: ignore[arg-type]
        gradient_method=args.gradient_method,  # type: ignore[arg-type]
        gumbel_tau=args.gumbel_tau_start,
        gumbel_tau_min=args.gumbel_tau_min,
    )
    log(
        f"[setup] GibbsConfig: beta={cfg.beta}, steps={cfg.gibbs_steps}, runs={cfg.runs}, "
        f"init={cfg.init}, init_p={cfg.init_p}, st_gradient_mode={cfg.st_gradient_mode}, "
        f"gradient_method={cfg.gradient_method}, gumbel_tau={cfg.gumbel_tau}"
    )

    log("[setup] building model")
    model = TinyViT(
        num_classes=num_classes,
        img_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        attention_type=args.attention,
        general_cfg=cfg,
        general_d_qk=args.general_d_qk,
        f1_type=args.f1_type,
        f2_type=args.f2_type,
        f1_query_mode=args.f1_query_mode,
        query_chunk_size=args.query_chunk_size,
        f1_concat_max_set_size=args.f1_concat_max_set_size,
        f1_concat_hidden=args.f1_concat_hidden,
        f2_neural_hidden=args.f2_neural_hidden,
        general_bias=args.general_bias,
        use_learned_tau=not args.disable_learned_tau,
        tau_init=args.tau_init,
    ).to(device)
    log("[setup] model built and moved to device")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    log(
        f"[setup] optimizer/scheduler ready: lr={args.lr}, wd={args.weight_decay}, "
        f"warmup_epochs={args.warmup_epochs}, amp={use_amp}"
    )

    n_params = count_trainable_params(model)
    log(f"device={device}, attention={args.attention}, dataset={args.dataset}, params={n_params:,}")
    if args.attention == "general":
        log(
            "general attention config:"
            f" f1={args.f1_type}, f2={args.f2_type}, f1_query={args.f1_query_mode}, "
            f"steps={args.gibbs_steps}, runs={args.gibbs_runs}, beta={args.gibbs_beta}, "
            f"learned_tau={not args.disable_learned_tau}, tau_init={args.tau_init}"
        )
        if (
            args.f1_type == "restricted_softmax"
            and args.f2_type == "full_set"
            and args.f1_query_mode == "none"
        ):
            log("[setup] exact_attention_special_case=True")

    metrics: list[EpochMetrics] = []
    best_val = -1.0
    best_epoch = -1
    run_start = time.perf_counter()
    run_name = args.run_name.strip()
    if not run_name:
        run_name = (
            f"{args.dataset}_{args.attention}"
            f"_d{args.embed_dim}_L{args.depth}"
            f"_seed{args.seed}"
        )
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if cfg.gradient_method == "gumbel":
            progress = (epoch - 1) / max(args.epochs - 1, 1)
            cfg.gumbel_tau = args.gumbel_tau_start - progress * (args.gumbel_tau_start - args.gumbel_tau_min)
            cfg.gumbel_tau = max(cfg.gumbel_tau, args.gumbel_tau_min)
            log(f"[epoch] gumbel_tau={cfg.gumbel_tau:.4f}")
        log(f"[epoch] ===== epoch {epoch}/{args.epochs} =====")
        train_loss, train_acc, epoch_seconds = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=scaler,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=args.epochs,
            log_every_batches=args.log_every_batches,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_every_batches=args.log_every_batches,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        imgs_sec = (len(train_loader) * args.batch_size) / max(epoch_seconds, 1e-9)
        rec = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
            epoch_seconds=epoch_seconds,
            images_per_sec=imgs_sec,
        )
        metrics.append(rec)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch

        log(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"lr={current_lr:.3e} imgs/s={imgs_sec:.1f}"
        )

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            periodic_ckpt_path = args.save_dir / f"{run_name}_epoch{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_acc": best_val,
                    "best_epoch": best_epoch,
                    "config": json_safe_config(args),
                },
                periodic_ckpt_path,
            )
            log(f"[ckpt] wrote periodic checkpoint: {periodic_ckpt_path}")

    total_seconds = time.perf_counter() - run_start
    summary_path = args.save_dir / f"{run_name}.json"
    out = {
        "run_name": run_name,
        "dataset": args.dataset,
        "attention": args.attention,
        "device": str(device),
        "num_params": n_params,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "total_seconds": total_seconds,
        "config": json_safe_config(args),
        "metrics": [m.__dict__ for m in metrics],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    log(f"done: best_val_acc={best_val:.2f}% at epoch {best_epoch}")
    log(f"wrote summary: {summary_path}")

    if args.save_checkpoint:
        ckpt_path = args.save_dir / f"{run_name}.pt"
        torch.save({"model": model.state_dict(), "config": json_safe_config(args)}, ckpt_path)
        log(f"wrote checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
