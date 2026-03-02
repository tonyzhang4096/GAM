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
        d_qk: Optional[int],
        cfg: GibbsConfig,
        f1_type: str,
        f2_type: str,
        query_chunk_size: int,
        f1_concat_max_set_size: int,
        f1_concat_hidden: int,
        f2_neural_hidden: int,
        proj_dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.attn = GeneralAttention(
            d_model=dim,
            d_qk=d_qk,
            d_v=dim,
            f2_type=f2_type,  # type: ignore[arg-type]
            f1_type=f1_type,  # type: ignore[arg-type]
            cfg=cfg,
            query_chunk_size=query_chunk_size,
            bias=bias,
            f1_concat_max_set_size=f1_concat_max_set_size,
            f1_concat_hidden=f1_concat_hidden,
            f2_neural_hidden=f2_neural_hidden,
        )
        # MHA includes an output projection; mirror that here for fairer comparison.
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(x)
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
        query_chunk_size: int,
        f1_concat_max_set_size: int,
        f1_concat_hidden: int,
        f2_neural_hidden: int,
        general_bias: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attention_type == "mha":
            self.attn = MHASelfAttention(dim, num_heads, attn_dropout, dropout)
        elif attention_type == "general":
            self.attn = GeneralSelfAttention(
                dim=dim,
                d_qk=general_d_qk,
                cfg=general_cfg,
                f1_type=f1_type,
                f2_type=f2_type,
                query_chunk_size=query_chunk_size,
                f1_concat_max_set_size=f1_concat_max_set_size,
                f1_concat_hidden=f1_concat_hidden,
                f2_neural_hidden=f2_neural_hidden,
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
        query_chunk_size: int,
        f1_concat_max_set_size: int,
        f1_concat_hidden: int,
        f2_neural_hidden: int,
        general_bias: bool,
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
                    query_chunk_size=query_chunk_size,
                    f1_concat_max_set_size=f1_concat_max_set_size,
                    f1_concat_hidden=f1_concat_hidden,
                    f2_neural_hidden=f2_neural_hidden,
                    general_bias=general_bias,
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
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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
    try:
        import torchvision
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required. Install with: pip install torchvision"
        ) from exc

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
        train_ds = torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=download,
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=download,
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=download,
        )
        test_ds = torchvision.datasets.CIFAR100(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=download,
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0
    start = time.perf_counter()

    for images, targets in loader:
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

    elapsed = time.perf_counter() - start
    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = (total_correct / max(total_seen, 1)) * 100.0
    return avg_loss, avg_acc, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bsz = images.shape[0]
        total_loss += loss.item() * bsz
        total_correct += (logits.argmax(dim=1) == targets).float().sum().item()
        total_seen += bsz

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = (total_correct / max(total_seen, 1)) * 100.0
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
    parser = argparse.ArgumentParser(description="Tiny ViT CIFAR baseline vs general attention")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--attention", choices=["mha", "general"], default="mha")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)

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
        choices=["mean", "mlp_mean", "mlp_concat"],
        default="mean",
    )
    parser.add_argument(
        "--f2-type",
        choices=["modular_dot", "logsumexp", "dot_repulsion", "neural_mlp"],
        default="logsumexp",
    )
    parser.add_argument("--gibbs-beta", type=float, default=1.0)
    parser.add_argument("--gibbs-steps", type=int, default=16)
    parser.add_argument("--gibbs-runs", type=int, default=2)
    parser.add_argument("--gibbs-init", choices=["empty", "random"], default="empty")
    parser.add_argument("--gibbs-init-p", type=float, default=0.5)
    parser.add_argument("--gibbs-logsumexp-eps", type=float, default=1e-6)
    parser.add_argument("--gibbs-repulsion-lambda", type=float, default=0.1)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--f1-concat-max-set-size", type=int, default=8)
    parser.add_argument("--f1-concat-hidden", type=int, default=128)
    parser.add_argument("--f2-neural-hidden", type=int, default=128)
    parser.add_argument("--general-bias", action="store_true")

    parser.add_argument("--save-dir", type=Path, default=Path("./results"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args.device)
    pin_memory = device.type == "cuda"

    train_loader, val_loader, num_classes = build_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        download=args.download,
        pin_memory=pin_memory,
    )

    cfg = GibbsConfig(
        beta=args.gibbs_beta,
        gibbs_steps=args.gibbs_steps,
        runs=args.gibbs_runs,
        init=args.gibbs_init,  # type: ignore[arg-type]
        init_p=args.gibbs_init_p,
        logsumexp_eps=args.gibbs_logsumexp_eps,
        repulsion_lambda=args.gibbs_repulsion_lambda,
    )

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
        query_chunk_size=args.query_chunk_size,
        f1_concat_max_set_size=args.f1_concat_max_set_size,
        f1_concat_hidden=args.f1_concat_hidden,
        f2_neural_hidden=args.f2_neural_hidden,
        general_bias=args.general_bias,
    ).to(device)

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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device}, attention={args.attention}, dataset={args.dataset}, params={n_params:,}")
    if args.attention == "general":
        print(
            "general attention config:"
            f" f1={args.f1_type}, f2={args.f2_type}, "
            f"steps={args.gibbs_steps}, runs={args.gibbs_runs}, beta={args.gibbs_beta}"
        )

    metrics: list[EpochMetrics] = []
    best_val = -1.0
    best_epoch = -1
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
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
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
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

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"lr={current_lr:.3e} imgs/s={imgs_sec:.1f}"
        )

    total_seconds = time.perf_counter() - run_start

    run_name = args.run_name.strip()
    if not run_name:
        run_name = (
            f"{args.dataset}_{args.attention}"
            f"_d{args.embed_dim}_L{args.depth}"
            f"_seed{args.seed}"
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
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

    print(f"done: best_val_acc={best_val:.2f}% at epoch {best_epoch}")
    print(f"wrote summary: {summary_path}")

    if args.save_checkpoint:
        ckpt_path = args.save_dir / f"{run_name}.pt"
        torch.save({"model": model.state_dict(), "config": json_safe_config(args)}, ckpt_path)
        print(f"wrote checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
