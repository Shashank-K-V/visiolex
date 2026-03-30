#!/usr/bin/env python
"""SilentRead training entry-point.

Usage
-----
  # Train with default config
  python scripts/train.py --config configs/train.yaml

  # Override specific fields
  python scripts/train.py \\
      --config configs/train.yaml \\
      --grid_root data/grid \\
      --processed_dir data/processed \\
      --epochs 30 \\
      --batch_size 8 \\
      --speakers 1 2 3

  # Resume from checkpoint
  python scripts/train.py \\
      --config configs/train.yaml \\
      --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import build_dataloaders
from src.models import SilentReadModel
from src.training import Trainer
from src.utils import get_logger
from src.utils.text import BLANK_IDX, VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SilentRead")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--grid_root", default=None,
                   help="Override config data.grid_root")
    p.add_argument("--processed_dir", default=None,
                   help="Override config data.processed_dir")
    p.add_argument("--speakers", nargs="+", type=int, default=None,
                   help="Restrict to these speaker IDs")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint .pt file to resume from")
    p.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"])
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging regardless of config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("silentread.train")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.grid_root:
        config["data"]["grid_root"] = args.grid_root
    if args.processed_dir:
        config["data"]["processed_dir"] = args.processed_dir
    if args.speakers:
        config["data"]["train_speakers"] = args.speakers
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.no_wandb:
        config["wandb"]["enabled"] = False

    dc = config["data"]
    tc = config["training"]
    mc = config["model"]

    # Data
    logger.info("Building data loaders…")
    train_loader, val_loader = build_dataloaders(
        grid_root=dc["grid_root"],
        processed_dir=dc.get("processed_dir"),
        speakers=dc.get("train_speakers"),
        val_split=dc.get("val_split", 0.1),
        num_frames=dc.get("num_frames", 75),
        img_size=dc.get("img_size", 64),
        batch_size=tc.get("batch_size", 16),
        num_workers=dc.get("num_workers", 4),
    )
    logger.info(
        f"  Train batches: {len(train_loader)}  "
        f"Val batches: {len(val_loader)}"
    )

    # Model
    model = SilentReadModel(
        vocab_size=VOCAB_SIZE,
        cnn_channels=mc.get("cnn_channels", [32, 64, 96]),
        gru_hidden=mc.get("gru_hidden", 256),
        gru_layers=mc.get("gru_layers", 2),
        dropout=mc.get("dropout", 0.5),
    )
    logger.info(f"  Model parameters: {model.num_parameters:,}")

    # Optionally resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(
            f"  Resumed from {args.resume} "
            f"(epoch {ckpt.get('epoch')}, WER={ckpt.get('val_wer', '?'):.3f})"
        )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
