"""SilentRead training loop.

Usage example
-------------
>>> from src.training import Trainer
>>> trainer = Trainer(model, train_loader, val_loader, config)
>>> trainer.fit()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..decoding.decoder import GreedyCTCDecoder
from ..utils.logging import AverageMeter, get_logger
from ..utils.text import BLANK_IDX
from .ctc_loss import CTCLossWrapper

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _word_error_rate(pred_strings, true_strings) -> float:
    """Macro-averaged WER over a batch."""
    import editdistance

    total_dist = 0
    total_words = 0
    for pred, true in zip(pred_strings, true_strings):
        ref_words = true.split()
        hyp_words = pred.split()
        total_dist += editdistance.eval(ref_words, hyp_words)
        total_words += max(len(ref_words), 1)
    return total_dist / max(total_words, 1)


class Trainer:
    """Manages training, validation, checkpointing, and W&B logging.

    Args:
        model: The ``SilentReadModel`` instance.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        config: Dict of training hyper-parameters (matches ``configs/train.yaml``
                training / wandb sections).
        device: ``"cuda"`` or ``"cpu"``; auto-detected if ``None``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: Optional[str] = None,
    ) -> None:
        self.logger = get_logger("silentread.trainer")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Training hyper-parameters
        tc = config.get("training", {})
        self.epochs: int = tc.get("epochs", 50)
        self.lr: float = tc.get("lr", 3e-4)
        self.weight_decay: float = tc.get("weight_decay", 1e-5)
        self.grad_clip: float = tc.get("grad_clip", 5.0)
        self.log_interval: int = tc.get("log_interval", 20)
        self.checkpoint_dir = Path(tc.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = CTCLossWrapper(blank_idx=BLANK_IDX)

        # Optimiser
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        sched_type = tc.get("scheduler", "cosine")
        warmup = tc.get("warmup_epochs", 2)
        if sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(self.epochs - warmup, 1), eta_min=1e-6
            )
        elif sched_type == "step":
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            self.scheduler = None
        self._warmup_epochs = warmup

        # Decoder for WER computation
        self.decoder = GreedyCTCDecoder(blank_idx=BLANK_IDX)

        # W&B
        wc = config.get("wandb", {})
        self._use_wandb = wc.get("enabled", False) and _WANDB_AVAILABLE
        if self._use_wandb:
            wandb.init(
                project=wc.get("project", "silentread"),
                entity=wc.get("entity", None),
                config=config,
            )

        self.best_val_wer: float = float("inf")

    # ---------------------------------------------------------------------- #
    # Public                                                                  #
    # ---------------------------------------------------------------------- #

    def fit(self) -> None:
        """Run the full training loop."""
        self.logger.info(
            f"Starting training on {self.device} — "
            f"{self.model.num_parameters:,} trainable parameters"
        )
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_wer = self._val_epoch(epoch)

            # LR scheduling (skip warmup)
            if self.scheduler is not None and epoch > self._warmup_epochs:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch:3d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_wer={val_wer:.3f}  lr={lr:.2e}"
            )

            if self._use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/wer": val_wer,
                    "lr": lr,
                })

            self._save_checkpoint(epoch, val_wer)

    # ---------------------------------------------------------------------- #
    # Private                                                                 #
    # ---------------------------------------------------------------------- #

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        meter = AverageMeter("loss")

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for step, batch in enumerate(pbar, 1):
            frames = batch["frames"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_lens = batch["label_lens"].to(self.device)

            self.optimizer.zero_grad()
            log_probs = self.model(frames)                     # (T, B, vocab)
            loss = self.criterion(log_probs, labels, label_lens)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Warmup: linearly increase LR for the first warmup_epochs
            if epoch <= self._warmup_epochs:
                warmup_steps = self._warmup_epochs * len(self.train_loader)
                done = (epoch - 1) * len(self.train_loader) + step
                factor = done / warmup_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr * factor

            meter.update(loss.item(), n=frames.size(0))
            if step % self.log_interval == 0:
                pbar.set_postfix({"loss": f"{meter.recent_avg:.4f}"})

        return meter.avg

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        loss_meter = AverageMeter("val_loss")
        all_preds, all_refs = [], []

        for batch in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            frames = batch["frames"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_lens = batch["label_lens"].to(self.device)

            log_probs = self.model(frames)
            loss = self.criterion(log_probs, labels, label_lens)
            loss_meter.update(loss.item(), n=frames.size(0))

            preds = self.decoder.decode_batch(log_probs)
            all_preds.extend(preds)
            all_refs.extend(batch["label_strs"])

        wer = _word_error_rate(all_preds, all_refs)
        return loss_meter.avg, wer

    def _save_checkpoint(self, epoch: int, val_wer: float) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_wer": val_wer,
            "config": self.config,
        }
        # Always save latest
        torch.save(ckpt, self.checkpoint_dir / "latest.pt")

        # Save best
        if val_wer < self.best_val_wer:
            self.best_val_wer = val_wer
            torch.save(ckpt, self.checkpoint_dir / "best.pt")
            self.logger.info(f"  ✓ New best checkpoint  WER={val_wer:.3f}")
