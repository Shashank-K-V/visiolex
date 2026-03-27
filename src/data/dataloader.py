"""DataLoader factory for VisioLex.

Handles variable-length label sequences by padding them in ``collate_fn``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split, Subset

from .augmentation import build_train_transforms, build_val_transforms
from .dataset import GRIDDataset


def _collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    """Collate a list of samples into a batched dict.

    Labels are padded with 0 to the length of the longest label in the batch.
    """
    frames = torch.stack([s["frames"] for s in batch])              # (B,1,T,H,W)
    label_lens = torch.stack([s["label_len"] for s in batch])       # (B,)

    max_len = int(label_lens.max().item())
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        lbl = s["label"]
        labels_padded[i, : len(lbl)] = lbl

    return {
        "frames": frames,
        "labels": labels_padded,
        "label_lens": label_lens,
        "label_strs": [s["label_str"] for s in batch],
        "video_paths": [s["video_path"] for s in batch],
    }


def build_dataloaders(
    grid_root: str | Path,
    processed_dir: Optional[str | Path] = None,
    speakers: Optional[List[int]] = None,
    val_split: float = 0.1,
    num_frames: int = 75,
    img_size: int = 64,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Args:
        grid_root: Path to GRID corpus root.
        processed_dir: Optional directory with pre-extracted ``.npy`` crops.
        speakers: Speaker IDs to include (``None`` = all).
        val_split: Fraction of data used for validation.
        num_frames: Frames per clip.
        img_size: Crop size in pixels.
        batch_size: Batch size.
        num_workers: DataLoader worker count.
        seed: Random seed for the train/val split.

    Returns:
        ``(train_loader, val_loader)``
    """
    train_dataset = GRIDDataset(
        grid_root=grid_root,
        processed_dir=processed_dir,
        speakers=speakers,
        num_frames=num_frames,
        img_size=img_size,
        transform=build_train_transforms(num_frames),
    )
    val_dataset = GRIDDataset(
        grid_root=grid_root,
        processed_dir=processed_dir,
        speakers=speakers,
        num_frames=num_frames,
        img_size=img_size,
        transform=build_val_transforms(),
    )

    n_total = len(train_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    rng = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_val], generator=rng
    )

    train_subset = Subset(train_dataset, list(train_idx))
    val_subset = Subset(val_dataset, list(val_idx))

    loader_kwargs = dict(
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader
