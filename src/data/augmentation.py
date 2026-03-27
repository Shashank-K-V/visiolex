"""Data augmentation for mouth-crop frame sequences.

Each transform accepts and returns a ``numpy.ndarray`` of shape ``(T, H, W)``
with values in ``[0, 1]``.  They can be composed with ``Compose``.
"""

from __future__ import annotations

import random
from typing import Callable, List, Sequence

import cv2
import numpy as np


class Compose:
    """Apply a list of transforms in order."""

    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            frames = t(frames)
        return frames


class RandomHorizontalFlip:
    """Flip the sequence left-right with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return frames[:, :, ::-1].copy()
        return frames


class RandomBrightnessJitter:
    """Multiply pixel values by a random factor in ``[1-delta, 1+delta]``."""

    def __init__(self, delta: float = 0.15) -> None:
        self.delta = delta

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        factor = 1.0 + random.uniform(-self.delta, self.delta)
        return np.clip(frames * factor, 0.0, 1.0).astype(np.float32)


class RandomCrop:
    """Pad then randomly crop each frame to original size.

    This adds ``pad`` pixels of zero-padding and takes a random crop,
    simulating small spatial jitter.
    """

    def __init__(self, pad: int = 4) -> None:
        self.pad = pad

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        T, H, W = frames.shape
        p = self.pad
        padded = np.pad(frames, ((0, 0), (p, p), (p, p)), mode="edge")
        top = random.randint(0, 2 * p)
        left = random.randint(0, 2 * p)
        return padded[:, top: top + H, left: left + W].copy()


class RandomTemporalJitter:
    """Randomly drop or repeat a small number of frames.

    The output is still trimmed/padded to ``num_frames`` so tensor shapes
    remain consistent.
    """

    def __init__(self, max_drop: int = 3, num_frames: int = 75) -> None:
        self.max_drop = max_drop
        self.num_frames = num_frames

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        T = frames.shape[0]
        n_drop = random.randint(0, min(self.max_drop, T - 1))
        if n_drop == 0:
            return frames
        drop_idxs = sorted(random.sample(range(T), n_drop), reverse=True)
        frames = np.delete(frames, drop_idxs, axis=0)
        # Pad back to num_frames
        if len(frames) < self.num_frames:
            pad = np.repeat(frames[-1:], self.num_frames - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        return frames[: self.num_frames]


def build_train_transforms(num_frames: int = 75) -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomBrightnessJitter(delta=0.15),
        RandomCrop(pad=4),
        RandomTemporalJitter(max_drop=3, num_frames=num_frames),
    ])


def build_val_transforms() -> Compose:
    """No augmentation for validation — identity transform."""
    return Compose([])
