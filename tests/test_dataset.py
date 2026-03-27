"""Tests for data utilities (no GRID corpus required)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import (
    Compose,
    RandomBrightnessJitter,
    RandomCrop,
    RandomHorizontalFlip,
    RandomTemporalJitter,
    build_train_transforms,
    build_val_transforms,
)
from src.data.preprocessing import MouthCropExtractor


# ------------------------------------------------------------------ #
# Augmentation tests                                                   #
# ------------------------------------------------------------------ #

def _random_frames(T=75, H=64, W=64) -> np.ndarray:
    return np.random.rand(T, H, W).astype(np.float32)


class TestAugmentations:
    def test_flip_shape_preserved(self):
        frames = _random_frames()
        out = RandomHorizontalFlip(p=1.0)(frames)
        assert out.shape == frames.shape

    def test_flip_is_actually_flipped(self):
        frames = _random_frames()
        out = RandomHorizontalFlip(p=1.0)(frames)
        np.testing.assert_array_equal(out, frames[:, :, ::-1])

    def test_brightness_clip(self):
        frames = _random_frames()
        out = RandomBrightnessJitter(delta=0.5)(frames)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_random_crop_shape_preserved(self):
        frames = _random_frames()
        out = RandomCrop(pad=4)(frames)
        assert out.shape == frames.shape

    def test_temporal_jitter_shape_preserved(self):
        frames = _random_frames()
        out = RandomTemporalJitter(max_drop=5, num_frames=75)(frames)
        assert out.shape == (75, 64, 64)

    def test_compose_pipeline(self):
        frames = _random_frames()
        transform = build_train_transforms(num_frames=75)
        out = transform(frames)
        assert out.shape == frames.shape
        assert out.dtype == np.float32

    def test_val_transform_identity(self):
        frames = _random_frames()
        out = build_val_transforms()(frames)
        np.testing.assert_array_equal(out, frames)


# ------------------------------------------------------------------ #
# MouthCropExtractor: pad/trim logic only (no video needed)           #
# ------------------------------------------------------------------ #

class TestMouthCropExtractorPadTrim:
    @pytest.fixture
    def extractor(self):
        return MouthCropExtractor(img_size=64, num_frames=75)

    def test_trim_long_sequence(self, extractor):
        frames = [np.zeros((64, 64), dtype=np.float32)] * 100
        out = extractor._pad_or_trim(frames)
        assert out.shape == (75, 64, 64)

    def test_pad_short_sequence(self, extractor):
        frames = [np.zeros((64, 64), dtype=np.float32)] * 30
        out = extractor._pad_or_trim(frames)
        assert out.shape == (75, 64, 64)

    def test_exact_length_unchanged(self, extractor):
        frames = [np.ones((64, 64), dtype=np.float32)] * 75
        out = extractor._pad_or_trim(frames)
        assert out.shape == (75, 64, 64)
        assert out.sum() == 75 * 64 * 64
