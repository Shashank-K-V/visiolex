"""PyTorch Dataset for the GRID Corpus.

GRID directory layout expected
-------------------------------
  <grid_root>/
    s1/
      video/
        mpg_6000/
          bbaf2n.mpg
          ...
      align/
        bbaf2n.align
        ...
    s2/
      ...

Each ``.align`` file contains one word per line with start/end frame numbers:
  0 23750 sil
  23750 29375 bin
  29375 35000 blue
  ...
  94000 100000 sil

We concatenate the non-silence words (lowercased) as the label string.

Pre-processed mode
------------------
If a ``processed_dir`` is given and pre-extracted ``.npy`` crops exist there,
the dataset loads them directly (fast).  Otherwise it runs MediaPipe on the fly
(slow but fine for prototyping).

Processed file path:  ``<processed_dir>/<speaker>/<clip_id>.npy``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.text import encode_text
from .preprocessing import MouthCropExtractor


def _parse_align(path: Path) -> str:
    """Return the sentence label from a .align file (silence stripped)."""
    words: List[str] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2].lower() not in ("sil", "sp"):
                words.append(parts[2].lower())
    return " ".join(words)


def _discover_clips(
    grid_root: Path,
    speakers: Optional[List[int]] = None,
) -> List[Tuple[Path, Path, str]]:
    """Return list of (video_path, align_path, speaker_id) triples."""
    clips: List[Tuple[Path, Path, str]] = []
    pattern = [f"s{i}" for i in speakers] if speakers else None

    for spk_dir in sorted(grid_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        if pattern and spk_dir.name not in pattern:
            continue

        video_dir = spk_dir / "video" / "mpg_6000"
        align_dir = spk_dir / "align"

        if not video_dir.exists() or not align_dir.exists():
            continue

        for vid_path in sorted(video_dir.glob("*.mpg")):
            align_path = align_dir / (vid_path.stem + ".align")
            if align_path.exists():
                clips.append((vid_path, align_path, spk_dir.name))

    return clips


class GRIDDataset(Dataset):
    """GRID Corpus dataset returning (frames_tensor, label_tensor) pairs.

    Args:
        grid_root: Path to the GRID corpus root directory.
        processed_dir: Path to pre-processed ``.npy`` crops.  If a clip's
            ``.npy`` file exists here it is loaded instead of running MediaPipe.
            Pass ``None`` to always run MediaPipe (slow).
        speakers: List of speaker integers to include, e.g. ``[1, 2, 3]``.
            ``None`` = all available speakers.
        num_frames: Number of frames per clip (clips are padded/trimmed).
        img_size: Spatial size of each mouth crop (pixels).
        transform: Optional callable applied to the float32 frames array
            before converting to tensor (e.g. augmentation).
    """

    def __init__(
        self,
        grid_root: str | Path,
        processed_dir: Optional[str | Path] = None,
        speakers: Optional[List[int]] = None,
        num_frames: int = 75,
        img_size: int = 64,
        transform=None,
    ) -> None:
        self.grid_root = Path(grid_root)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform

        self._clips = _discover_clips(self.grid_root, speakers)
        if not self._clips:
            raise FileNotFoundError(
                f"No GRID clips found under {self.grid_root}. "
                "Check the path and that video/align sub-directories exist."
            )

        # Lazy MediaPipe extractor — created on first __getitem__ call to avoid
        # forking issues when num_workers > 0.
        self._extractor: Optional[MouthCropExtractor] = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        vid_path, align_path, speaker = self._clips[idx]

        frames = self._load_frames(vid_path, speaker)
        label_str = _parse_align(align_path)
        label_ids = encode_text(label_str)

        if self.transform is not None:
            frames = self.transform(frames)

        # frames: (T, H, W) -> (1, T, H, W)  (channel-first for 3-D CNN)
        frames_tensor = torch.from_numpy(frames).unsqueeze(0)
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        return {
            "frames": frames_tensor,          # (1, T, H, W) float32
            "label": label_tensor,             # (L,) long
            "label_len": torch.tensor(len(label_ids), dtype=torch.long),
            "label_str": label_str,
            "video_path": str(vid_path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_frames(self, vid_path: Path, speaker: str) -> np.ndarray:
        """Return (T, H, W) float32 array for the clip."""
        if self.processed_dir is not None:
            npy_path = self.processed_dir / speaker / (vid_path.stem + ".npy")
            if npy_path.exists():
                return np.load(str(npy_path))

        # Fallback: extract on the fly
        if self._extractor is None:
            self._extractor = MouthCropExtractor(
                img_size=self.img_size,
                num_frames=self.num_frames,
            )
        frames = self._extractor.extract_from_video(vid_path)
        if frames is None:
            # Return blank frames if MediaPipe fails
            return np.zeros((self.num_frames, self.img_size, self.img_size),
                            dtype=np.float32)
        return frames
