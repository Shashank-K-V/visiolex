#!/usr/bin/env python
"""Pre-process GRID Corpus into mouth-crop numpy arrays.

Iterates over all (or specified) speakers and clips, runs MediaPipe FaceMesh,
and saves the resulting (T, H, W) float32 arrays as ``.npy`` files.

Running once saves significant time during training since MediaPipe is the
bottleneck (not the GPU).  Pre-processed files are stored under
``<processed_dir>/<speaker>/<clip_id>.npy``.

Usage
-----
  # All speakers, default settings
  python scripts/preprocess_grid.py \\
      --grid_root data/grid \\
      --processed_dir data/processed

  # Only speakers 1–5, 48 frames, 56 px crops
  python scripts/preprocess_grid.py \\
      --grid_root data/grid \\
      --processed_dir data/processed \\
      --speakers 1 2 3 4 5 \\
      --num_frames 48 \\
      --img_size 56
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import MouthCropExtractor
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-process GRID Corpus mouth crops")
    p.add_argument("--grid_root", required=True,
                   help="Path to the GRID corpus root (contains s1/, s2/, …)")
    p.add_argument("--processed_dir", required=True,
                   help="Output directory for .npy files")
    p.add_argument("--speakers", nargs="+", type=int, default=None,
                   help="Speaker IDs to process (default: all)")
    p.add_argument("--num_frames", type=int, default=75)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process even if .npy already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid_root = Path(args.grid_root)
    out_root = Path(args.processed_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect speaker directories
    spk_filter = {f"s{i}" for i in args.speakers} if args.speakers else None
    spk_dirs = [
        d for d in sorted(grid_root.iterdir())
        if d.is_dir() and (spk_filter is None or d.name in spk_filter)
    ]

    if not spk_dirs:
        print(f"No speaker directories found under {grid_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(spk_dirs)} speaker(s)…")

    with MouthCropExtractor(img_size=args.img_size, num_frames=args.num_frames) as extractor:
        for spk_dir in spk_dirs:
            video_dir = spk_dir / "video" / "mpg_6000"
            if not video_dir.exists():
                video_dir = spk_dir / "video"   # flat layout (s2–s10)
            if not video_dir.exists():
                print(f"  Skipping {spk_dir.name}: no video dir found")
                continue

            out_spk = out_root / spk_dir.name
            out_spk.mkdir(parents=True, exist_ok=True)

            clips = sorted(video_dir.glob("*.mpg"))
            success = fail = skip = 0

            for clip in tqdm(clips, desc=spk_dir.name, leave=False):
                out_path = out_spk / (clip.stem + ".npy")
                if out_path.exists() and not args.overwrite:
                    skip += 1
                    continue

                frames = extractor.extract_from_video(clip)
                if frames is None:
                    fail += 1
                    continue

                np.save(str(out_path), frames)
                success += 1

            print(
                f"  {spk_dir.name}: {success} saved, "
                f"{skip} skipped, {fail} failed"
            )

    print("Done.")


if __name__ == "__main__":
    main()
