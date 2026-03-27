"""Mouth Region Of Interest (ROI) extraction using MediaPipe Face Landmarker.

MediaPipe 0.10+ uses the Tasks API (mp.tasks) instead of mp.solutions.
The face_landmarker.task model file is downloaded automatically on first run
and cached at <repo_root>/models/face_landmarker.task.

For each video frame we:
  1. Detect 478 facial landmarks with MediaPipe Face Landmarker.
  2. Use the outer-lip landmark indices to compute a tight bounding box.
  3. Add a padding margin, crop, and resize to ``img_size × img_size``.
  4. Convert to grayscale and normalise to [0, 1].

Landmark references
-------------------
MediaPipe FaceMesh outer lip landmarks (same indices as the legacy API):
  upper_lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
  lower_lip: 146, 91, 181, 84, 17, 314, 405, 321, 375
"""

from __future__ import annotations

import urllib.request
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Model download URL (official MediaPipe hosted model, ~1 MB)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"

# Outer-lip landmark indices (MediaPipe 478-point topology)
_OUTER_LIP_IDXS: List[int] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
]

_MARGIN_RATIO = 0.30  # padding around the tight lip bbox


def _ensure_model() -> str:
    """Download the face landmarker model if not already cached."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        print(f"Downloading face landmarker model (~1 MB) → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        print("Download complete.")
    return str(_MODEL_PATH)


class MouthCropExtractor:
    """Extract mouth ROI frames from video using MediaPipe Face Landmarker.

    Args:
        img_size: Side length (pixels) of the square output crop.
        num_frames: Fixed number of frames to sample / pad to.
    """

    def __init__(self, img_size: int = 64, num_frames: int = 75) -> None:
        self.img_size = img_size
        self.num_frames = num_frames

        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_video(self, video_path) -> Optional[np.ndarray]:
        """Read a video file and return an array of mouth crop frames.

        Returns:
            Float32 array of shape ``(num_frames, img_size, img_size)`` with
            values in ``[0, 1]``, or ``None`` if the video cannot be opened.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            warnings.warn(f"Cannot open video: {video_path}")
            return None

        raw_frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            crop = self.extract_from_frame(frame)
            if crop is not None:
                raw_frames.append(crop)
        cap.release()

        if not raw_frames:
            warnings.warn(f"No mouth detected in: {video_path}")
            return None

        return self._pad_or_trim(raw_frames)

    def extract_from_frame(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single mouth crop from a BGR frame.

        Returns:
            Float32 array of shape ``(img_size, img_size)`` or ``None`` if no
            face is detected.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w = bgr_frame.shape[:2]

        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)]
             for i in _OUTER_LIP_IDXS],
            dtype=np.int32,
        )

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        bw = x_max - x_min
        bh = y_max - y_min
        margin_x = int(bw * _MARGIN_RATIO)
        margin_y = int(bh * _MARGIN_RATIO)

        x1 = max(0, x_min - margin_x)
        y1 = max(0, y_min - margin_y)
        x2 = min(w, x_max + margin_x)
        y2 = min(h, y_max + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = bgr_frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "MouthCropExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_or_trim(self, frames: List[np.ndarray]) -> np.ndarray:
        """Pad (repeat-last) or centre-trim to exactly ``self.num_frames``."""
        T = self.num_frames
        n = len(frames)
        arr = np.stack(frames, axis=0)  # (n, H, W)

        if n == T:
            return arr
        if n > T:
            start = (n - T) // 2
            return arr[start: start + T]
        pad = np.repeat(arr[-1:], T - n, axis=0)
        return np.concatenate([arr, pad], axis=0)
