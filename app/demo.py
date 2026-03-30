"""SilentRead — Gradio demo application.

Upload a video clip → see the extracted mouth-crop frames → get the
transcription.  Optionally uses beam-search decoding if pyctcdecode is
installed, otherwise falls back to greedy.

Run locally
-----------
  python app/demo.py --checkpoint checkpoints/best.pt

Deploy to Hugging Face Spaces
------------------------------
  # Place this file at the root of a new HF Space repo (Gradio SDK).
  # Set the checkpoint path via the HF_CHECKPOINT env var or edit
  # DEFAULT_CHECKPOINT below.

Environment variables
---------------------
  SILENTREAD_CHECKPOINT  Path to model checkpoint (overrides CLI arg).
  SILENTREAD_LM_PATH     Path to KenLM .binary file (optional).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# Allow running from repo root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import gradio as gr

from src.data.preprocessing import MouthCropExtractor
from src.decoding.decoder import BeamCTCDecoder, GreedyCTCDecoder
from src.models import SilentReadModel
from src.utils.text import BLANK_IDX, VOCAB_SIZE

DEFAULT_CHECKPOINT = str(_ROOT / "checkpoints" / "best.pt")
NUM_FRAMES = 75
IMG_SIZE = 64

# ------------------------------------------------------------------ #
# Model + extractor setup (loaded once at startup)                    #
# ------------------------------------------------------------------ #

_model: Optional[SilentReadModel] = None
_extractor: Optional[MouthCropExtractor] = None
_decoder = None
_device: torch.device = torch.device("cpu")


def _load_model(checkpoint_path: str, lm_path: Optional[str] = None) -> None:
    global _model, _extractor, _decoder, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = SilentReadModel(vocab_size=VOCAB_SIZE)

    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=_device)
        state = ckpt.get("model_state_dict", ckpt)
        _model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(
            f"[WARNING] Checkpoint not found at {checkpoint_path}. "
            "Running with random weights — output will be nonsense."
        )

    _model.to(_device).eval()
    _extractor = MouthCropExtractor(img_size=IMG_SIZE, num_frames=NUM_FRAMES)
    _decoder = BeamCTCDecoder(beam_width=10, lm_path=lm_path, blank_idx=BLANK_IDX)


# ------------------------------------------------------------------ #
# Inference helpers                                                   #
# ------------------------------------------------------------------ #

def _extract_mouth_frames(video_path: str) -> Tuple[Optional[np.ndarray], list]:
    """Return (frames_array, list_of_crop_images_for_gallery)."""
    assert _extractor is not None

    cap = cv2.VideoCapture(video_path)
    raw_crops = []
    raw_bgr = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = _extractor.extract_from_frame(frame)
        if crop is not None:
            raw_crops.append(crop)
            raw_bgr.append(frame)
    cap.release()

    if not raw_crops:
        return None, []

    frames_array = _extractor._pad_or_trim(raw_crops)

    # Build gallery: show every 5th frame as a PIL image
    gallery = []
    for i in range(0, len(raw_crops), max(1, len(raw_crops) // 15)):
        img = (raw_crops[i] * 255).astype(np.uint8)
        gallery.append(img)

    return frames_array, gallery


def _run_inference(frames_array: np.ndarray) -> str:
    assert _model is not None and _decoder is not None

    # (T, H, W) → (1, 1, T, H, W)
    tensor = torch.from_numpy(frames_array).unsqueeze(0).unsqueeze(0).to(_device)
    with torch.no_grad():
        log_probs = _model(tensor)   # (T, 1, vocab)

    text = _decoder.decode(log_probs[:, 0, :])
    return text if text.strip() else "(no prediction)"


# ------------------------------------------------------------------ #
# Gradio interface                                                    #
# ------------------------------------------------------------------ #

def transcribe(video_file) -> Tuple[str, list, str]:
    """Main Gradio callback.

    Returns:
        (transcription, gallery_images, status_message)
    """
    if video_file is None:
        return "", [], "Please upload a video file."

    if _model is None:
        return "", [], "Model not loaded. Start the demo with --checkpoint."

    video_path = video_file if isinstance(video_file, str) else video_file.name
    frames_array, gallery = _extract_mouth_frames(video_path)

    if frames_array is None:
        return "", [], "No face/mouth detected in the video. Try a clearer clip."

    transcription = _run_inference(frames_array)
    status = f"Decoded {len(gallery)*5} frames → {len(transcription.split())} word(s)"
    return transcription, gallery, status


_CSS = """
.transcription-box textarea {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    text-align: center;
}
"""

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="SilentRead — Silent Lip Reading", css=_CSS) as demo:
        gr.Markdown(
            """
            # SilentRead
            **Visual Speech Recognition · No microphone · No audio**

            Upload a short video of someone speaking.  SilentRead watches the
            mouth, extracts 75 frames of lip movement, and transcribes the
            words — purely from video.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload video clip", height=300)
                run_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=1):
                transcription_out = gr.Textbox(
                    label="Transcription",
                    lines=3,
                    elem_classes=["transcription-box"],
                    interactive=False,
                )
                status_out = gr.Textbox(label="Status", lines=1, interactive=False)

        gallery_out = gr.Gallery(
            label="Extracted mouth crops (every 5th frame)",
            columns=8,
            height=160,
            object_fit="contain",
        )

        run_btn.click(
            fn=transcribe,
            inputs=[video_input],
            outputs=[transcription_out, gallery_out, status_out],
        )

        gr.Markdown(
            """
            ---
            **How it works:** MediaPipe FaceMesh → mouth ROI crop → 3D CNN →
            BiGRU → CTC beam-search decode.  Trained on the GRID Corpus
            (34 speakers, 34k clips).

            *Accuracy note: GRID Corpus models work best on clean,
            frontal-face video with the same fixed-vocabulary sentences
            used in training.*
            """
        )

    return demo


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SilentRead Gradio demo")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("SILENTREAD_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    p.add_argument(
        "--lm_path",
        default=os.environ.get("SILENTREAD_LM_PATH", None),
        help="Optional path to KenLM .binary file",
    )
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _load_model(args.checkpoint, lm_path=args.lm_path)
    demo = build_interface()
    demo.launch(share=args.share, server_port=args.port)
