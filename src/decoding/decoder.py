"""CTC decoders for VisioLex.

Two implementations are provided:

GreedyCTCDecoder
  Fast, no dependencies.  Takes argmax at each timestep then collapses
  repeated tokens and removes blanks.  Suitable for training-time WER
  monitoring where speed matters more than accuracy.

BeamCTCDecoder
  Uses ``pyctcdecode`` for beam-search decoding.  Optionally loads a KenLM
  n-gram language model to rescore candidates, giving +10–15% WER improvement
  on GRID sentences.  Falls back to greedy if ``pyctcdecode`` is not installed.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
from torch import Tensor

from ..utils.text import BLANK_IDX, VOCAB, decode_indices


class GreedyCTCDecoder:
    """Greedy (best-path) CTC decoder.

    Args:
        blank_idx: CTC blank token index.
    """

    def __init__(self, blank_idx: int = BLANK_IDX) -> None:
        self.blank_idx = blank_idx

    def decode(self, log_probs: Tensor) -> str:
        """Decode a single sequence.

        Args:
            log_probs: ``(T, vocab_size)`` log-probability tensor.

        Returns:
            Decoded string.
        """
        indices = log_probs.argmax(dim=-1).tolist()
        return decode_indices(indices, remove_duplicates=True, remove_blank=True)

    def decode_batch(self, log_probs: Tensor) -> List[str]:
        """Decode a batch.

        Args:
            log_probs: ``(T, B, vocab_size)`` log-probability tensor.

        Returns:
            List of decoded strings, length B.
        """
        T, B, V = log_probs.shape
        results = []
        for b in range(B):
            results.append(self.decode(log_probs[:, b, :]))
        return results


class BeamCTCDecoder:
    """Beam-search CTC decoder backed by ``pyctcdecode``.

    If ``pyctcdecode`` is not installed, falls back to greedy decoding with a
    warning.

    Args:
        beam_width: Number of beams to track.
        lm_path: Path to a KenLM ``.binary`` file.  ``None`` = no LM.
        lm_alpha: LM weight (language model score coefficient).
        lm_beta: Word insertion bonus.
        blank_idx: CTC blank token index.
    """

    def __init__(
        self,
        beam_width: int = 10,
        lm_path: Optional[str] = None,
        lm_alpha: float = 0.5,
        lm_beta: float = 1.5,
        blank_idx: int = BLANK_IDX,
    ) -> None:
        self.blank_idx = blank_idx
        self._greedy_fallback = GreedyCTCDecoder(blank_idx)
        self._decoder = None

        try:
            from pyctcdecode import build_ctcdecoder

            # pyctcdecode expects labels as list of strings; blank is ""
            # Index 0 in our vocab is PAD (never predicted), so we pass
            # an empty string for it, then the actual characters.
            labels = [""] + list(VOCAB) + [""]  # pad, a-z+space, blank
            # Ensure list length == vocab_size
            self._decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model=lm_path,
                alpha=lm_alpha,
                beta=lm_beta,
            )
            self._beam_width = beam_width
        except ImportError:
            warnings.warn(
                "pyctcdecode is not installed — BeamCTCDecoder falls back to "
                "greedy decoding.  Install with: pip install pyctcdecode",
                stacklevel=2,
            )

    def decode(self, log_probs: Tensor) -> str:
        """Decode a single sequence.

        Args:
            log_probs: ``(T, vocab_size)`` log-probability tensor.

        Returns:
            Decoded string.
        """
        if self._decoder is None:
            return self._greedy_fallback.decode(log_probs)

        import numpy as np
        probs_np = log_probs.exp().cpu().numpy()  # (T, vocab_size)
        text = self._decoder.decode(probs_np, beam_width=self._beam_width)
        return text.strip().lower()

    def decode_batch(self, log_probs: Tensor) -> List[str]:
        """Decode a batch.

        Args:
            log_probs: ``(T, B, vocab_size)`` log-probability tensor.

        Returns:
            List of decoded strings, length B.
        """
        T, B, V = log_probs.shape
        return [self.decode(log_probs[:, b, :]) for b in range(B)]
