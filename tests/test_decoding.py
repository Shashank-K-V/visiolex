"""Unit tests for CTC decoders."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decoding import GreedyCTCDecoder
from src.utils.text import BLANK_IDX, VOCAB_SIZE, encode_text, decode_indices


class TestTextUtils:
    def test_encode_decode_roundtrip(self):
        text = "bin blue at f two"
        encoded = encode_text(text)
        decoded = decode_indices(encoded, remove_duplicates=False,
                                 remove_blank=False)
        assert decoded == text

    def test_encode_unknown_chars_dropped(self):
        encoded = encode_text("hello!!")  # '!' not in vocab
        assert encode_text("hello") == encoded

    def test_blank_removed_on_decode(self):
        indices = [2, BLANK_IDX, 9, 14]
        decoded = decode_indices(indices)
        assert str(BLANK_IDX) not in decoded

    def test_duplicates_collapsed(self):
        indices = [2, 2, 2, 9, 9, 14]
        decoded = decode_indices(indices, remove_duplicates=True, remove_blank=False)
        assert decoded == "bin"


class TestGreedyDecoder:
    @pytest.fixture
    def decoder(self):
        return GreedyCTCDecoder(blank_idx=BLANK_IDX)

    def test_decode_single_sequence(self, decoder):
        T, V = 75, VOCAB_SIZE
        log_probs = torch.randn(T, V).log_softmax(dim=-1)
        result = decoder.decode(log_probs)
        assert isinstance(result, str)

    def test_decode_batch(self, decoder):
        T, B, V = 75, 4, VOCAB_SIZE
        log_probs = torch.randn(T, B, V).log_softmax(dim=-1)
        results = decoder.decode_batch(log_probs)
        assert len(results) == B
        assert all(isinstance(r, str) for r in results)

    def test_high_confidence_output(self, decoder):
        """If the model is very confident about 'bin', it should decode to 'bin'."""
        # b=2, i=9, n=14 (1-indexed)
        T, V = 10, VOCAB_SIZE
        log_probs = torch.full((T, V), -1e9)
        # First 3 frames: confident b, i, n; rest: confident blank
        for t, idx in enumerate([2, 9, 14]):
            log_probs[t, idx] = 0.0
        for t in range(3, T):
            log_probs[t, BLANK_IDX] = 0.0
        log_probs = log_probs.log_softmax(dim=-1)
        result = decoder.decode(log_probs)
        assert result == "bin", f"Expected 'bin', got '{result}'"
