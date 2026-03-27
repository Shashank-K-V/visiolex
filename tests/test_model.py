"""Unit tests for the VisioLex model architecture."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VisioLexModel
from src.utils.text import VOCAB_SIZE


@pytest.fixture
def model():
    return VisioLexModel(
        vocab_size=VOCAB_SIZE,
        cnn_channels=[8, 16, 24],   # small for speed
        gru_hidden=32,
        gru_layers=1,
        dropout=0.0,
    )


def test_output_shape(model):
    """Model should return (T, B, vocab_size) log-probs."""
    B, T, H, W = 2, 75, 64, 64
    x = torch.randn(B, 1, T, H, W)
    out = model(x)
    assert out.shape == (T, B, VOCAB_SIZE), f"Expected ({T},{B},{VOCAB_SIZE}), got {out.shape}"


def test_log_probs_sum_to_one(model):
    """Log-softmax output should be valid probabilities."""
    x = torch.randn(1, 1, 75, 64, 64)
    out = model(x)  # (T, 1, vocab)
    # exp(log_probs).sum(dim=-1) should be ~1 at every timestep
    probs = out.exp()
    sums = probs.sum(dim=-1)  # (T, 1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        "Probability distributions do not sum to 1"


def test_parameter_count(model):
    assert model.num_parameters > 0


def test_gradient_flow(model):
    """Loss should produce non-None gradients on all parameters."""
    x = torch.randn(2, 1, 75, 64, 64)
    log_probs = model(x)             # (T, B, vocab)
    T, B, V = log_probs.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)
    targets = torch.randint(1, 27, (B, 5))
    target_lengths = torch.tensor([5, 5])

    ctc = torch.nn.CTCLoss(blank=28, zero_infinity=True)
    loss = ctc(log_probs, targets, input_lengths, target_lengths)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_different_batch_sizes(model):
    for B in [1, 4, 8]:
        x = torch.randn(B, 1, 75, 64, 64)
        out = model(x)
        assert out.shape[1] == B
