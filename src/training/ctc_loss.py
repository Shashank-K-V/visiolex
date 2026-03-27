"""CTC loss wrapper with input-length computation.

``torch.nn.CTCLoss`` requires:
  - ``log_probs``: (T, B, vocab_size) — already log-softmaxed
  - ``targets``:   (B, max_label_len) or flattened (sum_label_lens,)
  - ``input_lengths``:  (B,) — number of valid timesteps per sample
  - ``target_lengths``: (B,) — actual label length per sample

The model always outputs the full temporal dimension T; ``input_lengths``
is therefore all-T.  ``target_lengths`` comes from the batch's
``label_lens`` field.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CTCLossWrapper(nn.Module):
    """Thin wrapper around :class:`torch.nn.CTCLoss`.

    Args:
        blank_idx: Index of the CTC blank token (default 28).
        reduction: ``"mean"`` or ``"sum"``.
        zero_infinity: Ignore inf losses (useful early in training).
    """

    def __init__(
        self,
        blank_idx: int = 28,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(
            blank=blank_idx,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        log_probs: Tensor,   # (T, B, vocab)
        targets: Tensor,     # (B, max_len)  padded label tensor
        target_lengths: Tensor,  # (B,)
    ) -> Tensor:
        T, B, _ = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long,
                                   device=log_probs.device)

        # CTCLoss is not implemented on MPS; move to CPU for this op only.
        if log_probs.device.type == "mps":
            loss = self.ctc(
                log_probs.cpu(), targets.cpu(),
                input_lengths.cpu(), target_lengths.cpu(),
            )
            return loss.to(log_probs.device)

        return self.ctc(log_probs, targets, input_lengths, target_lengths)
