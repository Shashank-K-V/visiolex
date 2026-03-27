"""VisioLex model: 3D CNN spatial encoder + BiGRU temporal decoder.

Architecture overview
---------------------
  Input: (B, 1, T, H, W)   — grayscale mouth-crop video tensor
                               B=batch, 1=channels, T=75 frames, H=W=64

  Stage 1 — Spatio-temporal feature extraction (3D CNN)
    Block 1: Conv3d(1→32,   kernel=(3,5,5), pad=(1,2,2)) → BN → ReLU
             MaxPool3d((1,2,2))           → (B, 32,  T,   32, 32)
    Block 2: Conv3d(32→64,  kernel=(3,5,5), pad=(1,2,2)) → BN → ReLU
             MaxPool3d((1,2,2))           → (B, 64,  T,   16, 16)
    Block 3: Conv3d(64→96,  kernel=(3,3,3), pad=(1,1,1)) → BN → ReLU
             MaxPool3d((1,2,2))           → (B, 96,  T,   8,  8)
    Flatten spatial: (B, T, 96*8*8) = (B, T, 6144)

  Stage 2 — Temporal sequence modelling (BiGRU)
    BiGRU(input=6144, hidden=256, layers=2, dropout=0.5)
             → (B, T, 512)   (concat of fwd + bwd hidden states)

  Stage 3 — Per-timestep character logits
    Linear(512, vocab_size)  → (B, T, vocab_size)

  CTC loss is computed externally; the model just returns log-softmax logits.

Parameter count: ~4.5M (comparable to the original LipNet).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _Conv3dBlock(nn.Module):
    """Conv3d → BatchNorm3d → ReLU → MaxPool3d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        conv_kernel: tuple,
        conv_pad: tuple,
        pool_kernel: tuple = (1, 2, 2),
        pool_stride: tuple = (1, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, conv_kernel, padding=conv_pad)
        self.bn = nn.BatchNorm3d(out_ch)
        self.pool = nn.MaxPool3d(pool_kernel, stride=pool_stride)
        self.drop = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class VisioLexModel(nn.Module):
    """VisioLex lip-reading model (3D CNN + BiGRU + CTC-ready).

    Args:
        vocab_size: Number of output classes (including CTC blank).
        cnn_channels: Output channels for each of the three 3D-CNN blocks.
        gru_hidden: Hidden size of each GRU direction.
        gru_layers: Number of stacked BiGRU layers.
        dropout: Dropout probability applied inside the GRU and CNN blocks.
    """

    def __init__(
        self,
        vocab_size: int = 29,
        cnn_channels: List[int] = (32, 64, 96),
        gru_hidden: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        c1, c2, c3 = cnn_channels

        # ------------------------------------------------------------------ #
        # Stage 1 — 3D CNN backbone                                           #
        # ------------------------------------------------------------------ #
        self.cnn = nn.Sequential(
            _Conv3dBlock(1,  c1, (3, 5, 5), (1, 2, 2), dropout=0.0),
            _Conv3dBlock(c1, c2, (3, 5, 5), (1, 2, 2), dropout=0.0),
            _Conv3dBlock(c2, c3, (3, 3, 3), (1, 1, 1), dropout=0.0),
        )
        # After three max-pool layers of stride (1,2,2):
        #   spatial: 64 → 32 → 16 → 8
        #   temporal: unchanged
        self._cnn_out_dim = c3 * 8 * 8  # 96 * 8 * 8 = 6144

        # ------------------------------------------------------------------ #
        # Stage 2 — BiGRU                                                     #
        # ------------------------------------------------------------------ #
        self.gru = nn.GRU(
            input_size=self._cnn_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_drop = nn.Dropout(p=dropout)

        # ------------------------------------------------------------------ #
        # Stage 3 — classifier head                                           #
        # ------------------------------------------------------------------ #
        self.classifier = nn.Linear(gru_hidden * 2, vocab_size)

        self._init_weights()

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Float tensor of shape ``(B, 1, T, H, W)``.

        Returns:
            Log-softmax output of shape ``(T, B, vocab_size)`` — the layout
            expected by ``torch.nn.CTCLoss``.
        """
        # Stage 1: CNN  (B,1,T,H,W) → (B, C3, T, 8, 8)
        feat = self.cnn(x)

        # Reshape: (B, C3, T, 8, 8) → (B, T, C3*8*8)
        B, C, T, H, W = feat.shape
        feat = feat.permute(0, 2, 1, 3, 4).contiguous().view(B, T, C * H * W)

        # Stage 2: GRU  (B, T, D) → (B, T, 2*gru_hidden)
        gru_out, _ = self.gru(feat)
        gru_out = self.gru_drop(gru_out)

        # Stage 3: logits  (B, T, vocab)
        logits = self.classifier(gru_out)

        # CTC expects (T, B, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
