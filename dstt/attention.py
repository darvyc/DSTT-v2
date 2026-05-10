"""Lightweight sequence mixer used by DSTT-v2.

This module removes transformer attention and replaces it with a
lightweight state-space inspired mixer built from depthwise convolutions,
channel mixing, and a tensor-fold operation.

Tensor folding introduces higher-dimensional structure by projecting token
features into a rank-factored 4D manifold and folding it back to the model
dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dstt.config import DSTTConfig


class LightweightTensorMixer(nn.Module):
    """Lightweight replacement for transformer attention.

    Uses depthwise temporal convolution + pointwise channel mixing and a
    tensor-fold residual branch to provide higher-dimensional feature
    interactions without quadratic attention cost.
    """

    def __init__(self, config: DSTTConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.hidden = max(config.d_model // 2, 64)
        self.in_proj = nn.Linear(config.d_model, self.hidden * 2, bias=False)
        self.depthwise = nn.Conv1d(
            self.hidden,
            self.hidden,
            kernel_size=5,
            padding=2,
            groups=self.hidden,
            bias=False,
        )
        self.channel_mix = nn.Linear(self.hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Higher-dimensional tensor fold parameters.
        self.fold_rank = 4
        self.fold_rows = max(config.d_model // 16, 8)
        self.fold_cols = max(config.d_model // (self.fold_rank * self.fold_rows), 4)
        self.fold_in = nn.Linear(config.d_model, self.fold_rank * self.fold_rows * self.fold_cols, bias=False)
        self.fold_out = nn.Linear(self.fold_rank * self.fold_rows * self.fold_rows, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        prev_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the lightweight tensor mixer.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            context: Global context vector, shape (batch, d_model).
            prev_state: Previous layer's output for Ramsey coherence,
                shape (batch, seq_len, d_model) or None.
            attention_mask: Mask tensor, shape (batch, 1, seq_len, seq_len)
                or broadcastable. 0 = masked, 1 = attend.

        Returns:
            Attention output, shape (batch, seq_len, d_model).
        """
        del context, prev_state, attention_mask
        gated = self.in_proj(x)
        u, v = gated.chunk(2, dim=-1)
        mixed = torch.sigmoid(v) * u

        seq = mixed.transpose(1, 2)
        seq = self.depthwise(seq).transpose(1, 2)
        base = self.channel_mix(seq)

        fold = self.fold_in(x)
        b, t, _ = fold.shape
        fold = fold.view(b, t, self.fold_rank, self.fold_rows, self.fold_cols)
        gram = torch.einsum("btrij,btrkj->btrik", fold, fold)
        fold = gram.reshape(b, t, -1)
        fold = self.fold_out(fold)

        return self.dropout(base + 0.1 * fold)


# Backward-compatible alias
RPMultiHeadAttention = LightweightTensorMixer
