"""
Wittgenstein Contextual Gating (WCG).

A sigmoid gate on the residual stream that implements the
Wittgensteinian principle: a representation's contribution is
determined by its *use in context*, not its intrinsic properties.

    x' = x + w_i · sublayer(x)

where w_i = σ(W_w · [x_i; context] + b_w).

Tokens whose sub-layer outputs are not contextually meaningful
are gated out, preventing them from degrading the residual stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WittgensteinGate(nn.Module):
    """Contextual gate for the residual connection.

    Produces a per-token scalar gate in [0, 1] based on the
    concatenation of the token representation and the global
    context vector.

    When the gate is ~1, the sub-layer output passes through normally.
    When the gate is ~0, the residual stream is preserved unchanged.

    Args:
        d_model: Model dimension.
        bias_init: Initial bias value. A small negative value
            (e.g., -1.0) encourages the gate to start slightly
            closed, promoting conservative information flow early
            in training.
    """

    def __init__(self, d_model: int, bias_init: float = 0.0):
        super().__init__()
        # Input: concatenation of token rep + context → 2 * d_model
        self.gate_proj = nn.Linear(2 * d_model, 1, bias=True)

        # Initialise with small weights and specified bias
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, bias_init)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token Wittgenstein gate values.

        Args:
            x: Token representations, shape (batch, seq_len, d_model).
            context: Global context vector, shape (batch, d_model).

        Returns:
            Gate values, shape (batch, seq_len, 1) in [0, 1].
        """
        # Expand context to match sequence dimension
        ctx = context.unsqueeze(1).expand_as(x)
        # (batch, seq_len, d_model)

        # Concatenate token and context
        combined = torch.cat([x, ctx], dim=-1)
        # (batch, seq_len, 2 * d_model)

        # Sigmoid gate
        gate = torch.sigmoid(self.gate_proj(combined))
        # (batch, seq_len, 1)

        return gate
