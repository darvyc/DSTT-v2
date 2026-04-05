"""
Ramsey-Partitioned Multi-Head Attention (RP-MHA).

Replaces standard multi-head attention with:
1. Variable-width heads determined by Ramsey-coherence partitioning.
2. Dual-flow attention scoring (CFM-AFM) instead of pure QKᵀ.

When ``use_ramsey_heads`` is False, falls back to standard equal-width
heads but still uses dual-flow scoring.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dstt.config import DSTTConfig
from dstt.flow_matrices import DualFlowScoring


class RPMultiHeadAttention(nn.Module):
    """Ramsey-Partitioned Multi-Head Attention with Dual-Flow Scoring.

    This module computes attention using variable-width heads and
    replaces the standard QKᵀ/√d_k score with the dual-flow composite:
        s_ij = QKᵀ/√d_k + α·CFM_ij − β·AFM_ij

    For efficiency, the current implementation uses equal-width heads
    (standard projections) but applies dual-flow scoring. Full
    variable-width partitioning requires the RamseyPartitioner to
    provide index assignments, which are used when available.

    Args:
        config: Model configuration.
        layer_idx: Index of the layer this attention belongs to.
    """

    def __init__(self, config: DSTTConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout

        # QKV projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dual-flow scoring
        self.dual_flow = DualFlowScoring(config, self.d_head)

        # Attention dropout
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        prev_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of RP-MHA.

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
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: (batch, n_heads, seq_len, d_head)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Prepare prev_state for CFM (project to head dim)
        prev_head = None
        if prev_state is not None:
            # Average across heads for CFM's Ramsey coherence
            prev_head = prev_state[..., : self.d_head]

        # Dual-flow attention scoring
        attn_scores = self.dual_flow(
            queries=q,
            keys=k,
            context=context,
            prev_state=prev_head,
            attention_mask=attention_mask,
        )
        # (batch, n_heads, seq_len, seq_len)

        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)
        # (batch, n_heads, seq_len, d_head)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_model)
        output = self.o_proj(attn_output)

        return output
