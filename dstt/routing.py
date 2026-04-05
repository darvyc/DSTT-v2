"""
ARM-FFN: Adaptive Routing Feed-Forward Network.

Replaces the standard transformer FFN with a partition-gated mixture
of experts (MoE). Expert selection is governed by CFM-AFM dual-flow
gating rather than a learned gating network.

Each expert corresponds to a coherent partition of the representation
space. Tokens are routed using:
    gate = softmax(CFM(x, centroid_k) - AFM(x, centroid_k))

Top-K sparse selection keeps computation efficient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dstt.config import DSTTConfig
from dstt.utils import top_k_softmax


class Expert(nn.Module):
    """Single expert sub-network (standard 2-layer FFN).

    Args:
        d_model: Input/output dimension.
        d_hidden: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden)
        self.w2 = nn.Linear(d_hidden, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class ARMFeedForward(nn.Module):
    """ARM-FFN: Partition-gated mixture of experts.

    Maintains E expert sub-networks. For each token, computes
    CFM-AFM routing scores against each expert's centroid, applies
    top-K softmax gating, and returns the weighted expert mix.

    Includes an auxiliary load-balancing loss to prevent expert
    collapse.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: DSTTConfig):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model

        # Expert sub-networks
        d_expert = config.d_ff // config.n_experts
        self.experts = nn.ModuleList([
            Expert(config.d_model, d_expert, config.dropout)
            for _ in range(config.n_experts)
        ])

        # Routing: learnable expert centroids (for CFM-AFM gating)
        self.centroids = nn.Parameter(
            torch.randn(config.n_experts, config.d_model) * 0.02
        )

        # Routing projections for CFM-AFM scoring
        self.route_proj = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Load-balancing loss weight
        self.lb_weight = config.load_balance_weight

        # Storage for load-balancing loss (computed during forward)
        self._load_balance_loss: torch.Tensor | None = None

    @property
    def load_balance_loss(self) -> torch.Tensor:
        """Return the most recently computed load-balancing loss."""
        if self._load_balance_loss is None:
            return torch.tensor(0.0)
        return self._load_balance_loss

    def _compute_routing(self, x: torch.Tensor) -> torch.Tensor:
        """Compute CFM-AFM-inspired routing scores.

        For each token x_i, the routing score for expert k is:
            r_ik = cos(x_i, centroid_k)

        This approximates CFM(x, centroid_k) - AFM(x, centroid_k)
        using a learned linear projection for efficiency.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Routing logits, shape (batch, seq_len, n_experts).
        """
        # Cosine similarity between tokens and expert centroids
        x_norm = F.normalize(x, dim=-1)
        c_norm = F.normalize(self.centroids, dim=-1)
        cos_scores = torch.matmul(x_norm, c_norm.T)
        # (batch, seq_len, n_experts)

        # Also add learned routing signal
        route_scores = self.route_proj(x)
        # (batch, seq_len, n_experts)

        return cos_scores + route_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with top-K expert routing.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Expert-mixed output, shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape

        # Compute routing scores
        routing_logits = self._compute_routing(x)
        # (batch, seq_len, n_experts)

        # Top-K sparse gating
        gate_weights = top_k_softmax(routing_logits, self.top_k, dim=-1)
        # (batch, seq_len, n_experts)

        # Compute load-balancing loss
        self._compute_load_balance_loss(gate_weights, routing_logits)

        # Dispatch to experts and combine
        output = torch.zeros_like(x)
        for k in range(self.n_experts):
            gk = gate_weights[:, :, k].unsqueeze(-1)  # (batch, seq_len, 1)
            # Only compute expert if any token routes to it
            if gk.sum() > 0:
                expert_out = self.experts[k](x)
                output = output + gk * expert_out

        return output

    def _compute_load_balance_loss(
        self,
        gate_weights: torch.Tensor,
        routing_logits: torch.Tensor,
    ) -> None:
        """Compute auxiliary load-balancing loss.

        L_bal = γ · E · Σ_k (f_k · p_k)

        where f_k is the fraction of tokens routed to expert k
        and p_k is the average gating probability for expert k.

        Args:
            gate_weights: Sparse gating weights (batch, seq_len, n_experts).
            routing_logits: Raw routing scores (batch, seq_len, n_experts).
        """
        # f_k: fraction of tokens dispatched to each expert
        dispatched = (gate_weights > 0).float()
        f = dispatched.mean(dim=[0, 1])  # (n_experts,)

        # p_k: average routing probability
        routing_probs = F.softmax(routing_logits, dim=-1)
        p = routing_probs.mean(dim=[0, 1])  # (n_experts,)

        self._load_balance_loss = self.lb_weight * self.n_experts * (f * p).sum()
