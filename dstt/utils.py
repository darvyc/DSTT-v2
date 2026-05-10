"""
Utility functions for DSTT.

Includes mathematical helpers for partition theory, coherence
computation, and general-purpose neural network utilities.
"""

from __future__ import annotations

import math
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Partition Theory ──────────────────────────────────────────────

def hardy_ramanujan_approx(n: int) -> float:
    """Compute the Hardy-Ramanujan asymptotic approximation of p(n).

    p(n) ~ (1 / 4n√3) · exp(π√(2n/3))

    Args:
        n: Positive integer for which to approximate the partition count.

    Returns:
        Approximate value of p(n).
    """
    if n <= 0:
        return 1.0
    return (1.0 / (4.0 * n * math.sqrt(3))) * math.exp(
        math.pi * math.sqrt(2.0 * n / 3.0)
    )


def partition_count_to_heads(d: int, clamp_min: int = 4, clamp_max_divisor: int = 8) -> int:
    """Compute Ramsey-derived head count from model dimension.

    h* = ⌊log₂ p(d)⌋ clamped to [clamp_min, d // clamp_max_divisor].

    Args:
        d: Model dimension.
        clamp_min: Minimum number of heads.
        clamp_max_divisor: d // this value gives the maximum head count.

    Returns:
        Target number of attention heads.
    """
    log2_p = math.pi * math.sqrt(2.0 * d / 3.0) / math.log(2.0)
    raw = int(math.floor(log2_p))
    return max(clamp_min, min(raw, d // clamp_max_divisor))


# ── Coherence Metrics ─────────────────────────────────────────────

def pairwise_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        x: Tensor of shape (n, d).

    Returns:
        Similarity matrix of shape (n, n) with values in [-1, 1].
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    return x_norm @ x_norm.transpose(-2, -1)


def ramsey_coherence(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute Ramsey coherence between two representations.

    R_c(x, y) = 1 - d_H(x, y) / d

    where d_H is the Hamming distance of the binarised representations
    (sign of each dimension), normalised by dimension.

    Args:
        x: Tensor of shape (..., d).
        y: Tensor of shape (..., d).

    Returns:
        Coherence score in [0, 1], shape (...).
    """
    x_bin = (x > 0).float()
    y_bin = (y > 0).float()
    hamming = (x_bin != y_bin).float().sum(dim=-1)
    d = x.shape[-1]
    return 1.0 - hamming / d


# ── Neural Network Utilities ─────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    More stable and slightly faster than standard LayerNorm.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def top_k_softmax(
    logits: torch.Tensor,
    k: int,
    dim: int = -1,
) -> torch.Tensor:
    """Compute softmax over only the top-k values, zeroing the rest.

    Args:
        logits: Raw scores of shape (..., n).
        k: Number of top values to keep.
        dim: Dimension along which to apply softmax.

    Returns:
        Sparse probability distribution of shape (..., n).
    """
    top_vals, top_idx = logits.topk(k, dim=dim)
    mask = torch.zeros_like(logits).scatter_(dim, top_idx, 1.0)
    masked_logits = logits.masked_fill(mask == 0, float("-inf"))
    return F.softmax(masked_logits, dim=dim)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
