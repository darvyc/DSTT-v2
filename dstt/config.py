"""
Configuration for DSTT models.

Every architectural and training hyperparameter is specified here.
Use preset class methods for standard configurations::

    config = DSTTConfig.base()   # 125M-param model
    config = DSTTConfig.tiny()   # 15M-param model for debugging
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DSTTConfig:
    """Complete configuration for a DSTTransformer model.

    Parameters are grouped by the component they control:
    model dimensions, RP-MHA, dual-flow attention, ARM-FFN,
    Wittgenstein gating, EML, and training.
    """

    # ── Model dimensions ──────────────────────────────────────────
    d_model: int = 768
    """Model (embedding) dimension."""

    n_layers: int = 12
    """Number of DSTT blocks."""

    vocab_size: int = 50257
    """Vocabulary size for token embedding."""

    max_seq_len: int = 2048
    """Maximum sequence length."""

    dropout: float = 0.1
    """Dropout rate applied throughout."""

    # ── RP-MHA (Ramsey-Partitioned Multi-Head Attention) ──────────
    n_heads: int = 16
    """Number of attention heads. When ``use_ramsey_heads`` is True
    this serves as the initial / fallback count; the actual count
    may be refined by EML or by the Ramsey partitioner."""

    use_ramsey_heads: bool = True
    """If True, use Ramsey-coherence partitioning to determine
    head count and variable-width head assignment."""

    coherence_threshold: float = 0.25
    """Threshold τ for the Ramsey coherence clustering that assigns
    model dimensions to attention heads."""

    # ── Dual-Flow Attention (CFM / AFM) ───────────────────────────
    cfm_alpha_init: float = 0.1
    """Initial value for the learnable CFM scaling coefficient α.
    The system starts near standard attention behaviour and gradually
    incorporates DSTT signals during training."""

    afm_beta_init: float = 0.1
    """Initial value for the learnable AFM scaling coefficient β."""

    # ── ARM-FFN (Adaptive Routing Feed-Forward) ───────────────────
    d_ff: int = 3072
    """Total feed-forward hidden dimension across all experts."""

    n_experts: int = 8
    """Number of ARM-FFN expert sub-networks."""

    top_k_experts: int = 2
    """Number of experts activated per token (sparse routing)."""

    load_balance_weight: float = 0.01
    """Weight γ of the auxiliary load-balancing loss."""

    # ── Wittgenstein Contextual Gating ────────────────────────────
    use_wittgenstein_gate: bool = True
    """If True, apply Wittgenstein contextual gates on residual
    connections after each sub-layer."""

    # ── Evolutionary Meta-Optimisation Layer (EML) ────────────────
    use_eml: bool = False
    """If True, enable the evolutionary meta-optimisation layer.
    Typically enabled only during architecture search (Phase 1)."""

    evo_population: int = 32
    """EA population size (number of candidate architectures)."""

    evo_generations: int = 50
    """Number of EA generations in the architecture search phase."""

    evo_period: int = 500
    """Number of gradient steps between EA generation evaluations."""

    evo_mutation_rate: float = 0.1
    """Per-gene mutation probability."""

    evo_tournament_size: int = 5
    """Tournament size for parent selection."""

    evo_elitism_rate: float = 0.1
    """Fraction of top individuals preserved across generations."""

    # ── EML fitness weights ───────────────────────────────────────
    evo_fitness_perplexity_weight: float = 0.5
    evo_fitness_coherence_weight: float = 0.3
    evo_fitness_efficiency_weight: float = 0.2

    # ── Derived properties ────────────────────────────────────────
    @property
    def d_head(self) -> int:
        """Default head dimension (used when Ramsey partitioning is off)."""
        return self.d_model // self.n_heads

    @property
    def d_expert(self) -> int:
        """Hidden dimension per expert."""
        return self.d_ff // self.n_experts

    @property
    def ramsey_head_count(self) -> int:
        """Head count from Ramanujan-Hardy partition theory.

        h* = ⌊π√(2d/3) / ln(2)⌋, clamped to [4, d//8].
        """
        raw = math.floor(math.pi * math.sqrt(2 * self.d_model / 3) / math.log(2))
        return max(4, min(raw, self.d_model // 8))

    def validate(self) -> None:
        """Raise ValueError if configuration is inconsistent."""
        if self.d_model % self.n_heads != 0 and not self.use_ramsey_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}) when Ramsey partitioning is disabled."
            )
        if self.d_ff % self.n_experts != 0:
            raise ValueError(
                f"d_ff ({self.d_ff}) must be divisible by n_experts "
                f"({self.n_experts})."
            )
        if not 0 < self.coherence_threshold < 1:
            raise ValueError("coherence_threshold must be in (0, 1).")
        if self.top_k_experts > self.n_experts:
            raise ValueError("top_k_experts cannot exceed n_experts.")

    # ── Preset configurations ─────────────────────────────────────
    @classmethod
    def tiny(cls) -> DSTTConfig:
        """~15M params. For debugging and unit tests."""
        return cls(
            d_model=256, n_layers=4, n_heads=8, d_ff=1024,
            n_experts=4, top_k_experts=2, vocab_size=10000,
            max_seq_len=512, use_ramsey_heads=False,
        )

    @classmethod
    def base(cls) -> DSTTConfig:
        """~125M params. Standard research configuration."""
        return cls(
            d_model=768, n_layers=12, n_heads=16, d_ff=3072,
            n_experts=8, top_k_experts=2,
        )

    @classmethod
    def large(cls) -> DSTTConfig:
        """~350M params."""
        return cls(
            d_model=1024, n_layers=24, n_heads=32, d_ff=4096,
            n_experts=16, top_k_experts=2,
        )

    @classmethod
    def xl(cls) -> DSTTConfig:
        """~1.3B params."""
        return cls(
            d_model=1536, n_layers=32, n_heads=48, d_ff=6144,
            n_experts=32, top_k_experts=2,
        )
