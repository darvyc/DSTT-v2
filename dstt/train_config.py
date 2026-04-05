"""
Training configuration for GPT-style pre-training.

Separates model architecture (DSTTConfig) from training procedure
(TrainConfig). Both are needed to fully reproduce a training run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """All hyperparameters controlling the GPT-style training loop.

    Usage::

        from dstt.train_config import TrainConfig
        tc = TrainConfig(max_steps=100_000, batch_size=64)
    """

    # ── Data ──────────────────────────────────────────────────────
    data_dir: str = "data"
    """Directory containing train.bin and val.bin (memory-mapped
    token arrays produced by ``prepare_data.py``)."""

    # ── Batch / sequence ──────────────────────────────────────────
    batch_size: int = 32
    """Micro-batch size (per GPU)."""

    block_size: int = 1024
    """Context window (sequence length) in tokens."""

    gradient_accumulation_steps: int = 1
    """Number of micro-batches accumulated before a weight update.
    Effective batch = batch_size * gradient_accumulation_steps."""

    # ── Optimiser ─────────────────────────────────────────────────
    max_steps: int = 100_000
    """Total number of weight-update steps."""

    learning_rate: float = 3e-4
    """Peak learning rate (reached after warmup)."""

    min_lr: float = 3e-5
    """Minimum learning rate at the end of cosine decay."""

    warmup_steps: int = 2000
    """Linear warmup from 0 to ``learning_rate``."""

    weight_decay: float = 0.1
    """AdamW weight decay applied to non-bias, non-LayerNorm params."""

    beta1: float = 0.9
    """Adam β₁."""

    beta2: float = 0.95
    """Adam β₂."""

    grad_clip: float = 1.0
    """Maximum gradient norm (0 to disable)."""

    # ── Evaluation ────────────────────────────────────────────────
    eval_interval: int = 500
    """Steps between validation evaluations."""

    eval_steps: int = 50
    """Number of batches used for each validation evaluation."""

    # ── Logging / checkpointing ───────────────────────────────────
    log_interval: int = 10
    """Steps between training-loss log prints."""

    save_interval: int = 5000
    """Steps between checkpoint saves."""

    save_dir: str = "checkpoints"
    """Directory for saving checkpoints."""

    resume_from: Optional[str] = None
    """Path to a checkpoint to resume from. If None, train from scratch."""

    # ── Device ────────────────────────────────────────────────────
    device: str = "auto"
    """Device string: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'."""

    compile_model: bool = False
    """If True, use ``torch.compile()`` (requires PyTorch 2.0+)."""

    dtype: str = "float32"
    """Training dtype: 'float32', 'float16', 'bfloat16'."""

    # ── Generation during training ────────────────────────────────
    sample_interval: int = 5000
    """Steps between sample-generation during training (0 to disable)."""

    sample_prompt: str = "\n"
    """Prompt string used for sample generation."""

    sample_max_tokens: int = 200
    """Maximum tokens to generate per sample."""

    sample_temperature: float = 0.8
    """Sampling temperature for generation."""

    # ── Derived ───────────────────────────────────────────────────
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def total_tokens(self) -> int:
        """Total tokens seen during training."""
        return self.max_steps * self.effective_batch_size * self.block_size

    def get_lr(self, step: int) -> float:
        """Compute learning rate at a given step.

        Implements linear warmup followed by cosine decay to ``min_lr``.
        """
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)
