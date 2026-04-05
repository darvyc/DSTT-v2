"""
GPT-style Trainer for DSTT-T.

Implements the standard GPT pre-training loop:
- AdamW optimiser with decoupled weight decay
- Linear warmup followed by cosine learning-rate decay
- Gradient accumulation for large effective batch sizes
- Gradient clipping by global norm
- Periodic validation evaluation
- Checkpoint save / resume
- Optional sample generation during training
- Mixed precision (float16 / bfloat16) support

The training objective is **next-token prediction**: given a sequence
of tokens [t₀, t₁, ..., tₙ], predict [t₁, t₂, ..., tₙ₊₁]. This
is the same objective used by GPT-2 and GPT-3.

Usage::

    from dstt import DSTTConfig, DSTTTransformer
    from dstt.train_config import TrainConfig
    from dstt.trainer import Trainer

    model_cfg = DSTTConfig.tiny()
    train_cfg = TrainConfig(max_steps=1000, batch_size=16)
    model = DSTTTransformer(model_cfg)
    trainer = Trainer(model, train_cfg, train_dataset, val_dataset)
    trainer.train()
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dstt.train_config import TrainConfig

logger = logging.getLogger(__name__)


class Trainer:
    """GPT-style trainer for DSTT-T models.

    Handles the complete training lifecycle: optimiser setup, learning-
    rate scheduling, gradient accumulation, evaluation, checkpointing,
    and optional sample generation.

    Args:
        model: A ``DSTTTransformer`` instance.
        config: Training configuration.
        train_dataset: Training dataset (TextDataset or MemmapDataset).
        val_dataset: Validation dataset (may be None to skip eval).
        tokenizer: Tokenizer instance (needed for sample generation).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_dataset,
        val_dataset=None,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

        # Device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Dtype / autocast
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(config.dtype, torch.float32)
        self.use_amp = self.dtype != torch.float32
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")

        # Optional torch.compile
        if config.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

        # Optimiser
        self.optimiser = self._create_optimiser()

        # State
        self.step = 0
        self.best_val_loss = float("inf")
        self.tokens_seen = 0

    def _create_optimiser(self) -> torch.optim.Optimizer:
        """Create AdamW optimiser with decoupled weight decay.

        Weight decay is applied to all 2D+ parameters (weight matrices)
        but NOT to biases, LayerNorm/RMSNorm parameters, or embedding
        vectors — matching the GPT-2/3 convention.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        logger.info(
            f"Optimiser: {len(decay_params)} decayed param groups, "
            f"{len(no_decay_params)} non-decayed"
        )

        return torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            fused=self.device.type == "cuda",
        )

    def _set_lr(self, step: int) -> float:
        """Apply learning rate schedule and return current LR."""
        lr = self.config.get_lr(step)
        for group in self.optimiser.param_groups:
            group["lr"] = lr
        return lr

    def _get_batch(self, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch from a dataset."""
        from dstt.data import get_batch
        return get_batch(dataset, self.config.batch_size, self.device)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run validation evaluation and return mean loss."""
        if self.val_dataset is None:
            return float("nan")

        self.model.eval()
        losses = []

        for _ in range(self.config.eval_steps):
            x, y = self._get_batch(self.val_dataset)
            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())

        self.model.train()
        return sum(losses) / len(losses)

    def _generate_sample(self) -> str:
        """Generate a text sample for monitoring training progress."""
        if self.tokenizer is None:
            return ""
        from dstt.generate import generate_text
        return generate_text(
            self.model,
            self.tokenizer,
            prompt=self.config.sample_prompt,
            max_new_tokens=self.config.sample_max_tokens,
            temperature=self.config.sample_temperature,
            top_k=50,
            device=self.device,
        )

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save a training checkpoint.

        Saves model weights, optimiser state, training step, and
        configuration — everything needed to resume training.

        Args:
            path: Override save path. If None, uses save_dir/step_N.pt.

        Returns:
            Path where the checkpoint was saved.
        """
        if path is None:
            os.makedirs(self.config.save_dir, exist_ok=True)
            path = os.path.join(self.config.save_dir, f"step_{self.step}.pt")

        raw_model = getattr(self.model, "_orig_mod", self.model)
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "tokens_seen": self.tokens_seen,
            "config": raw_model.config,
            "train_config": self.config,
        }, path)

        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.tokens_seen = ckpt.get("tokens_seen", 0)

        logger.info(f"Resumed from {path} at step {self.step}")

    def train(self) -> dict:
        """Run the full GPT-style training loop.

        This is the main entry point. It runs for ``max_steps`` weight
        updates, performing gradient accumulation, LR scheduling,
        periodic evaluation, checkpointing, and sample generation.

        Returns:
            Dict with final training statistics.
        """
        cfg = self.config
        model = self.model
        model.train()

        # Resume if requested
        if cfg.resume_from is not None:
            self.load_checkpoint(cfg.resume_from)

        logger.info(f"Starting training from step {self.step}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {self.dtype}")
        logger.info(f"  Effective batch size: {cfg.effective_batch_size}")
        logger.info(f"  Block size: {cfg.block_size}")
        logger.info(f"  Max steps: {cfg.max_steps}")
        logger.info(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")

        t0 = time.time()
        running_loss = 0.0
        tokens_per_step = cfg.effective_batch_size * cfg.block_size

        while self.step < cfg.max_steps:
            # ── Learning rate schedule ──
            lr = self._set_lr(self.step)

            # ── Gradient accumulation loop ──
            self.optimiser.zero_grad(set_to_none=True)
            micro_loss_sum = 0.0

            for micro_step in range(cfg.gradient_accumulation_steps):
                x, y = self._get_batch(self.train_dataset)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.use_amp,
                ):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    )
                    # Scale loss by accumulation steps for correct averaging
                    scaled_loss = loss / cfg.gradient_accumulation_steps

                # Backward (scaled for AMP)
                self.scaler.scale(scaled_loss).backward()
                micro_loss_sum += loss.item()

            # ── Gradient clipping ──
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            # ── Optimiser step ──
            self.scaler.step(self.optimiser)
            self.scaler.update()

            # ── Bookkeeping ──
            avg_loss = micro_loss_sum / cfg.gradient_accumulation_steps
            running_loss += avg_loss
            self.step += 1
            self.tokens_seen += tokens_per_step

            # ── Logging ──
            if self.step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                tokens_sec = self.tokens_seen / elapsed if elapsed > 0 else 0
                avg = running_loss / cfg.log_interval
                ppl = math.exp(min(avg, 20))  # cap to avoid overflow
                logger.info(
                    f"step {self.step:>7d} | "
                    f"loss {avg:.4f} | "
                    f"ppl {ppl:8.2f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tokens_sec:,.0f} | "
                    f"tokens {self.tokens_seen:,d}"
                )
                running_loss = 0.0

            # ── Validation ──
            if self.step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20))
                logger.info(
                    f"  → val loss {val_loss:.4f} | val ppl {val_ppl:.2f}"
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(cfg.save_dir, "best.pt")
                    )

            # ── Checkpointing ──
            if self.step % cfg.save_interval == 0:
                self.save_checkpoint()

            # ── Sample generation ──
            if (
                cfg.sample_interval > 0
                and self.step % cfg.sample_interval == 0
                and self.tokenizer is not None
            ):
                sample = self._generate_sample()
                logger.info(f"  → sample: {sample[:300]}...")
                model.train()

        # Final save
        final_path = self.save_checkpoint(
            os.path.join(cfg.save_dir, "final.pt")
        )

        total_time = time.time() - t0
        return {
            "final_step": self.step,
            "best_val_loss": self.best_val_loss,
            "total_tokens": self.tokens_seen,
            "total_time_seconds": total_time,
            "checkpoint_path": final_path,
        }
