#!/usr/bin/env python3
"""
Train DSTT-T like a GPT.

This script implements the standard GPT pre-training pipeline:
next-token prediction with AdamW, linear warmup, cosine LR decay,
gradient accumulation, and periodic validation + checkpointing.

Quick start (Tiny Shakespeare):
    python prepare_data.py --download shakespeare
    python train.py --data_dir data --config tiny --max_steps 5000

Resume training:
    python train.py --resume_from checkpoints/best.pt

Custom configuration:
    python train.py \
        --data_dir data \
        --config base \
        --batch_size 32 \
        --block_size 1024 \
        --gradient_accumulation_steps 4 \
        --learning_rate 3e-4 \
        --max_steps 100000 \
        --eval_interval 500 \
        --save_interval 5000
"""

import argparse
import logging
import os
import sys

import torch
import numpy as np

from dstt import DSTTConfig, DSTTTransformer
from dstt.data import TextDataset, MemmapDataset, create_datasets
from dstt.tokenizer import CharTokenizer, get_tokenizer
from dstt.train_config import TrainConfig
from dstt.trainer import Trainer
from dstt.utils import count_parameters, format_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_CONFIGS = {
    "tiny": DSTTConfig.tiny,
    "base": DSTTConfig.base,
    "large": DSTTConfig.large,
    "xl": DSTTConfig.xl,
}


def load_data(args):
    """Load datasets and tokenizer from prepared data directory.

    Supports two modes:
    1. Pre-tokenized: data_dir contains train.bin and val.bin
       (created by prepare_data.py).
    2. Raw text: data_dir contains a .txt file -- tokenized on the fly.
    """
    data_dir = args.data_dir
    block_size = args.block_size

    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    tok_path = os.path.join(data_dir, "tokenizer.json")

    # Mode 1: pre-tokenized .bin files
    if os.path.exists(train_bin):
        logger.info(f"Loading pre-tokenized data from {data_dir}/")
        train_ds = MemmapDataset(train_bin, block_size)
        val_ds = MemmapDataset(val_bin, block_size) if os.path.exists(val_bin) else None

        if os.path.exists(tok_path):
            tokenizer = CharTokenizer.load(tok_path)
        else:
            try:
                tokenizer = get_tokenizer("gpt2")
            except ImportError:
                tokenizer = None

        vocab_size = tokenizer.vocab_size if tokenizer else 50257
        logger.info(f"  train: {len(train_ds):,d} windows")
        if val_ds:
            logger.info(f"  val:   {len(val_ds):,d} windows")
        logger.info(f"  vocab: {vocab_size:,d}")
        return train_ds, val_ds, tokenizer, vocab_size

    # Mode 2: find a .txt file and tokenize on the fly
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if txt_files:
        txt_path = os.path.join(data_dir, txt_files[0])
        logger.info(f"Loading raw text from {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = get_tokenizer("char", text=text)
        train_ds, val_ds = create_datasets(text, tokenizer, block_size)
        logger.info(f"  train: {len(train_ds):,d} windows")
        logger.info(f"  val:   {len(val_ds):,d} windows")
        logger.info(f"  vocab: {tokenizer.vocab_size}")
        return train_ds, val_ds, tokenizer, tokenizer.vocab_size

    raise FileNotFoundError(
        f"No data found in {data_dir}/. "
        "Run prepare_data.py first, or put a .txt file in the data directory."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train DSTT-T like a GPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--config", type=str, default="tiny", choices=list(MODEL_CONFIGS))

    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--block_size", type=int, default=256)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=20)

    # Logging / checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    # Sampling
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--sample_prompt", type=str, default="\n")
    parser.add_argument("--sample_max_tokens", type=int, default=200)

    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")

    args = parser.parse_args()

    # Load data
    train_ds, val_ds, tokenizer, vocab_size = load_data(args)

    # Build model
    model_cfg = MODEL_CONFIGS[args.config]()
    model_cfg.vocab_size = vocab_size
    model_cfg.max_seq_len = args.block_size
    model = DSTTTransformer(model_cfg)
    n_params = count_parameters(model)

    logger.info(f"Model: DSTT-T {args.config}")
    logger.info(f"  Parameters: {format_params(n_params)} ({n_params:,d})")
    logger.info(f"  d_model={model_cfg.d_model}, layers={model_cfg.n_layers}, "
                f"heads={model_cfg.n_heads}, experts={model_cfg.n_experts}")

    # Training config
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dtype=args.dtype,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        resume_from=args.resume_from,
        device=args.device,
        compile_model=args.compile,
        sample_interval=args.sample_interval,
        sample_prompt=args.sample_prompt,
        sample_max_tokens=args.sample_max_tokens,
    )

    logger.info(f"Training: {train_cfg.max_steps} steps, "
                f"effective batch={train_cfg.effective_batch_size}, "
                f"~{format_params(train_cfg.total_tokens)} tokens total")

    # Train
    trainer = Trainer(model, train_cfg, train_ds, val_ds, tokenizer)
    results = trainer.train()

    # Summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Steps:         {results['final_step']:,d}")
    logger.info(f"  Best val loss: {results['best_val_loss']:.4f}")
    logger.info(f"  Tokens seen:   {results['total_tokens']:,d}")
    logger.info(f"  Time:          {results['total_time_seconds']:.0f}s")
    logger.info(f"  Checkpoint:    {results['checkpoint_path']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
