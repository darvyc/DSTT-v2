#!/usr/bin/env python3
"""
Example: Train DSTT-T on Tiny Shakespeare.

A complete end-to-end example that downloads the dataset, tokenizes
it, trains a small model, and generates text — all in one script.

Usage:
    python examples/train_shakespeare.py
"""

import os
import sys
import urllib.request

import torch

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dstt import DSTTConfig, DSTTTransformer
from dstt.data import create_datasets
from dstt.tokenizer import CharTokenizer
from dstt.train_config import TrainConfig
from dstt.trainer import Trainer
from dstt.generate import generate_text
from dstt.utils import count_parameters, format_params

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    # ── 1. Download Tiny Shakespeare ─────────────────────────────
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "shakespeare.txt")

    if not os.path.exists(txt_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        logger.info(f"Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(url, txt_path)
    
    with open(txt_path, "r") as f:
        text = f.read()
    logger.info(f"Dataset: {len(text):,d} characters")

    # ── 2. Tokenize ──────────────────────────────────────────────
    tokenizer = CharTokenizer.from_text(text)
    logger.info(f"Vocabulary: {tokenizer.vocab_size} characters")

    # ── 3. Create datasets ───────────────────────────────────────
    block_size = 256
    train_ds, val_ds = create_datasets(text, tokenizer, block_size, train_split=0.9)
    logger.info(f"Train: {len(train_ds):,d} windows | Val: {len(val_ds):,d} windows")

    # ── 4. Build model ───────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = DSTTConfig.tiny()
    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.max_seq_len = block_size
    model = DSTTTransformer(model_cfg)
    logger.info(f"Model: {format_params(count_parameters(model))} parameters")

    # ── 5. Train ─────────────────────────────────────────────────
    train_cfg = TrainConfig(
        batch_size=32,
        block_size=block_size,
        max_steps=3000,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=100,
        weight_decay=0.1,
        eval_interval=250,
        eval_steps=10,
        log_interval=50,
        save_interval=1000,
        save_dir="checkpoints",
        sample_interval=500,
        sample_prompt="\nROMEO:\n",
        sample_max_tokens=200,
        device=device,
    )

    trainer = Trainer(model, train_cfg, train_ds, val_ds, tokenizer)
    results = trainer.train()

    # ── 6. Generate samples ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Generating samples from trained model:")
    logger.info("=" * 60)

    prompts = [
        "\nROMEO:\n",
        "\nJULIET:\n",
        "\nTo be, or not to be",
    ]

    for prompt in prompts:
        logger.info(f"\nPrompt: {repr(prompt)}")
        logger.info("-" * 40)
        sample = generate_text(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.8,
            top_k=40,
            device=device,
        )
        print(sample)

    logger.info(f"\nTraining complete! Checkpoint: {results['checkpoint_path']}")
    logger.info(f"Generate more with: python generate.py --checkpoint {results['checkpoint_path']}")


if __name__ == "__main__":
    main()
