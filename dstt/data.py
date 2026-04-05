"""
Data pipeline for GPT-style pre-training.

Provides two dataset classes:

1. ``TextDataset``: Loads a text file, tokenizes it, and serves
   random contiguous chunks of ``block_size + 1`` tokens. The extra
   token provides the target for next-token prediction.

2. ``MemmapDataset``: Loads pre-tokenized data from a NumPy
   memory-mapped file (.bin), enabling training on datasets that
   don't fit in RAM.

Both return ``(x, y)`` pairs where ``x[t]`` is the input at position
``t`` and ``y[t] = x[t+1]`` is the prediction target — the standard
GPT autoregressive objective.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """In-memory text dataset for small-scale experiments.

    Tokenizes an entire text file into a 1-D array of token ids and
    serves random windows of ``block_size`` tokens (input) with their
    shifted targets (next token at each position).

    Args:
        text: Raw text string.
        tokenizer: Tokenizer with ``encode()`` method.
        block_size: Context window length.
    """

    def __init__(self, text: str, tokenizer, block_size: int):
        self.block_size = block_size
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"Text too short ({len(self.data)} tokens) for "
                f"block_size={block_size}. Need at least {block_size + 1}."
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # input tokens
        y = chunk[1:]   # target tokens (shifted right by 1)
        return x, y


class MemmapDataset(Dataset):
    """Memory-mapped dataset for large-scale pre-training.

    Reads from a flat binary file of uint16 token ids (produced by
    ``prepare_data.py``). Supports datasets larger than RAM.

    Args:
        data_path: Path to the ``.bin`` file.
        block_size: Context window length.
    """

    def __init__(self, data_path: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"Data file too short ({len(self.data)} tokens) for "
                f"block_size={block_size}."
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(
            self.data[idx : idx + self.block_size + 1].astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_batch(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from a dataset.

    This is the simple random-sampling approach used in nanoGPT.
    For each batch, ``batch_size`` random starting positions are
    drawn and a contiguous window of ``block_size`` tokens is
    extracted from each.

    Args:
        dataset: TextDataset or MemmapDataset.
        batch_size: Number of sequences per batch.
        device: Target device for the tensors.

    Returns:
        (x, y) each of shape ``(batch_size, block_size)``.
    """
    n = len(dataset)
    ix = torch.randint(0, n, (batch_size,))
    x_list, y_list = [], []
    for i in ix:
        xi, yi = dataset[i.item()]
        x_list.append(xi)
        y_list.append(yi)
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y


def create_datasets(
    text: str,
    tokenizer,
    block_size: int,
    train_split: float = 0.9,
) -> Tuple[TextDataset, TextDataset]:
    """Split text into train/val datasets.

    Args:
        text: Full text corpus.
        tokenizer: Tokenizer instance.
        block_size: Context window length.
        train_split: Fraction of data used for training.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    n = int(len(text) * train_split)
    train_text = text[:n]
    val_text = text[n:]
    train_ds = TextDataset(train_text, tokenizer, block_size)
    val_ds = TextDataset(val_text, tokenizer, block_size)
    return train_ds, val_ds
