#!/usr/bin/env python3
"""
Prepare text data for DSTT-T training.

Downloads a dataset (or reads a local file), tokenizes it, and writes
train.bin and val.bin — flat arrays of uint16 token ids that the
trainer memory-maps at runtime.

Usage:
    # Prepare from a local text file (character-level tokenizer)
    python prepare_data.py --input data/shakespeare.txt --tokenizer char

    # Prepare from a local file with GPT-2 BPE tokenizer
    python prepare_data.py --input data/corpus.txt --tokenizer gpt2

    # Download Tiny Shakespeare as a quick start
    python prepare_data.py --download shakespeare
"""

import argparse
import os
import urllib.request

import numpy as np


DOWNLOADS = {
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "input.txt",
    ),
}


def download_dataset(name: str, out_dir: str) -> str:
    """Download a named dataset and return the file path."""
    if name not in DOWNLOADS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DOWNLOADS)}")

    url, filename = DOWNLOADS[name]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    if not os.path.exists(path):
        print(f"Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
    else:
        print(f"Already exists: {path}")

    return path


def prepare(
    input_path: str,
    out_dir: str,
    tokenizer_name: str = "char",
    train_split: float = 0.9,
) -> None:
    """Tokenize a text file and write train.bin / val.bin.

    Args:
        input_path: Path to the input text file.
        out_dir: Directory for output .bin files.
        tokenizer_name: 'char' or 'gpt2'.
        train_split: Fraction used for training.
    """
    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,d} characters")

    # Tokenize
    from dstt.tokenizer import get_tokenizer

    if tokenizer_name == "char":
        tokenizer = get_tokenizer("char", text=text)
    else:
        tokenizer = get_tokenizer("gpt2")

    print(f"Tokenizing with {tokenizer_name} (vocab_size={tokenizer.vocab_size})...")
    ids = tokenizer.encode(text)
    print(f"  {len(ids):,d} tokens")

    # Split
    n = int(len(ids) * train_split)
    train_ids = np.array(ids[:n], dtype=np.uint16)
    val_ids = np.array(ids[n:], dtype=np.uint16)

    # Write
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Wrote {train_path} ({len(train_ids):,d} tokens)")
    print(f"Wrote {val_path}   ({len(val_ids):,d} tokens)")

    # Save tokenizer if char-level
    if tokenizer_name == "char":
        tok_path = os.path.join(out_dir, "tokenizer.json")
        tokenizer.save(tok_path)
        print(f"Wrote {tok_path} (vocab_size={tokenizer.vocab_size})")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for DSTT-T training")
    parser.add_argument("--input", type=str, default=None, help="Path to input text file")
    parser.add_argument("--download", type=str, default=None, choices=list(DOWNLOADS), help="Download a dataset")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "gpt2"])
    parser.add_argument("--train_split", type=float, default=0.9)
    args = parser.parse_args()

    if args.download:
        args.input = download_dataset(args.download, args.out_dir)

    if args.input is None:
        parser.error("Provide --input or --download")

    prepare(args.input, args.out_dir, args.tokenizer, args.train_split)


if __name__ == "__main__":
    main()
