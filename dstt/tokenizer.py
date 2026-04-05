"""
Tokenizer utilities for DSTT-T.

Provides a simple character-level tokenizer (zero dependencies) and
an optional wrapper around ``tiktoken`` for GPT-2 BPE tokenization.

The character-level tokenizer is useful for small-scale experiments
(Shakespeare, code files). For production pre-training, use the
GPT-2 BPE tokenizer.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional


class CharTokenizer:
    """Character-level tokenizer.

    Builds a vocabulary from a text corpus. Every unique character
    becomes a token. Simple, deterministic, no dependencies.

    Usage::

        tok = CharTokenizer.from_text(open("data.txt").read())
        ids = tok.encode("Hello world")
        text = tok.decode(ids)
        tok.save("tokenizer.json")
        tok = CharTokenizer.load("tokenizer.json")
    """

    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build tokenizer from a text corpus."""
        return cls(text)

    @classmethod
    def from_file(cls, path: str) -> CharTokenizer:
        """Build tokenizer from a text file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls(f.read())

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token ids."""
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        return "".join(self.itos.get(i, "?") for i in ids)

    def save(self, path: str) -> None:
        """Save tokenizer to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> CharTokenizer:
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.chars = data["chars"]
        tok.stoi = {ch: i for i, ch in enumerate(tok.chars)}
        tok.itos = {i: ch for i, ch in enumerate(tok.chars)}
        return tok


class GPT2Tokenizer:
    """Wrapper around ``tiktoken`` for GPT-2 BPE tokenization.

    Requires ``pip install tiktoken``. Falls back to CharTokenizer
    if tiktoken is not available.

    Usage::

        tok = GPT2Tokenizer()
        ids = tok.encode("Hello world")
        text = tok.decode(ids)
    """

    def __init__(self):
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("gpt2")
        except ImportError:
            raise ImportError(
                "GPT2Tokenizer requires tiktoken. "
                "Install with: pip install tiktoken"
            )

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab  # 50257

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)


def get_tokenizer(name: str = "char", text: str = "") -> CharTokenizer | GPT2Tokenizer:
    """Factory function to create a tokenizer.

    Args:
        name: 'char' for character-level, 'gpt2' for BPE.
        text: Training text (required for 'char' tokenizer).

    Returns:
        Tokenizer instance.
    """
    if name == "gpt2":
        return GPT2Tokenizer()
    elif name == "char":
        if not text:
            raise ValueError("CharTokenizer requires text to build vocabulary.")
        return CharTokenizer.from_text(text)
    else:
        raise ValueError(f"Unknown tokenizer: {name}. Use 'char' or 'gpt2'.")
