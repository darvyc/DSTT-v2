"""
FDMP-E: Fundamental Data Matrix Processor — Embedding Layer.

Fuses DSTT's multimodal encoding with the transformer embedding layer.
Produces embeddings that carry modality-specific structure, partition-
aware positional encoding, and pre-computed Wittgenstein scores.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dstt.config import DSTTConfig


class FDMPEmbedding(nn.Module):
    """Hybrid embedding layer combining token/patch embedding with
    FDMP's modality-structured encoding and partition-aware positional
    encoding.

    For text:
        e_i = TokenEmbed(x_i) + PE(i) + ME(modality)

    The Wittgenstein scores are computed at embedding time and stored
    for consumption by WittgensteinGate modules downstream.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: DSTTConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding (learnable)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Modality embedding (for future multimodal extension)
        # 0 = text, 1 = image, 2 = video
        self.modality_embed = nn.Embedding(3, config.d_model)

        # Context projection for Wittgenstein score pre-computation
        # Produces a global context vector from the embedded sequence
        self.context_proj = nn.Linear(config.d_model, config.d_model)

        # Embedding dropout
        self.dropout = nn.Dropout(config.dropout)

        # Embedding scale (standard transformer practice)
        self.scale = math.sqrt(config.d_model)

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.modality_embed.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute FDMP-E embeddings.

        Args:
            input_ids: Token indices, shape (batch, seq_len).
            modality: Modality identifier (0=text, 1=image, 2=video).

        Returns:
            Tuple of:
                - embeddings: (batch, seq_len, d_model)
                - context: Global context vector (batch, d_model)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Token embedding with scaling
        tok_emb = self.token_embed(input_ids) * self.scale

        # Positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)

        # Modality embedding
        mod_id = torch.tensor([modality], device=device).expand(batch)
        mod_emb = self.modality_embed(mod_id).unsqueeze(1)  # (batch, 1, d_model)

        # Combine
        embeddings = tok_emb + pos_emb + mod_emb
        embeddings = self.dropout(embeddings)

        # Compute global context vector (mean-pooled, then projected)
        context = self.context_proj(embeddings.mean(dim=1))
        # (batch, d_model)

        return embeddings, context
