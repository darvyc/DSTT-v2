"""
DSTT-T Model: DSTTv2 and DSTTBlock.

The full DSTT-Transformer architecture. Each DSTTBlock composes:
1. LayerNorm → LTM (Ramsey-partitioned dual-flow attention)
2. Wittgenstein Gate on attention output
3. Residual connection
4. LayerNorm → ARM-FFN (partition-gated mixture of experts)
5. Wittgenstein Gate on FFN output
6. Residual connection

An L-layer model stacks L DSTTBlocks, preceded by FDMP-E embedding
and followed by a language-model head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dstt.config import DSTTConfig
from dstt.attention import LightweightTensorMixer
from dstt.embedding import FDMPEmbedding
from dstt.routing import ARMFeedForward
from dstt.gating import WittgensteinGate
from dstt.utils import RMSNorm, count_parameters, format_params


class DSTTBlock(nn.Module):
    """A single DSTT-T DSTT-v2 block.

    Computation flow::

        z  = LayerNorm(x)
        a  = LTM(z, context, prev_state)
        x' = x + w₁ ⊙ a                     (Wittgenstein-gated residual)
        z' = LayerNorm(x')
        f  = ARM-FFN(z')
        x''= x' + w₂ ⊙ f                    (Wittgenstein-gated residual)

    Args:
        config: Model configuration.
        layer_idx: Index of this layer in the stack.
    """

    def __init__(self, config: DSTTConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-norm layers
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        # Core sub-layers
        self.attention = LightweightTensorMixer(config, layer_idx)
        self.feed_forward = ARMFeedForward(config)

        # Wittgenstein gates (optional)
        if config.use_wittgenstein_gate:
            self.attn_gate = WittgensteinGate(config.d_model)
            self.ffn_gate = WittgensteinGate(config.d_model)
        else:
            self.attn_gate = None
            self.ffn_gate = None

        # Dropout on residual
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        prev_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of a single DSTT-T block.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            context: Global context vector, shape (batch, d_model).
            prev_state: Previous layer output for Ramsey coherence.
            attention_mask: Causal or padding mask.

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        # ── Attention sub-layer ──
        z = self.attn_norm(x)
        attn_out = self.attention(z, context, prev_state, attention_mask)
        attn_out = self.residual_dropout(attn_out)

        # Wittgenstein gate on attention output
        if self.attn_gate is not None:
            gate = self.attn_gate(x, context)  # (batch, seq_len, 1)
            x = x + gate * attn_out
        else:
            x = x + attn_out

        # ── FFN sub-layer ──
        z = self.ffn_norm(x)
        ffn_out = self.feed_forward(z)
        ffn_out = self.residual_dropout(ffn_out)

        # Wittgenstein gate on FFN output
        if self.ffn_gate is not None:
            gate = self.ffn_gate(x, context)
            x = x + gate * ffn_out
        else:
            x = x + ffn_out

        return x


class DSTTv2(nn.Module):
    """Complete DSTT-T model.

    Architecture::

        Input IDs → FDMP-E (embedding + context) → [DSTTBlock × L] → LM Head → Logits

    The model produces a global context vector from the embedding layer
    and passes it through every block. Each block also receives the
    previous block's output as ``prev_state`` for Ramsey coherence in
    the CFM computation.

    Args:
        config: Model configuration.

    Example::

        config = DSTTConfig.tiny()
        model = DSTTv2(config)
        x = torch.randint(0, config.vocab_size, (2, 128))
        logits = model(x)  # (2, 128, vocab_size)
    """

    def __init__(self, config: DSTTConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embedding
        self.embedding = FDMPEmbedding(config)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DSTTBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = RMSNorm(config.d_model)

        # Language model head (tied weights with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.token_embed.weight

        # Initialise weights
        self.apply(self._init_weights)

        # Log parameter count
        n_params = count_parameters(self)
        self._n_params = n_params

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _make_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal attention mask.

        Returns:
            Mask of shape (1, 1, seq_len, seq_len) where 1 = attend, 0 = mask.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        modality: int = 0,
    ) -> torch.Tensor:
        """Forward pass of the full DSTT-T model.

        Args:
            input_ids: Token indices, shape (batch, seq_len).
            attention_mask: Optional attention mask. If None, a causal
                mask is generated automatically.
            modality: Modality identifier (0=text, 1=image, 2=video).

        Returns:
            Logits, shape (batch, seq_len, vocab_size).
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding
        x, context = self.embedding(input_ids, modality=modality)
        # x: (batch, seq_len, d_model)
        # context: (batch, d_model)

        # Causal mask
        if attention_mask is None:
            attention_mask = self._make_causal_mask(seq_len, device, x.dtype)

        # Process through DSTT-T blocks
        prev_state = None
        for block in self.blocks:
            x = block(x, context, prev_state, attention_mask)
            prev_state = x.detach()  # Detach for Ramsey coherence (no grad through state)

        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return total parameter count.

        Args:
            non_embedding: If True, exclude embedding parameters.
        """
        n = self._n_params
        if non_embedding:
            n -= self.embedding.token_embed.weight.numel()
        return n

    def __repr__(self) -> str:
        return (
            f"DSTTv2(\n"
            f"  layers={self.config.n_layers}, "
            f"d_model={self.config.d_model}, "
            f"heads={self.config.n_heads}, "
            f"experts={self.config.n_experts}\n"
            f"  params={format_params(self._n_params)}, "
            f"tensor_fold=True, "
            f"wittgenstein_gate={self.config.use_wittgenstein_gate}\n"
            f")"
        )


# Backward-compatible alias
DSTTTransformer = DSTTv2
