"""
Flow Matrices: CFM, AFM, and Dual-Flow Scoring.

The core innovation of DSTT. These modules replace the standard
scaled dot-product attention score with a composite signal that
integrates contextual relevance (CFM) and incoherence penalty (AFM).

Dual-flow attention score:
    s_ij = A_ij + α · CFM_ij − β · AFM_ij

where A_ij is the standard QKᵀ/√d_k score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dstt.config import DSTTConfig


class CorrectFlowMatrix(nn.Module):
    """Correct Flow Matrix (CFM).

    Computes a contextual-relevance score for each key vector,
    combining Wittgenstein semantic alignment with Ramsey coherence
    against the prior state.

    CFM_j = W_s(k_j, C) + R_c(k_j, S_prev)

    In practice this is implemented as a learned projection that
    produces per-key relevance scores from the key-context interaction.
    """

    def __init__(self, d_head: int, d_model: int):
        super().__init__()
        self.d_head = d_head
        # Wittgenstein score: learned alignment between key and context
        self.context_proj = nn.Linear(d_model, d_head, bias=False)
        # Ramsey coherence: consistency with prior state
        self.state_proj = nn.Linear(d_head, d_head, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        keys: torch.Tensor,
        context: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CFM scores for each key position.

        Args:
            keys: Key vectors, shape (batch, n_heads, seq_len, d_head).
            context: Global context, shape (batch, d_model).
            prev_state: Previous layer output, shape (batch, seq_len, d_head).
                If None, the Ramsey coherence term is omitted.

        Returns:
            CFM scores, shape (batch, n_heads, 1, seq_len).
            Broadcast-ready for addition to the attention score matrix.
        """
        batch, n_heads, seq_len, d_head = keys.shape

        # Wittgenstein score: cosine similarity between keys and context
        ctx = self.context_proj(context)  # (batch, d_head)
        ctx = ctx.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, d_head)
        witt_score = F.cosine_similarity(keys, ctx.expand_as(keys), dim=-1)
        # (batch, n_heads, seq_len)

        # Ramsey coherence against prior state
        if prev_state is not None:
            # Project prior state into head space
            state = self.state_proj(prev_state)  # (batch, seq_len, d_head)
            state = state.unsqueeze(1).expand_as(keys)
            # Coherence: 1 - normalised Hamming distance of sign patterns
            key_sign = (keys > 0).float()
            state_sign = (state > 0).float()
            hamming = (key_sign != state_sign).float().mean(dim=-1)
            ramsey_score = 1.0 - hamming  # (batch, n_heads, seq_len)
        else:
            ramsey_score = torch.zeros_like(witt_score)

        cfm = (witt_score + ramsey_score) * self.scale
        return cfm.unsqueeze(2)  # (batch, n_heads, 1, seq_len)


class AdversarialFlowMatrix(nn.Module):
    """Adversarial Flow Matrix (AFM).

    Computes an incoherence penalty for each key vector, combining
    contradiction score (negative alignment) with entropy score
    (selection uncertainty).

    AFM_j = C_a(k_j, C) + E_s(k_j)
    """

    def __init__(self, d_head: int, d_model: int):
        super().__init__()
        self.d_head = d_head
        self.context_proj = nn.Linear(d_model, d_head, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        keys: torch.Tensor,
        context: torch.Tensor,
        key_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute AFM scores for each key position.

        Args:
            keys: Key vectors, shape (batch, n_heads, seq_len, d_head).
            context: Global context, shape (batch, d_model).
            key_probs: Prior selection probabilities for keys,
                shape (batch, n_heads, seq_len). If None, uniform is assumed.

        Returns:
            AFM scores, shape (batch, n_heads, 1, seq_len).
        """
        batch, n_heads, seq_len, d_head = keys.shape

        # Contradiction score: rectified negative cosine similarity
        ctx = self.context_proj(context)  # (batch, d_head)
        ctx = ctx.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, d_head)
        cos_sim = F.cosine_similarity(keys, ctx.expand_as(keys), dim=-1)
        contradiction = F.relu(-cos_sim)  # (batch, n_heads, seq_len)

        # Entropy score: self-information under current distribution
        if key_probs is not None:
            # Shannon self-information: -p * log2(p)
            probs_clamped = key_probs.clamp(min=1e-8)
            entropy = -probs_clamped * torch.log2(probs_clamped)
        else:
            # Uniform distribution: constant entropy
            entropy = torch.full_like(contradiction, 1.0 / seq_len)

        afm = (contradiction + entropy) * self.scale
        return afm.unsqueeze(2)  # (batch, n_heads, 1, seq_len)


class DualFlowScoring(nn.Module):
    """Dual-Flow Attention Scoring.

    Combines standard dot-product attention with CFM and AFM signals:
        s_ij = (Q·Kᵀ / √d_k) + α · CFM_ij − β · AFM_ij

    α and β are learnable parameters initialised to small values
    so the model starts near standard attention and gradually
    incorporates DSTT signals.
    """

    def __init__(self, config: DSTTConfig, d_head: int):
        super().__init__()
        self.cfm = CorrectFlowMatrix(d_head, config.d_model)
        self.afm = AdversarialFlowMatrix(d_head, config.d_model)
        self.alpha = nn.Parameter(torch.tensor(config.cfm_alpha_init))
        self.beta = nn.Parameter(torch.tensor(config.afm_beta_init))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        context: torch.Tensor,
        prev_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dual-flow attention scores.

        Args:
            queries: (batch, n_heads, seq_len, d_head)
            keys: (batch, n_heads, seq_len, d_head)
            context: (batch, d_model)
            prev_state: (batch, seq_len, d_head) or None
            attention_mask: (batch, 1, seq_len, seq_len) or None

        Returns:
            Composite attention scores, shape (batch, n_heads, seq_len, seq_len).
        """
        d_k = queries.shape[-1]

        # Standard scaled dot-product
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (d_k ** 0.5)

        # CFM: contextual relevance boost (broadcast across queries)
        cfm_scores = self.cfm(keys, context, prev_state)
        # (batch, n_heads, 1, seq_len) → broadcasts to (batch, n_heads, seq_len, seq_len)

        # AFM: incoherence penalty
        afm_scores = self.afm(keys, context)
        # (batch, n_heads, 1, seq_len)

        # Composite dual-flow score
        composite = attn_scores + self.alpha * cfm_scores - self.beta * afm_scores

        # Apply attention mask (e.g., causal mask)
        if attention_mask is not None:
            composite = composite.masked_fill(attention_mask == 0, float("-inf"))

        return composite
