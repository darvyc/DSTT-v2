"""
Autoregressive text generation for DSTT-T.

Implements the standard GPT generation loop: given a prompt, feed it
through the model, sample the next token from the output distribution,
append it, and repeat.

Supports:
- **Temperature scaling**: Controls randomness (lower = more deterministic).
- **Top-k sampling**: Restricts sampling to the top-k most probable tokens.
- **Top-p (nucleus) sampling**: Restricts sampling to the smallest set
  of tokens whose cumulative probability exceeds p.
- **Greedy decoding**: Equivalent to temperature=0 or top_k=1.
- **Repetition penalty**: Discourages the model from repeating tokens.
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively.

    This is the core generation loop. It mirrors the procedure used
    by GPT-2/3: at each step, the full context (up to ``max_seq_len``)
    is fed through the model, logits for the last position are
    extracted, sampling is applied, and the new token is appended.

    Args:
        model: A ``DSTTTransformer`` instance (or any model that takes
            ``(batch, seq_len)`` int tensors and returns
            ``(batch, seq_len, vocab_size)`` logits).
        prompt_ids: Starting token ids, shape ``(1, prompt_len)``.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Values < 1.0 sharpen the
            distribution (more deterministic); > 1.0 flatten it
            (more random). Use 0.0 for greedy decoding.
        top_k: If > 0, only sample from the top-k most probable tokens.
        top_p: If < 1.0, apply nucleus sampling: keep the smallest
            set of tokens whose cumulative probability exceeds p.
        repetition_penalty: Penalty factor for previously generated
            tokens. Values > 1.0 discourage repetition.
        eos_token_id: If set, stop generation when this token is produced.

    Returns:
        Full token sequence (prompt + generated), shape ``(1, total_len)``.
    """
    model.eval()
    device = prompt_ids.device
    max_seq_len = getattr(model.config, "max_seq_len", 2048)
    ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len if needed
        context = ids if ids.shape[1] <= max_seq_len else ids[:, -max_seq_len:]

        # Forward pass
        logits = model(context)

        # Extract logits for the last position
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Greedy decoding
        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            continue

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val = logits.topk(top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative probability above the threshold
            # Shift right to keep at least one token
            sorted_mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            # Scatter back to original order
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Append and check EOS
        ids = torch.cat([ids, next_id], dim=1)
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return ids


def generate_text(
    model,
    tokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    device: torch.device | str = "cpu",
) -> str:
    """High-level generation function: prompt string in, text string out.

    Args:
        model: DSTTTransformer instance.
        tokenizer: Tokenizer with encode/decode methods.
        prompt: Input text prompt.
        max_new_tokens: Tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filter (0 = disabled).
        top_p: Nucleus sampling threshold.
        repetition_penalty: Repetition penalty factor.
        device: Device for computation.

    Returns:
        Generated text string (including the prompt).
    """
    if isinstance(device, str):
        device = torch.device(device)

    prompt_ids = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)

    output_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    return tokenizer.decode(output_ids[0].tolist())
