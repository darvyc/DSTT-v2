"""
DSTT-T Quickstart Example.

Creates a small DSTT-T model, runs a forward pass, and demonstrates
the key components.
"""

import torch
from dstt import DSTTConfig, DSTTTransformer
from dstt.utils import count_parameters, format_params


def main():
    # ── 1. Create configuration ──────────────────────────────────
    print("=" * 60)
    print("DSTT-T: Dynamic Semi-Trained Topology — Transformer")
    print("=" * 60)

    config = DSTTConfig.tiny()  # ~15M params, good for CPU
    print(f"\nConfiguration:")
    print(f"  d_model:    {config.d_model}")
    print(f"  n_layers:   {config.n_layers}")
    print(f"  n_heads:    {config.n_heads}")
    print(f"  n_experts:  {config.n_experts}")
    print(f"  top_k:      {config.top_k_experts}")
    print(f"  vocab_size: {config.vocab_size}")

    # ── 2. Build model ───────────────────────────────────────────
    model = DSTTTransformer(config)
    print(f"\nModel: {model}")
    print(f"Total parameters: {format_params(count_parameters(model))}")

    # ── 3. Forward pass ──────────────────────────────────────────
    batch_size = 2
    seq_len = 64

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"\nInput shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids)

    print(f"Output shape: {logits.shape}")
    print(f"  → (batch={batch_size}, seq_len={seq_len}, vocab={config.vocab_size})")

    # ── 4. Inspect components ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Component Inspection:")
    print(f"{'─' * 60}")

    # Dual-flow attention parameters
    block = model.blocks[0]
    alpha = block.attention.dual_flow.alpha.item()
    beta = block.attention.dual_flow.beta.item()
    print(f"\nDual-Flow Attention (Layer 0):")
    print(f"  α (CFM scale): {alpha:.4f}")
    print(f"  β (AFM scale): {beta:.4f}")

    # ARM-FFN expert centroids
    centroids = block.feed_forward.centroids
    print(f"\nARM-FFN (Layer 0):")
    print(f"  Experts: {config.n_experts}")
    print(f"  Centroid shape: {centroids.shape}")
    print(f"  Load balance loss: {block.feed_forward.load_balance_loss:.6f}")

    # Wittgenstein gates
    if block.attn_gate is not None:
        print(f"\nWittgenstein Gate (Layer 0):")
        print(f"  Gate projection: {block.attn_gate.gate_proj}")

    print(f"\n{'=' * 60}")
    print("Quickstart complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
