"""
Example: Custom DSTT-T configurations.

Demonstrates how to customise every aspect of the architecture.
"""

from dstt import DSTTConfig, DSTTTransformer
from dstt.utils import count_parameters, format_params


def main():
    # ── Fully custom configuration ───────────────────────────────
    config = DSTTConfig(
        d_model=512,
        n_layers=8,
        vocab_size=32000,
        max_seq_len=1024,
        n_heads=16,
        use_ramsey_heads=True,       # Enable Ramsey partitioning
        coherence_threshold=0.3,     # Higher τ = more, smaller partitions
        cfm_alpha_init=0.15,         # Slightly stronger CFM signal
        afm_beta_init=0.05,          # Weaker AFM penalty
        d_ff=2048,
        n_experts=8,
        top_k_experts=2,
        load_balance_weight=0.02,
        use_wittgenstein_gate=True,
        dropout=0.05,
    )

    model = DSTTTransformer(config)
    print(f"Custom model: {format_params(count_parameters(model))} params")
    print(f"Ramsey head count target: {config.ramsey_head_count}")

    # ── Configuration without DSTT enhancements ──────────────────
    # This produces a near-standard transformer (for ablation)
    vanilla_config = DSTTConfig(
        d_model=512,
        n_layers=8,
        vocab_size=32000,
        n_heads=8,
        use_ramsey_heads=False,      # Fixed equal-width heads
        cfm_alpha_init=0.0,          # No CFM signal
        afm_beta_init=0.0,           # No AFM signal
        n_experts=1,                 # Single expert = standard FFN
        top_k_experts=1,
        use_wittgenstein_gate=False,  # No gating
        d_ff=2048,
    )

    vanilla = DSTTTransformer(vanilla_config)
    print(f"\nVanilla-equivalent: {format_params(count_parameters(vanilla))} params")

    # ── Compare parameter counts ─────────────────────────────────
    dstt_params = count_parameters(model)
    vanilla_params = count_parameters(vanilla)
    overhead = (dstt_params - vanilla_params) / vanilla_params * 100
    print(f"DSTT-T overhead: {overhead:.1f}% more parameters")


if __name__ == "__main__":
    main()
