"""
Example: Train a small DSTT-T language model.

Demonstrates the complete training loop with dummy data.
Replace the data generation with your own dataset for real training.
"""

import torch
from torch.optim import AdamW

from dstt import DSTTConfig, DSTTTransformer
from dstt.losses import DSTTLoss
from dstt.utils import count_parameters, format_params


def main():
    # Use tiny config for fast iteration
    config = DSTTConfig.tiny()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building DSTT-T model on {device}...")
    model = DSTTTransformer(config).to(device)
    print(f"Parameters: {format_params(count_parameters(model))}")

    # Optimiser and loss
    optimiser = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = DSTTLoss(load_balance_weight=config.load_balance_weight)

    # Training loop (dummy data)
    model.train()
    for step in range(1, 51):
        # Dummy data — replace with real tokenised text
        input_ids = torch.randint(0, config.vocab_size, (8, 128), device=device)
        targets = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)

        optimiser.zero_grad()
        logits = model(input_ids)
        losses = loss_fn(logits, targets, model)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if step % 10 == 0:
            print(
                f"Step {step:3d} | "
                f"Loss: {losses['loss'].item():.4f} | "
                f"Task: {losses['task_loss'].item():.4f} | "
                f"LB: {losses['lb_loss'].item():.6f}"
            )

    print("\nTraining complete!")

    # Inspect learned dual-flow parameters
    print("\nLearned Dual-Flow Parameters:")
    for i, block in enumerate(model.blocks):
        alpha = block.attention.dual_flow.alpha.item()
        beta = block.attention.dual_flow.beta.item()
        print(f"  Layer {i}: α={alpha:.4f}, β={beta:.4f}")


if __name__ == "__main__":
    main()
