"""
Loss functions for DSTT training.

Includes the composite DSTT loss that combines standard cross-entropy
with the ARM-FFN load-balancing auxiliary loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoadBalanceLoss(nn.Module):
    """Auxiliary loss encouraging uniform expert utilisation.

    L_bal = γ · E · Σ_k (f_k · p_k)

    where f_k is the fraction of tokens dispatched to expert k
    and p_k is the average gating probability for expert k.

    This loss is computed inside ARMFeedForward and collected here.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Collect and sum load-balancing losses from all ARM-FFN layers.

        Args:
            model: The full DSTTv2 model.

        Returns:
            Scalar load-balancing loss.
        """
        total = torch.tensor(0.0, device=next(model.parameters()).device)
        for module in model.modules():
            if hasattr(module, "load_balance_loss"):
                lb = module.load_balance_loss
                if lb is not None:
                    total = total + lb
        return total


class DSTTLoss(nn.Module):
    """Composite loss for DSTT training.

    L = L_task + L_balance

    where L_task is the standard cross-entropy (for language modelling)
    or task-specific loss, and L_balance is the ARM-FFN load-balancing
    auxiliary loss summed across all layers.

    Args:
        load_balance_weight: Weight for the load-balancing loss term.
        label_smoothing: Label smoothing for cross-entropy.
    """

    def __init__(
        self,
        load_balance_weight: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.task_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )
        self.lb_loss = LoadBalanceLoss(weight=load_balance_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss.

        Args:
            logits: Model predictions, shape (batch, seq_len, vocab_size).
            targets: Target token ids, shape (batch, seq_len).
            model: The full model (for collecting auxiliary losses).

        Returns:
            Dict with 'loss' (total), 'task_loss', and 'lb_loss'.
        """
        # Reshape for cross-entropy: (batch*seq, vocab) vs (batch*seq,)
        task_loss = self.task_loss(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        # Collect load-balancing loss
        lb_loss = self.lb_loss(model)

        total = task_loss + lb_loss

        return {
            "loss": total,
            "task_loss": task_loss,
            "lb_loss": lb_loss,
        }
