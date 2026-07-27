"""
precision_controller.py

Purpose
-------
Small neural controller that maps a pair of scalar signals (prediction error,
latent change magnitude) to a positive precision weight beta. The controller
is intended to modulate the KL weight (precision) in the variational free
energy objective dynamically.

What is implemented
-------------------
- `PrecisionController(nn.Module)`:
    - A compact MLP taking two inputs per sample and producing a single
      positive scalar output per sample (clamped to [0.01, 2.0]).
    - Input preprocessing uses tanh to bound inputs before the MLP.
    - Designed to be used in batched settings: accepts tensors of shape
      compatible with broadcasting to [B, 1].

Design notes (for Methods / Appendix)
-------------------------------------
- The network is intentionally small (32 hidden units) to keep the precision
  controller lightweight and stable during training.
- Inputs are squashed with `tanh` before being concatenated; this bounds the
  dynamic range and reduces sensitivity to outliers.
- Final output is mapped via sigmoid to (0.01, 2.0) to ensure differentiability.
- The final layer bias is initialised to 1.0 to avoid getting stuck at the lower bound.

Suggested experiments
---------------------
- Ablate the controller by replacing it with a fixed beta (e.g., 1.0) to
  measure stability and performance impact.
- Sweep hidden width (e.g., 8, 32, 128) to test sensitivity to controller capacity.
- Compare `tanh` preprocessing vs raw inputs or `softplus` preprocessing.
"""

import torch
import torch.nn as nn


class PrecisionController(nn.Module):
    """
    PrecisionController

    A compact MLP that maps (prediction_error, latent_delta) -> beta.

    Usage:
        pc = PrecisionController()
        beta = pc(error_tensor, delta_tensor)  # beta shape: [B, 1] or [B]

    Notes:
    - The forward method ensures inputs are 2D column vectors ([B,1]) so the
      same code works for scalars and batched tensors.
    - The output is mapped via sigmoid to (0.01, 2.0) to keep beta in a numerically
      reasonable range and ensure differentiability.
    """

    def __init__(self):
        super().__init__()
        # Small MLP: 2 -> 32 -> 1. Kept intentionally compact.
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Initialize the final layer bias to 1.0 so that raw output starts positive,
        # preventing the sigmoid from mapping to near 0.01.
        nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(self, error, delta):
        """
        Compute precision beta from prediction error and latent delta.

        Args:
            error: tensor-like scalar or shape [B] or [B,1]; prediction error.
            delta: tensor-like scalar or shape [B] or [B,1]; latent change magnitude.

        Returns:
            beta: tensor shape [B,1] with values in (0.01, 2.0).
        """
        # Ensure both are 2D column vectors: [B,1]. This handles scalars too.
        error = error.view(-1, 1)
        delta = delta.view(-1, 1)

        # Squash inputs to [-1, 1] to reduce sensitivity to outliers.
        e = torch.tanh(error)
        d = torch.tanh(delta)

        # Concatenate into [B, 2] and run through the network.
        inp = torch.cat([e, d], dim=-1)
        beta_raw = self.net(inp)

        # Differentiable mapping to (0.01, 2.0)
        beta = 0.01 + 1.99 * torch.sigmoid(beta_raw)
        return beta
