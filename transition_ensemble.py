"""
transition_ensemble.py

-------
Ensemble of small MLP transition networks that map an input vector
(e.g., concatenated [z, a, s]) to a predicted latent vector. The module
returns per-sample predictive mean and variance across ensemble members.

What is implemented
-------------------
- Class `TransitionEnsemble(nn.Module)` implementing an ensemble of MLPs.
- Supports eager construction (pass input_dim at init) or lazy construction
  (builds on first forward pass using observed input width).
- Each ensemble member is an MLP: Linear(input_dim, 128) -> ReLU -> Linear(128, latent_dim).
- Forward returns (mean, var) with shapes:
    mean: [B, latent_dim], var: [B, latent_dim].

Design choices and caveats
--------------------------
- Rebuilding the ensemble (when input width changes) resets member weights.
  In experiments we construct the ensemble eagerly to avoid weight resets.
- Variance is computed across ensemble members (torch.var with default unbiased
  estimator). Downstream code interprets this variance as an epistemic score.
- The module selects device at construction (CUDA if available) and moves
  parameters accordingly; inputs are moved to the module device if needed.

Defaults used in experiments (suggested)
---------------------------------------
- ensemble_size = 5
- hidden width = 128
- activation = ReLU

Suggested experiments (brief)
-----------------------------
- Sweep ensemble_size ∈ {1,3,5,10} and report calibration (corr(var, prediction_error)).
- Compare eager vs lazy construction to confirm no unintended resets.
- Replace MLP members with deeper or residual blocks to test capacity vs calibration.
- Measure forward-pass latency vs ensemble_size to quantify compute trade-offs.
"""

import torch
import torch.nn as nn


class TransitionEnsemble(nn.Module):
    """
    Ensemble of transition models used to estimate epistemic uncertainty.

    Implemented behaviour
    --------------------
    - Constructor accepts positional args (input_dim, latent_dim) or keywords.
    - If input_dim is omitted, the ensemble is built on the first forward call
      using the observed input width.
    - Each ensemble member is a small MLP returning a raw latent vector.
    - forward(x) returns (mean, var) across ensemble members.

    Important implementation details
    -------------------------------
    - Rebuilding the ensemble resets weights. If stable weights are required,
      pass input_dim at initialization (eager construction).
    - The module chooses device at construction (CUDA if available) and moves
      parameters accordingly; inputs are moved to the module device if needed.
    """

    def __init__(self, *args, input_dim=None, latent_dim=None, ensemble_size=3, **kwargs):
        super().__init__()

        # Backwards-compatible parsing: allow TransitionEnsemble(16, 32)
        if len(args) >= 2:
            # If positional args are provided, prefer them unless explicit
            # keyword args override them.
            input_dim = input_dim if input_dim is not None else args[0]
            latent_dim = latent_dim if latent_dim is not None else args[1]
        else:
            # Accept common alternative names for input_dim for compatibility.
            if input_dim is None:
                input_dim = kwargs.get('input_dim', kwargs.get('action_dim', None))
            if latent_dim is None:
                latent_dim = kwargs.get('latent_dim', None)

        if latent_dim is None:
            # latent_dim is required; raise a clear error for the caller.
            raise TypeError('TransitionEnsemble requires latent_dim (positional or keyword).')

        # Store core attributes. Cast to int to avoid accidental float types.
        self.input_dim = int(input_dim) if input_dim is not None else None
        self.latent_dim = int(latent_dim)
        self.ensemble_size = int(ensemble_size)

        # Choose device once. Module parameters will be moved to this device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ModuleList holds ensemble members so parameters are registered.
        self.models = nn.ModuleList()
        if self.input_dim is not None:
            # If input_dim known at init, build models now (eager construction).
            self._build_models(self.input_dim)

        # Ensure parameters are on the chosen device.
        self.to(self.device)

    def _build_models(self, input_dim):
        """
        Build the ensemble members for a given input dimension.

        Notes:
        - This replaces any existing models with newly initialised ones.
        - Calling this during training will reset weights; document this in
          experimental protocols if lazy building is used.
        """
        # Simple two-layer MLP per ensemble member. These are intentionally
        # small for speed; increase capacity if needed for complex tasks.
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),  # hidden width chosen for experiments
                nn.ReLU(),
                nn.Linear(128, self.latent_dim)
            ) for _ in range(self.ensemble_size)
        ])
        # Update stored input dimension and move to device.
        self.input_dim = int(input_dim)
        self.to(self.device)

    def forward(self, x):
        """
        Compute ensemble predictions and return mean and variance.

        Args:
            x (torch.Tensor): shape [B, input_dim]. Must be 2D.

        Returns:
            mean (torch.Tensor): [B, latent_dim] mean across ensemble members.
            var  (torch.Tensor): [B, latent_dim] variance across ensemble members.

        Behavioural notes
        -----------------
        - If the module was created without input_dim, the first forward will
          build the ensemble to match x.size(-1).
        - If x.size(-1) differs from the current input_dim, the ensemble will
          be rebuilt to match the new width (weights reset).
        - Inputs are moved to the module device if necessary.
        """
        # Expect a 2D tensor [B, input_dim]; fail early if not.
        assert x.dim() == 2, f"Expected [B, input_dim], got {x.shape}"
        observed_in = x.size(-1)

        # Lazy init or adapt to changed input width.
        if self.input_dim is None:
            # First forward: build models to match observed input size.
            self._build_models(observed_in)
        elif observed_in != self.input_dim:
            # Input width changed; rebuild to adapt. This resets weights.
            self._build_models(observed_in)

        # Move input to module device if needed. Prefer caller to handle device,
        # but this is a safe fallback to avoid device mismatch errors.
        if x.device != self.device:
            x = x.to(self.device)

        # Compute each member's prediction. This is a Python-level loop over
        # ensemble members; for large ensembles or many candidates, consider
        # vectorised simulation in downstream code.
        preds = [m(x) for m in self.models]  # list of [B, latent_dim]

        # Stack predictions along ensemble axis: [E, B, latent_dim]
        stacked = torch.stack(preds, dim=0)

        # Mean and variance across ensemble members -> [B, latent_dim]
        mean = stacked.mean(dim=0)
        # NOTE (bugfix): torch.var() defaults to the unbiased (Bessel-corrected)
        # estimator, which divides by (n - 1). With ensemble_size=1 -- a value
        # this module's own docstring suggests sweeping -- that means dividing
        # by zero, silently producing NaN that then propagates into the
        # epistemic term, the overshoot pseudo-variance, and the loss.
        # We use population variance (unbiased=False) instead: with n=1 this
        # correctly returns 0 (a single model has no ensemble disagreement,
        # which is the mathematically sensible value), and for n>1 it's still
        # a reasonable variance estimate across the full set of members we
        # actually have (we aren't sampling from a larger population).
        var = stacked.var(dim=0, unbiased=False)

        return mean, var


# -------------------------
# Implementation summary (for inclusion in Methods)
# -------------------------
# - TransitionEnsemble implements an ensemble of small MLPs that predict a
#   latent vector from an input vector. The ensemble returns per-sample mean
#   and variance across members. In experiments we use ensemble_size=5 and
#   hidden width 128. The ensemble variance is used as an epistemic signal
#   in planning.
#
# - The module supports lazy construction for convenience, but eager
#   construction (passing input_dim at init) is recommended in training to
#   avoid weight resets when input dimensionality changes.
#
# Suggested short description for the paper:
# "We implement the transition ensemble as an ensemble of two-layer MLPs
# (128 hidden units, ReLU). The ensemble returns per-sample predictive mean
# and variance by stacking member outputs and computing mean and variance
# across the ensemble axis. We use an ensemble size of 5 in our experiments."
#
# -------------------------
# Minimal test ideas (to include in repo tests)
# -------------------------
# - Forward pass shape and finiteness test with dummy input.
# - Lazy build behavior test: call forward with two different input widths.
# - Device test: forward with CPU and GPU inputs.
#
# -------------------------
# Small TODOs (non-functional suggestions)
# -------------------------
# - Optionally expose `predict_all()` to return raw per-member outputs.
# - Add an argument to control unbiased vs population variance in forward().
# - Consider an option to preserve weights when adapting input_dim (e.g.
#   by projecting old weights into new input space).
