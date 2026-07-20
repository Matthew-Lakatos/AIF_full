"""
planners/efe_planner.py

Purpose
-------
Short-horizon Expected Free Energy (EFE) planner used to score candidate
action sequences for policy-gradient style updates. The planner evaluates
imagined trajectories under a frozen world model and returns a scalar EFE
loss composed of a risk term (distance to preferred observations) and an
epistemic term (uncertainty / ensemble variance). The planner is intentionally
short-horizon and lightweight to be used inside the Active Inference agent's
update loop.

What is implemented
-------------------
- Sampling of candidate action sequences using `policy_dist.rsample()` so that
  gradients flow from EFE -> actions -> policy parameters.
- Deterministic rollout of the provided `world_model` for a fixed horizon H.
- Risk computed as mean squared error between predicted observation mean and
  preferred observations.
- Epistemic term computed from an ensemble predictive variance when available,
  otherwise approximated by the prior predictive variance from the RSSM.
- Aggregation of per-step terms across horizon, particles, and batch to produce
  scalar EFE, risk, and epistemic values.

Design choices and caveats
--------------------------
- **World model freezing**: The planner assumes the caller has set the
  world_model parameters to `requires_grad=False`. This prevents gradients
  from flowing into the world model during planner rollouts. The planner
  itself uses reparameterized sampling from the policy so gradients flow to
  policy parameters only.
- **Short-horizon approximation**: The planner samples actions independently
  per step from the current policy (no conditioning on imagined future states).
  This is a pragmatic approximation that reduces computational cost.
- **Particles**: Multiple action trajectories (particles) are sampled and
  averaged to reduce variance in the EFE estimate.
- **Epistemic summarization**: Ensemble variance is summarized as the trace
  (sum of variances across latent dims) per sample. Other summaries (max,
  mean, or learned projections) are possible and left for ablation.
- **Numerical stability**: Small Gaussian noise (`action_noise`) is added to
  sampled actions to avoid degenerate gradients and improve numerical stability.

Recommended experimental uses
-----------------------------
- Use this planner inside the AIF agent's update step with the world model
  parameters frozen for the duration of planner execution.
- Sweep `horizon`, `num_particles`, and `lambda_epistemic` to study trade-offs
  between foresight, variance, and computational cost.
- Compare ensemble-based epistemic term vs prior-logvar fallback to quantify
  the ensemble's contribution to exploration and calibration.

Minimal tests to include
------------------------
- Shape and device test: run planner with dummy world_model and policy_dist
  and assert returned tensors are scalars on the expected device.
- Gradient flow test: ensure gradients flow to policy parameters but not to
  world_model parameters when world_model parameters are frozen.
- Epistemic fallback test: run planner with `ensemble=None` and verify the
  epistemic term is computed from prior logvar.
"""

import torch
import torch.nn as nn


class EFEPlanner:
    """
    Short-horizon Expected Free Energy planner.

    Responsibilities
    ----------------
    - Sample candidate action sequences using `policy_dist.rsample()` so that
      gradients propagate to policy parameters.
    - Roll out the provided `world_model` for `horizon` steps per particle.
    - Compute per-step risk (MSE to preferred observation) and epistemic term
      (ensemble variance trace or prior variance fallback).
    - Aggregate across horizon, particles, and batch to return scalar metrics.

    Constructor arguments
    ---------------------
    - horizon (int): planning horizon in steps.
    - lambda_epistemic (float): weight applied to epistemic term in EFE.
    - num_particles (int): number of sampled action trajectories to average.
    - action_noise (float): small additive noise applied to sampled actions
      for numerical stability (still differentiable).
    """

    def __init__(self, horizon: int = 3, lambda_epistemic: float = 0.1, num_particles: int = 8, action_noise: float = 1e-3):
        self.horizon = int(horizon)
        self.lambda_epistemic = float(lambda_epistemic)
        self.num_particles = int(num_particles)  # number of sampled action trajectories
        self.action_noise = float(action_noise)  # small additive noise for stability

    def __call__(self, world_model, ensemble, policy_dist, h_start, z_start, preferred_obs):
        """
        Execute the planner and return EFE and its components.

        Args:
            world_model: RSSMWorldModel instance. Caller should set its parameters
                         `requires_grad=False` before calling the planner.
            ensemble: TransitionEnsemble instance or None. If provided, its
                      predictive variance is used as the epistemic signal.
            policy_dist: a torch.distributions-like object with `.rsample()` and
                         `.mean` attributes. Sampling shape: [B, A].
            h_start: [B, Hdim] tensor representing the current deterministic belief.
            z_start: [B, D] tensor representing the current stochastic latent.
            preferred_obs: [1, obs_dim] or [B, obs_dim] tensor of preferred observations.

        Returns:
            efe: scalar tensor (risk_mean - lambda_epistemic * epistemic_mean)
            risk_mean: scalar tensor (mean risk across particles and batch)
            epistemic_mean: scalar tensor (mean epistemic across particles and batch)

        Notes:
            - h_start and z_start are **not** detached here; the caller controls
              whether gradients should flow by freezing world_model parameters.
            - preferred_obs may be broadcast from shape [1, obs_dim] to [B, obs_dim].
        """
        device = h_start.device
        B = h_start.shape[0]

        # Determine action dimensionality from policy mean
        A = policy_dist.mean.shape[-1]

        # Sample action sequences for each horizon step using rsample so gradients
        # flow to policy parameters. Each particle gets its OWN independently
        # sampled action sequence.
        #
        # BUGFIX: this previously called policy_dist.rsample() once per horizon
        # step and then used .expand() to broadcast that single sample across
        # the particle dimension. .expand() is a broadcast view, not independent
        # sampling, so every particle received a byte-identical action sequence.
        # Combined with a deterministic world_model.transition() (the default
        # whenever an ensemble is attached -- it returns the ensemble mean, not
        # a stochastic sample), this made every particle's rollout identical,
        # so num_particles contributed zero variance reduction: risk_acc.mean(0)
        # was just averaging num_particles copies of one number. We now draw an
        # independent rsample() per (horizon step, particle) pair so particles
        # can actually diverge.
        actions_particles = []
        for t in range(self.horizon):
            a_t_per_particle = []
            for _ in range(self.num_particles):
                a_t = policy_dist.rsample()  # independent reparameterized sample [B, A]
                a_t = a_t + (self.action_noise * torch.randn_like(a_t))
                a_t_per_particle.append(a_t)
            # Stack this horizon step's independent particle samples: [P, B, A]
            actions_particles.append(torch.stack(a_t_per_particle, dim=0))
        # Stack to shape [P, H, B, A]
        actions_particles = torch.stack(actions_particles, dim=1)

        # Accumulators for per-particle, per-batch sums of risk and epistemic terms
        risk_acc = torch.zeros(self.num_particles, B, device=device)
        epistemic_acc = torch.zeros(self.num_particles, B, device=device)

        # For each particle, roll out from the same starting belief/state
        for p in range(self.num_particles):
            # Start from the provided belief and latent. The caller controls whether
            # these tensors require gradients; typically world_model params are frozen.
            h_roll = h_start
            z_roll = z_start

            # Attempt to obtain a slow latent `s` from the world model if present.
            # If not present, create a zero tensor with the expected slow_dim.
            try:
                s_roll = world_model.s.detach() if hasattr(world_model, "s") else torch.zeros(B, world_model.slow_dim, device=device)
            except Exception:
                # Defensive fallback: if accessing world_model.s fails, create zeros.
                s_roll = torch.zeros(B, world_model.slow_dim, device=device)

            # Roll forward for the planning horizon and accumulate per-step metrics.
            for t in range(self.horizon):
                a_t = actions_particles[p, t]  # [B, A]

                # Transition through the world model. The world_model should have
                # been frozen by the caller to prevent gradients flowing into it.
                h_roll, z_roll, mu_p, logvar_p = world_model.transition(h_roll, z_roll, a_t, s=s_roll)

                # Decode predicted observation mean and logvar for risk computation.
                mu_o, logvar_o = world_model.decode(h_roll, z_roll, s_roll)

                # Broadcast preferred observation to batch if necessary.
                pref = preferred_obs
                if pref.shape[0] == 1 and B > 1:
                    pref = pref.expand(B, -1)

                # Risk: mean squared error between predicted mean and preferred observation.
                # We average over observation dimensions to produce a per-sample scalar.
                risk_step = ((mu_o - pref) ** 2).mean(dim=-1)  # [B]
                risk_acc[p] += risk_step

                # Epistemic term: prefer ensemble predictive variance when available.
                if ensemble is not None:
                    # Ensemble expects the same input contract as transition: [z, a, s]
                    ensemble_inp = torch.cat([z_roll, a_t, s_roll], dim=-1)
                    mean_z, var_z = ensemble(ensemble_inp)  # var_z: [B, D]
                    # Summarize epistemic uncertainty as the trace (sum of variances).
                    epistemic_step = var_z.sum(dim=-1)  # [B]
                    epistemic_acc[p] += epistemic_step
                else:
                    # Fallback: use prior predictive variance (exp(logvar_p)) as proxy.
                    epistemic_acc[p] += logvar_p.exp().sum(dim=-1)

                # Update slow latent deterministically using the slow prior mean for next step.
                # This keeps the slow latent differentiable while relying on prior dynamics.
                if world_model.slow_dim > 0:
                    _, mu_s_p, _ = world_model.slow_prior_dist(h_roll, s_roll)
                    s_roll = mu_s_p  # differentiable tensor; world_model params should be frozen

        # Aggregate: compute mean risk and epistemic across particles and batch.
        risk_mean = risk_acc.mean(dim=0).mean()       # scalar
        epistemic_mean = epistemic_acc.mean(dim=0).mean()  # scalar

        # Compose Expected Free Energy: risk - lambda * epistemic (we minimize EFE)
        efe = risk_mean - self.lambda_epistemic * epistemic_mean

        # Return scalar tensors (efe, risk, epistemic) on the planner device.
        return efe, risk_mean, epistemic_mean


# -------------------------
# Minimal test snippets (to include in tests/test_efe_planner.py)
# -------------------------
# These tests are intentionally small and focus on shape, device, and gradient flow.
#
# import torch
# from planners.efe_planner import EFEPlanner
#
# class DummyWorld:
#     def __init__(self, obs_dim=4, latent_dim=3, slow_dim=2):
#         self.slow_dim = slow_dim
#         self.s = torch.zeros(1, slow_dim)
#     def transition(self, h, z, a, s=None):
#         # simple deterministic transition: increment h and z by small amounts
#         h_new = h + 0.01
#         z_new = z + 0.01
#         mu_p = torch.zeros_like(z_new)
#         logvar_p = torch.zeros_like(z_new)
#         return h_new, z_new, mu_p, logvar_p
#     def decode(self, h, z, s=None):
#         mu_o = torch.zeros(h.shape[0], 4)
#         logvar_o = torch.zeros_like(mu_o)
#         return mu_o, logvar_o
#     def slow_prior_dist(self, h, s_prev):
#         mu_s_p = torch.zeros_like(s_prev)
#         return None, mu_s_p, None
#
# def test_efe_shapes_and_grad_flow():
#     planner = EFEPlanner(horizon=2, num_particles=4)
#     world = DummyWorld()
#     # Simple policy: Normal with learnable mean parameter
#     mean = torch.zeros(1, 2, requires_grad=True)
#     std = torch.ones(1, 2)
#     class SimpleDist:
#         def __init__(self, mean, std):
#             self.mean = mean
#             self.std = std
#         def rsample(self):
#             return self.mean + self.std * torch.randn_like(self.mean)
#     policy = SimpleDist(mean, std)
#
#     h_start = torch.zeros(1, 8)
#     z_start = torch.zeros(1, 3)
#     pref = torch.zeros(1, 4)
#
#     # Freeze world model parameters (none in DummyWorld), but demonstrate intent:
#     # for p in world.parameters(): p.requires_grad = False
#
#     efe, risk, epi = planner(world, None, policy, h_start, z_start, pref)
#     assert torch.isfinite(efe)
#     # Backpropagate to ensure gradients reach policy mean
#     efe.backward()
#     assert mean.grad is not None
#
# Note: adapt these snippets to your real world_model and policy classes.
