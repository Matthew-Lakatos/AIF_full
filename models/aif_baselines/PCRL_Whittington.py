"""
models/aif_baselines/PCRL_Whittington.py

Purpose
-------
Predictive Coding RL (PCRL) baseline inspired by Whittington-style predictive
coding. This model performs iterative latent inference (predictive coding)
to minimise observation prediction error, then updates generative model
parameters to improve predictions. A policy acts on the inferred latent.

What is implemented
-------------------
- Generative network `gen_net` mapping latent -> predicted observation.
- Policy network `policy_net` mapping latent -> action mean (Gaussian policy).
- Iterative latent optimisation (predictive coding inference) for a fixed
  number of iterations (`iters_pc`) with a simple gradient-ascent update on
  the latent to reduce prediction error.
- Parameter update step that backpropagates through the generative model
  using the final inferred latent.
- Simple belief management: `latent_state` stores the current inferred latent.
- Diagnostics for logging prediction error and gradient norms.

Design notes and caveats
-----------------------
- The inference step treats the latent as a trainable variable and performs
  a small number of gradient steps to reduce prediction error. This is a
  common predictive-coding style approach and is intentionally lightweight.
- The latent optimisation uses `requires_grad_(True)` and manual gradient
  updates; the generative model parameters are updated separately in the
  learning phase.
- The learning step detaches the inferred latent when used as input to the
  parameter update to avoid second-order gradients and to keep the update
  stable.
- The policy is conditioned on the current `latent_state` (updated after
  inference) and uses a shared learnable `log_std` for action stochasticity.
- This baseline is single-step and online; it does not implement replay,
  overshooting, or ensemble epistemics.

Suggested experiments
---------------------
- Sweep `iters_pc` and `lr_state` to study the trade-off between inference
  accuracy and computational cost.
- Compare predictive-coding inference vs amortized encoder approaches on
  blackout and latency environments.
- Test sensitivity to the number of inference iterations and to learning
  rate schedules for the generative model.

Minimal tests to include
------------------------
- Ensure `update()` runs without error and reduces prediction error after a
  few parameter updates.
- Verify `latent_state` changes after `update()` and that `get_action()`
  returns correctly shaped outputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PCRL_Whittington(nn.Module):
    """
    Predictive Coding RL-style agent.

    Responsibilities
    ----------------
    - Maintain a latent state `latent_state`.
    - Perform iterative latent inference to minimise prediction error on the
      next observation (`iters_pc` iterations of gradient-based updates).
    - Update generative model parameters to reduce prediction error using the
      final inferred latent.
    - Provide a Gaussian policy conditioned on the inferred latent.
    """

    def __init__(self, obs_dim=16, action_dim=6, latent_dim=32, iters_pc=3, lr_state=0.1, gamma=0.99):
        super().__init__()
        # Dimensions and hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.iters_pc = int(iters_pc)    # number of inference iterations
        self.lr_state = float(lr_state)  # step size for latent updates during inference
        self.gamma = gamma

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generative network: latent -> predicted observation
        self.gen_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, obs_dim),
        )

        # Policy network: latent -> action mean
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        # Shared learnable log-std for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # BUGFIX: added critic, needed for policy_net to receive ANY gradient
        # at all. Previously the only backpropagated loss (final prediction
        # error after predictive-coding inference) depended solely on gen_net
        # -- policy_net and log_std were registered with the optimizer but
        # never appeared in any loss computation, so they stayed at their
        # random initialization for the entire run.
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Optimizer for model parameters (generative + policy + critic)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        # Current belief (latent state) initialised to zeros (batch size 1)
        self.latent_state = torch.zeros(1, latent_dim, device=self.device)

        # Diagnostics for logging and analysis
        self.last_prediction_error = 0.0
        self.last_grad_norm = 0.0
        self.last_blackout_active = False
        self.last_seed = None
        # BUGFIX: new diagnostics for the actor-critic term.
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0
        self.last_pseudo_reward = 0.0

        # Move module to device
        self.to(self.device)

    # -------------------------
    # Belief management
    # -------------------------
    def reset_belief(self):
        """Reset the internal latent state to zeros (use at episode boundaries)."""
        self.latent_state = torch.zeros(1, self.latent_dim, device=self.device)

    # -------------------------
    # Policy helpers
    # -------------------------
    def _policy_dist(self):
        """
        Return a Normal distribution parameterised by the policy network
        conditioned on the current latent_state.
        """
        mu_a = self.policy_net(self.latent_state)
        std_a = torch.exp(torch.clamp(self.log_std, -5, 2)).expand_as(mu_a)
        return Normal(mu_a, std_a)

    # -------------------------
    # Action selection
    # -------------------------
    def get_action(self, obs):
        """
        Sample an action from the policy conditioned on the current latent_state.

        Note: predictive coding inference is performed in `update()`; `get_action`
        assumes `latent_state` is already up-to-date.
        """
        dist = self._policy_dist()
        a = dist.sample()
        logprob = dist.log_prob(a).sum(-1)
        entropy = dist.entropy().sum(-1)
        return (
            a.detach().squeeze(0).cpu().numpy(),
            float(logprob.item()),
            float(entropy.item()),
            0.0,
        )

    # -------------------------
    # Training update
    # -------------------------
    def update(self, prev_obs, action, next_obs):
        """
        Perform predictive-coding inference followed by parameter learning.

        Steps:
        1. Create a trainable copy of the current latent_state `s` with gradients.
        2. Iteratively update `s` to reduce prediction error on `next_obs`:
           - compute prediction = gen_net(s)
           - compute loss_state = mean squared error
           - backpropagate to obtain grad w.r.t. s
           - update s with a small step in the direction of negative gradient
             (here implemented as gradient ascent on -loss or descent on loss)
        3. After inference, perform a parameter update:
           - compute final prediction from gen_net(s.detach())
           - compute parameter loss (MSE) and backpropagate to update model params
        4. Set `latent_state` to the inferred latent (detached).
        """
        # Convert next observation to tensor on device with batch dim
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ---- Inference (latent optimisation) ----
        # Create a trainable copy of the current latent (detach to avoid backprop through history)
        s = self.latent_state.detach().clone().requires_grad_(True)

        for _ in range(self.iters_pc):
            # Predict observation from current latent estimate
            pred = self.gen_net(s)
            # Prediction error
            err = next_obs_t - pred
            # Loss on latent (MSE)
            loss_state = (err ** 2).mean()
            # Zero any existing gradients on s and backpropagate to compute grad(s)
            if s.grad is not None:
                s.grad.zero_()
            loss_state.backward()
            # Gradient step on latent: move s to reduce loss (gradient descent)
            # We use with torch.no_grad() to update s in-place without tracking this step in autograd.
            with torch.no_grad():
                # s.grad contains d(loss_state)/d(s); step in negative gradient direction
                s -= self.lr_state * s.grad

        # ---- Learning (parameter update) ----
        # Use the final inferred latent (detached) to update generative model parameters
        self.optimizer.zero_grad()
        pred_final = self.gen_net(s.detach())
        err_final = next_obs_t - pred_final
        loss_params = (err_final ** 2).mean()

        # BUGFIX: actor-critic term so policy_net actually receives gradient.
        # Previously the only loss (final prediction error) never depended on
        # policy_net's output -- it stayed at its random init forever. We use
        # negative post-inference prediction error as a pseudo-reward
        # (successfully predicting the next observation = good), directly
        # analogous to FEAC_Friston's existing pseudo-reward pattern in this
        # same baseline suite. self.latent_state here is still the belief
        # that was current when get_action() chose `action` (it's only
        # reassigned at the very end of this method), so logprob(action) is
        # evaluated under the correct distribution.
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self._policy_dist()
        logprob = dist.log_prob(action_t).sum(-1)

        pseudo_reward = -loss_params.detach()
        v_prev = self.critic(self.latent_state.detach())
        v_next = self.critic(s.detach())
        target = pseudo_reward + self.gamma * v_next
        td_error = target - v_prev
        critic_loss = (td_error ** 2).mean()
        actor_loss = -(logprob * td_error.detach()).mean()

        total_loss = loss_params + critic_loss + actor_loss
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Update belief to inferred latent (detached to avoid retaining graph)
        self.latent_state = s.detach()

        # Diagnostics
        self.last_prediction_error = float(loss_params.item())
        self.last_grad_norm = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        self.last_pseudo_reward = float(pseudo_reward.item())

        return float(total_loss.item())

    # -------------------------
    # Diagnostics
    # -------------------------
    def get_diagnostics(self):
        """
        Return a dictionary of scalar diagnostics for logging and aggregation.
        """
        return {
            "pred_error": self.last_prediction_error,
            "grad_norm": self.last_grad_norm,
            "critic_loss": self.last_critic_loss,
            "actor_loss": self.last_actor_loss,
            "pseudo_reward": self.last_pseudo_reward,
            "blackout_active": self.last_blackout_active,
            "seed": self.last_seed,
        }


# -------------------------
# Minimal test snippets (to include in tests/test_pcrl_whittington.py)
# -------------------------
# These tests are small smoke checks to ensure the predictive-coding loop runs.
#
# import torch
# from models.aif_baselines.PCRL_Whittington import PCRL_Whittington
#
# def test_pcrl_inference_and_update_runs():
#     agent = PCRL_Whittington(obs_dim=4, action_dim=2, latent_dim=8, iters_pc=2, lr_state=0.05)
#     prev_obs = torch.zeros(4)
#     next_obs = torch.ones(4) * 0.1
#     action = torch.zeros(2)
#     loss = agent.update(prev_obs, action, next_obs)
#     assert isinstance(loss, float)
#
# def test_latent_state_changes_after_update():
#     agent = PCRL_Whittington(obs_dim=4, action_dim=2, latent_dim=8, iters_pc=2)
#     initial_latent = agent.latent_state.clone()
#     prev_obs = torch.zeros(4)
#     next_obs = torch.randn(4)
#     action = torch.zeros(2)
#     agent.update(prev_obs, action, next_obs)
#     assert not torch.allclose(initial_latent, agent.latent_state)
#
# Note: adapt tests to your test framework and device configuration.
