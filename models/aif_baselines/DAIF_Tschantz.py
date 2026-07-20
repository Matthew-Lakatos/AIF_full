"""
models/aif_baselines/DAIF_Tschantz.py

Purpose
-------
Deep Active Inference baseline inspired by Tschantz-style formulations.
Implements a compact latent-state generative model with:
- Transition prior p(s_t | s_{t-1}, a_{t-1}) (Gaussian)
- Likelihood p(o_t | s_t) (Gaussian)
- Amortized posterior q(s_t | o_t) (Gaussian)
- Policy q(a_t | s_t) (Gaussian)
Training objective is a negative ELBO: reconstruction (NLL) + KL(q||p).

What is implemented
-------------------
- Parameterised networks for transition, likelihood, encoder, and policy.
- Reparameterized sampling from the posterior for gradient-based optimisation.
- Closed-form Gaussian KL and Gaussian NLL utilities.
- Online per-step update that:
    1. Computes prior stats from previous latent and action.
    2. Computes posterior stats from current observation.
    3. Samples latent from posterior (reparameterized).
    4. Computes likelihood NLL and KL, forms loss, backpropagates, and steps optimizer.
    5. Updates belief to posterior mean (detached).
- Simple Gaussian policy for action selection.
- Diagnostics for logging and plotting.

Design notes and caveats
-----------------------
- This baseline is intentionally compact and single-step (online). It does not
  implement a recurrent belief (no GRU) or overshooting/ensemble mechanisms.
- The implementation assumes batch size 1 for internal latent_state, but the
  networks operate on batched tensors and can be adapted to larger batches.
- Numerical stability: small epsilons are added in variance denominators and
  exponentials to avoid NaNs.
- The KL and NLL utilities return scalar means across the batch.

Recommended experiments
-----------------------
- Sweep `latent_dim` and `lambda_kl` to study representation vs regularisation.
- Compare to RSSM-based AIF agent to isolate the contribution of recurrent
  belief and planning.
- Test on blackout and latency environments to evaluate robustness differences.

Minimal tests to include
------------------------
- Shape and dtype checks for `get_action()` outputs.
- Update runs without error on synthetic transitions.
- After update, `latent_state` should change (posterior mean applied).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class DAIF_Tschantz(nn.Module):
    """
    Deep Active Inference (Tschantz-style) baseline.

    Responsibilities
    ----------------
    - Maintain a current latent_state (belief) representing q(s_t).
    - Provide `get_action(obs)` that samples from policy q(a_t | s_t).
    - Provide `update(prev_obs, action, next_obs)` that performs one learning step:
        * compute prior p(s_t | s_{t-1}, a_{t-1})
        * compute posterior q(s_t | o_t)
        * sample s_t via reparameterization
        * compute likelihood p(o_t | s_t) and KL(q||p)
        * backpropagate loss = NLL + lambda_kl * KL
        * update latent_state to posterior mean (detached)
    """

    def __init__(self, obs_dim=16, action_dim=6, latent_dim=32, lambda_kl=1.0, gamma=0.99):
        super().__init__()
        # Dimensions and hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.lambda_kl = lambda_kl
        self.gamma = gamma

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transition prior network: outputs concatenated [mu_p, logvar_p]
        self.trans_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 2 * latent_dim),  # mean, logvar
        )

        # Likelihood network: outputs concatenated [mu_o, logvar_o]
        self.lik_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 2 * obs_dim),
        )

        # Encoder (amortized posterior): outputs concatenated [mu_q, logvar_q]
        self.enc_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 2 * latent_dim),
        )

        # Policy network: maps latent -> action mean
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        # Learnable log-std parameter for policy
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # BUGFIX: added critic, needed for policy_net to receive ANY gradient
        # at all. Previously the only backpropagated loss (NLL + lambda_kl*KL)
        # depended solely on trans_net/lik_net/enc_net -- policy_net and
        # log_std were registered with the optimizer but never appeared in any
        # loss computation, so they stayed at their random initialization for
        # the entire run (see update() for the actor-critic term this feeds).
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        # Current belief (latent state) initialised to zeros (batch size 1)
        self.latent_state = torch.zeros(1, latent_dim, device=self.device)

        # Diagnostics
        self.last_accuracy = 0.0
        self.last_complexity = 0.0
        self.last_kl = 0.0
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
    # Utilities
    # -------------------------
    def reset_belief(self):
        """Reset latent state to zeros (useful at episode boundaries)."""
        self.latent_state = torch.zeros(1, self.latent_dim, device=self.device)

    def _split_gauss(self, out, dim=-1):
        """
        Split a concatenated Gaussian parameter output into (mu, logvar).

        Args:
            out: tensor with last dim = 2 * D
        Returns:
            mu, logvar each with last dim = D
        """
        mu, logvar = torch.chunk(out, 2, dim=dim)
        return mu, logvar

    def _kl_gauss(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Compute mean KL(q||p) for diagonal Gaussians.

        Returns:
            scalar mean KL across batch
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            logvar_p - logvar_q
            + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
            - 1.0
        )
        # Sum over latent dims, then mean over batch
        return kl.sum(-1).mean()

    def _nll_gauss(self, x, mu, logvar):
        """
        Negative log-likelihood (Gaussian) averaged over batch.

        Args:
            x: target observations [B, obs_dim]
            mu: predicted mean [B, obs_dim]
            logvar: predicted log-variance [B, obs_dim]

        Returns:
            scalar mean NLL across batch
        """
        var = torch.exp(logvar) + 1e-8
        nll = 0.5 * (((x - mu) ** 2) / var + logvar)
        return nll.sum(-1).mean()

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
        Sample an action from the policy given the current latent_state.

        Args:
            obs: raw observation (unused for belief update here; belief updated in update()).

        Returns:
            (action_numpy, logprob, entropy, value_placeholder)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Policy conditioned on current latent_state
        dist = self._policy_dist()
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        # No critic value in this baseline
        return (
            action.detach().squeeze(0).cpu().numpy(),
            float(logprob.item()),
            float(entropy.item()),
            0.0,
        )

    # -------------------------
    # Training update
    # -------------------------
    def update(self, prev_obs, action, next_obs):
        """
        One-step DAIF update:
        - Compute prior p(s_t | s_{t-1}, a_{t-1}) via trans_net
        - Compute posterior q(s_t | o_t) via enc_net
        - Sample s_t from q (reparameterized)
        - Compute likelihood p(o_t | s_t) and KL(q||p)
        - Backpropagate loss = NLL + lambda_kl * KL and step optimizer
        - Update belief to posterior mean (detached)
        """
        # Convert inputs to tensors on device with batch dim
        prev_obs_t = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.optimizer.zero_grad()

        # Prior p(s_t | s_{t-1}, a_{t-1}) -> outputs [mu_p, logvar_p]
        prior_in = torch.cat([self.latent_state.detach(), action_t], dim=-1)
        prior_out = self.trans_net(prior_in)
        mu_p, logvar_p = self._split_gauss(prior_out, dim=-1)

        # Posterior q(s_t | o_t) -> outputs [mu_q, logvar_q]
        enc_out = self.enc_net(next_obs_t)
        mu_q, logvar_q = self._split_gauss(enc_out, dim=-1)

        # Reparameterized sample from posterior q(s_t)
        std_q = torch.exp(0.5 * logvar_q)
        eps = torch.randn_like(std_q)
        s_t = mu_q + eps * std_q

        # Likelihood p(o_t | s_t) -> outputs [mu_o, logvar_o]
        lik_out = self.lik_net(s_t)
        mu_o, logvar_o = self._split_gauss(lik_out, dim=-1)
        nll = self._nll_gauss(next_obs_t, mu_o, logvar_o)

        # KL between posterior and prior
        kl = self._kl_gauss(mu_q, logvar_q, mu_p, logvar_p)

        # BUGFIX: actor-critic term so policy_net actually receives gradient.
        # Previously the only loss (NLL + lambda_kl*KL) never depended on
        # policy_net's output at all -- it stayed at its random init forever.
        # We use negative free energy (-(NLL + lambda_kl*KL)) as a pseudo-reward
        # -- consistent with the active-inference framing this whole baseline
        # already embodies (lower free energy = better), directly analogous to
        # FEAC_Friston's existing "negative surprise" pseudo-reward pattern in
        # this same baseline suite. self.latent_state here is still the belief
        # that was current when get_action() chose `action` (update() only
        # reassigns it at the very end), so logprob(action) is evaluated under
        # the correct distribution.
        dist = self._policy_dist()
        logprob = dist.log_prob(action_t).sum(-1)

        pseudo_reward = -(nll.detach() + self.lambda_kl * kl.detach())
        v_prev = self.critic(self.latent_state.detach())
        v_next = self.critic(mu_q.detach())
        target = pseudo_reward + self.gamma * v_next
        td_error = target - v_prev
        critic_loss = (td_error ** 2).mean()
        actor_loss = -(logprob * td_error.detach()).mean()

        # Total loss: negative ELBO (NLL + lambda * KL) + actor-critic term
        loss = nll + self.lambda_kl * kl + critic_loss + actor_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Update belief to posterior mean (detached to avoid retaining graph)
        self.latent_state = mu_q.detach()

        # Diagnostics
        self.last_accuracy = float(nll.item())
        self.last_complexity = float(kl.item())
        self.last_kl = float(kl.item())
        self.last_grad_norm = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        self.last_pseudo_reward = float(pseudo_reward.item())

        return float(loss.item())

    # -------------------------
    # Diagnostics
    # -------------------------
    def get_diagnostics(self):
        """
        Return a dictionary of scalar diagnostics for logging and aggregation.
        """
        return {
            "accuracy": self.last_accuracy,
            "complexity": self.last_complexity,
            "kl": self.last_kl,
            "grad_norm": self.last_grad_norm,
            "critic_loss": self.last_critic_loss,
            "actor_loss": self.last_actor_loss,
            "pseudo_reward": self.last_pseudo_reward,
            "blackout_active": self.last_blackout_active,
            "seed": self.last_seed,
        }


# -------------------------
# Minimal test snippets (to include in tests/test_daif_tschantz.py)
# -------------------------
# These tests are small smoke checks to ensure the baseline runs and updates.
#
# import torch
# from models.aif_baselines.DAIF_Tschantz import DAIF_Tschantz
#
# def test_daif_tschantz_update_runs():
#     agent = DAIF_Tschantz(obs_dim=4, action_dim=2, latent_dim=8)
#     prev_obs = torch.zeros(4)
#     next_obs = torch.ones(4) * 0.1
#     action = torch.zeros(2)
#     loss = agent.update(prev_obs, action, next_obs)
#     assert isinstance(loss, float)
#
# def test_latent_state_updates_to_posterior_mean():
#     agent = DAIF_Tschantz(obs_dim=4, action_dim=2, latent_dim=8)
#     initial_latent = agent.latent_state.clone()
#     prev_obs = torch.zeros(4)
#     next_obs = torch.randn(4)
#     action = torch.zeros(2)
#     agent.update(prev_obs, action, next_obs)
#     assert not torch.allclose(initial_latent, agent.latent_state)
#
# Note: adapt tests to your test framework and device configuration.
