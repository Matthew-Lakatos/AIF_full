"""
models/aif_baselines/FEAC_Friston.py

Purpose
-------
Free Energy Actor-Critic (FEAC) baseline inspired by Friston-style formulations.
This lightweight baseline combines a simple encoder-based latent representation
with an actor-critic architecture where the critic approximates a free-energy
like value and the actor is trained with a TD-style policy gradient. A
pseudo-reward derived from encoder change (surprise) is used to shape learning.

What is implemented
-------------------
- Encoder mapping observations -> latent state.
- Actor network producing action means; learnable shared log-std for Gaussian policy.
- Critic network predicting scalar value from latent.
- One-step TD update where the pseudo-reward is the negative squared change in
  encoder outputs (interpreted as surprise).
- Online per-step updates and simple belief management (latent_state).
- Diagnostics for logging and plotting.

Design notes and caveats
-----------------------
- The pseudo-reward is intentionally simple: negative squared change in encoder
  outputs. This encourages the agent to prefer states that reduce surprise.
- The critic is trained with a one-step TD target; the actor uses the TD error
  as an advantage signal for a policy gradient update.
- The encoder output used as the latent is detached when used as the next belief
  to avoid retaining computation graphs across steps.
- This baseline is intentionally compact and suitable as a control for the full
  RSSM + planner AIF agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class FEAC_Friston(nn.Module):
    """
    Free Energy Actor-Critic style baseline.

    Responsibilities
    ----------------
    - Encode observations into a compact latent representation.
    - Provide a Gaussian policy q(a | s) parameterised by an actor network and
      a shared learnable log-std.
    - Provide a critic that predicts a scalar value from the latent.
    - Perform a one-step TD update using a pseudo-reward derived from encoder
      change (negative squared difference) and update actor + critic jointly.
    - Maintain a current latent_state for action selection between updates.
    """

    def __init__(self, obs_dim=16, action_dim=6, latent_dim=32, gamma=0.99):
        super().__init__()
        # Dimensions and hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder: maps observation -> latent representation
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        )

        # Actor: maps latent -> action mean
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        # Shared learnable log-std for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic: maps latent -> scalar value (approximates free energy / value)
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Optimizer over all parameters
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        # Current belief (latent state) initialised to zeros (batch size 1)
        self.latent_state = torch.zeros(1, latent_dim, device=self.device)

        # Diagnostics for logging and analysis
        self.last_critic_loss = 0.0
        self.last_actor_loss = 0.0
        self.last_td_error = 0.0
        self.last_grad_norm = 0.0
        self.last_blackout_active = False
        self.last_seed = None

        # Move module to device
        self.to(self.device)

    # -------------------------
    # Helpers
    # -------------------------
    def reset_belief(self):
        """Reset the internal latent state to zeros (use at episode boundaries)."""
        self.latent_state = torch.zeros(1, self.latent_dim, device=self.device)

    def _policy_dist(self, latent):
        """
        Return a Normal distribution parameterised by the actor network for a given latent.

        Args:
            latent: tensor [B, latent_dim]

        Returns:
            torch.distributions.Normal with mean [B, action_dim] and std [B, action_dim]
        """
        mu_a = self.actor(latent)
        std_a = torch.exp(torch.clamp(self.log_std, -5, 2)).expand_as(mu_a)
        return Normal(mu_a, std_a)

    # -------------------------
    # Action selection
    # -------------------------
    def get_action(self, obs):
        """
        Encode observation to latent and sample an action from the policy.

        Args:
            obs: raw observation (array-like or tensor) of shape (obs_dim,)

        Returns:
            (action_numpy, logprob, entropy, value)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Encode current observation to get latent state (detached to avoid graph retention)
        self.latent_state = self.encoder(obs_t).detach()
        dist = self._policy_dist(self.latent_state)
        a = dist.sample()
        logprob = dist.log_prob(a).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(self.latent_state)
        return (
            a.detach().squeeze(0).cpu().numpy(),
            float(logprob.item()),
            float(entropy.item()),
            float(value.item()),
        )

    # -------------------------
    # Training update
    # -------------------------
    def update(self, prev_obs, action, next_obs):
        """
        One-step TD update with pseudo-reward based on encoder change (surprise).

        Steps:
        - Encode prev_obs and next_obs to obtain s_prev and s_next.
        - Compute pseudo-reward = -mean((s_next - s_prev)^2) (negative surprise).
        - Compute TD target: pseudo_rew + gamma * V(s_next).
        - Compute TD error and critic loss (MSE).
        - Compute actor loss using policy gradient with TD error as advantage.
        - Backpropagate combined loss and step optimizer.
        - Update latent_state to s_next (detached).
        """
        # Convert inputs to tensors on device with batch dim
        prev_obs_t = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.optimizer.zero_grad()

        # Encode states
        s_prev = self.encoder(prev_obs_t)            # [1, latent_dim], requires_grad=True
        s_next = self.encoder(next_obs_t).detach()   # target latent (no grad through target)

        # Critic values
        v_prev = self.critic(s_prev)  # [1, 1]
        v_next = self.critic(s_next)  # [1, 1]

        # Pseudo-reward: negative squared change in encoding (interpreted as surprise)
        # Detach to avoid backpropagating through the target encoder change
        pseudo_rew = -((s_next - s_prev) ** 2).mean().detach()  # scalar tensor

        # TD target and error
        target = pseudo_rew + self.gamma * v_next
        td_error = target - v_prev
        critic_loss = (td_error ** 2).mean()

        # Actor loss: policy gradient using TD error as advantage (detach advantage)
        dist = self._policy_dist(s_prev)
        a_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        logprob = dist.log_prob(a_t).sum(-1)
        # Use negative sign because we minimise loss; detach td_error to avoid critic->actor gradients
        actor_loss = -(logprob * td_error.detach()).mean()

        # Combined loss and optimization
        loss = critic_loss + actor_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Update belief to next latent (detached to avoid retaining graph)
        self.latent_state = s_next.detach()

        # Diagnostics
        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        self.last_td_error = float(td_error.mean().item())
        self.last_grad_norm = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        return float(loss.item())

    # -------------------------
    # Diagnostics
    # -------------------------
    def get_diagnostics(self):
        """
        Return a dictionary of scalar diagnostics for logging and aggregation.
        """
        return {
            "critic_loss": self.last_critic_loss,
            "actor_loss": self.last_actor_loss,
            "td_error": self.last_td_error,
            "grad_norm": self.last_grad_norm,
            "blackout_active": self.last_blackout_active,
            "seed": self.last_seed,
        }


# -------------------------
# Minimal test snippets (to include in tests/test_feac_friston.py)
# -------------------------
# These tests are small smoke checks to ensure the baseline runs and updates.
#
# import torch
# from models.aif_baselines.FEAC_Friston import FEAC_Friston
#
# def test_feac_update_runs():
#     agent = FEAC_Friston(obs_dim=4, action_dim=2, latent_dim=8)
#     prev_obs = torch.zeros(4)
#     next_obs = torch.ones(4) * 0.1
#     action = torch.zeros(2)
#     loss = agent.update(prev_obs, action, next_obs)
#     assert isinstance(loss, float)
#
# def test_latent_state_updates_after_update():
#     agent = FEAC_Friston(obs_dim=4, action_dim=2, latent_dim=8)
#     initial_latent = agent.latent_state.clone()
#     prev_obs = torch.zeros(4)
#     next_obs = torch.randn(4)
#     action = torch.zeros(2)
#     agent.update(prev_obs, action, next_obs)
#     assert not torch.allclose(initial_latent, agent.latent_state)
#
# Note: adapt tests to your test framework and device configuration.

