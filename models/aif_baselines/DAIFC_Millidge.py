"""
models/aif_baselines/DAIFC_Millidge.py

Purpose
-------
Deep Active Inference Control (DAIFC) style baseline implementation.
This lightweight baseline implements:
- A latent-state transition network (predict next latent given current latent and action)
- A likelihood network mapping latent -> observation prediction
- An amortized encoder mapping observation -> posterior latent
- A simple policy network producing actions from the latent
- A state-cost regulariser that encourages latent states to remain near zero
- Online per-step updates (no replay buffer)

Design notes
------------
- This baseline is intentionally simple to serve as a control for the AIF agent:
  it lacks an explicit world model with GRU belief, overshooting, or ensemble
  epistemics. It is useful to compare performance when using a compact
  amortized latent model and control-as-inference style state-cost shaping.
- The implementation assumes single-batch operation (batch size 1) for agent
  internals, but tensors are shaped to support batching where convenient.
- The policy uses a Gaussian parameterisation with a learnable log-std.
- The encoder output is used to correct the predicted latent after the update.

Recommended experiments
-----------------------
- Sweep `latent_dim` to compare representational capacity vs stability.
- Vary `lambda_state_cost` to measure the effect of state-cost shaping on
  behaviour and robustness to blackout/latency.
- Compare this baseline to the full AIF agent on the same environments to
  isolate the contribution of the RSSM, ensemble, and EFE planner.

Minimal tests to include
------------------------
- Shape and dtype checks for `get_action()` outputs.
- Update runs without error on synthetic transitions.
- Posterior correction moves latent_state toward encoder output.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class DAIFC_Millidge(nn.Module):
    """
    Deep Active Inference Control-style agent.

    Responsibilities
    ----------------
    - Maintain a compact latent state `latent_state`.
    - Predict next latent given current latent and action (trans_net).
    - Predict observation from latent (lik_net).
    - Encode observation to posterior latent (enc_net).
    - Produce actions from latent via a Gaussian policy (policy_net + log_std).
    - Perform online per-step updates: minimise observation prediction error
      plus a state-cost regulariser, then correct latent via encoder.
    """

    def __init__(self, obs_dim=16, action_dim=6, latent_dim=32, lambda_state_cost=0.1, gamma=0.99):
        super().__init__()
        # Dimensions and hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.lambda_state_cost = lambda_state_cost
        self.gamma = gamma

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transition network: predicts next latent from [latent, action]
        # Output is deterministic latent prediction (no stochasticity here).
        self.trans_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        )

        # Likelihood network: maps latent -> predicted observation
        self.lik_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, obs_dim),
        )

        # Encoder (amortized posterior): maps observation -> latent
        self.enc_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        )

        # Policy network: maps latent -> action mean
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        # Learnable log-std parameter shared across batch
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # BUGFIX: added critic, needed for policy_net to receive ANY gradient
        # at all. Previously the only backpropagated loss (obs_loss +
        # lambda_state_cost * state_cost) depended solely on
        # trans_net/lik_net/enc_net -- policy_net and log_std were registered
        # with the optimizer but never appeared in any loss computation, so
        # they stayed at their random initialization for the entire run.
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Optimizer over all parameters
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        # Initialize latent state on device (batch size 1)
        self.latent_state = torch.zeros(1, latent_dim, device=self.device)

        # Diagnostics for logging and analysis
        self.last_accuracy = 0.0
        self.last_state_cost = 0.0
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
    # Belief / latent management
    # -------------------------
    def reset_belief(self):
        """
        Reset the internal latent state to zeros. Useful at episode boundaries.
        """
        self.latent_state = torch.zeros(1, self.latent_dim, device=self.device)

    # -------------------------
    # Policy helpers
    # -------------------------
    def _policy_dist(self):
        """
        Return a Normal distribution parameterised by the policy network.
        The distribution supports `.sample()` for environment interaction.
        """
        mu_a = self.policy_net(self.latent_state)
        std_a = torch.exp(torch.clamp(self.log_std, -5, 2)).expand_as(mu_a)
        return Normal(mu_a, std_a)

    # -------------------------
    # Action selection
    # -------------------------
    def get_action(self, obs):
        """
        Sample an action for environment interaction.

        Args:
            obs: raw observation (array-like or tensor). This baseline assumes
                 the latent_state is already updated in `update()`.

        Returns:
            (action_numpy, logprob, entropy, value_placeholder)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Policy conditioned on current latent_state
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
        Perform a single online update step.

        Steps:
        - Convert inputs to tensors on device.
        - Predict next latent from current latent and action (prior).
        - Predict observation from predicted latent and compute observation loss.
        - Compute state cost (penalise large latent magnitudes).
        - Backpropagate combined loss and step optimizer.
        - Compute amortized posterior via encoder and blend with predicted latent
          to form the new latent_state (posterior correction).
        - Update diagnostics and return scalar loss.
        """
        # Convert inputs to tensors on device with batch dim
        prev_obs_t = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.optimizer.zero_grad()

        # Prior latent prediction: use detached latent_state to avoid backprop
        # through previous steps (online update).
        prior_in = torch.cat([self.latent_state.detach(), action_t], dim=-1)
        s_pred = self.trans_net(prior_in)

        # Predict observation from predicted latent
        o_pred = self.lik_net(s_pred)
        obs_loss = ((next_obs_t - o_pred) ** 2).mean()

        # State cost: encourage latent to remain near zero (control-as-inference style)
        state_cost = (s_pred ** 2).mean()

        # BUGFIX: actor-critic term so policy_net actually receives gradient.
        # Previously the only loss (obs_loss + lambda_state_cost*state_cost)
        # never depended on policy_net's output at all -- it stayed at its
        # random init forever. We use negative free energy
        # (-(obs_loss + lambda_state_cost*state_cost)) as a pseudo-reward,
        # directly analogous to FEAC_Friston's existing pseudo-reward pattern
        # in this same baseline suite. self.latent_state here is still the
        # belief that was current when get_action() chose `action` (it's only
        # reassigned further below), so logprob(action) is evaluated under
        # the correct distribution.
        dist = self._policy_dist()
        logprob = dist.log_prob(action_t).sum(-1)

        pseudo_reward = -(obs_loss.detach() + self.lambda_state_cost * state_cost.detach())
        v_prev = self.critic(self.latent_state.detach())
        v_next = self.critic(s_pred.detach())
        target = pseudo_reward + self.gamma * v_next
        td_error = target - v_prev
        critic_loss = (td_error ** 2).mean()
        actor_loss = -(logprob * td_error.detach()).mean()

        # Total loss: observation reconstruction + weighted state cost + actor-critic term
        loss = obs_loss + self.lambda_state_cost * state_cost + critic_loss + actor_loss
        loss.backward()
        # Clip gradients for stability and compute grad norm for diagnostics
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Posterior correction: blend predicted latent and encoder output
        # Both are detached to avoid retaining computation graph across steps.
        s_enc = self.enc_net(next_obs_t)
        # Simple averaging blend; alternative blending strategies can be used.
        self.latent_state = 0.5 * s_pred.detach() + 0.5 * s_enc.detach()

        # Diagnostics
        self.last_accuracy = float(obs_loss.item())
        self.last_state_cost = float(state_cost.item())
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
            "state_cost": self.last_state_cost,
            "grad_norm": self.last_grad_norm,
            "critic_loss": self.last_critic_loss,
            "actor_loss": self.last_actor_loss,
            "pseudo_reward": self.last_pseudo_reward,
            "blackout_active": self.last_blackout_active,
            "seed": self.last_seed,
        }


# -------------------------
# Minimal test snippets (to include in tests/test_daifc_millidge.py)
# -------------------------
# These tests are small smoke checks to ensure the baseline runs and updates.
#
# import torch
# from models.aif_baselines.DAIFC_Millidge import DAIFC_Millidge
#
# def test_daifc_update_runs():
#     agent = DAIFC_Millidge(obs_dim=4, action_dim=2, latent_dim=8)
#     prev_obs = torch.zeros(4)
#     next_obs = torch.ones(4) * 0.1
#     action = torch.zeros(2)
#     loss = agent.update(prev_obs, action, next_obs)
#     assert isinstance(loss, float)
#
# def test_posterior_correction_moves_latent():
#     agent = DAIFC_Millidge(obs_dim=4, action_dim=2, latent_dim=8)
#     initial_latent = agent.latent_state.clone()
#     prev_obs = torch.zeros(4)
#     next_obs = torch.randn(4)
#     action = torch.zeros(2)
#     agent.update(prev_obs, action, next_obs)
#     # After update, latent_state should have changed from initial value
#     assert not torch.allclose(initial_latent, agent.latent_state)
#
# Note: adapt tests to your test framework and device configuration.
