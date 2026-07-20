"""
models/AIF.py

Purpose
-------
Active Inference agent implementation built on an RSSM world model. This
implementation routes differentiable Expected Free Energy (EFE) gradients to
the policy only: the planner uses reparameterized sampling (rsample) from the
policy distribution while the world model parameters are temporarily frozen
by the caller during planner execution.

What is implemented
-------------------
- `ActiveInferenceAgent(nn.Module)`:
    - RSSM-based agent with fast latent z and optional slow latent s.
    - Policy network producing a Gaussian distribution (mu, std).
    - PrecisionController that maps prediction error and latent change to a
      dynamic KL weight beta.
    - EFE planner integration (planners/efe_planner.EFEPlanner) that samples
      action sequences with `rsample()` so gradients flow to policy parameters.
    - Training update that computes VFE (reconstruction + complexity + overshoot)
      and an auxiliary EFE loss that is differentiable for the policy only.
    - Temporary freezing of world_model parameters during planner execution
      to prevent planner gradients from updating the world model.
    - Lightweight gradient-check debug warnings to detect unintended gradient flow.

Design notes and caveats
-----------------------
- The planner must use `policy_dist.rsample()` internally; otherwise gradients
  will not flow to policy parameters. See planners/efe_planner.py.
- The caller (here, `update`) temporarily sets `requires_grad=False` on all
  world_model parameters while the planner runs, then restores the original flags.
- `h_prior` and `z_post` are intentionally NOT detached when passed to the
  planner so that gradients can flow from EFE -> actions -> policy parameters.
- After the update, belief state tensors are detached to prevent accidental
  gradient retention across steps.
- The implementation assumes single-batch operation for agent internals (batch=1)
  but supports batched tensors where appropriate.

Recommended experiments
-----------------------
- Verify gradient routing by toggling `use_efe` and checking policy gradients.
- Ablate `use_dynamic_beta` to compare learned vs fixed KL weighting.
- Compare ensemble-based epistemic term vs prior-logvar fallback in planner.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import warnings

from rssm_world_model import RSSMWorldModel
from precision_controller import PrecisionController
from transition_ensemble import TransitionEnsemble
from planners.efe_planner import EFEPlanner


class ActiveInferenceAgent(nn.Module):
    """
    RSSM-based Active Inference agent with EFE gradients routed to policy only.

    Responsibilities
    ----------------
    - Maintain belief state (h, z, s, step) and update it each environment step.
    - Provide `get_action(obs)` for environment interaction (non-differentiable).
    - Provide `update(prev_obs, action, next_obs)` to perform a single learning
      update step: compute VFE, optionally compute differentiable EFE for the
      policy, backpropagate, and update parameters.
    - Expose `get_diagnostics()` to return scalar diagnostics for logging.
    """

    def __init__(
        self,
        obs_dim=16,
        action_dim=6,
        latent_dim=64,
        hidden_dim=128,
        efe_horizon=3,
        lambda_epistemic=0.1,
        lambda_policy=0.1,
        lambda_slow=0.1,
        overshoot_horizon=1,
        overshoot_horizon_start=1,
        overshoot_horizon_final=3,
        overshoot_horizon_ramp_episodes=200,
        use_dynamic_beta=True,
        use_persistence=True,
        use_posterior_correction=True,
        use_efe=True,
        use_context=False,
        preferred_obs=None,
        pseudo_var_floor=1.0,
    ):
        super().__init__()

        # Store dimensions and hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.efe_horizon = efe_horizon
        self.overshoot_horizon = overshoot_horizon          # kept for compatibility
        self.overshoot_horizon_start = overshoot_horizon_start
        self.overshoot_horizon_final = overshoot_horizon_final
        self.overshoot_horizon_ramp_episodes = overshoot_horizon_ramp_episodes

        self.lambda_epistemic = lambda_epistemic
        self.lambda_policy = lambda_policy
        self.lambda_slow = lambda_slow

        self.use_dynamic_beta = use_dynamic_beta
        self.use_persistence = use_persistence
        self.use_posterior_correction = use_posterior_correction
        self.use_efe = use_efe
        self.use_context = use_context

        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Preferred observation used by planner risk term; default to zeros
        if preferred_obs is None:
            self.preferred_obs = torch.zeros(1, obs_dim, device=device)
        else:
            self.preferred_obs = torch.as_tensor(
                preferred_obs, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Ensemble and world model
        # TransitionEnsemble is constructed with latent_dim and action_dim.
        # Note: TransitionEnsemble supports lazy construction if input_dim omitted.
        self.ensemble = TransitionEnsemble(
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

        # RSSMWorldModel holds priors/posteriors, decoder, overshoot utilities.
        self.world_model = RSSMWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lambda_slow=lambda_slow,
            pseudo_var_floor=pseudo_var_floor,
            ensemble=self.ensemble,
        )

        # Policy network: input is [h, z, (s if use_context)]
        self.context_dim = self.world_model.slow_dim if use_context else 0
        pol_in_dim = self.hidden_dim + self.latent_dim + (self.context_dim if self.use_context else 0)
        self.policy_net = nn.Sequential(
            nn.Linear(pol_in_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        # Learnable log-std parameter shared across batch
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Precision controller maps (prediction error, latent delta) -> beta
        self.precision = PrecisionController()

        # EFE planner used to compute auxiliary policy loss (differentiable for policy)
        self.planner = EFEPlanner(
            horizon=efe_horizon,
            lambda_epistemic=lambda_epistemic,
            num_particles=8,            # default particles for planner; tune as needed
            action_noise=1e-3,          # small action noise for stability
        )

        # Optimizer over all agent parameters (world model, policy, precision, ensemble)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)

        # Initialize belief state on device (batch=1 by default)
        self.h, self.z, self.s, self.step = self.world_model.init_state(batch=1)
        self.h = self.h.to(device)
        self.z = self.z.to(device)
        self.s = self.s.to(device)
        self.step = self.step.to(device)

        # Buffers for overshooting (store recent actions and posterior z)
        self._past_actions = []
        self._past_post_z = []

        # BUGFIX: tracks whether get_action() has already folded in an
        # observation for the current episode. See get_action() and
        # reset_belief() for why this is needed.
        self._episode_step_count = 0

        # Overshoot schedule
        self.episode_count = 0
        self.current_overshoot_horizon = self.overshoot_horizon_start

        # Diagnostics (populated after each update)
        self.last_accuracy = 0.0
        self.last_complexity = 0.0
        self.last_beta = 0.0
        self.last_efe = 0.0
        self.last_risk = 0.0
        self.last_epistemic = 0.0
        self.last_kl_prior = 0.0
        self.last_kl_balanced = 0.0
        self.last_overshoot = 0.0
        self.last_latent_std_mean = 0.0
        self.last_grad_norm = 0.0
        self.last_slow_persist = 0.0
        self.last_mse = 0.0

        self.last_blackout_active = False
        self.last_seed = None

        # Move agent to device
        self.to(device)

    # -------------------------
    # Overshoot horizon schedule
    # -------------------------
    def _update_overshoot_horizon(self):
        """Update the current overshoot horizon based on episode count."""
        if self.episode_count >= self.overshoot_horizon_ramp_episodes:
            self.current_overshoot_horizon = self.overshoot_horizon_final
        else:
            # linear interpolation
            ratio = self.episode_count / self.overshoot_horizon_ramp_episodes
            self.current_overshoot_horizon = int(
                self.overshoot_horizon_start + ratio * (self.overshoot_horizon_final - self.overshoot_horizon_start)
            )
            # ensure at least 1
            self.current_overshoot_horizon = max(1, self.current_overshoot_horizon)

    # -------------------------
    # Belief management
    # -------------------------
    def reset_belief(self):
        """
        Reset internal belief state to zeros and clear overshoot buffers.
        Increments episode counter and updates overshoot horizon.
        """
        self.h, self.z, self.s, self.step = self.world_model.init_state(batch=1)
        self.h = self.h.to(self.device)
        self.z = self.z.to(self.device)
        self.s = self.s.to(self.device)
        self.step = self.step.to(self.device)
        self._past_actions = []
        self._past_post_z = []

        # BUGFIX: reset the episode-step counter so the first get_action() call
        # of the new episode folds in the initial observation (see get_action()).
        self._episode_step_count = 0

        # Episode tracking for overshoot schedule
        self.episode_count += 1
        self._update_overshoot_horizon()

    def set_preferred_obs(self, preferred_obs):
        """
        Update the agent's preferred-observation target after construction.

        DESIGN NOTE / FIX: preferred_obs was previously captured exactly once,
        at agent construction time, by calling env.preferred_obs() before any
        env.step() had run. For environments with a non-stationary goal (e.g.
        TransferEntropy, whose preferred_obs() correctly encodes a goal flip
        at step 250) this meant the agent's actual target was permanently
        frozen at whatever the environment's preference was at step 0 --
        the goal-flip the environment was built to test was never reflected
        in the agent's behavior at all, regardless of training duration.

        This method lets the training loop re-query env.preferred_obs() and
        push the current value in, typically once per step (see train_all.py).

        SCOPE NOTE: this is a pragmatic, "oracle-fed" fix, not a claim that
        the agent autonomously infers its goal changed. In strict active
        inference theory, prior preferences are usually an intrinsic,
        stationary property of the agent, not something dynamically supplied
        by the environment; genuinely inferring a goal shift from evidence
        would need a higher-level (e.g. hierarchical) generative model, which
        is out of scope here. What this DOES meaningfully test is narrower
        but still real: given a re-conditioned preference target, does the
        agent's planning correctly re-orient its behavior -- as opposed to
        baselines, none of which take a preferred_obs concept at all.
        """
        self.preferred_obs = torch.as_tensor(
            preferred_obs, dtype=torch.float32, device=self.device
        ).reshape(1, -1)

    # -------------------------
    # Policy helpers
    # -------------------------
    def _policy_input(self):
        """Construct the policy input vector from current belief."""
        if self.use_context:
            return torch.cat([self.h, self.z, self.s], dim=-1)
        return torch.cat([self.h, self.z], dim=-1)

    def _policy_dist(self):
        """
        Return a Normal distribution object parameterised by the policy network.
        The distribution supports `.rsample()` for reparameterized sampling.
        """
        pol_in = self._policy_input()
        mu = self.policy_net(pol_in)
        std = torch.exp(torch.clamp(self.log_std, -5, 2)).expand_as(mu)
        return Normal(mu, std)

    # -------------------------
    # Action selection
    # -------------------------
    def get_action(self, obs):
        """
        Select an action for environment interaction.

        Behaviour:
        - On the FIRST step of an episode (no belief has been conditioned on any
          observation yet), the observation is folded in via world_model.observe()
          under `torch.no_grad()` to produce an initial posterior z and slow s.
        - On every SUBSEQUENT step, this method does NOT re-observe. The belief
          (self.h, self.z, self.s) was already correctly updated by the previous
          call to update(prev_obs, action, obs) -- that call processed this exact
          `obs` as `next_obs` and produced a posterior-corrected z (see update()'s
          `z_new = z_prior + beta * (z_post - z_prior)`). Re-observing here would
          silently discard that precision-weighted belief and replace it with a
          fresh, uncorrected posterior sample.
        - The policy samples an action (non-rsamped) for environment execution.
        - Returns: (action_numpy, logprob, entropy_placeholder, value_placeholder)

        BUGFIX: previously this method called world_model.observe() unconditionally
        on every step, overwriting self.z with a freshly-sampled posterior before it
        was ever read. Since update() sets self.z to the posterior-corrected belief
        and nothing else reads self.z in between, this meant `use_posterior_correction`
        had no effect on agent behavior whatsoever -- it was computed and then
        immediately discarded on the very next get_action() call. This method now
        only re-observes on the first step of an episode, when there genuinely is no
        prior belief to reuse.
        """
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs_t = obs.to(self.device).unsqueeze(0) if obs.dim() == 1 else obs.to(self.device)

        if self._episode_step_count == 0:
            # First action of the episode: no belief has been conditioned on any
            # observation yet (self.h/self.z/self.s are still the zeros from
            # init_state/reset_belief), so fold in this initial observation.
            with torch.no_grad():
                z_post, mu_q, logvar_q, s_next, _, _, _, _ = self.world_model.observe(
                    self.h, obs_t, s_prev=self.s, update_s=True, step=self.step
                )
                self.z = z_post
                self.s = s_next
        # else: reuse self.h / self.z / self.s as already set by the previous
        # update() call -- do NOT re-observe, to preserve the posterior correction.

        self._episode_step_count += 1

        # Sample action for environment (non-differentiable sampling)
        dist = self._policy_dist()
        action = dist.sample()  # sampling for environment interaction can remain non-rsamped
        logprob = dist.log_prob(action).sum(-1)

        return (
            action.detach().squeeze(0).cpu().numpy(),
            float(logprob.item()),
            0.0,
            0.0,
        )

    # -------------------------
    # Training update
    # -------------------------
    def update(self, prev_obs, action, next_obs):
        """
        Perform a single learning update using the previous observation, action,
        and next observation.

        Key steps:
        - Compute prior via world_model.transition(h, z, a).
        - Compute posterior via world_model.observe(h_prior, next_obs).
        - Compute reconstruction (accuracy) loss and KL complexity terms.
        - Optionally compute overshoot loss from recent history.
        - Optionally compute differentiable EFE for the policy only:
            * Temporarily freeze world_model parameters (requires_grad=False).
            * Call planner which uses policy_dist.rsample() internally.
            * Restore world_model requires_grad flags after planner returns.
        - Compute slow persistence penalty.
        - Compute mean squared error.
        - Backpropagate total loss and step optimizer.
        - Update belief state and overshoot buffers (detach to avoid gradient retention).
        """
        self.optimizer.zero_grad()

        # Convert inputs to tensors on agent device
        a_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_obs_t = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Prior transition from current belief and action
        h_prior, z_prior, mu_p, logvar_p = self.world_model.transition(
            self.h, self.z, a_t, s=self.s
        )

        # Posterior from next observation (conditions on h_prior and next_obs)
        z_post, mu_q, logvar_q, s_next, mu_s_q, logvar_s_q, mu_s_p, logvar_s_p = \
            self.world_model.observe(
                h_prior, next_obs_t, s_prev=self.s, update_s=True, step=self.step
            )

        # Reconstruction / accuracy term (Gaussian decoder)
        mu_o, logvar_o = self.world_model.decode(h_prior, z_post, s_next)
        var_o = torch.exp(logvar_o)
        pred_error = next_obs_t - mu_o
        recon_term = 0.5 * ((pred_error ** 2) / (var_o + 1e-8) + logvar_o)
        accuracy_loss = recon_term.mean()
        mse = (pred_error ** 2).mean().item()
        self.last_mse = mse

        # KL terms: prior KL and balanced KL loss
        kl_prior = self.world_model.gaussian_kl(mu_q, logvar_q, mu_p, logvar_p).mean()
        kl_bal = self.world_model.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
        complexity_raw = kl_bal

        # Dynamic precision beta computed from prediction error and latent delta
        if self.use_dynamic_beta:
            pred_error_norm = torch.tanh(pred_error.pow(2).mean())
            latent_delta = z_post - z_prior
            latent_delta_norm = torch.tanh(latent_delta.pow(2).mean())
            beta = self.precision(pred_error_norm, latent_delta_norm)
            # No extra clamping – the controller already outputs in (0.01, 2.0)
        else:
            beta = torch.tensor(1.0, device=self.device)

        complexity_loss = beta * complexity_raw

        # Overshoot loss computed from recent action/posterior history (if available)
        overshoot_loss = torch.tensor(0.0, device=self.device)
        if self.current_overshoot_horizon > 0 and len(self._past_actions) > 0:
            # Build tensors of past actions and past posteriors (detached)
            actions_hist = self._past_actions + [a_t.detach()]
            z_hist = self._past_post_z + [z_post.detach()]
            actions_tm = torch.stack(actions_hist, dim=0)
            z_tm = torch.stack(z_hist, dim=0)
            overshoot_loss = self.world_model.overshoot_loss(
                h=self.h.detach(),
                z=self.z.detach(),
                actions=actions_tm,
                target_z=z_tm,
                s=self.s.detach(),
                target_s=None,
                horizon=self.current_overshoot_horizon,
            )

        # Variational free energy (VFE) composed of accuracy, complexity, and overshoot
        vfe_loss = accuracy_loss + complexity_loss + overshoot_loss

        # -------------------------
        # EFE auxiliary loss (differentiable for policy only)
        # -------------------------
        efe_loss = torch.tensor(0.0, device=self.device)
        risk_term = torch.tensor(0.0, device=self.device)
        epistemic_term = torch.tensor(0.0, device=self.device)

        if self.use_efe:
            # Save original requires_grad flags for world_model parameters
            wm_params = list(self.world_model.parameters())
            saved_requires_grad = [p.requires_grad for p in wm_params]
            # Freeze world_model parameters so planner gradients only reach policy params
            for p in wm_params:
                p.requires_grad = False

            # IMPORTANT: do NOT detach h_prior or z_post here; planner must see tensors
            # that allow gradients to flow back to policy via sampled actions (rsample).
            dist = self._policy_dist()
            efe_out = self.planner(
                world_model=self.world_model,
                ensemble=self.ensemble,
                policy_dist=dist,
                h_start=h_prior,     # NOT detached
                z_start=z_post,      # NOT detached
                preferred_obs=self.preferred_obs.to(self.device),
            )
            efe_loss, risk_term, epistemic_term = efe_out

            # Restore world_model requires_grad flags to their original values
            for p, flag in zip(wm_params, saved_requires_grad):
                p.requires_grad = flag

        # Slow persistence penalty (auxiliary regulariser on slow latent dynamics)
        slow_persist = self.world_model.slow_persistence_loss(self.s, s_next, slope=0.01)

        # Total loss includes VFE, weighted EFE for policy, and slow persistence penalty
        total_loss = vfe_loss + self.lambda_policy * efe_loss + self.lambda_slow * slow_persist

        # Backpropagate and step optimizer
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # -------------------------
        # Debug / gradient routing check (non-fatal warning)
        # -------------------------
        # Verify that EFE produced gradients for policy parameters.
        #
        # BUGFIX: removed the "world_model has non-zero grads after EFE" check.
        # It inspected world_model.parameters()[i].grad AFTER total_loss.backward(),
        # where total_loss = vfe_loss + lambda_policy*efe_loss + lambda_slow*slow_persist
        # in a single combined backward pass. vfe_loss legitimately trains the
        # world model every step (that's its job) and is computed with the
        # world model unfrozen, so wm grads are expected to be non-zero after
        # this call for reasons that have nothing to do with EFE. The actual
        # guarantee against EFE leaking into the world model is the
        # freeze/unfreeze block above (requires_grad=False during the planner
        # call) -- a structural property of autograd, not something this
        # runtime check could ever validate correctly, since it can't isolate
        # which loss term contributed what to a shared .grad. It fired on
        # every step where use_efe=True regardless of whether anything was
        # actually wrong.
        
        if self.use_efe:
            policy_params = list(self.policy_net.parameters()) + [self.log_std]
            policy_grad_norm = 0.0
            for p in policy_params:
                if p.grad is not None:
                    policy_grad_norm += float(p.grad.norm().item())
            if policy_grad_norm == 0.0:
                warnings.warn("EFE did not produce gradients for policy parameters. Check planner uses rsample() and that policy parameters are included in optimizer.")

        # -------------------------
        # Belief update with optional posterior correction
        # -------------------------
        if self.use_posterior_correction:
            # Posterior correction blends prior and posterior using dynamic beta
            z_new = z_prior + beta * (z_post - z_prior)
        else:
            z_new = z_post

        # Detach belief state to avoid retaining computation graph across steps
        self.h = h_prior.detach()
        self.z = z_new.detach()
        self.s = s_next.detach()
        self.step = (self.step + 1).to(self.device)

        # Update overshoot buffers (store detached tensors)
        self._past_actions.append(a_t.detach())
        self._past_post_z.append(z_post.detach())
        max_buf = max(1, self.current_overshoot_horizon)
        if len(self._past_actions) > max_buf:
            self._past_actions = self._past_actions[-max_buf:]
            self._past_post_z = self._past_post_z[-max_buf:]

        # -------------------------
        # Diagnostics (store scalars for logging)
        # -------------------------
        self.last_accuracy = float(accuracy_loss.item())
        self.last_complexity = float(complexity_raw.item())
        try:
            self.last_beta = float(beta.item())
        except Exception:
            # beta may be a tensor of shape [B,1]; take mean if necessary
            self.last_beta = float(beta.mean().item())
        self.last_efe = float(efe_loss.item())
        self.last_risk = float(risk_term.item())
        self.last_epistemic = float(epistemic_term.item())
        self.last_kl_prior = float(kl_prior.item())
        self.last_kl_balanced = float(kl_bal.item())
        self.last_overshoot = float(overshoot_loss.item())
        self.last_latent_std_mean = float(torch.exp(0.5 * logvar_q).mean().item())
        self.last_grad_norm = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        self.last_slow_persist = float(slow_persist.item())

        # Return VFE (useful for logging/training loops)
        return float(vfe_loss.item())

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    def get_diagnostics(self):
        """
        Return a dictionary of current diagnostics. Used by the training driver
        to aggregate per-step metrics into per-episode summaries.
        """
        latent_norm = float(torch.cat([self.h, self.z], dim=-1).norm().item())
        return {
            "accuracy": self.last_accuracy,
            "complexity": self.last_complexity,
            "beta": self.last_beta,
            "efe": self.last_efe,
            "risk": self.last_risk,
            "epistemic": self.last_epistemic,
            "kl_prior": self.last_kl_prior,
            "kl_balanced": self.last_kl_balanced,
            "overshoot_loss": self.last_overshoot,
            "latent_norm": latent_norm,
            "latent_std_mean": self.last_latent_std_mean,
            "grad_norm": self.last_grad_norm,
            "slow_persist": self.last_slow_persist,
            "blackout_active": self.last_blackout_active,
            "seed": self.last_seed,
            "mse": self.last_mse,
        }
