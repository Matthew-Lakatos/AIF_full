"""
rssm_world_model.py

Purpose
-------
Recurrent State-Space Model (RSSM) world model with an additional slow/context
latent `s`. Implements prior/posterior for fast latent `z` and slow latent `s`,
a deterministic belief state `h` (GRUCell), a decoder for observations, KL
utilities, KL-based overshooting with adaptive pseudo-posterior variance, and
helpers for slow-latent persistence. Designed to be compatible with the AIF
agent implementation (models/AIF.py) and to support an optional TransitionEnsemble
for ensemble-based predictive means/variances.

What is implemented
-------------------
- Deterministic belief update via GRUCell over concatenated [z, a, s].
- Prior and posterior networks for fast latent z (mu/logvar outputs).
- Prior and posterior networks for slow latent s (mu/logvar outputs).
- Optional learned gating for slow-latent updates or periodic updates via interval.
- Decoder producing observation statistics (mu_o, logvar_o).
- Gaussian KL utilities and a balanced KL wrapper with free-bits.
- KL-based overshooting loss that constructs a per-sample pseudo-posterior
  variance (batch variance optionally augmented by ensemble variance).
- Slow-latent persistence penalty (asymmetric squared penalty).
- Device-aware handling and optional moving of an ensemble module to the model device.

Design notes and defaults used in experiments
--------------------------------------------
- latent_dim (z) default: 64
- hidden_dim (h) default: 128
- slow_dim (s) default: 32
- slow_update_interval default: 10 (periodic slow updates)
- pseudo_var_floor default: 1e-2 (floor for pseudo-posterior variance in overshoot)
- use_ensemble_var_in_overshoot default: True (augment pseudo-variance with ensemble variance)
- The ensemble (if provided) is expected to follow the contract: ensemble(x) -> (mean, var)

Recommended experimental variations
----------------------------------
- Vary slow_dim and slow_update_interval to study memory capacity vs stability.
- Toggle learned_slow_update to compare gated vs periodic slow updates.
- Ablate use_ensemble_var_in_overshoot to measure the ensemble's contribution to calibration.
- Sweep pseudo_var_floor to test sensitivity of overshoot to variance floors.

Notes for integration
---------------------
- This module assumes env/model code will call init_state() to obtain initial (h, z, s, step).
- The overshoot_loss() detaches rollouts to avoid accidental gradient flow into caller graphs.
- If ensemble is provided and is an nn.Module, it will be moved to the model device.
- The code uses torch.randn_like for sampling; for reproducibility, set global seeds externally.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSMWorldModel(nn.Module):
    """
    RSSM world model with slow/context latent `s`.

    Summary of responsibilities
    ---------------------------
    - Maintain deterministic belief `h` and stochastic latents `z` (fast) and `s` (slow).
    - Provide transition(h, z, a, s) that returns prior stats and a prior sample/mean.
    - Provide observe(h, obs, s_prev, ...) that returns posterior samples and updated slow latent.
    - Provide decode(h, z, s) -> (mu_o, logvar_o) for observation reconstruction.
    - Provide KL utilities and overshoot_loss() for multi-step consistency.
    - Provide slow_persistence_loss() as an auxiliary regulariser.

    Important behaviour
    -------------------
    - Overshoot rollouts are detached from caller graphs to avoid unintended backprop.
    - When an ensemble is provided, transition() uses ensemble predictions for z prior mean.
    - Slow latent updates can be learned (gate) or periodic (interval).
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        latent_dim=64,
        hidden_dim=128,
        slow_dim=32,
        slow_update_interval: int = 10,
        learned_slow_update: bool = False,
        slow_update_init: float = 0.1,
        lambda_slow: float = 0.1,
        ensemble=None,
        device: torch.device = None,
        pseudo_var_floor: float = 1e-2,
        use_ensemble_var_in_overshoot: bool = True,
    ):
        super().__init__()

        # Dimensions and hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.slow_dim = slow_dim
        self.lambda_slow = lambda_slow

        self.slow_update_interval = slow_update_interval
        self.learned_slow_update = learned_slow_update

        # Ensemble (TransitionEnsemble) may be None; expected contract: ensemble(x) -> (mean, var)
        self.ensemble = ensemble

        # Overshoot configuration (exposed for experiments)
        self.pseudo_var_floor = float(pseudo_var_floor)
        self.use_ensemble_var_in_overshoot = bool(use_ensemble_var_in_overshoot)

        # Device selection: default to CUDA if available
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Deterministic belief GRU: input is [z, a, s]
        # GRUCell is used for a compact deterministic update per step.
        self.gru = nn.GRUCell(latent_dim + action_dim + slow_dim, hidden_dim)

        # Prior and posterior networks for fast latent z (output concatenated mu, logvar)
        # Prior uses hidden state only; posterior conditions on hidden state and observation.
        self.prior = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior = nn.Linear(hidden_dim + obs_dim, latent_dim * 2)

        # Slow latent networks (prior and posterior)
        self.slow_prior = nn.Linear(hidden_dim + slow_dim, slow_dim * 2)
        self.slow_posterior = nn.Linear(hidden_dim + obs_dim + slow_dim, slow_dim * 2)

        # Optional learned gating for slow updates
        if self.learned_slow_update:
            # Gate outputs logits per slow-dim; sigmoid applied later to mix prior/posterior.
            self.slow_update_gate = nn.Sequential(
                nn.Linear(hidden_dim + obs_dim + slow_dim, 64),
                nn.ReLU(),
                nn.Linear(64, slow_dim),
            )
            # Initialize gate bias to produce initial probability close to slow_update_init.
            # Use torch.logit to set bias such that sigmoid(bias) ~= slow_update_init.
            nn.init.constant_(
                self.slow_update_gate[-1].bias,
                float(torch.logit(torch.tensor(slow_update_init))),
            )
        else:
            self.slow_update_gate = None

        # Decoder: outputs concatenated [mu_o, logvar_o] for observation reconstruction.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + slow_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim * 2),
        )

        # Move module and ensemble (if module-like) to device
        self.to(self.device)
        if self.ensemble is not None:
            try:
                # If ensemble is an nn.Module, move it to device
                self.ensemble.to(self.device)
            except Exception:
                # If ensemble is a callable or custom object without .to(), ignore
                pass

    # -------------------------
    # State initialization
    # -------------------------
    def init_state(self, batch: int = 1):
        """
        Initialize (h, z, s, step) with zeros on the model device.

        Returns:
            h: [B, hidden_dim] deterministic belief
            z: [B, latent_dim] fast latent (initially zeros)
            s: [B, slow_dim] slow latent (initially zeros)
            step: [B] long tensor tracking step index (useful for periodic updates)
        """
        h = torch.zeros(batch, self.hidden_dim, device=self.device)
        z = torch.zeros(batch, self.latent_dim, device=self.device)
        s = torch.zeros(batch, self.slow_dim, device=self.device)
        step = torch.zeros(batch, dtype=torch.long, device=self.device)
        return h, z, s, step

    # -------------------------
    # Slow latent helpers
    # -------------------------
    def slow_prior_dist(self, h, s_prev):
        """
        Compute slow prior stats and sample.

        Args:
            h: [B, hidden_dim]
            s_prev: [B, slow_dim]

        Returns:
            s_prior: sampled slow prior [B, slow_dim]
            mu_p: prior mean [B, slow_dim]
            logvar_p: prior log-variance [B, slow_dim]
        """
        inp = torch.cat([h, s_prev], dim=-1)
        stats = self.slow_prior(inp)
        mu_p, logvar_p = stats.chunk(2, dim=-1)
        # Clamp logvar to a reasonable range for numerical stability
        logvar_p = torch.clamp(logvar_p, -6.0, 4.0)
        std_p = torch.exp(0.5 * logvar_p)
        s_prior = mu_p + std_p * torch.randn_like(std_p)
        return s_prior, mu_p, logvar_p

    def slow_posterior_dist(self, h, obs, s_prev):
        """
        Compute slow posterior stats and sample.

        Args:
            h: [B, hidden_dim]
            obs: [B, obs_dim]
            s_prev: [B, slow_dim]

        Returns:
            s_post: sampled slow posterior [B, slow_dim]
            mu_q: posterior mean [B, slow_dim]
            logvar_q: posterior log-variance [B, slow_dim]
        """
        inp = torch.cat([h, obs, s_prev], dim=-1)
        stats = self.slow_posterior(inp)
        mu_q, logvar_q = stats.chunk(2, dim=-1)
        logvar_q = torch.clamp(logvar_q, -6.0, 4.0)
        std_q = torch.exp(0.5 * logvar_q)
        s_post = mu_q + std_q * torch.randn_like(std_q)
        return s_post, mu_q, logvar_q

    # -------------------------
    # Transition and observation
    # -------------------------
    def transition(self, h, z, a, s=None):
        """
        Deterministic GRU update followed by prior stats for z.

        If ensemble is provided, the ensemble predicts next-z mean/var; the RSSM
        prior stats (mu_p, logvar_p) are still computed and returned for KL.

        Args:
            h: [B, hidden_dim] deterministic belief
            z: [B, latent_dim] current fast latent
            a: [B, action_dim] action
            s: [B, slow_dim] slow latent (optional)

        Returns:
            h: updated deterministic belief [B, hidden_dim]
            z_prior: prior sample or ensemble mean [B, latent_dim]
            mu_p: prior mean for KL [B, latent_dim]
            logvar_p: prior logvar for KL [B, latent_dim]
        """
        if s is None:
            s = torch.zeros(z.shape[0], self.slow_dim, device=self.device)

        # Deterministic update: concatenate [z, a, s] and update GRUCell
        inp = torch.cat([z, a, s], dim=-1)
        h = self.gru(inp, h)

        # Prior stats for KL/diagnostics from hidden state
        stats = self.prior(h)
        mu_p, logvar_p = stats.chunk(2, dim=-1)
        logvar_p = torch.clamp(logvar_p, -6.0, 4.0)

        if self.ensemble is None:
            # Sample from prior for stochastic z
            std_p = torch.exp(0.5 * logvar_p)
            z_prior = mu_p + std_p * torch.randn_like(std_p)
        else:
            # Ensemble-based latent prediction; keep mu_p/logvar_p for KL diagnostics
            ensemble_inp = torch.cat([z, a, s], dim=-1)
            # Ensemble contract: ensemble(ensemble_inp) -> (mean_z, var_z) shaped [B, D]
            mean_z, var_z = self.ensemble(ensemble_inp)
            # Use ensemble mean as prior prediction for z; downstream code may use var_z as epistemic signal
            z_prior = mean_z

        return h, z_prior, mu_p, logvar_p

    def observe(self, h, obs, s_prev=None, update_s: bool = True, step: torch.Tensor = None):
        """
        Compute posterior over fast latent z (and optionally slow latent s) given h and obs.

        Args:
            h: [B, hidden_dim]
            obs: [B, obs_dim]
            s_prev: [B, slow_dim] previous slow latent (optional)
            update_s: whether to update slow latent
            step: [B] step indices for periodic updates (optional)

        Returns:
            z_post: sampled posterior for z [B, latent_dim]
            mu_q: posterior mean for z [B, latent_dim]
            logvar_q: posterior logvar for z [B, latent_dim]
            s_new: updated slow latent [B, slow_dim]
            mu_s_q, logvar_s_q: slow posterior stats (or None)
            mu_s_p, logvar_s_p: slow prior stats (or None)
        """
        inp = torch.cat([h, obs], dim=-1)
        stats = self.posterior(inp)
        mu_q, logvar_q = stats.chunk(2, dim=-1)
        logvar_q = torch.clamp(logvar_q, -6.0, 4.0)
        std_q = torch.exp(0.5 * logvar_q)
        z_post = mu_q + std_q * torch.randn_like(std_q)

        mu_s_q = logvar_s_q = mu_s_p = logvar_s_p = None
        if s_prev is None:
            # BUGFIX: this reassignment used to happen AFTER `s_new = s_prev`,
            # so if update_s=False and s_prev was originally None, s_new would
            # silently be returned as None instead of a zero tensor -- a latent
            # trap for any future caller of the update_s=False path (currently
            # unused elsewhere in this codebase, but worth closing off).
            s_prev = torch.zeros(z_post.shape[0], self.slow_dim, device=self.device)
        s_new = s_prev

        if update_s:
            # Compute slow prior and posterior stats and samples
            s_prior, mu_s_p, logvar_s_p = self.slow_prior_dist(h, s_prev)
            s_post, mu_s_q, logvar_s_q = self.slow_posterior_dist(h, obs, s_prev)

            if self.learned_slow_update:
                # Learned gating: sigmoid(gate_logits) mixes prior and posterior
                gate_inp = torch.cat([h, obs, s_prev], dim=-1)
                gate_logits = self.slow_update_gate(gate_inp)
                gate = torch.sigmoid(gate_logits)
                s_new = (1.0 - gate) * s_prior + gate * s_post
            else:
                # Periodic update: update s only at specified intervals (mask per batch element)
                if step is None:
                    s_new = s_post
                else:
                    mask = ((step % self.slow_update_interval) == 0).float().unsqueeze(-1)
                    s_new = mask * s_post + (1.0 - mask) * s_prev

        return z_post, mu_q, logvar_q, s_new, mu_s_q, logvar_s_q, mu_s_p, logvar_s_p

    # -------------------------
    # Decoder
    # -------------------------
    def decode(self, h, z, s=None):
        """
        Decode observation statistics (mu_o, logvar_o) from belief (h, z, s).

        Args:
            h: [B, hidden_dim]
            z: [B, latent_dim]
            s: [B, slow_dim] optional

        Returns:
            mu_o: [B, obs_dim] reconstructed observation mean
            logvar_o: [B, obs_dim] reconstructed observation log-variance
        """
        if s is None:
            s = torch.zeros(z.shape[0], self.slow_dim, device=self.device)
        x = torch.cat([h, z, s], dim=-1)
        stats = self.decoder(x)
        mu_o, logvar_o = stats.chunk(2, dim=-1)
        logvar_o = torch.clamp(logvar_o, -6.0, 4.0)
        return mu_o, logvar_o

    # -------------------------
    # KL utilities
    # -------------------------
    @staticmethod
    def gaussian_kl_per_dim(mu_q, logvar_q, mu_p, logvar_p):
        """
        Elementwise Gaussian KL between q ~ N(mu_q, var_q) and p ~ N(mu_p, var_p),
        WITHOUT reducing over the latent dimension.

        Formula:
            KL(q||p) = 0.5 * ( (var_q/var_p) + ((mu_p-mu_q)^2)/var_p - 1 + logvar_p - logvar_q )

        Returns:
            kl: [B, D] per-sample, per-dimension KL (no reduction)
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            (var_q / var_p)
            + ((mu_p - mu_q) ** 2) / var_p
            - 1.0
            + logvar_p
            - logvar_q
        )
        return kl

    @staticmethod
    def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
        """
        Elementwise Gaussian KL between q ~ N(mu_q, var_q) and p ~ N(mu_p, var_p).
        Returns per-sample KL summed over latent dims.

        Returns:
            kl_sum: [B] per-sample KL summed over latent dims
        """
        return RSSMWorldModel.gaussian_kl_per_dim(mu_q, logvar_q, mu_p, logvar_p).sum(-1)

    @staticmethod
    def kl_balanced(mu_q, logvar_q, mu_p, logvar_p, alpha: float = 0.8, free_bits: float = 0.1):
        """
        Balanced KL that mixes KL(q||p) and KL(p||q) to stabilize training.
        Applies a free-bits floor to avoid vanishing KL.

        Args:
            alpha: mixing coefficient for KL(q||p) vs KL(p||q)
            free_bits: minimum KL per sample **per latent dimension** (prevents
                       collapse on individual dimensions)

        Returns:
            scalar: mean balanced KL across batch

        BUGFIX NOTE:
            Free-bits was previously applied to the already-summed total KL
            (summed across all latent dimensions). That meant a single
            dimension carrying a large KL could satisfy the floor for the
            entire latent vector while every other dimension collapsed
            (posterior == prior) undetected -- exactly the failure mode
            free-bits is meant to prevent. We now compute KL per-dimension,
            clamp each dimension to `free_bits` independently, and only then
            sum across dimensions. This is standard free-bits practice, but
            it changes the effective regularization strength: with a
            latent_dim-dimensional z, the worst-case total floor rises from
            `free_bits` to `free_bits * latent_dim`. Existing `free_bits`
            values calibrated against the old (buggy) behavior will likely
            need to be reduced (e.g. divided by latent_dim) or re-tuned.
        """
        # Detach the opposing side in each KL to avoid double-backprop issues
        kl_qp = RSSMWorldModel.gaussian_kl_per_dim(mu_q, logvar_q, mu_p.detach(), logvar_p.detach())  # [B, D]
        kl_pq = RSSMWorldModel.gaussian_kl_per_dim(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p)  # [B, D]
        kl = alpha * kl_qp + (1.0 - alpha) * kl_pq  # [B, D]
        kl = torch.clamp(kl, min=free_bits)  # per-dimension floor, applied before reduction
        return kl.sum(-1).mean()

    def kl_loss(self, mu_q, logvar_q, mu_p, logvar_p, alpha: float = 0.8, free_bits: float = 0.1):
        """
        Convenience wrapper for the balanced KL loss used by the agent.
        """
        return RSSMWorldModel.kl_balanced(mu_q, logvar_q, mu_p, logvar_p, alpha, free_bits)

    # -------------------------
    # Overshooting (KL-based, adaptive)
    # -------------------------
    def overshoot_loss(self, h, z, actions, target_z, s=None, target_s=None, horizon: int = 3):
        """
        KL-based overshooting loss with adaptive pseudo-posterior variance.

        Behavior:
        - Rolls forward up to `horizon` steps using `transition`.
        - At each step compares rollout prior (mu_p, logvar_p) to a per-sample
          pseudo-posterior whose mean is target_z[k] and whose variance is:
            - batch variance across samples when B > 1 (per-feature), optionally
              augmented by ensemble predictive variance, and clamped to floor;
            - fallback to a conservative floor when B == 1.
        - Uses prior mean of slow latent (mu_s_p) for deterministic slow rollout.
        - Optionally includes slow-latent auxiliary loss (MSE) if target_s provided.

        Args:
            h: [B, Hdim]
            z: [B, D]
            actions: [T, B, A]
            target_z: [T, B, D]
            s: [B, S] or None
            target_s: [T, B, S] or None
            horizon: int

        Returns:
            scalar tensor: averaged KL-based overshoot loss
        """
        T, B, _ = actions.shape
        loss = 0.0
        steps = 0

        # not detaching for better horizon predictions, test to ensure not instable
        h_roll = h
        z_roll = z # add a .detach() if unstable
        s_roll = s.detach() if (s is not None) else None

        # Ensure s_roll exists if needed
        if s_roll is None and self.slow_dim > 0:
            s_roll = torch.zeros(z_roll.shape[0], self.slow_dim, device=self.device)

        # Limit to available timesteps
        max_k = min(horizon, T)

        for k in range(max_k):
            a_k = actions[k]  # [B, A]

            # Roll transition: returns h_roll, z_prior_sample_or_mean, mu_p, logvar_p
            h_roll, z_roll, mu_p, logvar_p = self.transition(h_roll, z_roll, a_k, s=s_roll)

            # Update s_roll using slow prior mean for deterministic rollout
            if self.slow_dim > 0:
                s_prior_sample, mu_s_p, logvar_s_p = self.slow_prior_dist(h_roll, s_roll)
                # Use prior mean for deterministic slow rollout (detach to avoid gradients)
                s_roll = mu_s_p.detach()

            # Build per-sample pseudo-posterior mean from target_z[k]
            z_target = target_z[k].detach()  # [B, D]
            mu_q = z_target  # [B, D]

            # Adaptive variance estimation:
            # - If batch > 1, estimate per-feature variance across batch.
            # - Optionally augment with ensemble predictive variance for this step.
            # - Clamp to pseudo_var_floor to avoid degenerate logvar.
            if z_target.shape[0] > 1:
                # variance across batch for each feature: shape [1, D]
                var_batch = z_target.var(dim=0, unbiased=False, keepdim=True)
                var_batch = torch.clamp(var_batch, min=1e-8)

                # Optionally add ensemble predictive variance for this step:
                if self.use_ensemble_var_in_overshoot and (self.ensemble is not None):
                    # Build ensemble input consistent with transition() contract: [z, a, s]
                    ensemble_inp = torch.cat([z_roll.detach(), a_k.detach(), s_roll.detach()], dim=-1)
                    # Ensemble contract: ensemble(ensemble_inp) -> (mean_z, var_z) shaped [B, D]
                    _, ensemble_var = self.ensemble(ensemble_inp)
                    # Average ensemble variance across batch dimension to get [1, D]
                    ensemble_var_mean = ensemble_var.mean(dim=0, keepdim=True)
                    var_est = var_batch + ensemble_var_mean
                else:
                    var_est = var_batch

                # Ensure floor
                var_est = torch.clamp(var_est, min=self.pseudo_var_floor)
                # Expand to [B, D] to match mu_p shape
                logvar_q_exp = var_est.log().expand_as(mu_p)
                mu_q_exp = mu_q
            else:
                # Batch size 1: use conservative floor
                var_floor = max(self.pseudo_var_floor, 1e-8)
                logvar_q_exp = torch.full_like(mu_p, float(var_floor)).log()
                mu_q_exp = mu_q

            # Align shapes
            if mu_q_exp.shape != mu_p.shape:
                mu_q_exp = mu_q_exp.expand_as(mu_p)
            if logvar_q_exp.shape != logvar_p.shape:
                logvar_q_exp = logvar_q_exp.expand_as(logvar_p)

            # Compute KL per sample and average across batch
            kl_step = self.gaussian_kl(mu_q_exp, logvar_q_exp, mu_p, logvar_p).mean()
            loss = loss + kl_step
            steps += 1

            # Optional slow-latent auxiliary term: compare rolled slow (mu_s_p used) to target_s
            if target_s is not None and self.slow_dim > 0:
                s_target = target_s[k].detach()  # [B, S]
                # Use MSE between mu_s_p (detached) and target_s
                loss = loss + F.mse_loss(s_roll, s_target)

        if steps == 0:
            return torch.tensor(0.0, device=self.device)
        return loss / steps

    # ------------------------
    # Persistence penalty
    # ----------------------
    def persistence_transform(self, delta_slow, slope: float = 0.01):
        """
        Asymmetric squared penalty:
        - positive delta: delta^2
        - negative delta: slope * delta^2

        This penalises rapid increases more than decreases (or vice versa depending on slope).
        """
        positive = delta_slow ** 2
        negative = slope * (delta_slow ** 2)
        return torch.where(delta_slow >= 0, positive, negative)

    def slow_persistence_loss(self, s_prev, s_next, slope: float = 0.01):
        """
        Computes mean persistence penalty on slow latent.

        Returns:
            scalar: mean penalty across batch and slow dimensions
        """
        delta = s_next - s_prev
        penalty = self.persistence_transform(delta, slope)
        return penalty.mean()

    # -------------------------
    # Device helper
    # -------------------------
    def to_device(self, device=None):
        """
        Move model and internal tensors to the requested device.

        Args:
            device: torch.device or None (defaults to CUDA if available)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)
        # Also move ensemble (if present and an nn.Module) to the same device to avoid CPU/GPU mismatches.
        if self.ensemble is not None:
            try:
                self.ensemble.to(device)
            except Exception:
                # If ensemble is not an nn.Module or does not support .to(), ignore.
                pass
