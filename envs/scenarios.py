"""
envs/research_envs.py

Purpose
-------
Collection of research-oriented environment scenarios used for experiments.
Provides:
- A configurable base environment `ResearchEnvironment` with unified sensor
  blackout support, seeding, and noisy observation generation.
- Ten concrete scenario classes (AffectiveTutor, ConflictResolution, EmotionExploration,
  HITL_CoCreation, LongHaulMission, ResourceGathering, SocialNavigation,
  SensorBlackout, Gaslighting, TransferEntropy) that implement simple latent
  dynamics and reward functions for controlled experiments.

Design notes
------------
- Observations are noisy projections of a latent state; this lets agents rely
  on world models and epistemic signals rather than raw reward alone.
- Blackout logic is unified and configurable via `set_blackout(...)`.
- All scenarios return an `info` dict containing `blackout_active` and other
  scenario-specific diagnostics to support logging and plotting.
- The module is intentionally lightweight and deterministic when seeds are set.
"""

import torch
import numpy as np
from scipy.special import expit  # sigmoid for non-linear dynamic


class ResearchEnvironment:
    """
    Base research environment providing:
    - latent-state to noisy-observation mapping
    - unified, configurable blackout support
    - simple seed control for reproducibility
    - per-step diagnostics via the info dict
    """

    def __init__(self, name=None, obs_dim=16, action_dim=6, max_steps=500, seed=None):
        """
        Args:
            name: optional environment name for logging
            obs_dim: dimensionality of observation vector returned to agents
            action_dim: dimensionality of action vectors expected by step()
            max_steps: episode length (used to compute blackout start from fraction)
            seed: optional RNG seed for reproducibility (numpy + torch)
        """
        self.name = name if name is not None else "ResearchEnv"
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.steps = 0
        self.max_steps = int(max_steps)
        self.history = []

        # Unified blackout configuration (defaults suitable for research)
        self.blackout_enabled = False
        self.blackout_fraction = 0.5
        self.blackout_noise_std = 0.0
        self.blackout_duration_steps = None
        self.blackout_start_step = -1
        self.blackout_active = False

        # Reproducible seeds
        self._seed = None
        if seed is not None:
            self.set_seed(seed)

    # -------------------------
    # Configuration helpers
    # -------------------------
    def set_seed(self, seed: int):
        """
        Set RNG seeds for reproducibility (numpy and torch, including CUDA if available).
        Call this before running episodes to ensure deterministic draws where possible.
        """
        self._seed = int(seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

    def set_blackout(self, enabled: bool = True, fraction: float = 0.5,
                     duration_steps: int = None, noise_std: float = 0.0):
        """
        Configure blackout behavior.

        Args:
            enabled: whether blackout is active
            fraction: fraction of episode after which blackout begins (0.0-1.0)
            duration_steps: number of steps to blackout; if None, blackout continues to episode end
            noise_std: if 0.0, observations are zeroed; if >0, Gaussian noise std used
        """
        self.blackout_enabled = bool(enabled)
        self.blackout_fraction = float(fraction)
        self.blackout_noise_std = float(noise_std)
        self.blackout_duration_steps = None if duration_steps is None else int(duration_steps)
        # compute start step immediately if possible
        self.blackout_start_step = int(self.max_steps * self.blackout_fraction) if self.blackout_enabled else -1

    # -------------------------
    # Episode lifecycle
    # -------------------------
    def reset(self):
        """
        Reset episode counters and latent state, compute blackout start step,
        and return initial observation + info dict.
        """
        self.steps = 0
        self.blackout_start_step = int(self.max_steps * self.blackout_fraction) if self.blackout_enabled else -1
        self.blackout_active = False
        # Latent state initialised with small Gaussian noise (10D internal state)
        self.latent_state = np.random.normal(0, 0.2, 10)
        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, info

    def _generate_noisy_obs(self):
        """Maps 10D latent state -> obs_dim noisy observation tensor (torch.Tensor)."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        n = min(len(self.latent_state), self.obs_dim)
        obs[:n] = self.latent_state[:n]
        # Add epistemic noise to observations; AIF agents can filter this noise.
        obs += np.random.normal(0, 0.08, self.obs_dim).astype(np.float32)
        return torch.tensor(obs, dtype=torch.float32)

    def _apply_blackout(self, obs: torch.Tensor):
        """
        Apply sensor blackout if configured:
        - replaces observation with zeros or Gaussian noise depending on blackout_noise_std
        - does NOT affect reward or dynamics (only observation channel)
        - sets self.blackout_active flag for diagnostics
        """
        if not self.blackout_enabled or self.blackout_start_step < 0:
            self.blackout_active = False
            return obs

        start = self.blackout_start_step
        if self.blackout_duration_steps is None:
            end = self.max_steps
        else:
            end = start + int(self.blackout_duration_steps)

        if self.steps >= start and self.steps < end:
            self.blackout_active = True
            if self.blackout_noise_std == 0.0:
                return torch.zeros_like(obs)
            else:
                return torch.randn_like(obs) * float(self.blackout_noise_std)
        else:
            self.blackout_active = False
            return obs

    def preferred_obs(self):
        """
        Default preferred observation prior for EFE risk term.
        By default, neutral around 0 (matches original AIF behaviour).
        Specific environments override this to encode task preferences.
        """
        return torch.zeros(self.obs_dim)

    def step(self, action):
        """Base step method – should be overridden by subclasses."""
        raise NotImplementedError("Concrete environments must implement step().")


# -------------------------
# Concrete scenario classes
# -------------------------
# Each scenario implements simple latent dynamics and returns (obs, reward, done, info).
# The implementations are intentionally simple and interpretable for controlled experiments.

class AffectiveTutor(ResearchEnvironment):
    def step(self, action):
        """
        Latent dynamics modelled as simple mood variables:
        - latent[1]: Frustration, increases with complexity, decreases with support
        - latent[2]: Mastery, increases with pace*challenge modulated by frustration
        Reward: mastery*2 - frustration
        """
        self.steps += 1
        tiredness = self.steps / self.max_steps
        self.latent_state[1] += (action[1] * 0.2) - (action[2] * 0.15) + (tiredness * 0.05)
        self.latent_state[2] += (action[0] * action[4] * 0.1) * (1 - self.latent_state[1])

        self.latent_state = np.clip(self.latent_state, 0, 1)
        reward = self.latent_state[2] * 2.0 - self.latent_state[1]
        done = self.steps >= self.max_steps or self.latent_state[1] > 0.95

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "latent": self.latent_state,
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info

    def preferred_obs(self):
        """Prefer low frustration (obs[1] ~ 0) and high mastery (obs[2] ~ 1)."""
        pref = torch.zeros(self.obs_dim)
        pref[1] = 0.0
        pref[2] = 1.0
        return pref


class ConflictResolution(ResearchEnvironment):
    def step(self, action):
        """
        Trust dynamics: empathy increases trust, threatening actions reduce it.
        Reward depends on success which itself depends on trust.
        """
        self.steps += 1
        trust = self.latent_state[4]
        self.latent_state[4] += (action[2] * 0.1) - (action[1] * 0.4)
        self.latent_state[4] = np.clip(self.latent_state[4], -1, 1)

        success = (action[0] * 0.2) + (action[3] * 0.3 * max(0, trust))
        reward = success if trust > 0 else -0.5
        done = self.steps >= self.max_steps

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "trust": trust,
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info


class EmotionExploration(ResearchEnvironment):
    def step(self, action):
        """
        Zero-reward information-seeking task: agent must modulate visibility of
        a latent cause via action[5] to observe structure. AIF agents should
        exploit epistemic value to learn here.
        """
        self.steps += 1
        latent_cause = np.sin(self.steps * 0.05)
        self.latent_state[0] = latent_cause * action[5]

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, 0.0, self.steps >= self.max_steps, info

    def preferred_obs(self):
        """Prefer some visible structure in obs[0]."""
        pref = torch.zeros(self.obs_dim)
        pref[0] = 0.5
        return pref


class HITL_CoCreation(ResearchEnvironment):
    def step(self, action):
        """
        Human-in-the-loop co-creation: human preference drifts over time (non-stationary).
        Reward is noisy feedback aligned with a time-varying preference.
        """
        self.steps += 1
        pref = np.sin(self.steps * 0.02)
        alignment = np.dot(action[:3], [pref, 1 - pref, 0.5])
        reward = alignment + np.random.normal(0, 0.2)
        done = self.steps >= self.max_steps

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "pref": pref,
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info


class LongHaulMission(ResearchEnvironment):
    def step(self, action):
        """
        Long-horizon homeostatic survival: systems decay slowly; agent repairs one
        system per step (argmax of action). Sparse reward at arrival if systems healthy.
        """
        self.steps += 1
        self.latent_state[:6] -= np.random.uniform(0, 0.01, 6)
        repair_idx = np.argmax(action)
        self.latent_state[repair_idx] += 0.04

        self.latent_state = np.clip(self.latent_state, 0, 1)
        reward = 100.0 if (self.steps == 499 and np.all(self.latent_state[:6] > 0.1)) else 0.0
        done = np.any(self.latent_state[:6] <= 0) or self.steps >= self.max_steps

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "integrity": self.latent_state[:6],
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info

    def preferred_obs(self):
        """Prefer first 6 systems to be healthy (obs[:6] ~ 1)."""
        pref = torch.zeros(self.obs_dim)
        pref[:6] = 1.0
        return pref


class ResourceGathering(ResearchEnvironment):
    def step(self, action):
        """
        Risk vs reward: greed increases yield but builds hidden predator tension.
        Random strikes penalise the agent; tension resets on strike.
        """
        self.steps += 1
        self.latent_state[7] += action[0] * 0.1
        strike = np.random.random() < (self.latent_state[7] ** 2)
        reward = -20.0 if strike else action[0] * 5.0
        if strike:
            self.latent_state[7] = 0
        done = self.steps >= self.max_steps

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "tension": self.latent_state[7],
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info


class SocialNavigation(ResearchEnvironment):
    def step(self, action):
        """
        2D navigation under social friction: reward penalises friction proportional
        to distance and a social action component.
        """
        self.steps += 1
        self.latent_state[8:10] += action[:2] * 0.1
        friction = np.linalg.norm(self.latent_state[8:10]) * action[2]
        reward = -friction
        done = self.steps >= self.max_steps

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info


class SensorBlackout(ResearchEnvironment):
    def step(self, action):
        """
        Stress test: sensor blackout. Dynamics continue but observations may be masked.
        Reward encourages latent_state[0] near 5.
        """
        self.steps += 1
        self.latent_state[0] += action[0] * 0.2

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)

        reward = 1.0 if abs(self.latent_state[0] - 5.0) < 0.5 else 0.0
        done = self.steps >= self.max_steps
        info = {
            "blind": self.blackout_active,
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info

    def preferred_obs(self):
        """Prefer obs[0] ~ 5 (maps to latent_state[0] near 5)."""
        pref = torch.zeros(self.obs_dim)
        pref[0] = 5.0
        return pref


class Gaslighting(ResearchEnvironment):
    def step(self, action):
        """
        Epistemic trust test: reward flips during certain intervals. AIF agents
        should rely on prediction error to detect and ignore flipped feedback.
        """
        self.steps += 1
        is_flipping = (100 < self.steps < 200) or (350 < self.steps < 450)
        base_reward = 1.0 if action[0] > 0.5 else -1.0
        reward = -base_reward if is_flipping else base_reward
        done = self.steps >= self.max_steps

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "gaslight": is_flipping,
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info


class TransferEntropy(ResearchEnvironment):
    def step(self, action):
        """
        Structure test: goal flips mid-episode. AIF should adapt by changing
        prior preferences (preferred_obs) rather than relearning dynamics.
        """
        self.steps += 1
        goal = 1.0 if self.steps < 250 else -1.0
        self.latent_state[0] += action[0] * 0.1
        reward = -abs(self.latent_state[0] - goal)
        done = self.steps >= self.max_steps

        obs = self._generate_noisy_obs()
        obs = self._apply_blackout(obs)
        info = {
            "goal": goal,
            "blackout_active": self.blackout_active,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
        return obs, reward, done, info

    def preferred_obs(self):
        """Preferred observation tracks current goal sign in obs[0]."""
        goal = 1.0 if self.steps < 250 else -1.0
        pref = torch.zeros(self.obs_dim)
        pref[0] = goal
        return pref


# Compatibility alias for tests
#
# BUGFIX: this previously aliased ScenarioEnv directly to the ABSTRACT base
# class `ResearchEnvironment`, whose step() unconditionally raises
# NotImplementedError. tests/compare_blackout_ablations.py imports
# `ScenarioEnv`, constructs it, calls `env.set_blackout(...)`, and then calls
# `env.step(action)` in its rollout loop -- which would crash immediately on
# the very first step. ScenarioEnv is now aliased to the concrete
# `SensorBlackout` environment, which is what that script's blackout-ablation
# comparisons actually need (reward encourages latent_state[0] near 5, with
# blackout applied to the observation channel only, matching the script's
# assumptions).
ScenarioEnv = SensorBlackout
