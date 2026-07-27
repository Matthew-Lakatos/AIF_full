"""
Emotion-Modulated PPO
Used as a baseline comparison model
Online per-step update with exploration rate (entropy) modulated
by Value Prediction Error (Surprise).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from .PPO_standard import PPOActorCritic


class EmotionModulatedPPO(PPOActorCritic):
    def __init__(self, obs_dim=16, action_dim=6):
        super().__init__(obs_dim, action_dim)
        self.surprise_mva = 0.0
        self.alpha = 0.05
        self.base_entropy_coef = 0.01
        self.dynamic_entropy_coef = self.base_entropy_coef

        # Minimal extra diagnostics
        self.last_surprise = 0.0
        self.last_modulation = 1.0

    def update(self, prev_obs, action, reward, done, next_obs):
        prev_obs_t = torch.as_tensor(prev_obs, dtype=torch.float32).unsqueeze(0)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        reward_t = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(0)
        done_t = torch.as_tensor(float(done), dtype=torch.float32).unsqueeze(0)

        value = self.critic(prev_obs_t)
        with torch.no_grad():
            next_value = self.critic(next_obs_t)

        target = reward_t + self.gamma * next_value * (1.0 - done_t)
        advantage = target - value

        current_surprise = advantage.abs().mean().item()
        self.surprise_mva = (1 - self.alpha) * self.surprise_mva + self.alpha * current_surprise
        modulation = current_surprise / (self.surprise_mva + 1e-6)
        modulation = float(np.clip(modulation, 0.5, 3.0))
        self.dynamic_entropy_coef = self.base_entropy_coef * modulation

        mu_old = self.actor(prev_obs_t).detach()
        std_old = torch.exp(self.log_std.detach())
        dist_old = Normal(mu_old, std_old)
        old_logprob = dist_old.log_prob(action_t).sum(-1).detach()

        for _ in range(self.k_epochs):
            mu = self.actor(prev_obs_t)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)

            logprob = dist.log_prob(action_t).sum(-1)
            entropy = dist.entropy().sum(-1)
            value_pred = self.critic(prev_obs_t)

            ratios = torch.exp(logprob - old_logprob)
            surr1 = ratios * advantage.detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.detach()

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (target - value_pred).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_loss + self.dynamic_entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Minimal diagnostics
        self.last_policy_loss = float(policy_loss.item())
        self.last_value_loss = float(value_loss.item())
        self.last_entropy = float(entropy.mean().item())
        self.last_advantage = float(advantage.mean().item())
        self.last_surprise = current_surprise
        self.last_modulation = modulation

        return float(loss.item())

    def get_diagnostics(self):
        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "advantage": self.last_advantage,
            "surprise": self.last_surprise,
            "modulation": self.last_modulation,
        }
