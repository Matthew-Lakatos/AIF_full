"""
Standard Proximal Policy Optimization (PPO)
Used as baseline model comparison
Online actor-critic with per-step updates for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim=16, action_dim=6):
        super().__init__()
        self.obs_dim = obs_dim   # <-- ADD THIS LINE
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.entropy_coef = 0.01

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        # Minimal diagnostics
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.last_advantage = 0.0

    def get_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mu = self.actor(state_t)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        value = self.critic(state_t)

        return (
            action.detach().squeeze(0).numpy(),
            dist.log_prob(action).sum(-1).item(),
            dist.entropy().sum(-1).item(),
            value.item(),
        )

    def evaluate(self, states, actions):
        mu = self.actor(states)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        logprobs = dist.log_prob(actions).sum(-1)
        dist_entropy = dist.entropy().sum(-1)
        state_values = self.critic(states).squeeze(-1)
        return logprobs, state_values, dist_entropy

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

            loss = policy_loss + value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Minimal diagnostics
        self.last_policy_loss = float(policy_loss.item())
        self.last_value_loss = float(value_loss.item())
        self.last_entropy = float(entropy.mean().item())
        self.last_advantage = float(advantage.mean().item())

        return float(loss.item())

    def get_diagnostics(self):
        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "advantage": self.last_advantage,
        }
