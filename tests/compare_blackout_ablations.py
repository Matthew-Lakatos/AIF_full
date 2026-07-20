# experiments/compare_blackout_ablations.py
# Purpose: Run short experiments comparing blackout ablations and save CSV results.

import argparse
import csv
import os
import sys
import statistics
import numpy as np
import torch
import pandas as pd
from datetime import datetime

from envs.scenarios import ScenarioEnv
from models.AIF import ActiveInferenceAgent
from utilities.seed import set_all_seeds   # added

DEFAULT_EPISODES = 200
DEFAULT_SEEDS = 3
OUTPUT_DIR = "experiments/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_agent_variant(variant_name, seed, use_context=True):
    """
    Create agent variants:
      - 'full': default agent with ensemble and persistence
      - 'no-ensemble': disable ensemble in world model
      - 'no-persistence': set lambda_slow to 0
      - 'fixed-average': force posterior correction to fixed average (posterior correction disabled or fixed beta)
    """
    set_all_seeds(seed)   # added
    agent = ActiveInferenceAgent(obs_dim=16, action_dim=6, latent_dim=32, hidden_dim=64, use_context=use_context, lambda_slow=0.1)
    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    # store seed
    agent.last_seed = seed
    # apply variant modifications
    if variant_name == "full":
        pass
    elif variant_name == "no-ensemble":
        agent.ensemble = None
        agent.world_model.ensemble = None
    elif variant_name == "no-persistence":
        agent.lambda_slow = 0.0
        agent.world_model.lambda_slow = 0.0
    elif variant_name == "fixed-average":
        # replace precision controller with fixed beta=0.5
        class FixedPrecision:
            def __call__(self, *args, **kwargs):
                return torch.tensor(0.5)
        agent.precision = FixedPrecision()
    else:
        raise ValueError(f"Unknown variant {variant_name}")
    return agent


def run_episode(env, agent, max_steps=None):
    """Run one episode and return diagnostics and rewards split by blackout."""
    obs, info = env.reset()
    total_reward = 0.0
    pre_blackout_rewards = []
    blackout_rewards = []
    for t in range(env.max_steps if max_steps is None else max_steps):
        action_np, _, _, _ = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action_np)
        # agent update (train step)
        agent.update(prev_obs=obs, action=action_np, next_obs=next_obs)
        # track blackout status in agent
        if hasattr(agent, "last_blackout_active"):
            agent.last_blackout_active = info.get("blackout_active", False)
        total_reward += reward
        if info["blackout_active"]:
            blackout_rewards.append(reward)
        else:
            pre_blackout_rewards.append(reward)
        obs = next_obs
        if done:
            break
    pre_mean = float(np.mean(pre_blackout_rewards)) if len(pre_blackout_rewards) > 0 else 0.0
    blackout_mean = float(np.mean(blackout_rewards)) if len(blackout_rewards) > 0 else 0.0
    return {
        "total_reward": float(total_reward),
        "pre_blackout_reward": pre_mean,
        "blackout_reward": blackout_mean,
        "last_epistemic": float(agent.last_epistemic) if hasattr(agent, "last_epistemic") else 0.0,
    }


def main(args):
    seeds = args.seeds
    episodes = args.episodes
    variants = ["full", "no-ensemble", "no-persistence", "fixed-average"]
    csv_path = os.path.join(OUTPUT_DIR, f"compare_blackout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["seed", "episode", "total_reward", "pre_blackout_reward", "blackout_reward", "last_epistemic", "model_variant"])
        writer.writeheader()
        for seed in range(args.seed, args.seed + seeds):
            for variant in variants:
                set_all_seeds(seed)   # added
                agent = make_agent_variant(variant, seed)
                env = ScenarioEnv(obs_dim=16, action_dim=6, max_steps=50, seed=seed)
                env.set_blackout(enabled=True, fraction=0.5, duration_steps=None, noise_std=0.0)
                for ep in range(episodes):
                    stats = run_episode(env, agent)
                    writer.writerow({
                        "seed": seed,
                        "episode": ep,
                        "total_reward": stats["total_reward"],
                        "pre_blackout_reward": stats["pre_blackout_reward"],
                        "blackout_reward": stats["blackout_reward"],
                        "last_epistemic": stats["last_epistemic"],
                        "model_variant": variant,
                    })
    # Summarize results
    # load CSV and compute mean ± std of blackout drop per variant
    df = pd.read_csv(csv_path)
    summary = []
    for variant in variants:
        sub = df[df["model_variant"] == variant]
        # compute mean blackout drop: pre - blackout
        drops = (sub["pre_blackout_reward"] - sub["blackout_reward"]).values
        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))
        summary.append((variant, mean_drop, std_drop))
    print("Blackout performance drop (pre - during) by variant:")
    for v, m, s in summary:
        print(f"{v:15s}: {m:.4f} ± {s:.4f}")
    print(f"CSV saved to: {csv_path}")

    # Smoke test: check for NaNs in critical columns
    if df[["total_reward", "pre_blackout_reward", "blackout_reward", "last_epistemic"]].isnull().any().any():
        raise RuntimeError("NaNs detected in output CSV!")
    print("Smoke test passed: no NaNs in output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Episodes per variant")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    args = parser.parse_args()
    # set deterministic seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
