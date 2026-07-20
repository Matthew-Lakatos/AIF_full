"""
train_all.py

Purpose
-------
Unified training script for running experiments across multiple environments
and agents (Active Inference, PPO variants, and AIF baselines). Handles
seeding, environment construction, agent instantiation, episode rollouts,
per-step updates, diagnostics aggregation, per-seed result saving, and
aggregation/plotting of final results.

What is implemented
-------------------
- Mapping of environment names to environment classes (ENVS).
- Mapping of agent names to agent classes (AGENTS).
- build_agent(agent_name, env, args): constructs an agent with environment-specific
  configuration for the AIF agent and default construction for other agents.
- train_single(env_name, agent_name, episodes, seed, args): runs one seed's
  training loop, including optional latency buffering and sensor blackout.
- aggregate_results(df): aggregates per-episode metrics across seeds (mean/std).
- main(): CLI entry point to run experiments across agents and seeds, save CSVs,
  and produce summary plots.

Design choices and caveats
--------------------------
- The AIF agent receives several ablation flags (dynamic beta, persistence,
  posterior correction, EFE) via command-line arguments; these flags are passed
  into the ActiveInferenceAgent constructor.
- The script supports a latency buffer wrapper (LatencyBuffer) to simulate
  delayed observations; when enabled, the agent may receive delayed or repeated
  observations.
- Sensor blackout can be enabled per-run via the --blackout flag; the environment
  must implement set_blackout(enabled, fraction, duration_steps, noise_std).
- PPO variants are asserted to contain no recurrent layers for fairness.
- The environment factory attempts to call env_cls(name=...) if the constructor
  accepts a 'name' argument; otherwise it calls the constructor with no args.
- The script expects env.reset() to return (obs, info) and env.step(action)
  to return (next_obs, reward, done, info). If environments differ, adaptors
  are required.

Notes for the paper (concise)
-----------------------------
- This script orchestrates experiments reported in the paper. For reproducibility,
  we set global seeds at the start of each seed run (set_all_seeds) and save
  per-seed CSVs and aggregated CSVs. The AIF agent is configured with the
  ablation flags used in the experiments.
- Default experimental settings used in the repository: ensemble_size=5,
  latent_dim and other model hyperparameters are defined in the model modules.
"""

import os
import argparse
import random
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Import environment classes. Each environment should expose:
# - obs_dim, action_dim attributes
# - reset() -> (obs, info)
# - step(action) -> (next_obs, reward, done, info)
# - optional: preferred_obs(), set_blackout(), set_seed()
from envs.scenarios import (
    AffectiveTutor,
    ConflictResolution,
    EmotionExploration,
    HITL_CoCreation,
    LongHaulMission,
    ResourceGathering,
    SocialNavigation,
    SensorBlackout,
    Gaslighting,
    TransferEntropy,
)

# Import agent classes. Each agent should implement:
# - get_action(obs) -> (action, logprob, entropy, value)
# - update(...) with signature depending on agent type
# - optional: reset_belief(), get_diagnostics()
from models.AIF import ActiveInferenceAgent
from models.PPO_standard import PPOActorCritic
from models.PPO_emotion_standard import EmotionModulatedPPO
from models.aif_baselines.DAIF_Tschantz import DAIF_Tschantz
from models.aif_baselines.PCRL_Whittington import PCRL_Whittington
from models.aif_baselines.FEAC_Friston import FEAC_Friston
from models.aif_baselines.DAIFC_Millidge import DAIFC_Millidge

# Utilities: latency buffer for delayed observations, deterministic seeding helper
from utilities.latency import LatencyBuffer
from utilities.seed import set_all_seeds   # sets global RNGs for reproducibility
from plot_results import plot_reward_curves, plot_diagnostics


# Environment registry: map string names used on CLI to environment classes.
ENVS = {
    "AffectiveTutor": AffectiveTutor,
    "ConflictResolution": ConflictResolution,
    "EmotionExploration": EmotionExploration,
    "HITL_CoCreation": HITL_CoCreation,
    "LongHaulMission": LongHaulMission,
    "ResourceGathering": ResourceGathering,
    "SocialNavigation": SocialNavigation,
    "SensorBlackout": SensorBlackout,
    "Gaslighting": Gaslighting,
    "TransferEntropy": TransferEntropy,
}

# Agent registry: map string names to agent classes used by build_agent.
AGENTS = {
    "AIF": ActiveInferenceAgent,
    "PPO": PPOActorCritic,
    "EmotionPPO": EmotionModulatedPPO,
    "DAIF": DAIF_Tschantz,
    "PCRL": PCRL_Whittington,
    "FEAC": FEAC_Friston,
    "DAIFC": DAIFC_Millidge,
}

# Set of agent names that are Active Inference variants (used to branch update logic).
AIF_NAMES = {"AIF", "DAIF", "PCRL", "FEAC", "DAIFC"}


def build_agent(agent_name: str, env, args) -> torch.nn.Module:
    """
    Construct and return an agent instance for the given agent_name and environment.

    Implemented behaviour:
    - For "AIF", construct ActiveInferenceAgent with environment-specific
      preferred observations and ablation flags passed from CLI args.
    - For other agents, instantiate the class from AGENTS with obs_dim and action_dim.

    Args:
        agent_name: string key identifying the agent.
        env: environment instance (must expose obs_dim and action_dim).
        args: parsed CLI args containing ablation flags and other options.

    Returns:
        agent: instantiated agent object (PyTorch module or agent wrapper).
    """
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    if agent_name == "AIF":
        # Environment-specific preferred observations for the EFE risk term.
        preferred = env.preferred_obs() if hasattr(env, "preferred_obs") else None

        # Construct ActiveInferenceAgent with ablation flags controlled by CLI args.
        agent = ActiveInferenceAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            preferred_obs=preferred,
            use_dynamic_beta=not args.no_beta,
            use_persistence=not args.no_persistence,
            use_posterior_correction=not args.no_posterior,
            use_efe=not args.no_efe,
            use_context=args.use_context,
        )
        # Disable ensemble if requested
        if args.no_ensemble:
            agent.ensemble = None
            agent.world_model.ensemble = None
        return agent

    # For non-AIF agents, instantiate the mapped class with obs/action dims.
    cls = AGENTS[agent_name]
    return cls(obs_dim=obs_dim, action_dim=action_dim)


def is_aif_agent(agent_name: str) -> bool:
    """Return True if the agent_name corresponds to an Active Inference variant."""
    return agent_name in AIF_NAMES


def train_single(
    env_name: str,
    agent_name: str,
    episodes: int,
    seed: int,
    args,
    log_per_step: bool = False,
) -> pd.DataFrame:
    """
    Run training for a single seed and return a DataFrame of per-episode metrics.

    Behaviour:
    - Sets global RNG seeds for reproducibility.
    - Constructs the environment and seeds it.
    - Optionally enables sensor blackout on the environment.
    - Builds the agent and runs `episodes` episodes, performing agent-specific
      updates each step.
    - Collects per-episode metrics and diagnostics into a pandas DataFrame.

    Returns:
        df: pandas DataFrame with one row per episode containing metrics and diagnostics.
    """
    # Set deterministic seeds for reproducibility across numpy, torch, random.
    set_all_seeds(seed)

    # Instantiate environment. Some env classes accept a 'name' argument in __init__.
    env_cls = ENVS[env_name]
    env = env_cls(name=env_name) if "name" in env_cls.__init__.__code__.co_varnames else env_cls()
    # Ensure environment RNG is seeded if the env exposes set_seed.
    env.set_seed(seed)

    # Global blackout configuration (applies to the whole run if requested).
    if args.blackout and hasattr(env, "set_blackout"):
        # Default blackout parameters used in experiments; tune as needed.
        env.set_blackout(enabled=True, fraction=0.5, duration_steps=20, noise_std=0.0)

    # Build the agent for this environment and seed.
    agent = build_agent(agent_name, env, args)
    if hasattr(agent, "last_seed"):
        # Store seed in agent for logging/debugging if agent supports it.
        agent.last_seed = seed

    aif_flag = is_aif_agent(agent_name)

    # Enforce PPO fairness: PPO agents must not contain recurrent modules.
    if agent_name in ["PPO", "EmotionPPO"]:
        for name, module in agent.named_modules():
            assert not isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)), \
                f"PPO cannot have recurrent layer {name}"

    # Optional latency buffer to simulate delayed observations (Dreamer-style).
    latency_buffer = None
    if args.latency > 0:
        latency_buffer = LatencyBuffer(latency_steps=args.latency)

    # History collects per-episode rows for later conversion to DataFrame.
    history: List[Dict[str, Any]] = []

    for ep in range(episodes):
        # Reset environment; updated environments return (obs, info).
        obs, info = env.reset()
        if latency_buffer is not None:
            latency_buffer.reset()

        # Reset agent belief state for AIF agents if supported.
        if aif_flag and hasattr(agent, "reset_belief"):
            agent.reset_belief()

        done = False
        ep_reward = 0.0
        ep_vfe = 0.0
        step_count = 0

        # Episode-level accumulators for diagnostics and blackout metrics.
        #
        # BUGFIX / FEATURE: previously only `Reward` was split into
        # PreBlackoutReward/BlackoutReward, and every other diagnostic (beta,
        # epistemic, complexity, kl_balanced, mse, etc.) was summed over the
        # WHOLE episode regardless of blackout status -- so even if a
        # mechanism (e.g. the precision controller) genuinely behaved
        # differently during blackout, that signal was diluted into a
        # whole-episode average and invisible in the logs. We now track four
        # phases for every metric (reward AND every diagnostic key):
        #   - "pre":           before blackout starts
        #   - "blackout":      during the blackout window itself
        #   - "recovery":      the `recovery_window` steps immediately after
        #                      blackout ends (the "fallout" -- does the agent
        #                      recover quickly, or does corrupted belief linger?)
        #   - "post_recovery": remaining steps after the recovery window, once
        #                      behavior should have returned to baseline
        # `phase_sums[phase][metric]` accumulates the per-step sum;
        # `phase_steps[phase]` counts steps in that phase, for computing means.
        phase_sums: Dict[str, Dict[str, float]] = {
            "pre": {}, "blackout": {}, "recovery": {}, "post_recovery": {}
        }
        phase_steps: Dict[str, int] = {"pre": 0, "blackout": 0, "recovery": 0, "post_recovery": 0}
        recovery_window = max(0, int(getattr(args, "recovery_window", 20)))
        # Whole-episode diagnostic sums, kept for backward compatibility with
        # existing unsuffixed diagnostic columns (e.g. "beta", "epistemic").
        diag_sums: Dict[str, float] = {}

        # Per-step data collection (if requested)
        step_data = [] if log_per_step else None

        prev_obs_eff = None  # last effective observation seen by the agent

        while not done:
            # Apply latency wrapper: agent may see delayed observations.
            obs_eff = obs
            if latency_buffer is not None:
                latency_buffer.push(obs)
                delayed = latency_buffer.pop()
                if delayed is not None:
                    obs_eff = delayed
                elif prev_obs_eff is not None:
                    # If buffer not yet full, reuse previous effective observation.
                    obs_eff = prev_obs_eff

            # Convert observation to numpy array if it's a torch.Tensor.
            if isinstance(obs_eff, torch.Tensor):
                obs_np = obs_eff.numpy()
            else:
                obs_np = np.asarray(obs_eff, dtype=np.float32)

            # Verify observation dimensionality matches agent expectation.
            assert obs_np.shape[-1] == agent.obs_dim, \
                f"Observation shape mismatch: expected {agent.obs_dim}, got {obs_np.shape[-1]}"

            # BUGFIX: previously preferred_obs was only ever read once, at agent
            # construction (see build_agent), so environments with a
            # non-stationary goal (e.g. TransferEntropy's mid-episode goal flip)
            # never actually updated the agent's target. Re-query it every step
            # for any agent that supports updating it (currently just AIF; see
            # AIF.set_preferred_obs for the scope/theory caveats of doing this).
            if hasattr(agent, "set_preferred_obs") and hasattr(env, "preferred_obs"):
                agent.set_preferred_obs(env.preferred_obs())

            # Query agent for action. Agents return (action, logprob, entropy, value).
            action, logprob, entropy, value = agent.get_action(obs_np)

            # Step the environment with the chosen action.
            next_obs, reward, done, info = env.step(action)

            # Determine which phase this step belongs to: pre / blackout /
            # recovery (the "fallout" window right after blackout ends) /
            # post_recovery. Computed once per step and reused below for both
            # reward and diagnostics accumulation, so every logged metric --
            # not just reward -- can be inspected per-phase.
            blackout_active = info.get("blackout_active", False)
            current_step_num = info.get("steps", step_count + 1)
            phase = "pre"
            if blackout_active:
                phase = "blackout"
            elif (
                getattr(env, "blackout_enabled", False)
                and getattr(env, "blackout_start_step", -1) >= 0
                and getattr(env, "blackout_duration_steps", None) is not None
            ):
                blackout_end_step = env.blackout_start_step + env.blackout_duration_steps
                if current_step_num >= blackout_end_step:
                    if current_step_num < blackout_end_step + recovery_window:
                        phase = "recovery"
                    else:
                        phase = "post_recovery"
                # else: current_step_num < blackout_start_step -> still "pre" (default)

            phase_steps[phase] += 1
            phase_sums[phase]["Reward"] = phase_sums[phase].get("Reward", 0.0) + float(reward)

            if aif_flag:
                # AIF-style update: agent.update(prev_obs, action, next_obs) returns VFE.
                vfe = agent.update(obs_np, action, next_obs)
                ep_vfe += float(vfe)
                # Compute per-step VFE from components (for logging)
                acc = getattr(agent, 'last_accuracy', float('nan'))
                comp = getattr(agent, 'last_complexity', float('nan'))
                beta = getattr(agent, 'last_beta', float('nan'))
                overshoot = getattr(agent, 'last_overshoot', float('nan'))
                efe = getattr(agent, 'last_efe', float('nan'))
                slow_persist = getattr(agent, 'last_slow_persist', float('nan'))
                mse = getattr(agent, 'last_mse', float('nan'))
                vfe_step = acc + beta * comp + overshoot if not any(pd.isna([acc, beta, comp, overshoot])) else float('nan')
            else:
                # PPO-style update: agent.update(prev_obs, action, reward, done, next_obs)
                agent.update(obs_np, action, reward, done, next_obs)
                vfe_step = float('nan')
                acc = float('nan')
                comp = float('nan')
                beta = float('nan')
                overshoot = float('nan')
                efe = float('nan')
                slow_persist = float('nan')
                mse = float('nan')

            # Collect diagnostics if the agent exposes them.
            if hasattr(agent, "get_diagnostics"):
                diags = agent.get_diagnostics()
                for k, v in diags.items():
                    diag_sums[k] = diag_sums.get(k, 0.0) + float(v)
                    # BUGFIX / FEATURE: also accumulate per-phase, so e.g. beta's
                    # behavior specifically during blackout/recovery isn't
                    # diluted into a whole-episode average (see phase_sums
                    # docstring above for the full rationale).
                    phase_sums[phase][k] = phase_sums[phase].get(k, 0.0) + float(v)

            # Optionally store blackout status on the agent for logging.
            if hasattr(agent, "last_blackout_active"):
                agent.last_blackout_active = info.get("blackout_active", False)

            # Record per-step data if requested
            if log_per_step:
                step_row = {
                    "step": step_count,
                    "blackout_active": info.get("blackout_active", False),
                    "phase": phase,
                    "reward": reward,
                    "vfe_step": vfe_step,
                    "accuracy": acc,
                    "complexity": comp,
                    "beta": beta,
                    "overshoot": overshoot,
                    "efe": efe,
                    "slow_persist": slow_persist,
                    "mse": mse,
                }
                step_data.append(step_row)

            ep_reward += float(reward)
            step_count += 1
            prev_obs_eff = obs_eff
            obs = next_obs

        # Save per-step CSV if requested and data exists
        if log_per_step and step_data:
            step_df = pd.DataFrame(step_data)
            step_filename = f"{env_name}_{agent_name}_seed{seed}_ep{ep}_per_step.csv"
            step_path = os.path.join(args.outdir, step_filename)
            step_df.to_csv(step_path, index=False)

        # Compute mean diagnostics over ALL episode steps (backward-compatible,
        # unsuffixed columns -- e.g. "beta", "epistemic").
        diag_means = {f"{k}": (v / max(step_count, 1)) for k, v in diag_sums.items()}

        # Compute per-phase means for every metric (reward + every diagnostic),
        # covering pre / blackout / recovery ("fallout") / post_recovery.
        # This is the fix for the logging-granularity issue: previously only
        # Reward was split (into pre/blackout), and no phase captured recovery
        # dynamics at all, so "does the agent recover quickly after blackout
        # ends, or does corrupted belief linger" was invisible in the logs.
        phase_mean_cols: Dict[str, float] = {}
        for phase_name, sums in phase_sums.items():
            n_steps_phase = max(phase_steps[phase_name], 1)
            for metric_name, total in sums.items():
                phase_mean_cols[f"{metric_name}_{phase_name}"] = total / n_steps_phase

        # Backward-compatible aliases (previous column names).
        pre_avg = phase_mean_cols.get("Reward_pre", float("nan"))
        blackout_avg = phase_mean_cols.get("Reward_blackout", float("nan"))

        # Compose a row summarising the episode.
        row = {
            "Episode": ep,
            "Reward": ep_reward,
            "VFE": ep_vfe if aif_flag else np.nan,
            "Model": agent_name,
            "Seed": seed,
            "PreBlackoutReward": pre_avg,
            "BlackoutReward": blackout_avg,
        }
        # Merge whole-episode diagnostics, then per-phase breakdown, into the row.
        row.update(diag_means)
        row.update(phase_mean_cols)
        history.append(row)

    # Convert history to DataFrame and return.
    df = pd.DataFrame(history)
    return df


def aggregate_blackout_pooled(df_all: pd.DataFrame, last_n_episodes: int = 50) -> dict:
    """
    Pool blackout-phase reward across ALL episodes (within the last
    `last_n_episodes`, to reflect converged behavior) AND all seeds for a
    single model, to produce a properly-powered summary statistic.

    METHODOLOGY FIX:
    aggregate_results() groups by (Model, Episode) and reports mean/std
    across only `len(seeds)` seeds (5 in the example CLI usage). Since each
    environment fires exactly one blackout window per episode at a fixed
    position, any single (Model, Episode) BlackoutReward data point is an
    average over just `len(seeds)` independent single-event samples --
    nowhere near enough to distinguish a genuine difference in
    blackout-handling between models from seed-to-seed noise in
    initialization, exploration, and the stochastic environment.

    This function instead pools every blackout occurrence across the last
    `last_n_episodes` episodes of EVERY seed into one flat sample
    (n = len(seeds) * last_n_episodes, typically 50-100x larger), and
    reports mean, std, standard error, and the actual sample size used.
    Any claim of the form "model A handles blackout better than model B"
    should be checked against this pooled statistic (ideally with a
    two-sample test using the reported n), not against a single point on
    the per-episode curve.

    Args:
        df_all: per-seed, per-episode DataFrame as produced by train_single,
                 concatenated across seeds for one model.
        last_n_episodes: number of final episodes to pool per seed, to
                 approximate converged behavior rather than mixing in
                 early-training noise.

    Returns:
        dict with pooled n / mean / std / sem for BlackoutReward, PreBlackoutReward,
        RecoveryReward (the "fallout" window right after blackout ends), and
        PostRecoveryReward (steady state once recovery should be complete).
    """
    if df_all is None or df_all.empty:
        return {}

    max_ep = df_all["Episode"].max()
    window_df = df_all[df_all["Episode"] > max_ep - last_n_episodes]

    out = {}
    # BlackoutReward/PreBlackoutReward are the original backward-compatible
    # column names; Reward_recovery/Reward_post_recovery are the new
    # phase-broken-down columns covering the fallout window and steady state
    # after recovery. Pooling the fallout window specifically is the direct
    # answer to "does the agent actually recover, and how fast" -- a single
    # per-episode BlackoutReward number can't distinguish "handled the
    # blackout fine" from "handled it badly but recovered instantly" from
    # "recovery lingers for many steps afterward".
    cols_to_pool = {
        "BlackoutReward": "BlackoutReward",
        "PreBlackoutReward": "PreBlackoutReward",
        "RecoveryReward": "Reward_recovery",
        "PostRecoveryReward": "Reward_post_recovery",
    }
    for out_name, col in cols_to_pool.items():
        if col not in window_df.columns:
            continue
        vals = window_df[col].dropna().to_numpy()
        n = int(len(vals))
        mean = float(vals.mean()) if n > 0 else float("nan")
        std = float(vals.std(ddof=1)) if n > 1 else float("nan")
        sem = float(std / np.sqrt(n)) if n > 1 else float("nan")
        out[f"{out_name}_pooled_n"] = n
        out[f"{out_name}_pooled_mean"] = mean
        out[f"{out_name}_pooled_std"] = std
        out[f"{out_name}_pooled_sem"] = sem
    return out


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-episode metrics across seeds.

    Behaviour:
    - Groups by Model and Episode and computes mean and std for numeric metrics.
    - Flattens the resulting MultiIndex columns into single-level names.

    Returns:
        agg: aggregated DataFrame with columns like 'Reward_mean', 'Reward_std', etc.
    """
    # Identify metric columns to aggregate (exclude grouping keys).
    metrics = [c for c in df.columns if c not in ["Episode", "Model", "Seed"]]
    agg = (
        df.groupby(["Model", "Episode"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten MultiIndex columns into single strings for CSV-friendly output.
    agg.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns.values
    ]
    return agg


def main():
    """
    CLI entry point: parse arguments, run experiments for each agent and seed,
    save per-seed and aggregated CSVs, and produce summary plots.
    """
    parser = argparse.ArgumentParser(description="Unified training for AIF, PPO, and baselines.")

    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=list(ENVS.keys()),
        help="Environment name.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["AIF", "PPO", "EmotionPPO"],
        choices=list(AGENTS.keys()),
        help="Agents to train.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of episodes per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Output directory for logs and plots.",
    )

    # Ablation flags (apply only to the AIF model)
    parser.add_argument("--no-beta", action="store_true", help="Disable dynamic beta in AIF.")
    parser.add_argument("--no-persistence", action="store_true", help="Disable persistence transform in AIF.")
    parser.add_argument("--no-posterior", action="store_true", help="Disable posterior correction in AIF.")
    parser.add_argument("--no-efe", action="store_true", help="Disable EFE term in AIF.")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble in AIF.")
    parser.add_argument("--use-context", action="store_true", help="Enable slow-timescale context latent in AIF.")

    # Experimental controls
    parser.add_argument("--latency", type=int, default=0, help="Observation latency (steps).")
    parser.add_argument("--blackout", action="store_true", help="Enable sensor blackout wrapper.")
    parser.add_argument(
        "--recovery-window", type=int, default=20,
        help="Number of steps immediately after blackout ends to log as a separate "
             "'recovery' phase (captures fallout / recovery dynamics, not just the "
             "blackout window itself). Defaults to the same length as the default "
             "blackout duration."
    )

    parser.add_argument("--log_per_step", action="store_true", help="Save per‑step diagnostics for each episode (for debugging).")

    args = parser.parse_args()

    # Ensure output directory exists.
    os.makedirs(args.outdir, exist_ok=True)

    all_agent_agg = []
    # BUGFIX: collect a properly-powered pooled blackout statistic per agent,
    # instead of relying only on the per-(Model, Episode) curve whose n is
    # just len(seeds). See aggregate_blackout_pooled() docstring.
    all_pooled_rows = []

    # Iterate over requested agents and run per-seed experiments.
    for agent_name in args.agents:
        print(f"\n=== Training {agent_name} on {args.env} ===")
        per_seed_dfs = []

        # RESUMABILITY FIX: previously, per-seed results were only written to
        # disk after ALL seeds for this agent finished (`{env}_{agent}_all_seeds.csv`).
        # If the process was interrupted partway through the seed loop (e.g. a
        # Kaggle session hitting its runtime limit), every seed already
        # completed for this agent was silently lost and had to be re-run from
        # scratch. We now checkpoint each seed's result individually, the
        # moment it finishes, and skip re-running any seed whose checkpoint
        # already exists -- so a re-run of the exact same command picks up
        # from wherever it left off instead of restarting the whole agent.
        ckpt_dir = os.path.join(args.outdir, "seed_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        for seed in args.seeds:
            ckpt_path = os.path.join(ckpt_dir, f"{args.env}_{agent_name}_seed{seed}.csv")

            if os.path.exists(ckpt_path):
                df_seed = pd.read_csv(ckpt_path)
                # CORRECTNESS GUARD: a checkpoint is only valid if it actually
                # covers the requested number of episodes. Without this check,
                # resuming with a DIFFERENT --episodes value than the
                # interrupted run (e.g. 100 during a quick pass, 300 for the
                # final run) would silently treat a shorter, stale checkpoint
                # as "done" and skip training entirely -- producing results
                # that look complete but aren't.
                completed_episodes = int(df_seed["Episode"].max()) + 1 if len(df_seed) else 0
                if completed_episodes >= args.episodes:
                    print(f"  Seed {seed}: found valid checkpoint "
                          f"({completed_episodes} episodes >= requested {args.episodes}), "
                          f"skipping training -> {ckpt_path}")
                else:
                    print(f"  Seed {seed}: found STALE checkpoint ({completed_episodes} "
                          f"episodes, but {args.episodes} requested) -- retraining from "
                          f"scratch and overwriting checkpoint.")
                    df_seed = train_single(
                        env_name=args.env,
                        agent_name=agent_name,
                        episodes=args.episodes,
                        seed=seed,
                        args=args,
                        log_per_step=args.log_per_step,
                    )
                    df_seed.to_csv(ckpt_path, index=False)
                    print(f"  Seed {seed}: done, checkpoint saved -> {ckpt_path}")
            else:
                print(f"  Seed {seed}: no checkpoint found, training...")
                df_seed = train_single(
                    env_name=args.env,
                    agent_name=agent_name,
                    episodes=args.episodes,
                    seed=seed,
                    args=args,
                    log_per_step=args.log_per_step,
                )
                # Write the checkpoint immediately -- if the process is killed
                # on the very next seed, this seed's work is not lost.
                df_seed.to_csv(ckpt_path, index=False)
                print(f"  Seed {seed}: done, checkpoint saved -> {ckpt_path}")

            per_seed_dfs.append(df_seed)

        # Concatenate per-seed DataFrames for this agent.
        df_all = pd.concat(per_seed_dfs, ignore_index=True)

        # Save per-seed results to CSV for reproducibility and later analysis.
        per_seed_path = os.path.join(
            args.outdir, f"{args.env}_{agent_name}_all_seeds.csv"
        )
        df_all.to_csv(per_seed_path, index=False)

        # Aggregate across seeds and save aggregated CSV.
        df_agg = aggregate_results(df_all)
        agg_path = os.path.join(
            args.outdir, f"{args.env}_{agent_name}_agg.csv"
        )
        df_agg.to_csv(agg_path, index=False)

        all_agent_agg.append(df_agg)

        # BUGFIX: also compute the pooled, properly-powered blackout statistic
        # for this agent (pools across episodes and seeds rather than
        # comparing at n=len(seeds) per episode).
        pooled = aggregate_blackout_pooled(df_all, last_n_episodes=min(50, args.episodes))
        all_pooled_rows.append({"Model": agent_name, **pooled})

    # Combine aggregated results across agents for plotting and comparison.
    combined_agg = pd.concat(all_agent_agg, ignore_index=True)
    combined_path = os.path.join(args.outdir, f"{args.env}_combined_agg.csv")
    combined_agg.to_csv(combined_path, index=False)

    # Save the pooled blackout-robustness summary across agents. This is the
    # statistic to use when claiming one model handles blackout better than
    # another -- it has a much larger, honestly-reported sample size than the
    # per-episode curve in combined_agg.
    pooled_summary_df = pd.DataFrame(all_pooled_rows)
    pooled_summary_path = os.path.join(args.outdir, f"{args.env}_blackout_pooled_summary.csv")
    pooled_summary_df.to_csv(pooled_summary_path, index=False)
    print(f"Pooled blackout-robustness summary saved to {pooled_summary_path}")
    print(pooled_summary_df.to_string(index=False))

    # Plot reward comparison and diagnostics using provided plotting utilities.
    plot_path = os.path.join(args.outdir, f"{args.env}_reward_comparison.png")
    plot_reward_curves(combined_agg, plot_path)

    diag_dir = os.path.join(args.outdir, f"{args.env}_diagnostics")
    plot_diagnostics(combined_agg, diag_dir)

    print(f"\nAll done. Aggregated results saved to {combined_path}")
    print(f"Reward comparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()
