"""
run_all_experiments.py

Purpose
-------
train_all.py runs ONE environment per invocation (--env is a single required
choice). There was previously no driver that loops over every environment,
runs every agent, and combines results into a single cross-scenario
comparison. This script is that driver.

What it does
------------
For every environment in ENVS and both blackout conditions (off/on), it
shells out to train_all.py with the requested agents/seeds/episodes, then
concatenates every resulting `{env}_{blackout}_combined_agg.csv` and
`{env}_{blackout}_blackout_pooled_summary.csv` into two master CSVs:

    <outdir>/ALL_SCENARIOS_combined_agg.csv
    <outdir>/ALL_SCENARIOS_blackout_pooled_summary.csv

each tagged with `Environment` and `Blackout` columns, so you can filter/
compare across models, seeds, scenarios, and blackout condition all at once
in a single dataframe.

Resumability
------------
Both this driver and train_all.py are checkpointed, so re-running the exact
same command after an interruption (e.g. a Kaggle session hitting its runtime
limit) resumes instead of restarting:
  - Fully-completed (environment, blackout) combos are detected up front and
    skipped entirely (no subprocess re-invoked).
  - Partially-completed combos fall through to train_all.py, which itself
    skips any individual seed that already has a checkpoint under
    `<run_outdir>/seed_checkpoints/`, and only trains the remaining ones.
No manual bookkeeping is required -- just re-run the same command again.

Usage
-----
    python run_all_experiments.py \\
        --agents AIF PPO EmotionPPO DAIF PCRL FEAC DAIFC \\
        --seeds 0 1 2 3 4 \\
        --episodes 300 \\
        --outdir results_all

Scale down for a quicker/Kaggle-session-friendly pass, e.g.:
    python run_all_experiments.py --seeds 0 1 2 --episodes 150 --outdir results_quick

Notes
-----
- This literally invokes `python train_all.py ...` as a subprocess once per
  (environment, blackout-condition) pair, so all of train_all.py's existing
  behavior (per-seed CSVs, per-agent aggregation, pooled blackout/recovery
  stats, reward/diagnostic plots) is preserved and simply run 18 times
  (9 environments x 2 blackout conditions) instead of once.
- Total runs = len(ENVS) * 2 * len(agents) * len(seeds) episodes-worth of
  training. With the full default agent/seed list (7 agents x 5 seeds) across
  9 environments x 2 blackout conditions, that's 630 independent training
  runs. Scale --seeds/--episodes down first if you're time-constrained
  (e.g. a Kaggle session), then scale back up for a final reported run.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

# Keep this in sync with train_all.py's ENVS registry.
ALL_ENVS = [
    "AffectiveTutor",
    "ConflictResolution",
    "EmotionExploration",
    "HITL_CoCreation",
    "LongHaulMission",
    "ResourceGathering",
    "SocialNavigation",
    "SensorBlackout",
    "Gaslighting",
    "TransferEntropy",
]

ALL_AGENTS = ["AIF", "PPO", "EmotionPPO", "DAIF", "PCRL", "FEAC", "DAIFC"]


def run_one(env_name, blackout, args):
    """Invoke train_all.py once for a given environment and blackout condition.

    RESUMABILITY: if this (env, blackout) combo was already fully completed by
    a previous invocation (its combined_agg.csv and pooled_summary.csv both
    already exist on disk), skip re-invoking train_all.py entirely rather than
    paying subprocess/import overhead just to re-confirm already-finished work.
    Partially-completed combos (interrupted mid-seed-loop) are NOT skipped
    here -- they fall through to train_all.py, which resumes efficiently on
    its own via per-seed checkpoints (see train_all.py's RESUMABILITY FIX),
    only re-training whichever seeds don't yet have a checkpoint.
    """
    tag = "blackout" if blackout else "clean"
    run_outdir = os.path.join(args.outdir, f"{env_name}_{tag}")
    os.makedirs(run_outdir, exist_ok=True)

    combined_path = os.path.join(run_outdir, f"{env_name}_combined_agg.csv")
    pooled_path = os.path.join(run_outdir, f"{env_name}_blackout_pooled_summary.csv")

    if os.path.exists(combined_path) and os.path.exists(pooled_path):
        # CORRECTNESS GUARD: only treat this as "fully done" if the existing
        # results actually cover the requested number of episodes -- otherwise
        # a completed quick pass (e.g. --episodes 100) would be mistaken for a
        # finished full-scale run (--episodes 300) on a later resume.
        try:
            existing = pd.read_csv(combined_path)
            completed_episodes = int(existing["Episode"].max()) + 1 if len(existing) else 0
        except Exception:
            completed_episodes = 0

        if completed_episodes >= args.episodes:
            print(f"\nSkipping env={env_name} blackout={blackout}: already fully completed "
                  f"({completed_episodes} episodes >= requested {args.episodes}) -- "
                  f"found {combined_path} and {pooled_path}.")
            return combined_path, pooled_path
        else:
            print(f"\nenv={env_name} blackout={blackout}: existing results only cover "
                  f"{completed_episodes} episodes but {args.episodes} were requested -- "
                  f"not skipping; train_all.py will detect the mismatch per-seed and "
                  f"retrain (from scratch, not incrementally) any seed whose checkpoint "
                  f"doesn't cover enough episodes.")

    cmd = [
        sys.executable, "train_all.py",
        "--env", env_name,
        "--agents", *args.agents,
        "--episodes", str(args.episodes),
        "--seeds", *[str(s) for s in args.seeds],
        "--outdir", run_outdir,
        "--recovery-window", str(args.recovery_window),
    ]
    if blackout:
        cmd.append("--blackout")

    print(f"\n{'='*70}\nRunning: env={env_name} blackout={blackout}\n{'='*70}")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: run failed for env={env_name} blackout={blackout} "
              f"(exit code {result.returncode}) -- continuing with remaining runs. "
              f"Any seeds that did complete before the failure are checkpointed "
              f"under {run_outdir}/seed_checkpoints/ and will be reused on the "
              f"next re-run instead of being retrained.")
        return None, None

    return (
        combined_path if os.path.exists(combined_path) else None,
        pooled_path if os.path.exists(pooled_path) else None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run every environment x every agent x every seed, with and without "
                    "blackout, and produce master cross-scenario comparison CSVs."
    )
    parser.add_argument("--envs", type=str, nargs="+", default=ALL_ENVS, choices=ALL_ENVS,
                        help="Environments to run (default: all 9).")
    parser.add_argument("--agents", type=str, nargs="+", default=ALL_AGENTS, choices=ALL_AGENTS,
                        help="Agents to run (default: all 7).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds to run (default: 0-4).")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Episodes per seed (default: 300).")
    parser.add_argument("--recovery-window", type=int, default=20,
                        help="Recovery/fallout window length in steps (default: 20).")
    parser.add_argument("--blackout-modes", type=str, nargs="+", default=["clean", "blackout"],
                        choices=["clean", "blackout"],
                        help="Which blackout conditions to run (default: both).")
    parser.add_argument("--outdir", type=str, default="results_all",
                        help="Top-level output directory.")
    parser.add_argument(
        "--post-combo-hook", type=str, default=None,
        help="Optional shell command to run after EACH (environment, blackout) combo "
             "finishes (whether freshly run or skipped as already-done). Useful for "
             "syncing --outdir to remote/persistent storage (e.g. Google Drive) after "
             "every combo, rather than only at the very end -- so an interrupted "
             "session still has everything up to the last completed combo backed up. "
             "The command is run via the shell with cwd set to the original working "
             "directory (not --outdir), so use paths accordingly."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    combined_frames = []
    pooled_frames = []

    for env_name in args.envs:
        for mode in args.blackout_modes:
            blackout = (mode == "blackout")
            combined_path, pooled_path = run_one(env_name, blackout, args)

            if combined_path is not None:
                df = pd.read_csv(combined_path)
                df["Environment"] = env_name
                df["Blackout"] = blackout
                combined_frames.append(df)

            if pooled_path is not None:
                df = pd.read_csv(pooled_path)
                df["Environment"] = env_name
                df["Blackout"] = blackout
                pooled_frames.append(df)

            if args.post_combo_hook:
                print(f"Running post-combo hook after env={env_name} blackout={blackout}: "
                      f"{args.post_combo_hook}")
                hook_result = subprocess.run(args.post_combo_hook, shell=True)
                if hook_result.returncode != 0:
                    print(f"WARNING: post-combo hook exited with code {hook_result.returncode} "
                          f"-- continuing anyway (this should sync/back up results, not gate them).")

    if combined_frames:
        all_combined = pd.concat(combined_frames, ignore_index=True)
        all_combined_path = os.path.join(args.outdir, "ALL_SCENARIOS_combined_agg.csv")
        all_combined.to_csv(all_combined_path, index=False)
        print(f"\nMaster cross-scenario per-episode comparison saved to {all_combined_path}")
    else:
        print("\nWARNING: no combined_agg results were produced -- check the run logs above.")

    if pooled_frames:
        all_pooled = pd.concat(pooled_frames, ignore_index=True)
        all_pooled_path = os.path.join(args.outdir, "ALL_SCENARIOS_blackout_pooled_summary.csv")
        all_pooled.to_csv(all_pooled_path, index=False)
        print(f"Master cross-scenario pooled blackout/recovery summary saved to {all_pooled_path}")

        # Print a compact, human-readable summary table.
        display_cols = [c for c in [
            "Environment", "Blackout", "Model",
            "BlackoutReward_pooled_mean", "BlackoutReward_pooled_sem", "BlackoutReward_pooled_n",
            "RecoveryReward_pooled_mean", "RecoveryReward_pooled_sem", "RecoveryReward_pooled_n",
        ] if c in all_pooled.columns]
        if display_cols:
            print("\nSummary (blackout + recovery/fallout, pooled across seeds and episodes):")
            print(all_pooled[display_cols].to_string(index=False))
    else:
        print("WARNING: no pooled blackout summaries were produced -- check the run logs above.")

    print(f"\nAll done. Full results tree under: {args.outdir}")


if __name__ == "__main__":
    main()
