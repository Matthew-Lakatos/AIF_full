"""
plot_results.py

Purpose
-------
Plotting utilities for experiment outputs. Provides:
- `plot_reward_curves`: plots mean reward with std shading across seeds for each model.
- `plot_diagnostics`: automatically detects diagnostic metrics in an aggregated
  DataFrame and produces per-metric plots across agents.

What is implemented
-------------------
- Smoothing helper `_smooth` using pandas rolling mean.
- `plot_reward_curves(agg_df, outpath, ...)`:
    - Expects aggregated DataFrame with columns: Model, Episode, Reward_mean, Reward_std.
    - Produces a single PNG comparing reward curves across agents.
- `plot_diagnostics(agg_df, outdir, ...)`:
    - Detects columns ending with `_mean` (excluding core columns) and plots each
      diagnostic across agents with optional std shading.
    - Saves one PNG per diagnostic into `outdir`.

Design choices and caveats
--------------------------
- Uses simple rolling mean smoothing (window default 10). This is appropriate
  for visualising trends but should not be used for statistical analysis.
- Expects aggregated DataFrame (mean/std per episode per model). If raw per-seed
  data is passed, aggregate first using the same aggregation used in the repo.
- Plotting uses seaborn/matplotlib defaults tuned for paper figures (dpi, size).
- `plot_diagnostics` looks for columns ending with `_mean`; ensure diagnostic
  names follow the `<name>_mean` / `<name>_std` convention.

Suggested experiments / variants
--------------------------------
- Use median and interquartile range instead of mean ± std for heavy-tailed metrics.
- Add command-line flags to control smoothing, color palettes, and figure size.
- Export CSVs of the smoothed series for reproducible figure regeneration.
"""

import os
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _smooth(series, window=10):
    """
    Smooth a pandas Series using a rolling mean.

    Args:
        series: pandas Series
        window: int, rolling window size

    Returns:
        pandas Series of smoothed values
    """
    return series.rolling(window=window, min_periods=1).mean()


def plot_reward_curves(
    agg_df: pd.DataFrame,
    outpath: str,
    smooth_window: int = 10,
    dpi: int = 300,
    title: Optional[str] = None,
):
    """
    Plot mean reward with std shading across seeds for each model.

    Expected input DataFrame columns:
      - 'Model' (str)
      - 'Episode' (int)
      - 'Reward_mean' (float)
      - 'Reward_std' (float)

    Args:
        agg_df: aggregated DataFrame (one row per Model x Episode with mean/std)
        outpath: path to save PNG
        smooth_window: rolling window for smoothing mean curve
        dpi: figure DPI
        title: optional figure title
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 6))

    # Work on a copy to avoid mutating caller's DataFrame
    df = agg_df.copy()
    df = df.sort_values(["Model", "Episode"])

    # Add a smoothed mean column per model for clearer trend visualization
    df["Reward_mean_smooth"] = (
        df.groupby("Model")["Reward_mean"]
        .transform(lambda x: _smooth(x, window=smooth_window))
    )

    models = df["Model"].unique()
    for model in models:
        sub = df[df["Model"] == model]
        episodes = sub["Episode"].values
        mean = sub["Reward_mean_smooth"].values
        std = sub["Reward_std"].values

        # Plot mean line and shaded std band
        plt.plot(episodes, mean, label=model, linewidth=2)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward (mean ± std)", fontsize=12)
    plt.title(title or "Reward Comparison Across Agents", fontsize=14, fontweight="bold")
    plt.legend(title="Agent")
    plt.tight_layout()

    # Ensure output directory exists and save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=dpi)
    plt.close()


# -------------------------------------------------------------
# Diagnostic plotting
# -------------------------------------------------------------

def plot_diagnostics(
    agg_df: pd.DataFrame,
    outdir: str,
    smooth_window: int = 10,
    dpi: int = 300,
):
    """
    Detect diagnostic metrics and generate plots.

    Behaviour:
      - Finds columns ending with '_mean' (excluding core metrics).
      - For each diagnostic, plots smoothed mean across episodes for each model.
      - If a corresponding '_std' column exists, plots shaded std band.

    Args:
        agg_df: aggregated DataFrame with columns like '<metric>_mean' and '<metric>_std'
        outdir: directory to save diagnostic PNGs
        smooth_window: smoothing window for mean curves
        dpi: figure DPI
    """
    os.makedirs(outdir, exist_ok=True)

    # Identify diagnostic metrics (exclude core columns)
    exclude = {"Episode", "Model", "Reward_mean", "Reward_std", "VFE_mean", "VFE_std"}
    # Select columns that look like aggregated diagnostics (end with '_mean')
    diag_metrics = [c for c in agg_df.columns if c not in exclude and c.endswith("_mean")]

    if len(diag_metrics) == 0:
        print("No diagnostics found to plot.")
        return

    # Helper to convert 'accuracy_mean' -> 'accuracy'
    clean_name = lambda m: m.replace("_mean", "")

    # One combined plot per diagnostic across agents
    for metric in diag_metrics:
        metric_name = clean_name(metric)
        std_col = metric.replace("_mean", "_std")

        sns.set_theme(style="whitegrid", context="paper")
        plt.figure(figsize=(10, 6))

        # Sort and smooth per model
        df = agg_df.sort_values(["Model", "Episode"])
        df[f"{metric}_smooth"] = (
            df.groupby("Model")[metric]
            .transform(lambda x: _smooth(x, window=smooth_window))
        )

        for model in df["Model"].unique():
            sub = df[df["Model"] == model]
            episodes = sub["Episode"].values
            mean = sub[f"{metric}_smooth"].values
            std = sub[std_col].values if std_col in sub else None

            plt.plot(episodes, mean, label=model, linewidth=2)
            if std is not None:
                plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)

        plt.xlabel("Episode", fontsize=12)
        plt.ylabel(f"{metric_name} (mean ± std)", fontsize=12)
        plt.title(f"{metric_name} Across Agents", fontsize=14, fontweight="bold")
        plt.legend(title="Agent")
        plt.tight_layout()

        outpath = os.path.join(outdir, f"diagnostic_{metric_name}.png")
        plt.savefig(outpath, dpi=dpi)
        plt.close()

    print(f"Diagnostic plots saved to {outdir}")
