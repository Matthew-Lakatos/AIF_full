# Active Inference with Recurrent State-Space Models and Epistemic Planning

A unified framework for active inference agents that combine a recurrent world model (RSSM) with a slow latent context, dynamic precision (β) weighting, Expected Free Energy (EFE) planning, and a transition ensemble for epistemic uncertainty. This repository provides the code for experiments reported in my paper.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Agents & Environments](#agents--environments)
- [Running Experiments](#running-experiments)
  - [Basic Usage](#basic-usage)
  - [Ablation Flags](#ablation-flags)
  - [Per‑Step Logging](#per-step-logging)
- [Outputs & Analysis](#outputs--analysis)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Overview

I propose an active inference agent built on a **Recurrent State-Space Model (RSSM)** that learns a world model with fast (`z`) and slow (`s`) latents. The agent:

- Uses a **dynamic β** (precision) controller that weighs the KL complexity term based on prediction error and latent change.
- Plans actions via a short‑horizon **Expected Free Energy (EFE)** planner that balances risk (deviation from preferred observations) and epistemic value (ensemble variance).
- Employs a **transition ensemble** to estimate epistemic uncertainty.
- Includes an **overshoot loss** that forces multi‑step predictive consistency, gradually increasing the horizon during training.
- Regularises the slow latent with a **persistence penalty**.

The code implements the agent alongside several baselines: DAIF (Tschantz), FEAC (Friston), DAIFC (Millidge), PCRL (Whittington), PPO, and Emotion‑Modulated PPO. All agents are trained in a unified script with full reproducibility (seeds, deterministic flags).

---

## Installation

Clone the repository and install the required packages.

```bash
git clone https://github.com/Matthew-Lakatos/active-inference-research.git
cd active-inference-research
pip install -r requirements.txt
```
If you do not have a requirements.txt, you can install the core dependencies manually:

```bash
pip install torch numpy pandas matplotlib seaborn scipy einops torchmetrics hydra-core pytest
```
I recommend using a Python 3.8+ environment with PyTorch 2.x. For GPU training, ensure CUDA is available (the code will automatically use GPU if present).

## Agents & Environments
Agents
AIF – My proposed agent (RSSM + EFE + ensemble + dynamic β).

DAIF – Deep Active Inference (Tschantz‑style, no recurrence).

FEAC – Free Energy Actor‑Critic (Friston‑style).

DAIFC – Deep Active Inference Control (Millidge‑style).

PCRL – Predictive Coding RL (Whittington‑style).

PPO – Standard Proximal Policy Optimization.

EmotionPPO – PPO with surprise‑modulated entropy.

Environments
All environments inherit from ResearchEnvironment (see envs/research_envs.py). They expose a latent state and provide noisy observations. The available scenarios are:

SensorBlackout – Agent must control a latent variable while observations are masked mid‑episode.

LongHaulMission – Sparse reward, long‑horizon homeostatic task.

Gaslighting – Reward flips during specific intervals; tests epistemic trust.

ResourceGathering – Risk‑reward trade‑off.

AffectiveTutor, ConflictResolution, EmotionExploration, HITL_CoCreation, SocialNavigation, TransferEntropy.

Each environment provides a preferred_obs() method for the EFE risk term and supports configurable sensor blackout.

## Running Experiments
# Basic Usage
The main script is train_all.py. It takes command‑line arguments for environment, agents, number of episodes, seeds, and output directory.

```bash
python train_all.py \
  --env SensorBlackout \
  --agents AIF DAIF FEAC DAIFC PPO \
  --episodes 300 \
  --seeds 0 1 2 3 4 \
  --outdir results
```
This will run the specified agents for 5 seeds on the SensorBlackout environment, saving per‑seed CSV files and aggregated plots.

# Ablation Flags
The following flags apply only to the AIF agent:

Flag	Effect
--no-beta	Disable dynamic β; use fixed β = 1.0.
--no-efe	Disable Expected Free Energy planning.
--no-persistence	Disable slow‑latent persistence penalty.
--no-posterior	Disable posterior correction (use z_post directly).
--no-ensemble	Remove the transition ensemble (epistemic uncertainty).
--use-context	Enable the slow latent s (context).
Example: run AIF without EFE and without the ensemble:

```bash
python train_all.py \
  --env SensorBlackout \
  --agents AIF \
  --no-efe \
  --no-ensemble \
  --episodes 200 \
  --seeds 0 1 2 \
  --outdir results
```
# Per‑Step Logging
For debugging or detailed analysis, you can enable per‑step logging:

```bash
python train_all.py \
  --env SensorBlackout \
  --agents AIF \
  --episodes 5 \
  --seeds 0 \
  --outdir results \
  --log_per_step
```
This will generate a CSV for each episode containing step‑by‑step diagnostics (reward, VFE, accuracy, complexity, overshoot loss, etc.).

Other Options
--latency N – simulate observation delay (Dreamer‑style).

--blackout – enable sensor blackout (environment‑specific).

--outdir – output directory (default: results).

## Outputs & Analysis
For each combination of (environment, agent, seed), train_all.py saves:

<env>_<agent>_all_seeds.csv – raw per‑episode data for all seeds (all rows).

<env>_<agent>_agg.csv – aggregated data (mean ± std across seeds) per episode.

<env>_combined_agg.csv – combined aggregated data across agents.

<env>_reward_comparison.png – reward curves (mean ± std) for all agents.

<env>_diagnostics/ – diagnostic plots for each metric.

If --log_per_step is used, additional <env>_<agent>_seed<seed>_ep<ep>_per_step.csv files are created.

You can further analyse the CSVs with the provided plot_results.py functions:

```python
from plot_results import plot_reward_curves, plot_diagnostics
plot_reward_curves(combined_agg, "reward.png")
plot_diagnostics(combined_agg, "diagnostics/")
```
## Reproducibility
I set seeds for Python, NumPy, and PyTorch at the start of each run via utilities.seed.set_all_seeds. CuDNN is set to deterministic mode. To reproduce exact results, use the same software versions and hardware (CPU/GPU) – some GPU operations may still introduce minor nondeterminism, but the configuration maximises consistency.

The agent hyperparameters are hard‑coded in models/AIF.py and can be overridden via the constructor (see the file for the full list). Key defaults:

Latent dimension: 64

Hidden dimension: 128

Slow latent dimension: 32

Overshoot horizon starts at 1, linearly increases to 3 over 200 episodes.

Pseudo‑posterior variance floor: 1.0

Learning rate: 2e‑4

Citation
If you use this code in your research, please cite my paper:

```text
@article{lakatos2026active,
  title={ --- },
  author={Lakatos, Matthew},
  journal={...},
  year={2026}
}
```
For any questions, please open an issue on GitHub.
