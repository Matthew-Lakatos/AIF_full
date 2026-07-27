# =============================================================================
# Kaggle script: run the full Active Inference experiment suite
# (no Google Drive, codebase pulled from GitHub, output local only)
# =============================================================================
# What this does:
#   1. Clones the codebase directly from GitHub (no dataset upload needed).
#   2. Installs dependencies.
#   3. Runs run_all_experiments.py, looping over every environment (10),
#      every agent (7), every seed, and both blackout conditions (off/on).
#   4. Writes everything to a single local directory, /kaggle/working/results_all
#      -- nothing is uploaded anywhere. That directory IS the output: when you
#      commit/save the notebook, Kaggle publishes it as this version's Output,
#      which is what you download or point a later session at.
#
# RESUMABILITY (no Drive involved -- this is Kaggle's own mechanism):
#   train_all.py checkpoints each (env, agent, seed) run the moment it
#   finishes, and skips retraining anything whose checkpoint already covers
#   the requested episode count. run_all_experiments.py additionally skips
#   any (environment, blackout) combo that's already fully completed.
#   So if a session is interrupted: commit the notebook (publishes
#   /kaggle/working as that version's Output), start a new session, add the
#   previous version's OWN output as a data source (Add Data -> Notebook
#   Output -> this notebook -> previous version), set RESUME_FROM_PATH below
#   to wherever that shows up under /kaggle/input/, and re-run with the same
#   QUICK_MODE/seeds/episodes. That's it -- no external service, no Drive.
#   Leave RESUME_FROM_PATH as None for a first, fresh run.
#
# SETUP REQUIRED BEFORE RUNNING:
#   - Set GITHUB_REPO_URL below to your repo.
#   - If the repo is private, either:
#       (a) set GITHUB_TOKEN via a Kaggle Secret and use the token-embedded
#           clone URL (see Cell 1), or
#       (b) make the repo public / use a deploy key -- your call.
#   - Settings -> Internet -> ON (required for both the clone and pip installs).
# =============================================================================

import os
import glob
import shutil
import subprocess
import sys
import time

# -----------------------------------------------------------------------------
# Cell 0: guard against accidentally running this whole script more than once
# at the same time in this session.
#
# BUG REPORT #10: a prior run's log showed the entire pipeline -- including
# the initial git clone -- executing twice concurrently (three near-
# simultaneous "Cloning into ..." events, every training block duplicated to
# the tenth of a second, byte-identical pooled statistics between the two
# copies). That doubles GPU-hour spend for nothing. This can't stop Kaggle
# from spinning up a genuinely separate second session/container, but it
# WILL catch the more common cause: re-running this cell (or the whole
# notebook) without restarting the kernel while a previous run is still live,
# since both would share this session's /kaggle/working filesystem.
#
# If you hit the RuntimeError below and you're sure nothing else is actually
# running, just delete the lock file it names and re-run.
# -----------------------------------------------------------------------------
_LOCK_PATH = "/kaggle/working/.aif_pipeline.lock"
_LOCK_STALE_AFTER_SECONDS = 6 * 60 * 60  # long enough to cover one real run

if os.path.exists(_LOCK_PATH):
    _lock_age = time.time() - os.path.getmtime(_LOCK_PATH)
    if _lock_age < _LOCK_STALE_AFTER_SECONDS:
        raise RuntimeError(
            f"{_LOCK_PATH} exists and is only {_lock_age:.0f}s old -- this pipeline "
            f"may already be running in this session (re-running this cell without "
            f"restarting the kernel, or committing the notebook while an interactive "
            f"run is still in progress, are common causes; see bug report #10). "
            f"If you're certain nothing else is actually running, delete this file "
            f"and re-run."
        )
    else:
        print(f"Found a stale lock file ({_lock_age:.0f}s old) -- removing and continuing.")
        os.remove(_LOCK_PATH)

os.makedirs("/kaggle/working", exist_ok=True)
with open(_LOCK_PATH, "w") as _f:
    _f.write(f"pid={os.getpid()} started={time.ctime()}\n")

# -----------------------------------------------------------------------------
# Cell 1: clone the codebase from GitHub
# -----------------------------------------------------------------------------
GITHUB_REPO_URL = "https://github.com/Matthew-Lakatos/AIF_full.git"  # <-- set this
WORKDIR = "/kaggle/working/aif_project"
OUTDIR = os.path.join(WORKDIR, "results_all")  # the ONLY place this script writes to

# If the repo is PRIVATE, uncomment the next few lines instead of the plain
# clone below, and add a Kaggle Secret named GITHUB_TOKEN (a GitHub Personal
# Access Token with repo read access):
#
# from kaggle_secrets import UserSecretsClient
# token = UserSecretsClient().get_secret("GITHUB_TOKEN")
# auth_url = GITHUB_REPO_URL.replace("https://", f"https://{token}@")
# clone_url = auth_url

clone_url = GITHUB_REPO_URL  # plain clone, assumes a public repo

if os.path.exists(WORKDIR):
    shutil.rmtree(WORKDIR)

result = subprocess.run(["git", "clone", "--depth", "1", clone_url, WORKDIR])
if result.returncode != 0:
    raise RuntimeError(
        "git clone failed. Check GITHUB_REPO_URL, that Internet is ON in notebook "
        "Settings, and (if private) that GITHUB_TOKEN is set correctly."
    )

print(f"Cloned {GITHUB_REPO_URL} -> {WORKDIR}")
os.chdir(WORKDIR)
print(os.listdir("."))

# -----------------------------------------------------------------------------
# Cell 2: restore a previous session's progress, if resuming (optional)
# -----------------------------------------------------------------------------
# Set this to a previous session's carried-forward output once you've added it
# as a data source (see RESUMABILITY above), e.g.:
#   RESUME_FROM_PATH = "/kaggle/input/my-notebook-slug/results_all"
# Leave as None for a first, fresh run -- this does NOT touch Google Drive or
# any external service, it only looks at other Kaggle inputs attached to
# this notebook.
RESUME_FROM_PATH = None

if RESUME_FROM_PATH is None:
    auto_candidates = glob.glob("/kaggle/input/*/results_all")
    if auto_candidates:
        RESUME_FROM_PATH = auto_candidates[0]
        print(f"Auto-detected a previous session's results at {RESUME_FROM_PATH}. "
              f"Set RESUME_FROM_PATH = None explicitly if this is wrong and you "
              f"want a fresh run instead.")

if RESUME_FROM_PATH is not None:
    if os.path.exists(RESUME_FROM_PATH):
        if os.path.exists(OUTDIR):
            shutil.rmtree(OUTDIR)
        shutil.copytree(RESUME_FROM_PATH, OUTDIR)
        n_ckpts = len(glob.glob(os.path.join(OUTDIR, "**", "seed_checkpoints", "*.csv"),
                                 recursive=True))
        print(f"Restored previous results from {RESUME_FROM_PATH} to {OUTDIR} "
              f"({n_ckpts} seed checkpoints found -- these will be skipped, not retrained).")
    else:
        print(f"WARNING: RESUME_FROM_PATH={RESUME_FROM_PATH} does not exist. Starting fresh.")
else:
    print("No RESUME_FROM_PATH set and none auto-detected -- starting a fresh run.")

# -----------------------------------------------------------------------------
# Cell 3: install dependencies
# -----------------------------------------------------------------------------
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "scipy", "seaborn"],
    check=False,
)

# -----------------------------------------------------------------------------
# Cell 4: sanity-check import before burning session time on a broken setup
# -----------------------------------------------------------------------------
sanity = subprocess.run(
    [sys.executable, "-c", "import torch, numpy, pandas, scipy, matplotlib; "
                           "from models.AIF import ActiveInferenceAgent; "
                           "print('imports OK, torch', torch.__version__)"],
    cwd=WORKDIR,
)
if sanity.returncode != 0:
    raise RuntimeError("Sanity import failed -- fix this before running the full sweep.")

# -----------------------------------------------------------------------------
# Cell 5: configure scale and run the full experiment suite
# -----------------------------------------------------------------------------
# Keep these THE SAME across every session used to resume this particular
# sweep -- only change them when deliberately starting a new/different sweep.
QUICK_MODE = True  # flip to False for a full-scale run

if QUICK_MODE:
    seeds = ["0", "1", "2"]
    episodes = "100"
else:
    seeds = ["0", "1", "2", "3", "4"]
    episodes = "300"

cmd = [
    sys.executable, "run_all_experiments.py",
    "--agents", "AIF", "PPO", "EmotionPPO", "DAIF", "PCRL", "FEAC", "DAIFC",
    "--seeds", *seeds,
    "--episodes", episodes,
    "--recovery-window", "20",
    "--blackout-modes", "clean", "blackout",
    "--outdir", OUTDIR,
]
print("Running:", " ".join(cmd))
print(f"All output goes to {OUTDIR} only -- nothing is uploaded anywhere during the run. "
      f"If this session is killed mid-way, whatever's on disk under {OUTDIR} is what "
      f"survives (per-seed checkpoints included) once you commit the notebook.")

result = subprocess.run(cmd, cwd=WORKDIR)
if result.returncode != 0:
    print(f"run_all_experiments.py exited with code {result.returncode}. "
          f"Partial results (and their checkpoints) still exist under {OUTDIR}.")

# -----------------------------------------------------------------------------
# Cell 6: quick look at the master comparison tables and progress report
# -----------------------------------------------------------------------------
import pandas as pd

pooled_path = os.path.join(OUTDIR, "ALL_SCENARIOS_blackout_pooled_summary.csv")
combined_path = os.path.join(OUTDIR, "ALL_SCENARIOS_combined_agg.csv")

if os.path.exists(pooled_path):
    pooled_df = pd.read_csv(pooled_path)
    print("\n=== Blackout/recovery (fallout) summary, pooled across seeds+episodes ===")
    display_cols = [c for c in [
        "Environment", "Blackout", "Model",
        "BlackoutReward_pooled_mean", "BlackoutReward_pooled_sem", "BlackoutReward_pooled_n",
        "RecoveryReward_pooled_mean", "RecoveryReward_pooled_sem", "RecoveryReward_pooled_n",
    ] if c in pooled_df.columns]
    print(pooled_df[display_cols].to_string(index=False))
else:
    print(f"No pooled summary yet at {pooled_path} -- check Cell 5's output above.")

if os.path.exists(combined_path):
    combined_df = pd.read_csv(combined_path)
    print(f"\nFull per-episode, per-scenario comparison table so far: "
          f"{combined_df.shape[0]} rows, {combined_df.shape[1]} columns -> {combined_path}")

all_combos = [(e, bo) for e in [
    "AffectiveTutor", "ConflictResolution", "EmotionExploration", "HITL_CoCreation",
    "LongHaulMission", "ResourceGathering", "SocialNavigation", "SensorBlackout",
    "Gaslighting", "TransferEntropy",
] for bo in ["clean", "blackout"]]
done_combos = [
    (e, bo) for (e, bo) in all_combos
    if os.path.exists(os.path.join(OUTDIR, f"{e}_{bo}", f"{e}_combined_agg.csv"))
]
print(f"\nProgress: {len(done_combos)}/{len(all_combos)} (environment, blackout-condition) "
      f"combos fully completed.")
if len(done_combos) < len(all_combos):
    print(f"Remaining: {[c for c in all_combos if c not in done_combos]}")
    print("To continue: commit this notebook, start a new session, add this notebook's "
          "own previous-version output as a data source, set RESUME_FROM_PATH to it, "
          "and re-run with the same QUICK_MODE/seeds/episodes.")
else:
    print("All combos complete.")

# -----------------------------------------------------------------------------
# Cell 7: zip the results for convenient single-file download (still local --
# this just archives /kaggle/working/results_all into /kaggle/working, no
# upload of any kind)
# -----------------------------------------------------------------------------
if os.path.exists(OUTDIR):
    shutil.make_archive("/kaggle/working/results_all", "zip", OUTDIR)
    print("\nZipped to /kaggle/working/results_all.zip (local only) -- both the folder "
          "and the zip will appear in this notebook's Output tab once committed.")
else:
    print(f"Nothing to zip -- {OUTDIR} does not exist.")

# -----------------------------------------------------------------------------
# Cell 8: release the double-execution lock acquired in Cell 0
# -----------------------------------------------------------------------------
try:
    os.remove(_LOCK_PATH)
except OSError:
    pass
