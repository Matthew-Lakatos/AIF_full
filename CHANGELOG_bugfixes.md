# Changelog — fixes applied from the bug report

This documents every change made in response to `AIF_codebase_bug_report.md`,
in the same order/numbering as that report. Every item below is one of:

- **Fixed** — behavior changed to match the code's own stated intent; nothing
  that was working on purpose was touched.
- **Documented / tagged, not changed** — the underlying design was left alone
  (changing it would be a research-design decision, not a bug fix), but the
  ambiguity is now self-documenting in the code and in the output data.
- **Mitigated** — can't be fixed from inside this repo (it's an external/
  infrastructure issue), but a guard was added to catch the common case.

Every edited file still compiles (`python3 -m py_compile`), and the two fixes
that don't depend on PyTorch (the statistics fix and the plotting fix) were
empirically re-run against synthetic data to confirm they do what's claimed
below — see "Verified" notes. Everything that depends on PyTorch (all the
gradient-routing fixes) could not be executed in the environment this fix was
made in (no GPU/torch available there) — see "How to verify" at the bottom
for exact commands to run yourself.

---

## #1 — DAIF/PCRL/FEAC/DAIFC never see the environment's reward
**Status: Documented / tagged, not changed.**

This is a research-design property, not a bug — changing what these agents
optimize is a substantive experimental decision that's yours to make, not
something to silently change while "fixing bugs." What *was* fixed: the fact
was invisible unless you read the source. Now:

- `models/AIF.py`, `models/PPO_standard.py`, and all four
  `models/aif_baselines/*.py` classes each carry two new class attributes:
  `USES_ENV_REWARD` (bool) and `REWARD_SOURCE` (human-readable string).
  `EmotionModulatedPPO` inherits `PPOActorCritic`'s values unchanged.
- `train_all.py`'s `main()` now reads these off the agent class (no need to
  instantiate one) and adds `OptimizesTaskReward` / `RewardSource` columns to
  every row of `{env}_blackout_pooled_summary.csv` — right next to the reward
  numbers themselves, not buried in a separate file.
- No agent's training objective changed. No column was removed or renamed.

If you *do* want a reward-aware ablation of the pseudo-reward baselines later,
that's a real design decision (e.g. `target = reward + gamma * v_next`
instead of `target = pseudo_reward + gamma * v_next`) — happy to help with it
once you've decided how you want to frame it, but I didn't make that call for
you here.

---

## #2 — AIF's policy gradient (the log warning)
**Status: The two confirmed contributing bugs underneath it are fixed (#3,
#5); whether this fully resolves the symptom needs a live run to confirm —
see "How to verify".**

No single line was silently "swallowing" the gradient that I could find by
static reading (rsample usage, the ensemble, world-model calls, all checked
out). What I did find, fix, and can defend confidently, is #3 and #5 below —
both directly relevant to this warning's mechanism. The diagnostic itself
(`AIF.py`'s `update()`) is unchanged in the policy-gradient check (it still
warns exactly if `policy_grad_norm == 0.0`); only the (broken) world-model
check next to it was removed — see #5.

---

## #3 — EFE planner's slow-context always reset to zero
**Status: Fixed.**

- `planners/efe_planner.py`: `EFEPlanner.__call__` now takes an explicit
  `s_start` parameter. The old `hasattr(world_model, "s")` lookup (which was
  always `False` against the real `RSSMWorldModel`, silently falling back to
  a zero tensor every call) is gone.
- `models/AIF.py`: the caller now passes `s_start=s_next` (the agent's real,
  just-updated slow context) through to the planner, via the new
  `_compute_efe_loss` method (see #5).

---

## #5 — World-model gradient leak from EFE + the diagnostic that couldn't catch it
**Status: Fixed (leak path), and the misleading diagnostic replaced.**

- `models/AIF.py`: extracted the freeze → plan → unfreeze block out of
  `update()` into a new method, `_compute_efe_loss(h_start, z_start,
  s_start)`. It now does `h_start.detach()`, `z_start.detach()`,
  `s_start.detach()` before calling the planner. Per the reasoning in the
  report: the policy-gradient path runs entirely through `policy_dist`
  (built from `self.policy_net`), not through these three tensors, so
  detaching them doesn't touch that path — it only severs the pre-existing
  graph edge these tensors carried back into the (nominally frozen) world
  model from *before* the freeze.
- Also wrapped the freeze/restore in `try`/`finally`, so a single failed
  planner call can't leave the world model permanently frozen for the rest
  of training.
- Removed the old `wm_grad_norm > 1e-8` runtime warning. As written, it was
  measured after the *combined* `total_loss.backward()` (VFE + EFE together),
  and VFE is *supposed to*, and does, produce non-zero world-model gradient
  — so that check could never actually isolate an EFE-specific leak; it
  would either cry wolf on every step or give false reassurance. A comment
  now explains this and points at the properly isolated version of the
  check, which lives in the fixed test (#8).
- Updated `models/AIF.py`'s module docstring, which previously asserted the
  (incorrect) claim that not detaching these tensors was necessary for the
  policy gradient.

---

## #6 — `TransitionEnsemble` kwarg parsing built the wrong shape
**Status: Fixed.**

- `models/AIF.py`: now constructs the ensemble with an explicit
  `input_dim=latent_dim + action_dim + slow_dim` instead of the previous
  `action_dim=action_dim`, which isn't a real parameter of
  `TransitionEnsemble.__init__` and was silently misinterpreted.
- `transition_ensemble.py`: removed the buggy fallback
  (`kwargs.get('input_dim', kwargs.get('action_dim', None))`) that aliased
  `action_dim` as `input_dim` for *any* caller, not just AIF's. If
  `input_dim` isn't given explicitly now, construction correctly falls
  through to lazy (build-on-first-forward-call) construction instead of
  guessing.
- Added `slow_dim=32` as an explicit constructor parameter on
  `ActiveInferenceAgent` (previously implicit via `RSSMWorldModel`'s own
  default, with no way for `AIF.py` to compute the ensemble's true input
  width from it). Default value unchanged, so no existing caller is
  affected — every call site in the repo uses keyword arguments only
  (verified by grep), so inserting a new parameter can't shift any
  positional argument either.

---

## #4 — Early-terminating environments and blackout survivorship bias
**Status: Fixed (now visible; the underlying environment behavior — ending
episodes early on failure — was left alone, since that's a legitimate
environment design, not a bug).**

- `train_all.py::train_single`: each episode's row now includes
  `EpisodeLength` (steps actually taken) and `ReachedBlackoutWindow` (1.0 /
  0.0 / NaN if blackout wasn't enabled for this run) — both plain numeric
  columns, safe for `aggregate_results()`'s existing `.agg(["mean","std"])`.
- `train_all.py::aggregate_blackout_pooled`: added
  `EarlyTermination_BeforeBlackout_rate` and `ReachedBlackoutWindow_n_episodes`
  to the pooled summary, computed from the above. An agent that fails the
  base task before ever reaching the blackout window now shows up as e.g.
  "73% of episodes never reached blackout" instead of a silent `NaN` that
  looks identical to "blackout wasn't configured for this run."
- **Verified** (pandas/numpy only, no torch needed): ran the fixed
  `aggregate_blackout_pooled` against synthetic data where one seed reaches
  the blackout window in only 1/5 of episodes and two seeds always reach it;
  it correctly reported `EarlyTermination_BeforeBlackout_rate = 0.2667`,
  matching the constructed ground truth exactly.

---

## #7 — Pooled SEM treated within-seed episodes as independent samples
**Status: Fixed (additively — nothing removed).**

- `train_all.py::aggregate_blackout_pooled`: added
  `{metric}_n_seeds`, `{metric}_between_seed_mean`, and
  `{metric}_between_seed_sem` for each pooled metric, computed by first
  averaging within each seed (yielding one number per seed — typically 3–5),
  then computing the mean/SEM across *those*. The original `_pooled_n` /
  `_pooled_mean` / `_pooled_std` / `_pooled_sem` columns are all still there,
  unchanged, for backward compatibility and as a legitimate descriptive
  statistic of this specific set of trained seeds.
- Use `_between_seed_sem`, not `_pooled_sem`, for anything of the form "would
  this reproduce with a new seed" (e.g. a significance claim).
- **Verified**: ran the fixed function against synthetic data (3 seeds, 60
  episodes each). Got `BlackoutReward_pooled_sem = 0.0207` (n=110, the old
  number) vs. `BlackoutReward_between_seed_sem = 0.0394` (n_seeds=3, the new
  number) — about 1.9x larger, correctly reflecting that there are really
  only 3 independent replicates, not 110.

---

## #8 — Gradient-routing unit test asserted the wrong invariant
**Status: Fixed.**

- `tests/test_overshoot_and_gradients.py::test_efe_policy_gradients_and_worldmodel_frozen`
  rewritten. It still checks (Part 1) that a full `agent.update()` call gives
  the policy non-zero gradient. It now ALSO (Part 2) reconstructs
  `h_prior`/`z_post`/`s_next` the same way `update()` does, calls the same
  `_compute_efe_loss()` helper `update()` now uses internally, and backwards
  *only* that loss (with grads zeroed first) — so any world-model gradient
  observed there can only have come from EFE, never from VFE. This is the
  test that would have caught the original `wm_grad_norm == 0.0` assertion
  being unsatisfiable in the first place.
- Also dropped two genuinely-unused local variables (`ensemble`, `slow_dim`)
  that the original test constructed but never referenced.

---

## #9 — `plot_diagnostics` DataFrame fragmentation
**Status: Fixed.**

- `plot_results.py::plot_diagnostics`: now sorts and copies the frame once,
  computes every `{metric}_smooth` series into a dict, and joins them all in
  a single `pd.concat(..., axis=1)` — instead of re-sorting the whole frame
  and assigning one column at a time, once per diagnostic metric (which is
  what fragmented the frame and triggered pandas' warning repeatedly).
- **Verified**: built a synthetic `combined_agg` shaped like the real
  pipeline's (208 columns, 101 diagnostic metrics, differently-populated per
  model, produced via the same `groupby(...).agg(["mean","std"])` + column-
  flatten pipeline `aggregate_results()` uses). The original code pattern
  raised exactly 101 `PerformanceWarning: DataFrame is highly fragmented`
  warnings on this data (one per metric — matching what you saw in the log).
  The fixed `plot_diagnostics` raised zero, on the identical data, and
  produced the same number of output PNGs either way.

---

## #10 — Kaggle log shows the whole script running twice
**Status: Mitigated (can't be fixed from inside this repo).**

- `kaggle_run_all_experiments.py`: added a lock file
  (`/kaggle/working/.aif_pipeline.lock`) written at the very start and
  removed at the very end. If the script is started again while a lock file
  younger than 6 hours already exists, it raises immediately with an
  explanation, instead of silently running the entire pipeline twice. A
  stale lock (older than 6 hours — long enough to cover one real run) is
  treated as leftover from a crashed run and cleared automatically.
- This can only catch double-execution that shares one Kaggle session's
  `/kaggle/working` filesystem (e.g. re-running the cell without restarting
  the kernel). It can't detect two genuinely separate Kaggle sessions/
  containers running concurrently — if that's what's actually happening,
  it's a Kaggle accelerator/session-settings question, not something fixable
  from inside the script.

---

## #11 — Minor items
**Status: Fixed / clarified, all individually small.**

- `envs/scenarios.py::TransferEntropy.preferred_obs`: now computes the goal
  from `self.steps + 1` instead of `self.steps`, matching what `step()`
  itself will use for this same upcoming transition (previously off by one
  right at the flip point).
- `tests/compare_blackout_ablations.py`: removed the dead
  `agent.world_model.lambda_slow = 0.0` line (that attribute is stored on
  `RSSMWorldModel` but never read anywhere — only `agent.lambda_slow`
  actually gates the persistence penalty in `AIF.update()`); the ablation
  still works via the line that does matter. Also corrected the
  `"fixed-average"` variant's docstring, which claimed it might disable
  posterior correction — it only ever replaces the precision controller.
- `models/PPO_standard.py`: added a docstring note that this is a per-step,
  batch-of-one PPO variant (not batched, no GAE) — a design choice, not a
  bug, but worth being explicit about wherever this is written up.
- The redundant `seed_mean` diagnostic column (harmless, just duplicates the
  existing `Seed` column) was left alone — genuinely cosmetic, not worth the
  churn of touching working aggregation code for it.

---

## Full list of files touched

```
models/AIF.py
planners/efe_planner.py
transition_ensemble.py
models/PPO_standard.py
models/aif_baselines/DAIF_Tschantz.py
models/aif_baselines/PCRL_Whittington.py
models/aif_baselines/FEAC_Friston.py
models/aif_baselines/DAIFC_Millidge.py
train_all.py
envs/scenarios.py
plot_results.py
tests/test_overshoot_and_gradients.py
tests/compare_blackout_ablations.py
kaggle_run_all_experiments.py
```

Not touched: `models/PPO_emotion_standard.py` (inherits the new class tags
from `PPOActorCritic` automatically, nothing else needed), `rssm_world_model.py`,
`precision_controller.py`, `utilities/*.py`, `sync_to_drive.py`,
`tests/test_agent_update_blackout.py`, `tests/test_blackout_behavior.py`,
`tests/test_rssm_ensemble_integration.py` — read carefully during the review,
no bugs found in them beyond what's already noted above.

Every file in the repo (and the standalone Kaggle script) passes
`python3 -m py_compile` after these changes.

---

## How to verify the PyTorch-dependent fixes (I couldn't run these myself —
## no GPU/torch in the environment this fix was made in)

```bash
# 1. The rewritten gradient-routing test -- this is the most direct check.
#    It should pass; if it doesn't, that's the next thing to dig into.
pytest tests/test_overshoot_and_gradients.py -v

# 2. Full existing test suite, to make sure nothing else regressed.
pytest tests/ -v

# 3. A quick real training run to see whether AIF's own reward improves
#    now relative to before (compare against your existing quick-mode logs).
python3 train_all.py --env AffectiveTutor --agents AIF --seeds 0 --episodes 100 \
    --outdir /tmp/aif_fix_check
#    Watch for: does "EFE did not produce gradients for policy parameters"
#    still appear? Does AIF's PreBlackoutReward move noticeably off ~0.2?
```
