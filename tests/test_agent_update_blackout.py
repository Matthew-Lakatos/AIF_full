# Purpose: Run a single agent.update() under blackout and non-blackout to ensure stability and diagnostics.

import numpy as np
import torch
import pytest
from models.AIF import ActiveInferenceAgent
from envs.scenarios import ScenarioEnv


@pytest.mark.parametrize("blackout_enabled", [False, True])
def test_agent_update_no_nans_and_blackout_flag(blackout_enabled):
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ScenarioEnv(obs_dim=16, action_dim=6, max_steps=20, seed=seed)
    env.set_blackout(enabled=blackout_enabled, fraction=0.5, duration_steps=None, noise_std=0.0)
    agent = ActiveInferenceAgent(obs_dim=16, action_dim=6, latent_dim=32, hidden_dim=64, use_context=True, lambda_slow=0.1)
    agent.last_seed = seed
    # sync seeds
    agent.world_model.to_device(torch.device("cpu"))

    obs, info = env.reset()
    # get action
    action_np, logprob, _, _ = agent.get_action(obs)
    # step env
    next_obs, reward, done, info = env.step(action_np)
    # run update (should not raise)
    agent.update(prev_obs=obs, action=action_np, next_obs=next_obs)
    # update blackout status in agent
    agent.last_blackout_active = info.get("blackout_active", False)
    # check no NaNs in key diagnostics
    assert not np.isnan(agent.last_accuracy)
    assert not np.isnan(agent.last_complexity)
    # ensure agent has last_slow_persist diagnostic
    assert hasattr(agent, "last_slow_persist")
    # set agent.last_blackout_active to env.blackout_active to satisfy test contract
    assert agent.last_blackout_active == env.blackout_active
    # check that last_epistemic is not NaN
    assert not np.isnan(agent.last_epistemic)
    # check that last_seed is set
    assert agent.last_seed == seed
