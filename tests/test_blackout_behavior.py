# tests/test_blackout_behavior.py
# Purpose: Unit tests for blackout timing, masking behavior, and info diagnostics.

import numpy as np
import pytest
from envs.scenarios import ScenarioEnv


@pytest.mark.parametrize("max_steps, fraction, expected_start", [
    (10, 0.5, 5),
    (11, 0.5, 5),
    (8, 0.25, 2),
])
def test_blackout_start_timing(max_steps, fraction, expected_start):
    env = ScenarioEnv(obs_dim=4, action_dim=2, max_steps=max_steps, seed=42)
    env.set_blackout(enabled=True, fraction=fraction, duration_steps=None, noise_std=0.0)
    obs, info = env.reset()
    # steps start at 0; blackout should activate at expected_start
    for step in range(0, max_steps):
        obs, reward, done, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
        if step + 1 < expected_start:
            assert info["blackout_active"] is False
        else:
            assert info["blackout_active"] is True

def test_blackout_zero_vs_noisy_masking():
    env = ScenarioEnv(obs_dim=3, action_dim=1, max_steps=6, seed=123)
    # zero masking
    env.set_blackout(enabled=True, fraction=0.0, duration_steps=6, noise_std=0.0)
    obs, info = env.reset()
    assert info["blackout_active"] is True
    assert np.allclose(obs, 0.0)
    # noisy masking
    env.set_blackout(enabled=True, fraction=0.0, duration_steps=6, noise_std=0.5)
    env.set_seed(7)
    obs2, info2 = env.reset()
    assert info2["blackout_active"] is True
    assert not np.allclose(obs2, 0.0)
    # reproducibility with seed
    env.set_seed(7)
    obs3, _ = env.reset()
    assert np.allclose(obs2, obs3)

def test_set_seed_reproducible():
    env = ScenarioEnv(obs_dim=2, action_dim=1, max_steps=5, seed=0)
    env.set_blackout(enabled=False)
    env.set_seed(99)
    o1, _ = env.reset()
    env.set_seed(99)
    o2, _ = env.reset()
    assert np.allclose(o1, o2)
