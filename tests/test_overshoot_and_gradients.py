# PyTest suite: gradient routing, overshoot sanity, device check, end-to-end smoke run.
import os
import tempfile
import torch
import pytest
import numpy as np

from models.AIF import ActiveInferenceAgent
from rssm_world_model import RSSMWorldModel
from transition_ensemble import TransitionEnsemble

# Helper: set deterministic seeds for tests
def set_test_seed(seed=0):
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_overshoot_near_zero_and_finite(device):
    """
    Overshoot should be finite and near-zero when target_z equals posterior samples
    for a short horizon and trivial actions.
    """
    set_test_seed(0)
    device = torch.device(device)
    # small dims for speed
    obs_dim = 6
    action_dim = 2
    latent_dim = 8
    slow_dim = 4

    ensemble = TransitionEnsemble(latent_dim=latent_dim, action_dim=action_dim)
    rssm = RSSMWorldModel(obs_dim, action_dim, latent_dim=latent_dim, hidden_dim=16,
                          slow_dim=slow_dim, ensemble=ensemble, device=device)
    rssm.to_device(device)

    # init state
    h, z, s, step = rssm.init_state(batch=1)
    h = h.to(device); z = z.to(device); s = s.to(device)

    # create a fake action sequence and target_z equal to posterior samples
    T = 3
    actions = torch.zeros(T, 1, action_dim, device=device)
    # produce target_z by calling observe repeatedly
    target_z_list = []
    for t in range(T):
        # fake obs
        obs = torch.randn(1, obs_dim, device=device)
        z_post, mu_q, logvar_q, s, *_ = rssm.observe(h, obs, s_prev=s, update_s=True, step=step)
        target_z_list.append(z_post.detach())
        # step transition to update h for next iteration
        h, z_prior, mu_p, logvar_p = rssm.transition(h, z_post, actions[t], s=s)
    target_z = torch.stack(target_z_list, dim=0)  # [T, B, D]

    loss = rssm.overshoot_loss(h.detach(), z.detach(), actions, target_z, s=s.detach(), horizon=3)
    assert torch.isfinite(loss).all(), "Overshoot loss produced non-finite values"
    # Expect small loss because target_z came from the model's own posterior (not exactly zero but small)
    assert float(loss.item()) < 20.0, f"Overshoot loss unexpectedly large: {loss.item()}"

def test_efe_policy_gradients_and_worldmodel_frozen():
    """
    Ensure EFE produces gradients for policy parameters, and that EFE
    *specifically* -- isolated from VFE's own gradient contribution --
    produces no gradient for world_model parameters.

    FIX (bug report #8): this test previously checked `wm_grad_norm == 0.0`
    after a full `agent.update()` call, which combines VFE's backward pass
    (whose accuracy_loss/complexity_loss are SUPPOSED to, and do, produce
    non-zero world-model gradient -- that's the entire point of VFE) with
    EFE's. That assertion could only ever legitimately pass if VFE itself
    were ALSO failing to reach the world model, which would be a much bigger
    problem than the one this test claims to check for -- it was measuring
    the wrong quantity. To actually verify "EFE alone doesn't leak into the
    world model" (see bug report #5), we measure EFE's gradient contribution
    in isolation, via a separate backward pass on `_compute_efe_loss`'s
    output alone -- the same helper `AIF.update()` itself now calls.
    """
    set_test_seed(1)
    device = torch.device("cpu")
    obs_dim = 6
    action_dim = 2
    latent_dim = 8

    agent = ActiveInferenceAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=32,
        efe_horizon=2,
        lambda_epistemic=0.1,
        lambda_policy=0.5,
        lambda_slow=0.1,
        overshoot_horizon=1,
        use_dynamic_beta=True,
        use_persistence=True,
        use_posterior_correction=True,
        use_efe=True,
        use_context=False,
        preferred_obs=None,
    )
    agent.to(device)
    # small fake observation and action
    obs = np.zeros(obs_dim, dtype=np.float32)
    next_obs = np.zeros(obs_dim, dtype=np.float32)
    action = np.zeros(action_dim, dtype=np.float32)

    # --- Part 1: a full update() call should give the policy a non-zero
    # gradient (this is the check the original test also made). ---
    for p in agent.policy_net.parameters():
        if p.grad is not None:
            p.grad.zero_()
    if agent.log_std.grad is not None:
        agent.log_std.grad.zero_()

    agent.update(obs, action, next_obs)

    policy_grad_norm = 0.0
    for p in list(agent.policy_net.parameters()) + [agent.log_std]:
        if p.grad is not None:
            policy_grad_norm += float(p.grad.norm().item())
    assert policy_grad_norm > 0.0, "Policy parameters did not receive gradients from a full update() call"

    # --- Part 2: EFE's OWN contribution, isolated from VFE. Reconstruct the
    # same h_prior/z_post/s_next that update() itself computes, call the same
    # _compute_efe_loss() helper update() uses internally, and backward ONLY
    # that loss -- so any world-model gradient we see here can only have come
    # from EFE, not from VFE. ---
    agent.optimizer.zero_grad()
    a_t = torch.as_tensor(action, dtype=torch.float32, device=agent.device).unsqueeze(0)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)

    h_prior, z_prior, mu_p, logvar_p = agent.world_model.transition(agent.h, agent.z, a_t, s=agent.s)
    z_post, mu_q, logvar_q, s_next, *_ = agent.world_model.observe(
        h_prior, next_obs_t, s_prev=agent.s, update_s=True, step=agent.step
    )

    efe_loss, _, _ = agent._compute_efe_loss(h_prior, z_post, s_next)
    efe_loss.backward()

    policy_grad_norm_isolated = 0.0
    for p in list(agent.policy_net.parameters()) + [agent.log_std]:
        if p.grad is not None:
            policy_grad_norm_isolated += float(p.grad.norm().item())
    wm_grad_norm_isolated = 0.0
    for p in agent.world_model.parameters():
        if p.grad is not None:
            wm_grad_norm_isolated += float(p.grad.norm().item())

    assert policy_grad_norm_isolated > 0.0, "EFE loss alone did not produce gradients for policy parameters"
    assert wm_grad_norm_isolated == 0.0, (
        "EFE loss alone produced gradients for world_model parameters -- this should be "
        "impossible, since h_start/z_start/s_start are detached before being passed to "
        "the planner and world_model params are frozen for the duration of the call"
    )

def test_end_to_end_smoke_run(tmp_path):
    """
    Run a very short training loop using the training driver script entrypoint (if available).
    This test is permissive: it only asserts that training completes and writes CSV outputs.
    """
    set_test_seed(2)
    # Try to import train_all; if not present, skip this test
    try:
        from train_all import train_single, aggregate_results
    except Exception:
        pytest.skip("train_all entrypoint not available in this environment")

    # Run a single-seed, single-episode AIF training (very short)
    df = train_single(env_name="SensorBlackout", agent_name="AIF", episodes=2, seed=0, args=type("A", (), {"latency":0, "blackout":False, "no_beta":False, "no_persistence":False, "no_posterior":False, "no_efe":False, "use_context":False}))
    assert not df.empty
    # Basic checks on columns
    for col in ["Episode", "Reward", "Model", "Seed"]:
        assert col in df.columns
