# tests/test_rssm_ensemble_integration.py
# Purpose: Unit tests to verify RSSM.transition returns prior stats and ensemble path executes when provided.

import torch
import numpy as np
from rssm_world_model import RSSMWorldModel
from transition_ensemble import TransitionEnsemble


def test_transition_returns_prior_stats_and_ensemble_path():
    obs_dim = 4
    action_dim = 2
    latent_dim = 6
    hidden_dim = 8
    slow_dim = 3

    # create a small ensemble with matching input dim (z + a + s)
    ensemble_input_dim = latent_dim + action_dim + slow_dim
    ensemble = TransitionEnsemble(input_dim=ensemble_input_dim, latent_dim=latent_dim, ensemble_size=3)

    model = RSSMWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        slow_dim=slow_dim,
        ensemble=ensemble,
    )

    batch = 2
    h = torch.zeros(batch, hidden_dim)
    z = torch.zeros(batch, latent_dim)
    a = torch.zeros(batch, action_dim)
    s = torch.zeros(batch, slow_dim)

    # call transition with ensemble present
    h_next, z_prior, mu_p, logvar_p = model.transition(h, z, a, s=s)
    # mu_p and logvar_p must be returned and finite
    assert mu_p.shape == (batch, latent_dim)
    assert logvar_p.shape == (batch, latent_dim)
    assert torch.isfinite(mu_p).all()
    assert torch.isfinite(logvar_p).all()
    # when ensemble is provided, z_prior should equal ensemble mean (approx)
    ensemble_inp = torch.cat([z, a, s], dim=-1)
    mean_z, var_z = ensemble(ensemble_inp)
    assert z_prior.shape == mean_z.shape
    assert torch.allclose(z_prior, mean_z, atol=1e-5)

def test_transition_without_ensemble_samples_prior():
    model = RSSMWorldModel(obs_dim=4, action_dim=2, latent_dim=6, hidden_dim=8, slow_dim=3, ensemble=None)
    h = torch.zeros(1, 8)
    z = torch.zeros(1, 6)
    a = torch.zeros(1, 2)
    h_next, z_prior, mu_p, logvar_p = model.transition(h, z, a, s=None)
    # z_prior should be sampled from mu_p/logvar_p and finite
    assert z_prior.shape == mu_p.shape
    assert torch.isfinite(z_prior).all()
