# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:14:15 2025
author: ADMIN

qpt/qpt_simulation.py

Generates synthetic Kraus‐map tomography data:
 - Simulates excited‐state populations P(t) = exp(-(γ_f+γ_b)*t/2)
 - Simulates inferred jump‐rate ratio R_obs ≈ λ_env + noise
 - Saves P(t), R_obs and λ_env to an NPZ for downstream fitting

Usage:
    python qpt_simulation.py
"""

import os
import sys
import yaml
import numpy as np

# ─── allow imports from project root ──────────────────────────────────────────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

from utilities.seed_manager import load_state, save_state


def load_params():
    """Load QPT parameters from default_params.yml in this folder."""
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'default_params.yml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['qpt']


def simulate_populations(t, gamma_f, gamma_b_vals):
    """
    Simulate excited-state populations for each gamma_b over time axis t:
      P(t) = exp[-(gamma_f + gamma_b) * t / 2].
    """
    return {gb: np.exp(-(gamma_f + gb) * t / 2.0) for gb in gamma_b_vals}


def simulate_R(lambda_env, noise_std, seed):
    """
    Simulate observed jump-rate ratios:
      R_obs = lambda_env + Gaussian noise,
    clipped at zero.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=lambda_env.shape)
    return np.clip(lambda_env + noise, 0.0, None)


def main():
    # ─── reproducibility ──────────────────────────────────────────────────────
    load_state()  # no-op on first run
    params = load_params()
    np.random.seed(params.get('seed', 0))
    save_state()  # freeze RNG for downstream steps
    # ─────────────────────────────────────────────────────────────────────────

    # Time axis for population decay
    t = np.linspace(0, params['t_max'], params['n_t'])

    # 1) Simulate populations
    pops_dict = simulate_populations(t, params['gamma_f'], params['gamma_b_vals'])

    # 2) Simulate R vs environmental coupling
    lambda_env = np.linspace(
        params['lambda_env_min'],
        params['lambda_env_max'],
        params['n_lambda_env']
    )
    R_obs = simulate_R(lambda_env, params['noise_R'], params.get('seed', 0))

    # Prepare output directory
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Save to NPZ
    out_path = os.path.join(out_dir, 'qpt_sim_data.npz')
    np.savez(
        out_path,
        t=t,
        gamma_b_vals=params['gamma_b_vals'],
        pops=[pops_dict[gb] for gb in params['gamma_b_vals']],
        lambda_env=lambda_env,
        R_obs=R_obs
    )
    print(f"Saved synthetic QPT data (populations + R_obs) to {out_path}")


if __name__ == '__main__':
    main()
