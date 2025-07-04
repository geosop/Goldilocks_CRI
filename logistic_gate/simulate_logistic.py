# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 13:45:52 2025

author: ADMIN

logistic_gate/simulate_logistic.py

Simulate noisy observations of the logistic “tipping‐point” curve G(p).
Outputs a CSV with columns: p, G_true, G_obs.

Usage:
    python simulate_logistic.py
"""

import os
import sys

# ─── make sure 'utilities' is on the import path ───────────────────────────────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

from utilities.seed_manager import load_state, save_state
import yaml
import numpy as np
import pandas as pd

def load_params():
    """Read schedule parameters from default_params.yml in the same folder."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, 'default_params.yml')
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['logistic']

def simulate(p, p0, alpha, noise_std, seed=0):
    """Compute true G(p) and add Gaussian noise."""
    k = 1.0 / alpha
    G_true = 1.0 / (1.0 + np.exp(-k * (p - p0)))
    rng     = np.random.default_rng(seed)
    noise   = rng.normal(loc=0.0, scale=noise_std, size=p.shape)
    G_obs   = np.clip(G_true + noise, 0.0, 1.0)
    return G_true, G_obs

def main():
    # ─── reproducibility ──────────────────────────────────────────────────────
    load_state()                              # restore prior RNG state (noop 1st run)
    params = load_params()
    np.random.seed(params.get('seed', 0))     # set master seed (e.g. 52)
    save_state()                              # freeze state for downstream steps
    # ─────────────────────────────────────────────────────────────────────────

    # Parameter grid
    p = np.linspace(
        params['p_min'],
        params['p_max'],
        params['n_points']
    )

    # Simulate true and observed G
    G_true, G_obs = simulate(
        p,
        p0=params['p0'],
        alpha=params['alpha'],
        noise_std=params['noise_std'],
        seed=params.get('seed', 0)
    )

    # Prepare output directory
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Save data
    df = pd.DataFrame({
        'p':       p,
        'G_true':  G_true,
        'G_obs':   G_obs
    })
    csv_path = os.path.join(out_dir, 'logistic_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved simulated logistic data to {csv_path}")

if __name__ == '__main__':
    main()

