#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 13:48:33 2025

author: ADMIN

logistic_gate/fit_logistic.py

Fit a logistic function to observed (p, G_obs) data and bootstrap CIs,
all controlled by the single master seed for full reproducibility.

Usage:
    python fit_logistic.py
"""

import os
import sys

# ─── ensure we can import utilities/seed_manager ─────────────────────────────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ─────────────────────────────────────────────────────────────────────────────

import yaml
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utilities.seed_manager import load_state, save_state


def load_params():
    """
    Load logistic parameters (including 'seed') from default_params.yml.
    """
    here    = os.path.dirname(__file__)
    cfg_path= os.path.join(here, "default_params.yml")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['logistic']


def logistic_function(p, p0, alpha):
    """G(p) = 1 / (1 + exp(-(p - p0)/alpha))."""
    return 1.0 / (1.0 + np.exp(- (p - p0) / alpha))


def fit_logistic(p, G_obs, p0_guess, alpha_guess):
    """Fit p0 and alpha by least squares, return (popt, perr)."""
    popt, pcov = curve_fit(
        logistic_function,
        p, G_obs,
        p0=[p0_guess, alpha_guess],
        bounds=([0, 1e-6], [1, np.inf])
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def bootstrap_ci(p, G_obs, p0_guess, alpha_guess, n_boot, ci, seed):
    """
    Bootstrap confidence intervals for p0 and alpha using master seed.
    """
    rng = np.random.default_rng(seed)
    boots = []
    n = len(p)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        popt, _ = fit_logistic(p[idx], G_obs[idx], p0_guess, alpha_guess)
        boots.append(popt)
    boots = np.array(boots)
    lower = np.percentile(boots, (100-ci)/2, axis=0)
    upper = np.percentile(boots, 100-(100-ci)/2, axis=0)
    return lower, upper


def main():
    # ─── reproducibility hook ────────────────────────────────────────────────
    load_state()                                 # no-op on first run
    params = load_params()
    np.random.seed(params.get('seed', 0))        # seed numpy RNG for legacy draws
    save_state()                                 # freeze state for downstream
    # ─────────────────────────────────────────────────────────────────────────

    # Load observed data
    here      = os.path.dirname(__file__)
    data_path = os.path.join(here, 'output', 'logistic_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data at {data_path}")
    df     = pd.read_csv(data_path)
    p      = df['p'].values
    G_obs  = df['G_obs'].values

    # Initial guesses from YAML
    p0_guess    = params['p0_guess']
    alpha_guess = params['alpha_guess']

    # Fit logistic
    (p0_hat, alpha_hat), (p0_err, alpha_err) = fit_logistic(
        p, G_obs, p0_guess, alpha_guess
    )

    # Bootstrap CIs using same master seed
    ci_low, ci_high = bootstrap_ci(
        p, G_obs,
        p0_guess, alpha_guess,
        n_boot=params['n_bootstrap'],
        ci=params['ci_percent'],
        seed=params.get('seed', 0)
    )

    # Report
    print(f"Fitted p0 = {p0_hat:.3f} ± {p0_err:.3f}")
    print(f"Fitted alpha = {alpha_hat:.4f} ± {alpha_err:.4f}")
    print(f"{params['ci_percent']}% CI p0:    [{ci_low[0]:.3f}, {ci_high[0]:.3f}]")
    print(f"{params['ci_percent']}% CI alpha: [{ci_low[1]:.4f}, {ci_high[1]:.4f}]")

    # Save results
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)
    results = {
        'p0_hat':       p0_hat,
        'alpha_hat':    alpha_hat,
        'p0_err':       p0_err,
        'alpha_err':    alpha_err,
        'p0_ci_low':    ci_low[0],
        'p0_ci_high':   ci_high[0],
        'alpha_ci_low': ci_low[1],
        'alpha_ci_high':ci_high[1]
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(out_dir, "fit_logistic_results.csv"),
        index=False
    )
    print(f"Saved fit results to {out_dir}/fit_logistic_results.csv")


if __name__ == '__main__':
    main()
