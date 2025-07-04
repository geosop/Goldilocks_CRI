# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 13:45:17 2025

@author: ADMIN

decay/fit_decay.py

Fit ln A_pre = ln(A0) − Δ/τ_fut using weighted least squares and bootstrap CIs.
Wired into the master seed for full reproducibility.

Usage:
    python fit_decay.py
"""

import os, sys

# ─── project root on path so we can import utilities/seed_manager ────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

import yaml
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utilities.seed_manager import load_state, save_state


def load_params():
    """
    Load decay parameters (including 'seed') from default_params.yml.
    """
    here    = os.path.dirname(__file__)
    cfg_path= os.path.join(here, "default_params.yml")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['decay']


def decay_model(delta, lnA0, inv_tau):
    """ln A_pre = lnA0 − inv_tau * Δ."""
    return lnA0 - inv_tau * delta


def fit_decay(delta, lnA, se, p0_guess):
    popt, pcov = curve_fit(
        decay_model, delta, lnA,
        sigma=se, absolute_sigma=True,
        p0=p0_guess
    )
    perr = np.sqrt(np.diag(pcov))
    lnA0_hat, inv_tau_hat = popt
    lnA0_err, inv_tau_err = perr

    tau_hat = 1.0 / inv_tau_hat
    tau_err = inv_tau_err / (inv_tau_hat ** 2)
    return (lnA0_hat, tau_hat), (lnA0_err, tau_err)


def bootstrap_ci(delta, lnA, se, p0_guess, n_boot, ci, seed):
    """
    Bootstrap CIs for τ_fut using the master seed.
    """
    # reproducible RNG seeded from params['seed']
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_boot):
        lnA_sim = rng.normal(lnA, se)
        (_, tau_hat), _ = fit_decay(delta, lnA_sim, se, p0_guess)
        taus.append(tau_hat)
    low, high = np.percentile(taus, [(100-ci)/2, 100-(100-ci)/2])
    return low, high


def main():
    # ─── Reproducibility hook ────────────────────────────────────────────────
    load_state()                             # restore last RNG state (no-op first run)
    params = load_params()
    np.random.seed(params.get("seed", 0))    # seed legacy RNG
    save_state()                             # freeze state for next scripts
    # ─────────────────────────────────────────────────────────────────────────

    # Read data
    here      = os.path.dirname(__file__)
    df        = pd.read_csv(os.path.join(here, 'output', 'decay_data.csv'))
    delta     = df['delta'].values
    lnA       = df['lnA_pre'].values
    se        = df['se_lnA'].values

    # Initial guess from YAML
    p0_guess  = [np.log(params['A0']), 1.0/params['tau_fut']]

    # Fit
    (lnA0_hat, tau_hat), (_, tau_err) = fit_decay(delta, lnA, se, p0_guess)

    # Bootstrap using the same master seed
    ci_low, ci_high = bootstrap_ci(
        delta, lnA, se, p0_guess,
        n_boot=params['n_boot'],
        ci=params['ci_percent'],
        seed=params.get("seed", 0)
    )

    # Print
    print(f"τ_fut = {tau_hat:.2f} ± {tau_err:.2f} s")
    print(f"{params['ci_percent']}% CI: [{ci_low:.2f}, {ci_high:.2f}] s")

    # Save results as a one‐row CSV
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{
        'tau_hat':      tau_hat,
        'tau_err':      tau_err,
        'tau_ci_low':   ci_low,
        'tau_ci_high':  ci_high
    }]).to_csv(os.path.join(out_dir, 'fit_decay_results.csv'),
               index=False)

    print(f"Saved fit results in {out_dir}/")

if __name__ == '__main__':
    main()
