# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:15:36 2025
author: ADMIN

qpt/qpt_fit.py

Fits:
  1) P(t) → recover (γ_f + γ_b) via linear regression on ln P(t)
  2) R_obs vs λ_env → linear fit R = λ_env (zero intercept) with bootstrap 95% CI

Outputs fit parameters & CIs to CSV.
Usage:
    python qpt_fit.py
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import linregress


def load_params():
    """Load QPT parameters from default_params.yml in the same folder."""
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'default_params.yml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['qpt']


def fit_population(t, pops):
    """
    Fit ln P = –(γ_f + γ_b) t/2 via linear regression.
    Returns (gamma_sum, gamma_err).
    """
    slope, _, _, _, stderr = linregress(t, np.log(pops))
    gamma_sum = -2.0 * slope
    gamma_err = 2.0 * stderr
    return gamma_sum, gamma_err


def fit_R(lambda_env, R_obs):
    """
    Fit R_obs vs lambda_env via linear regression:
       R_obs = slope * lambda_env + intercept
    Returns (slope, intercept, slope_err).
    """
    slope, intercept, _, _, stderr = linregress(lambda_env, R_obs)
    return slope, intercept, stderr


def bootstrap_slope(lambda_env, R_obs, n_boot, ci_percent):
    """
    Bootstrap CI for the slope of R vs lambda_env at given ci_percent.
    """
    rng = np.random.default_rng(0)
    slopes = []
    n = len(lambda_env)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        slope, _, _ = fit_R(lambda_env[idx], R_obs[idx])
        slopes.append(slope)
    lower = (100 - ci_percent) / 2.0
    upper = 100 - lower
    return np.percentile(slopes, [lower, upper])


def main():
    # Load parameters
    params = load_params()

    # Load simulation data
    here = os.path.dirname(__file__)
    data = np.load(os.path.join(here, 'output', 'qpt_sim_data.npz'))
    t            = data['t']
    gamma_b_vals = data['gamma_b_vals']
    pops         = data['pops']       # list of arrays
    lambda_env   = data['lambda_env']
    R_obs        = data['R_obs']

    # 1) Fit populations for each backward rate
    pop_results = []
    for gb, pop in zip(gamma_b_vals, pops):
        sum_rate, sum_err = fit_population(t, pop)
        pop_results.append({
            'gamma_b':    gb,
            'gamma_sum':  sum_rate,
            'gamma_err':  sum_err
        })
    df_pops = pd.DataFrame(pop_results)

    # 2) Fit R_obs vs lambda_env
    slope, intercept, slope_err = fit_R(lambda_env, R_obs)
    ci_low, ci_high = bootstrap_slope(
        lambda_env, R_obs,
        params['n_bootstrap'],
        params.get('ci_percent', 95)
    )
    df_R = pd.DataFrame({
        'slope':     [slope],
        'intercept': [intercept],
        'slope_err': [slope_err],
        'ci_low':    [ci_low],
        'ci_high':   [ci_high]
    })

    # Save fit results
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)
    df_pops.to_csv(os.path.join(out_dir, 'qpt_pop_fit.csv'), index=False)
    df_R.to_csv(os.path.join(out_dir, 'qpt_R_fit.csv'), index=False)

    print(f"Population fits saved to {out_dir}/qpt_pop_fit.csv")
    print(f"Ratio fits saved to      {out_dir}/qpt_R_fit.csv")
    print(f"Slope = {slope:.3f} ± {slope_err:.3f} (CI {params.get('ci_percent',95)}%: [{ci_low:.3f}, {ci_high:.3f}])")


if __name__ == '__main__':
    main()
