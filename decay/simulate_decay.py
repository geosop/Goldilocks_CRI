# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 13:53:17 2025

@author: ADMIN

decay/simulate_decay.py

Simulate log-linear decay data:
  ln A_pre = ln(A0) − Δ/τ_fut
and save discrete points + continuous curve to CSV.

Usage:
    python simulate_decay.py
"""

import os, sys
import yaml
import numpy as np
import pandas as pd

# import from the project root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
    
# seed_manager for full‐pipeline reproducibility
from utilities.seed_manager import load_state, save_state

def load_params():
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, "default_params.yml")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['decay']

def main():
    # ─── reproducibility ──────────────────────────────────────────────────────
    load_state()                                # restore prior RNG state (no-op first run)
    params = load_params()
    np.random.seed(params.get("seed", 0))       # set the master seed
    save_state()                                # freeze state for downstream scripts
    # ─────────────────────────────────────────────────────────────────────────

    # Discrete delays
    deltas = np.arange(
        params['delta_start'],
        params['delta_end'] + params['delta_step'],
        params['delta_step']
    )

    # Compute true values
    A_pre     = params['A0'] * np.exp(-deltas / params['tau_fut'])
    lnA_pre   = np.log(A_pre)
    se_lnA    = np.full_like(lnA_pre, params['noise_log'])

    # Continuous curve for plotting
    deltas_cont   = np.linspace(
        params['delta_start'],
        params['delta_end'],
        params['n_cont']
    )
    A_pre_cont    = params['A0'] * np.exp(-deltas_cont / params['tau_fut'])
    lnA_pre_cont  = np.log(A_pre_cont)

    # Prepare output directory
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Write discrete data
    pd.DataFrame({
        'delta':      deltas,
        'lnA_pre':    lnA_pre,
        'se_lnA':     se_lnA
    }).to_csv(os.path.join(out_dir, 'decay_data.csv'), index=False)

    # Write continuous curve data
    pd.DataFrame({
        'delta_cont':     deltas_cont,
        'lnA_pre_cont':   lnA_pre_cont
    }).to_csv(os.path.join(out_dir, 'decay_curve.csv'), index=False)

    print(f"Saved discrete & continuous decay data in {out_dir}/")

if __name__ == '__main__':
    main()
