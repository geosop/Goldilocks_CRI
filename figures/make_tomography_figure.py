# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:09:37 2025

@author: ADMIN

figures/make_tomography_figure.py

Generate the Quantum Process Tomography figure for Perspective:
  - Left: simulated excited-state population decay P(t)
  - Right: observed R vs κ with fitted linear relation
  - Exports as PDF at 180 mm width, vector format with embedded fonts

Usage:
    python make_tomography_figure.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# — Publication-quality styling —
mpl.rcParams['font.family']     = 'Arial'
mpl.rcParams['font.size']       = 8
mpl.rcParams['axes.linewidth']  = 0.5    # ≈0.25 pt after reduction
mpl.rcParams['lines.linewidth'] = 0.75   # ≈0.5 pt
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
# —————————————————————————

def load_params(path):
    """Load QPT simulation parameters from YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['qpt']


def main():
    here    = os.path.dirname(__file__)
    repo    = os.path.abspath(os.path.join(here, os.pardir))
    qpt_dir = os.path.join(repo, 'qpt')
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Load parameters and simulation data
    params = load_params(os.path.join(qpt_dir, 'default_params.yml'))
    sim_path = os.path.join(qpt_dir, 'output', 'qpt_sim_data.npz')
    sim = np.load(sim_path)
    if 'lambda_env' not in sim.files:
        raise KeyError("Missing 'lambda_env' in QPT simulation data")

    t            = sim['t']
    gamma_b_vals = sim['gamma_b_vals']
    pops_list    = sim['pops']
    lambda_env   = sim['lambda_env']
    R_obs        = sim['R_obs']

    # Load fit results
    df_R = pd.read_csv(os.path.join(qpt_dir, 'output', 'qpt_R_fit.csv'))
    slope     = df_R['slope'].iloc[0]
    intercept = df_R['intercept'].iloc[0]

    # Figure size: 180 mm wide, 3:1 aspect → 60 mm tall
    width_mm  = 180
    height_mm = width_mm / 3
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize=(width_mm/25.4, height_mm/25.4),
        gridspec_kw={'width_ratios': [1, 1]}
    )

    # Left: population decay
    for gb, pop in zip(gamma_b_vals, pops_list):
        ax1.plot(t, pop, label=rf'$\gamma_b={gb:.1f}$')
    ax1.set_xlabel('Time $t$ (s)')
    ax1.set_ylabel('Population $P(t)$')
    ax1.set_title('Simulated Excited-State Populations')
    ax1.legend(title='Backward rates', loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right: R vs lambda_env
    ax2.scatter(lambda_env, R_obs, s=25, label='Observed $R$')
    lam_cont = np.linspace(
        params['lambda_env_min'],
        params['lambda_env_max'],
        100
    )
    ax2.plot(
        lam_cont,
        slope * lam_cont + intercept,
        linestyle='--',
        label=rf'Fit: $R = {slope:.2f}\,\lambda_{{env}} + {intercept:.2f}$'
    )
    ax2.set_xlabel(r'Environmental coupling $\lambda_{env}$')
    ax2.set_ylabel(r'Jump-rate ratio $R$')
    ax2.set_title('Inferred $R$ vs. $\lambda_{env}$')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Save as vector PDF
    pdf_path = os.path.join(out_dir, 'tomography_refined_figure.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved tomography figure to {pdf_path}")

if __name__ == '__main__':
    main()
