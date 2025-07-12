# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:06:52 2025

@author: ADMIN

figures/make_logistic_figure.py

Generate the logistic “Tipping-Point” figure for Perspective:
  - Scatter of observed R vs p
  - Fitted logistic curve with ± noise band
  - Vertical line at fitted p0
  - Inset of derivative dG/dp
  - Exports vector PDF at 88 mm width, publication-quality styling

Usage:
    python make_logistic_figure.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# — Publication-quality styling —
mpl.rcParams['font.family']     = 'Arial'
mpl.rcParams['font.size']       = 8
mpl.rcParams['axes.linewidth']  = 0.5    # ~0.25 pt after reduction
mpl.rcParams['lines.linewidth'] = 0.75   # ~0.5 pt
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
# ————————————————————————————————

def load_params(path):
    """Load logistic parameters from YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['logistic']

def logistic(p, p0, alpha):
    """G(p) = 1 / (1 + exp(-(p - p0)/alpha))."""
    return 1.0 / (1.0 + np.exp(-(p - p0) / alpha))

def derivative(p, p0, alpha):
    """dG/dp analytic."""
    k = 1.0 / alpha
    exp_t = np.exp(-k * (p - p0))
    return k * exp_t / (1.0 + exp_t)**2

def main():
    here    = os.path.dirname(__file__)
    repo    = os.path.abspath(os.path.join(here, os.pardir))
    log_dir = os.path.join(repo, 'logistic_gate')
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Load
    params = load_params(os.path.join(log_dir, 'default_params.yml'))
    df     = pd.read_csv(os.path.join(log_dir, 'output', 'logistic_data.csv'))
    df_fit = pd.read_csv(os.path.join(log_dir, 'output', 'fit_logistic_results.csv'))

    p0_hat    = df_fit['p0_hat'].iloc[0]
    alpha_hat = df_fit['alpha_hat'].iloc[0]
    noise     = params['noise_std']

    # Domain
    p_cont = np.linspace(params['p_min'], params['p_max'], params['n_points'])
    G_fit  = logistic(p_cont, p0_hat, alpha_hat)
    upper  = np.clip(G_fit + noise, 0.0, 1.0)
    lower  = np.clip(G_fit - noise, 0.0, 1.0)

    # Figure size: 88 mm-wide single column, 6:4 aspect ratio
    width_mm   = 88
    height_mm  = width_mm / 1.5
    fig, ax    = plt.subplots(constrained_layout=True)
    fig.set_size_inches(width_mm/25.4, height_mm/25.4)

    # Main panel
    ax.plot(p_cont, G_fit, label='Fitted logistic')
    ax.fill_between(p_cont, lower, upper, alpha=0.3, label=rf'±{noise:.2f} CI')
    ax.scatter(df['p'], df['G_obs'], s=25, edgecolors='k', label='Observed')
    ax.axvline(p0_hat, color='grey', linestyle='--',
               label=rf'$\hat p_0={p0_hat:.2f}$')

    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$G(p \mid a)$')
    ax.legend(loc='upper left')

    # Inset: derivative
    ax_ins = inset_axes(
        ax, width="75%", height="75%",
        loc='lower left',
        bbox_to_anchor=(0.65, 0.3, 0.4, 0.4),
        bbox_transform=ax.transAxes
    )
    ax_ins.plot(p_cont, derivative(p_cont, p0_hat, alpha_hat), linewidth=0.75)
    ax_ins.set_title(r'$\mathrm{d}G/\mathrm{d}p$', fontsize=8)
    ax_ins.set_xlabel(r'$p$', fontsize=7)
    ax_ins.set_ylabel('Rate', fontsize=7)
    ax_ins.tick_params(labelsize=6)

    # Save vector PDF
    pdf_path = os.path.join(out_dir, 'logistic_refined_figure.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved logistic figure to {pdf_path}")

if __name__ == '__main__':
    main()

