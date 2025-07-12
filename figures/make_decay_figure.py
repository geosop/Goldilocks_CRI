# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:26:27 2025

@author: ADMIN

figures/make_decay_figure.py

Generate Fig. 2 (single-column width) for Perspective:
  - Log-linear decay plot with ± noise band
  - Inset of raw A_pre vs Δ
  - Exports as PDF at 88 mm width, vector format with embedded fonts

Usage:
    python make_decay_figure.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ─── Publication-quality styling ─────────────────────────────────────────────
mpl.rcParams['font.family']       = 'Arial'
mpl.rcParams['font.size']         = 8
mpl.rcParams['axes.linewidth']    = 0.5     # ~0.25 pt after reduction
mpl.rcParams['lines.linewidth']   = 0.75    # ~0.5 pt after reduction
mpl.rcParams['legend.fontsize']   = 6
mpl.rcParams['xtick.labelsize']   = 6
mpl.rcParams['ytick.labelsize']   = 6
# ──────────────────────────────────────────────────────────────────────────────

def load_params(path):
    """Load decay parameters from the shared YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['decay']

def main():
    # Paths
    here      = os.path.dirname(__file__)
    repo      = os.path.abspath(os.path.join(here, os.pardir))
    decay_dir = os.path.join(repo, 'decay')
    out_dir   = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Load data & params
    params   = load_params(os.path.join(decay_dir, 'default_params.yml'))
    noise    = params['noise_log']
    df_pts   = pd.read_csv(os.path.join(decay_dir, 'output', 'decay_data.csv'))
    df_curve = pd.read_csv(os.path.join(decay_dir, 'output', 'decay_curve.csv'))

    # Figure: set to 88 mm wide, same 6:4 aspect ratio (1.5)
    width_mm  = 88
    height_mm = width_mm / 1.5
    fig, ax  = plt.subplots()
    fig.set_size_inches(width_mm/25.4, height_mm/25.4)  # mm→inches

    # Main panel: ln A_pre vs Δ
    ax.plot(
        df_curve['delta_cont'], df_curve['lnA_pre_cont'],
        label=r'$\ln A_{\rm pre}(\Delta)$'
    )
    ax.fill_between(
        df_curve['delta_cont'],
        df_curve['lnA_pre_cont'] - noise,
        df_curve['lnA_pre_cont'] + noise,
        alpha=0.2,
        label=rf'$\pm{noise:.2f}$'
    )
    ax.errorbar(
        df_pts['delta'], df_pts['lnA_pre'],
        yerr=noise, fmt='o', markersize=3,
        label='Sampled delays'
    )

    ax.set_xlabel(r'$\Delta$ (s)')
    ax.set_ylabel(r'$\ln A_{\rm pre}(\Delta)$')
    ax.legend(loc='upper right')

    # Inset: raw A_pre vs Δ
    ax_ins = inset_axes(
        ax, width="70%", height="70%",
        loc='lower left',
        bbox_to_anchor=(0.15, 0.17, 0.4, 0.4),
        bbox_transform=ax.transAxes
    )
    A_pre_cont = params['A0'] * np.exp(-df_curve['delta_cont'] / params['tau_fut'])
    A_pre_pts  = params['A0'] * np.exp(-df_pts['delta'] / params['tau_fut'])
    ax_ins.plot(df_curve['delta_cont'], A_pre_cont, linewidth=0.75)
    ax_ins.scatter(df_pts['delta'], A_pre_pts, s=10)
    ax_ins.set_title('Raw $A_{\\rm pre}(\\Delta)$', fontsize=8)
    ax_ins.set_xlabel(r'$\Delta$ (s)', fontsize=7)
    ax_ins.set_ylabel(r'$A_{\rm pre}$', fontsize=7)
    ax_ins.tick_params(labelsize=6)

    # Tight layout (vector objects only)
    fig.tight_layout()

    # Save as vector PDF (fonts embedded automatically)
    pdf_path = os.path.join(out_dir, 'decay_refined_figure.pdf')
    fig.savefig(
        pdf_path,
        format='pdf',
        bbox_inches='tight'
    )
    plt.close(fig)
    print(f"Saved decay figure to {pdf_path}")

if __name__ == '__main__':
    main()
