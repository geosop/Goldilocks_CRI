# -*- coding: utf-8 -*-
"""
@author: ADMIN

stimulus_presentation/psychopy_cue_scheduler.py

Generate a cue schedule CSV based on YAML parameters.

Usage:
    python psychopy_cue_scheduler.py
"""

import os
import yaml
import pandas as pd

def load_params():
    """Read schedule parameters from default_params.yml in the same folder."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, 'default_params.yml')
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['stimulus']

def main():
    params = load_params()
    deltas = params['deltas_min']    # List of delays in minutes
    pvals = params['p_values']       # List of probabilities

    # Convert to seconds and build DataFrame
    onset_s = [d * 60 for d in deltas]
    df = pd.DataFrame({
        'trial': list(range(1, len(deltas) + 1)),
        'delta_min': deltas,
        'onset_s': onset_s,
        'p_value': pvals
    })

    # Write to output folder
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    csvpath = os.path.join(outdir, 'cue_schedule.csv')
    df.to_csv(csvpath, index=False)
    print(f"Saved cue schedule to {csvpath}")

if __name__ == '__main__':
    main()
