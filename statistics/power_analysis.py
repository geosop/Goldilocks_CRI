#!/usr/bin/env python
"""
@author: ADMIN

statistics/power_analysis.py

A priori power calculations for two‐sample t‐tests over a range of effect sizes.

Saves:
  statistics/output/power_analysis_results.csv
"""

import os
import yaml
import pandas as pd
from statsmodels.stats.power import TTestPower

def load_params():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "default_params.yml")      # define the path
    with open(path, "r", encoding="utf-8") as f:         # force UTF-8 decoding
        cfg = yaml.safe_load(f)
    return cfg["statistics"]

def main():
    params = load_params()
    analysis = TTestPower()

    results = []
    for es in params["effect_sizes"]:
        n_per_group = analysis.solve_power(
            effect_size=es,
            power=params["power"],
            alpha=params["alpha"],
            alternative="two-sided"
        )
        results.append({
            "effect_size": es,
            "required_n_per_group": float(n_per_group)
        })

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, "power_analysis_results.csv"),
        index=False
    )
    print("Saved power analysis results to", out_dir)

if __name__ == "__main__":
    main()
