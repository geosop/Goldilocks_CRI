#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: ADMIN

statistics/permutation_test.py

Within‐subject permutation test comparing peak envelope between TierA and TierB
for each frequency band.

Loads:
  ../epochs_features/output/epoch_metrics.csv

Saves:
  statistics/output/permutation_test_results.csv
"""

import os
import sys

# ─── ensure project root on path so we can import utilities/seed_manager ────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

import yaml
import numpy as np
import pandas as pd
from utilities.seed_manager import load_state, save_state


def load_params():
    """Load statistics parameters (including 'seed') from default_params.yml."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "default_params.yml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["statistics"]


def permutation_test(data1, data2, n_perm, seed):
    """Two‐sample permutation test; returns observed difference and p‐value."""
    obs_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm1 = combined[: len(data1)]
        perm2 = combined[len(data1) :]
        if abs(np.mean(perm1) - np.mean(perm2)) >= abs(obs_diff):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return obs_diff, p_value


def main():
    # ─── reproducibility hook ────────────────────────────────────────────────
    load_state()   # no-op on first run
    params = load_params()
    np.random.seed(params.get("seed", 0))  # seed numpy RNG for consistency
    save_state()   # freeze state before next script
    # ─────────────────────────────────────────────────────────────────────────

    # Load per‐epoch metrics
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), 
            "..", "epochs_features", "output", "epoch_metrics.csv"
        )
    )

    results = []
    for band in df["band"].unique():
        df_band = df[df["band"] == band]
        data_A = df_band[df_band["file"].str.contains("TierA")]["peak_env"].values
        data_B = df_band[df_band["file"].str.contains("TierB")]["peak_env"].values

        obs_diff, p_val = permutation_test(
            data_A, 
            data_B,
            n_perm=params["n_perm"],
            seed=params.get("seed", 0)
        )
        results.append({
            "band":     band,
            "obs_diff": float(obs_diff),
            "p_value":  float(p_val)
        })

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, "permutation_test_results.csv"),
        index=False
    )
    print("Saved permutation test results to", out_dir)


if __name__ == "__main__":
    main()
