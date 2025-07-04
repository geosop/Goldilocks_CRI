# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:19:14 2025

@author: ADMIN

epochs_features/extract_epochs.py

Load cleaned EEG, extract epochs around events, compute band‐specific envelopes,
and save per-epoch metrics & envelope NPZs.

Usage:
    python extract_epochs.py
"""

import os
import yaml
import numpy as np
import mne
from scipy.signal import hilbert
import pandas as pd

# Suppress MNE filename‐convention warnings
mne.utils.set_config('MNE_IGNORE_STUPID_WARNINGS', 'true')

def load_params():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, 'default_params.yml'), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['epochs_features']

def main():
    params = load_params()
    here = os.path.dirname(__file__)
    raw_dir = os.path.join(here, '..', 'preprocessing', 'output')
    evt_path = os.path.join(here, '..', 'synthetic_EEG', 'output', 'events.tsv')
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Load events.tsv
    events_df = pd.read_csv(evt_path, sep='\t')
    # Convert to MNE events array: sample index, 0, event_id
    raw0 = mne.io.read_raw_fif(
        os.path.join(raw_dir, os.listdir(raw_dir)[0]),
        preload=False, verbose=False
    )
    sfreq = raw0.info['sfreq']
    events = []
    event_id = {'spindle': 1}
    for _, row in events_df.iterrows():
        sample = int(row['onset_s'] * sfreq)
        events.append([sample, 0, event_id[row['trial_type']]])
    events = np.array(events, int)

    metrics = []

    # Process each cleaned file
    for fname in os.listdir(raw_dir):
        if not fname.endswith('_clean_eeg.fif'):
            continue
        raw_path = os.path.join(raw_dir, fname)
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

        picks = mne.pick_types(raw.info, eeg=True)
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=params['tmin'], tmax=params['tmax'],
            picks=picks, baseline=None,
            preload=True, verbose=False
        )

        for band_name, (l, h) in params['bands'].items():
            # Filter
            ep_filt = epochs.copy().filter(
                l, h, fir_design='firwin', verbose=False
            ).get_data()  # shape: (n_epochs, n_ch, n_times)

            # Hilbert envelope
            env = np.abs(hilbert(ep_filt, axis=2))

            # Save NPZ per band
            np.savez(
                os.path.join(out_dir, f'epochs_{band_name}_env.npz'),
                envelope=env, times=epochs.times
            )

            # Per-epoch metrics: peak & duration above mean+thr*std
            peak_vals = env.max(axis=2).mean(axis=1)  # average across channels
            thr = peak_vals.mean() + params['threshold_multiplier'] * peak_vals.std()
            durations = []
            for e in env.mean(axis=1):  # per-epoch mean envelope
                above = e > thr
                # approximate duration in seconds
                durations.append(above.sum() / sfreq)
            for idx, (pk, dur) in enumerate(zip(peak_vals, durations)):
                metrics.append({
                    'file': fname,
                    'band': band_name,
                    'epoch': idx,
                    'peak_env': float(pk),
                    'dur_above_thr_s': float(dur)
                })

    # Save metrics CSV
    pd.DataFrame(metrics).to_csv(
        os.path.join(out_dir, 'epoch_metrics.csv'),
        index=False
    )
    print(f"Epoch extraction complete. Outputs in {out_dir}")

if __name__ == '__main__':
    main()
