# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:15:21 2025

@author: ADMIN

epochs_features/compute_x_t.py

Compute noise-suppression index x(t) from theta-band epoch envelopes.
Outputs a CSV of time vs x(t).

Usage:
    python compute_x_t.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import mne  # EEG processing library


def load_params():
    """Load parameters from default_params.yml."""
    here = os.path.dirname(__file__)
    cfg = yaml.safe_load(open(os.path.join(here, 'default_params.yml')))
    return cfg['epochs_features']


def main():
    params = load_params()

    # Directories and files
    here = os.path.dirname(__file__)
    raw_dir = os.path.abspath(os.path.join(here, '..', 'preprocessing', 'output'))
    events_path = os.path.abspath(os.path.join(here, '..', 'synthetic_EEG', 'output', 'events.tsv'))

    # Load events
    events_df = pd.read_csv(events_path, sep='	')
    sfreq = params['sfreq']
    events = np.column_stack([
        (events_df['onset_s'] * sfreq).astype(int),
        np.zeros(len(events_df), int),
        np.ones(len(events_df), int)
    ])
    event_id = {'spindle': 1}

    # Accumulate envelopes
    all_env = []
    times = None
    for fname in os.listdir(raw_dir):
        if not fname.endswith('_clean.fif'):
            continue
        raw = mne.io.read_raw_fif(os.path.join(raw_dir, fname), preload=True, verbose=False)
        picks = mne.pick_types(raw.info, eeg=True)
        # Extract theta-band epochs
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=params['tmin'], tmax=params['tmax'],
            picks=picks, baseline=None,
            preload=True, verbose=False
        )
        # Filter for theta
        theta = params['bands']['theta']
        ep_theta = epochs.copy().filter(theta[0], theta[1], fir_design='firwin', verbose=False)
        data = ep_theta.get_data()  # shape: (n_epochs, n_channels, n_times)
        # Hilbert envelope
        env = np.abs(np.fft.ifft(np.fft.fft(data, axis=2)))  # placeholder for hilbert
        # average across channels and epochs
        mean_env = env.mean(axis=(0,1))
        all_env.append(mean_env)
        if times is None:
            times = ep_theta.times

    if not all_env:
        raise FileNotFoundError("No cleaned FIF files or events found for compute_x_t.")

    # Average across recordings
    env_avg = np.mean(all_env, axis=0)

    # Sliding window average
    wlen = int(params['window_s'] * sfreq)
    kernel = np.ones(wlen) / wlen
    x_t = np.convolve(env_avg, kernel, mode='same')

    # Save x(t)
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({'time_s': times, 'x_t': x_t})
    df.to_csv(os.path.join(out_dir, 'x_t.csv'), index=False)
    print(f"Saved x(t) to {out_dir}/x_t.csv")

if __name__ == '__main__':
    main()

