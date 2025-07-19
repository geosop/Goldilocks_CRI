# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:10:18 2025

@author: ADMIN
"""
import picard
# Monkey-patch a _version_ so MNE's version check passes:
picard.__version__ = "0.8.0"

import os
import sys
import yaml
import mne
import numpy as np

# ─── suppress MNE “stupid warnings” about file naming ──────────────────────────
mne.utils.set_config('MNE_IGNORE_STUPID_WARNINGS', 'true')


# ─── 'utilities' is on the import path ───────────────────────────────
here = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(here, '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

from utilities.seed_manager import load_state, save_state

# ─── here we add these third-party imports ───────────────────────────────────────────────
from scipy.signal import hilbert
import pandas as pd
# ──────────────────────────────────────────────────────────────────────────────

def load_params():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, 'default_params.yml'), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['preprocessing']



def detect_bad_channels(raw, params):
    eeg_ch_names = [ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == 'eeg']
    data = raw.get_data(picks='eeg')
    ptp_vals = np.ptp(data, axis=1)

    ptp_thresh = float(params.get('ptp_threshold', 150e-6))
    flat_thresh = float(params.get('flat_threshold', 1e-6))

    bad_ptp = [ch for ch, val in zip(eeg_ch_names, ptp_vals) if val > ptp_thresh]
    bad_flat = [ch for ch, val in zip(eeg_ch_names, ptp_vals) if val < flat_thresh]

    # Variance-based
    vars_ = np.var(data, axis=1)
    thr = vars_.mean() + params['bads_threshold'] * vars_.std()
    bad_var = [ch for ch, val in zip(eeg_ch_names, vars_) if val > thr]

    print(f"EEG channel peak-to-peak range (min, max): {ptp_vals.min()*1e6:.2f}µV, {ptp_vals.max()*1e6:.2f}µV")
    print(f"Bad PTP channels: {len(bad_ptp)}, Bad flat: {len(bad_flat)}, Bad var: {len(bad_var)}")

    bads = sorted(set(bad_ptp + bad_flat + bad_var))
    print(f"Marking bad channels (PTP > {ptp_thresh*1e6:.1f}µV, < {flat_thresh*1e6:.1f}µV, or high var): {bads}" if bads else "No bad channels detected.")
    return bads




def artifact_pipeline(raw, params):
    # 1) Band-pass filter
    raw.filter(params['l_freq'], params['h_freq'], fir_design='firwin', verbose=False)
    # 2) Notch filter
    raw.notch_filter(params['notch_freq'], fir_design='firwin', verbose=False)
    # 3) Bad-channel detection
    bads = detect_bad_channels(raw, params)
    raw.info['bads'] = bads

    # 4) Interpolate (try, fallback to nan if no dig points)
    try:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    except Exception as e:
        print(f"Interpolation failed: {e}. Bad channels set to NaN.")
        raw.interpolate_bads(method='nan', reset_bads=True, verbose=False)

    # 5) ICA (on valid EEG only)
    picks_eeg = mne.pick_types(raw.info, eeg=True, exclude='bads')
    valid_eeg = []
    for idx in picks_eeg:
        d = raw.get_data(picks=[idx])
        if not np.all(np.isnan(d)):
            valid_eeg.append(idx)
    if not valid_eeg:
        print("All EEG channels are NaN after interpolation. Skipping ICA.")
        return raw

    # Robust EOG handling
    eog_ch = params.get('eog_ch', 'EOG 061')
    picks_eog = []
    if eog_ch in raw.ch_names:
        eog_idx = raw.ch_names.index(eog_ch)
        d = raw.get_data(picks=[eog_idx])
        if not np.all(np.isnan(d)):
            picks_eog = [eog_idx]
    

    # ICA
    from mne.preprocessing import ICA
    n_components = min(params['ica_n_components'], len(valid_eeg))
    ica = ICA(
        n_components=n_components,
        method='picard',
        fit_params={'ortho': False},
        random_state=0,
        verbose=False
    )
    ica.fit(raw, picks=valid_eeg, verbose=False)

    # EOG artifact removal if present
    if picks_eog:
        try:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch, verbose=False)
            ica.exclude = eog_inds
        except Exception as e:
            print(f"Warning: EOG artifact detection failed ({e}), skipping.")

    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean

def main():
    params = load_params()
    here = os.path.dirname(__file__)
    raw_dir = os.path.normpath(os.path.join(here, '..', 'synthetic_EEG', 'output'))
    out_dir = os.path.join(here, 'output')
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(raw_dir):
        if not fname.endswith('.fif'):
            continue
        raw = mne.io.read_raw_fif(os.path.join(raw_dir, fname), preload=True, verbose=False)
        raw_clean = artifact_pipeline(raw, params)
        clean_name = fname.replace('.fif', '_clean_eeg.fif')
        raw_clean.save(os.path.join(out_dir, clean_name), overwrite=True, verbose=False)
        print(f"Saved cleaned data to {out_dir}/{clean_name}")

if __name__ == '__main__':
    main()
