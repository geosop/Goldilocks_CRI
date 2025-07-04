# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:51:41 2025

@author: ADMIN
"""
# visualize_synthetic_eeg.py
import os
import mne

# Path to the synthetic EEG files
output_dir = os.path.join(os.path.dirname(__file__), 'output')

file_A = os.path.join(output_dir, 'TierA_sleep_raw.fif')
file_B = os.path.join(output_dir, 'TierB_wake_raw.fif')

# Load Tier A (with artifact and flat channel)
print(f"Loading {file_A} ...")
raw_A = mne.io.read_raw_fif(file_A, preload=True)
print(f"Channels: {raw_A.ch_names}")

# Print basic info about the signal
ptp = raw_A.get_data(picks='eeg').ptp(axis=1)
print(f"EEG peak-to-peak per channel (µV): min={ptp.min()*1e6:.2f}, max={ptp.max()*1e6:.2f}")

# Plot the EEG (first 10 seconds, all channels)
print("Launching interactive plot for TierA_sleep_raw.fif ...")
raw_A.plot(duration=10.0, n_channels=20, scalings='auto', block=True)

# Repeat for Tier B
print(f"\nLoading {file_B} ...")
raw_B = mne.io.read_raw_fif(file_B, preload=True)
print(f"Channels: {raw_B.ch_names}")
ptp_B = raw_B.get_data(picks='eeg').ptp(axis=1)
print(f"TierB EEG ptp (µV): min={ptp_B.min()*1e6:.2f}, max={ptp_B.max()*1e6:.2f}")
raw_B.plot(duration=10.0, n_channels=20, scalings='auto', block=True)
