# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:39:48 2025

@author: ADMIN
"""
# utilities/seed_manager.py

import os
import pickle
from numpy.random import get_state, set_state

# Path to store the RNG state
STATE_FILE = os.path.join(os.path.dirname(__file__), 'rng_state.pkl')

def save_state():
    """
    Save the current NumPy RNG state to disk.
    Call this at key points (e.g. before a simulation) to enable exact replay.
    """
    state = get_state()
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(state, f)

def load_state():
    """
    Load the saved RNG state from disk.
    On first run (no state file) simply returns.
    """
    if not os.path.exists(STATE_FILE):
        return     # <— quietly do nothing
    with open(STATE_FILE, "rb") as f:
        state = pickle.load(f)
    set_state(state)
