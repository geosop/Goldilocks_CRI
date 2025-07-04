# -*- coding: utf-8 -*-
"""
@author: ADMIN

simulation_core/toy_model_master_eq.py

Integrate the toy master‐equation (no randomness) and save output to HDF5.

Usage:
    python toy_model_master_eq.py
"""

import os
import sys

# ─── ensure project root on path for utilities import ────────────────────────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

from utilities.seed_manager import load_state, save_state
import yaml
import numpy as np
from scipy.integrate import odeint
from retro_kernel_weight import w
import h5py

def load_params(path="default_params.yml"):
    """Load parameters from YAML under the key 'master_eq'."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["master_eq"]

def p_delta_function(delta: np.ndarray) -> np.ndarray:
    """
    Example p(Δ): linear ramp from 0→1 over max Δ.
    Returns zeros if max(delta)==0 to avoid divide-by-zero.
    """
    max_delta = np.max(delta)
    if max_delta == 0:
        return np.zeros_like(delta)
    return delta / max_delta

def master_rhs(X, t, params, past_times, past_X):
    """
    RHS of the toy master equation:
        dX/dt = -X / tau_mem  +  sum_i w(Δ_i, p(Δ_i)) * X_i * Δt
    where w is your retro-kernel weight function and f(X)=X.
    """
    # unpack parameters
    tau_mem = params["tau_mem"]
    lam     = params["lambda_decay"]
    p0      = params["p0"]
    alpha   = params["alpha"]

    # 1) exponential leak term
    leak = -X / tau_mem

    # 2) convolution over past states
    #    consider only times <= t
    mask     = past_times <= t
    deltas   = t - past_times[mask]
    X_past   = past_X[mask]
    p_vals   = p_delta_function(deltas)
    weights  = w(deltas, p_vals, p0, lam, alpha)
    # assume uniform spacing in past_times
    if len(past_times) > 1:
        dt = past_times[1] - past_times[0]
    else:
        dt = 0.0
    conv = np.sum(weights * X_past) * dt

    return leak + conv

def simulate_master(params):
    """
    Step through times with scipy.odeint, using master_rhs and past history.
    Returns (times, X_sol).
    """
    # time grid
    t_max = params["t_max"]
    dt    = params["dt"]
    times = np.arange(0.0, t_max + dt, dt)

    # initialize solution array
    X_sol = np.zeros_like(times)
    X_sol[0] = params.get("X0", 1.0)

    # integrate one step at a time
    for i in range(1, len(times)):
        t_prev, t_curr = times[i-1], times[i]
        # history up to current time
        past_times = times[: i + 1]
        past_X     = X_sol[: i + 1]
        # integrate from t_prev → t_curr
        Xi = odeint(
            master_rhs,
            X_sol[i - 1],
            [t_prev, t_curr],
            args=(params, past_times, past_X)
        )
        # Xi has shape (2,1); take the endpoint scalar
        X_sol[i] = Xi[-1, 0]

    return times, X_sol

def main():
    # ─── reproducibility hooks (no randomness here) ───────────────────────────
    load_state()   # no-op on first run
    # no np.random.seed() needed — purely deterministic
    save_state()
    # ─────────────────────────────────────────────────────────────────────────

    # load parameters
    here   = os.path.dirname(__file__)
    params = load_params(os.path.join(here, "default_params.yml"))

    # run the simulation
    times, X_sol = simulate_master(params)

    # save to HDF5
    out_file = os.path.join(here, "master_eq_output.h5")
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("time", data=times)
        hf.create_dataset("X",    data=X_sol)

    print(f"Saved master‐equation output to {out_file}")

if __name__ == "__main__":
    main()
