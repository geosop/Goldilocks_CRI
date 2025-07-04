# -*- coding: utf-8 -*-
"""
@author: ADMIN

retro_kernel_weight.py

Implements the retrocausal kernel
    w(Δ) = G(p(Δ) − p0) · exp(−λΔ)
where
    G(x) = 1 / (1 + exp(−x/α))

Usage:
    from retro_kernel_weight import w
    kernel_vals = w(delta_array, p_delta_array, p0=0.5, lam=0.1, alpha=0.05)
"""

import os
import sys

# ─── ensure we can import that utilities folder if needed ────────────────────
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np

def gating_function(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Logistic gating G(x) = 1 / (1 + exp(−x / alpha)).
    """
    return 1.0 / (1.0 + np.exp(-x / alpha))

def w(
    delta: np.ndarray,
    p_delta: np.ndarray,
    p0: float,
    lam: float,
    alpha: float
) -> np.ndarray:
    """
    Compute the retrocausal weight w(Δ) for each delay Δ.

    Parameters
    ----------
    delta : array-like
        Delay times Δ.
    p_delta : array-like
        The probability function p(Δ) evaluated at each Δ.
    p0 : float
        Reference probability threshold in the logistic gating.
    lam : float
        Exponential decay rate λ.
    alpha : float
        Scale parameter for the logistic gain.

    Returns
    -------
    w_vals : np.ndarray
        Retrocausal kernel values w(Δ).
    """
    delta   = np.asarray(delta)
    p_delta = np.asarray(p_delta)

    # 1) Compute logistic gating
    G     = gating_function(p_delta - p0, alpha)

    # 2) Exponential decay
    decay = np.exp(-lam * delta)

    # 3) Final weight = gate · decay
    return G * decay

