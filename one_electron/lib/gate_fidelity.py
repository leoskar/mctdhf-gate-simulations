"""Gauge-corrected two-qubit gate trace fidelity (single source of truth).

Validated against an exact two-well TD-FCI propagation.

Basis-order convention (rows/columns of M): (|00>, |10>, |01>, |11>).
"""

from __future__ import annotations

import numpy as np


def _localZ(alpha, beta):
    """Single-qubit Z⊗Z phase matrix in basis order (|00>,|10>,|01>,|11>)."""
    return np.diag([1.0, np.exp(1j * alpha), np.exp(1j * beta),
                    np.exp(1j * (alpha + beta))])


def gate_fidelity_up_to_localZ(M_ref, M):
    """Normalised gate overlap maximised over local Z⊗Z (both sides) + global
    phase: F = max |Tr(M_ref† Z_L M Z_R)|² / (‖M_ref‖_F² ‖M‖_F²). → 1 iff the
    gates agree up to single-qubit Z's. The idle dynamical phases are exactly
    such a local Z⊗Z, so this removes that trivial gauge."""
    from scipy.optimize import minimize
    if not np.all(np.isfinite(M)):
        return np.nan
    nref = np.vdot(M_ref, M_ref).real
    nm = np.vdot(M, M).real

    def neg(x):
        t = np.trace(M_ref.conj().T @ (_localZ(x[0], x[1]) @ M @ _localZ(x[2], x[3])))
        return -(abs(t) ** 2) / (nref * nm)

    best = None
    rng = np.random.default_rng(0)
    for _ in range(30):
        r = minimize(neg, rng.uniform(0, 2 * np.pi, 4), method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-12, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return float(-best.fun)
