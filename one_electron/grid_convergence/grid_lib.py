"""Parameterized grid / FCI builders for the grid convergence check.

`td_fci.py` hardcodes the two-well sine-DVR at NUM_DVR=64
(its T_S, X_S, U_01, _well_v, static_H_dense are all built at import). The check needs
the SAME physics at arbitrary grid size N (the DVR-convergence axis), so this
module re-parameterizes only the N-dependent pieces, reusing every N-independent
constant from td_fci (well centres/widths, KAPPA, the soft-Coulomb, SineDVR).

The FCI reference is the maximal-SPF limit: the lowest-k eigenstates of the full
interacting product-grid Hamiltonian (no orbital truncation), via matrix-free
`eigsh`, so large N stays affordable.
"""

from __future__ import annotations

import os
import sys
from collections import namedtuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                               # td_fci
sys.path.insert(0, os.path.join(HERE, "..", "lib"))    # geometry_config

import td_fci as tf  # noqa: E402  (N-independent constants + SineDVR + schedule)

# N-independent physical constants (single source of truth = td_fci).
WELL_CENTERS = [float(c) for c in np.asarray(tf.WELL_CENTERS)]
WELL_WIDTHS = [float(w) for w in np.asarray(tf.WELL_WIDTHS)]
NUM_WELLS = tf.NUM_WELLS
TYPICAL_FREQ_GHZ = tf.TYPICAL_FREQ_GHZ

Grid = namedtuple("Grid", ["N", "T_s", "X_s", "U_01"])


def build_grid(N):
    """Two-well sine-DVR at num_points=N. Mirrors td_fci.py:87-99 for arbitrary N."""
    bases = [
        tf.SineDVR(
            x_min=WELL_CENTERS[s] - 0.5 * WELL_WIDTHS[s],
            x_max=WELL_CENTERS[s] + 0.5 * WELL_WIDTHS[s],
            num_points=N,
        )
        for s in range(NUM_WELLS)
    ]
    zero = lambda x: np.zeros_like(x)
    T_s = [np.asarray(bases[s].get_h(potential=zero)) for s in range(NUM_WELLS)]
    X_s = [np.asarray(bases[s].get_grid()) for s in range(NUM_WELLS)]
    U_01 = np.asarray(bases[0].get_g_cross_species(bases[1], tf._coulomb))
    return Grid(N=N, T_s=T_s, X_s=X_s, U_01=U_01)


def well_v(s, x4, grid):
    """Diagonal potential on grid s for config x4=[a0,a1,b0,b1] GHz (parameterized
    copy of td_fci._well_v with the grid passed in)."""
    a = x4[s] / TYPICAL_FREQ_GHZ
    b = x4[2 + s] / TYPICAL_FREQ_GHZ
    dx = grid.X_s[s] - WELL_CENTERS[s]
    return 0.5 * a ** 2 * dx ** 2 + (b * a ** 2 / 3.0) * dx ** 4


def static_matvec(x4, grid):
    """Matrix-free H(x4) psi on the (N, N) product grid."""
    N = grid.N
    T0, T1 = grid.T_s[0], grid.T_s[1]
    U = grid.U_01
    v0 = well_v(0, x4, grid)
    v1 = well_v(1, x4, grid)

    def mv(psi_flat):
        psi = psi_flat.reshape(N, N)
        out = T0 @ psi + psi @ T1 + v0[:, None] * psi + v1[None, :] * psi + U * psi
        return out.ravel()

    return mv


def fci_eigs(x4, grid, k=6):
    """Lowest k eigenvalues (and eigenvectors (k, N, N)) of the interacting grid H
    at static config x4 — the FCI reference (no orbital truncation)."""
    N = grid.N
    op = LinearOperator((N * N, N * N), matvec=static_matvec(x4, grid), dtype=float)
    evals, evecs = eigsh(op, k=k, which="SA", maxiter=10000, tol=0)
    order = np.argsort(evals)
    evals = evals[order]
    vecs = evecs[:, order].T.reshape(k, N, N)
    return evals, vecs


def natural_occupations(C, weights=None):
    """State-averaged natural-orbital occupations per well (descending). Copy of
    run_mctdh._natural_occupations (kept local to avoid importing run_mctdh, which
    parses argv / sets the JAX platform at import)."""
    C = np.asarray(C)
    if not np.all(np.isfinite(C)):
        return np.array([np.nan]), np.array([np.nan])
    S = C.shape[0]
    weights = np.ones(S) / S if weights is None else np.asarray(weights)
    rho0 = np.zeros((C.shape[1], C.shape[1]), complex)
    rho1 = np.zeros((C.shape[2], C.shape[2]), complex)
    for s in range(S):
        rho0 += weights[s] * (C[s] @ C[s].conj().T)
        rho1 += weights[s] * (C[s].conj().T @ C[s])

    def _occ(rho):
        try:
            w = np.linalg.eigvalsh(rho).real
        except np.linalg.LinAlgError:
            w = np.linalg.eigvals(rho).real
        return np.sort(w)[::-1]

    return _occ(rho0), _occ(rho1)
