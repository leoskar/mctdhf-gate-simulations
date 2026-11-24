"""Shared conventions for the routed-sqiSWAP circuits:

  (1)  ANCHOR PHASE.  The four (or sixteen) anchors are separately computed
        eigenstates with independent arbitrary phases (measured
        [-180,-180,-180,0] deg).  Under Psi_i -> e^{i th_i} Psi_i a gate matrix
        transforms as M -> diag(e^{-i th}) M diag(e^{i th}), so a PRODUCT gate
        stops looking like a product unless th_11 = th_10 + th_01 - th_00.
        Two-qubit fidelities are exactly invariant to this; single-qubit gate
        matrices are NOT.  Fix: rotate each anchor so its dominant CI
        coefficient is real positive.

  (2)  LEG LOCAL-Z FRAME.  A measured leg is an excellent sqiSWAP wearing ~1e3
        rad of accumulated dynamical phase (F_bar 0.999 but raw overlap 0.023).
        The decomposition's local layers must be designed against the
        gauge-corrected leg -- exactly what an experimentalist gets for free
        with virtual Z.  strip_localZ() returns (z_L, z_R, M_stripped).

  (3)  SINGLE-QUBIT GATES as one-body operators on the anchor qubit orbitals.
        Tensor ordering: the 4x4 pair basis is (|00>,|10>,|01>,|11>), i.e.
        index = n_first + 2*n_second, so the SECOND well of the pair is the more
        significant bit and kron(A,B) means A on PAIR[1], B on PAIR[0].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_S2 = 1.0 / np.sqrt(2.0)
SQI = np.array([[1, 0, 0, 0], [0, _S2, 1j * _S2, 0],
                [0, 1j * _S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
HAD = np.array([[1, 1], [1, -1]], dtype=complex) * _S2
I2 = np.eye(2, dtype=complex)


# ── (1) anchor phase convention ─────────────────────────────────────────────

def phase_fix_anchors(C4):
    """Rotate each anchor so its dominant CI coefficient is real positive.

    Returns (C4_phased, phases) where phases[j] is the applied e^{-i arg}.
    Accepts jax or numpy arrays; returns numpy complex128.
    """
    out, ph = [], []
    for C in C4:
        C = np.asarray(C, dtype=complex)
        k = np.unravel_index(np.argmax(np.abs(C)), C.shape)
        p = np.exp(-1j * np.angle(C[k]))
        ph.append(p)
        out.append(C * p)
    return out, np.array(ph)


def anchor_phase_report(C4):
    """Degrees of the dominant coefficient of each anchor, pre-fix (diagnostic)."""
    d = []
    for C in C4:
        C = np.asarray(C, dtype=complex)
        k = np.unravel_index(np.argmax(np.abs(C)), C.shape)
        d.append(np.degrees(np.angle(C[k])))
    return np.array(d)


# ── (2) local-Z gauge stripping ─────────────────────────────────────────────

def _z4(a, b):
    """diag local-Z on the pair in the (|00>,|10>,|01>,|11>) basis."""
    return np.diag([1.0, np.exp(1j * a), np.exp(1j * b), np.exp(1j * (a + b))])


def strip_localZ(M, M_ref=SQI, ntries=40, seed=0):
    """Maximise the RAW overlap |<M_ref | z_L M z_R>| over 4 phases.

    Returns (M_stripped, (aL,bL,aR,bR), raw_overlap_before, raw_overlap_after)
    where raw_overlap = |<M_ref,M>|/(||M_ref|| ||M||).

    NOTE the maximisation is over the RAW overlap, not F_bar: F_bar is already
    local-Z invariant, so it cannot select the frame.  This is what folds into
    the local layers of a decomposition.
    """
    M = np.asarray(M, dtype=complex)
    nrm = np.linalg.norm(M_ref) * np.linalg.norm(M)

    def negov(p):
        return -abs(np.vdot(M_ref, _z4(p[0], p[1]) @ M @ _z4(p[2], p[3]))) / nrm

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(ntries):
        r = minimize(negov, rng.uniform(0, 2 * np.pi, 4), method="BFGS",
                     options=dict(maxiter=5000, gtol=1e-14))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    Ms = _z4(p[0], p[1]) @ M @ _z4(p[2], p[3])
    return Ms, p, abs(np.vdot(M_ref, M)) / nrm, -best.fun


# ── locality diagnostic ─────────────────────────────────────────────────

def schmidt(M):
    """Operator-Schmidt spectrum: rank 1 <=> M = A (x) B, i.e. genuinely local.

    kron(A,B) -> [2,0,0,0];  SWAP -> [1,1,1,1].
    """
    M = np.asarray(M, dtype=complex)
    return np.linalg.svd(M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4),
                         compute_uv=False)


# ── SU(2) parameterisation + the ideal-level decomposition ───────────────────

def su2(p):
    a, b, c = p
    return np.array([[np.cos(a / 2) * np.exp(1j * (b + c) / 2),
                      np.sin(a / 2) * np.exp(1j * (b - c) / 2)],
                     [-np.sin(a / 2) * np.exp(-1j * (b - c) / 2),
                      np.cos(a / 2) * np.exp(-1j * (b + c) / 2)]])


def solve_decomposition(target=SWAP, E=SQI, nlegs=3, ntries=300, seed=0):
    """target = L_n E ... E L_1 E L_0 with Lk = kron(a_k, b_k), E the IDEAL gate.

    Solved against the IDEAL E, not fitted to a measured leg: that is how a
    designer deploys a decomposition, and fitting to M_leg would calibrate away
    part of the leg error and flatter the result.
    Returns (L, fidelity) with L a list of (a_k, b_k) 2x2 pairs, len nlegs+1.
    """
    d = target.shape[0]

    def build(x):
        M = np.kron(su2(x[0:3]), su2(x[3:6]))
        for k in range(nlegs):
            M = E @ M
            o = 6 * (k + 1)
            M = np.kron(su2(x[o:o + 3]), su2(x[o + 3:o + 6])) @ M
        return M

    def f(x):
        return -abs(np.vdot(target, build(x))) ** 2 / d ** 2

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(ntries):
        r = minimize(f, rng.uniform(0, 2 * np.pi, 6 * (nlegs + 1)), method="BFGS",
                     options=dict(maxiter=5000, gtol=1e-14))
        if best is None or r.fun < best.fun:
            best = r
        if -best.fun > 1 - 1e-13:
            break
    L = [(su2(best.x[6 * k:6 * k + 3]), su2(best.x[6 * k + 3:6 * k + 6]))
         for k in range(nlegs + 1)]
    return L, -best.fun
