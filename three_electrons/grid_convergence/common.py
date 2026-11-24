"""Shared pieces of the leakage study: paths, single-well FCI at a control
point, and the projection of ML states onto products of single-well FCI
states.

Everything from ``gate_timing`` is imported, never edited. The projection is
``level_check.project``'s construction; the only new thing is
:func:`det_overlaps_tensor`, the same <Phi_K|chi_m> by contracting the
antisymmetrised FCI tensor with the orbitals instead of forming 3x3 minors
(validated against the minors route).
"""

import json
import os

import numpy as np

from gate_timing import mf_lib as L
from gate_timing.level_check import fci_levels, minors
from gate_timing.timing import FCIWell, h_well
from system import fourwell as fw
from mctdhf.MLMCTDHF_n_species import unpack_y_to_ABb
from mctdhf.SemiDirectCI_n_species import SemiDirectCI_n_species

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MF = os.path.join(os.path.dirname(HERE), "gate_timing")
MF_RESULTS = os.path.join(MF, "results")
SWEEP_DIR = "diagonalize_mix1e-4_dt0.01_tol1e-10_rg0.1_s16o4_rnstep"
LEGS3 = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3)}

# The representative pulse of the sweep's verification runs (w1w2).
TAU_RAMP, TAU_HOLD = 28.3, 105.4


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
    os.replace(tmp, path)


def x_scaled(leg):
    """(x_idle, x_target) in code units (GHz / 10)."""
    xi, xg = L.x_vectors(leg)
    return xi / fw.TYPICAL_FREQ_GHZ, xg / fw.TYPICAL_FREQ_GHZ


def well_levels(fci, w, a, b, k=10, vH=None):
    """Exact 3e FCI of well ``w`` at (alpha, beta) = (a, b) [code units]:
    ``(E, V, resid)`` with V (D, k) real, ascending E."""
    _H, E, V, r = fci_levels(fci, h_well(w, a, b, vH), k=k)
    return E, V, r


# ── projection onto single-well FCI products ──────────────────────────────

def chi_tensor(fci, V):
    """Antisymmetrised FCI tensors X[m, p, q, r] (64^3 per level) with
    X[m, p, q, r] = sgn(perm) c_m[sorted(p, q, r)]."""
    n, k = fci.n, V.shape[1]
    X = np.zeros((k, n, n, n))
    p, q, r = fci.dets.T
    for (a, b, c), s in (((p, q, r), 1), ((q, r, p), 1), ((r, p, q), 1),
                         ((q, p, r), -1), ((p, r, q), -1), ((r, q, p), -1)):
        X[:, a, b, c] = s * V.T
    return X


def slater_occ(norb, N=fw.NUM_PARTICLES):
    """Occupied-orbital triples of the MCTDHF determinants, in the
    library's determinant order (``SemiDirectCI_n_species.get_slater_dets``)."""
    I = SemiDirectCI_n_species.get_slater_dets(norb, N)
    return np.array([np.flatnonzero(d) for d in I])


def det_overlaps_tensor(X, b, occ):
    """c[K, m] = <Phi_K|chi_m> for MCTDHF determinants Phi_K built from the
    orbitals ``b`` (64, N): sum_{pqr} X[m,p,q,r] b*[p,i] b*[q,j] b*[r,k] at
    (i, j, k) = occ[K]. Equal to ``minors(b, dets, occ).conj().T @ V``."""
    bc = b.conj()
    Y = np.tensordot(X, bc, axes=([3], [0]))            # m p q k
    Y = np.einsum("mpqk,qj->mpjk", Y, bc)
    Y = np.einsum("mpjk,pi->mijk", Y, bc)
    return Y[:, occ[:, 0], occ[:, 1], occ[:, 2]].T


def det_overlaps_minors(fci, V, b, occ):
    """``level_check.project``'s route (3x3 minors)."""
    T = minors(b, fci.dets, occ)
    return T.conj().T @ V


def unpack(y, leg, layout):
    """(A, Bs, bs) from a flat state, without building the solver."""
    A, Bs, bs = unpack_y_to_ABb(
        np.asarray(y), num_states=4,
        num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_mctdhf_orbitals=layout.norbs(leg), num_spfs=layout.nspfs(leg))
    return np.asarray(A), [np.asarray(B) for B in Bs], [np.asarray(b) for b in bs]


def pair_amplitudes(A, Bs, leg, c_i, c_j):
    """amp[a, m, n] = <chi_m|_i <chi_n|_j <spectator SPFs| psi_a>, from the
    determinant overlaps c_w[K, m] = <Phi_K|chi_m> of the two active wells.
    The spectators carry one SPF each and are traced as that SPF."""
    i, j = L.LEGS[leg]
    ovl_i = c_i.conj().T @ Bs[i]                    # <chi_m|phi_k>
    ovl_j = c_j.conj().T @ Bs[j]
    sl = [slice(None)] + [slice(None) if s in (i, j) else 0
                          for s in range(fw.NUM_WELLS)]
    Ap = A[tuple(sl)]                               # (4, n_i, n_j)
    return np.einsum("akl,mk,nl->amn", Ap, ovl_i, ovl_j)


class Projector:
    """Projects states of one (leg, layout) onto |m>_i |n>_j, products of
    single-well FCI states ``V_i``, ``V_j`` (D, k), by the tensor route."""

    def __init__(self, fci, leg, layout, V_i, V_j):
        self.leg, self.layout = leg, layout
        self.i, self.j = L.LEGS[leg]
        self.X = {self.i: chi_tensor(fci, V_i), self.j: chi_tensor(fci, V_j)}
        self.occ = {w: slater_occ(layout.norbs(leg)[w]) for w in (self.i, self.j)}

    def weights(self, y):
        A, Bs, bs = unpack(y, self.leg, self.layout)
        c = {w: det_overlaps_tensor(self.X[w], bs[w], self.occ[w])
             for w in (self.i, self.j)}
        amp = pair_amplitudes(A, Bs, self.leg, c[self.i], c[self.j])
        return np.abs(amp) ** 2


def theta_from_W(W):
    """theta = 2 atan2(sqrt(W[|10> col, 0, 1]), sqrt(W[|10> col, 1, 0]));
    W[..., column, m, n]."""
    return 2.0 * np.arctan2(np.sqrt(W[..., 1, 0, 1]), np.sqrt(W[..., 1, 1, 0]))


def leak_traces(W):
    """P11 (|11> column on the four computational products), P30_03,
    P40_04 (|11> column), and the captured weight per column."""
    col = W[..., 3, :, :]
    P11 = col[..., 0, 0] + col[..., 1, 0] + col[..., 0, 1] + col[..., 1, 1]
    P30_03 = col[..., 3, 0] + col[..., 0, 3]
    P40_04 = col[..., 4, 0] + col[..., 0, 4]
    captured = W.sum(axis=(-1, -2))
    return P11, P30_03, P40_04, captured


def load_state_file(path, ml):
    """A relaxed-state npz (A, B0.., b0..) as a flat y for ``ml``
    (``exact_check.load_relaxed``'s lines, by explicit path)."""
    import jax.numpy as jnp
    d = np.load(path)
    W = range(fw.NUM_WELLS)
    return ml._ABb_to_y(jnp.asarray(d["A"]),
                        [jnp.asarray(d[f"B{s}"]) for s in W],
                        [jnp.asarray(d[f"b{s}"]) for s in W])
