"""Shared machinery for the exact (full-CI) reference propagation.

The exact state is the product-grid tensor psi[i0,i1,i2,i3], i_s = 0..63, and the
Hamiltonian is

    H(t) = sum_s h_s(t) acting on axis s  +  V(x0,x1,x2,x3)

with V a single multiplicative diagonal built by broadcasting the six DVR pair
tables.  No two-body tensor is ever constructed.

Everything that is not the wavefunction ansatz is imported from the MCTDH code
(lib/, routed_gates/): the geometry, the erf ramp/hold schedule, the target
points, the anchors, the qubit orbitals and the scoring primitives.  Two
exceptions:

  * PULSE_SETS mirrors routed_gates/slurm/_common.sh, which is shell and cannot
    be imported.  It is parsed from that file at import and asserted equal.
  * fbar16() assembles the three-line F_bar formula of run_route_sa.score and
    run_arm_decomp._score_matrix from target_sqiswap_03 and fxc_4q; a
    self-test scores a stored MCTDH M back to its stored Fbar16 to 1e-12.

THE TWO-BODY FACTOR, checked not assumed.  MCTDHF_n_species._transform_all_g
fills BOTH (s,t) and (t,s), and SemiDirectCI.precompute_ci_ops then applies
factor 0.5 to each, so the physical Hamiltonian is
    H = sum_s h_s + sum_{s<t} V(x_s, x_t)
with g_dict[(s,t)] entering once, unscaled (checked numerically against the
solver's own contraction path).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER_DIR = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(ER_DIR, ".."))         # one_electron
LIB = os.path.join(ROOT, "lib")
RW = os.path.join(ROOT, "routed_gates")
GS = os.path.join(ROOT, "gate_sweeps")
sys.path.insert(0, LIB)

RESULTS = os.path.join(ER_DIR, "results")
DIAG = os.path.join(ER_DIR, "diagnostics")

# ── pulse sets, mirrored from routed_gates/slurm/_common.sh and asserted ──────
#
# The shell file holds the four leg pulses.  It cannot be imported, so it is
# parsed and compared.

PULSE_SETS = {
    "best": {"w0w1": (82.3, 73.2), "w1w2": (57.2, 95.4), "w2w3": (34.1, 112.8)},
    "fast": {"w0w1": (82.3, 73.2), "w1w2": (8.9, 137.1), "w2w3": (11.9, 131.7)},
    "sync": {"w0w1": (57.2, 92.7), "w1w2": (57.2, 95.4), "w2w3": (57.2, 92.7)},
}
TAU_W0W3 = (81.3, 2160.7)


def _check_pulse_sets():
    src = open(os.path.join(RW, "slurm", "_common.sh")).read()
    for name, table in PULSE_SETS.items():
        m = re.search(rf"^\s*(?:[a-z|]*\|)?{name}\)\s*export\s+(.*?);;", src,
                      re.M | re.S)
        assert m, f"pulse set {name!r} not found in _common.sh"
        got = dict(re.findall(r"TAU_(W\dW\d)=([\d.,]+)", m.group(1)))
        for key, (tr, th) in table.items():
            shell = got[key.upper()]
            assert shell == f"{tr:g},{th:g}", (
                f"PULSE_SETS[{name}][{key}] = {(tr, th)} but _common.sh says "
                f"{shell}; the two must agree")
    m = re.search(r"^export TAU_W0W3=([\d.,]+)", src, re.M)
    assert m and m.group(1) == f"{TAU_W0W3[0]:g},{TAU_W0W3[1]:g}", \
        f"TAU_W0W3 mirror forked: {m and m.group(1)}"


_check_pulse_sets()


def leg_T(taus):
    return round(2 * taus[0] + taus[1], 6)


# ── jax / MCTDH imports ─────────────────────────────────────────────────────
# qoct_gate_leg pins jax_platform_name=cpu at import (it is a VMF module).  That
# only sets the DEFAULT backend; the GPU is selected per-run with
# jax.default_device, exactly as run_arm_decomp does.  TAU_SQI12 must be set
# before run_route_par is imported or it aborts on a missing entangler pulse.
os.environ.setdefault("TAU_SQI12", "57.2,95.4")

import jax                                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                           # noqa: E402

from qoct_gate_leg import G_DICT_64, NUM_ORB, mctdh_overlap       # noqa: E402
from rte_step import NUM_DVR, NUM_WELLS                           # noqa: E402
from geometry_config import ANCHOR_PATHS                          # noqa: E402
from mctdhf import MCTDHF_n_species_DVR, unpack_y_to_Cb           # noqa: E402
import run_route_sa as _B                                         # noqa: E402
import run_route_par as _P                                        # noqa: E402

from common import SQI, SWAP, HAD, phase_fix_anchors, \
    anchor_phase_report, strip_localZ                             # noqa: E402

assert NUM_WELLS == 4 and NUM_DVR == 64, "the einsum strings below are 4x64"
assert NUM_ORB == 4, "the MCTDH side of the comparison is norb=4"

CONFIGS = _B.CONFIGS
N = NUM_DVR

# plateau vectors, read from the same anchor files the MCTDH runs read
RES = {"w0w1": jnp.asarray(np.load(ANCHOR_PATHS[2])["best_x"], dtype=jnp.float64),
       "w1w2": jnp.asarray(np.load(ANCHOR_PATHS["3a"])["best_x"], dtype=jnp.float64),
       "w2w3": jnp.asarray(np.load(ANCHOR_PATHS["4a"])["best_x"], dtype=jnp.float64),
       "w0w3": jnp.asarray(np.load(ANCHOR_PATHS[5])["best_x"], dtype=jnp.float64)}
RES_KEY = {"w0w1": 2, "w1w2": "3a", "w2w3": "4a", "w0w3": 5}
X_IDLE = jnp.asarray(_P.X1, dtype=jnp.float64)


# ── precision ────────────────────────────────────────────────────────────────

def dtypes(prec):
    if prec == "x64":
        return jnp.complex128, jnp.float64
    if prec == "x32":
        return jnp.complex64, jnp.float32
    raise SystemExit(f"prec must be x64|x32, got {prec!r}")


# ── the Hamiltonian ──────────────────────────────────────────────────────────

def build_Vd(rdtype=jnp.float64):
    """The whole two-body term as one 64^4 multiplicative diagonal.

    g_dict[(s,t)][i,j] = V(x^s_i, x^t_j) and s < t for every stored key, so the
    (64,64) table reshapes straight into axes (s,t) with singleton axes
    elsewhere -- row-major order already matches.
    """
    Vd = jnp.zeros((N,) * NUM_WELLS, dtype=rdtype)
    for (s, t), g in G_DICT_64.items():
        assert s < t
        shp = [1] * NUM_WELLS
        shp[s] = N
        shp[t] = N
        Vd = Vd + jnp.reshape(jnp.asarray(g, dtype=rdtype), shp)
    return Vd


def h_fns_leg(pair_key, taus, pulse_set=None):
    """Per-well h_s(t) for a single-pair leg, from the MCTDH code's builder."""
    tr, th = taus
    return [_P.make_h_par(s, RES[pair_key], tr, th) for s in range(NUM_WELLS)]


def h_fns_par(taus01, taus23):
    """Per-well h_s(t) for a CONCURRENT (0,1)+(2,3) leg: spliced plateau, per-well
    taus, the shorter pair idling at its plateau -- propagated, not assumed."""
    taus = {0: taus01, 1: taus01, 2: taus23, 3: taus23}
    return [_P.make_h_par(s, _P.X_PAR, *taus[s]) for s in range(NUM_WELLS)]


def h_idle():
    """The idle one-body Hamiltonians: envelope identically 0, i.e. the plateau
    vector the ITE anchors were prepared at.

    NOTE the schedule's own h_s(0) is NOT exactly this: the erf top-hat is at
    -3 sigma at t = 0, so envelope(0) = 1.1e-5.  Both MCTDH and exact see the
    same h_s(0), so the comparison is unaffected; the exact ANCHORS are defined
    against the pure idle H, which is what the anchors were prepared with.
    """
    return [_P.make_h_par(s, X_IDLE, 1.0, 0.0) for s in range(NUM_WELLS)]


def h_at(h_fns, t, rdtype=jnp.float64):
    return [jnp.asarray(h_fns[s](jnp.asarray(float(t), jnp.float64)), dtype=rdtype)
            for s in range(NUM_WELLS)]


_SUBS = ["ai,ibcd->abcd", "bi,aicd->abcd", "ci,abid->abcd", "di,abci->abcd"]
# with a leading ensemble axis, so one call serves all states at once
_SUBS_B = ["ai,zibcd->zabcd", "bi,zaicd->zabcd",
           "ci,zabid->zabcd", "di,zabci->zabcd"]


def hpsi(hs, Vd, v):
    """H|psi>: four one-body contractions plus one elementwise multiply.

    Accepts a single (64,)*4 state or a batch (B,) + (64,)*4; Vd broadcasts over
    the leading axis either way.  split_op grew the batched form first and this
    did not follow, which is what broke the Chebyshev reference on its first run.
    """
    subs = _SUBS_B if v.ndim == NUM_WELLS + 1 else _SUBS
    out = Vd * v
    for s in range(NUM_WELLS):
        out = out + jnp.einsum(subs[s], hs[s], v, preferred_element_type=v.dtype)
    return out


def apply_axis(U, s, v):
    """Apply a 64x64 one-body operator along axis s (single-qubit layers).

    Batch-aware, like hpsi.  The other rank-4 helpers (expand, project_orb,
    occupations) work on a single state v[k]; see the batch_ok() self-test.
    """
    subs = _SUBS_B if v.ndim == NUM_WELLS + 1 else _SUBS
    return jnp.einsum(subs[s], jnp.asarray(U, dtype=v.dtype), v,
                      preferred_element_type=v.dtype)


def batch_ok():
    """Assert every batch-aware helper agrees with its per-state form.

    Cheap enough to run at import of any driver that batches.
    """
    rng = np.random.default_rng(0)
    v1 = jnp.asarray(rng.normal(size=(N,) * NUM_WELLS)
                     + 1j * rng.normal(size=(N,) * NUM_WELLS), dtype=jnp.complex128)
    vb = jnp.stack([v1, 2.0 * v1])
    hs = h_at(h_idle(), 0.0)
    Vd = build_Vd(jnp.float64)
    worst = float(jnp.abs(hpsi(hs, Vd, vb)[1] - 2.0 * hpsi(hs, Vd, v1)).max())
    U = np.linalg.qr(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))[0]
    for s in range(NUM_WELLS):
        worst = max(worst, float(jnp.abs(apply_axis(U, s, vb)[1]
                                         - 2.0 * apply_axis(U, s, v1)).max()))
    assert worst < 1e-10, f"batched helpers disagree with per-state: {worst:.2e}"
    return worst


def cdot(a, b):
    """<a|b>, ACCUMULATED IN complex128 whatever the state dtype.

    A 16.8M-element reduction in float32 carries ~1e-4 relative rounding, which
    would put a floor under the x32 precision gate for a reason that has nothing
    to do with the physics.  Inner products are ~2000x cheaper than a matvec, so
    buying them at double precision costs nothing measurable.
    """
    # rank-agnostic: reshape(-1) is a free view for contiguous arrays, and it
    # lets the same inner product serve the 64^4 grid and a 256-dim CI vector.
    return jnp.einsum("i,i->", a.reshape(-1).conj(), b.reshape(-1),
                      preferred_element_type=jnp.complex128)


def norm2(v):
    return jnp.real(cdot(v, v))


# ── MCTDH <-> exact ──────────────────────────────────────────────────────────

def expand(C, bs0):
    """psi[ijkl] = sum_abcd C[abcd] bs0[0][i,a] bs0[1][j,b] ... -- the MCTDH state
    written out exactly.  Bit-identical initial conditions."""
    v = jnp.einsum("ia,abcd->ibcd", jnp.asarray(bs0[0]), jnp.asarray(C))
    v = jnp.einsum("jb,ibcd->ijcd", jnp.asarray(bs0[1]), v)
    v = jnp.einsum("kc,ijcd->ijkd", jnp.asarray(bs0[2]), v)
    v = jnp.einsum("ld,ijkd->ijkl", jnp.asarray(bs0[3]), v)
    return v


def project_orb(bs0, v):
    """c[abcd] = <phi_a phi_b phi_c phi_d | psi>, the exact state's component in
    the MCTDH orbital-product space.

    Every anchor lives in this 256-dimensional space (they are norb=4 MCTDH
    states sharing bs0), so BOTH the 16x16 gate matrix and the computational-
    subspace projector reduce to this one contraction -- ~1/16 of a matvec,
    against 16 full 64^4 inner products done naively.
    """
    c = jnp.einsum("ia,ijkl->ajkl", jnp.asarray(bs0[0]).conj(), v)
    c = jnp.einsum("jb,ajkl->abkl", jnp.asarray(bs0[1]).conj(), c)
    c = jnp.einsum("kc,abkl->abcl", jnp.asarray(bs0[2]).conj(), c)
    c = jnp.einsum("ld,abcl->abcd", jnp.asarray(bs0[3]).conj(), c)
    return c


def anchor_overlaps(C16, bs0, v):
    """<anchor_i | psi> for all 16, via the orbital-space projection."""
    c = project_orb(bs0, v).astype(jnp.complex128)
    return jnp.stack([jnp.sum(jnp.asarray(C16[i]).conj() * c) for i in range(16)])


def load_primary_anchors():
    """The 16 MCTDH anchors, phase-fixed exactly as run_arm_decomp does."""
    C16, bs0 = _B.basis_states()
    before = anchor_phase_report(C16)
    C16f, _ = phase_fix_anchors(C16)
    C16f = [jnp.asarray(c, dtype=jnp.complex128) for c in C16f]
    return C16f, bs0, before


# run_arm_decomp cannot be imported without the PULSE_SET/TAU_* environment and
# creates result directories on import, so qubit_basis16 is reproduced here
# from run_arm_decomp.qubit_basis16.
def orthonormal_gauge(C16, bs0):
    """Return the SAME physical states in the orthonormal orbital gauge.

    MCTDH has an exact gauge freedom: (b_s T_s, C x_s T_s^-1) represents the same
    Psi.  With T_s from the QR factorisation and the positive-diagonal sign
    convention, this is regauge.make_regauge (applied by the MCTDH runs once per
    SNAP_DT chunk), done once at t = 0.

    WHY IT IS NEEDED HERE.  The stored anchor's `best_y` is complex64, so the
    anchor orbital set is orthonormal only to 2.6e-7 (float32 eps).  MCTDH's own
    generator drops the overlap factors S = b^H b (transform_h gives b^H h b and
    get_sigma assumes orthonormality), so with a non-orthonormal b the CI
    Hamiltonian is not <Phi_I|H|Phi_J> and the sigma identity cannot hold better
    than ||S - I||.

    IT CHANGES NOTHING PHYSICAL.  expand(C, b) == expand(R C, Q) exactly, so the
    exact initial state is identical either way; only the representation moves.
    """
    C_out, b_out, Rs = [], [], []
    for s in range(NUM_WELLS):
        b = np.asarray(bs0[s], dtype=complex)
        Q, R = np.linalg.qr(b)
        sgn = np.sign(np.real(np.diag(R)))
        sgn = np.where(sgn == 0, 1.0, sgn).astype(R.dtype)
        Q = Q * sgn[None, :]
        R = sgn[:, None] * R
        b_out.append(jnp.asarray(Q, dtype=jnp.complex128))
        Rs.append(R)
    for C in C16:
        C = np.asarray(C, dtype=complex)
        for s in range(NUM_WELLS):
            C = np.moveaxis(np.tensordot(Rs[s], np.moveaxis(C, s, 0),
                                         axes=([1], [0])), 0, s)
        C_out.append(jnp.asarray(C, dtype=jnp.complex128))
    return C_out, b_out


def qubit_basis16(s, C16, bs0):
    """Qubit orbitals of well s: dominant natural orbital of the well-s 1-RDM,
    from the anchors labelled n_s = 0 and n_s = 1 (all other wells in |0>)."""
    B = np.asarray(bs0[s])

    def nat(C):
        C = np.asarray(C)
        Cs = np.moveaxis(C, s, 0).reshape(C.shape[s], -1)
        w, v = np.linalg.eigh(Cs @ Cs.conj().T)
        return B @ v[:, -1], w[-1] / w.sum()

    zero = tuple(0 for _ in range(NUM_WELLS))
    one = tuple(1 if k == s else 0 for k in range(NUM_WELLS))
    j0 = CONFIGS.index(zero)
    j1 = CONFIGS.index(one)
    c0, o0 = nat(C16[j0])
    c1, o1 = nat(C16[j1])
    Q, _ = np.linalg.qr(np.stack([c0, c1], 1))
    dev = 0.0
    for j in range(16):
        cj, _ = nat(C16[j])
        ref = c1 if CONFIGS[j][s] == 1 else c0
        dev = max(dev, float(np.sqrt(max(0.0, 1.0 - abs(np.vdot(ref, cj)) ** 2))))
    return Q, (o0, o1, abs(np.vdot(c0, c1)), dev)


# ── circuit primitives (as run_arm_decomp) ───────────────────────
# B = sqiSWAP . (H x H)(Sd x Sd);  B^3 = e^{-i pi/4} SWAP exactly.
SDG = np.diag([1.0, -1j]).astype(complex)
BLOCK_U2 = HAD @ SDG


def layer_U64(Q, u2):
    """One-body operator implementing u2 in the span of Q: U = I - QQ^H + Q u2 Q^H."""
    Q = np.asarray(Q)
    return (np.eye(NUM_DVR, dtype=complex) - Q @ Q.conj().T
            + Q @ np.asarray(u2, dtype=complex) @ Q.conj().T)


_SUBS_PROJ = ["i,ijkl->jkl", "j,ijkl->ikl", "k,ijkl->ijl", "l,ijkl->ijk"]


def occupations(q1s, v):
    """n_s = <psi| |q1_s><q1_s| |psi> per well."""
    out = []
    for s in range(NUM_WELLS):
        u = jnp.einsum(_SUBS_PROJ[s], jnp.asarray(q1s[s], dtype=v.dtype).conj(), v,
                       preferred_element_type=v.dtype)
        out.append(jnp.real(jnp.sum(jnp.abs(u.astype(jnp.complex128)) ** 2)))
    return jnp.stack(out)


# ── scoring ──────────────────────────────────────────────────────

_BIT = np.array([(c[0] << 3) | (c[1] << 2) | (c[2] << 1) | c[3] for c in CONFIGS])
_PERM = np.zeros((16, 16))
for _q in range(16):
    _PERM[_BIT[_q], _q] = 1.0


def to_bit_order(M_cfg):
    """M indexed in CONFIGS (iter_bitstrings) order -> plain binary bit order."""
    return _PERM @ np.asarray(M_cfg) @ _PERM.T


def _sqiswap_general(pair):
    a, b = pair
    U = np.zeros((16, 16), complex)
    for i in range(16):
        q = [(i >> (3 - k)) & 1 for k in range(4)]
        for li in range(4):
            qa, qb = (li >> 1) & 1, li & 1
            amp = SQI[2 * q[a] + q[b], 2 * qa + qb]
            if amp != 0:
                o = list(q)
                o[a], o[b] = qa, qb
                U[sum(bit << (3 - k) for k, bit in enumerate(o)), i] = amp
    return U


# the general builder must reproduce the (0,3) target of run_route_sa bit for bit
assert np.array_equal(_sqiswap_general((0, 3)), _B.target_sqiswap_03()), \
    "pair-general sqiSWAP target disagrees with target_sqiswap_03()"


def target_sqiswap(pair):
    """sqiSWAP on `pair` (x) identity elsewhere, plain bit order -- the target
    confirm_k16 scores a single leg against.  (0,3) returns run_route_sa's
    target_sqiswap_03() object."""
    if tuple(pair) == (0, 3):
        return _B.target_sqiswap_03()
    return _sqiswap_general(pair)


def fbar16(M_bit, pair=(0, 3)):
    """F_bar16 / F_xc / logical corner, via the primitives of run_route_sa.

    Default pair (0,3) is the target of the routed circuits.  Passing a leg's own pair scores
    it against sqiSWAP(pair), the convention of confirm_k16 / rescore_k16."""
    U = target_sqiswap(pair)
    fxc = _B.fxc_4q(U, np.asarray(M_bit))
    d = 16
    Fbar = (np.vdot(M_bit, M_bit).real
            + fxc * np.vdot(U, U).real * np.vdot(M_bit, M_bit).real) / (d * (d + 1))
    spect = [s for s in range(NUM_WELLS) if s not in pair]
    sub = [i for i in range(16) if all(((i >> (3 - s)) & 1) == 0 for s in spect)]
    return float(Fbar), float(fxc), np.asarray(M_bit)[np.ix_(sub, sub)]


def fbar4(M_ref, M4):
    from qoct_gate_leg import avg_gate_fidelity_up_to_localZ as F
    return float(F(M_ref, M4))


def corner4(M16_bit, pair):
    a, b = pair
    idx = []
    for j in range(4):
        na, nb = j & 1, (j >> 1) & 1
        bits = [0, 0, 0, 0]
        bits[a], bits[b] = na, nb
        idx.append((bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3])
    return np.asarray(M16_bit)[np.ix_(idx, idx)]


# ── MCTDH checkpoint reading (no propagation, no GPU) ────────────────────────

def read_ckpt(path):
    """Unpack a routed_gates per-schedule-step checkpoint into (C, bs)."""
    y = jnp.asarray(np.load(path))
    return unpack_y_to_Cb(y, 16, [1] * NUM_WELLS, [NUM_DVR] * NUM_WELLS,
                          [NUM_ORB] * NUM_WELLS)


def mctdh_solver(h_fns, num_states=16):
    """A solver instance, only ever used for its integral-transform and CI
    contraction paths -- never propagated here."""
    sol = MCTDHF_n_species_DVR(
        h_list=h_fns, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=num_states, imag_time=False)
    assert sol.e_shift == 0.0, "e_shift must be 0 for x64 gate fidelity"
    return sol


def device(name=None):
    name = name or os.environ.get("EXACT_DEV", "cuda")
    if name == "cpu":
        return jax.devices("cpu")[0]
    return jax.devices("cuda")[0]


def banner(title):
    print(f"\n=== {title} ===", flush=True)
