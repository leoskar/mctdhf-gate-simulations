"""16-state register circuits from sqiSWAP legs and single-qubit layers (Sec. IV A 2).

Every SWAP is three sqiSWAP legs, each followed by a fixed single-qubit layer:

    SWAP(a,b) = B^3 . e^{+i pi/4},   B = E . (H(x)H)(Sd(x)Sd),  E = sqiSWAP(a,b)

Each leg's local-Z frame (measured on the 4x4 pair block of the leg's 16x16
matrix, against the ideal sqiSWAP) is applied as a virtual-Z frame update
directly next to the leg, z_R before and z_L after; the layers themselves stay
H . Sd.  Composition of one-body layers is exact
(U(u)U(v) = I - QQ^H + Q u v Q^H), so this is bookkeeping, not physics.

Modes (env ARM):
  legs16    each distinct sqiSWAP leg propagated alone in SA-16; stores its
            16x16 matrix M_leg (the other modes need it first).
  serial    SWAP(2,3) SWAP(1,2) sqiSWAP(0,1) SWAP(1,2) SWAP(2,3)
            = 13 sqiSWAP legs + 12 single-qubit layers.
  parallel  [SWAP(0,1) || SWAP(2,3)] sqiSWAP(1,2) [SWAP(0,1) || SWAP(2,3)]
            = 7 leg slots + 6 four-well layers.  The two pairs have different
            leg durations, so the shorter pair waits at the idle point inside
            the slot; that idle time is propagated.
  direct    the single w0w3 leg (the paper's direct gate is the 16-state
            confirmation in gate_sweeps/scripts/confirm_k16.py).

The pulses come from the environment (TAU_W0W1 ... TAU_W0W3, no defaults, set
by slurm/_common.sh), and PULSE_SET (best | fast | sync) names the output
folder results/arms_<set>/.  Basis states and scoring (16x16 M, F_bar16, (0,3)
pair block) are those of lib/run_route_sa.py; the parallel layers use
lib/run_route_par.make_h_par.

Conventions:
  (1) anchor phases are fixed so each anchor's dominant CI coefficient is real
      positive (lib/common.py).
  (2) the per-leg local-Z frame, as above.
  (3) a layer acts as U = I - QQ^H + Q u2 Q^H on that well's orbitals; the
      qubit orbitals Q are the dominant natural orbitals of the well's 1-RDM.
  (4) kron(A,B) means A on PAIR[1], B on PAIR[0].

The per-leg frame corrects the active pair's Z frame only.  No virtual-Z frame
is tracked on the spectator wells across legs: those phases do not commute
through later legs in which the spectator becomes active, so they stay in the
circuit and are not absorbed by score()'s final per-qubit-Z gauge.
"""
from __future__ import annotations
import os, sys, time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RG_DIR = os.path.abspath(os.path.join(HERE, ".."))
LIB = os.path.abspath(os.path.join(RG_DIR, "..", "lib"))   # one_electron/lib
for p in (HERE, LIB):
    sys.path.insert(0, p)

os.environ.setdefault("TAU_SQI12", "30.0,116.8")   # consumed by run_route_par import

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import qoct_gate_leg as _gl
from qoct_gate_leg import (DT_PROD, G_DICT_64, NUM_ORB, _make_h_s_x64,
                           mctdh_overlap)
from sweep_map import SNAP_DT
from rte_step import NUM_DVR, NUM_WELLS
from geometry_config import ANCHOR_PATHS
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_cmf_half_step_integrator
from regauge import make_regauge

import run_route_sa as _B
import run_route_par as _P

from common import (SQI, SWAP, HAD, strip_localZ,
                    phase_fix_anchors, anchor_phase_report, schmidt)

ARM = os.environ.get("ARM", "legs16")
DT_DIV = float(os.environ.get("SWEEP_DT_DIV", "0.1"))          # 0.1 => 10 x dt
DT = DT_PROD / DT_DIV
STEPS = int(round(SNAP_DT / DT))
assert STEPS >= 1 and abs(STEPS * DT - SNAP_DT) < 1e-12, \
    f"dt={DT} does not divide the SNAP_DT={SNAP_DT} chunk grid"
DEV = os.environ.get("SWEEP_DEV", "cuda")
assert NUM_ORB == 4, f"production arms use norb=[4,4,4,4]; qoct_gate_leg has {NUM_ORB}"
if abs(DT_DIV - 0.1) > 1e-12:
    # Every production arm uses 10dt.  A stability diagnostic
    # needs to vary it, so there is one narrow escape -- explicit, and loud.
    assert os.environ.get("ALLOW_DT_OVERRIDE") == "1", \
        f"production arms use 10dt (SWEEP_DT_DIV=0.1); got {DT_DIV}"
    print(f"  !! DT OVERRIDE: SWEEP_DT_DIV={DT_DIV}, not the production "
          f"0.1. Requires explicit ALLOW_DT_OVERRIDE=1. Results at this step are "
          f"NOT like-for-like with 10dt arms and must be labelled with their dt.",
          flush=True)

RES = {"w0w1": jnp.asarray(np.load(ANCHOR_PATHS[2])["best_x"], dtype=jnp.float64),
       "w1w2": jnp.asarray(np.load(ANCHOR_PATHS["3a"])["best_x"], dtype=jnp.float64),
       "w2w3": jnp.asarray(np.load(ANCHOR_PATHS["4a"])["best_x"], dtype=jnp.float64),
       # direct long-range arm: stage-5 w0w3 anchor + the hold-scan maximum
       # pulse, T = 2*10 + 2111.3 = 2131.3 t.u.  Same values as
       # diag_integ_screen.LEG_SPECS["w0w3"], read from the same file.
       "w0w3": jnp.asarray(np.load(ANCHOR_PATHS[5])["best_x"], dtype=jnp.float64)}

# Calibrated sqiSWAP legs, from the environment (slurm/_common.sh).
def _tau(name):
    """(tau_ramp, tau_hold) from the environment.  No default: an
    unset value must abort at import, never propagate as a placeholder."""
    v = os.environ[name]
    tr, th = (float(x) for x in v.split(","))
    assert tr > 0 and th > 0, f"{name}={v!r}: both taus must be positive"
    return (tr, th)


TAU = {"w0w1": _tau("TAU_W0W1"), "w1w2": _tau("TAU_W1W2"),
       "w2w3": _tau("TAU_W2W3"), "w0w3": _tau("TAU_W0W3")}
SET = os.environ["PULSE_SET"]      # 'best' | 'fast' | 'sync', names the output
assert SET in ("best", "fast", "sync"), \
    f"PULSE_SET must be best|fast|sync, got {SET!r}"
PAIRS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3), "w0w3": (0, 3)}
T_LEG = {k: round(2 * tr + th, 6) for k, (tr, th) in TAU.items()}

OUT = os.path.join(RG_DIR, "results", f"arms_{SET}")
os.makedirs(OUT, exist_ok=True)
# LEGS_TAG names a second legs16 store (and its arms' checkpoints and outputs),
# so a re-run at another dt neither overwrites legs16.npz nor resumes from the
# old arm's checkpoints.  Empty = the default names.
LEGS_TAG = os.environ.get("LEGS_TAG", "")
_SFX = f"_{LEGS_TAG}" if LEGS_TAG else ""
LEGS16 = os.path.join(OUT, f"legs16{_SFX}.npz")
CKPT = os.path.join(OUT, f"ckpt_{ARM}_dt{DT_DIV:g}{_SFX}")
os.makedirs(CKPT, exist_ok=True)


# ── anchors, qubit orbitals, layer application ───────────────────────────────

def load_anchors16():
    """16 anchors with fixed phases (lib/common.py), plus the shared orbital set."""
    C16, bs0 = _B.basis_states()
    ph_before = anchor_phase_report(C16)
    C16f, _ = phase_fix_anchors(C16)
    C16f = [jnp.asarray(c, dtype=jnp.complex128) for c in C16f]
    print(f"  anchor dominant-coeff phases (deg), before: "
          f"{np.round(ph_before, 1)}", flush=True)
    print(f"                                          after : "
          f"{np.round(anchor_phase_report(C16f), 6)}", flush=True)
    return C16f, bs0


def qubit_basis16(s, C16, bs0):
    """Qubit orbitals of well s: dominant natural orbital of the well-s 1-RDM,
    taken from the anchors labelled n_s = 0 and n_s = 1.

    Uses the two CLEANEST representatives (all other wells in |0>), then checks
    state-independence across all 16 anchors -- expected ~3e-4, and a
    silent violation would mean 'the qubit subspace' is not well defined.
    """
    B = np.asarray(bs0[s])
    cfg = _B.CONFIGS

    def nat(C):
        C = np.asarray(C)
        Cs = np.moveaxis(C, s, 0).reshape(C.shape[s], -1)
        w, v = np.linalg.eigh(Cs @ Cs.conj().T)
        return B @ v[:, -1], w[-1] / w.sum()

    zero = tuple(0 for _ in range(NUM_WELLS))
    one = tuple(1 if k == s else 0 for k in range(NUM_WELLS))
    j0 = cfg.index(zero)
    j1 = cfg.index(one)
    c0, o0 = nat(C16[j0]); c1, o1 = nat(C16[j1])
    Q, _ = np.linalg.qr(np.stack([c0, c1], 1))

    dev = 0.0
    for j in range(16):
        cj, _ = nat(C16[j])
        ref = c1 if cfg[j][s] == 1 else c0
        dev = max(dev, float(np.sqrt(max(0.0, 1.0 - abs(np.vdot(ref, cj)) ** 2))))
    return Q, (o0, o1, abs(np.vdot(c0, c1)), dev)


def make_layer_U(Q, u2):
    """One-body operator implementing u2 in the span of Q."""
    return np.eye(NUM_DVR, dtype=complex) - Q @ Q.conj().T + Q @ u2 @ Q.conj().T


def apply_layer(solver, y, assign, QB):
    """assign: {well: 2x2}.  Acts on the shared orbitals only; C is unchanged."""
    C, bs = solver._y_to_Cb(y)
    bs = list(bs)
    for s, u2 in assign.items():
        bs[s] = jnp.asarray(make_layer_U(QB[s], u2)) @ bs[s]
    return solver._Cb_to_y(C, bs)


# ── solvers and propagation ──────────────────────────────────────────────────

def make_solver_single(pair_key):
    """SA-16 solver for one sqiSWAP leg on `pair_key` (one active pair).

    The plateau vector is passed EXPLICITLY via run_route_par.make_h_par rather
    than through qoct_gate_leg's module global.  `_make_h_s_x64` closes over
    `x_star_2_ghz` and reads it at TRACE time, not at construction time, and
    `run_arm` builds all three leg solvers before jax traces any of them -- so
    the global-rebinding idiom (fine in run_route_sa, which builds and traces one
    leg at a time) would have given all three legs the LAST leg's resonance, with
    no error anywhere.  make_h_par is algebraically identical to _make_h_s_x64
    with x_star_2_ghz = x2; `probe_h` below pins that.
    """
    tr, th = TAU[pair_key]
    return _solver([_P.make_h_par(s, RES[pair_key], tr, th)
                    for s in range(NUM_WELLS)])


def probe_h(solver, t):
    """Fingerprint a solver's time-dependent one-body Hamiltonians at time t."""
    return np.array([float(jnp.sum(jnp.abs(h(jnp.asarray(t)))))
                     for h in solver.h_list])


def make_solver_par():
    """SA-16 solver for one CONCURRENT sqiSWAP(0,1)+sqiSWAP(2,3) sub-leg.

    Per-well taus and the entry-wise spliced plateau come from run_route_par;
    wells 2,3 finish their 175.2 t.u. leg and then sit at the idle plateau for
    the remaining 34.1 t.u. of the 209.3 t.u. sub-leg.
    """
    taus = {0: TAU["w0w1"], 1: TAU["w0w1"], 2: TAU["w2w3"], 3: TAU["w2w3"]}
    h_list = [_P.make_h_par(s, _P.X_PAR, *taus[s]) for s in range(NUM_WELLS)]
    return _solver(h_list), round(max(2 * tr + th for tr, th in taus.values()), 6)


def _solver(h_list):
    sol = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=16, imag_time=False)
    assert sol.e_shift == 0.0, "e_shift must be 0 for x64 gate fidelity"
    return sol


def make_chunk(solver):
    """One SNAP_DT of CMF half-step, then an EXACT gauge re-orthonormalisation.

    The regauge is mandatory, not cosmetic: b_propagator solves a frozen-projector
    substep whose generator is non-Hermitian, so exp(-i dt A_s) is not unitary and
    b^H b = I decays LINEARLY in T.  Without it this configuration lost a 1065 t.u.
    leg entirely (drift 9.91e-01).  See regauge.py.
    """
    integ = get_cmf_half_step_integrator(solver, DT, c_propagator="diagonalize",
                                         b_propagator="diagonalize")
    rg = make_regauge(solver)
    inner = jax.jit(lambda c: jax.lax.scan(
        lambda a, _: (integ(a[0], a[1]), None), c, None, length=STEPS)[0])
    return jax.jit(lambda c: (lambda tv: (tv[0], rg(tv[1])))(inner(c)))


def diagnostics(solver, y, n=16):
    C, bs = solver._y_to_Cb(y)
    nn = np.array([float(jnp.real(mctdh_overlap(C[j], bs, C[j], bs)))
                   for j in range(n)])
    ortho = max(float(np.abs(np.asarray(b).conj().T @ np.asarray(b)
                             - np.eye(NUM_ORB)).max()) for b in bs)
    return nn, ortho


CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "2000"))
ABORT_DRIFT = float(os.environ.get("ABORT_DRIFT", "1e-2"))


def propagate(solver, chunk, y, T_leg, label, midckpt=None, start_chunk=0):
    """Advance T_leg with per-SNAP_DT CMF chunks, from chunk index start_chunk."""
    n_chunks = int(round(T_leg / SNAP_DT))
    carry = (jnp.asarray(start_chunk * SNAP_DT), y)
    t0 = time.perf_counter()
    carry = chunk(carry); jax.block_until_ready(carry)
    print(f"    [{label}] compile+chunk1 {time.perf_counter()-t0:.1f}s "
          f"({n_chunks} chunks, T={T_leg}, from chunk {start_chunk})", flush=True)
    tic = time.perf_counter()
    for c in range(start_chunk + 1, n_chunks):
        carry = chunk(carry)
        if (c + 1) % 1000 == 0:
            jax.block_until_ready(carry)
            nn, ortho = diagnostics(solver, carry[1])
            drift = float(abs(nn - 1).max())
            el = time.perf_counter() - tic
            done = max(1, c - start_chunk)
            print(f"    [{label}] chunk {c+1}/{n_chunks} t={float(carry[0]):.1f} "
                  f"drift={drift:.2e} ortho={ortho:.2e} "
                  f"({el:.0f}s, eta {el/done*(n_chunks-c-1)/60:.1f}m)", flush=True)
            if drift > ABORT_DRIFT or not np.all(np.isfinite(np.asarray(carry[1]))):
                raise SystemExit(
                    f"ABORT [{label}] chunk {c+1} t={float(carry[0]):.1f}: drift "
                    f"{drift:.3e} > {ABORT_DRIFT:g} -- this step fails at this "
                    f"leg length; stopping rather than burning the rest of the run.")
        if midckpt and (c + 1) % CKPT_EVERY == 0:
            jax.block_until_ready(carry)
            np.save(midckpt + ".tmp.npy", np.asarray(carry[1]))
            np.save(midckpt + f".mid{c+1}.npy", np.asarray(carry[1]))
    jax.block_until_ready(carry)
    y_fin = carry[1]
    nn, ortho = diagnostics(solver, y_fin)
    fin = bool(np.all(np.isfinite(np.asarray(y_fin))))
    print(f"    [{label}] done {time.perf_counter()-tic:.0f}s  "
          f"drift={abs(nn-1).max():.2e}  ortho={ortho:.2e}  finite={fin}",
          flush=True)
    assert fin, f"leg {label} went non-finite"
    # END-OF-LEG ABORT.  The periodic check above only fires every 1000 chunks,
    # so the TAIL of every leg was unguarded: parallel/best ran
    # cleanly to chunk 2000 of its last leg (drift 3.89e-04), blew up in the
    # remaining 378 chunks to drift 2.89e-01, and still wrote a plausible-looking
    # F_bar16 = 0.408 because nothing checked again before scoring.  A silent
    # wrong number is the worst possible outcome, so the leg boundary is now a
    # hard gate as well.
    if float(abs(nn - 1).max()) > ABORT_DRIFT:
        raise SystemExit(
            f"ABORT [{label}] at leg end t={T_leg}: drift "
            f"{abs(nn-1).max():.3e} > {ABORT_DRIFT:g} -- the periodic check "
            f"passed but the leg tail diverged; refusing to score a corrupt state.")
    return y_fin, dict(drift=float(abs(nn - 1).max()), ortho=float(ortho))


# ── 4x4 active-pair corner of a 16x16 register matrix ────────────────────────

def corner4(M16, pair):
    """Rows/cols of M16 (BIT-ORDER, q0 most significant) with both spectators in
    |0>, ordered as the pair basis (|00>,|10>,|01>,|11>) with index =
    n_{pair[0]} + 2*n_{pair[1]}."""
    a, b = pair
    idx = []
    for j in range(4):
        na, nb = j & 1, (j >> 1) & 1
        bits = [0, 0, 0, 0]
        bits[a], bits[b] = na, nb
        idx.append((bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3])
    return M16[np.ix_(idx, idx)]


def _zfactors(a, b):
    """z(a,b) = diag(1,e^{ia},e^{ib},e^{i(a+b)}) = kron(hi,lo), hi on PAIR[1]."""
    return np.diag([1.0, np.exp(1j * b)]), np.diag([1.0, np.exp(1j * a)])


# ── the SWAP block  B = sqiSWAP . (H(x)H) . (Sd(x)Sd) ───────────────

SDG = np.diag([1.0, -1j]).astype(complex)        # S^dagger
BLOCK_U2 = HAD @ SDG                             # the per-qubit half of B's layer


def block_layer(pair):
    """B's single local layer, (H(x)H)(Sd(x)Sd), as {well: 2x2}.

    It is the SAME 2x2 on both wells of the pair, so it is trivially a product
    kron(a,b) and hence one-body implementable.
    """
    return {pair[0]: BLOCK_U2.copy(), pair[1]: BLOCK_U2.copy()}


def check_block():
    """B^3 = e^{-i pi/4} SWAP, exactly.  Returns the residual."""
    L = np.kron(BLOCK_U2, BLOCK_U2)
    B = SQI @ L
    return float(np.abs(B @ B @ B - np.exp(-1j * np.pi / 4) * SWAP).max())


# ── the leg's local-Z frame, applied ADJACENT to the leg ──────────

def zframe(p, pair, side):
    """Virtual-Z frame update for one measured leg, as {well: 2x2}.

    `strip_localZ` returns  M_stripped = z4(p0,p1) . M_leg . z4(p2,p3)  with
    z4(a,b) = diag(1, e^{ia}, e^{ib}, e^{i(a+b)}) in the (|00>,|10>,|01>,|11>)
    pair basis, index = n_{pair[0]} + 2 n_{pair[1]}.  So the LEFT
    factor acts AFTER the leg and the RIGHT factor BEFORE it, and the two
    together are exactly the frame in which the measured leg IS a sqiSWAP.

    This is applied next to the leg, not folded into block_layer(): the
    correction belongs to the sqiSWAP.  Composition of one-body layers on the
    same well is exact, so the choice costs nothing physically -- but it keeps
    the block's layers literally H . Sd, which is what makes B^3 = SWAP a
    checkable statement about the CIRCUIT rather than about a fitted object.
    """
    a, b = (p[0], p[1]) if side == "after" else (p[2], p[3])
    return {pair[0]: np.diag([1.0, np.exp(1j * a)]).astype(complex),
            pair[1]: np.diag([1.0, np.exp(1j * b)]).astype(complex)}


# ── ideal 16x16 image of a local layer (bit-order, q0 most significant) ───────

def layer16_ideal(assign):
    U = np.eye(1, dtype=complex)
    for s in range(NUM_WELLS):
        U = np.kron(U, assign.get(s, np.eye(2, dtype=complex)))
    return U


def M16_of(C16, bs0, solver, y):
    """16x16 gate matrix in BIT-ORDER -- the M half of run_route_sa.score(),
    without its fxc_4q optimisation (40 Nelder-Mead restarts, which would
    otherwise run once per LAYER).  Asserted identical to _B.score in stage2a."""
    C_T, bs_T = solver._y_to_Cb(y)
    M = np.zeros((16, 16), complex)
    for j in range(16):
        for i in range(16):
            M[i, j] = complex(mctdh_overlap(C16[i], bs0, C_T[j], bs_T))
    bit = np.array([(c[0] << 3) | (c[1] << 2) | (c[2] << 1) | c[3]
                    for c in _B.CONFIGS])
    P = np.zeros((16, 16))
    for q in range(16):
        P[bit[q], q] = 1.0
    return P @ M @ P.T


def layer16_implemented(solver, chunkless_y0, assign, QB, C16, bs0):
    """16x16 matrix of the layer as ACTUALLY implemented (one-body operator on
    the anchor qubit orbitals).  No propagation -- the layer only rotates the
    shared orbitals, so the finals are (C16[j], U bs0)."""
    return M16_of(C16, bs0, solver, apply_layer(solver, chunkless_y0, assign, QB))


# ── schedules ────────────────────────────────────────────────────────────────

def swap_block(key, pair, z):
    """One B, in PROPAGATION order (what the register meets first comes first).

    Matrix form  B = [z_L] E [z_R] . (H(x)H)(Sd(x)Sd), so propagation order is
    layer, z_R, leg, z_L.  Only the first is counted as a "layer": the z's are
    virtual-Z frame updates and cost no time.
    """
    return [("layer", block_layer(pair)),
            ("zframe", zframe(z, pair, "before")),
            ("leg", key),
            ("zframe", zframe(z, pair, "after"))]


def entangler(key, pair, z):
    """A bare sqiSWAP primitive: no block layer, only its own Z frame."""
    return [("zframe", zframe(z, pair, "before")),
            ("leg", key),
            ("zframe", zframe(z, pair, "after"))]


def par_block(z01, z23):
    """One concurrent B on (0,1) and (2,3) at once -- one four-well layer."""
    return [("layer", {**block_layer((0, 1)), **block_layer((2, 3))}),
            ("zframe", {**zframe(z01, (0, 1), "before"),
                        **zframe(z23, (2, 3), "before")}),
            ("leg", "par"),
            ("zframe", {**zframe(z01, (0, 1), "after"),
                        **zframe(z23, (2, 3), "after")})]


def coalesce(sched):
    """Merge each contiguous run of local operations into ONE application.

    WHY THIS IS NOT COSMETIC.  A one-body layer composes exactly:
    U(u)U(v) = (I-QQ^H) + Q u v Q^H = U(uv), so for the PROPAGATION merging
    changes nothing.  For the PRODUCT MODEL it changes a lot.  Each layer
    contributes a 16x16 matrix that is a PROJECTION of a unitary onto the anchor
    basis, with ||L||^2 = 15.968 rather than 16 because the anchors are not
    exactly product states in the qubit subspace.  Multiplying k such matrices
    compounds that projection k times, so splitting one physical rotation into
    (block layer, z_R) or (z_L, block layer, z_R) would charge the product model
    two or three projections for one instantaneous rotation and make it look
    leakier than the schedule it models.  Since the point is to compare the
    product model against the propagation, that bookkeeping asymmetry has to go:
    the number of matrix factors must equal the number of physical rotations.

    Applying e1 then e2 means bs <- U(u2)U(u1) bs, so the merged 2x2 is u2 @ u1.
    """
    out = []
    for kind, val in sched:
        if kind in ("layer", "zframe") and out and out[-1][0] == "local":
            merged = dict(out[-1][1])
            for w, u in val.items():
                merged[w] = u @ merged[w] if w in merged else u
            out[-1] = ("local", merged)
        elif kind in ("layer", "zframe"):
            out.append(("local", dict(val)))
        else:
            out.append((kind, val))
    return out


def serial_schedule(z):
    """Sequential:
    SWAP(2,3) SWAP(1,2) sqiSWAP(0,1) SWAP(1,2) SWAP(2,3)
    = 6x(2,3) + 6x(1,2) + 1x(0,1) = 13 legs + 12 layers.

    The matrix product is a palindrome, so propagation order and matrix order
    coincide and the list below reads the same either way.
    """
    sched = []
    for key, pair in (("w2w3", (2, 3)), ("w1w2", (1, 2))):
        for _ in range(3):
            sched += swap_block(key, pair, z[key])
    sched += entangler("w0w1", (0, 1), z["w0w1"])
    for key, pair in (("w1w2", (1, 2)), ("w2w3", (2, 3))):
        for _ in range(3):
            sched += swap_block(key, pair, z[key])
    return sched


def parallel_schedule(z01, z23, zent):
    """Parallel:
    [SWAP(0,1) || SWAP(2,3)] sqiSWAP(1,2) [SWAP(0,1) || SWAP(2,3)]
    = 6 concurrent leg-slots + 1x(1,2) = 7 leg-slots + 6 four-well layers.

    The two pairs have different leg durations, so the shorter pair IDLES at its
    plateau inside every concurrent slot; that idle time is propagated, not
    assumed away (make_h_par carries per-well taus).
    """
    sched = []
    for _ in range(3):
        sched += par_block(z01, z23)
    sched += entangler("w1w2", (1, 2), zent)
    for _ in range(3):
        sched += par_block(z01, z23)
    return sched


# ── main ─────────────────────────────────────────────────────────────────────

def _fbar4(M_ref, M):
    from qoct_gate_leg import avg_gate_fidelity_up_to_localZ as F
    return F(M_ref, M)


def stage2a(C16, bs0):
    """Each distinct leg propagated ALONE in SA-16; full 16x16 M_leg stored."""
    print("\n=== per-leg 16x16 matrices (SA-16) ===", flush=True)
    # pin the explicit-x2 builder against qoct_gate_leg's global-reading one
    _gl.x_star_2_ghz = RES["w1w2"]
    tr, th = TAU["w1w2"]
    _a = _make_h_s_x64(1, tr, th)(jnp.asarray(70.0))
    _b = _P.make_h_par(1, RES["w1w2"], tr, th)(jnp.asarray(70.0))
    _d = float(jnp.abs(_a - _b).max())
    print(f"  [check] make_h_par vs _make_h_s_x64: max|dh| = {_d:.2e}", flush=True)
    assert _d < 1e-12, "explicit-x2 Hamiltonian builder diverged from the global one"
    y0 = pack_Cb_to_y(jnp.stack(C16),
                      [jnp.asarray(b, dtype=jnp.complex128) for b in bs0])
    store = {}
    for key in ("w0w1", "w1w2", "w2w3", "par"):
        if key == "par":
            solver, T = make_solver_par()
        else:
            solver, T = make_solver_single(key), T_LEG[key]
        chunk = make_chunk(solver)
        y, diag = propagate(solver, chunk, y0, T, key)
        M16 = M16_of(C16, bs0, solver, y)
        if key == "w0w1":                # pin the fast path against score() once
            M_ref = _B.score(C16, bs0, solver, y)[0]
            d = float(np.abs(M16 - M_ref).max())
            print(f"    [check] M16_of vs run_route_sa.score(): max|dM| = {d:.2e}",
                  flush=True)
            assert d < 1e-12, "M16_of diverged from the shared scoring code"
        store[f"M16_{key}"] = M16
        store[f"diag_{key}"] = np.array([diag["drift"], diag["ortho"]])
        store[f"T_{key}"] = T
        pairs = [PAIRS[key]] if key != "par" else [(0, 1), (2, 3)]
        for pr in pairs:
            c4 = corner4(M16, pr)
            Ms, p, rb, ra = strip_localZ(c4, SQI)
            store[f"z_{key}_{pr[0]}{pr[1]}"] = p
            store[f"corner_{key}_{pr[0]}{pr[1]}"] = c4
            print(f"    [{key}] pair {pr}: F_bar4 = {_fbar4(SQI, c4):.6f}   "
                  f"||corner||^2 = {np.linalg.norm(c4)**2:.5f} (ideal 4)   "
                  f"raw overlap {rb:.4f} -> {ra:.6f} after Z-strip", flush=True)
        print(f"    [{key}] ||M16||^2 = {np.linalg.norm(M16)**2:.4f} (ideal 16)   "
              f"T = {T} t.u. = {T*0.0159155:.2f} ns", flush=True)
    dst = LEGS16
    assert not os.path.exists(dst), f"{dst} exists; refusing to overwrite"
    np.savez(dst, dt_div=DT_DIV, dt=DT, **store)
    print(f"  -> {dst}", flush=True)
    return store


def frames(store):
    """The measured per-leg local-Z frame parameters, keyed by leg.

    Nothing is fitted here: the block is fixed and only the frame is
    measured.
    """
    r = check_block()
    print(f"  B^3 vs e^{{-i pi/4}} SWAP, ideal block: max residual {r:.2e}",
          flush=True)
    assert r < 1e-12, f"the B^3 = SWAP identity does not hold ({r:.2e})"
    z = {k: store[f"z_{k}_{a}{b}"]
         for k, (a, b) in (("w0w1", (0, 1)), ("w1w2", (1, 2)), ("w2w3", (2, 3)))}
    z["par01"] = store["z_par_01"]
    z["par23"] = store["z_par_23"]
    for k, v in z.items():
        print(f"  z frame [{k}] = {np.round(v, 4)} rad", flush=True)
    return z


def run_arm(name, sched, C16, bs0, store):
    print(f"\n=== {name} arm ===", flush=True)
    n_legs = sum(1 for k, _ in sched if k == "leg")
    n_lay = sum(1 for k, _ in sched if k == "layer")
    n_z = sum(1 for k, _ in sched if k == "zframe")
    sched = coalesce(sched)
    n_app = sum(1 for k, _ in sched if k == "local")
    T_tot = sum(T_LEG[v] if v != "par" else float(store["T_par"])
                for k, v in sched if k == "leg")
    print(f"  {n_legs} sqiSWAP legs, {n_lay} local layers, {n_z} virtual-Z "
          f"frame updates (free), merged into {n_app} physical rotations, "
          f"T = {T_tot:.1f} t.u. = {T_tot*0.0159155:.2f} ns", flush=True)

    QB = {}
    for s in range(NUM_WELLS):
        Q, (o0, o1, ov, dev) = qubit_basis16(s, C16, bs0)
        QB[s] = Q
        print(f"  well {s} qubit orbitals: NO occ {o0:.5f}/{o1:.5f}  "
              f"|<c0|c1>| {ov:.1e}  state-dependence {dev:.1e}", flush=True)

    solvers, chunks = {}, {}
    for key in set(v for k, v in sched if k == "leg"):
        if key == "par":
            solvers[key], _T = make_solver_par()
        else:
            solvers[key] = make_solver_single(key)
        chunks[key] = make_chunk(solvers[key])
    any_solver = next(iter(solvers.values()))

    # Guard the trace-time-global hazard directly: distinct legs must have
    # DISTINCT Hamiltonians.  If make_solver_single ever reverts to reading
    # qoct_gate_leg.x_star_2_ghz, every entry here becomes identical and the arm
    # silently propagates one leg's pulse throughout.
    fp = {k: probe_h(v, 0.35 * (T_LEG[k] if k != "par" else float(store["T_par"])))
          for k, v in solvers.items()}
    for k, v in fp.items():
        print(f"  h-fingerprint[{k}] = {np.round(v, 6)}", flush=True)
    keys = sorted(solvers)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.allclose(fp[keys[i]], fp[keys[j]], rtol=0, atol=1e-9), (
                f"legs {keys[i]} and {keys[j]} have IDENTICAL Hamiltonians -- "
                f"the plateau vector is not being bound per leg")

    y = pack_Cb_to_y(jnp.stack(C16),
                     [jnp.asarray(b, dtype=jnp.complex128) for b in bs0])
    y0_ref = y

    start = 0
    for i in range(len(sched), 0, -1):
        f = os.path.join(CKPT, f"{name}_dt{DT_DIV:g}_step{i}.npy")
        if os.path.exists(f):
            y = jnp.asarray(np.load(f)); start = i
            print(f"  [resume] {f} -> starting at step {i+1}", flush=True)
            break

    # matrix-product model, built from the SAME schedule
    M_prod_impl = np.eye(16, dtype=complex)
    M_prod_idl = np.eye(16, dtype=complex)
    t_all = time.perf_counter()
    for i, (kind, val) in enumerate(sched):
        if kind == "local":
            Li = layer16_implemented(any_solver, y0_ref, val, QB, C16, bs0)
            M_prod_impl = Li @ M_prod_impl
            M_prod_idl = layer16_ideal(val) @ M_prod_idl
            if i >= start:
                y = apply_layer(any_solver, y, val, QB)
        else:
            M_prod_impl = store[f"M16_{val}"] @ M_prod_impl
            M_prod_idl = store[f"M16_{val}"] @ M_prod_idl
            if i >= start:
                T = T_LEG[val] if val != "par" else float(store["T_par"])
                y, _d = propagate(solvers[val], chunks[val], y, T,
                                  f"{name} {i+1}/{len(sched)} {val}")
        if i >= start:
            np.save(os.path.join(CKPT, f"{name}_dt{DT_DIV:g}_step{i+1}.npy"),
                    np.asarray(y))

    M, Fbar16, fxc, Mlog = _B.score(C16, bs0, any_solver, y)
    Fp, xp, Lp = _score_matrix(M_prod_impl)
    Fi, xi, Li_ = _score_matrix(M_prod_idl)

    print(f"\n=== RESULT: {name} arm ===", flush=True)
    print(f"  FULL PROPAGATION   F_bar16 = {Fbar16:.4f}   F_xc = {fxc:.4f}   "
          f"||M||^2 = {np.linalg.norm(M)**2:.3f} (ideal 16)", flush=True)
    print(f"  product, impl locals F_bar16 = {Fp:.4f}   F_xc = {xp:.4f}", flush=True)
    print(f"  product, ideal locals F_bar16 = {Fi:.4f}   F_xc = {xi:.4f}", flush=True)
    print(f"  (0,3) corner F_bar4  full = {_fbar4(SQI, Mlog):.4f}   "
          f"product = {_fbar4(SQI, Lp):.4f}", flush=True)
    print(f"  (0,3) corner survival per col: "
          f"{np.round((np.abs(Mlog)**2).sum(0), 3)}", flush=True)
    print(f"  |M_logical| (0,3 corner):\n{np.round(np.abs(Mlog), 3)}", flush=True)
    print(f"  wall {(time.perf_counter()-t_all)/3600:.2f} h", flush=True)

    dst = os.path.join(OUT, f"arm_{name}_dt{DT_DIV:g}{_SFX}_"
                            f"{datetime.now():%Y%m%d_%H%M%S}.npz")
    np.savez(dst, M=M, Fbar16=Fbar16, F_xc=fxc, M_logical=Mlog,
             M_prod_impl=M_prod_impl, M_prod_idl=M_prod_idl,
             Fbar_prod_impl=Fp, Fbar_prod_idl=Fi,
             T_total=T_tot, n_legs=n_legs, n_layers=n_lay, n_zframes=n_z,
             dt_div=DT_DIV, dt=DT, legs16=os.path.basename(LEGS16),
             n_applications=n_app,
             pulse_set=SET, taus=np.array([TAU[k] for k in
                                           ("w0w1", "w1w2", "w2w3", "w0w3")]),
             configs=np.array(_B.CONFIGS))
    print(f"  -> {dst}", flush=True)


def _score_matrix(M):
    """Score a 16x16 built by matrix product with the SAME code path as score()."""
    U = _B.target_sqiswap_03()
    fxc = _B.fxc_4q(U, M)
    d = 16
    Fbar = (np.vdot(M, M).real
            + fxc * np.vdot(U, U).real * np.vdot(M, M).real) / (d * (d + 1))
    sub = [i for i in range(16) if ((i >> 2) & 1) == 0 and ((i >> 1) & 1) == 0]
    return Fbar, fxc, M[np.ix_(sub, sub)]


def main():
    print(f"=== run_arm_decomp  ARM={ARM}  dt={DT:g} ({STEPS} steps/SNAP_DT) "
          f"c=diagonalize b=diagonalize regauge=per-chunk  dev={DEV} ===",
          flush=True)
    print(f"  PULSE_SET={SET}", flush=True)
    print(f"  legs: " + "  ".join(f"{k}{TAU[k]}->T={T_LEG[k]}" for k in TAU),
          flush=True)
    C16, bs0 = load_anchors16()

    if ARM == "legs16":
        stage2a(C16, bs0)
        return
    if ARM == "direct":
        solver = make_solver_single("w0w3")
        chunk = make_chunk(solver)
        y0 = pack_Cb_to_y(jnp.stack(C16),
                          [jnp.asarray(b, dtype=jnp.complex128) for b in bs0])
        mid = os.path.join(CKPT, f"direct_dt{DT_DIV:g}")
        import glob as _glob
        start, y_start = 0, y0
        cands = _glob.glob(mid + ".mid*.npy")
        if cands:
            best = max(cands, key=lambda f: int(f.rsplit(".mid", 1)[1][:-4]))
            start = int(best.rsplit(".mid", 1)[1][:-4])
            y_start = jnp.asarray(np.load(best))
            print(f"  [resume] {os.path.basename(best)} -> starting at chunk "
                  f"{start} (t={start*SNAP_DT:.1f})", flush=True)
        t0 = time.perf_counter()
        y, d = propagate(solver, chunk, y_start, T_LEG["w0w3"],
                         f"direct dt/{DT_DIV:g}", midckpt=mid, start_chunk=start)
        M = M16_of(C16, bs0, solver, y)
        Fb, fx, Mlog = _score_matrix(M)
        print(f"\n=== RESULT: direct arm at dt/{DT_DIV:g} ===", flush=True)
        print(f"  T = {T_LEG['w0w3']} t.u. = {T_LEG['w0w3']*0.0159155:.2f} ns",
              flush=True)
        print(f"  drift = {d['drift']:.3e}   ortho = {d['ortho']:.3e}   "
              f"(acceptance: drift <= 1e-5, ortho <= 1e-12)", flush=True)
        print(f"  F_bar16 = {Fb:.6f}   F_xc = {fx:.6f}   "
              f"||M||^2 = {np.linalg.norm(M)**2:.4f} (ideal 16)", flush=True)
        print(f"  (0,3) corner F_bar4 = {_fbar4(SQI, Mlog):.8f}", flush=True)
        print(f"  corner survival per col: "
              f"{np.round((np.abs(Mlog)**2).sum(0), 4)}", flush=True)
        print(f"  wall {(time.perf_counter()-t0)/3600:.2f} h", flush=True)
        dst = os.path.join(OUT, f"direct_dt{DT_DIV:g}.npz")
        np.savez(dst, M=M, Fbar16=Fb, F_xc=fx, M_logical=Mlog,
                 corner_Fbar4=_fbar4(SQI, Mlog), drift=d["drift"],
                 ortho=d["ortho"], dt_div=DT_DIV, T=T_LEG["w0w3"],
                 configs=np.array(_B.CONFIGS))
        print(f"  -> {dst}", flush=True)
        return
    legs = LEGS16
    assert os.path.exists(legs), (
        f"{legs} missing -- run ARM=legs16 first; the arms' local "
        f"layers are designed against the measured per-leg Z frames")
    store = dict(np.load(legs))
    # the Z frames and the product model come from these legs: same dt as the arm
    if "dt_div" in store:
        assert abs(float(store.pop("dt_div")) - DT_DIV) < 1e-12, (
            f"{legs} was made at a different dt than this arm (dt_div {DT_DIV})")
        store.pop("dt")
    else:
        print(f"  !! {os.path.basename(legs)} carries no dt; it predates the "
              f"dt check and cannot be verified against dt_div={DT_DIV}",
              flush=True)
    print(f"  legs store: {legs}", flush=True)
    z = frames(store)
    if ARM == "serial":
        run_arm("serial", serial_schedule(z), C16, bs0, store)
    elif ARM == "parallel":
        run_arm("parallel", parallel_schedule(z["par01"], z["par23"], z["w1w2"]),
                C16, bs0, store)
    else:
        raise SystemExit(
            f"ARM must be legs16|serial|parallel|direct, got {ARM!r}")


if __name__ == "__main__":
    if DEV == "cuda":
        with jax.default_device(jax.devices("cuda")[0]):
            main()
    else:
        main()
