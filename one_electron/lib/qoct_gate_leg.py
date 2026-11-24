"""Two-qubit gate leg: four-state gate fidelity and single-leg propagation.

A leg moves the active pair from the idle point to its target point and back
with erf-edged ramps around a hold.  Leg types: "sqiswap" (the 50/50
exchange point) and "swap" (about twice the hold), scored against different
ideal gates M_REFS[leg].

Fidelity. M_ij = <anchor_i | psi_j(T)> over the four active-pair anchor states
{|00>,|10>,|01>,|11>} (spectators in |0>).  The anchors are the idle-point
eigenstates themselves, identified by their dominant determinant, not by
energy order.  The loss is 1 - Fbar_4, the average gate fidelity up to local
Z (avg_gate_fidelity_up_to_localZ), which counts leakage out of the
computational subspace; the maximisation over local Z is that of
gate_fidelity.py.

Propagation. Each of the four states is propagated on its own with norb = 4
orbitals per well (a state-averaged propagation fails here: with all
spectators in |0> the averaged spectator density is rank-deficient).  The
four states run as one vmapped batch, in double precision (complex128
states, float64 integrals rebuilt from the DVR bases), with the VMF
integrator; the scan is compiled once, and the compile time is reported
separately.

Default leg: w0w1 sqiswap at tau_ramp = 3/Delta_2 and tau_hold = 12.75.

Usage:
    python lib/qoct_gate_leg.py verify  # loss checks only (from one_electron/)
    python lib/qoct_gate_leg.py leg     # + timed batched leg
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import jax

# VMF is latency-bound, so it runs on the CPU; set before the backend starts.
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))          # one_electron/lib/
sys.path.insert(0, HERE)

from rte_step import (        # geometry/schedule pieces + production settings
    DVR_BASES,
    _coulomb,
    _envelope_param,
    LENGTH_UNIT_UM,
    DELTA_2,
    NUM_WELLS,
    NUM_DVR,
    NUM_ORBITALS,
    TYPICAL_FREQ_GHZ,
    x_star_1_ghz,
    x_star_2_ghz,
    TAU_RAMP_DEFAULT,
    DT,
    NUM_NEWTON,
    MIN_SOLVER_TOL,
    MAX_SOLVER_TOL,
    MAX_SOLVER_ITER,
    C_anchor,
    bs_anchor,
)
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_vmf_integrator
from gate_fidelity import _localZ, gate_fidelity_up_to_localZ

assert NUM_WELLS == 4  # einsum strings below are 4-species

NUM_ORB = NUM_ORBITALS       # per-state norb=4


# ── x64 integrals (rebuilt from the DVR bases; rte_step's are float32) ────────

F64 = jnp.float64
from geometry_config import WELL_CENTERS_UM as _WELL_CENTERS_UM
WELL_CENTERS_64 = jnp.array(_WELL_CENTERS_UM / LENGTH_UNIT_UM, dtype=F64)
_zero_pot = lambda x: jnp.zeros_like(x)                       # noqa: E731
T_S_64 = [
    jnp.asarray(DVR_BASES[s].get_h(potential=_zero_pot), dtype=F64)
    for s in range(NUM_WELLS)
]
X_S_64 = [jnp.asarray(DVR_BASES[s].get_grid(), dtype=F64) for s in range(NUM_WELLS)]
G_DICT_64 = {
    (s1, s2): jnp.asarray(
        DVR_BASES[s1].get_g_cross_species(DVR_BASES[s2], _coulomb), dtype=F64
    )
    for s1 in range(NUM_WELLS)
    for s2 in range(s1 + 1, NUM_WELLS)
}


def _make_h_s_x64(s, tau_ramp, tau_hold):
    """Erf-edged ramp/hold h_s(t) at float64 (same schedule as rte_step's
    _make_h_s_param, WITHOUT the H_DTYPE=float32 downcast)."""
    centre = WELL_CENTERS_64[s]
    x_s = X_S_64[s]
    T_s = T_S_64[s]

    def h_s_t(t):
        x_ghz = x_star_1_ghz + (x_star_2_ghz - x_star_1_ghz) * _envelope_param(
            t, tau_ramp, tau_hold
        )
        a = x_ghz[s] / TYPICAL_FREQ_GHZ
        b = x_ghz[NUM_WELLS + s] / TYPICAL_FREQ_GHZ
        dx = x_s - centre
        v = 0.5 * a**2 * dx**2 + (b * a**2 / 3.0) * dx**4
        return T_s + jnp.diag(v)

    return h_s_t


# ── Active pair, loss basis, switchable M_ref ─────────────────────────────────
#
# Gate-1 leg: adjacent pair (0,1); spectators (2,3) stay in |0>. Row/column
# order (|00>,|10>,|01>,|11>) on the pair matches the _localZ convention inside
# gate_fidelity_up_to_localZ.

ACTIVE_PAIR = (0, 1)
PAIR_CONFIGS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)]
PAIR_LABELS = ["|00>", "|10>", "|01>", "|11>"]

_S2 = 1.0 / np.sqrt(2.0)
M_REFS = {
    # entangling leg: sqrt(iSWAP) on the pair
    "sqiswap": np.array(
        [
            [1, 0, 0, 0],
            [0, _S2, 1j * _S2, 0],
            [0, 1j * _S2, _S2, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    ),
    # routing leg: full SWAP permutation (|10> <-> |01>)
    "swap": np.eye(4)[:, [0, 2, 1, 3]].astype(complex),
}


def avg_gate_fidelity_up_to_localZ(M_ref, M):
    """Average gate fidelity up to local Z, including leakage:

        Fbar_d = [ Tr(M M†) + max_{Z_L,Z_R} |Tr(M_ref† Z_L M Z_R)|² ] / (d(d+1))

    Tr(M M†) = ‖M‖_F² is gauge-INVARIANT and carries the subspace-leakage
    penalty (population that leaks out of the 4-D computational subspace shrinks
    ‖M‖_F²); the coherent term REUSES the per-qubit-Z maximiser inside
    `gate_fidelity_up_to_localZ` (F_xc · ‖M_ref‖² · ‖M‖² recovers the
    un-normalised max|Tr|²). This is the metric F_xc is NOT: F_xc divides the
    coherent term by ‖M‖² and is therefore blind to leakage. Returns Fbar in
    [0, 1]."""
    M = np.asarray(M)
    M_ref = np.asarray(M_ref)
    d = M_ref.shape[0]
    nref = float(np.linalg.norm(M_ref) ** 2)            # = d for a unitary ref
    nm = float(np.linalg.norm(M) ** 2)                  # = Tr(M M†)
    fxc = gate_fidelity_up_to_localZ(M_ref, M)          # = max|Tr|²/(nref·nm)
    coherent = fxc * nref * nm                          # un-normalised max|Tr|²
    return (nm + coherent) / (d * (d + 1.0))


def leg_loss(M, leg):
    """1 - Fbar_4 against the leg's ideal gate (rows of M are the bare anchors)."""
    return 1.0 - avg_gate_fidelity_up_to_localZ(M_REFS[leg], M)


# ── State identification (dominant determinant, NEVER energy ordering) ────────


def identify_anchor_states(C):
    """Map each PAIR_CONFIG to its anchor state index by dominant determinant.

    For each config: pick the state carrying the most weight on that
    determinant, then assert that the state's own dominant determinant IS that
    config (mutual consistency) and that it genuinely dominates (>0.5).
    """
    C = np.asarray(C)
    w = np.abs(C) ** 2
    norms = w.reshape(C.shape[0], -1).sum(axis=1)
    picks = {}
    for cfg, lab in zip(PAIR_CONFIGS, PAIR_LABELS):
        w_cfg = w[(slice(None),) + cfg] / norms          # weight of det cfg per state
        k = int(np.argmax(w_cfg))
        dom = np.unravel_index(np.argmax(w[k]), w[k].shape)
        assert tuple(int(d) for d in dom) == cfg, (
            f"anchor identity check failed for {lab}: best-weight state {k} has "
            f"dominant determinant {dom}, expected {cfg}"
        )
        assert w_cfg[k] > 0.5, (
            f"anchor {lab}: dominant-determinant weight {w_cfg[k]:.3f} <= 0.5 "
            f"in state {k} — not a clean computational-basis anchor"
        )
        picks[cfg] = (k, float(w_cfg[k]))
    ks = [picks[cfg][0] for cfg in PAIR_CONFIGS]
    assert len(set(ks)) == 4, f"anchor states not distinct: {ks}"
    return picks


# ── Overlaps and gate matrix ──────────────────────────────────────────────────


def mctdh_overlap(C_a, bs_a, C_b, bs_b):
    """<A|B> between two 4-species MCTDH states with different orbital sets."""
    S = [jnp.einsum("xm,xn->mn", jnp.conj(bs_a[s]), bs_b[s]) for s in range(4)]
    return jnp.einsum(
        "abcd,aA,bB,cC,dD,ABCD->", jnp.conj(C_a), S[0], S[1], S[2], S[3], C_b
    )


def gate_matrix(C4, bs0, finals):
    """M_ij = <anchor_i | psi_j(T)>; finals = [(C_j, bs_j)] per propagated
    state (per-state runs: each state ends with its OWN orbital set)."""
    M = np.zeros((4, 4), dtype=complex)
    for j, (C_j, bs_j) in enumerate(finals):
        for i in range(4):
            M[i, j] = complex(mctdh_overlap(C4[i], bs0, C_j, bs_j))
    return M


def load_loss_anchors():
    """4 normalized anchor CI tensors (loss order) + shared 4-orbital set
    (canonical chain_step1.npz anchors via rte_step, cast to complex128)."""
    picks = identify_anchor_states(C_anchor)
    bs0 = [jnp.asarray(b, dtype=jnp.complex128) for b in bs_anchor]
    C4 = []
    for cfg, lab in zip(PAIR_CONFIGS, PAIR_LABELS):
        k, wdom = picks[cfg]
        Ck = jnp.asarray(C_anchor[k], dtype=jnp.complex128)
        n2 = float(jnp.real(mctdh_overlap(Ck, bs0, Ck, bs0)))
        print(f"  {lab}: anchor state {k:2d}  dominant-det weight {wdom:.4f}  "
              f"norm^2 {n2:.6f}")
        C4.append(Ck / jnp.sqrt(n2))
    return C4, bs0


# ── Loss verification ───────────────────────────────────────────────────────


def _rand_gauge(M, rng):
    """Random per-qubit Z's (both sides) + global phase applied to M."""
    aL, bL, aR, bR, g = rng.uniform(0, 2 * np.pi, 5)
    return np.exp(1j * g) * (_localZ(aL, bL) @ M @ _localZ(aR, bR))


def verify_loss():
    print("=== loss verification ===")
    rng = np.random.default_rng(7)

    # No-op check: M = anchors against themselves -> F_xc ~ 1 against the
    # IDENTITY target. Tolerance calibrated to the measured anchor
    # orthonormality residual (complex64-stored ITE anchors), not invented.
    C4, bs0 = load_loss_anchors()
    M_noop = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        for i in range(4):
            M_noop[i, j] = complex(mctdh_overlap(C4[i], bs0, C4[j], bs0))
    err = float(np.linalg.norm(M_noop - np.eye(4)))
    F_noop = gate_fidelity_up_to_localZ(np.eye(4), M_noop)
    tol_noop = max(1e-9, err**2)
    print(f"[no-op]  ||M-I||_F = {err:.3e}   1-F_xc = {1-F_noop:.3e}   "
          f"(tol {tol_noop:.1e})")
    assert 1 - F_noop <= tol_noop, "no-op check failed"

    # Gauge invariance: F_xc must be blind to per-qubit Z's + global phase,
    # for BOTH production references (catches a wrong gauge group).
    M_rand = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    worst = 0.0
    for leg in ("sqiswap", "swap"):
        for M in (M_noop, M_rand):
            F0 = gate_fidelity_up_to_localZ(M_REFS[leg], M)
            for _ in range(3):
                dF = abs(
                    gate_fidelity_up_to_localZ(M_REFS[leg], _rand_gauge(M, rng))
                    - F0
                )
                worst = max(worst, dF)
    print(f"[gauge]  max |dF_xc| over 12 random Z(x)Z+global gauges = {worst:.2e}")
    assert worst <= 1e-10, f"gauge invariance violated: {worst:.2e}"

    # CZ sensitivity: e^{i*phi} on the |11> column of an otherwise perfect M
    # must drop F_xc — the error a 2-state loss cannot see. Analytic max over
    # the Z gauge is cos^2(phi/4) for the identity and (by P^dag Z P staying in
    # the gauge family) the SWAP reference; for sqiswap assert the drop.
    for phi in (0.2, np.pi / 2):
        D_phi = np.diag([1.0, 1.0, 1.0, np.exp(1j * phi)])
        F_an = float(np.cos(phi / 4) ** 2)
        F_id = gate_fidelity_up_to_localZ(np.eye(4), D_phi)
        F_sw = gate_fidelity_up_to_localZ(M_REFS["swap"], M_REFS["swap"] @ D_phi)
        F_sq = gate_fidelity_up_to_localZ(
            M_REFS["sqiswap"], M_REFS["sqiswap"] @ D_phi
        )
        print(f"[CZ]     phi={phi:.3f}: F_xc id/swap/sqiswap = "
              f"{F_id:.8f}/{F_sw:.8f}/{F_sq:.8f}  (analytic id,swap {F_an:.8f})")
        assert abs(F_id - F_an) <= 1e-7 and abs(F_sw - F_an) <= 1e-7, (
            "F_xc deviates from analytic CZ value"
        )
        assert F_sq < 1 - 1e-3, "CZ phase did not drop F_xc (sqiswap ref)"

    # M_ref switch check: the ideal sqiswap M must score 1
    # against the sqiswap reference and (analytically) 6/16 = 0.375 against
    # the SWAP reference — and vice versa. Catches a silently wrong/ignored
    # target argument.
    F_cross_an = 6.0 / 16.0
    for leg, other in (("sqiswap", "swap"), ("swap", "sqiswap")):
        F_same = gate_fidelity_up_to_localZ(M_REFS[leg], M_REFS[leg])
        F_cross = gate_fidelity_up_to_localZ(M_REFS[other], M_REFS[leg])
        print(f"[switch] M=ideal {leg}: F_xc(ref={leg}) = {F_same:.8f}   "
              f"F_xc(ref={other}) = {F_cross:.8f}  (analytic {F_cross_an:.4f})")
        assert F_same >= 1 - 1e-7, f"M_ref switch: ideal {leg} not scored 1"
        assert abs(F_cross - F_cross_an) <= 1e-6, "M_ref cross-score not 0.375"

    # Reported baselines: "no gate happened" (M = I) against each reference.
    F_base_sq = gate_fidelity_up_to_localZ(M_REFS["sqiswap"], np.eye(4))
    F_base_sw = gate_fidelity_up_to_localZ(M_REFS["swap"], np.eye(4))
    base_sq_an = (6.0 + 4.0 * np.sqrt(2.0)) / 16.0
    print(f"[base]   F_xc(no-gate | sqiswap ref) = {F_base_sq:.6f} "
          f"(analytic {base_sq_an:.6f});  F_xc(no-gate | swap ref) = "
          f"{F_base_sw:.5f} (analytic 0.25)")

    # ── Fbar_4 (the loss used) ───────────────────────────────────────────────
    # Leakage sensitivity — the check 1 - F_xc silently fails. Scale the |11>
    # column of an otherwise-ideal sqiswap by sqrt(1-L): population L leaks OUT
    # of the 4-D computational subspace, so Tr(M M†) = 4 - L drops and
    #   Fbar(L) = [(4 - L) + (3 + sqrt(1-L))^2] / 20
    # drops with it; F_xc divides the coherent term by ‖M‖² and barely moves.
    print("[Fbar leak] L : Fbar(meas)=Fbar(an)    1-Fbar   1-F_xc (leakage-blind)")
    for L in (0.0, 0.25, 0.64, 0.90):
        Msc = M_REFS["sqiswap"].copy()
        Msc[:, 3] *= np.sqrt(1.0 - L)
        Fbar = avg_gate_fidelity_up_to_localZ(M_REFS["sqiswap"], Msc)
        Fbar_an = ((4.0 - L) + (3.0 + np.sqrt(1.0 - L)) ** 2) / 20.0
        Fxc = gate_fidelity_up_to_localZ(M_REFS["sqiswap"], Msc)
        print(f"            {L:.2f}: {Fbar:.6f}={Fbar_an:.6f}    "
              f"{1-Fbar:.4f}   {1-Fxc:.4f}")
        assert abs(Fbar - Fbar_an) <= 1e-9, f"Fbar leakage analytic @L={L}"
    # At 64% leakage the leakage-blind metric must under-report grossly.
    Msc = M_REFS["sqiswap"].copy()
    Msc[:, 3] *= np.sqrt(1.0 - 0.64)
    loss_fxc = 1 - gate_fidelity_up_to_localZ(M_REFS["sqiswap"], Msc)
    loss_fbar = 1 - avg_gate_fidelity_up_to_localZ(M_REFS["sqiswap"], Msc)
    assert loss_fxc < 0.25 * loss_fbar, (
        f"F_xc should grossly under-report 64% leakage: 1-F_xc={loss_fxc:.4f} "
        f"vs 1-Fbar={loss_fbar:.4f}"
    )

    # Fbar analytic baselines (replace the F_xc-era constants).
    s2 = np.sqrt(2.0)
    Fbar_checks = [
        ("ideal sqiswap | sqiswap ref", M_REFS["sqiswap"], "sqiswap", 1.0),
        ("ideal swap    | swap ref",    M_REFS["swap"],    "swap",    1.0),
        ("ideal sqiswap | swap ref",    M_REFS["sqiswap"], "swap",    0.5),
        ("ideal swap    | sqiswap ref", M_REFS["swap"],    "sqiswap", 0.5),
        ("no-gate (I)   | sqiswap ref", np.eye(4, dtype=complex),
         "sqiswap", (10.0 + 4.0 * s2) / 20.0),
        ("no-gate (I)   | swap ref",    np.eye(4, dtype=complex), "swap", 0.4),
    ]
    for name, M, leg, an in Fbar_checks:
        Fb = avg_gate_fidelity_up_to_localZ(M_REFS[leg], M)
        print(f"[Fbar base] {name}: Fbar = {Fb:.6f}  (analytic {an:.6f})")
        assert abs(Fb - an) <= 1e-6, f"Fbar baseline {name}: {Fb} != {an}"

    print("=== loss verification PASSED ===\n")
    return C4, bs0


# ── Task 2: timed per-state single-leg forward propagation (vmapped batch) ────

# Starting guess: the 50/50 point IS the sqiswap -> tau_hold ~ 12.75;
# tau_ramp at its flat QOCT value 3/Delta_2. A guess to be timed, not a result.
LEG_TYPE = "sqiswap"
TAU_RAMP_GUESS = TAU_RAMP_DEFAULT          # = 3/DELTA_2 ~= 29.24
TAU_HOLD_GUESS = 12.75                     # gradient-free 50/50 point
T_TOTAL = 2.0 * TAU_RAMP_GUESS + TAU_HOLD_GUESS

# Solve settings. dt = 1e-2 sits at the Newton stability edge of this system
# (dt = 2e-2 fails in the first ramp; at 1e-2 the correlated states lose norm,
# independent of the solver tolerances). At dt = 5e-3 the norm drift is
# below 1e-7, and the loose solver caps (1e-5, 20 iterations) give the same
# result as tight ones.
DT_PROD = 5e-3
VMF_MIN_TOL = MIN_SOLVER_TOL        # 1e-5
VMF_MAX_ITER = MAX_SOLVER_ITER      # 20

NUM_CHUNKS = 100                                            # saved snapshots
SAVE_EVERY = int(np.ceil(T_TOTAL / DT_PROD / NUM_CHUNKS))   # steps/snapshot
NUM_STEPS = NUM_CHUNKS * SAVE_EVERY

# drift tolerances (the VMF residual scale)
TOL_E_DRIFT = 1e-4      # |E(T)-E(0)|/|E(0)|
TOL_NORM_DRIFT = 1e-6   # | ||psi(T)||^2 - ||psi(0)||^2 |


def _natural_occupations(solver, C):
    """Natural-orbital occupations per species (ascending): state-averaged
    (the density that conditions D^-1; for num_states=1 this IS the per-state
    density) and per-state. Secondary diagnostic — guarded by the caller,
    must never abort the run."""
    w = np.asarray(solver.state_weights, dtype=float)
    occ_sa, occ_ps = [], []
    for D in solver.dCI.get_1p_RDMs(C):                 # (num_states, n, n)
        D = np.asarray(D)
        rho_sa = np.einsum("s,sij->ij", w, D)
        rho_sa = 0.5 * (rho_sa + rho_sa.conj().T)
        occ_sa.append(np.sort(np.linalg.eigvalsh(rho_sa).real))
        occ_ps.append(
            np.stack([
                np.sort(np.linalg.eigvalsh(0.5 * (d + d.conj().T)).real)
                for d in D
            ])
        )
    return np.stack(occ_sa), np.stack(occ_ps)   # (K,norb), (K,num_states,norb)


def run_leg(C4, bs0):
    print(f"=== timed Gate-1 {LEG_TYPE} leg (per-state x64, vmapped batch of 4, "
          f"pair 0-1) ===")
    print(f"tau_ramp={TAU_RAMP_GUESS:.4f}  tau_hold={TAU_HOLD_GUESS:.4f}  "
          f"T_total={T_TOTAL:.4f}  (DELTA_2={DELTA_2:.6f})")
    print(f"VMF: dt={DT_PROD}, num_steps={NUM_STEPS}, num_newton={NUM_NEWTON}, "
          f"solver_tol=[{VMF_MIN_TOL},{MAX_SOLVER_TOL}], "
          f"max_solver_iter={VMF_MAX_ITER}, norb={NUM_ORB}, num_states=1 "
          f"(x4 vmapped), x64, no e_shift, device={jax.devices()[0].platform}",
          flush=True)

    h_list = [
        _make_h_s_x64(s, TAU_RAMP_GUESS, TAU_HOLD_GUESS) for s in range(NUM_WELLS)
    ]
    assert h_list[0](0.0).dtype == jnp.float64          # x64 integrals
    solver = MCTDHF_n_species_DVR(
        h_list=h_list,
        g_dict=G_DICT_64,
        num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=1,
        imag_time=False,
    )
    # Batch of 4 independent per-state packed vectors (leading axis = state).
    y0s = jnp.stack([pack_Cb_to_y(C4[j][None], bs0) for j in range(4)])
    assert y0s.dtype == jnp.complex128                  # x64 state

    integrator = get_vmf_integrator(
        solver,
        DT_PROD,
        num_newton=NUM_NEWTON,
        min_solver_tol=VMF_MIN_TOL,
        max_solver_tol=MAX_SOLVER_TOL,
        max_solver_iter=VMF_MAX_ITER,
        precondition=True,
    )

    def _step(carry, _):
        t, y = carry
        return integrator(t, y), None

    def _chunk(carry, _):
        carry, _ = jax.lax.scan(_step, carry, None, length=SAVE_EVERY)
        return carry, carry

    def _run(y0):
        t0 = jnp.asarray(0.0, dtype=jnp.float64)
        (_, _), (ts, ys) = jax.lax.scan(_chunk, (t0, y0), None, length=NUM_CHUNKS)
        return ts, ys

    # AOT compile of the vmapped batch (clean JIT/steady-state split; the
    # BOBYQA driver reuses this pattern).
    t0 = time.perf_counter()
    compiled = jax.jit(jax.vmap(_run)).lower(y0s).compile()
    t_compile = time.perf_counter() - t0
    print(f"JIT compile: {t_compile:.1f}s", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"qoct_gate_leg_{stamp}.npz")
    out = dict(
        leg_type=LEG_TYPE, m_ref=M_REFS[LEG_TYPE],
        tau_ramp=TAU_RAMP_GUESS, tau_hold=TAU_HOLD_GUESS, t_total=T_TOTAL,
        dt=DT_PROD, num_steps=NUM_STEPS, save_every=SAVE_EVERY,
        num_orb=NUM_ORB, t_compile=t_compile,
        pair_configs=np.array(PAIR_CONFIGS),
        min_solver_tol=VMF_MIN_TOL, max_solver_tol=MAX_SOLVER_TOL,
        num_newton=NUM_NEWTON, max_solver_iter=VMF_MAX_ITER,
    )

    # ── Propagate the 4 per-state legs as one vmapped batch ──────────────────
    t0 = time.perf_counter()
    ts, ys = compiled(y0s)
    jax.block_until_ready(ys)
    wall = time.perf_counter() - t0
    ts_np = np.concatenate([[0.0], np.asarray(ts[0], dtype=float)])
    ys_np = np.concatenate(
        [np.asarray(y0s)[:, None], np.asarray(ys)], axis=1
    )                                   # (4, NUM_CHUNKS+1, len_y)
    finite_ps = [bool(np.all(np.isfinite(ys_np[j, -1]))) for j in range(4)]
    finite = all(finite_ps)
    print(f"  batched per-state legs: {wall:.1f}s  "
          f"({wall/NUM_STEPS*1e3:.1f} ms/step for the batch)  "
          f"final-state finite={finite_ps}", flush=True)
    out.update(wall=wall, ts=ts_np, ys=ys_np)
    if not finite:
        np.savez(out_path, **out)
        sys.exit(f"ABORT: non-finite per-state leg(s) {finite_ps} — D^-1 "
                 f"conditioning suspect; partial results in {out_path}")
    np.savez(out_path, **out)          # primary results saved before diagnostics

    # ── Gate matrix, loss, transfer ───────────────────────────────────────────
    finals = []
    for j in range(4):
        C_j, bs_j = solver._y_to_Cb(jnp.asarray(ys_np[j, -1]))
        finals.append((C_j[0], bs_j))   # squeeze the num_states=1 axis
    M = gate_matrix(C4, bs0, finals)
    loss = leg_loss(M, LEG_TYPE)                  # = 1 - Fbar_4 (pinned loss)
    Fbar = 1.0 - loss                             # leakage-penalised (headline)
    F_xc = gate_fidelity_up_to_localZ(M_REFS[LEG_TYPE], M)   # diagnostic only
    F_xc_swap = gate_fidelity_up_to_localZ(M_REFS["swap"], M)   # context only

    np.set_printoptions(precision=4, suppress=True)
    print(f"\n|M|^2 (rows: bare anchors; cols: propagated states):"
          f"\n{np.abs(M)**2}")
    # Transfer sanity (~0.5 expected at the 50/50 point; guess — report only).
    p_01_to_10 = abs(M[1, 2]) ** 2
    p_10_to_01 = abs(M[2, 1]) ** 2
    subspace = (np.abs(M) ** 2).sum(axis=0)
    print(f"transfer: |<10|psi_01(T)>|^2 = {p_01_to_10:.4f}   "
          f"|<01|psi_10(T)>|^2 = {p_10_to_01:.4f}")
    print(f"4-anchor-subspace population per state: {subspace}")
    print(f"  |11> subspace population (leakage indicator) = {subspace[3]:.4f}")
    print(f"\nFbar_4({LEG_TYPE}) = {Fbar:.6f}   loss = 1-Fbar = {loss:.6f}   "
          f"[diagnostic: F_xc(leakage-blind) = {F_xc:.6f}, "
          f"F_xc(swap) = {F_xc_swap:.6f}]")

    # ── Diagnostics (guarded — never abort after the physics is saved) ────────
    n_snap = len(ts_np)
    energies = np.full((n_snap, 4), np.nan)
    norms2 = np.full((n_snap, 4), np.nan)
    occ = np.full((n_snap, 4, NUM_WELLS, NUM_ORB), np.nan)  # ascending per well
    try:
        for j in range(4):
            for m, t in enumerate(ts_np):
                yj = jnp.asarray(ys_np[j, m])
                C_t, bs_t = solver._y_to_Cb(yj)
                energies[m, j] = float(np.real(np.asarray(
                    solver.calculate_energy(float(t), yj))[0]))
                norms2[m, j] = float(np.real(
                    mctdh_overlap(C_t[0], bs_t, C_t[0], bs_t)))
                o_sa, _ = _natural_occupations(solver, C_t)   # num_states=1:
                occ[m, j] = o_sa                              # SA == per-state
    except Exception as exc:                                    # noqa: BLE001
        print(f"WARNING: diagnostics failed ({exc!r}) — primary results are "
              f"already saved", flush=True)

    out.update(energies=energies, norms2=norms2, occ=occ,
               M=M, Fbar=Fbar, F_xc=F_xc, loss=loss,
               F_xc_swap=F_xc_swap, p_01_to_10=p_01_to_10, p_10_to_01=p_10_to_01)
    np.savez(out_path, **out)

    print("\nper-state diagnostics (closure and drift):")
    # Raw E(T)-E(0) is NOT a drift measure for an entangling leg: perfect
    # exchange between non-degenerate anchors changes the energy physically
    # (|10>/|01> at 0.794 transfer give exactly 1.9e-3). Gate the
    # energy on the mixing-corrected residual E(T) - sum_i |M_ij|^2 E_i(0),
    # valid while the state stays in the anchor subspace; if it leaked, the
    # prediction is undefined -> report raw, gate n/a (norm is the real
    # conservation check).
    subspace_pop = (np.abs(M) ** 2).sum(axis=0)
    ok = True
    for j, lab in enumerate(PAIR_LABELS):
        e0, eT = energies[0, j], energies[-1, j]
        de_raw = abs(eT - e0) / abs(e0)
        e_pred = float((np.abs(M[:, j]) ** 2) @ energies[0, :])
        de_mix = abs(eT - e_pred / max(subspace_pop[j], 1e-12)) / abs(e0)
        dn_T = abs(norms2[-1, j] - norms2[0, j])
        dn_max = np.nanmax(np.abs(norms2[:, j] - norms2[0, j]))
        n_pass = dn_T <= TOL_NORM_DRIFT
        if subspace_pop[j] > 0.99:
            e_pass = de_mix <= TOL_E_DRIFT
            e_str = (f"|dE_mix|/|E0|={de_mix:.2e} (raw {de_raw:.2e}) "
                     f"[{'PASS' if e_pass else 'FAIL'}]")
        else:
            e_pass = True   # leaked: prediction undefined; norm gate decides
            e_str = (f"|dE|/|E0|={de_raw:.2e} raw [n/a: subspace "
                     f"{subspace_pop[j]:.3f}]")
        ok &= e_pass and n_pass
        print(f"  state {lab}: E0={e0:.6f}  {e_str}  "
              f"|dnorm^2|(T)={dn_T:.2e} (max {dn_max:.2e}) "
              f"[{'PASS' if n_pass else 'FAIL'}]  "
              f"min NO occ={np.nanmin(occ[:, j]):.2e}")
    occ_min_rank = np.nanmin(occ, axis=(0, 1, 2))       # per orbital rank
    print(f"natural-occupation minima per orbital rank (asc, over states+"
          f"species+trajectory): {occ_min_rank}")
    print(f"global min occupation over trajectory: {np.nanmin(occ):.2e}  "
          f"(Tikhonov eps = {solver.eps:.0e})")

    # ── Budget ────────────────────────────────────────────────────────────────
    budget_h = wall * 30 / 3600.0
    print(f"\nwall time: compile {t_compile:.1f}s; batched propagation "
          f"{wall:.1f}s")
    print(f"BOBYQA budget estimate per leg: 1 batched eval x {wall:.0f}s x "
          f"~30 evals = {budget_h:.1f} h")
    print(f"drift criteria: {'ALL PASS' if ok else 'FAILURES — see above'}")
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    assert mode in ("verify", "leg"), f"mode must be verify|leg, got {mode!r}"
    # leg [leg_type] [tau_ramp] [tau_hold]: archive ONE full-trajectory
    # propagation (ts, ys snapshots + energies + occ) at a chosen pulse. With
    # no extra args it runs the default sqiswap 50/50 guess; the BOBYQA driver
    # invokes it at each leg's optimum so the converged dynamics are saved
    # (the optimizer npz keeps only the endpoint gate matrix M, not ys).
    if mode == "leg" and len(sys.argv) > 2:
        LEG_TYPE = sys.argv[2]
        assert LEG_TYPE in M_REFS, f"leg_type must be in {list(M_REFS)}"
    if mode == "leg" and len(sys.argv) > 4:
        TAU_RAMP_GUESS = float(sys.argv[3])
        TAU_HOLD_GUESS = float(sys.argv[4])
        T_TOTAL = 2.0 * TAU_RAMP_GUESS + TAU_HOLD_GUESS
        SAVE_EVERY = int(np.ceil(T_TOTAL / DT_PROD / NUM_CHUNKS))
        NUM_STEPS = NUM_CHUNKS * SAVE_EVERY
    C4, bs0 = verify_loss()
    if mode == "leg":
        run_leg(C4, bs0)
