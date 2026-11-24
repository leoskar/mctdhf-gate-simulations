"""Compare two screen runs of the SAME leg at different timesteps.

This is the pass/fail instrument of the timestep check. It takes the
final states saved by diag_integ_screen.py, builds each run's 16x16 gate matrix
against the COMMON anchor basis, and reports the four numbers that matter, at
the gauge level each claim needs:

    strict   |<M_A,M_B>|^2 / (||M_A||^2 ||M_B||^2)   -- global phase only.
             "Do these two simulations agree?"  Use for any claim about the
             DYNAMICS.
    local-Z  fxc_4q(M_A, M_B)                        -- global phase + 4 local Z.
             "Do they agree on the quantity F_bar quotients out?"  Use for any
             claim about the FIDELITY, which is itself defined up to local Z.

The gauge quotiented in the check must match the gauge quotiented in the claim.

Both metrics normalise by ||M||, so they are BLIND to overall amplitude. That is
checked separately by d||M|| and by per-input survival (|M|^2).sum(0) -- the
latter is what moved 1.5e-02 at 4dt-without-regauge while the aggregate
agreement still read 0.99998.

Comparisons are only meaningful between runs of EQUAL propagation length; a
filename collision once compared a 5 t.u. state against a 163.7 t.u. one, so the
length equality is asserted, not assumed.

    python cmp_screen.py <A.npz> <B.npz>
"""
from __future__ import annotations
import os, sys
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(os.path.join(HERE, "..", "..", "lib"))   # one_electron/lib
sys.path.insert(0, LIB)

import qoct_gate_leg as _gl
from qoct_gate_leg import (G_DICT_64, NUM_ORB, _make_h_s_x64, mctdh_overlap,
                           M_REFS, avg_gate_fidelity_up_to_localZ)
from rte_step import NUM_DVR, NUM_WELLS
from sweep_map import SNAP_DT
from mctdhf import MCTDHF_n_species_DVR
import run_route_sa as _B
sys.path.insert(0, HERE)
from diag_integ_screen import LEG_SPECS

# Which 2x2-well corner each leg drives, and the 4x4 target it is aimed at.
# Bits are named q0..q3 = wells 0..3; the corner keeps the two SPECTATORS at 0.
CORNERS = {
    "w0w1":  ((2, 3), "swap"),      # spectators q2,q3
    "w1w2":  ((0, 3), "swap"),
    "w2w3":  ((0, 1), "swap"),
    "sqi12": ((0, 3), "sqiswap"),
    "w0w3":  ((1, 2), "sqiswap"),
}


def build_solver(leg):
    X_RES, TAU = LEG_SPECS[leg]
    _gl.x_star_2_ghz = X_RES
    h_list = [_make_h_s_x64(s, TAU[0], TAU[1]) for s in range(NUM_WELLS)]
    return MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=16, imag_time=False)


def gate_matrix_16(C16, bs0, solver, y):
    """M[i,j] = <basis_i | psi_j(T)>, permuted to plain bit-order q0<<3|...|q3.

    diag_integ_screen propagates in CONFIGS order (iter_bitstrings); the corner
    extraction below assumes bit-order, so permute here exactly as
    run_route_sa.score does.
    """
    C_T, bs_T = solver._y_to_Cb(jnp.asarray(y))
    M = np.array([[complex(mctdh_overlap(C16[i], bs0, C_T[j], bs_T))
                   for j in range(16)] for i in range(16)])
    bit = np.array([(c[0] << 3) | (c[1] << 2) | (c[2] << 1) | c[3]
                    for c in _B.CONFIGS])
    P = np.zeros((16, 16))
    for p in range(16):
        P[bit[p], p] = 1.0
    return P @ M @ P.T


def corner(M, leg):
    (sa, sb), ref = CORNERS[leg]
    keep = [i for i in range(16)
            if ((i >> (3 - sa)) & 1) == 0 and ((i >> (3 - sb)) & 1) == 0]
    return M[np.ix_(keep, keep)], ref


def diagnostics(solver, y):
    nn = _B.norms(solver, np.asarray(y))
    _, bs = solver._y_to_Cb(jnp.asarray(y))
    oe = max(float(np.abs(np.asarray(b).conj().T @ np.asarray(b)
                          - np.eye(b.shape[1])).max()) for b in bs)
    return float(np.abs(nn - 1).max()), oe


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    da, db = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    leg = str(da["leg"])
    assert str(db["leg"]) == leg, f"different legs: {leg} vs {db['leg']}"
    # Different lengths silently produce meaningless agreement numbers.
    assert int(da["n_chunks"]) == int(db["n_chunks"]), (
        f"LENGTH MISMATCH {int(da['n_chunks'])} vs {int(db['n_chunks'])} chunks "
        f"-- these runs are not comparable")
    T = int(da["n_chunks"]) * SNAP_DT

    solver = build_solver(leg)
    C16, bs0 = _B.basis_states()
    MA = gate_matrix_16(C16, bs0, solver, da["y"])
    MB = gate_matrix_16(C16, bs0, solver, db["y"])
    drA, oA = diagnostics(solver, da["y"])
    drB, oB = diagnostics(solver, db["y"])

    strict = float(abs(np.vdot(MA, MB)) ** 2
                   / (np.vdot(MA, MA).real * np.vdot(MB, MB).real))
    localZ = float(_B.fxc_4q(MA, MB))
    dnorm = abs(np.linalg.norm(MA) - np.linalg.norm(MB))
    sA = (np.abs(MA) ** 2).sum(0); sB = (np.abs(MB) ** 2).sum(0)
    dsurv = float(np.abs(sA - sB).max())

    McA, ref = corner(MA, leg)
    McB, _ = corner(MB, leg)
    FA = float(avg_gate_fidelity_up_to_localZ(M_REFS[ref], McA))
    FB = float(avg_gate_fidelity_up_to_localZ(M_REFS[ref], McB))

    print(f"=== leg {leg}   T = {T:.1f} t.u.   corner target = {ref} ===")
    print(f"  A  dt/{float(da['div']):g}  dev={str(da['dev'])}  {fa}")
    print(f"  B  dt/{float(db['div']):g}  dev={str(db['dev'])}  {fb}")
    print(f"\n  agreement (amplitude divided out)")
    print(f"    strict  (global phase only)     {strict:.10f}   1-{1-strict:.2e}")
    print(f"    local-Z (F_bar's own gauge)     {localZ:.10f}   1-{1-localZ:.2e}")
    print(f"\n  amplitude (NOT covered above)")
    print(f"    d||M||                          {dnorm:.3e}")
    print(f"    max d(survival per input)       {dsurv:.3e}")
    print(f"\n  corner F_bar4 vs {ref}")
    print(f"    A {FA:.8f}    B {FB:.8f}    dF {FB-FA:+.2e}")
    print(f"\n  state health          drift        ortho")
    print(f"    A                   {drA:.2e}     {oA:.2e}")
    print(f"    B                   {drB:.2e}     {oB:.2e}")

    # Reported PER COMPONENT rather than as one
    # verdict. The two groups answer different questions and can disagree:
    # CONVERGENCE asks whether refining the step still moves the gate, and is the
    # thing that decides a timestep. HEALTH asks whether the state is intact, and
    # its 1e-5 drift threshold was calibrated on w2w3 — the long-range leg runs an
    # order of magnitude higher at every step size, so a single PASS/FAIL there
    # would read as a convergence failure when it is not one. Report both; do not
    # restate the threshold to fit the leg.
    conv = [("strict >= 1-1e-5", strict >= 1 - 1e-5, f"{1-strict:.2e} away"),
            ("|dF_bar4| <= 1e-5", abs(FB - FA) <= 1e-5, f"{abs(FB-FA):.2e}")]
    health = [("drift <= 1e-5", max(drA, drB) <= 1e-5, f"{max(drA, drB):.2e}"),
              ("ortho <= 1e-12", max(oA, oB) <= 1e-12, f"{max(oA, oB):.2e}")]
    for name, group in (("CONVERGENCE (decides the timestep)", conv),
                        ("HEALTH (w2w3-calibrated thresholds)", health)):
        verdict = "PASS" if all(ok for _, ok, _ in group) else "FAIL"
        print(f"\n  {name}: {verdict}")
        for label, ok, val in group:
            print(f"    {'ok ' if ok else 'NO '} {label:20s} {val}")


if __name__ == "__main__":
    main()
