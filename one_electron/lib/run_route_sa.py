"""16-state register code: basis states, scoring and the serial routing.

Used by routed_gates/scripts/run_arm_decomp.py (the routed circuits of
Sec. IV A 2), gate_sweeps/scripts/confirm_k16.py and rescore_k16.py (the
16-state scores) and the time-step checks: basis_states() (the 16 anchor-1
eigenstates, Hungarian-matched), target_sqiswap_03() and score() (the 16x16
gate matrix and the 4-qubit average gate fidelity up to local Z).

Run directly, it propagates all 16 basis states through the serial routing
SWAP(2,3) . SWAP(1,2) . sqiSWAP(0,1) . SWAP(1,2) . SWAP(2,3) in one
state-averaged run (VMF on the CPU, x64), with the pulses in TAU below and
per-leg checkpoints.  That stand-alone run is an earlier test, not one of
the paper's results; the paper's circuits use the pulse sets in
routed_gates/slurm/_common.sh and are run by run_arm_decomp.py.
"""
from __future__ import annotations
import os, sys, time
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qoct_gate_leg as _gl
from qoct_gate_leg import (DT_PROD, G_DICT_64, NUM_ORB, VMF_MIN_TOL, VMF_MAX_ITER,
                           _make_h_s_x64, mctdh_overlap)
from sweep_map import SNAP_DT, SNAP_STEPS
from rte_step import (MAX_SOLVER_TOL, NUM_DVR, NUM_NEWTON, NUM_WELLS,
                      C_anchor, bs_anchor)
from geometry_config import ANCHOR_PATHS
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "ckpt"); os.makedirs(CKPT, exist_ok=True)

# ── schedule (taus = best-Fbar sweep points) ──────────────────────────────────
RES = {"w0w1": jnp.asarray(np.load(ANCHOR_PATHS[2])["best_x"], dtype=jnp.float64),
       "w1w2": jnp.asarray(np.load(ANCHOR_PATHS["3a"])["best_x"], dtype=jnp.float64),
       "w2w3": jnp.asarray(np.load(ANCHOR_PATHS["4a"])["best_x"], dtype=jnp.float64)}
# Pulses of the stand-alone run.  The leg checkpoints in results/ckpt/ are
# keyed by leg index only: clear them after changing TAU.
TAU = {"w0w1": (37.7, 112.5), "w1w2": (40.0, 873.0), "w2w3": (110.0, 845.0)}
LEGS = ["w2w3", "w1w2", "w0w1", "w1w2", "w2w3"]

# ── 16 basis states (anchor-1 eigenstates, stored Hungarian matched_idx) ───────
_A1 = np.load(ANCHOR_PATHS[1])
MATCHED = _A1["matched_idx"][int(np.argmin(_A1["losses"]))]
CONFIGS = [tuple(int(x) for x in c) for c in _A1["configs"]]


def basis_states():
    bs0 = [jnp.asarray(b, dtype=jnp.complex128) for b in bs_anchor]
    C16 = []
    for n in range(16):
        Cn = jnp.asarray(C_anchor[MATCHED[n]], dtype=jnp.complex128)
        nrm = float(jnp.real(mctdh_overlap(Cn, bs0, Cn, bs0)))
        C16.append(Cn / jnp.sqrt(nrm))
    return C16, bs0


def make_solver(pair, tr, th):
    _gl.x_star_2_ghz = RES[pair]
    h_list = [_make_h_s_x64(s, tr, th) for s in range(NUM_WELLS)]
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=16, imag_time=False)
    from mctdhf.integrators import get_vmf_integrator
    integ = get_vmf_integrator(solver, DT_PROD, num_newton=NUM_NEWTON,
                               min_solver_tol=VMF_MIN_TOL, max_solver_tol=MAX_SOLVER_TOL,
                               max_solver_iter=VMF_MAX_ITER, precondition=True)
    return solver, integ


def norms(solver, y):
    C, bs = solver._y_to_Cb(y)
    return np.array([float(jnp.real(mctdh_overlap(C[j], bs, C[j], bs))) for j in range(16)])


def run_leg(y0, pair, leg_label):
    tr, th = TAU[pair]
    T_leg = round(2 * tr + th, 6)
    n_chunks = int(round(T_leg / SNAP_DT))
    solver, integ = make_solver(pair, tr, th)

    def step(carry, _):
        t, y = carry
        return integ(t, y), None
    chunk = jax.jit(lambda c: jax.lax.scan(step, c, None, length=SNAP_STEPS)[0])

    print(f"=== leg {leg_label} ({pair}) tr={tr} th={th} T_leg={T_leg} "
          f"{n_chunks} chunks ===", flush=True)
    t0 = time.perf_counter()
    carry = chunk((jnp.asarray(0.0), y0)); jax.block_until_ready(carry)
    print(f"  compile+chunk1: {time.perf_counter()-t0:.1f}s", flush=True)
    tic = time.perf_counter()
    for c in range(1, n_chunks):
        carry = chunk(carry)
        if (c + 1) % 500 == 0:
            jax.block_until_ready(carry); t, y = carry
            nn = norms(solver, np.asarray(y))
            print(f"  chunk {c+1}/{n_chunks} t={float(t):.1f} "
                  f"norm[min,max]=[{nn.min():.5f},{nn.max():.5f}] "
                  f"finite={bool(np.all(np.isfinite(np.asarray(y))))} "
                  f"({time.perf_counter()-tic:.0f}s)", flush=True)
    jax.block_until_ready(carry)
    _, y_fin = carry
    nn = norms(solver, np.asarray(y_fin))
    print(f"  leg {leg_label} done {time.perf_counter()-tic:.0f}s "
          f"norm[min,max]=[{nn.min():.5f},{nn.max():.5f}] "
          f"finite={bool(np.all(np.isfinite(np.asarray(y_fin))))}", flush=True)
    return y_fin, solver


# ── 16x16 target + 4-qubit gauge-corrected fidelity ──────────────────────────
_S2 = 1.0 / np.sqrt(2.0)
_SQ2 = np.array([[1, 0, 0, 0], [0, _S2, 1j*_S2, 0], [0, 1j*_S2, _S2, 0], [0, 0, 0, 1]],
                dtype=complex)


def target_sqiswap_03():
    U = np.zeros((16, 16), complex)
    for i in range(16):
        q = [(i >> (3 - k)) & 1 for k in range(4)]
        for li in range(4):
            q0o, q3o = (li >> 1) & 1, li & 1
            amp = _SQ2[2*q[0] + q[3], 2*q0o + q3o]
            if amp != 0:
                o = (q0o << 3) | (q[1] << 2) | (q[2] << 1) | q3o
                U[o, i] = amp
    return U


def _zphases(thetas):
    bits = np.array([[(i >> (3 - k)) & 1 for k in range(4)] for i in range(16)])
    return np.exp(1j * (bits @ np.asarray(thetas)))


def fxc_4q(U_ref, M, restarts=40, seed=0):
    from scipy.optimize import minimize
    if not np.all(np.isfinite(M)):
        return np.nan
    nref = np.vdot(U_ref, U_ref).real; nm = np.vdot(M, M).real
    Uc = U_ref.conj().T

    def neg(x):
        zl = _zphases(x[:4]); zr = _zphases(x[4:8])
        t = np.trace(Uc @ ((zl[:, None] * M) * zr[None, :]))
        return -(abs(t) ** 2) / (nref * nm)
    rng = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        r = minimize(neg, rng.uniform(0, 2*np.pi, 8), method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-11, maxiter=8000))
        if best is None or r.fun < best.fun:
            best = r
    return float(-best.fun)


def score(C16, bs0, solver, y_fin):
    C_T, bs_T = solver._y_to_Cb(y_fin)
    M = np.zeros((16, 16), complex)
    for j in range(16):
        for i in range(16):
            M[i, j] = complex(mctdh_overlap(C16[i], bs0, C_T[j], bs_T))
    # M is indexed in CONFIGS (iter_bitstrings) order; the target/corner below
    # assume plain binary bit-order (q0<<3|q1<<2|q2<<1|q3). Permute M to bit-order.
    bit = np.array([(c[0] << 3) | (c[1] << 2) | (c[2] << 1) | c[3] for c in CONFIGS])
    P = np.zeros((16, 16))
    for p in range(16):
        P[bit[p], p] = 1.0
    M = P @ M @ P.T
    U = target_sqiswap_03()
    fxc = fxc_4q(U, M)
    d = 16
    Fbar = (np.vdot(M, M).real + fxc * np.vdot(U, U).real * np.vdot(M, M).real) / (d*(d+1))
    # (0,3) logical corner (q1=q2=0 in & out) -> 4x4 vs 2-qubit sqiSWAP
    sub = [i for i in range(16) if ((i >> 2) & 1) == 0 and ((i >> 1) & 1) == 0]  # q1=q2=0
    Mlog = M[np.ix_(sub, sub)]
    return M, Fbar, fxc, Mlog


def main():
    print("=== SA-16 routed sqiSWAP(0,3), routing B, x64 VMF/CPU ===", flush=True)
    print(f"settings: norb={NUM_ORB} dt={DT_PROD} tol=[{VMF_MIN_TOL},{MAX_SOLVER_TOL}] "
          f"newton={NUM_NEWTON} legs={LEGS}", flush=True)
    C16, bs0 = basis_states()
    C_init = jnp.stack(C16); bs_init = bs0
    y = pack_Cb_to_y(C_init, bs_init)

    # resume from the latest existing per-leg checkpoint
    start = 0
    for i in range(len(LEGS), 0, -1):
        f = os.path.join(CKPT, f"after_leg{i}.npy")
        if os.path.exists(f):
            y = jnp.asarray(np.load(f)); start = i
            print(f"[resume] loaded {f}; starting at leg {i+1}", flush=True)
            break

    t_all = time.perf_counter()
    for i in range(start, len(LEGS)):
        y, solver = run_leg(y, LEGS[i], f"{i+1}/{len(LEGS)}")
        np.save(os.path.join(CKPT, f"after_leg{i+1}.npy"), np.asarray(y))
        if not bool(np.all(np.isfinite(np.asarray(y)))):
            print("!! non-finite — aborting", flush=True); return

    M, Fbar, fxc, Mlog = score(C16, bs0, solver, y)
    from datetime import datetime
    out = os.path.join(OUT, f"route_sa16_B_{datetime.now():%Y%m%d_%H%M%S}.npz")
    np.savez(out, M=M, Fbar=Fbar, F_xc=fxc, M_logical=Mlog,
             configs=np.array(CONFIGS), legs=np.array(LEGS),
             wall=time.perf_counter()-t_all)
    print("\n=== RESULT: full 4-qubit routed gate (routing B) ===", flush=True)
    print(f"  F_bar (16-dim, gauge-corrected) = {Fbar:.4f}   F_xc = {fxc:.4f}", flush=True)
    print(f"  (0,3) logical-corner survival per col (q0q3=00,10,01,11): "
          f"{np.round((np.abs(Mlog)**2).sum(0),3)}", flush=True)
    print("  |M_logical| (0,3 corner):\n", np.round(np.abs(Mlog), 3), flush=True)
    print(f"  total wall: {(time.perf_counter()-t_all)/3600:.2f} h  -> {out}", flush=True)


if __name__ == "__main__":
    main()
