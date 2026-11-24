"""Parallel routing of sqrt(iSWAP)(0,3) on the 16-state register.

    [SWAP(0,1) || SWAP(2,3)] . sqiSWAP(1,2) . [SWAP(0,1) || SWAP(2,3)]

Wells (0,1,2,3) hold qubits (qA,qB,qC,qD).  The first layer moves qA to well 1
and qD to well 2, sqiSWAP(1,2) entangles them, and the second layer undoes
the permutation, so the target is sqiSWAP(0,3) (x) I(1,2), scored by the code
of run_route_sa.py.  During a parallel layer the target point is the
entry-wise splice of the two single-pair target points (make_h_par, X_PAR);
each pair keeps its own pulse, and nothing is recalibrated for the joint
configuration.

routed_gates/scripts/run_arm_decomp.py uses make_h_par and X_PAR.  Run
directly, this file propagates the routing with VMF on the CPU and the
pulses below, with per-leg checkpoints in results/ckpt_par/; that run is an
earlier test, not one of the paper's results.
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

from qoct_gate_leg import (DT_PROD, G_DICT_64, NUM_ORB, VMF_MIN_TOL, VMF_MAX_ITER,
                           mctdh_overlap, WELL_CENTERS_64, X_S_64, T_S_64)
from sweep_map import SNAP_DT, SNAP_STEPS
from rte_step import (MAX_SOLVER_TOL, NUM_DVR, NUM_NEWTON, NUM_WELLS,
                      _envelope_param, x_star_1_ghz, TYPICAL_FREQ_GHZ)
from geometry_config import ANCHOR_PATHS
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_vmf_integrator

# identical basis states + scoring as routing B (same anchors, same Hungarian
# match, same gauge-corrected 4-qubit F_bar) — do not re-implement.
import run_route_sa as _B

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "ckpt_par"); os.makedirs(CKPT, exist_ok=True)

X1 = jnp.asarray(x_star_1_ghz, dtype=jnp.float64)
RES = {"w0w1": jnp.asarray(np.load(ANCHOR_PATHS[2])["best_x"], dtype=jnp.float64),
       "w1w2": jnp.asarray(np.load(ANCHOR_PATHS["3a"])["best_x"], dtype=jnp.float64),
       "w2w3": jnp.asarray(np.load(ANCHOR_PATHS["4a"])["best_x"], dtype=jnp.float64)}

# ── schedule ─────────────────────────────────────────────────────────────────
# Per-pair SWAP optima (k=1 revival, confirmed with norb=4).
TAU_01 = (40.0, 796.0)      # w0w1 SWAP  F=0.9538   T =  876.0 t.u. = 13.94 ns
TAU_23 = (110.0, 845.0)     # w2w3 SWAP  F=0.9963   T = 1065.0 t.u. = 16.95 ns
T_LAYER = 2 * TAU_23[0] + TAU_23[1]   # 1065.0 — the longer pair sets the layer;
                                      # pair (0,1) sits at idle for the last ~3 ns
TAU_PAR = {0: TAU_01, 1: TAU_01, 2: TAU_23, 3: TAU_23}

# Composite plateau: entry-wise splice. Anchor 2 sets wells 0,1 and leaves 2,3
# at idle; anchor 4a sets wells 2,3 and leaves 0,1 at idle — so this is exactly
# "both anchors at once" with no interpolation or refitting.
X_PAR = jnp.concatenate([RES["w0w1"][0:2], RES["w2w3"][2:4],
                         RES["w0w1"][4:6], RES["w2w3"][6:8]])

# sqiSWAP(1,2) entangler — SET FROM THE w1w2 SWEEP (Step 2). Env override lets
# the SLURM script pin it without editing the file; the assert stops the run
# from ever silently propagating a placeholder.
TAU_SQI12 = os.environ.get("TAU_SQI12", "")
assert TAU_SQI12, ("TAU_SQI12 unset: pass the sqiSWAP(1,2) optimum from the "
                   "w1w2 sweep, e.g. TAU_SQI12=37.7,123.5")
TAU_SQI12 = tuple(float(v) for v in TAU_SQI12.split(","))

# (label, plateau vector, per-well taus)
LEGS = [("par",   X_PAR,        TAU_PAR),
        ("sqi12", RES["w1w2"],  {s: TAU_SQI12 for s in range(NUM_WELLS)}),
        ("par",   X_PAR,        TAU_PAR)]


def make_h_par(s, x2, tau_ramp, tau_hold):
    """Erf-edged ramp/hold h_s(t) at x64 with plateau vector and ramp schedule
    passed EXPLICITLY. qoct_gate_leg._make_h_s_x64 reads the module global
    x_star_2_ghz at trace time and shares one tau across all wells; the parallel
    layer needs per-well taus, so this stays local and argument-driven."""
    centre = WELL_CENTERS_64[s]
    x_s = X_S_64[s]
    T_s = T_S_64[s]

    def h_s_t(t):
        x_ghz = X1 + (x2 - X1) * _envelope_param(t, tau_ramp, tau_hold)
        a = x_ghz[s] / TYPICAL_FREQ_GHZ
        b = x_ghz[NUM_WELLS + s] / TYPICAL_FREQ_GHZ
        dx = x_s - centre
        v = 0.5 * a**2 * dx**2 + (b * a**2 / 3.0) * dx**4
        return T_s + jnp.diag(v)

    return h_s_t


def make_solver(x2, taus):
    h_list = [make_h_par(s, x2, *taus[s]) for s in range(NUM_WELLS)]
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=16, imag_time=False)
    integ = get_vmf_integrator(solver, DT_PROD, num_newton=NUM_NEWTON,
                               min_solver_tol=VMF_MIN_TOL,
                               max_solver_tol=MAX_SOLVER_TOL,
                               max_solver_iter=VMF_MAX_ITER, precondition=True)
    return solver, integ


def norms(solver, y):
    C, bs = solver._y_to_Cb(y)
    return np.array([float(jnp.real(mctdh_overlap(C[j], bs, C[j], bs)))
                     for j in range(16)])


def run_leg(y0, leg, leg_label):
    label, x2, taus = leg
    T_leg = round(max(2 * tr + th for tr, th in taus.values()), 6)
    n_chunks = int(round(T_leg / SNAP_DT))
    solver, integ = make_solver(x2, taus)

    def step(carry, _):
        t, y = carry
        return integ(t, y), None
    chunk = jax.jit(lambda c: jax.lax.scan(step, c, None, length=SNAP_STEPS)[0])

    taus_str = " ".join(f"w{s}:({tr:g},{th:g})" for s, (tr, th) in sorted(taus.items()))
    print(f"=== leg {leg_label} [{label}] {taus_str} T_leg={T_leg} "
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


def main():
    T_tot = 2 * T_LAYER + (2 * TAU_SQI12[0] + TAU_SQI12[1])
    print("=== SA-16 PARALLEL-routed sqiSWAP(0,3), x64 VMF/CPU ===", flush=True)
    print(f"settings: norb={NUM_ORB} dt={DT_PROD} tol=[{VMF_MIN_TOL},{MAX_SOLVER_TOL}] "
          f"newton={NUM_NEWTON}", flush=True)
    print(f"schedule: [SWAP(0,1)||SWAP(2,3)] . sqiSWAP(1,2) tau={TAU_SQI12} . "
          f"[SWAP(0,1)||SWAP(2,3)]", flush=True)
    print(f"          T_layer={T_LAYER}  T_total={T_tot} t.u. = "
          f"{T_tot*15.915494309189535/1000:.2f} ns  "
          f"(routing B: 4223.9 t.u. = 67.23 ns)", flush=True)
    C16, bs0 = _B.basis_states()
    y = pack_Cb_to_y(jnp.stack(C16), bs0)

    start = 0
    for i in range(len(LEGS), 0, -1):
        f = os.path.join(CKPT, f"after_leg{i}.npy")
        if os.path.exists(f):
            y = jnp.asarray(np.load(f)); start = i
            print(f"[resume] loaded {f}; starting at leg {i+1}", flush=True)
            break

    t_all = time.perf_counter()
    solver = None
    for i in range(start, len(LEGS)):
        y, solver = run_leg(y, LEGS[i], f"{i+1}/{len(LEGS)}")
        np.save(os.path.join(CKPT, f"after_leg{i+1}.npy"), np.asarray(y))
        if not bool(np.all(np.isfinite(np.asarray(y)))):
            print("!! non-finite — aborting", flush=True); return
    if solver is None:      # fully resumed: rebuild the final leg's solver
        solver, _ = make_solver(*LEGS[-1][1:])

    M, Fbar, fxc, Mlog = _B.score(C16, bs0, solver, y)
    from datetime import datetime
    out = os.path.join(OUT, f"route_sa16_PAR_{datetime.now():%Y%m%d_%H%M%S}.npz")
    np.savez(out, M=M, Fbar=Fbar, F_xc=fxc, M_logical=Mlog,
             configs=np.array(_B.CONFIGS),
             legs=np.array([l[0] for l in LEGS]),
             tau_sqi12=np.array(TAU_SQI12), T_total=T_tot,
             wall=time.perf_counter() - t_all)
    print("\n=== RESULT: full 4-qubit PARALLEL-routed gate ===", flush=True)
    print(f"  F_bar (16-dim, gauge-corrected) = {Fbar:.4f}   F_xc = {fxc:.4f}",
          flush=True)
    print(f"    (routing B: F_bar16 = 0.425 @ 67.23 ns)", flush=True)
    print(f"  (0,3) logical-corner survival per col (q0q3=00,10,01,11): "
          f"{np.round((np.abs(Mlog)**2).sum(0),3)}", flush=True)
    print("  |M_logical| (0,3 corner):\n", np.round(np.abs(Mlog), 3), flush=True)
    print(f"    (routing B corner F_bar4 = 0.879 @ 67 ns; direct 0.811 @ 35 ns)",
          flush=True)
    print(f"  total wall: {(time.perf_counter()-t_all)/3600:.2f} h  -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
