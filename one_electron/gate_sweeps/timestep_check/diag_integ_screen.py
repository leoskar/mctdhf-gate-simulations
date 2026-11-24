"""Quick screen: which integrator is less dt-sensitive on PHASE?

One integrator, one timestep, a TRUNCATED segment — designed so the four
configurations (VMF/CMF x dt/dt-half) run as parallel jobs and the whole screen
costs the wall time of the slowest, not the sum.

Rationale: the binding error here is accumulated dynamical phase. VMF carries
an O(dt^2) phase error from implicit midpoint; CMF with diagonalize/diagonalize
propagates the CI block exactly in RTE and the orbital substep by a dense
exponential, leaving only the O(dt^3) splitting error.

No converged reference is needed: comparing each integrator's OWN dt-to-dt/2
disagreement is a fair head-to-head of dt sensitivity.

    LEG=sqi12 INTEG=cmf N_CHUNKS=1637 SCREEN_DEV=cuda DT_DIV=1 sbatch job_screen.sh
(from this directory; see job_screen.sh)
"""
from __future__ import annotations
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(os.path.join(HERE, "..", "..", "lib"))   # one_electron/lib
sys.path.insert(0, LIB)

import qoct_gate_leg as _gl
from qoct_gate_leg import (G_DICT_64, NUM_ORB, VMF_MIN_TOL, VMF_MAX_ITER,
                           DT_PROD, _make_h_s_x64)
from rte_step import MAX_SOLVER_TOL, NUM_DVR, NUM_NEWTON, NUM_WELLS
from sweep_map import SNAP_DT
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import (get_vmf_integrator,
                                get_cmf_half_step_integrator)
import run_route_sa as _B
import run_direct_sa_gauge as _G   # exact gauge re-orthonormalisation

LEG_SPECS = {k: (_B.RES[k], _B.TAU[k]) for k in ("w0w1", "w1w2", "w2w3")}
LEG_SPECS["sqi12"] = (_B.RES["w1w2"], (12.5, 138.7))
# The direct (long-range) arm is a SINGLE leg and is not in run_route_sa.RES --
# run_direct_sa.py reads its resonance from the stage-5 w0w3 anchor and uses the
# hold-scan maximum as its pulse. Replicated here:
# T = 2*10 + 2111.3 = 2131.3 t.u. -> N_CHUNKS = 21313 at SNAP_DT = 0.1.
from geometry_config import ANCHOR_PATHS
LEG_SPECS["w0w3"] = (jnp.asarray(np.load(ANCHOR_PATHS[5])["best_x"],
                                 dtype=jnp.float64), (10.0, 2111.3))
LEG = os.environ.get("LEG", "sqi12")
INTEG = os.environ.get("INTEG", "vmf")
DIV = float(os.environ.get("DT_DIV", "1"))
N_CHUNKS = int(os.environ.get("N_CHUNKS", "50"))
# abort as soon as the run is provably lost (norm is the cheap tell)
ABORT_DRIFT = float(os.environ.get("ABORT_DRIFT", "1e-2"))
# CMF's b_propagator="diagonalize" is exact only in the LINEARISED orbital
# ODE; the projector P = b b^H makes it nonlinear, so the exponential does
# NOT preserve b^H b = I. Measured: w2w3 @ 2dt ended with b singular values
# 0.011..1.807 while ||C|| stayed exactly 1. Regauge restores the constraint
# exactly (Psi unchanged to 8.9e-16) -- NOT the built-in normalize modes.
REGAUGE = os.environ.get("REGAUGE", "1") == "1"
# cn / sil / diagonalize all solve the SAME frozen-coefficient ODE db/dt = A b
# and differ only in accuracy, so none of them can fix the non-Hermitian A.
B_PROP = os.environ.get("B_PROP", "diagonalize")
C_PROP = os.environ.get("C_PROP", "diagonalize")
DEV = os.environ.get("SCREEN_DEV", "cpu")
X_RES, TAU = LEG_SPECS[LEG]


def _main():
    dt = DT_PROD / DIV
    steps = int(round(SNAP_DT / dt))
    _gl.x_star_2_ghz = X_RES
    h_list = [_make_h_s_x64(s, TAU[0], TAU[1]) for s in range(NUM_WELLS)]
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=16, imag_time=False)
    assert solver.e_shift == 0.0
    if INTEG == "vmf":
        integ = get_vmf_integrator(solver, dt, num_newton=NUM_NEWTON,
                                   min_solver_tol=VMF_MIN_TOL,
                                   max_solver_tol=MAX_SOLVER_TOL,
                                   max_solver_iter=VMF_MAX_ITER,
                                   precondition=True)
    elif INTEG == "cmf":
        integ = get_cmf_half_step_integrator(solver, dt,
                                             c_propagator=C_PROP,
                                             b_propagator=B_PROP)
    else:
        raise SystemExit(f"bad INTEG={INTEG}")

    C16, bs0 = _B.basis_states()
    y = pack_Cb_to_y(jnp.stack(C16), bs0)

    def step(carry, _):
        t, yy = carry
        return integ(t, yy), None
    _chunk = jax.jit(lambda c: jax.lax.scan(step, c, None, length=steps)[0])
    if REGAUGE:
        regauge = _G.make_regauge(solver)
        def chunk(c):
            t, yy = _chunk(c)
            return (t, regauge(yy))
    else:
        chunk = _chunk

    print(f"=== SCREEN {INTEG.upper()} dt/{DIV:g} dev={DEV} leg={LEG} "
          f"{N_CHUNKS} chunks ({N_CHUNKS*SNAP_DT:.1f} t.u.) regauge={REGAUGE} "
          f"c={C_PROP} b={B_PROP} ===",
          flush=True)
    print(f"  jax devices: {jax.devices()}", flush=True)
    carry = (jnp.asarray(0.0), y)
    t0 = time.perf_counter(); carry = chunk(carry); jax.block_until_ready(carry)
    print(f"  compile+chunk1 {time.perf_counter()-t0:.1f}s", flush=True)
    tic = time.perf_counter()
    # Progress lines carry drift AND orthonormality, and abort early on blow-up.
    # (A failed run can otherwise look normal until its end-of-run drift.)
    every = max(1, N_CHUNKS // 40)
    for c in range(1, N_CHUNKS):
        carry = chunk(carry)
        if (c + 1) % every == 0:
            jax.block_until_ready(carry)
            yy = np.asarray(carry[1])
            nn = _B.norms(solver, yy)
            drift = float(np.abs(nn - 1).max())
            _, bs = solver._y_to_Cb(carry[1])
            oe = max(float(np.abs(np.asarray(b).conj().T @ np.asarray(b)
                                  - np.eye(b.shape[1])).max()) for b in bs)
            el = time.perf_counter() - tic
            print(f"  chunk {c+1}/{N_CHUNKS} t={float(carry[0]):.1f} "
                  f"drift={drift:.2e} ortho={oe:.2e} "
                  f"({el:.0f}s, eta {el/c*(N_CHUNKS-c-1)/60:.1f}m)", flush=True)
            if drift > ABORT_DRIFT or not np.all(np.isfinite(yy)):
                print(f"  !! ABORT at chunk {c+1} (t={float(carry[0]):.1f}): "
                      f"drift {drift:.3e} > {ABORT_DRIFT:g} "
                      f"(ortho {oe:.2e}) -- this step size fails at this leg "
                      f"length; stopping rather than burning the full run.",
                      flush=True)
                return
    jax.block_until_ready(carry)
    wall = time.perf_counter() - tic
    yf = np.asarray(carry[1])
    nn = _B.norms(solver, yf)
    nsteps = (N_CHUNKS - 1) * steps
    print(f"  done {wall/60:.2f} min  {wall/max(nsteps,1):.4f} s/step  "
          f"{wall/((N_CHUNKS-1)*SNAP_DT):.2f} s per t.u.  "
          f"drift={np.abs(nn-1).max():.2e}  "
          f"finite={bool(np.all(np.isfinite(yf)))}", flush=True)
    out = os.path.join(HERE, "results", f"screen_{LEG}_{INTEG}_{DIV:g}_{DEV}_n{N_CHUNKS}"
                         f"{os.environ.get('SCREEN_TAG','')}.npz")
    np.savez(out, y=yf, integ=INTEG, div=DIV, leg=LEG, n_chunks=N_CHUNKS,
             wall=wall, dev=DEV, s_per_step=wall/max(nsteps,1))
    print(f"  -> {out}", flush=True)


def main():
    if DEV == "cuda":
        gpus = jax.devices("cuda")
        print(f"  pinning to {gpus[0]}", flush=True)
        with jax.default_device(gpus[0]):
            _main()
    else:
        _main()


if __name__ == "__main__":
    main()
