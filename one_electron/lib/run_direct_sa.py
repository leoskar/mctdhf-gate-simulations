"""Direct long-range sqrt(iSWAP)(0,3) on the 16-state register, SA-16, x64 VMF/CPU.

One leg at the w0w3 target point, no routing.  Pulse: tau_ramp = 10,
tau_hold = 2111.3 (the hold-scan maximum 2111.295, snapped to the
SNAP_DT = 0.1 checkpoint grid), propagated from t = 0 through ramp, hold and
ramp-down.  Basis states and scoring are imported from run_route_sa.py.
Built-in check: the (0,3) corner F_bar4 should reproduce the 4-state
hold-scan value 0.81103.

This VMF run is an earlier test, not one of the paper's results; the paper's
direct gate is the 16-state confirmation in gate_sweeps/scripts/confirm_k16.py.

    python lib/run_direct_sa.py      (from one_electron/)
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
from qoct_gate_leg import (DT_PROD, G_DICT_64, NUM_ORB, VMF_MIN_TOL,
                           VMF_MAX_ITER, _make_h_s_x64, mctdh_overlap)
from sweep_map import SNAP_DT, SNAP_STEPS
from rte_step import MAX_SOLVER_TOL, NUM_DVR, NUM_NEWTON, NUM_WELLS
from geometry_config import ANCHOR_PATHS, GEOM_TAG
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_vmf_integrator

# identical basis states + scoring as the routed arms — do not re-implement.
import run_route_sa as _B

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "ckpt_direct"); os.makedirs(CKPT, exist_ok=True)

# ── schedule ─────────────────────────────────────────────────────────────────
# Stage-5 w0w3 superposition anchor = the long-range target point.
_X5_PATH = ANCHOR_PATHS[5]
X_RES = jnp.asarray(np.load(_X5_PATH)["best_x"], dtype=jnp.float64)

TAU_RAMP = 10.0
TAU_HOLD = 2111.3          # hold-scan maximum 2111.295, snapped to SNAP_DT
T_LEG = round(2 * TAU_RAMP + TAU_HOLD, 6)
HOLDSCAN_FBAR4 = 0.81103   # 4-state value at this pulse (hold scan)

# Mid-leg checkpoint cadence: one leg of ~2131 t.u. is a single 15-18 h block,
# so save periodically rather than only at the end.
CKPT_EVERY = 5000          # chunks -> every 500 t.u. (~4 h)


def make_solver():
    _gl.x_star_2_ghz = X_RES          # read by _make_h_s_x64 at trace time
    h_list = [_make_h_s_x64(s, TAU_RAMP, TAU_HOLD) for s in range(NUM_WELLS)]
    assert h_list[0](0.0).dtype == jnp.float64, "h_list downcast — x64 is inert"
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=16, imag_time=False)
    # x64 gate fidelity REQUIRES e_shift = 0: "auto" strips the per-state
    # entangling phase (E11+E00-E10-E01)*T and fakes a per-state/SA gap.
    assert solver.e_shift == 0.0, f"e_shift={solver.e_shift} — must be 0 for x64 gates"
    integ = get_vmf_integrator(solver, DT_PROD, num_newton=NUM_NEWTON,
                               min_solver_tol=VMF_MIN_TOL,
                               max_solver_tol=MAX_SOLVER_TOL,
                               max_solver_iter=VMF_MAX_ITER, precondition=True)
    return solver, integ


def norms(solver, y):
    C, bs = solver._y_to_Cb(y)
    return np.array([float(jnp.real(mctdh_overlap(C[j], bs, C[j], bs)))
                     for j in range(16)])


def main():
    n_chunks = int(round(T_LEG / SNAP_DT))
    assert abs(n_chunks * SNAP_DT - T_LEG) < 1e-9, \
        f"T_leg={T_LEG} is not an integer multiple of SNAP_DT={SNAP_DT}"

    # Smoke hook: truncate the propagation to exercise imports, asserts, one
    # compile, the checkpoint path and the full scoring path in ~minutes. Unset
    # in production, where it changes nothing.
    _smoke = int(os.environ.get("DIRECT_SMOKE_CHUNKS", "0"))
    if _smoke:
        n_chunks = min(n_chunks, _smoke)
        print(f"!! SMOKE MODE: {n_chunks} chunks only — result is NOT the gate",
              flush=True)

    print("=== SA-16 DIRECT long-range sqiSWAP(0,3), x64 VMF/CPU ===", flush=True)
    print(f"settings: norb={NUM_ORB} dt={DT_PROD} tol=[{VMF_MIN_TOL},{MAX_SOLVER_TOL}] "
          f"newton={NUM_NEWTON} geom={GEOM_TAG}", flush=True)
    print(f"pulse: tau_ramp={TAU_RAMP} tau_hold={TAU_HOLD} T_leg={T_LEG} t.u. = "
          f"{T_LEG*15.915494309189535/1000:.2f} ns  ({n_chunks} chunks)", flush=True)
    print(f"resonance anchor: {os.path.basename(_X5_PATH)}", flush=True)
    print(f"  x_star_2 = {np.asarray(X_RES)}", flush=True)
    # Guard against the silent-fallback-geometry trap: the resonance plateau must
    # actually differ from idle, or we would propagate an idle window for 18 h.
    dx = float(jnp.max(jnp.abs(X_RES - jnp.asarray(_gl.x_star_1_ghz))))
    assert dx > 1e-6, "resonance vector == idle: wrong anchor file"
    print(f"  max|x_res - x_idle| = {dx:.6g}", flush=True)
    print(f"  reference: routing B F16=0.4253 corner=0.8790 @ 67.23 ns | "
          f"PAR 0.2751 / 0.7275 @ 36.51 ns | direct 4-state F4={HOLDSCAN_FBAR4}",
          flush=True)

    C16, bs0 = _B.basis_states()
    y = pack_Cb_to_y(jnp.stack(C16), bs0)
    solver, integ = make_solver()

    n0 = norms(solver, np.asarray(y))
    print(f"init norms [min,max] = [{n0.min():.8f},{n0.max():.8f}]", flush=True)
    assert np.allclose(n0, 1.0, atol=1e-8), "initial basis states not normalised"

    # resume from a mid-leg checkpoint if one exists
    start = 0
    f_res = os.path.join(CKPT, "mid.npz")
    if os.path.exists(f_res):
        d = np.load(f_res)
        y = jnp.asarray(d["y"]); start = int(d["chunk"])
        print(f"[resume] {f_res} at chunk {start}/{n_chunks} "
              f"(t={start*SNAP_DT:.1f})", flush=True)

    def step(carry, _):
        t, yy = carry
        return integ(t, yy), None
    chunk = jax.jit(lambda c: jax.lax.scan(step, c, None, length=SNAP_STEPS)[0])

    t_all = time.perf_counter()
    carry = (jnp.asarray(start * SNAP_DT), y)
    t0 = time.perf_counter()
    carry = chunk(carry); jax.block_until_ready(carry)
    print(f"  compile+chunk1: {time.perf_counter()-t0:.1f}s", flush=True)
    tic = time.perf_counter()
    for c in range(start + 1, n_chunks):
        carry = chunk(carry)
        if (c + 1) % 500 == 0:
            jax.block_until_ready(carry); t, yy = carry
            nn = norms(solver, np.asarray(yy))
            el = time.perf_counter() - tic
            print(f"  chunk {c+1}/{n_chunks} t={float(t):.1f} "
                  f"norm[min,max]=[{nn.min():.5f},{nn.max():.5f}] "
                  f"finite={bool(np.all(np.isfinite(np.asarray(yy))))} "
                  f"({el:.0f}s, eta {el/(c+1-start)*(n_chunks-c-1)/3600:.1f}h)",
                  flush=True)
        if (c + 1) % CKPT_EVERY == 0:
            jax.block_until_ready(carry)
            np.savez(f_res, y=np.asarray(carry[1]), chunk=c + 1)
    jax.block_until_ready(carry)
    _, y_fin = carry

    nn = norms(solver, np.asarray(y_fin))
    print(f"leg done {time.perf_counter()-tic:.0f}s "
          f"norm[min,max]=[{nn.min():.6f},{nn.max():.6f}] "
          f"finite={bool(np.all(np.isfinite(np.asarray(y_fin))))}", flush=True)
    if not bool(np.all(np.isfinite(np.asarray(y_fin)))):
        print("!! non-finite — aborting before scoring", flush=True); return

    M, Fbar, fxc, Mlog = _B.score(C16, bs0, solver, y_fin)
    surv = (np.abs(M) ** 2).sum(0)

    from datetime import datetime
    out = os.path.join(OUT, f"route_sa16_DIRECT_{datetime.now():%Y%m%d_%H%M%S}.npz")
    # Insurance save before the corner's Nelder-Mead fit; re-saved enriched below.
    np.savez(out, M=M, Fbar=Fbar, F_xc=fxc, M_logical=Mlog,
             configs=np.array(_B.CONFIGS), norms_final=nn, survival=surv,
             tau=np.array([TAU_RAMP, TAU_HOLD]), T_total=T_LEG,
             wall=time.perf_counter() - t_all)

    print("\n=== RESULT: DIRECT long-range sqiSWAP(0,3), K=16 ===", flush=True)
    print(f"  F_bar16 (gauge-corrected) = {Fbar:.4f}   F_xc = {fxc:.4f}", flush=True)
    print(f"    (routing B 0.4253 @ 67.23 ns | PAR 0.2751 @ 36.51 ns)", flush=True)
    print(f"  per-input survival: min {surv.min():.3f} mean {surv.mean():.3f}",
          flush=True)
    print(f"    (routing B min 0.782 mean 0.956 | PAR min 0.935 mean 0.984)",
          flush=True)
    print("  |M_logical| (0,3 corner):\n", np.round(np.abs(Mlog), 3), flush=True)
    print(f"  corner survival per col: {np.round((np.abs(Mlog)**2).sum(0), 3)}",
          flush=True)

    # ── consistency check against the 4-state hold scan ──────────────────────
    # Corner scored with the same metric as the 4-state hold scan
    # (avg_gate_fidelity_up_to_localZ, M_REFS["sqiswap"]), so the two numbers
    # are directly comparable. Index-order note: _B.score's corner comes out in
    # (q0q3) = (00,01,10,11) while gate_fidelity's convention is (00,10,01,11);
    # F_bar is invariant under that relabelling because the sqiSWAP reference is
    # symmetric under exchanging the two qubits and the Z-gauge is maximised
    # over both angles. Do not "fix" the order — it would change nothing.
    Fbar4 = _gl.avg_gate_fidelity_up_to_localZ(_gl.M_REFS["sqiswap"], Mlog)
    print(f"  (0,3) corner F_bar4 = {Fbar4:.5f}  vs 4-state hold-scan "
          f"{HOLDSCAN_FBAR4:.5f}  -> d = {Fbar4-HOLDSCAN_FBAR4:+.5f}", flush=True)
    print(f"    (routing B corner 0.8790 | PAR corner 0.7275)", flush=True)
    np.savez(out, M=M, Fbar=Fbar, F_xc=fxc, M_logical=Mlog, Fbar4_corner=Fbar4,
             configs=np.array(_B.CONFIGS), norms_final=nn, survival=surv,
             tau=np.array([TAU_RAMP, TAU_HOLD]), T_total=T_LEG,
             holdscan_fbar4=HOLDSCAN_FBAR4,
             wall=time.perf_counter() - t_all)
    print(f"  total wall: {(time.perf_counter()-t_all)/3600:.2f} h  -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
