"""The direct long-range sqrt(iSWAP)(0,3) of run_direct_sa.py with exact gauge
re-orthonormalisation of the orbitals between chunks.

Without it the VMF run fails near t = 2126: the orbitals of well 3 drift off
the orthonormal manifold (max|b^H b - I| reaches 0.12).  b^H b = I is only a
weak invariant of the projector equations, so nothing restores it once
numerical error has moved the orbitals off it.  make_regauge (b_s = Q_s R_s,
b_s <- Q_s, C <- R_s on mode s) uses the exact gauge freedom of MCTDH to
return to the orthonormal gauge without changing the wavefunction; see
regauge.py, which holds the same function.  It is not the solver's built-in
normalize/orthonormalize options, which change the wavefunction.

The time-step checks import make_regauge from here.  Run directly, this file
is an earlier VMF test, not one of the paper's results.
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
from qoct_gate_leg import NUM_ORB, VMF_MIN_TOL, VMF_MAX_ITER, DT_PROD
from rte_step import MAX_SOLVER_TOL, NUM_NEWTON
from sweep_map import SNAP_DT, SNAP_STEPS
from geometry_config import GEOM_TAG
from mctdhf import pack_Cb_to_y

import run_direct_sa as _D
import run_route_sa as _B

OUT = os.path.join(HERE, "results", "gauge"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "ckpt"); os.makedirs(CKPT, exist_ok=True)
CKPT_EVERY = 2500          # was 5000: run 1 aborted at 2500 with nothing saved

# chunk -> (max allowed drift, what the pre-fix runs measured there).
# Recalibrated after run 1 (see the EARLY GATE section of the docstring): the
# t=250 bar is set to catch a genuine no-op (1.5e-5) without tripping on the
# 1.45e-6 that run 1 actually achieved, and the decisive test is moved to
# t=1000 where the pre-fix drift was already 3.9e-3.
GATES = {2500: (3e-6, 1.50e-05), 10000: (1e-4, 3.86e-03)}


def make_regauge(solver):
    """Exact gauge re-orthonormalisation: b_s = Q_s R_s -> (Q_s, R_s x_s C).

    Psi is invariant by construction, so this changes the REPRESENTATION and
    not the state. The positive-diagonal convention makes Q, R unique, so the
    map is deterministic and the run is reproducible.
    """
    @jax.jit
    def regauge(y):
        C, bs = solver._y_to_Cb(y)
        bs_new = []
        for s, b in enumerate(bs):
            Q, R = jnp.linalg.qr(b)
            sgn = jnp.sign(jnp.real(jnp.diag(R)))
            sgn = jnp.where(sgn == 0, 1.0, sgn).astype(R.dtype)
            Q = Q * sgn[None, :]          # Q' = Q S
            R = sgn[:, None] * R          # R' = S R, so Q'R' = QR = b exactly
            bs_new.append(Q)
            # phi_m = sum_k R'[k,m] chi_k  =>  C'_k = sum_m R'[k,m] C_m
            C = jnp.moveaxis(
                jnp.tensordot(R, jnp.moveaxis(C, s + 1, 0), axes=([1], [0])),
                0, s + 1)
        return solver._Cb_to_y(C, bs_new)
    return regauge


def ortho_err(solver, y):
    """max_s max|b_s^H b_s - I| — the quantity that ran away in the first run."""
    _, bs = solver._y_to_Cb(y)
    out = []
    for b in bs:
        B = np.asarray(b)
        out.append(float(np.abs(B.conj().T @ B - np.eye(B.shape[1])).max()))
    return np.array(out)


def main():
    t_all = time.perf_counter()
    n_chunks = int(round(_D.T_LEG / SNAP_DT))

    print("=== SA-16 DIRECT sqiSWAP(0,3) — GAUGE RE-ORTHONORMALISED ===",
          flush=True)
    print(f"settings: norb={NUM_ORB} dt={DT_PROD} newton={NUM_NEWTON} "
          f"tol=[{VMF_MIN_TOL},{MAX_SOLVER_TOL}] max_iter={VMF_MAX_ITER} "
          f"geom={GEOM_TAG}  (production baseline, reverted from 4455)",
          flush=True)
    print(f"pulse: tau_ramp={_D.TAU_RAMP} tau_hold={_D.TAU_HOLD} "
          f"T_leg={_D.T_LEG} t.u. ({n_chunks} chunks)", flush=True)
    print(f"regauge: exact (b<-Q, C<-RC) every chunk = every {SNAP_DT} t.u.",
          flush=True)
    for _c, (_lim, _ref) in sorted(GATES.items()):
        print(f"gate: abort at chunk {_c} (t={_c*SNAP_DT:.0f}) if drift > "
              f"{_lim:g}   [pre-fix runs: {_ref:.2e} there]", flush=True)

    C16, bs0 = _B.basis_states()
    y = pack_Cb_to_y(jnp.stack(C16), bs0)
    solver, integ = _D.make_solver()          # production baseline settings
    regauge = make_regauge(solver)

    n0 = _D.norms(solver, np.asarray(y))
    print(f"init norms [min,max] = [{n0.min():.8f},{n0.max():.8f}]", flush=True)
    print(f"init ortho err per well = {np.array2string(ortho_err(solver, y), precision=2)}",
          flush=True)
    assert np.allclose(n0, 1.0, atol=1e-8), "initial basis states not normalised"

    # regauge is a no-op on an already-orthonormal state: verify Psi is untouched
    y_rg = regauge(y)
    dn = float(np.abs(_D.norms(solver, np.asarray(y_rg)) - n0).max())
    assert dn < 1e-12, f"regauge changed Psi by {dn:.3e} — NOT a gauge transform"
    print(f"regauge self-check: max|dNorm| = {dn:.2e} (must be ~0)", flush=True)

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
    _chunk = jax.jit(lambda c: jax.lax.scan(step, c, None, length=SNAP_STEPS)[0])

    def chunk(c):
        t, yy = _chunk(c)
        return (t, regauge(yy))

    carry = (jnp.asarray(start * SNAP_DT), y)
    t0 = time.perf_counter()
    carry = chunk(carry); jax.block_until_ready(carry)
    print(f"  compile+chunk1: {time.perf_counter()-t0:.1f}s", flush=True)

    tic = time.perf_counter()
    for c in range(start + 1, n_chunks):
        carry = chunk(carry)
        if (c + 1) % 500 == 0:
            jax.block_until_ready(carry); t, yy = carry
            nn = _D.norms(solver, np.asarray(yy))
            oe = ortho_err(solver, yy)
            el = time.perf_counter() - tic
            drift = float(np.max(np.abs(nn - 1.0)))
            print(f"  chunk {c+1}/{n_chunks} t={float(t):.1f} "
                  f"norm[min,max]=[{nn.min():.8f},{nn.max():.8f}] "
                  f"drift={drift:.2e} ortho={oe.max():.2e} "
                  f"finite={bool(np.all(np.isfinite(np.asarray(yy))))} "
                  f"({el:.0f}s, eta {el/(c+1-start)*(n_chunks-c-1)/3600:.1f}h)",
                  flush=True)
            if (c + 1) in GATES:
                lim, ref = GATES[c + 1]
                factor = ref / max(drift, 1e-30)
                if drift > lim:
                    print(f"\n!! GATE FAILED at t={float(t):.0f}: drift "
                          f"{drift:.3e} > {lim:g}", flush=True)
                    print(f"   pre-fix runs measured {ref:.2e} here, so this "
                          f"is only a {factor:.1f}x change — the gauge fix is "
                          f"not carrying the bleed. Aborting.", flush=True)
                    return
                print(f"\n>> GATE PASSED at t={float(t):.0f}: drift "
                      f"{drift:.3e} <= {lim:g} ({factor:.0f}x better than the "
                      f"pre-fix {ref:.2e}) — continuing\n", flush=True)
        if (c + 1) % CKPT_EVERY == 0:
            jax.block_until_ready(carry)
            np.savez(f_res, y=np.asarray(carry[1]), chunk=c + 1)
    jax.block_until_ready(carry)
    _, y_fin = carry

    nn = _D.norms(solver, np.asarray(y_fin))
    oe = ortho_err(solver, y_fin)
    print(f"leg done {time.perf_counter()-tic:.0f}s "
          f"norm[min,max]=[{nn.min():.6f},{nn.max():.6f}] "
          f"ortho={oe.max():.2e} "
          f"finite={bool(np.all(np.isfinite(np.asarray(y_fin))))}", flush=True)
    if not bool(np.all(np.isfinite(np.asarray(y_fin)))):
        print("!! non-finite — aborting before scoring", flush=True); return

    M, Fbar, fxc, Mlog = _B.score(C16, bs0, solver, y_fin)
    surv = (np.abs(M) ** 2).sum(0)
    Fbar4 = _gl.avg_gate_fidelity_up_to_localZ(_gl.M_REFS["sqiswap"], Mlog)

    from datetime import datetime
    out = os.path.join(OUT, f"route_sa16_DIRECT_gauge_{datetime.now():%Y%m%d_%H%M%S}.npz")
    np.savez(out, M=M, Fbar=Fbar, F_xc=fxc, M_logical=Mlog, Fbar4_corner=Fbar4,
             configs=np.array(_B.CONFIGS), norms_final=nn, survival=surv,
             ortho_final=oe, tau=np.array([_D.TAU_RAMP, _D.TAU_HOLD]),
             T_total=_D.T_LEG, holdscan_fbar4=_D.HOLDSCAN_FBAR4,
             wall=time.perf_counter() - t_all)

    print("\n=== RESULT: DIRECT sqiSWAP(0,3), K=16, GAUGE-FIXED ===", flush=True)
    print(f"  F_bar16 (gauge-corrected) = {Fbar:.4f}   F_xc = {fxc:.4f}", flush=True)
    print(f"    (routing B 0.4253 @ 67.23 ns | PAR 0.2751 @ 36.51 ns)", flush=True)
    print(f"  per-input survival: min {surv.min():.3f} mean {surv.mean():.3f}",
          flush=True)
    print(f"    (routing B min 0.782 mean 0.956 | PAR min 0.935 mean 0.984)",
          flush=True)
    print("  |M_logical| (0,3 corner):\n", np.round(np.abs(Mlog), 3), flush=True)
    print(f"  (0,3) corner F_bar4 = {Fbar4:.5f}  vs 4-state hold-scan "
          f"{_D.HOLDSCAN_FBAR4:.5f}  -> d = {Fbar4-_D.HOLDSCAN_FBAR4:+.5f}",
          flush=True)
    print(f"  final norm [min,max] = [{nn.min():.6f},{nn.max():.6f}]  "
          f"(REPORTED, never forced)", flush=True)
    print(f"  total wall: {(time.perf_counter()-t_all)/3600:.2f} h  -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
