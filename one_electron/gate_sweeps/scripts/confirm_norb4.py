"""Confirm a swept sqiSWAP optimum at full norb=4 (no mean-field surrogate).

Notes:
  * SWEEP_DEV=cuda is honoured (jax.default_device), needed for the w0w3 leg
  * SWEEP_LEG_TYPE defaults to (and is asserted) "sqiswap"
  * the report labels F_sqiSWAP as THE result and iSWAP/SWAP as rotation
    diagnostics

Propagates ONE full genuine leg (no checkpoint reuse) at the as-built anchor.

    LEG=w1w2 TAU_RAMP=36 TAU_HOLD=145.4 python confirm_norb4.py
"""
from __future__ import annotations
import os, sys, time

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
GS_DIR = os.path.abspath(os.path.join(HERE, ".."))
LIB = os.path.abspath(os.path.join(GS_DIR, "..", "lib"))   # one_electron/lib
sys.path.insert(0, LIB)

LEG = os.environ["LEG"]
os.environ["GATE_LEG"] = LEG
os.environ.setdefault("SWEEP_LEG_TYPE", "sqiswap")
os.environ["MEANFIELD_SPEC"] = "0"                # the whole point
os.environ.setdefault("SWEEP_INTEG", "cmf")
os.environ.setdefault("SWEEP_DT_DIV", "0.1")
assert os.environ["SWEEP_LEG_TYPE"] == "sqiswap"
assert "SWEEP_DT" not in os.environ, "SWEEP_DT is not SWEEP_DT_DIV"

import sweep_sa as S
from qoct_gate_leg import (G_DICT_64, _make_h_s_x64, gate_matrix,
                           load_loss_anchors,
                           avg_gate_fidelity_up_to_localZ as Fbar)
from rte_step import NUM_DVR, NUM_WELLS
from mctdhf import MCTDHF_n_species_DVR
from mctdhf.integrators import get_cmf_half_step_integrator
from regauge import make_regauge

assert S.NORB_LIST == [4, 4, 4, 4], f"expected norb=4 everywhere, got {S.NORB_LIST}"
assert abs(S.SNAP_DT - 0.1) < 1e-12

S2 = 1 / np.sqrt(2)
# Only sqiSWAP is the RESULT; iSWAP/SWAP are rotation-angle diagnostics.
TARGETS = {
    "sqiSWAP": np.array([[1,0,0,0],[0,S2,1j*S2,0],[0,1j*S2,S2,0],[0,0,0,1]], complex),
    "iSWAP": np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], complex),
    "SWAP": np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], complex),
}
TAU_RAMP = float(os.environ["TAU_RAMP"])
TAU_HOLD = float(os.environ["TAU_HOLD"])


def zz_phase(M):
    v = (np.angle(M[3, 3]) + np.angle(M[0, 0])
         - np.angle(M[2, 1]) - np.angle(M[1, 2]))
    return np.degrees((v + np.pi) % (2 * np.pi) - np.pi)


def _main():
    T_total = 2 * TAU_RAMP + TAU_HOLD
    n_chunks = int(round(T_total / S.SNAP_DT))
    print(f"=== norb=4 confirmation (sqiSWAP): {LEG} tau=({TAU_RAMP},{TAU_HOLD}) "
          f"T={T_total} t.u. = {T_total*15.9155e-3:.2f} ns  norb={S.NORB_LIST} "
          f"dt={S.SWEEP_DT:g} dev={os.environ.get('SWEEP_DEV','cpu')} ===",
          flush=True)
    C4, bs0 = load_loss_anchors()
    y0, _, _ = S._sa_y0()
    h_list = [_make_h_s_x64(s, TAU_RAMP, TAU_HOLD) for s in range(NUM_WELLS)]
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1]*NUM_WELLS,
        num_mctdhf_orbitals=S.NORB_LIST,
        num_spatial_orbitals=[NUM_DVR]*NUM_WELLS,
        num_states=4, imag_time=False)
    assert solver.e_shift == 0.0
    integ = get_cmf_half_step_integrator(solver, S.SWEEP_DT,
                                         c_propagator="diagonalize",
                                         b_propagator="diagonalize")
    rg = make_regauge(solver)

    def step(c, _):
        return integ(c[0], c[1]), None
    _c = jax.jit(lambda c: jax.lax.scan(step, c, None, length=S.SWEEP_STEPS)[0])
    chunk = jax.jit(lambda c: (lambda tv: (tv[0], rg(tv[1])))(_c(c)))

    carry = (jnp.asarray(0.0), jnp.asarray(y0))
    t0 = time.perf_counter()
    for c in range(n_chunks):
        carry = chunk(carry)
        if (c + 1) % max(1, n_chunks // 10) == 0:
            jax.block_until_ready(carry)
            print(f"  chunk {c+1}/{n_chunks} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    jax.block_until_ready(carry)
    wall = time.perf_counter() - t0

    C_fin, bs_fin = solver._y_to_Cb(carry[1])
    assert bool(np.all(np.isfinite(np.asarray(carry[1])))), "non-finite final state"
    M = gate_matrix(C4, bs0, [(C_fin[j], bs_fin) for j in range(4)])
    sub = (np.abs(M) ** 2).sum(axis=0)
    res = {f"F_{k}": float(Fbar(U, M)) for k, U in TARGETS.items()}
    print(f"  F_sqiSWAP  = {res['F_sqiSWAP']:.6f}   <-- THE RESULT", flush=True)
    print(f"  [diagnostic] F_iSWAP = {res['F_iSWAP']:.6f}  "
          f"F_SWAP = {res['F_SWAP']:.6f}  (rotation angle only, NOT the gate)",
          flush=True)
    print(f"  conditional phase = {zz_phase(M):+.2f} deg", flush=True)
    print(f"  |11> retention    = {sub[3]:.5f}   subspace {np.round(sub,5)}",
          flush=True)
    print(f"  wall {wall/60:.1f} min", flush=True)
    out = os.path.join(GS_DIR, "results", "norb4_confirm")
    os.makedirs(out, exist_ok=True)
    assert out.startswith(GS_DIR + os.sep)
    f = os.path.join(out, f"n4_{LEG}_tr{TAU_RAMP:g}_th{TAU_HOLD:g}.npz")
    np.savez(f, M=M, subspace=sub, tau_ramp=TAU_RAMP, tau_hold=TAU_HOLD,
             leg=LEG, wall=wall, norb=S.NORB_LIST, dt=S.SWEEP_DT, **res)
    print(f"  -> {f}", flush=True)


if __name__ == "__main__":
    if os.environ.get("SWEEP_DEV", "cpu") == "cuda":
        with jax.default_device(jax.devices("cuda")[0]):
            _main()
    else:
        _main()
