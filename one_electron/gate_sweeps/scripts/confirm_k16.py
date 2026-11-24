"""Register-level (K=16) confirmation of a calibrated sqiSWAP leg (Sec. IV A 1).

The norb=4 confirmations (confirm_norb4.py) propagate num_states=4: the four
configurations of the active pair with the two spectator wells in |0>.  That
gives the fidelity on the pair, not on the register.  This script propagates
all sixteen 4-qubit register basis states (num_states=16, shared orbitals)
and scores the 16x16 gate against sqiSWAP(pair) (x) I(spectators).  The two
are not related by a formula: errors on spectator-excited inputs are
invisible to a 4-state propagation with the spectators in |0>.

The 16 basis states, the local-Z gauge optimisation (fxc_4q) and the Fbar_16
formula are those of lib/run_route_sa.py; only the target is generalised from
the (0,3) pair to an arbitrary pair, and that generalisation is asserted to
reproduce run_route_sa.target_sqiswap_03() exactly.  Integrator: CMF
half-step + per-chunk regauge.  The paper's numbers are the rescored ones of
rescore_k16.py.

    LEG=w1w2 TAU_RAMP=57.2 TAU_HOLD=95.4 python confirm_k16.py
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
os.environ["MEANFIELD_SPEC"] = "0"
os.environ.setdefault("SWEEP_LEG_TYPE", "sqiswap")
os.environ.setdefault("SWEEP_INTEG", "cmf")
os.environ.setdefault("SWEEP_DT_DIV", "0.1")
assert "SWEEP_DT" not in os.environ, "SWEEP_DT is not SWEEP_DT_DIV"

# sweep_sa rebinds qoct_gate_leg.x_star_2_ghz to THIS leg's resonance anchor and
# gives NORB_LIST/SWEEP_DT/SNAP_DT -- the same objects the sweep used.
import sweep_sa as S
import run_route_sa as B                    # basis states + validated scoring
from qoct_gate_leg import G_DICT_64, _make_h_s_x64, mctdh_overlap
from rte_step import NUM_DVR, NUM_WELLS
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_cmf_half_step_integrator
from regauge import make_regauge
from common import phase_fix_anchors, anchor_phase_report

assert S.NORB_LIST == [4, 4, 4, 4], f"expected norb=4 everywhere, got {S.NORB_LIST}"
assert S.SWEEP_LEG_TYPE == "sqiswap"
assert abs(S.SNAP_DT - 0.1) < 1e-12

PAIR = S._LEG_PAIR
SPECT = tuple(s for s in range(NUM_WELLS) if s not in PAIR)
TAU_RAMP = float(os.environ["TAU_RAMP"])
TAU_HOLD = float(os.environ["TAU_HOLD"])

_S2 = 1.0 / np.sqrt(2.0)
_SQ2 = np.array([[1, 0, 0, 0], [0, _S2, 1j*_S2, 0],
                 [0, 1j*_S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)


def target_sqiswap(pair):
    """sqiSWAP on `pair` (x) identity on the other two wells, in plain bit order
    (q0<<3 | q1<<2 | q2<<1 | q3)."""
    a, b = pair
    U = np.zeros((16, 16), complex)
    for i in range(16):
        q = [(i >> (3 - k)) & 1 for k in range(4)]
        for li in range(4):
            qa, qb = (li >> 1) & 1, li & 1
            amp = _SQ2[2 * q[a] + q[b], 2 * qa + qb]
            if amp != 0:
                o = list(q); o[a], o[b] = qa, qb
                idx = sum(bit << (3 - k) for k, bit in enumerate(o))
                U[idx, i] = amp
    return U


# Equivalence gate: the generalisation must reproduce the validated (0,3) target.
assert np.allclose(target_sqiswap((0, 3)), B.target_sqiswap_03(), atol=0), \
    "generalised target disagrees with run_route_sa.target_sqiswap_03()"


def score16(C16, bs0, solver, y_fin):
    """As run_route_sa.score, but with the pair-general target and corner."""
    C_T, bs_T = solver._y_to_Cb(y_fin)
    M = np.zeros((16, 16), complex)
    for j in range(16):
        for i in range(16):
            M[i, j] = complex(mctdh_overlap(C16[i], bs0, C_T[j], bs_T))
    bit = np.array([(c[0] << 3) | (c[1] << 2) | (c[2] << 1) | c[3] for c in B.CONFIGS])
    P = np.zeros((16, 16))
    for p in range(16):
        P[bit[p], p] = 1.0
    M = P @ M @ P.T                                   # CONFIGS order -> bit order
    U = target_sqiswap(PAIR)
    fxc = B.fxc_4q(U, M)
    d = 16
    Fbar = (np.vdot(M, M).real
            + fxc * np.vdot(U, U).real * np.vdot(M, M).real) / (d * (d + 1))
    # logical corner: both spectators 0 in and out -> 4x4 vs the 2-qubit sqiSWAP
    sub = [i for i in range(16)
           if all(((i >> (3 - s)) & 1) == 0 for s in SPECT)]
    return M, float(Fbar), float(fxc), M[np.ix_(sub, sub)]


def _main():
    T_total = round(2 * TAU_RAMP + TAU_HOLD, 6)
    n_chunks = int(round(T_total / S.SNAP_DT))
    assert abs(n_chunks * S.SNAP_DT - T_total) < 1e-9
    print(f"=== K=16 register confirmation (sqiSWAP): {LEG} pair={PAIR} "
          f"spectators={SPECT} tau=({TAU_RAMP},{TAU_HOLD}) T={T_total} t.u. = "
          f"{T_total*15.9155e-3:.2f} ns  norb={S.NORB_LIST} dt={S.SWEEP_DT:g} "
          f"dev={os.environ.get('SWEEP_DEV','cpu')} ===", flush=True)

    C16, bs0 = B.basis_states()
    # ANCHOR PHASE (lib/common.py:phase_fix_anchors, applied
    # by run_arm_decomp.py). The 16 anchors are separately computed eigenstates
    # with independent arbitrary phases; under Psi_i -> e^{i th_i} Psi_i the gate
    # matrix transforms as M -> diag(e^{-i th}) M diag(e^{+i th}). fxc_4q's gauge
    # is PRODUCT-FORM ONLY (_zphases builds exp(i bits @ thetas) from 4+4
    # generators), so the non-product part of a 16-phase diagonal cannot be
    # absorbed and lands in Fbar16. Two-qubit fidelities ARE invariant to it,
    # which is why the corner F4 was right while Fbar16 was not.
    ph_before = anchor_phase_report(C16)
    C16 = [jnp.asarray(c, dtype=jnp.complex128)
           for c in phase_fix_anchors(C16)[0]]
    print(f"  anchor dominant-coeff phases (deg) before: "
          f"{np.round(ph_before, 1)}", flush=True)
    print(f"                                     after : "
          f"{np.round(anchor_phase_report(C16), 6)}", flush=True)
    y0 = pack_Cb_to_y(jnp.stack(C16), bs0)
    h_list = [_make_h_s_x64(s, TAU_RAMP, TAU_HOLD) for s in range(NUM_WELLS)]
    assert h_list[0](0.0).dtype == jnp.float64, "h_list downcast — x64 is inert"
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=S.NORB_LIST,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=16, imag_time=False)
    assert solver.e_shift == 0.0, "e_shift must be 0 for x64 gate fidelity"

    n0 = np.array([float(jnp.real(mctdh_overlap(C16[j], bs0, C16[j], bs0)))
                   for j in range(16)])
    print(f"  init norms [min,max] = [{n0.min():.8f},{n0.max():.8f}]", flush=True)
    assert np.allclose(n0, 1.0, atol=1e-8), "initial basis states not normalised"

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
            el = time.perf_counter() - t0
            print(f"  chunk {c+1}/{n_chunks} ({el:.0f}s, eta "
                  f"{el/(c+1)*(n_chunks-c-1)/60:.0f} min)", flush=True)
    jax.block_until_ready(carry)
    wall = time.perf_counter() - t0
    y_fin = carry[1]
    assert bool(np.all(np.isfinite(np.asarray(y_fin)))), "non-finite final state"

    M, Fbar16, fxc, Mlog = score16(C16, bs0, solver, y_fin)
    surv = (np.abs(M) ** 2).sum(axis=0)
    from qoct_gate_leg import avg_gate_fidelity_up_to_localZ as Fbar4fn
    F_corner = float(Fbar4fn(_SQ2, Mlog))

    print(f"\n=== RESULT {LEG} ({TAU_RAMP},{TAU_HOLD}) ===", flush=True)
    print(f"  F_bar16 (gauge-corrected) = {Fbar16:.6f}   F_xc16 = {fxc:.6f}",
          flush=True)
    print(f"  logical corner F_bar4     = {F_corner:.6f}   "
          f"(K=4 confirmation for cross-check)", flush=True)
    print(f"  per-input survival: min {surv.min():.4f} mean {surv.mean():.4f}",
          flush=True)
    print(f"  wall {wall/60:.1f} min", flush=True)

    out = os.path.join(GS_DIR, "results", "k16_confirm")
    os.makedirs(out, exist_ok=True)
    assert out.startswith(GS_DIR + os.sep)
    f = os.path.join(out, f"k16_{LEG}_tr{TAU_RAMP:g}_th{TAU_HOLD:g}.npz")
    np.savez(f, M=M, M_logical=Mlog, Fbar16=Fbar16, F_xc16=fxc,
             F_corner4=F_corner, survival=surv, tau_ramp=TAU_RAMP,
             tau_hold=TAU_HOLD, leg=LEG, pair=np.array(PAIR), T_total=T_total,
             anchors_phase_fixed=True,
             wall=wall, norb=S.NORB_LIST, dt=S.SWEEP_DT)
    print(f"  -> {f}", flush=True)


if __name__ == "__main__":
    if os.environ.get("SWEEP_DEV", "cpu") == "cuda":
        with jax.default_device(jax.devices("cuda")[0]):
            _main()
    else:
        _main()
