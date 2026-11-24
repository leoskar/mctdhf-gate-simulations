"""Spectator-correlation test: run a 4-well gate leg with the SPECTATOR wells
truncated to norb ∈ {1,2,4}, active pair fixed at norb=4.

norb_spectator = 1 is TDHF (dynamical mean-field) for those wells; norb=4 is the
full multiconfiguration treatment. The active pair, geometry, anchors, schedule
and solver settings are IDENTICAL across the runs — the ONLY variable is
spectator correlation. The norb=1→4 gap in F̄ is the dynamical spectator-
correlation effect on the gate.

The active pair (--leg w0w1|w1w2|w2w3|w0w3), scored gate (--gate sqiswap|swap)
and pulse (--tau-ramp/--tau-hold) are parameters; leg retargeting follows sweep_sa.py
exactly (rebind qoct_gate_leg.x_star_2_ghz to the leg's resonance anchor and
qoct_gate_leg.PAIR_CONFIGS to the pair's loss basis). --ref-pt compares F̄
against a saved norb=4 sweep point (sweep_sa pt_*.npz) and gates |ΔF̄| < 1e-3.
Defaults: w0w1 √iSWAP at (40, 107).

Projection: the spectators' single (or few) orbital(s) = leading
natural orbital(s) of the SA-averaged spectator 1-RDM from the norb=4 anchor;
the CI is Tucker-projected onto them. This IS the mean-field/factorized
truncation, i.e. the physically-correct initial state for the test. Overlap of
the truncated state with the norb=4 anchor is reported as a state-prep check
(the retained natural-orbital occupation). `build_reduced_init` is the shared
projection helper reused by the sweep/routed-gate mean-field machinery.

The gate matrix M_ij = <anchor_i^(norb4) | psi_j(T)^(reduced)> is scored against
the TRUE norb=4 anchors (mctdh_overlap handles heterogeneous orbital counts),
so leakage out of the true computational basis is penalised in F̄.

Usage:
    python lib/spectator_norb_test.py \
        [--leg w2w3] [--gate swap] [--tau-ramp 200] [--tau-hold 182.3] \
        [--ref-pt <pt_npz>] [k ...]
    (k default: 1 2 4 ; e.g. `... 1` runs only the mean-field case)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qoct_gate_leg as _gl
from qoct_gate_leg import (
    DT_PROD,
    G_DICT_64,
    M_REFS,
    NUM_ORB,
    PAIR_LABELS,
    VMF_MAX_ITER,
    VMF_MIN_TOL,
    _make_h_s_x64,
    avg_gate_fidelity_up_to_localZ,
    gate_matrix,
    load_loss_anchors,
    mctdh_overlap,
)
from rte_step import MAX_SOLVER_TOL, NUM_DVR, NUM_NEWTON, NUM_WELLS
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_vmf_integrator
from gate_fidelity import gate_fidelity_up_to_localZ

assert NUM_WELLS == 4

# ── Leg table (pair, loss basis, resonance-anchor key) — mirrors sweep_sa.py ──
# anchor=None keeps rte_step's default x_star_2_ghz (the w0w1 resonance).
LEGS = {
    "w0w1": dict(pair=(0, 1), anchor=None,
                 configs=[(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)]),
    "w1w2": dict(pair=(1, 2), anchor="3a",
                 configs=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0)]),
    "w2w3": dict(pair=(2, 3), anchor="4a",
                 configs=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1)]),
    "w0w3": dict(pair=(0, 3), anchor=5,
                 configs=[(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0, 1)]),
}


def apply_leg(leg):
    """Retarget qoct_gate_leg's module globals to `leg` (sweep_sa.py pattern:
    the schedule builder closes over x_star_2_ghz, load_loss_anchors over
    PAIR_CONFIGS — rebind, don't re-implement). Returns (pair, spectators)."""
    cfg = LEGS[leg]
    if cfg["anchor"] is not None:
        from geometry_config import ANCHOR_PATHS
        _gl.x_star_2_ghz = jnp.asarray(
            np.load(ANCHOR_PATHS[cfg["anchor"]])["best_x"], dtype=jnp.float64)
    _gl.PAIR_CONFIGS = cfg["configs"]
    _gl.ACTIVE_PAIR = cfg["pair"]
    spectators = tuple(s for s in range(NUM_WELLS) if s not in cfg["pair"])
    return cfg["pair"], spectators


def _spectator_natural_orbitals(C_stack, weights, well):
    """SA-averaged 1-RDM of `well` (in the shared-orbital basis), returned as
    descending (occupations, eigenvectors). C_stack: (num_states, 4,4,4,4)."""
    rho = np.zeros((NUM_ORB, NUM_ORB), dtype=complex)
    for j, w in enumerate(weights):
        C = np.moveaxis(C_stack[j], well, 0)          # (norb_well, other...)
        Cf = C.reshape(NUM_ORB, -1)
        rho += w * (Cf @ Cf.conj().T)
    rho = 0.5 * (rho + rho.conj().T)
    occ, U = np.linalg.eigh(rho)
    order = np.argsort(occ)[::-1]
    return occ[order].real, U[:, order]


def build_reduced_init(C4, bs0, k, spectators):
    """Project `spectators` to k leading SA natural orbitals; return the
    reduced (C_stack, bs_reduced) + retained occupations + per-state overlaps
    with the untruncated anchors."""
    n_states = len(C4)
    weights = np.full(n_states, 1.0 / n_states)
    C_stack = np.stack([np.asarray(c) for c in C4])    # (n_states, 4,4,4,4)
    bs = [np.asarray(b) for b in bs0]
    U = {}
    retained = {}
    for s in spectators:
        occ, Us = _spectator_natural_orbitals(C_stack, weights, s)
        U[s] = Us[:, :k]                               # (norb, k)
        retained[s] = float(occ[:k].sum())
    bs_red = list(bs)
    for s in spectators:
        bs_red[s] = bs[s] @ U[s]                        # (64, k)
    # Tucker-project the CI on the spectator modes (axis = well+1 in the stack)
    Cp = C_stack
    for s in spectators:
        Cp = np.tensordot(Cp, U[s].conj(), axes=([s + 1], [0]))   # moves axis to end
        Cp = np.moveaxis(Cp, -1, s + 1)
    # renormalise each state; report overlap with the norb=4 anchor
    overlaps = []
    C_red = np.empty_like(Cp)
    for j in range(n_states):
        n2 = float(np.real(mctdh_overlap(jnp.asarray(Cp[j]),
                                         [jnp.asarray(b) for b in bs_red],
                                         jnp.asarray(Cp[j]),
                                         [jnp.asarray(b) for b in bs_red])))
        C_red[j] = Cp[j] / np.sqrt(n2)
        # overlap of truncated (renormalised) with the full norb=4 anchor
        ov = complex(mctdh_overlap(jnp.asarray(C4[j]), [jnp.asarray(b) for b in bs],
                                   jnp.asarray(C_red[j]),
                                   [jnp.asarray(b) for b in bs_red]))
        overlaps.append(abs(ov) ** 2)
    return C_red, bs_red, retained, overlaps


def run_k(k, C4, bs0, pair, spectators, gate, tau_ramp, tau_hold):
    norb = [k if s in spectators else NUM_ORB for s in range(NUM_WELLS)]
    print(f"\n=== spectator norb={k}  (active pair {pair} norb={NUM_ORB}; "
          f"norb list {norb}) ===", flush=True)
    if k == NUM_ORB:
        # no truncation: use the anchors directly (exact baseline == leg_sa)
        C_red = np.stack([np.asarray(c) for c in C4])
        bs_red = [np.asarray(b) for b in bs0]
        retained = {s: 1.0 for s in spectators}
        ov = [1.0] * len(C4)
    else:
        C_red, bs_red, retained, ov = build_reduced_init(C4, bs0, k, spectators)
    ret_str = " ".join(f"w{s}={retained[s]:.6f}" for s in spectators)
    print(f"  SA spectator NO occupation retained @k={k}: {ret_str}")
    print(f"  |<anchor4 | truncated>|^2 per state "
          f"({', '.join(PAIR_LABELS)}): {np.round(ov,6)}", flush=True)

    T_total = 2.0 * tau_ramp + tau_hold
    num_steps = int(round(T_total / DT_PROD))
    save_every = max(1, num_steps // 50)
    n_full = num_steps // save_every
    rem = num_steps - n_full * save_every
    chunk_lengths = [save_every] * n_full + ([rem] if rem else [])

    h_list = [_make_h_s_x64(s, tau_ramp, tau_hold) for s in range(NUM_WELLS)]
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64, num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=norb, num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=4, imag_time=False)
    y0 = pack_Cb_to_y(jnp.asarray(C_red, dtype=jnp.complex128),
                      [jnp.asarray(b, dtype=jnp.complex128) for b in bs_red])
    integrator = get_vmf_integrator(
        solver, DT_PROD, num_newton=NUM_NEWTON, min_solver_tol=VMF_MIN_TOL,
        max_solver_tol=MAX_SOLVER_TOL, max_solver_iter=VMF_MAX_ITER,
        precondition=True)

    def _step(carry, _):
        t, y = carry
        return integrator(t, y), None

    def _make_chunk(length):
        return jax.jit(lambda carry: jax.lax.scan(_step, carry, None,
                                                  length=length)[0])
    cache = {save_every: _make_chunk(save_every)}
    if rem:
        cache[rem] = _make_chunk(rem)

    tic = time.perf_counter()
    carry = (jnp.asarray(0.0), y0)
    for L in chunk_lengths:
        carry = cache[L](carry)
        jax.block_until_ready(carry)
    wall = time.perf_counter() - tic
    t_fin, y_fin = carry
    finite = bool(np.all(np.isfinite(np.asarray(y_fin))))

    C_T, bs_T = solver._y_to_Cb(y_fin)
    # score against the TRUE norb=4 anchors (heterogeneous overlap)
    M = gate_matrix(C4, bs0, [(C_T[j], bs_T) for j in range(4)])
    Fbar = avg_gate_fidelity_up_to_localZ(M_REFS[gate], M)
    Fxc = gate_fidelity_up_to_localZ(M_REFS[gate], M)
    sub = (np.abs(M) ** 2).sum(axis=0)
    transfer = 0.5 * (abs(M[1, 2]) ** 2 + abs(M[2, 1]) ** 2)
    np.set_printoptions(precision=4, suppress=True)
    print(f"  |M|^2:\n{np.abs(M)**2}")
    print(f"  Fbar={Fbar:.6f}  F_xc={Fxc:.6f}  transfer={transfer:.4f}  "
          f"|11>sub={sub[3]:.4f}  finite={finite}  ({wall:.0f}s)", flush=True)
    return dict(k=k, norb=norb, M=M, Fbar=Fbar, F_xc=Fxc, transfer=transfer,
                subspace=sub, retained=retained, overlaps=ov, wall=wall,
                finite=finite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ks", nargs="*", type=int, default=None)
    ap.add_argument("--leg", default="w0w1", choices=sorted(LEGS))
    ap.add_argument("--gate", default="sqiswap", choices=sorted(M_REFS))
    ap.add_argument("--tau-ramp", type=float, default=40.0)
    ap.add_argument("--tau-hold", type=float, default=107.0)
    ap.add_argument("--ref-pt", default=None,
                    help="saved norb=4 sweep pt_*.npz to gate |dFbar| < --tol against")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    ks = args.ks or [1, 2, 4]

    pair, spectators = apply_leg(args.leg)
    T_total = 2.0 * args.tau_ramp + args.tau_hold
    print(f"spectator-correlation test: {args.gate}({args.leg}), active pair "
          f"{pair}, spectators {spectators}, norb ∈ {ks}\n"
          f"  tau_ramp={args.tau_ramp:g} tau_hold={args.tau_hold:g} "
          f"T={T_total:g} dt={DT_PROD} num_steps={int(round(T_total/DT_PROD))}",
          flush=True)
    C4, bs0 = load_loss_anchors()
    results = [run_k(k, C4, bs0, pair, spectators, args.gate,
                     args.tau_ramp, args.tau_hold) for k in ks]

    # reference F̄: a saved norb=4 sweep point if given, else the largest-k run
    ref_Fbar, ref_label = None, ""
    if args.ref_pt:
        d = np.load(args.ref_pt, allow_pickle=True)
        assert (abs(float(d["tau_ramp"]) - args.tau_ramp) < 1e-6
                and abs(float(d["tau_hold"]) - args.tau_hold) < 1e-6), \
            "--ref-pt (tau_ramp, tau_hold) does not match the requested point"
        ref_Fbar = float(d["Fbar"])
        ref_label = os.path.basename(args.ref_pt)
        print(f"\n  reference norb=4 point {ref_label}: Fbar={ref_Fbar:.6f}")
    else:
        base = next((r for r in results if r["k"] == max(ks)), results[-1])
        ref_Fbar, ref_label = base["Fbar"], f"norb={base['k']} run"

    print(f"\n=== SUMMARY ({args.gate}({args.leg}) — spectator norb -> gate; "
          f"ref = {ref_label}) ===")
    print("  norb_spec   Fbar     F_xc    transfer  |11>sub   NO-occ("
          + "/".join(f"w{s}" for s in spectators) + ")")
    for r in sorted(results, key=lambda r: r["k"]):
        ret = "/".join(f"{r['retained'][s]:.5f}" for s in spectators)
        print(f"    {r['k']:>4d}    {r['Fbar']:.5f}  {r['F_xc']:.5f}  "
              f"{r['transfer']:.4f}   {r['subspace'][3]:.4f}   {ret}   "
              f"dFbar_vs_ref={r['Fbar']-ref_Fbar:+.6f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, f"spectator_norb_{args.leg}_{args.gate}_{stamp}.npz")
    np.savez(out, ks=np.array(ks), leg=args.leg, gate=args.gate,
             tau_ramp=args.tau_ramp, tau_hold=args.tau_hold,
             ref_pt=(args.ref_pt or ""), ref_Fbar=(ref_Fbar if ref_Fbar is not None else np.nan),
             Fbar=np.array([r["Fbar"] for r in results]),
             F_xc=np.array([r["F_xc"] for r in results]),
             transfer=np.array([r["transfer"] for r in results]),
             subspace=np.stack([r["subspace"] for r in results]),
             Ms=np.stack([r["M"] for r in results]),
             retained=np.array([[r["retained"][s] for s in spectators]
                                for r in results]))
    print(f"\nSaved {out}")

    # Part-0 gate: mean-field (k=1) must reproduce the reference within tol
    if args.ref_pt:
        r1 = next((r for r in results if r["k"] == 1), None)
        if r1 is not None:
            dF = abs(r1["Fbar"] - ref_Fbar)
            verdict = "PASS" if dF < args.tol else "FAIL"
            print(f"[gate] |dFbar|(mean-field vs norb=4 ref) = {dF:.2e} "
                  f"{'<' if dF < args.tol else '>='} {args.tol:g}  -> {verdict}")
            if dF >= args.tol:
                sys.exit(1)


if __name__ == "__main__":
    main()
