"""Validation checks for the ML ITE_opt. GPU, through Slurm:

    sbatch slurm/gpu.sbatch -m operating_points.validate equiv
    sbatch slurm/gpu.sbatch -m operating_points.validate traced
    sbatch slurm/gpu.sbatch -m operating_points.validate loss
    sbatch slurm/gpu.sbatch -m operating_points.validate fixedpoint --arm base
    sbatch slurm/gpu.sbatch -m operating_points.validate smoke

Each check prints its measurements, a ``GATE <name> PASS|FAIL`` line and a
``RESULT {json}`` line, and exits 1 on failure.

``equiv``
    The ML ITE_opt class against the code it was built from,
    ``system.register.relax`` (``split="strang"``, ``precision=...``,
    per-iteration Hungarian matching onto the fixed one-hot reference) with
    the same five configurations, the same float64 weights as SA and gauge
    weights, from the same start (``ckpt/iso_n6_dt0.002.npz``). norb 6,
    n_s 3. Three comparisons:
    (a) ``x64d``, fixed N: per-iteration sum E of the reference, of the
        scanned ``integrate`` and of the per-iteration-matched fallback
        path, and the final states;
    (b) ``mixed``, fixed N: the same, REPORTED only — float32 BiCGSTAB at a
        relative tolerance of 1e-2 is path dependent by design, and only its
        fixed point is pinned to float64;
    (c) ``mixed``, both run the same long fixed count (default 4000;
        a |dE| stop lands at arbitrary points of a slow mode here).
``traced``
    Two operating points on one instance: no new compile for the second,
    and its result equal to a fresh instance BUILT with the second
    ``h_list`` as its placeholder. Any read of the baked ``h`` would make
    the two differ. Plus the default NaN placeholder stays finite.
``loss``
    Stage-1 loss on ``F_matched`` equals ``L_idle`` computed by hand; the
    matched ``F`` equals ``qubit_corner_fidelities`` (one_electron/lib/utils.py,
    as ``_qubit_corner``) on the same roots; stages 2-5 finite. norb 6 / n_s 3, relaxed.
``fixedpoint``
    norb 14 / n_s 4 at the appendix idle x0 (``--arm``: base, dt1e-2,
    x64d, norb16, timing). Run after the norb-6 checks have passed.
``smoke``
    The optimisation chain at norb 6, ``--maxfun 5``.
``leakage``
    Is the hybridisation of the pair's doubly excited configuration D (e.g.
    |0110> at stage 3: fidelity ~0.896) real, or an artefact
    of relaxing only |0000> + singles (D carries ZERO weight) with 2 spare
    SPFs per well? At ``--stem``'s best point: (1) the optimisation's own
    5-state relaxation: fid(D), the roots that hold D's weight (energies,
    per-well populations above the qubit levels), and a 1000-step tail;
    (2) ``sa6``: D added to the relaxed set (weight 1/2) and relaxed to 1e-9
    plus tail; (3) ``sa6-ns6``: the same with n_s 6 (the relaxed SPFs plus two
    orthogonal extras). A measurement: PASS means every arm stayed finite
    and healthy.
``blockweight``
    The intruder rejection's threshold (``C.BLOCK_WEIGHT_MIN``) against the
    best states of a finished chain (``--base``, default ``ml_20260923``):
    each is analysed at its own x. Stages 2/3 (real A-B mixing) must pass,
    stage 4 (|0010> hybridised with a well-1 intruder) must
    be rejected, stage 5 (unmixed) must pass. norb 14 / n_s 4.
``fullset``
    Is the stage loss right? The optimisation relaxes only |0000> + singles,
    but the loss also scores the doubles, triples and |1111>
    (|0110> at 0.896 with 5 states, 0.667 once relaxed). At ``--stem``'s
    best point: the loss, its terms and every configuration's population
    from (1) the optimisation's 5-state state and (2) ``sa16``: all 16
    register configurations relaxed, weights 2^-(excitations), from the
    regauged SPFs of (1), to ``--tol`` plus ``--tail``, n_s ``--spfs``
    (6: the relaxed SPFs plus orthogonal extras). The qubit levels are the
    natural SPFs of the WEIGHTED density, so (3) re-measures the sa16 SPFs in
    the optimisation's gauge (rows |0000> + singles, weights 1, 1/2): what
    changes from (1) to (3) is the SPFs, not the definition of |0>/|1>.
    ``--from-saved`` re-analyses the saved sa16 state. Is SA16 adequate?
    ``--spfs 6``, ``--weights`` (per excitation number; the upper states
    emphasised) and ``--norb 16`` (cold: isolated wells at this x, nothing
    carried over; ``--cold`` at norb 14 is its control) must reproduce it.
    A measurement: PASS means finite and healthy.
``warmsteps``
    Before rerunning the targets with all 16 configurations relaxed: from
    the converged 16-state state at ``--stem``, an optimiser-sized step in
    x, the evaluation's fixed warm steps, then on to 1e-8. The loss error
    of the fixed steps must be small against the loss change of the step.
``seedcheck``
    ``--stem`` is a LOSS name here (stage2_superpos_w0w1 ...). The first
    evaluation of a 16-state target stage at its x0: the seeded, cold-16 and
    cold-5-then-16 candidates side by side (sum E16, detunings, health).
"""

import argparse
import json
import os
import sys
import time

from operating_points import common as C  # noqa: I100  (sets x64 first)

import jax
import jax.numpy as jnp
import numpy as np

from mctdhf.MLMCTDHF_n_species_ITE_opt import _qubit_corner
from system import fourwell as fw

REGISTER_N14 = np.array([1.758244, 2.566739, 3.027495, 4.419907])
HARTREE = np.array([1.75827, 2.5666, 3.02715, 4.41964])
L_IDLE_W = np.array([3.0] * 5 + [1.0] * 6 + [0.3] * 5)
DEFAULT_FULLSET_WEIGHTS = "1,0.5,0.25,0.125,0.0625"


def _iso_ckpt(norb):
    d = np.load(os.path.join(C.CKPT_DIR, f"iso_n{norb}_dt0.002.npz"))
    return [jnp.asarray(d[f"b{w}"]) for w in range(fw.NUM_WELLS)]


def _projector_diff(sa, sb):
    """Largest elementwise difference of the orbital projectors ``b b†`` and
    the SPF projectors ``B B†``. Both are invariant under the arbitrary
    rotations (eigh phases, the empty-level block of the regauge) that make
    the raw coefficients incomparable."""
    _Aa, Ba, ba = sa
    _Ab, Bb, bb = sb
    db = max(float(jnp.max(jnp.abs(x @ x.conj().T - y @ y.conj().T)))
             for x, y in zip(ba, bb))
    dB = max(float(jnp.max(jnp.abs(x @ x.conj().T - y @ y.conj().T)))
             for x, y in zip(Ba, Bb))
    return db, dB


def _finish(name, ok, result):
    result = dict(result, gate=name, passed=bool(ok),
                  device=jax.default_backend())
    print(f"\nGATE {name} {'PASS' if ok else 'FAIL'}")
    print("RESULT " + json.dumps(result, default=float), flush=True)
    return 0 if ok else 1


# ── equiv ─────────────────────────────────────────────────────────────────


def _reference(norb, n_spf, solver, h, state0, steps, precision, tol):
    """``register.relax`` on an ML solver built with the REAL ``h``."""
    from system import register as reg
    from system.prep import make_solver

    ml = make_solver(norb, n_spf, solver.num_states, True, list(h),
                     state_weights=solver.relax_weights)
    ml.state_weights = jnp.asarray(solver.relax_weights)   # float64, as new
    A0, Bs0, bs0 = state0
    t0 = time.perf_counter()
    A, Bs, bs, _ev, _rk, hist = reg.relax(
        ml, A0, list(Bs0), list(bs0), steps, solver.dt,
        solver.relax_configs, verbose=False, B_propagator="cn",
        b_propagator="cn", gauge_w=solver.relax_weights, match="fixed",
        regauge=True, tol=tol, split="strang", precision=precision,
        mixed_tol=solver.inner_tol, mixed_maxiter=solver.inner_maxiter)
    wall = time.perf_counter() - t0
    return (A, Bs, bs), np.array([r["e_sum"] for r in hist]), wall


def gate_equiv(args):
    norb, n_spf, N = 6, 3, args.steps
    h = fw.h_static(fw.X_IDLE)
    bs_iso = _iso_ckpt(norb)
    res, ok = {}, True

    # (a) pins the PATH, so the inner solves must be converged: at the
    # production inner tol 1e-2 BiCGSTAB's stopping iteration can flip on
    # rounding and move an increment by up to 1 %, in float64 as much as in
    # float32 (measured: 4e-5 in sum E5 between two correct paths). At 1e-12
    # the only freedom left is the code. (b) keeps the production setting.
    for prec, itol, imax in (("x64d", 1e-12, 2000), ("mixed", 1e-2, 200)):
        solver = C.make_solver(norb, n_spf, precision=prec, inner_tol=itol,
                               inner_maxiter=imax)
        state0 = solver.default_state(bs_iso, h)
        s_ref, e_ref, w_ref = _reference(norb, n_spf, solver, h, state0, N,
                                         prec, 0.0)
        t0 = time.perf_counter()
        s_new, e_new, info = solver.integrate(N, h, state0)
        w_new = time.perf_counter() - t0
        t0 = time.perf_counter()
        s_mat, e_mat, _r = solver._matched_steps(state0, solver.as_h(h), N)
        w_mat = time.perf_counter() - t0
        d_new = float(np.max(np.abs(e_new - e_ref)))
        d_mat = float(np.max(np.abs(e_mat - e_ref)))
        pb_new, pB_new = _projector_diff(s_new, s_ref)
        pb_mat, pB_mat = _projector_diff(s_mat, s_ref)
        a_ref = solver.analyse(s_ref, h)
        a_new = solver.analyse(s_new, h)
        dE16 = float(np.max(np.abs(a_new["E_matched"] - a_ref["E_matched"])))
        first = [int(np.argmax(np.abs(e - e_ref) > 1e-9))
                 if np.any(np.abs(e - e_ref) > 1e-9) else None
                 for e in (e_new, e_mat)]
        print(f"\n[{prec}, inner tol {itol:g}] fixed N = {N} from iso_n6 "
              f"(norb 6, n_s 3)")
        print(f"  first iteration with |dE5| > 1e-9: scan {first[0]}  "
              f"matched {first[1]}")
        print(f"  sum E5 at it 0 / {N-1}: ref {e_ref[0]:.10f} / "
              f"{e_ref[-1]:.10f}")
        print(f"  max |dE5| per iteration: scan {d_new:.3e}   "
              f"per-iteration-matched {d_mat:.3e}")
        print(f"  final projectors  scan  b {pb_new:.2e}  B {pB_new:.2e}")
        print(f"                    match b {pb_mat:.2e}  B {pB_mat:.2e}")
        print(f"  16 matched energies, max diff {dE16:.3e}")
        print(f"  fallback chunks {info['fallback_chunks']}   ranks "
              f"{info['ranks']}")
        print(f"  wall  ref {w_ref:.1f} s   scan {w_new:.1f} s   "
              f"matched {w_mat:.1f} s  (all include compile)")
        res[prec] = dict(d_scan=d_new, d_matched=d_mat, proj_b=pb_new,
                         proj_B=pB_new, proj_b_matched=pb_mat,
                         proj_B_matched=pB_mat, dE16=dE16,
                         fallback=info["fallback_chunks"],
                         wall_ref=w_ref, wall_scan=w_new)
        if prec == "x64d":
            ok_a = (d_new < 1e-9 and d_mat < 1e-9 and
                    max(pb_new, pB_new, pb_mat, pB_mat) < 1e-8)
            print(f"  (a) x64d fixed-N agreement: "
                  f"{'PASS' if ok_a else 'FAIL'}  (tol 1e-9 in sum E, "
                  f"1e-8 in projectors)")
            ok &= ok_a

    # (c) the long-run state, mixed. NOT stopped on |dE|: at norb 6 / n_s 3
    # there is a slow mode (well 0, whose level-2 SPF holds 2e-5) along which
    # sum E5 keeps moving by ~3e-10 per iteration, so a |dE| < 1e-9 or 1e-10
    # stop fires at an arbitrary point of it (two pilot pairs: the two codes
    # stopped at 1394 / 1400 and at 1435 / 2050 iterations, 2e-7 apart in
    # sum E5). Both run the SAME fixed count instead, and the trajectory is
    # printed so the tail itself is measured.
    Nc = args.relax_steps
    solver = C.make_solver(norb, n_spf, precision="mixed")
    state0 = solver.default_state(bs_iso, h)
    s_ref, e_ref, w_ref = _reference(norb, n_spf, solver, h, state0, Nc,
                                     "mixed", 0.0)
    t0 = time.perf_counter()
    s_new, e_new, info = solver.integrate(Nc, h, state0)
    w_new = time.perf_counter() - t0
    a_ref, a_new = solver.analyse(s_ref, h), solver.analyse(s_new, h)
    dE5 = abs(a_new["e_relax"] - a_ref["e_relax"])
    dd = float(np.max(np.abs(a_new["detunings"] - a_ref["detunings"])))
    pb, pB = _projector_diff(s_new, s_ref)
    print(f"\n[mixed] both run {Nc} iterations (no |dE| stop)")
    print("  iteration   sum E5 ref           sum E5 new           "
          "diff        ref |dE| per it")
    for k in sorted({500, 1000, 1500, 2000, 3000, Nc} & set(range(1, Nc + 1))):
        print(f"  {k:9d}   {e_ref[k-1]:.10f}   {e_new[k-1]:.10f}   "
              f"{e_new[k-1] - e_ref[k-1]:+.2e}   "
              f"{abs(e_ref[k-1] - e_ref[k-2]):.1e}")
    print(f"  final (analyse) sum E5  ref {a_ref['e_relax']:.10f}   new "
          f"{a_new['e_relax']:.10f}   |diff| {dE5:.3e}")
    print("  detunings   ref " + " ".join(f"{v:.8f}" for v in
                                         a_ref["detunings"]))
    print("              new " + " ".join(f"{v:.8f}" for v in
                                         a_new["detunings"]))
    print(f"  max |d det| {dd:.3e}   projectors b {pb:.2e}  B {pB:.2e}"
          f"   fallback chunks {info['fallback_chunks']}")
    print(f"  healthy     ref {a_ref['healthy']}  new {a_new['healthy']}")
    print("  occupations (weights 1, 1/2 x4; ideal 0.8333 0.1667 0 ...)")
    for s, o in enumerate(a_new["occupations"]):
        print(f"    well {s}: " + " ".join(f"{v:.6f}" for v in o))
    print(f"  wall  ref {w_ref:.1f} s  new {w_new:.1f} s")
    ok_c = dE5 < 1e-8 and dd < 1e-7 and a_new["healthy"]
    print(f"  (c) long-run states: {'PASS' if ok_c else 'FAIL'}  "
          f"(tol 1e-8 in sum E5, 1e-7 in detunings)")
    ok &= ok_c
    res["relaxed"] = dict(dE5=dE5, d_det=dd, proj_b=pb, proj_B=pB,
                          steps=Nc, e5=a_new["e_relax"],
                          e5_traj_ref=[float(e_ref[k - 1]) for k in
                                       (500, 1000, 2000, Nc)],
                          e5_traj_new=[float(e_new[k - 1]) for k in
                                       (500, 1000, 2000, Nc)],
                          detunings=list(a_new["detunings"]),
                          healthy=a_new["healthy"],
                          wall_ref=w_ref, wall_new=w_new)
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    C.save_state(os.path.join(C.CACHE_DIR, "equiv_n6_s3_mixed.npz"), s_new,
                 x_ghz=fw.X_IDLE * fw.TYPICAL_FREQ_GHZ,
                 e_relax=a_new["e_relax"])
    return _finish("equiv", ok, res)


# ── traced ────────────────────────────────────────────────────────────────


def _traced_arm(prec, itol, imax, N, h1, h2, bs_iso):
    """One instance at x1 then x2, a fresh instance built at x2, a NaN-
    placeholder instance at x2, and the first instance at x2 AGAIN (the
    run-to-run floor)."""
    kw = dict(precision=prec, inner_tol=itol, inner_maxiter=imax)
    A = C.make_solver(6, 3, h_placeholder=h1, **kw)
    state0 = A.default_state(bs_iso, h1)
    t0 = time.perf_counter()
    _s1, e1, _i1 = A.integrate(N, h1, state0)
    t1 = time.perf_counter() - t0
    c1 = A.compile_counts()
    t0 = time.perf_counter()
    s2, e2, _i2 = A.integrate(N, h2, state0)
    t2 = time.perf_counter() - t0
    c2 = A.compile_counts()
    _s2b, e2b, _ = A.integrate(N, h2, state0)
    B = C.make_solver(6, 3, h_placeholder=h2, **kw)
    sB, eB, _iB = B.integrate(N, h2, state0)
    Cn = C.make_solver(6, 3, **kw)                   # NaN placeholder
    _sC, eC, _iC = Cn.integrate(N, h2, state0)
    r = dict(counts_x1=c1, counts_x2=c2, wall_x1=t1, wall_x2=t2,
             d_AB=float(np.max(np.abs(e2 - eB))),
             d_state=max(float(jnp.max(jnp.abs(a - b)))
                         for a, b in zip(jax.tree_util.tree_leaves(s2),
                                         jax.tree_util.tree_leaves(sB))),
             d_nan=float(np.max(np.abs(eC - eB))),
             nan_finite=bool(np.all(np.isfinite(eC))),
             d_repeat=float(np.max(np.abs(e2b - e2))),
             moved=float(np.max(np.abs(e2 - e1))))
    print(f"\n[{prec}, inner tol {itol:g}] {N} iterations per run")
    print(f"  compiled variants after x1 {c1}\n                    after x2 {c2}")
    print(f"  wall x1 {t1:.2f} s (with compile)   x2 {t2:.2f} s")
    print(f"  x1 -> x2 moves sum E5 by up to {r['moved']:.3e}")
    print(f"  built-at-x1 run at x2 vs built-at-x2:  |dE5| {r['d_AB']:.3e}  "
          f"|d state| {r['d_state']:.3e}")
    print(f"  NaN placeholder at x2: finite {r['nan_finite']}, |dE5| "
          f"{r['d_nan']:.3e}")
    print(f"  same instance at x2 twice (run-to-run floor): "
          f"|dE5| {r['d_repeat']:.3e}")
    return r


def gate_traced(args):
    """Gated at a PINNED path (x64d, inner tol 1e-12): at the production
    inner tol 1e-2 two runs of the same program already differ by ~1e-4
    (measured), so equality there cannot distinguish a baked-h
    read from BiCGSTAB's path dependence. The production setting is run
    too, for the compile count, the timing and the run-to-run floor."""
    N = args.steps
    x2 = fw.X_IDLE.copy()
    x2[0] *= 1.01
    h1, h2 = fw.h_static(fw.X_IDLE), fw.h_static(x2)
    bs_iso = _iso_ckpt(6)
    pin = _traced_arm("x64d", 1e-12, 2000, N, h1, h2, bs_iso)
    prod = _traced_arm("mixed", 1e-2, 200, N, h1, h2, bs_iso)
    ok = all(r["counts_x1"] == r["counts_x2"] and r["nan_finite"]
             and r["moved"] > 1e-3 for r in (pin, prod))
    # Wall times are reported, not gated: at x64d / 1e-12 the solves dwarf
    # the compile (30 s vs 21 s), so a speed ratio is no proxy for
    # a recompile. The compile counts are the test.
    ok &= pin["d_AB"] < 1e-9 and pin["d_nan"] < 1e-9
    return _finish("traced", ok, dict(pinned=pin, production=prod))


# ── loss ──────────────────────────────────────────────────────────────────


def gate_loss(args):
    norb, n_spf = 6, 3
    h = fw.h_static(fw.X_IDLE)
    solver = C.make_solver(norb, n_spf)
    state0 = solver.default_state(_iso_ckpt(norb), h)
    state, e, info = solver.relax_to_tol(h, state0, tol=args.tol)
    an = solver.analyse(state, h, info)
    ok = True

    # configuration order: the one-electron losses index positions
    print("configs (one_electron utils.iter_bitstrings order): "
          + " ".join("".join(map(str, c)) for c in C.CONFIGS))
    from system.register import iter_bitstrings as reg_order
    ro = reg_order(4, 2)
    print(f"  system.register.iter_bitstrings agrees: {ro == C.CONFIGS}"
          f"  (its singles: {' '.join(''.join(map(str, c)) for c in ro[1:5])})")
    ok_order = (C.CONFIGS[1] == (1, 0, 0, 0) and C.CONFIGS[4] == (0, 0, 0, 1)
                and C.CONFIGS[5] == (1, 1, 0, 0)
                and solver.relax_configs == C.CONFIGS[:5])
    ok &= ok_order

    # F_matched against qubit_corner_fidelities on the same roots
    _A, _B, evals, evecs, _occ = solver._analyse_dev(*state, solver.as_h(h))
    A_all = evecs.T.reshape((solver.dim,) + solver.num_spfs)
    F_ref = np.asarray(_qubit_corner(A_all, 4, 2))
    dF = float(np.max(np.abs(F_ref - an["F_energy"])))
    dF_m = float(np.max(np.abs(F_ref[an["ranks_all"]] - an["F_matched"])))
    print(f"  F_energy vs qubit_corner_fidelities: max diff {dF:.2e}; "
          f"F_matched rows: {dF_m:.2e}")
    # The eigenvectors here come from a SECOND call of the same jitted
    # function; on the GPU two calls differ at ~1e-11 (measured).
    # A wrong column order or definition would differ at O(1).
    ok &= dF < 1e-9 and dF_m < 1e-9

    ok_ranks = np.array_equal(an["ranks_relax"], an["ranks_all"][:5])
    print(f"  5-config match == first 5 of the 16-config match: {ok_ranks}")
    ok &= ok_ranks

    # stage 1 == L_idle by hand
    spec = C.load_loss("stage1_singles_ground")
    A, Bs, bs = state
    r = spec["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_matched"],
                        energies=an["E_matched"], num_wells=4)
    P = np.diag(an["F_matched"])
    L_hand = float(np.sum(L_IDLE_W * (1.0 - P)))
    dL = abs(r["loss"] - L_hand)
    diag_ok = np.array_equal(r["matched_idx"], np.arange(16))
    print(f"\n  stage1 loss on F_matched {r['loss']:.12f}   L_idle by hand "
          f"{L_hand:.12f}   |diff| {dL:.2e}   matched_idx diagonal {diag_ok}")
    ok &= dL < 1e-12 and diag_ok
    r_en = spec["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_energy"],
                           energies=an["E_all"], num_wells=4)
    print(f"  stage1 loss on the ENERGY-ordered F ({an['F_energy'].shape}): "
          f"{r_en['loss']:.6f}  (logged, not optimised)")

    print("\n  config  rank   E              P(config)")
    for i, c in enumerate(C.CONFIGS):
        print(f"   |{''.join(map(str, c))}>  {an['ranks_all'][i]:4d}   "
              f"{an['E_matched'][i]:.8f}   {P[i]:.6f}")
    print("  detunings " + " ".join(f"{v:.8f}" for v in an["detunings"])
          + f"   additivity {an['additivity']:.3e}   healthy {an['healthy']}")
    print("  leakage (weighted node density beyond level 1) "
          + " ".join(f"{v:.2e}" for v in an["leakage"]))
    ok &= bool(an["healthy"])

    stages = {}
    for name in ("stage2_superpos_w0w1", "stage3_superpos_w1w2",
                 "stage4_superpos_w2w3", "stage5_superpos_w0w3"):
        sp = C.load_loss(name)
        rm = sp["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_matched"],
                           energies=an["E_matched"], num_wells=4)
        re = sp["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_energy"],
                           energies=an["E_all"], num_wells=4)
        fin = bool(np.isfinite(rm["loss"]) and np.isfinite(re["loss"]))
        print(f"  {name}: matched {rm['loss']:.6f}   energy-ordered "
              f"{re['loss']:.6f}   finite {fin}")
        stages[name] = dict(matched=rm["loss"], energy=re["loss"])
        ok &= fin
    print(f"\n  loss sources: {C.load_loss('stage5_superpos_w0w3')['sha256']}")
    return _finish("loss", ok, dict(
        stage1=r["loss"], L_hand=L_hand, dL=dL, dF=dF, stages=stages,
        e5=an["e_relax"], detunings=list(an["detunings"]),
        ranks=list(map(int, an["ranks_all"])), register_order_agrees=(
            ro == C.CONFIGS), steps=info["steps"]))


# ── fixedpoint (norb 14) ──────────────────────────────────────────────────


def gate_fixedpoint(args):
    x0, x0_src = C.load_x0("stage1_singles_ground")
    arm = args.arm
    norb, dt, prec = 14, 2e-2, "mixed"
    if arm == "dt1e-2":
        dt = 1e-2
    elif arm == "x64d":
        prec = "x64d"
    elif arm == "norb16":
        norb = 16
    solver = C.make_solver(norb, 4, dt=dt, precision=prec)
    h = C.h_of(x0)
    tag = f"fp_n{norb}_s4_{prec}_dt{dt:g}"
    path = os.path.join(C.CACHE_DIR, tag + ".npz")

    if arm == "timing":
        base = os.path.join(C.CACHE_DIR, "fp_n14_s4_mixed_dt0.02.npz")
        state = C.load_state(base)
        spec = C.load_loss("stage1_singles_ground")
        prev = dict(state=state, detunings=solver.analyse(state, h)[
            "detunings"])
        out = {}
        # "+1% again" repeats the first evaluation from the same warm state:
        # at inner tol 1e-2 the GPU path is not reproducible run to run
        # (norb 6: ~4e-4 in sum E5 mid-path, gone by convergence), so this measures how much of that reaches the LOSS.
        for lbl, f in (("+1%", 1.01), ("-1%", 0.99), ("+1% again", 1.01)):
            x = x0.copy()
            x[0] *= f
            t0 = time.perf_counter()
            r = C.evaluate_params(solver, x, spec, prev, norb)
            out[lbl] = dict(wall=time.perf_counter() - t0, loss=r["loss"],
                            e5=r["e_relax"], cold=r["cold"],
                            guard=r["guard"], dE=r["dE_last_chunk"],
                            detunings=list(r["detunings"]))
            print(f"  warm {lbl} alpha_0: {out[lbl]}", flush=True)
        x = x0.copy()
        x[0] *= 1.01
        t0 = time.perf_counter()
        sc, _e, ic = C.cold_state(solver, x, norb)
        ac = solver.analyse(sc, C.h_of(x))
        out["cold+1%"] = dict(wall=time.perf_counter() - t0,
                              e5=ac["e_relax"], steps=ic["steps"],
                              detunings=list(ac["detunings"]))
        print(f"  cold +1% alpha_0: {out['cold+1%']}")
        print(f"  repeat of the same warm evaluation: |d loss| "
              f"{abs(out['+1% again']['loss'] - out['+1%']['loss']):.3e}  "
              f"|d sum E5| {abs(out['+1% again']['e5'] - out['+1%']['e5']):.3e}")
        print(f"  warm - cold sum E5 at +1%: "
              f"{out['+1%']['e5'] - ac['e_relax']:+.3e}")
        return _finish("fixedpoint-timing", True, out)

    t0 = time.perf_counter()
    state, e, info = C.cold_state(solver, x0, norb)
    wall = time.perf_counter() - t0
    an_stop = solver.analyse(state, h, info)
    # A |dE| < 1e-9 stop can sit on a slow mode (norb 6: sum E5 still moved
    # 2e-7 afterwards). A fixed tail measures how far this arm
    # still is from its fixed point, i.e. its own error bar for the cross-arm
    # comparison; the tail state is what is compared and saved.
    t0 = time.perf_counter()
    state, e_tail, info_t = solver.integrate(args.tail, h, state)
    wall_tail = time.perf_counter() - t0
    an = solver.analyse(state, h, info_t)
    tail_dE5 = an["e_relax"] - an_stop["e_relax"]
    tail_ddet = an["detunings"] - an_stop["detunings"]
    C.save_state(path, state, x_ghz=x0, e_relax=an["e_relax"])
    print(f"arm {arm}: norb {norb} n_s 4 {prec} dt {dt:g}, x0 from {x0_src}")
    print(f"  {info['steps']} iterations, converged {info['converged']}, "
          f"fallback chunks {info['fallback_chunks']}, {wall:.0f} s")
    print(f"  tail of {args.tail} iterations ({wall_tail:.0f} s): sum E5 moved "
          f"{tail_dE5:+.3e}, detunings moved "
          + " ".join(f"{v:+.1e}" for v in tail_ddet)
          + f", last |dE| {abs(e_tail[-1] - e_tail[-2]):.1e}, fallback "
          f"chunks {info_t['fallback_chunks']}")
    print(f"  sum E5 {an['e_relax']:.10f}  (after the tail)")
    print("  detunings " + " ".join(f"{v:.6f}" for v in an["detunings"]))
    print("  register  " + " ".join(f"{v:.6f}" for v in REGISTER_N14))
    print("  Hartree   " + " ".join(f"{v:.6f}" for v in HARTREE))
    print(f"  healthy {an['healthy']}  additivity {an['additivity']:.3e}")
    for s, o in enumerate(an["occupations"]):
        print(f"    well {s}: " + " ".join(f"{v:.6f}" for v in o[:4]))
    print(f"  ranks {list(an['ranks_all'])}")
    for other in sorted(os.listdir(C.CACHE_DIR)):
        if other.startswith("fp_n") and other != tag + ".npz":
            d = np.load(os.path.join(C.CACHE_DIR, other))
            print(f"  vs {other}: d sum E5 = "
                  f"{an['e_relax'] - float(d['e_relax']):+.3e}")
    return _finish(f"fixedpoint-{arm}", bool(an["healthy"]), dict(
        e5=an["e_relax"], detunings=list(an["detunings"]),
        e5_at_stop=an_stop["e_relax"], tail_dE5=tail_dE5,
        tail_ddet=list(tail_ddet), tail=args.tail,
        steps=info["steps"], wall=wall, healthy=an["healthy"],
        occupations=an["occupations"][:, :4].tolist()))


# ── smoke ─────────────────────────────────────────────────────────────────


def gate_smoke(args):
    from operating_points import optimize_idle, optimize_target
    rc = optimize_idle.main(["smoke", "--norb", "6", "--spfs", "3",
                             "--maxfun", "5"])
    idle = os.path.join(C.RESULTS_DIR, "smoke_stage1_singles_ground.npz")
    rc |= optimize_target.main(["smoke", "stage2_superpos_w0w1", idle,
                                "--norb", "6", "--spfs", "3",
                                "--maxfun", "5"])
    return _finish("smoke", rc == 0, {})


def _root_profile(solver, state, h, idx_cfg, top=3):
    """The roots holding configuration ``all_configs[idx_cfg]``: for the
    ``top`` roots with the largest population on it, their energy above the
    matched ground root, that population, the qubit-corner weight, and each
    well's population above level 1."""
    an = solver.analyse(state, h)
    _A, _B, evals, evecs, _occ = solver._analyse_dev(*state, solver.as_h(h))
    evals = np.asarray(evals)
    V = np.abs(np.asarray(evecs)) ** 2              # (dim, n_roots)
    F_col = an["F_energy"][:, idx_cfg]
    e0 = an["E_matched"][0]
    rows = []
    for r in np.argsort(F_col)[::-1][:top]:
        P = V[:, r].reshape(solver.num_spfs)
        above = [float(P.sum(axis=tuple(a for a in range(P.ndim) if a != w))
                       [2:].sum()) for w in range(P.ndim)]
        rows.append(dict(rank=int(r), E=float(evals[r] - e0),
                         pop=float(F_col[r]),
                         qubit=float(an["F_energy"][r].sum()),
                         above_qubit=above))
    return an, rows


def _print_profile(label, an, rows, idx_cfg):
    cfg = "".join(map(str, C.CONFIGS[idx_cfg]))
    print(f"  [{label}] fid(|{cfg}>) = {an['F_matched'][idx_cfg, idx_cfg]:.5f}"
          f"  E(|{cfg}>) - E0 = {an['E_matched'][idx_cfg] - an['E_matched'][0]:.6f}"
          f"  detunings {np.round(an['detunings'], 6)}  healthy "
          f"{an['healthy']}  sum E_relax {an['e_relax']:.8f}")
    for r in rows:
        print(f"      root {r['rank']:4d}  E - E0 = {r['E']:.6f}  pop on "
              f"|{cfg}> {r['pop']:.5f}  qubit-corner weight {r['qubit']:.5f}"
              f"  per-well above level 1 "
              + " ".join(f"{v:.4f}" for v in r["above_qubit"]), flush=True)


def gate_leakage(args):
    stem = args.stem
    path = os.path.join(C.RESULTS_DIR, stem + "_best_state.npz")
    loss_name = stem.split("_target_")[1]
    with np.load(path) as d:
        x = np.asarray(d["x_ghz"], dtype=float)
    h = C.h_of(x)
    a, b = C.superpos_block(loss_name)
    wa, wb = a - 1, b - 1
    D = tuple(1 if w in (wa, wb) else 0 for w in range(C.NUM_WELLS))
    iD = C.CONFIGS.index(D)
    print(f"leakage at {stem} (x = {np.round(x, 6)}), D = "
          f"|{''.join(map(str, D))}>", flush=True)
    ok, out = True, {}

    # (1) the optimisation's own relaxation, then a fixed tail
    solver = C.make_solver(14, 4)
    state = C.load_state(path)
    an, rows = _root_profile(solver, state, h, iD)
    _print_profile("base, as optimised", an, rows, iD)
    out["base"] = dict(fid=float(an["F_matched"][iD, iD]), roots=rows)
    t0 = time.perf_counter()
    state_t, _e, _i = solver.integrate(args.tail, h, state)
    an_t, rows_t = _root_profile(solver, state_t, h, iD)
    _print_profile(f"base + {args.tail} steps ({time.perf_counter() - t0:.0f}"
                   " s)", an_t, rows_t, iD)
    out["base_tail"] = dict(fid=float(an_t["F_matched"][iD, iD]),
                            roots=rows_t)
    ok &= bool(an_t["healthy"])
    A_rg, Bs_rg, bs = an_t["state_regauged"]

    relax = list(C.CONFIGS[:5]) + [D]
    weights = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    for label, n_spf in (("sa6", 4), ("sa6-ns6", 6)):
        sv = C.make_solver(14, n_spf, relax_configs=relax,
                           relax_weights=weights)
        if n_spf == 4:
            Bs0 = Bs_rg
        else:
            # the relaxed SPFs, completed by the lowest bare SPFs at these
            # orbitals, orthonormalised (QR keeps the span of the first 4)
            _A, B_def, _b = sv.default_state(bs, h)
            Bs0 = [jnp.linalg.qr(jnp.concatenate([Bs_rg[s], B_def[s]],
                                                 axis=1))[0][:, :n_spf]
                   for s in range(C.NUM_WELLS)]
        A0 = sv.default_state(bs, h)[0]
        t0 = time.perf_counter()
        st, _e, info = sv.relax_to_tol(h, (A0, Bs0, bs), tol=1e-9)
        st, _e, _i = sv.integrate(args.tail, h, st)
        wall = time.perf_counter() - t0
        an_s, rows_s = _root_profile(sv, st, h, iD)
        print(f"  {label}: {info['steps']} it + {args.tail} tail, converged "
              f"{info['converged']}, fallback chunks "
              f"{len(info['fallback_chunks'])}, {wall:.0f} s; occupations "
              + "; ".join(np.array2string(o[:4], precision=4)
                          for o in an_s["occupations"]), flush=True)
        _print_profile(label, an_s, rows_s, iD)
        ok &= bool(an_s["healthy"]) and bool(np.isfinite(an_s["e_relax"]))
        out[label] = dict(fid=float(an_s["F_matched"][iD, iD]), roots=rows_s,
                          steps=info["steps"], converged=info["converged"],
                          detunings=list(an_s["detunings"]), wall=wall)
    return _finish("leakage", ok, dict(stem=stem, D="".join(map(str, D)),
                                      **out))


def _loss_terms(loss_name, r, F):
    """The loss split into the part on |0000> + singles, on the doubles and
    on the triples + |1111>, from the ``fidelities`` the loss returns."""
    f = np.asarray(r["fidelities"], dtype=float)
    if loss_name.startswith("stage1"):
        w = L_IDLE_W
        return dict(low=float(np.sum(w[:5] * (1 - f[:5]))),
                    doubles=float(np.sum(w[5:11] * (1 - f[5:11]))),
                    triples=float(np.sum(w[11:] * (1 - f[11:]))))
    C._one_e_losses()
    from losses import _superpos as S
    a, b = C.superpos_block(loss_name)
    W = S.WEIGHTS_BY_CFG_IDX
    pinned = [i for i in range(5) if i not in (a, b)]
    sup = S.W_SUPER * float(sum((F[i, j] - 0.5) ** 2
                                for i in (a, b) for j in (a, b)))
    return dict(super=sup,
                pinned=float(np.sum(W[pinned] * (1 - f[pinned]))),
                doubles=float(np.sum(W[5:11] * (1 - f[5:11]))),
                triples=float(np.sum(W[11:] * (1 - f[11:]))))


def _full_report(label, loss_name, spec, sv, st, h):
    an = sv.analyse(st, h)
    A, Bs, bs = st
    r = spec["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_matched"],
                        energies=an["E_matched"], num_wells=4)
    terms = _loss_terms(loss_name, r, an["F_matched"])
    P = np.diag(an["F_matched"])
    e0 = an["E_matched"][0]
    print(f"  [{label}] loss {r['loss']:.6f}  terms "
          + " ".join(f"{k} {v:.6f}" for k, v in terms.items())
          + f"  healthy {an['healthy']}", flush=True)
    print("     config  E - E0       P(own)   matched row's pop per config "
          "> 0.02")
    for i, c in enumerate(C.CONFIGS):
        big = [f"|{''.join(map(str, C.CONFIGS[j]))}> {an['F_matched'][i, j]:.4f}"
               for j in np.argsort(an["F_matched"][i])[::-1]
               if j != i and an["F_matched"][i, j] > 0.02]
        print(f"     |{''.join(map(str, c))}>  {an['E_matched'][i] - e0:.6f}  "
              f"{P[i]:.5f}  qubit-corner {an['F_matched'][i].sum():.5f}  "
              + "  ".join(big))
    print("     detunings " + " ".join(f"{v:.6f}" for v in an["detunings"])
          + "   level>=2 occupation " + " ".join(f"{v:.2e}"
                                                for v in an["leakage"]),
          flush=True)
    return an, dict(loss=float(r["loss"]), terms=terms, P=P.tolist(),
                    E=(an["E_matched"] - e0).tolist(),
                    detunings=list(an["detunings"]),
                    healthy=bool(an["healthy"]),
                    leakage=list(map(float, an["leakage"])))


def gate_fullset(args):
    """The stage loss from a relaxation of ALL 16 register configurations
    (weights 1, 1/2, 1/4, 1/8, 1/16 by excitation number) against the
    optimisation's 5-state relaxation, at ``--stem``'s best point."""
    stem = args.stem
    path = os.path.join(C.RESULTS_DIR, stem + "_best_state.npz")
    loss_name = stem.split("_target_")[1] if "_target_" in stem \
        else stem.split("_idle_")[1]
    with np.load(path) as d:
        x = np.asarray(d["x_ghz"], dtype=float)
    h = C.h_of(x)
    spec = C.load_loss(loss_name)
    print(f"fullset at {stem} ({loss_name}), x = {np.round(x, 8)}, "
          f"n_s {args.spfs}", flush=True)

    solver = C.make_solver(14, 4)
    state = C.load_state(path)
    an_b, out_b = _full_report("5-state, as optimised", loss_name, spec,
                               solver, state, h)
    ok = out_b["healthy"]
    A_rg, Bs_rg, bs = an_b["state_regauged"]

    w_exc = [float(v) for v in args.weights.split(",")]
    weights = [w_exc[sum(c)] for c in C.CONFIGS]
    norb = args.norb
    sv = C.make_solver(norb, args.spfs, relax_configs=list(C.CONFIGS),
                       relax_weights=weights)
    cold = args.cold or norb != 14
    if cold:
        # isolated wells at this x, bare SPFs, one-hot A: nothing carried
        # over from the optimisation (a norb-16 state cannot be)
        bs_iso, _p = C.isolated_orbitals(x, norb)
        _A, Bs0, bs = sv.default_state(bs_iso, h)
    elif args.spfs == 4:
        Bs0 = Bs_rg
    else:
        _A, B_def, _b = sv.default_state(bs, h)
        Bs0 = [jnp.linalg.qr(jnp.concatenate([Bs_rg[s], B_def[s]],
                                             axis=1))[0][:, :args.spfs]
               for s in range(C.NUM_WELLS)]
    tag = f"ns{args.spfs}"
    if norb != 14:
        tag += f"_n{norb}"
    if cold:
        tag += "_cold"
    if args.weights != DEFAULT_FULLSET_WEIGHTS:
        tag += "_w" + args.weights.replace(",", "-")
    print(f"  sa16 arm {tag}: norb {norb}, weights by excitation {w_exc}",
          flush=True)
    save_path = os.path.join(C.RESULTS_DIR, f"{stem}_sa16_{tag}_state.npz")
    out = {}
    if args.from_saved:
        st = C.load_state(save_path)
        print(f"  sa16 state from {save_path}", flush=True)
        an_t, out_t = _full_report("sa16 (saved)", loss_name, spec, sv, st,
                                   h)
        info = dict(steps=None, converged=None)
        wall = None
    else:
        A0 = sv.default_state(bs, h)[0]
        t0 = time.perf_counter()
        # Caps from the measured runs: 16 states n_s 4 needed 1050 (idle)
        # and 19 700 (stage 3, ~0.08 s/it); 6 states n_s 6 needed 2400 at
        # stage 3 (~0.4 s/it). At the idle point n_s 6 never met the 1e-9
        # stop (cancelled after 2 h 51 min: two near-empty SPFs).
        max_steps = args.max_steps or (25000 if args.spfs <= 4 else 6000)
        print(f"  sa16: cap {max_steps} iterations; progress every "
              f"{sv.chunk}", flush=True)
        st, _e, info = sv.relax_to_tol(h, (A0, Bs0, bs), tol=args.tol,
                                       max_steps=max_steps, verbose=True)
        t1 = time.perf_counter()
        wall = t1 - t0
        print(f"  sa16: {info['steps']} it, converged {info['converged']}, "
              f"fallback chunks {len(info['fallback_chunks'])}, "
              f"{wall:.0f} s", flush=True)
        _an_s, out["sa16"] = _full_report("sa16", loss_name, spec, sv, st, h)
        st, _e, _i = sv.integrate(args.tail, h, st)
        an_t, out_t = _full_report(f"sa16 + {args.tail} steps "
                                   f"({time.perf_counter() - t1:.0f} s)",
                                   loss_name, spec, sv, st, h)
        C.save_state(save_path, st, x_ghz=x, loss=out_t["loss"],
                     relax_weights=np.asarray(weights))
        print(f"  saved {save_path}", flush=True)
    ok &= out_t["healthy"] and bool(np.isfinite(out_t["loss"]))
    out["sa16_tail"] = out_t
    # The qubit levels are the natural SPFs of the WEIGHTED density, so the
    # 2^-n weights define a slightly different |0>/|1> per well than the
    # optimisation's 1, 1/2. Same SPFs, measured in the optimisation's gauge:
    # the rows of |0000> + singles regauged with its weights.
    sv5 = solver if (args.spfs, norb) == (4, 14) else \
        C.make_solver(norb, args.spfs)
    A16, Bs16, bs16 = st
    _an5, out["sa16_gauge5"] = _full_report(
        "sa16 SPFs in the 5-state gauge", loss_name, spec, sv5,
        (A16[:5], Bs16, bs16), h)
    return _finish("fullset", ok, dict(
        stem=stem, spfs=args.spfs, norb=norb, cold=cold, weights=w_exc,
        steps=info["steps"],
        converged=info["converged"], wall=wall, base=out_b, **out))


def gate_warmsteps(args):
    """Is a fixed 300-step warm evaluation enough with all 16 configurations
    relaxed? From the converged 16-state state at ``--stem``'s best point
    (``fullset``), step x like the optimiser does, run the evaluation's
    fixed warm steps, then relax on to 1e-8: the loss after the fixed steps
    against the loss converged there, and against the loss change the step
    itself makes (what BOBYQA has to resolve)."""
    stem = args.stem
    loss_name = stem.split("_target_")[1]
    with np.load(os.path.join(C.RESULTS_DIR, stem + "_best_state.npz")) as d:
        x_star = np.asarray(d["x_ghz"], dtype=float)
    spec = C.load_loss(loss_name)
    sv = C.make_solver(14, 4, **C.relax_set("16"))
    st0 = C.load_state(os.path.join(C.RESULTS_DIR,
                                    f"{stem}_sa16_ns4_state.npz"))

    def loss_of(st, x):
        an = sv.analyse(st, C.h_of(x))
        A, Bs, bs = st
        r = spec["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_matched"],
                            energies=an["E_matched"], num_wells=4)
        return float(r["loss"]), an

    l_star, _an = loss_of(st0, x_star)
    print(f"warmsteps at {stem}: x* = {np.round(x_star, 6)}, converged "
          f"16-state loss {l_star:.6f}", flush=True)
    # rhobeg 0.02 of the scaled-to-bounds z: alpha window ~[9.49, 12.96] at
    # stage 3 -> 0.069 GHz, beta [0.05, 2] -> 0.039; the search moves beta
    # by up to 0.15 with 1000 warm steps.
    steps = [("beta1 +0.039 (rhobeg)", 5, +0.039, C.WARM_STEPS),
             ("alpha1 -0.069 (rhobeg)", 1, -0.069, C.WARM_STEPS),
             ("beta2 -0.10 (search)", 6, -0.10, C.SEED_STEPS)]
    ok, out = True, {}
    for label, idx, dx, n in steps:
        x = x_star.copy()
        x[idx] += dx
        h = C.h_of(x)
        t0 = time.perf_counter()
        st, e, info = sv.integrate(n, h, st0)
        l_n, an_n = loss_of(st, x)
        dE_last = float(e[-1] - e[-2])
        st, e2, info2 = sv.relax_to_tol(h, st, tol=1e-8,
                                        max_steps=args.max_steps or 8000)
        l_c, an_c = loss_of(st, x)
        wall = time.perf_counter() - t0
        err, sig = l_n - l_c, l_c - l_star
        print(f"  {label}: after {n} steps loss {l_n:.6f} (dE last "
              f"{dE_last:+.1e}, healthy {an_n['healthy']}); relaxed on "
              f"{info2['steps']} it (converged {info2['converged']}, last "
              f"dE {e2[-1] - e2[-2]:+.1e}) loss {l_c:.6f}; error of the "
              f"fixed steps {err:+.2e} against the step's own change "
              f"{sig:+.2e}  (ratio {abs(err) / max(abs(sig), 1e-12):.2f}); "
              f"detunings {np.round(an_n['detunings'], 6)} -> "
              f"{np.round(an_c['detunings'], 6)}; {wall:.0f} s", flush=True)
        ok &= bool(an_c["healthy"])
        out[label] = dict(loss_fixed=l_n, loss_conv=l_c, err=err, signal=sig,
                          dE_last=dE_last, steps_more=info2["steps"],
                          converged=info2["converged"], wall=wall)
    return _finish("warmsteps", ok, dict(stem=stem, loss_star=l_star, **out))


def gate_seedcheck(args):
    """The first evaluation of a 16-state target stage at its x0 (the x of
    the ``ml_20260924`` run's first evaluation): which candidate is right?
    The stage-2 run kept the seeded state with well 0's detuning at 6.83
    against 3.41 in every 5-state evaluation at that x, because the cold
    16-state candidate had a still higher sum E. Arms: seeded (the idle
    optimum's 16-state state + SEED_STEPS), cold 16 (isolated wells, 1e-6,
    then on to 1e-8), and cold 5 -> 16 (the 5-state cold relaxation, which
    worked there, then the 16-state relaxation from its SPFs)."""
    import json
    name = args.stem
    with open(os.path.join(C.RESULTS_DIR, f"ml_20260924_target_{name}"
                           ".evals.jsonl")) as fh:
        x = np.asarray(json.loads(fh.readline())["x"], dtype=float)
    h = C.h_of(x)
    spec = C.load_loss(name)
    sv = C.make_solver(14, 4, **C.relax_set("16"))
    sv5 = C.make_solver(14, 4)
    print(f"seedcheck {name} at x0 = {np.round(x, 6)}", flush=True)

    def report(label, st, t0, extra=""):
        an = sv.analyse(st, h)
        A, Bs, bs = st
        r = spec["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_matched"],
                            energies=an["E_matched"], num_wells=4)
        print(f"  {label}: sum E16 {an['e_relax']:.8f}  detunings "
              + " ".join(f"{v:.6f}" for v in an["detunings"])
              + f"  healthy {an['healthy']}  loss {r['loss']:.5f}  min P "
              f"{np.min(np.diag(an['F_matched'])):.4f}  occupations "
              + "; ".join(np.array2string(o[:4], precision=4)
                          for o in an["occupations"])
              + f"  {extra}{time.perf_counter() - t0:.0f} s", flush=True)
        return dict(e=float(an["e_relax"]), det=list(an["detunings"]),
                    healthy=bool(an["healthy"]), loss=float(r["loss"]))

    out = {}
    t0 = time.perf_counter()
    seed = C.load_state(os.path.join(
        C.RESULTS_DIR, "ml_20260923_idle_stage1_singles_ground_sa16_ns4_"
        "state.npz"))
    st, _e, _i = sv.integrate(C.SEED_STEPS, h, seed)
    out["seeded"] = report(f"seeded ({C.SEED_STEPS} steps)", st, t0)

    t0 = time.perf_counter()
    st, _e, info = C.cold_state(sv, x, 14, tol=C.COLD_TOL_OPT)
    out["cold16"] = report("cold 16 (1e-6)", st, t0,
                           f"{info['steps']} it, ")
    st, e2, info2 = sv.relax_to_tol(h, st, tol=1e-8, max_steps=10000)
    out["cold16_more"] = report("cold 16, on to 1e-8", st, t0,
                                f"+{info2['steps']} it (conv. "
                                f"{info2['converged']}), ")

    t0 = time.perf_counter()
    st5, _e, info5 = C.cold_state(sv5, x, 14, tol=C.COLD_TOL_OPT)
    an5 = sv5.analyse(st5, h)
    print(f"  cold 5 (1e-6): {info5['steps']} it, sum E5 "
          f"{an5['e_relax']:.8f}, detunings "
          + " ".join(f"{v:.6f}" for v in an5["detunings"])
          + f", healthy {an5['healthy']}", flush=True)
    _A, Bs5, bs5 = an5["state_regauged"]
    A0 = sv.default_state(bs5, h)[0]
    st, _e, info = sv.relax_to_tol(h, (A0, Bs5, bs5), tol=C.COLD_TOL_OPT)
    out["cold5to16"] = report("cold 5 -> 16 (1e-6)", st, t0,
                              f"{info['steps']} it, ")
    ok = all(v["healthy"] for v in out.values())
    return _finish("seedcheck", ok, dict(stage=name, x=list(x), **out))


def gate_blockweight(args):
    solver = C.make_solver(14, 4)
    expect = {"stage2_superpos_w0w1": True, "stage3_superpos_w1w2": True,
              "stage4_superpos_w2w3": False, "stage5_superpos_w0w3": True}
    ok, out = True, {}
    for name, keep in expect.items():
        path = os.path.join(C.RESULTS_DIR,
                            f"{args.base}_target_{name}_best_state.npz")
        with np.load(path) as d:
            x = np.asarray(d["x_ghz"], dtype=float)
            loss_saved = float(d["loss"])
        state = C.load_state(path)
        spec = C.load_loss(name)
        spec["block"] = C.superpos_block(name)
        a, b = spec["block"]
        an = solver.analyse(state, C.h_of(x))
        F = an["F_matched"]
        A, Bs, bs = state
        loss = spec["compute"](C_final=A, bs_final=(Bs, bs), F=F,
                               energies=an["E_matched"], num_wells=4)["loss"]
        wa, wb = F[a, a] + F[a, b], F[b, a] + F[b, b]
        bw = float(min(wa, wb))
        passes = bw >= C.BLOCK_WEIGHT_MIN
        good = passes == keep
        ok &= good
        print(f"  {name}: x = {np.round(x, 6)}")
        print(f"    block [[F_AA F_AB] [F_BA F_BB]] = [[{F[a, a]:.4f} "
              f"{F[a, b]:.4f}] [{F[b, a]:.4f} {F[b, b]:.4f}]]  row sums "
              f"{wa:.4f} {wb:.4f}  -> block weight {bw:.4f}  "
              f"{'kept' if passes else 'REJECTED'} (expected "
              f"{'kept' if keep else 'rejected'})  {'ok' if good else 'WRONG'}")
        print(f"    loss re-scored {loss:.6f} (saved {loss_saved:.6f})  "
              f"leakage {np.array2string(an['leakage'], precision=4)}",
              flush=True)
        out[name] = dict(block_weight=bw, row_sums=[wa, wb], loss=loss,
                         loss_saved=loss_saved, kept=passes)
    return _finish("blockweight", ok, dict(threshold=C.BLOCK_WEIGHT_MIN,
                                          base=args.base, stages=out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gate", choices=("equiv", "traced", "loss", "fixedpoint",
                                    "smoke", "blockweight", "leakage", "fullset",
                                    "warmsteps", "seedcheck"))
    p.add_argument("--stem",
                   default="ml_20260924_target_stage3_superpos_w1w2",
                   help="leakage/fullset: results stem whose best state to "
                   "probe")
    p.add_argument("--spfs", type=int, default=4,
                   help="fullset: SPFs per well of the 16-state relaxation")
    p.add_argument("--max-steps", type=int, default=None,
                   help="fullset: iteration cap (default 25000 for n_s 4, "
                   "6000 above)")
    p.add_argument("--weights", default=DEFAULT_FULLSET_WEIGHTS,
                   help="fullset: relax weight per excitation number 0..4")
    p.add_argument("--norb", type=int, default=14,
                   help="fullset: orbitals per well (not 14 implies --cold)")
    p.add_argument("--cold", action="store_true",
                   help="fullset: start from isolated wells, not the "
                   "optimisation's state")
    p.add_argument("--from-saved", action="store_true",
                   help="fullset: re-analyse the saved sa16 state, no relax")
    p.add_argument("--base", default="ml_20260923",
                   help="blockweight: the chain whose best states to check")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--relax-steps", type=int, default=4000)
    p.add_argument("--tail", type=int, default=1000,
                   help="fixedpoint: fixed iterations after the |dE| stop")
    p.add_argument("--arm", default="base",
                   choices=("base", "dt1e-2", "x64d", "norb16", "timing"))
    args = p.parse_args(argv)
    print(f"jax {jax.__version__} on {jax.devices()}", flush=True)
    return dict(equiv=gate_equiv, traced=gate_traced, loss=gate_loss,
                fixedpoint=gate_fixedpoint, smoke=gate_smoke,
                blockweight=gate_blockweight,
                leakage=gate_leakage, fullset=gate_fullset,
                warmsteps=gate_warmsteps,
                seedcheck=gate_seedcheck)[args.gate](args)


if __name__ == "__main__":
    sys.exit(main())
