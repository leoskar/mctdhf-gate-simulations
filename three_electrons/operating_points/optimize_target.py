"""Stages 2-5 at 3 e-/well: the entangling targets, spectators frozen.

``optimize_qubit_params_frozen.py`` with the ML ITE_opt. The active wells
come from the loss module's ``A, B``; the two spectators' alpha AND beta
are frozen at ``freeze_from`` — the ML IDLE optimum (a chain), not
the appendix idle point. Active alphas are held ``FREEZE_GAP_GHZ`` clear of
the frozen neighbours on the ``ascending`` stages 2-4 and are free on
stage 5. x0 for the active wells is the appendix target optimum, clamped
into the window. Whenever there is no warm state (the first evaluation, or
after a failure) the evaluation relaxes BOTH cold and from the ML idle
optimum's state and keeps the lower sum E5.

Two additions (after stages 4 and 5 of
``ml_20260923`` stayed on the plateau):

* every evaluation is REJECTED (sentinel, warm state kept) when a
  superposition root keeps less than ``C.BLOCK_WEIGHT_MIN`` of its weight on
  the pair {|A>, |B>}: the super term of the one-electron loss rewards A hybridising with an
  intruder;
* ``--lever beta3`` (``alpha<w>`` / ``beta<w>``, an active well) runs a
  RESONANCE SEARCH before BOBYQA: a 1-D root search on the active wells'
  detuning gap along that one parameter, starting from the appendix x0.
  BOBYQA then starts from the lowest-loss search point where the pair
  mixes (else the smallest gap). Far from resonance the loss is flat to
  ~1e-3, which BOBYQA cannot follow.

    python -u -m operating_points.optimize_target <run_name> <loss_name> <freeze_from>
        [--norb 14 --spfs 4] [--maxfun 400] [--lever beta3 [--search-only]]

``--relax 16`` relaxes all 16
register configurations (weights 2^-n) instead of |0000> + singles; the
seed is then the idle optimum's state re-relaxed with that set
(``<idle stem>_sa16_ns4_state.npz``, ``validate fullset``). Everything else,
including x0 and the search, is unchanged.

``freeze_from`` is a results .npz (its ``best_x``) or a run stem under
``results/``.
"""

import argparse
import os
import sys

from operating_points import common as C  # noqa: I100  (sets x64 first)

import jax
import numpy as np


def _resolve(freeze_from):
    if freeze_from.endswith(".npz") or "/" in freeze_from:
        path = os.path.abspath(freeze_from)
    else:
        path = os.path.join(C.RESULTS_DIR, freeze_from + ".npz")
    return C.load_best_x(path), path


def _seed_state(anchor_path, solver, relax="5"):
    """The ML idle optimum's relaxed state (``<stem>_best_state.npz``),
    which seeds the first evaluation alongside the cold start
    (``common.evaluate_params``). With ``relax="16"``, the same
    point re-relaxed with all 16 configurations (``validate fullset``)."""
    path = anchor_path[:-4] + ("_best_state.npz" if relax == "5"
                               else "_sa16_ns4_state.npz")
    A, Bs, bs = C.load_state(path)
    want_A = (solver.num_states,) + solver.num_spfs
    if A.shape != want_A or bs[0].shape[1] != solver.ml.num_mctdhf_orbitals[0]:
        raise ValueError(f"seed {path} has A {A.shape}, norb {bs[0].shape[1]};"
                         f" this run needs A {want_A}, norb "
                         f"{solver.ml.num_mctdhf_orbitals[0]}")
    return (A, Bs, bs), path


# Largest lever move per search step. The guard rejects a warm evaluation
# whose detunings jump by > 25 %; at stage 5, d0 moves 1.58 per unit beta0
# (d0 ~ 1.74), so 0.15 is a ~14 % move. alpha moves a detuning by ~0.1-0.2
# per GHz. Measured on the ml_20260923 probe evaluations.
SEARCH_MAX_STEP = {"alpha": 1.0, "beta": 0.15}


def _parse_lever(lever, active, n_act):
    """``"beta3"`` -> index into z = [alpha_active..., beta_active...]."""
    for kind, off in (("alpha", 0), ("beta", n_act)):
        if lever.startswith(kind):
            w = int(lever[len(kind):])
            if w not in active:
                raise ValueError(f"lever {lever}: well {w} is not active "
                                 f"{active}")
            return kind, off + active.index(w)
    raise ValueError(f"lever {lever!r}: expected alpha<w> or beta<w>")


def resonance_search(run, z0, k, lb_z, ub_z, wells, block, max_step,
                     max_evals=12, mixed_fid=0.75, bracket_tol=1e-4):
    """Walk z[k] until the active wells' dressed detuning gap closes.

    g = d[wa] - d[wb] (from the MATCHED roots). Secant steps, capped at
    ``max_step`` and clipped to the bounds; once two accepted points bracket
    a sign change, regula falsi (Illinois) inside the bracket. Across the
    resonance the matched labels swap, so g jumps by the dressed splitting
    (~2J, ~0.012-0.014 at stages 2/3) instead of passing through zero; the
    bracket closes on the jump, i.e. on the 50/50 point.

    Stops when the pair is clearly mixed (fidelity of A <= ``mixed_fid``),
    the bracket is narrower than ``bracket_tol``, the gap stops closing,
    the lever is pinned at a bound, an evaluation is rejected, or after
    ``max_evals``. There is deliberately no |g| tolerance: the splitting
    differs by > 15x between stages (~0.012 at stages 2-4, <~7e-4 at stage
    5), and stage 5 stopped UNMIXED at |g| = 0.015 under a
    0.02 tolerance.

    Returns the accepted point with the lowest loss among those where the
    pair visibly mixes (fidelity of A <= 0.95); if none does, the point with
    the smallest |g| (on the plateau the loss does not point anywhere).
    """
    wa, wb = wells
    a = block[0]
    pts = []                               # accepted (v, g, loss, z)

    def evaluate(z):
        n0 = len(run.records)
        run(z)
        rec = run.records[n0]
        if rec["oob"]:
            return None
        d = rec["detunings"]
        g = float(d[wa] - d[wb])
        fa = float(rec["fidelities"][a])
        print(f"      search: z[{k}] = {z[k]:.6f}  gap = {g:+.5f}  "
              f"fid(A) = {fa:.4f}  loss = {rec['loss']:.5f}", flush=True)
        pts.append((float(z[k]), g, float(rec["loss"]), z.copy(), fa))
        return g, fa

    def done(r):
        return r is not None and r[1] <= mixed_fid

    z = np.asarray(z0, dtype=float).copy()
    r = evaluate(z)
    if r is None:
        raise RuntimeError("resonance search: the start point was rejected")
    # first probe: a quarter step, away from a bound if pinned
    h = 0.25 * max_step if z[k] + 0.25 * max_step <= ub_z[k] else \
        -0.25 * max_step
    lo_side = hi_side = None               # (v, g) either side of the jump
    reason = "max evaluations"
    while not done(r) and len(pts) < max_evals:
        v1, g1 = pts[-1][0], pts[-1][1]
        if len(pts) == 1:
            v_new = v1 + h
        else:
            # bracket bookkeeping on all accepted points
            neg = [(v, g) for v, g, *_ in pts if g < 0]
            pos = [(v, g) for v, g, *_ in pts if g > 0]
            if neg and pos:
                # the closest pair of opposite signs in lever value
                lo_side, hi_side = min(
                    ((n, p) for n in neg for p in pos),
                    key=lambda np_: abs(np_[0][0] - np_[1][0]))
            if lo_side is not None:
                (va, ga), (vb, gb) = lo_side, hi_side
                if abs(va - vb) < bracket_tol:
                    reason = "bracket closed"
                    break
                # Illinois: halve the stale end's value
                stale = pts[-2][1] if len(pts) > 1 else None
                if stale is not None and np.sign(stale) == np.sign(g1):
                    if np.sign(ga) == np.sign(g1):
                        gb = 0.5 * gb
                    else:
                        ga = 0.5 * ga
                v_new = va - ga * (vb - va) / (gb - ga)
            else:
                v0, g0 = pts[-2][0], pts[-2][1]
                if len(pts) >= 3 and abs(g1) >= abs(g0):
                    # no bracket and a SECANT step did not close the gap:
                    # the gap saturates along this lever (a secant would
                    # cycle about the extremum). Not after the first probe:
                    # its direction is blind, and a wrong-way probe is what
                    # the secant turns round (stage 2).
                    reason = "gap stopped closing along the lever"
                    break
                slope = (g1 - g0) / (v1 - v0) if v1 != v0 else 0.0
                if slope == 0.0 or not np.isfinite(slope):
                    reason = "flat gap along the lever"
                    break
                step = float(np.clip(-g1 / slope, -max_step, max_step))
                v_new = v1 + step
        v_new = float(np.clip(v_new, lb_z[k], ub_z[k]))
        if abs(v_new - v1) < 1e-12:
            reason = "lever pinned at a bound"
            break
        z = pts[-1][3].copy()
        z[k] = v_new
        r = evaluate(z)
        if r is None:
            reason = "evaluation rejected"
            break
    else:
        reason = "resonance reached" if done(r) else reason
    mixed = [p for p in pts if p[4] <= 0.95]
    best = (min(mixed, key=lambda p: p[2]) if mixed
            else min(pts, key=lambda p: abs(p[1])))
    print(f"  resonance search: {reason} after {len(pts)} accepted "
          f"evaluation(s); best z[{k}] = {best[0]:.6f}  gap = {best[1]:+.5f}"
          f"  loss = {best[2]:.5f}", flush=True)
    return best[3], dict(reason=reason, n=len(pts), lever_value=best[0],
                         gap=best[1], loss=best[2])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_name")
    p.add_argument("loss_name")
    p.add_argument("freeze_from")
    p.add_argument("--norb", type=int, default=14)
    p.add_argument("--spfs", type=int, default=4)
    p.add_argument("--maxfun", type=int, default=C.MAXFUN)
    p.add_argument("--rhobeg", type=float, default=C.RHOBEG)
    p.add_argument("--rhoend", type=float, default=C.RHOEND)
    p.add_argument("--warm-steps", type=int, default=C.WARM_STEPS)
    p.add_argument("--dt", type=float, default=2e-2)
    p.add_argument("--precision", default="mixed", choices=("mixed", "x64d"))
    p.add_argument("--inner-tol", type=float, default=1e-2)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--lever", default=None,
                   help="resonance search before BOBYQA along this "
                        "parameter: alpha<w> or beta<w> of an active well")
    p.add_argument("--search-evals", type=int, default=12)
    p.add_argument("--search-warm-steps", type=int, default=C.SEED_STEPS,
                   help="warm steps per search evaluation (the lever moves "
                        "much further than a BOBYQA step)")
    p.add_argument("--relax", default="5", choices=("5", "16"),
                   help="relaxed set: |0000> + singles, or all 16 "
                   "configurations at 2^-n (C.relax_set)")
    p.add_argument("--search-only", action="store_true",
                   help="stop after the resonance search (a test)")
    args = p.parse_args(argv)

    spec = C.load_loss(args.loss_name)
    spec["block"] = C.superpos_block(args.loss_name)
    anchor_x, anchor_src = _resolve(args.freeze_from)
    x0_app, x0_src = C.load_x0(args.loss_name)
    alpha_b = C._normalise_bounds(spec.get("alpha_bounds_ghz",
                                           C.ALPHA_BOUNDS_GHZ), "alpha")
    beta_b = C._normalise_bounds(spec.get("beta_bounds_ghz",
                                          C.BETA_BOUNDS_GHZ), "beta")
    ordering = spec.get("alpha_ordering", "none")
    apply_gap = ordering == "ascending"

    active = C._active_wells_for_loss(args.loss_name)
    spectators = sorted(set(range(C.NUM_WELLS)) - set(active))
    frozen_alpha = anchor_x[:C.NUM_WELLS].copy()
    frozen_beta = anchor_x[C.NUM_WELLS:].copy()
    beta_wells = active if C.FREEZE_BETA else list(range(C.NUM_WELLS))

    a_lo, a_hi = C._build_active_alpha_window(
        active, frozen_alpha, alpha_b, C.FREEZE_GAP_GHZ, apply_gap=apply_gap)
    b_lo, b_hi = beta_b[beta_wells, 0], beta_b[beta_wells, 1]
    lb_z = np.concatenate([a_lo, b_lo])
    ub_z = np.concatenate([a_hi, b_hi])

    z0_alpha = np.clip(x0_app[:C.NUM_WELLS][active], a_lo, a_hi)
    if C.FREEZE_BETA:
        z0_beta = np.clip(x0_app[C.NUM_WELLS:][active], b_lo, b_hi)
    else:
        z0_beta = np.clip(frozen_beta[beta_wells], b_lo, b_hi)
    z0 = np.concatenate([z0_alpha, z0_beta])
    n_act = len(active)

    def to_x(z):
        alpha, beta = frozen_alpha.copy(), frozen_beta.copy()
        alpha[active] = z[:n_act]
        for j, w in enumerate(beta_wells):
            beta[w] = z[n_act + j]
        return np.concatenate([alpha, beta])

    x0 = to_x(z0)
    settings = dict(norb=args.norb, n_spf=args.spfs, dt=args.dt,
                    precision=args.precision, inner_tol=args.inner_tol,
                    chunk=args.chunk, warm_steps=args.warm_steps,
                    maxfun=args.maxfun, rhobeg=args.rhobeg,
                    rhoend=args.rhoend,
                    seek_global_minimum=C.SEEK_GLOBAL_MINIMUM,
                    active=active, spectators=spectators,
                    apply_gap=apply_gap, gap_ghz=C.FREEZE_GAP_GHZ,
                    freeze_beta=C.FREEZE_BETA, freeze_from=anchor_src,
                    appendix_x0=x0_src, block=spec["block"],
                    block_weight_min=C.BLOCK_WEIGHT_MIN, lever=args.lever,
                    search_evals=args.search_evals,
                    search_warm_steps=args.search_warm_steps,
                    relax=args.relax, **C.relax_set(args.relax),
                    device=jax.default_backend(), jax=jax.__version__)
    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"ML target optimisation  run={args.run_name}  "
          f"loss={args.loss_name}")
    print(f"  active wells {active}   frozen {spectators} from {anchor_src}")
    print(f"  frozen α = {np.round(frozen_alpha[spectators], 6)}  "
          f"β = {np.round(frozen_beta[spectators], 6)}")
    for j, w in enumerate(active):
        print(f"  active well {w}: α ∈ [{a_lo[j]:.4f}, {a_hi[j]:.4f}] GHz")
    print(f"  appendix x0 {x0_app}  from {x0_src}")
    print(f"  x0 (clamped, spectators frozen) {x0}")
    print(f"  loss sources: {spec['sha256']}")
    print(f"  settings: {settings}", flush=True)

    solver = C.make_solver(args.norb, args.spfs, dt=args.dt,
                           precision=args.precision,
                           inner_tol=args.inner_tol, chunk=args.chunk,
                           **C.relax_set(args.relax))
    seed, seed_path = _seed_state(anchor_src, solver, args.relax)
    print(f"  seed state (first evaluation, alongside cold): {seed_path}",
          flush=True)
    settings["seed_state"] = seed_path
    run = C.Run(solver, spec, args.norb, to_x, lb_z, ub_z, args.warm_steps,
                seed=seed,
                log_stem=os.path.join(C.RESULTS_DIR, f"{args.run_name}_{args.loss_name}"))
    z_start = z0
    if args.lever is not None:
        kind, k = _parse_lever(args.lever, active, n_act)
        print(f"\nresonance search along {args.lever} (z[{k}] in "
              f"[{lb_z[k]:.4f}, {ub_z[k]:.4f}], step <= "
              f"{SEARCH_MAX_STEP[kind]}, {args.search_warm_steps} warm "
              f"steps)", flush=True)
        run.phase, run.warm_steps = "search", args.search_warm_steps
        z_start, search = resonance_search(
            run, z0, k, lb_z, ub_z, active, spec["block"],
            SEARCH_MAX_STEP[kind], max_evals=args.search_evals)
        run.phase, run.warm_steps = "bobyqa", args.warm_steps
        settings["search"] = search
        print(f"  BOBYQA starts at {to_x(z_start)}\n", flush=True)
        if args.search_only:
            return 0
    soln, wall = C.run_bobyqa(run, z_start, lb_z, ub_z, args.maxfun,
                              args.rhobeg, args.rhoend)

    print(f"\nfinished in {wall:.1f} s ({len(run.records)} evaluations)")
    print(f"solver flag {soln.flag}: {soln.msg}")
    print(f"best loss {run.best['loss']:.6f}  at x = {run.best['x']}")

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = os.path.join(C.RESULTS_DIR,
                       f"{args.run_name}_{args.loss_name}.npz")
    C.save_trajectory(
        out, run, soln, wall, spec,
        alpha_bounds_ghz=alpha_b, beta_bounds_ghz=beta_b, x0_ghz=x0,
        x0_source=f"frozen:{anchor_src}|active:{x0_src}",
        alpha_ordering=("frozen_spectators_gap" if apply_gap
                        else "frozen_spectators_free"),
        solver_x_physical=to_x(np.asarray(soln.x)), settings=settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
