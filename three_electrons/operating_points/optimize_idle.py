"""Stage 1 at 3 e-/well: re-optimise the idle point with the ML ITE_opt.

The 8 parameters (alpha_0..3, beta_0..3) under the loss module's
``ascending`` stick-breaking, exactly as ``optimize_qubit_params.py``; the
loss is the paper's ``L_idle`` (``stage1_singles_ground``) on the matched
16x16 ``F``. Local PyBOBYQA from the appendix idle optimum.

    python -u -m operating_points.optimize_idle <run_name> [--loss stage1_singles_ground]
        [--norb 14 --spfs 4] [--maxfun 400]

Writes ``results/<run_name>_<loss>.npz`` and ``..._best_state.npz``.
"""

import argparse
import os
import sys

from operating_points import common as C  # noqa: I100  (sets x64 first)

import jax
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_name")
    p.add_argument("--loss", default="stage1_singles_ground")
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
    args = p.parse_args(argv)

    spec = C.load_loss(args.loss)
    x0, x0_src = C.load_x0("stage1_singles_ground")
    alpha_b = C._normalise_bounds(spec.get("alpha_bounds_ghz",
                                           C.ALPHA_BOUNDS_GHZ), "alpha")
    beta_b = C._normalise_bounds(spec.get("beta_bounds_ghz",
                                          C.BETA_BOUNDS_GHZ), "beta")
    ordering = spec.get("alpha_ordering", "none")
    eps = 0.0          # the one-electron optimiser's alpha_ordering_eps default

    if ordering == "ascending":
        lb_z, ub_z, a_lo, a_hi, warn = C._derive_z_bounds(alpha_b, beta_b, eps)
        if warn:
            print(warn)
        z0 = np.clip(C._x_to_z(x0, a_lo, a_hi, eps), lb_z, ub_z)

        def to_x(z):
            return C._z_to_x(z, a_lo, a_hi, eps)
    else:
        lb_z = np.concatenate([alpha_b[:, 0], beta_b[:, 0]])
        ub_z = np.concatenate([alpha_b[:, 1], beta_b[:, 1]])
        z0 = x0.copy()

        def to_x(z):
            return np.asarray(z, dtype=float)

    settings = dict(norb=args.norb, n_spf=args.spfs, dt=args.dt,
                    precision=args.precision, inner_tol=args.inner_tol,
                    chunk=args.chunk, warm_steps=args.warm_steps,
                    maxfun=args.maxfun, rhobeg=args.rhobeg,
                    rhoend=args.rhoend,
                    seek_global_minimum=C.SEEK_GLOBAL_MINIMUM,
                    device=jax.default_backend(), jax=jax.__version__)
    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"ML idle optimisation  run={args.run_name}  loss={args.loss}")
    print(f"  x0 = {x0}  from {x0_src}")
    print(f"  loss sources: {spec['sha256']}")
    print(f"  settings: {settings}", flush=True)

    solver = C.make_solver(args.norb, args.spfs, dt=args.dt,
                           precision=args.precision,
                           inner_tol=args.inner_tol, chunk=args.chunk)
    run = C.Run(solver, spec, args.norb, to_x, lb_z, ub_z, args.warm_steps,
                log_stem=os.path.join(C.RESULTS_DIR, f"{args.run_name}_{args.loss}"))
    soln, wall = C.run_bobyqa(run, z0, lb_z, ub_z, args.maxfun,
                              args.rhobeg, args.rhoend)

    print(f"\nfinished in {wall:.1f} s ({len(run.records)} evaluations)")
    print(f"solver flag {soln.flag}: {soln.msg}")
    print(f"best loss {run.best['loss']:.6f}  at x = {run.best['x']}")

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = os.path.join(C.RESULTS_DIR, f"{args.run_name}_{args.loss}.npz")
    C.save_trajectory(out, run, soln, wall, spec,
                      alpha_bounds_ghz=alpha_b, beta_bounds_ghz=beta_b,
                      x0_ghz=x0, x0_source=x0_src, alpha_ordering=ordering,
                      solver_x_physical=to_x(soln.x), settings=settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
