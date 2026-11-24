"""Relax the ML state at the idle control point and save it as a checkpoint.

Every integrator arm restarts from the SAME byte-identical state, which is
what makes the arms comparable: drift can then be attributed to the
integrator rather than to where in the pulse each arm happened to be.

The checkpoint is always relaxed in DOUBLE precision and saved as
complex128. A complex64 arm casts it down at load, so the two precisions
start from the same physical state rather than from two separately relaxed
ones.

Relaxation uses ``a_propagator="diagonalize"``: in imaginary time
``_diag_solve_C`` replaces the top tensor by the lowest ``num_states``
eigenvectors of ``H_top`` — improved relaxation, not propagation — which
converges in far fewer steps than SIL and, being Krylov-free, cannot bias
the starting state toward any arm.

⚠ dt <= 2e-3. See ``check_replica``'s docstring: above that, imaginary time
converges smoothly onto the WRONG fixed point.

    python -m system.prep --norb 7 --spfs 4 --states 4
"""

import argparse
import os
import sys
import time

import jax

from ._precision import X64  # noqa: F401
# ^ sets jax_enable_x64 before any array exists; True unless
#   ML_PRECISION=c64 is in the environment.

import jax.numpy as jnp
import numpy as np

from mctdhf.MLMCTDHF_n_species import MLMCTDHF_n_species_DVR
from mctdhf.integrators_ml import get_ml_cmf_integrator

from . import fourwell as fw
from .check_replica import relax_isolated

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "ckpt")


def ckpt_path(norb, n_spf, num_states):
    return os.path.abspath(os.path.join(
        CKPT_DIR, f"idle_n{norb}_s{n_spf}_st{num_states}.npz"))


def make_solver(norb, n_spf, num_states, imag_time, h_list, eps=1e-6,
                state_weights=None, e_shift=0.0):
    return MLMCTDHF_n_species_DVR(
        h_list=h_list, g_dict=fw.g_dict(),
        num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_mctdhf_orbitals=[norb] * fw.NUM_WELLS,
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_spfs=[n_spf] * fw.NUM_WELLS,
        # e_shift moves the whole spectrum without touching any observable
        # (the solver propagates H - e_shift). At 0 the register sits at
        # ~2593 per state while the detunings are 1.8-4.6, so a float32
        # Sum E has an ulp of 0.0039 and an absolute |dE| stopping rule
        # cannot resolve anything. Irrelevant in x64, decisive in c64.
        num_states=num_states, imag_time=imag_time, eps=eps,
        e_shift=e_shift,
        state_weights=state_weights,
    )


def relax_ml(norb, n_spf, num_states, steps, dt, iso_steps=8000,
             verbose=True):
    h_list = fw.h_static(fw.X_IDLE)
    ml = make_solver(norb, n_spf, num_states, True, h_list)

    if verbose:
        print(f"  D_s = {ml.num_dets[0]}  n_s = {n_spf}  "
              f"top tensor {num_states} x {n_spf}^4 = "
              f"{num_states * n_spf ** fw.NUM_WELLS}")

    # Orbitals from the isolated-well relaxations: the coupled relaxation
    # then has only the inter-well polarisation left to find.
    bs = []
    for w in range(fw.NUM_WELLS):
        b, e_w = relax_isolated(w, norb, iso_steps, dt, fw.X_IDLE)
        bs.append(jnp.asarray(b, dtype=jnp.complex128))
        if verbose:
            print(f"    well {w} isolated: E = {e_w:.8f}", flush=True)

    Bs, spf_evals = ml._default_spfs(0.0, bs)
    Bs = [jnp.asarray(B, dtype=jnp.complex128) for B in Bs]
    A = jnp.asarray(ml._default_top_tensor(spf_evals, dtype=jnp.complex128))

    step = get_ml_cmf_integrator(
        ml, dt, a_propagator="diagonalize", B_propagator="diagonalize",
        b_propagator="cn",
        B_kwargs={"orbital_x64": True},
        b_kwargs={"solver_tol": 1e-12, "max_solver_iter": 400},
    )
    norm_fn = ml._make_norm_fn("orthonormalize")

    t, y = 0.0, norm_fn(ml._ABb_to_y(A, Bs, bs))
    e_prev = float(np.real(ml.calculate_energy(t, y)).sum())
    t0 = time.perf_counter()
    for i in range(steps):
        t, y = step(t, y)
        y = norm_fn(y)
        if (i + 1) % 250 == 0:
            e = float(np.real(ml.calculate_energy(t, y)).sum())
            if verbose:
                print(f"    step {i+1:6d}  E = {e:.8f}  "
                      f"dE = {e - e_prev:+.3e}", flush=True)
            if abs(e - e_prev) < 1e-11:
                break
            e_prev = e
    if verbose:
        print(f"    relaxed in {time.perf_counter() - t0:.1f} s")
    return ml, y


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--norb", type=int, required=True)
    p.add_argument("--spfs", type=int, required=True)
    p.add_argument("--states", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--dt", type=float, default=2e-3)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if args.dt > 2e-3 + 1e-12:
        print("REFUSED: dt > 2e-3 selects the collapsed fixed point.")
        return 2

    out = ckpt_path(args.norb, args.spfs, args.states)
    if os.path.exists(out) and not args.force:
        print(f"exists, not regenerating: {out}")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"relaxing norb={args.norb} spfs={args.spfs} "
          f"states={args.states} at the idle point")
    ml, y = relax_ml(args.norb, args.spfs, args.states, args.steps, args.dt)

    e = np.real(ml.calculate_energy(0.0, y))
    pops = [np.asarray(p) for p in ml.spf_populations(y)]
    nos = [np.asarray(o) for o in ml.natural_occupations(y)]
    print(f"\n  E per state = {np.array2string(np.asarray(e), precision=8)}")
    print(f"  smallest SPF population per well = "
          f"{[float(p[-1]) for p in pops]}")
    print(f"  smallest natural occupation per well = "
          f"{[float(o[-1]) for o in nos]}")

    A, Bs, bs = ml._y_to_ABb(y)
    np.savez(
        out,
        A=np.asarray(A), **{f"B{s}": np.asarray(Bs[s])
                            for s in range(fw.NUM_WELLS)},
        **{f"b{s}": np.asarray(bs[s]) for s in range(fw.NUM_WELLS)},
        norb=args.norb, n_spf=args.spfs, num_states=args.states,
        energy=np.asarray(e), dt_relax=args.dt,
        spf_pop_min=np.array([float(p[-1]) for p in pops]),
        no_min=np.array([float(o[-1]) for o in nos]),
    )
    print(f"\n  wrote {out}")
    return 0


def load_ckpt(path, dtype=jnp.complex128):
    d = np.load(path)
    A = jnp.asarray(d["A"], dtype=dtype)
    Bs = [jnp.asarray(d[f"B{s}"], dtype=dtype) for s in range(fw.NUM_WELLS)]
    bs = [jnp.asarray(d[f"b{s}"], dtype=dtype) for s in range(fw.NUM_WELLS)]
    meta = {k: d[k] for k in ("norb", "n_spf", "num_states", "energy",
                              "spf_pop_min", "no_min")}
    return A, Bs, bs, meta


if __name__ == "__main__":
    sys.exit(main())
