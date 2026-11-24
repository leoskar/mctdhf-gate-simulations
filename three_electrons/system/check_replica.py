"""Check of the four-well system against reference energies.

Reference energies of an earlier standard-MCTDHF three-electron calculation
(not included) at the idle control point:

    Hartree                       2586.80203116
    norb-6 MCTDHF                 2586.86273224
    isolated well 0, norb 6        342.36408

By default the module checks the isolated-well energy: it tests the quasi-1D
``veff`` intra-well kernel and the CI machinery together, and it costs
seconds rather than hours.

The four-well number is reproduced by ``--four-well``. It is a demanding check on every
ingredient at once — the units, the well geometry, the per-well DVR boxes,
the quasi-1D ``veff`` intra-well kernel, the bare inter-well Coulomb and the
anharmonic potential — because any one of them being wrong moves the energy
far more than the tolerance below.

    python -m system.check_replica                 # isolated wells (fast)
    python -m system.check_replica --four-well     # the K=4 energy too

⚠⚠ The K=4 relaxation has TWO traps.

**The basin.** With ``sil`` and dt = 2e-3 the imaginary-time relaxation at
K = 4 finds a single-determinant basin and stops near **2587.94** instead of
2586.86 (a run from a cold start here descended smoothly to 2588.03 and
flattened).  A **Hartree warm start at dt = 5e-4** avoids it, which is why
``--four-well`` defaults to that step size and warm-starts from the isolated
wells.  It is slow; budget hours, not minutes.

⚠ The relaxation step size is NOT free: dt > 2e-3 selects a collapsed fixed
point.  At dt = 0.02 the four-well relaxation converges smoothly, reports
``dE = 4.6e-12``, and lands on 6448.5 instead of 2586.9.  A converged-looking
imaginary-time run at too large a step is a WRONG stationary point, not a
noisy one, so there is no drift or residual to warn you. Keep dt <= 2e-3.
"""

import argparse
import sys
import time

import jax

from ._precision import X64  # noqa: F401
# ^ sets jax_enable_x64 before any array exists; True unless
#   ML_PRECISION=c64 is in the environment.

import jax.numpy as jnp
import numpy as np

from mctdhf.MCTDHF_n_species import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_cmf_integrator

from . import fourwell as fw

TOL = 5e-5          # on an energy of 2.6e3, i.e. 2e-8 relative
TOL_ISO = 5e-6      # the isolated-well check; the reference has 8 figures
E_ISO_W0_NORB6 = 342.36408   # reference: isolated well 0, relaxed norb-6 optimum


def hf_start(h_list, norb):
    """Lowest ``norb`` eigenvectors of each well's one-body Hamiltonian."""
    bs = []
    for s in range(fw.NUM_WELLS):
        _w, v = np.linalg.eigh(np.asarray(h_list[s], dtype=np.complex128))
        bs.append(jnp.asarray(v[:, :norb], dtype=jnp.complex128))
    return bs


def relax_isolated(w, norb, steps, dt, x, krylov=20):
    """Relax ONE well on its own. Cheap (D = C(norb,3) determinants), and its
    orbitals are a far better starting point for the coupled run than the
    one-body eigenvectors — the four-well relaxation then has only the
    inter-well polarisation left to find."""
    h_list = fw.h_static(x)
    solver = MCTDHF_n_species_DVR(
        h_list=[h_list[w]], g_dict={(0, 0): fw.g_dict()[(w, w)]},
        num_particles=[fw.NUM_PARTICLES], num_mctdhf_orbitals=[norb],
        num_spatial_orbitals=[fw.NUM_DVR], num_states=1, imag_time=True,
        eps=1e-6, e_shift=0.0,
    )
    _ev, vec = np.linalg.eigh(np.asarray(h_list[w], dtype=np.complex128))
    bs = [jnp.asarray(vec[:, :norb], dtype=jnp.complex128)]
    C = np.zeros((1,) + tuple(solver.num_dets), dtype=np.complex128)
    C[(0,) + (0,) * 1] = 1.0
    step = get_cmf_integrator(solver, dt, c_propagator="sil",
                              b_propagator="cn",
                              c_kwargs={"krylov_dim": krylov},
                              b_kwargs={"solver_tol": 1e-12,
                                        "max_solver_iter": 400})
    norm_fn = solver._make_norm_fn("orthonormalize")
    t, y = 0.0, norm_fn(pack_Cb_to_y(jnp.asarray(C), bs))
    e_prev = float(np.real(solver.calculate_energy(t, y)).sum())
    for i in range(steps):
        t, y = step(t, y)
        y = norm_fn(y)
        if (i + 1) % 250 == 0:          # an isolated well converges long
            e = float(np.real(solver.calculate_energy(t, y)).sum())
            if abs(e - e_prev) < 1e-11:  # before the step budget runs out
                break
            e_prev = e
    _C, bs_out = solver._y_to_Cb(y)
    return bs_out[0], float(np.real(solver.calculate_energy(t, y)).sum())


def relax(norb, steps, dt, krylov=20, num_states=1, x=None, verbose=True,
          bs_init=None):
    """Imaginary-time relaxation of the four-well system at control point x."""
    x = fw.X_IDLE if x is None else x
    h_list = fw.h_static(x)
    solver = MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=fw.g_dict(),
        num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_mctdhf_orbitals=[norb] * fw.NUM_WELLS,
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_states=num_states, imag_time=True, eps=1e-6, e_shift=0.0,
    )
    bs = hf_start(h_list, norb) if bs_init is None else list(bs_init)

    # Hartree-Fock determinant: the lowest determinant of each species.
    C = np.zeros((num_states,) + tuple(solver.num_dets), dtype=np.complex128)
    for a in range(num_states):
        C[(a,) + (0,) * fw.NUM_WELLS] = 1.0
    if num_states > 1:                    # distinct seeds per state
        rng = np.random.default_rng(0)
        C = C + 1e-2 * (rng.standard_normal(C.shape)
                        + 1j * rng.standard_normal(C.shape))
    C = jnp.asarray(C)

    step = get_cmf_integrator(solver, dt, c_propagator="sil",
                              b_propagator="cn",
                              c_kwargs={"krylov_dim": krylov},
                              b_kwargs={"solver_tol": 1e-12,
                                        "max_solver_iter": 400})
    norm_fn = solver._make_norm_fn("orthonormalize")

    t, y = 0.0, norm_fn(pack_Cb_to_y(C, bs))
    e_prev = float(np.real(solver.calculate_energy(t, y)).sum())
    t0 = time.perf_counter()
    for i in range(steps):
        t, y = step(t, y)
        y = norm_fn(y)
        if (i + 1) % 250 == 0:
            e = float(np.real(solver.calculate_energy(t, y)).sum())
            if verbose:
                print(f"    step {i+1:6d}  E = {e:.8f}  dE = {e - e_prev:+.2e}",
                      flush=True)
            if abs(e - e_prev) < 1e-12:
                e_prev = e
                break
            e_prev = e
    wall = time.perf_counter() - t0
    e = float(np.real(solver.calculate_energy(t, y)).sum())
    if verbose:
        print(f"    relaxed in {wall:.1f} s")
    return solver, y, e


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--norb", type=int, default=6)
    p.add_argument("--four-well", action="store_true",
                   help="also relax the coupled K=4 system (hours)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--dt", type=float, default=5e-4,
                   help="MUST be <= 2e-3; K=4 needs 5e-4 (module docstring)")
    args = p.parse_args(argv)

    print(f"jax {jax.__version__} on {jax.devices()}\n")
    print("replica ingredients:")
    fw.self_check()
    print()

    if args.dt > 2e-3 + 1e-12:
        print(f"REFUSED: dt = {args.dt} > 2e-3 selects the collapsed fixed "
              f"point (see the module docstring).")
        return 2

    print(f"relaxing each well on its own at norb {args.norb}")
    bs_init, e_iso = [], []
    for w in range(fw.NUM_WELLS):
        b, e_w = relax_isolated(w, args.norb, min(args.steps, 8000), args.dt,
                                fw.X_IDLE)
        bs_init.append(b)
        e_iso.append(e_w)
        print(f"    well {w}: E = {e_w:.8f}")
    print(f"    sum of isolated wells = {sum(e_iso):.8f}")

    d_iso = abs(e_iso[0] - E_ISO_W0_NORB6)
    if args.norb == 6:
        print(f"\n  E(well 0, replica)  = {e_iso[0]:.8f}")
        print(f"  E(well 0, reference)= {E_ISO_W0_NORB6:.8f}")
        print(f"  |difference|        = {d_iso:.3e}   "
              f"(tolerance {TOL_ISO:.1e})")
        print(f"\n  ISOLATED-WELL CHECK "
              f"{'PASSES' if d_iso < TOL_ISO else 'FAILS'}")
        if d_iso >= TOL_ISO:
            return 1

    if not args.four_well:
        return 0

    print(f"\nrelaxing [{args.norb}]*4, N = 3, idle control point")
    _solver, _y, e = relax(args.norb, args.steps, args.dt, bs_init=bs_init)

    ref = fw.E_MCTDHF6_IDLE
    print(f"\n  E(replica, norb {args.norb}) = {e:.8f}")
    print(f"  E(reference, norb 6)      = {ref:.8f}")
    if args.norb == 6:
        d = abs(e - ref)
        print(f"  |difference|              = {d:.3e}   "
              f"(tolerance {TOL:.1e})")
        print(f"\n  {'CHECK PASSES' if d < TOL else 'CHECK FAILS'}")
        return 0 if d < TOL else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
