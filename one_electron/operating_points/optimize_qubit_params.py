"""Optimise (alpha, beta) for the four-anharmonic-wells setup so the lowest
many-body eigenstates are close to 4-qubit Hartree products.

Strategy: scipy Nelder-Mead on the 8-vector [alpha_0..alpha_3, beta_0..beta_3]
(in GHz), with warm-starting across evaluations — the previously converged
50-state manifold is re-used as the initial guess for the next ITE run.

Run from one_electron/operating_points/ (see slurm/job_idle.sh):
    python optimize_qubit_params.py <run_name> stage1_singles_ground
"""

from __future__ import annotations

import os
import sys
import time
from itertools import product

if __name__ == "__main__" and (
    len(sys.argv) < 3
    or len(sys.argv) > 4
    or not sys.argv[1]
    or not sys.argv[2]
):
    sys.exit(
        "usage: optimize_qubit_params.py <run_name> <loss_name> [<warm_from>]\n"
        "  loss_name  must match losses/<loss_name>.py\n"
        "  warm_from  (optional) prior run stem under "
        "ITE_trajectories/, or a path to an .npz file. Loads "
        "`best_x` and uses it as the new x0."
    )

import numpy as np
import jax.numpy as jnp
from opt_einsum import contract
import pybobyqa

# Make `mctdhf` and `utils` importable regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)

from mctdhf import SineDVR
from mctdhf.MCTDHF_n_species_ITE_opt import MCTDHF_n_species_ITE_opt
from utils import (
    anharm_pot,
    iter_bitstrings,
    qubit_corner_fidelities,
)
import losses

# ── Fixed geometry (independent of alpha, beta) ───────────────────────────────
# Geometry, units, and the Coulomb constant come from the shared single-source
# module geometry_config.py (toggle the well spacing there). NUM_MCTDHF_ORBITALS
# and NUM_STATES are solver settings, not geometry, so they stay here.
from geometry_config import (
    TYPICAL_FREQ_GHZ,
    LENGTH_UNIT_UM,
    TIME_UNIT_PS,
    KAPPA,
    NUM_WELLS,
    NUM_DVR,
    WELL_CENTERS_UM,
    WELL_WIDTHS_UM,
    WELL_CENTERS,
    WELL_WIDTHS,
)

NUM_MCTDHF_ORBITALS = 4
NUM_STATES = 50

# Default per-well alpha/beta bounds in GHz, shape (NUM_WELLS, 2).
# Loss modules may override via ALPHA_BOUNDS_GHZ / BETA_BOUNDS_GHZ.
ALPHA_BOUNDS_GHZ = np.tile(np.array([5.0, 15.0]), (NUM_WELLS, 1))
BETA_BOUNDS_GHZ = np.tile(np.array([0.05, 2.0]), (NUM_WELLS, 1))
DEFAULT_X0_GHZ = np.array([7.0, 9.0, 11.0, 13.0, 0.5, 0.5, 0.5, 0.5])

CONFIGS = iter_bitstrings(NUM_WELLS, 2)  # 16 four-qubit configs


def _normalise_bounds(b, name):
    arr = np.asarray(b, dtype=float)
    if arr.shape == (2,):
        arr = np.tile(arr, (NUM_WELLS, 1))
    if arr.shape != (NUM_WELLS, 2):
        raise ValueError(
            f"{name} must have shape (2,) or ({NUM_WELLS}, 2); got {arr.shape}"
        )
    if not np.all(arr[:, 0] < arr[:, 1]):
        raise ValueError(f"{name} requires lo < hi per row; got {arr}")
    return arr


def _resolve_warm_from(arg, traj_dir):
    """Return (x0_array, source_path_str) from a previous .npz archive."""
    if arg.endswith(".npz") or "/" in arg:
        path = os.path.abspath(arg)
    else:
        path = os.path.join(traj_dir, arg + ".npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"warm_from archive not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "best_x" not in data.files:
            raise KeyError(f"{path} has no `best_x` field")
        x = np.asarray(data["best_x"], dtype=float)
    if x.shape != (2 * NUM_WELLS,):
        raise ValueError(
            f"warm_from best_x has shape {x.shape}; expected ({2 * NUM_WELLS},)"
        )
    return x, path


def _z_to_x(z, alpha_lo, alpha_hi, eps, num_wells=NUM_WELLS):
    """Stick-breaking: fractional [t_0..t_{K-1}, β...] → physical [α..., β...].

    Each tⱼ ∈ [0, 1] is the fraction of the *remaining* feasible α-headroom
    consumed at step j, so for any t in the unit box we get
    α_lo ≤ α₀ < α₁ < ... < α_{K-1} ≤ α_hi with each gap ≥ eps. Concretely:

        α₀ = α_lo + t₀ · (α_hi − α_lo − (K−1)·eps)
        αⱼ = α_{j−1} + eps + tⱼ · (α_hi − α_{j−1} − (K−j)·eps)   for j ≥ 1
    """
    z = np.asarray(z, dtype=float)
    K = num_wells
    alpha = np.empty(K)
    alpha[0] = alpha_lo + z[0] * (alpha_hi - alpha_lo - (K - 1) * eps)
    for j in range(1, K):
        headroom = alpha_hi - alpha[j - 1] - (K - j) * eps
        alpha[j] = alpha[j - 1] + eps + z[j] * headroom
    return np.concatenate([alpha, z[K:]])


def _x_to_z(x, alpha_lo, alpha_hi, eps, num_wells=NUM_WELLS):
    """Inverse stick-breaking: physical [α..., β...] → fractional [t..., β...].

    Inverts `_z_to_x`. For physically valid x (α ascending with gaps ≥ eps,
    α_{K−1} ≤ α_hi) the resulting tⱼ live in [0, 1]. Floating-point can put
    them slightly outside; the caller should clip to [0, 1] afterwards.
    Degenerate denominators (zero headroom) map to t = 0.
    """
    x = np.asarray(x, dtype=float)
    K = num_wells
    alpha = x[:K]
    t = np.empty(K)
    span0 = alpha_hi - alpha_lo - (K - 1) * eps
    t[0] = (alpha[0] - alpha_lo) / span0 if span0 > 0 else 0.0
    for j in range(1, K):
        headroom = alpha_hi - alpha[j - 1] - (K - j) * eps
        gap = alpha[j] - alpha[j - 1] - eps
        t[j] = gap / headroom if headroom > 1e-12 else 0.0
    return np.concatenate([t, x[K:]])


def _derive_z_bounds(alpha_b, beta_b, eps, num_wells=NUM_WELLS):
    """Build z-space bounds for the stick-breaking parametrisation.

    z = [t_0..t_{K-1}, β_0..β_{K-1}] with each tⱼ ∈ [0, 1]. The α part is
    reconstructed via `_z_to_x`, which guarantees α_lo ≤ α₀ < α₁ < ... <
    α_{K-1} ≤ α_hi by construction. Per-well α bounds are collapsed to a
    single global range [min α_lo, max α_hi] — the ordering constraint
    cannot be expressed against per-well bounds. Returns
    (lb_z, ub_z, alpha_lo, alpha_hi, warn_msg) where warn_msg is non-None
    when the input α rows are non-uniform.
    """
    alpha_b = np.asarray(alpha_b, dtype=float)
    beta_b = np.asarray(beta_b, dtype=float)
    alpha_lo = float(alpha_b[:, 0].min())
    alpha_hi = float(alpha_b[:, 1].max())

    span = alpha_hi - alpha_lo
    if span <= (num_wells - 1) * eps:
        raise ValueError(
            f"alpha range [{alpha_lo}, {alpha_hi}] too narrow for "
            f"{num_wells - 1} gaps of >= {eps} GHz each"
        )

    lb_z = np.empty(2 * num_wells)
    ub_z = np.empty(2 * num_wells)
    lb_z[:num_wells] = 0.0
    ub_z[:num_wells] = 1.0
    lb_z[num_wells:] = beta_b[:, 0]
    ub_z[num_wells:] = beta_b[:, 1]

    rows_uniform = (
        np.allclose(alpha_b[:, 0], alpha_b[0, 0])
        and np.allclose(alpha_b[:, 1], alpha_b[0, 1])
    )
    warn_msg = (
        None if rows_uniform
        else (
            f"alpha_ordering=ascending: per-well α bounds collapsed to "
            f"global range [{alpha_lo}, {alpha_hi}] GHz "
            f"(input rows were not uniform)"
        )
    )
    return lb_z, ub_z, alpha_lo, alpha_hi, warn_msg


def _ghz_to_dimensionless(x_ghz):
    return np.asarray(x_ghz) / TYPICAL_FREQ_GHZ


def _coulomb(x1, x2):
    return KAPPA / jnp.sqrt((x1 - x2) ** 2)


def _build_dvr_bases():
    return [
        SineDVR(
            x_min=centre - 0.5 * width,
            x_max=centre + 0.5 * width,
            num_points=NUM_DVR,
        )
        for centre, width in zip(WELL_CENTERS, WELL_WIDTHS)
    ]


def _build_g_dict(dvr_bases):
    g_dict = {}
    for s1 in range(NUM_WELLS):
        for s2 in range(s1 + 1, NUM_WELLS):
            g = dvr_bases[s1].get_g_cross_species(dvr_bases[s2], _coulomb)
            g_dict[(s1, s2)] = jnp.array(g, dtype=jnp.float32)
    return g_dict


DVR_BASES = _build_dvr_bases()
G_DICT = _build_g_dict(DVR_BASES)

# State-weighted ITE: ground state at full weight, next 4 at half, rest unused.
STATE_WEIGHTS = np.concatenate(
    [np.ones(1), 0.5 * np.ones(4), np.zeros(NUM_STATES - 5)]
)

# Solver is constructed once at module load. h_list is passed to .integrate()
# as a traced argument, so the JIT'd step compiles on the first call and
# cache-hits on every subsequent call with a different (alpha, beta).
MCTDHF = MCTDHF_n_species_ITE_opt(
    g_dict=G_DICT,
    num_particles=[1] * NUM_WELLS,
    num_mctdhf_orbitals=[NUM_MCTDHF_ORBITALS] * NUM_WELLS,
    num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
    num_states=NUM_STATES,
    state_weights=STATE_WEIGHTS,
)


def _build_h_list(alphas_dim, betas_dim):
    h_list = []
    for s, (a, b, centre) in enumerate(zip(alphas_dim, betas_dim, WELL_CENTERS)):
        pot = anharm_pot(a, b, centre)
        h = DVR_BASES[s].get_h(potential=pot)
        h_list.append(jnp.array(h, dtype=jnp.float32))
    return h_list


# ── Rayleigh–Ritz post-diagonalisation ────────────────────────────────────────


def _rayleigh_ritz(mctdhf, C, bs, h_list):
    """Diagonalise H on the span of the converged states.

    With `c_propagator="diagonalize"`, every CI step already replaces C by the
    lowest-N eigenvectors of H_CI(b_t), so at convergence this RR call is a
    no-op modulo row sign/ordering. We still run it to (a) recover the
    per-state eigenvalues (the integrator does not surface them) and (b)
    repin near-degenerate blocks that FP noise can leave as a mixed
    orthonormal basis.
    """
    h_MO_list = [
        mctdhf.transform_h(h_list[s], bs[s])[1]
        for s in range(mctdhf.num_species)
    ]
    g_MO_dict, _ = mctdhf._transform_all_g(bs)

    sigma = mctdhf.dCI.get_sigma(C, h_MO_list, g_MO_dict)

    S = C.shape[0]
    C_flat = jnp.reshape(C, (S, -1))
    sigma_flat = jnp.reshape(sigma, (S, -1))

    H_mat = C_flat.conj() @ sigma_flat.T
    H_mat = 0.5 * (H_mat + H_mat.conj().T)

    evals, evecs = jnp.linalg.eigh(H_mat)
    C_new = contract("Jk,J...->k...", evecs, C)
    return C_new, np.real(evals)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_params(
    alphas_ghz,
    betas_ghz,
    compute_loss,
    y_prev=None,
    num_steps=2000,
    dt=1e-2,
    save_every=100,
    verbose=False,
):
    """Run ITE at (alpha, beta), return qubit fidelities and the final state.

    When `y_prev` is given, it is used as the initial guess (warm start) with
    noise_C = 0. For cold starts (y_prev=None) we use the default single-particle
    product-state construction plus noise_C=1e-3, matching the notebook.
    """
    alphas_dim = _ghz_to_dimensionless(alphas_ghz)
    betas_dim = _ghz_to_dimensionless(betas_ghz)

    h_list = _build_h_list(alphas_dim, betas_dim)

    if y_prev is None:
        ts, ys = MCTDHF.integrate(
            num_steps=num_steps,
            dt=dt,
            h_list=h_list,
            t_init=0.0,
            noise_bs=0,
            noise_C=1e-3,
            save_every=save_every,
        )
    else:
        C_prev, bs_prev = MCTDHF._y_to_Cb(y_prev)
        ts, ys = MCTDHF.integrate(
            num_steps=num_steps,
            dt=dt,
            h_list=h_list,
            t_init=0.0,
            bs_init_list=bs_prev,
            C_init=C_prev,
            noise_bs=0.0,
            noise_C=0.0,
            save_every=save_every,
        )

    y_final = ys[-1]
    C_final, bs_final = MCTDHF._y_to_Cb(y_final)

    # Rayleigh–Ritz to get energies
    C_final, energies = _rayleigh_ritz(MCTDHF, C_final, bs_final, h_list)
    y_final = MCTDHF._Cb_to_y(C_final, bs_final)

    # State-averaged NO rotation: per-species, eigh of the sum-over-states
    # 1-RDM and sort by descending occupation. After this the leading 2 columns
    # of each b_s are the qubit |0>_s, |1>_s modes (most-occupied across the
    # 50 propagated states), so the qubit fidelity is just the (0/1)^K corner
    # of |C[k, ...]|^2.
    y_final = MCTDHF.rotate_to_natural_orbitals(y_final)
    C_final, bs_final = MCTDHF._y_to_Cb(y_final)

    F = qubit_corner_fidelities(C_final, NUM_WELLS, n_levels=2)
    loss_result = compute_loss(
        C_final=C_final,
        bs_final=bs_final,
        F=F,
        energies=energies,
        num_wells=NUM_WELLS,
    )
    loss = loss_result["loss"]
    fidelities = loss_result["fidelities"]
    matched_idx = loss_result["matched_idx"]

    if verbose:
        print(f"  loss={loss:.4f}  min_fid={fidelities.min():.4f}")

    return dict(
        loss=loss,
        fidelities=fidelities,
        matched_idx=matched_idx,
        F=np.asarray(F),
        energies=energies,
        y_final=y_final,
        mctdhf=MCTDHF,
    )


# ── Optimisation ──────────────────────────────────────────────────────────────


def optimize(
    compute_loss,
    loss_name,
    x0=None,
    x0_source="default",
    alpha_bounds_ghz=None,
    beta_bounds_ghz=None,
    alpha_ordering="none",
    alpha_ordering_eps=0,
    num_steps_cold=1000,
    num_steps_warm=200,
    maxiter=5000,
    rhobeg=0.1,
    rhoend=1e-3,
    seek_global_minimum=True,
    log_path=None,
):
    """Run PyBOBYQA over [alpha_0..alpha_3, beta_0..beta_3] in GHz.

    Trust-region derivative-free optimizer with quadratic local models.
    Variables are internally rescaled to [0, 1] using the bounds
    (`scaling_within_bounds=True`), so `rhobeg` and `rhoend` are interpreted
    in normalized space — i.e. as fractions of each parameter's bound range.
    Defaults: `rhoend=1e-3` ≈ 10 MHz on α (range 10 GHz) and ≈ 0.4 MHz on β
    (range 0.38 GHz); `rhobeg=0.1` is 10% of each range. Warm-starts ITE from
    the previous converged state. Bounds are enforced natively by PyBOBYQA.

    `alpha_ordering`:
      - "none" (default): direct optimisation over [α₀..α_{K-1}, β...].
      - "ascending": stick-breaking reparametrisation to [t₀..t_{K-1}, β...]
        with each tⱼ ∈ [0, 1]. The α-part is reconstructed via
        αⱼ = α_{j−1} + ε + tⱼ · (α_hi − α_{j−1} − (K−j)·ε) (with α_{-1}≡α_lo
        and the j=0 step using α_lo + t₀ · (α_hi − α_lo − (K−1)·ε)).
        This guarantees α_lo ≤ α₀ < α₁ < ... < α_{K-1} ≤ α_hi with gaps
        ≥ alpha_ordering_eps for every t in the unit box. Per-well α
        bounds collapse to a single global range [min α_lo, max α_hi];
        if the input α rows are non-uniform a warning is printed.
        Trajectory and prints stay in physical α space.
    """
    alpha_b = _normalise_bounds(
        ALPHA_BOUNDS_GHZ if alpha_bounds_ghz is None else alpha_bounds_ghz,
        "alpha_bounds_ghz",
    )
    beta_b = _normalise_bounds(
        BETA_BOUNDS_GHZ if beta_bounds_ghz is None else beta_bounds_ghz,
        "beta_bounds_ghz",
    )
    lb = np.concatenate([alpha_b[:, 0], beta_b[:, 0]])
    ub = np.concatenate([alpha_b[:, 1], beta_b[:, 1]])

    x0 = np.asarray(DEFAULT_X0_GHZ if x0 is None else x0, dtype=float)
    if x0.shape != (2 * NUM_WELLS,):
        raise ValueError(
            f"x0 must have shape ({2 * NUM_WELLS},); got {x0.shape}"
        )
    if not (np.all(lb <= x0) and np.all(x0 <= ub)):
        raise ValueError(
            f"x0 outside bounds.\n  x0={x0}\n  lb={lb}\n  ub={ub}"
        )

    state = dict(
        y_prev=None,
        best_loss=np.inf,
        best_x=None,
        trajectory=[],
        eval_idx=0,
    )

    def loss_fn(x):
        state["eval_idx"] += 1
        i = state["eval_idx"]
        t0 = time.time()

        num_steps = num_steps_cold if state["y_prev"] is None else num_steps_warm
        try:
            result = evaluate_params(
                alphas_ghz=x[:NUM_WELLS],
                betas_ghz=x[NUM_WELLS:],
                compute_loss=compute_loss,
                y_prev=state["y_prev"],
                num_steps=num_steps,
            )
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            wall = time.time() - t0
            # Large sentinel so PyBOBYQA's quadratic surrogate sees a steep
            # gradient out of this region. Distance-scaled (in bound-normalised
            # space) when we have a best_x, so points farther from the current
            # best register as proportionally worse — this gives the local
            # model a real direction to move in rather than a flat penalty.
            base_penalty = 1e6
            if state["best_x"] is not None:
                dist = np.linalg.norm((x - state["best_x"]) / (ub - lb))
                sentinel_loss = base_penalty * (1.0 + dist)
            else:
                sentinel_loss = base_penalty
            print(
                f"[{i:03d}] α={np.round(x[:4], 3)}  β={np.round(x[4:], 3)}  "
                f"FAILED ({type(exc).__name__}: {exc}) "
                f"sentinel={sentinel_loss:.2e}  "
                f"warm={state['y_prev'] is not None}  {wall:.1f}s",
                flush=True,
            )
            # Drop the warm-start cache — its state may be corrupted.
            state["y_prev"] = None
            state["trajectory"].append(
                dict(
                    eval=i,
                    x=x.copy(),
                    loss=sentinel_loss,
                    min_fid=0.0,
                    fidelities=np.zeros(len(CONFIGS)),
                    matched_idx=np.zeros(len(CONFIGS), dtype=int),
                    energies=np.zeros(NUM_STATES),
                    wall=wall,
                    oob=True,
                )
            )
            return sentinel_loss

        loss = result["loss"]
        min_fid = float(result["fidelities"].min())
        wall = time.time() - t0

        worst_idx = int(np.argmin(result["fidelities"]))
        worst_cfg = CONFIGS[worst_idx]
        print(
            f"[{i:03d}] α={np.round(x[:4], 3)}  β={np.round(x[4:], 3)}  "
            f"loss={loss:.4f}  min_fid={min_fid:.4f} (cfg {worst_cfg})  "
            f"warm={state['y_prev'] is not None}  {wall:.1f}s",
            flush=True,
        )

        state["y_prev"] = result["y_final"]

        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_x"] = x.copy()
            state["best_result"] = result

        state["trajectory"].append(
            dict(
                eval=i,
                x=x.copy(),
                loss=loss,
                min_fid=min_fid,
                fidelities=result["fidelities"].copy(),
                matched_idx=result["matched_idx"].copy(),
                energies=result["energies"].copy(),
                wall=wall,
                oob=False,
            )
        )
        return loss

    if alpha_ordering not in ("none", "ascending"):
        raise ValueError(
            f"alpha_ordering must be 'none' or 'ascending'; got {alpha_ordering!r}"
        )

    if alpha_ordering == "ascending":
        lb_z, ub_z, alpha_lo_global, alpha_hi_global, warn_msg = _derive_z_bounds(
            alpha_b, beta_b, alpha_ordering_eps
        )
        if warn_msg is not None:
            print(warn_msg)
        alpha_part = x0[:NUM_WELLS]
        if (
            not np.all(np.diff(alpha_part) >= alpha_ordering_eps)
            or alpha_part[0] < alpha_lo_global
            or alpha_part[-1] > alpha_hi_global
        ):
            raise ValueError(
                f"alpha_ordering='ascending' requires x0 alpha part to be "
                f"strictly ascending with gaps >= {alpha_ordering_eps} and "
                f"contained in [{alpha_lo_global}, {alpha_hi_global}]; "
                f"got {alpha_part}"
            )
        z0 = _x_to_z(x0, alpha_lo_global, alpha_hi_global, alpha_ordering_eps)
        # Stick-breaking is exact in real arithmetic; FP can land z slightly
        # outside [0, 1] when x0 sits exactly on a boundary. Clip to be safe.
        z0 = np.clip(z0, lb_z, ub_z)

        def loss_fn_z(z):
            return loss_fn(
                _z_to_x(z, alpha_lo_global, alpha_hi_global, alpha_ordering_eps)
            )

        solver_fn = loss_fn_z
        solver_x0 = z0
        solver_lb, solver_ub = lb_z, ub_z
    else:
        solver_fn = loss_fn
        solver_x0 = x0
        solver_lb, solver_ub = lb, ub

    print("Starting PyBOBYQA optimisation ...")
    print(f"  x0 (GHz) = α={x0[:4]}, β={x0[4:]}")
    print(f"  x0_source = {x0_source}")
    print(f"  α bounds (GHz) per well =\n{alpha_b}")
    print(f"  β bounds (GHz) per well =\n{beta_b}")
    print(f"  alpha_ordering = {alpha_ordering}"
          + (f" (eps={alpha_ordering_eps})" if alpha_ordering == "ascending" else ""))
    print(f"  num_steps cold/warm = {num_steps_cold}/{num_steps_warm}")
    print(f"  maxfun={maxiter}, rhobeg={rhobeg}, rhoend={rhoend}, "
          f"seek_global_minimum={seek_global_minimum}")
    t_start = time.time()
    soln = pybobyqa.solve(
        solver_fn,
        solver_x0,
        bounds=(solver_lb, solver_ub),
        rhobeg=rhobeg,
        rhoend=rhoend,
        maxfun=maxiter,
        scaling_within_bounds=True,
        print_progress=False,
        seek_global_minimum=seek_global_minimum,
    )
    wall_total = time.time() - t_start
    if alpha_ordering == "ascending":
        solver_x_physical = _z_to_x(
            soln.x, alpha_lo_global, alpha_hi_global, alpha_ordering_eps
        )
    else:
        solver_x_physical = np.asarray(soln.x)
    print(f"\nOptimisation finished in {wall_total:.1f}s ({state['eval_idx']} evals).")
    print(f"Solver flag: {soln.flag} ({soln.msg})")
    print(f"Best loss = {state['best_loss']:.4f}")
    print(
        f"Best α (GHz) = {state['best_x'][:4]}\nBest β (GHz) = {state['best_x'][4:]}"
    )

    if log_path is not None:
        _save_trajectory(
            log_path, state, soln, wall_total, loss_name,
            alpha_bounds_ghz=alpha_b,
            beta_bounds_ghz=beta_b,
            x0_ghz=x0,
            x0_source=x0_source,
            alpha_ordering=alpha_ordering,
            solver_x_physical=solver_x_physical,
            seek_global_minimum=seek_global_minimum,
        )

    return state, soln


def _save_trajectory(
    log_path, state, soln, wall_total, loss_name,
    *, alpha_bounds_ghz, beta_bounds_ghz, x0_ghz,
    x0_source="default",
    alpha_ordering="none", solver_x_physical=None,
    seek_global_minimum=True,
):
    """Save optimisation trajectory and best result to an .npz archive."""
    traj = state["trajectory"]
    n = len(traj)
    xs = np.stack([t["x"] for t in traj])
    losses = np.array([t["loss"] for t in traj])
    min_fids = np.array([t["min_fid"] for t in traj])
    walls = np.array([t["wall"] for t in traj])
    oob = np.array([t["oob"] for t in traj])

    fid_rows = [t for t in traj if not t["oob"]]
    fid_xs = (
        np.stack([t["x"] for t in fid_rows]) if fid_rows else np.zeros((0, 8))
    )
    fid_matrix = (
        np.stack([t["fidelities"] for t in fid_rows])
        if fid_rows
        else np.zeros((0, len(CONFIGS)))
    )
    matched_matrix = (
        np.stack([t["matched_idx"] for t in fid_rows])
        if fid_rows
        else np.zeros((0, len(CONFIGS)), dtype=int)
    )
    energies_matrix = (
        np.stack([t["energies"] for t in fid_rows])
        if fid_rows
        else np.zeros((0, NUM_STATES))
    )

    np.savez(
        log_path,
        loss_name=loss_name,
        xs=xs,
        losses=losses,
        min_fids=min_fids,
        walls=walls,
        oob=oob,
        best_x=state["best_x"],
        best_loss=state["best_loss"],
        best_y=state["best_result"]["y_final"],
        solver_x=soln.x,
        solver_fun=soln.f,
        solver_flag=soln.flag,
        solver_msg=soln.msg,
        wall_total=wall_total,
        configs=np.array(CONFIGS),
        fid_xs=fid_xs,
        fidelities=fid_matrix,
        matched_idx=matched_matrix,
        energies=energies_matrix,
        alpha_bounds_ghz=np.asarray(alpha_bounds_ghz),
        beta_bounds_ghz=np.asarray(beta_bounds_ghz),
        x0_ghz=np.asarray(x0_ghz),
        x0_source=x0_source,
        alpha_ordering=alpha_ordering,
        solver_x_physical=np.asarray(
            soln.x if solver_x_physical is None else solver_x_physical
        ),
        seek_global_minimum=seek_global_minimum,
    )
    print(f"Trajectory saved to {log_path}")


if __name__ == "__main__":

    name = sys.argv[1]
    loss_name = sys.argv[2]
    warm_from_arg = sys.argv[3] if len(sys.argv) >= 4 and sys.argv[3] else None
    spec = losses.load(loss_name)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(out_dir, "ITE_trajectories/" + name + "_" + loss_name + ".npz")

    if warm_from_arg is not None:
        traj_dir = os.path.join(out_dir, "ITE_trajectories")
        x0, x0_source = _resolve_warm_from(warm_from_arg, traj_dir)
        print(f"Warm-starting x0 from {x0_source}: {x0}")
    elif "x0_ghz" in spec:
        x0 = np.asarray(spec["x0_ghz"], dtype=float)
        x0_source = f"loss_spec:{loss_name}"
    else:
        x0 = None
        x0_source = "default"

    state, soln = optimize(
        compute_loss=spec["compute"],
        loss_name=loss_name,
        alpha_bounds_ghz=spec.get("alpha_bounds_ghz"),
        beta_bounds_ghz=spec.get("beta_bounds_ghz"),
        x0=x0,
        x0_source=x0_source,
        alpha_ordering=spec.get("alpha_ordering", "none"),
        seek_global_minimum=spec.get("seek_global_minimum", True),
        maxiter=spec.get("maxiter", 5000),
        log_path=log_path,
    )
