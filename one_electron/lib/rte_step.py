"""Single-step RTE: 1 -> 2 -> 1 swap via erf-ramped parametric modulation.

Propagates |1000> (state index 1 of the anchor-1 NO basis) through anchor 2
and back. The `__main__` baseline uses the VMF integrator; the QOCT
`propagate` driver uses the CMF half-step integrator with diagonalize/
diagonalize sub-propagators, which is autodiff-clean — each substep is a
closed-form eigh + matrix exponential, with no iterative-solver custom_vjp
to truncate the reverse pass. Runs mixed precision: float32 integrals +
complex64 state, with the orbital substep's non-Hermitian expm promoted to
complex128 (orbital_x64=True); pure complex64 is unstable.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf

import pint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mctdhf import (
    MCTDHF_n_species_DVR,
    SineDVR,
    get_vmf_integrator,
    get_cmf_half_step_integrator,
    pack_Cb_to_y,
    unpack_y_to_Cb,
)
from utils import hartree_overlap, iter_bitstrings


# ── Anchors ───────────────────────────────────────────────────────────────────
# Anchor set is tied to the geometry in geometry_config.py (toggle there), so the
# grid and the anchors cannot mismatch.
from geometry_config import ANCHOR_PATHS as _ANCHOR_PATHS

ANCHOR_1_PATH = _ANCHOR_PATHS[1]
ANCHOR_2_PATH = _ANCHOR_PATHS[2]

anchor_1 = np.load(ANCHOR_1_PATH)
anchor_2 = np.load(ANCHOR_2_PATH)

x_star_1_ghz = jnp.array(anchor_1["best_x"], dtype=jnp.float64)
x_star_2_ghz = jnp.array(anchor_2["best_x"], dtype=jnp.float64)
y_1_packed = anchor_1["best_y"]

DELTA_2 = float(anchor_2["energies"][-1, 2] - anchor_2["energies"][-1, 1])


# ── Geometry constants (from shared geometry_config.py) ──────────────────────

from geometry_config import (
    NUM_WELLS,
    NUM_DVR,
    TYPICAL_FREQ_GHZ,
    LENGTH_UNIT_UM,
    KAPPA,
    WELL_CENTERS_UM,
    WELL_WIDTHS_UM,
    GEOM_TAG,
)

NUM_ORBITALS = 4
NUM_STATES_ANCHOR = 50

# Precision of the (real) Hamiltonian integrals h_list / g_dict and the
# geometry arrays. Mixed-precision CMF half-step: float32 integrals +
# complex64 state, with the orbital substep's non-Hermitian expm promoted
# to complex128 (orbital_x64=True). Flip to float64 for an all-x64 run.
H_DTYPE = jnp.float32

WELL_CENTERS = jnp.array(WELL_CENTERS_UM / LENGTH_UNIT_UM, dtype=H_DTYPE)
WELL_WIDTHS = np.array(WELL_WIDTHS_UM) / LENGTH_UNIT_UM


def _coulomb(x1, x2):
    return KAPPA / jnp.sqrt((x1 - x2) ** 2)


DVR_BASES = [
    SineDVR(
        x_min=float(WELL_CENTERS[s]) - 0.5 * WELL_WIDTHS[s],
        x_max=float(WELL_CENTERS[s]) + 0.5 * WELL_WIDTHS[s],
        num_points=NUM_DVR,
    )
    for s in range(NUM_WELLS)
]

# Kinetic-only h (constant) and DVR grid per species, dtype H_DTYPE.
_zero_pot = lambda x: jnp.zeros_like(x)
T_S = [
    jnp.array(DVR_BASES[s].get_h(potential=_zero_pot), dtype=H_DTYPE)
    for s in range(NUM_WELLS)
]
X_S = [
    jnp.array(DVR_BASES[s].get_grid(), dtype=H_DTYPE)
    for s in range(NUM_WELLS)
]

G_DICT = {}
for s1 in range(NUM_WELLS):
    for s2 in range(s1 + 1, NUM_WELLS):
        g = DVR_BASES[s1].get_g_cross_species(DVR_BASES[s2], _coulomb)
        G_DICT[(s1, s2)] = jnp.array(g, dtype=H_DTYPE)


# ── Schedule ──────────────────────────────────────────────────────────────────

TAU_RAMP = 3.0 / DELTA_2
TAU_HOLD = float(np.pi) / DELTA_2
T_RAMP_UP_END = TAU_RAMP
T_HOLD_END = T_RAMP_UP_END + TAU_HOLD
T_TOTAL = T_HOLD_END + TAU_RAMP

# Erf-edged top-hat: argument is ±3 at each ramp's endpoints.
_ERF_SCALE = 6.0 / TAU_RAMP
_RAMP_UP_MID = 0.5 * TAU_RAMP
_RAMP_DOWN_MID = T_HOLD_END + 0.5 * TAU_RAMP


def envelope(t):
    """0 outside [0, T_TOTAL], rises smoothly to 1 during the hold."""
    return 0.5 * (
        erf(_ERF_SCALE * (t - _RAMP_UP_MID))
        - erf(_ERF_SCALE * (t - _RAMP_DOWN_MID))
    )


def x_of_t_ghz(t):
    return x_star_1_ghz + (x_star_2_ghz - x_star_1_ghz) * envelope(t)


# ── Time-dependent h_list (one closure per species) ──────────────────────────


def _make_h_s(s: int):
    centre = WELL_CENTERS[s]
    x_s = X_S[s]
    T_s = T_S[s]

    def h_s_t(t):
        x_ghz = x_of_t_ghz(t)
        a = (x_ghz[s] / TYPICAL_FREQ_GHZ).astype(H_DTYPE)
        b = (x_ghz[NUM_WELLS + s] / TYPICAL_FREQ_GHZ).astype(H_DTYPE)
        dx = x_s - centre
        v = 0.5 * a ** 2 * dx ** 2 + (b * a ** 2 / 3.0) * dx ** 4
        return T_s + jnp.diag(v)

    return h_s_t


h_list = [_make_h_s(s) for s in range(NUM_WELLS)]


# ── Solver ────────────────────────────────────────────────────────────────────

mctdhf = MCTDHF_n_species_DVR(
    h_list=h_list,
    g_dict=G_DICT,
    num_particles=[1] * NUM_WELLS,
    num_mctdhf_orbitals=[NUM_ORBITALS] * NUM_WELLS,
    num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
    num_states=1,
    imag_time=False,
)


# ── Initial state: state-index 1 (|1000>) from anchor 1 ──────────────────────

C_anchor, bs_anchor = unpack_y_to_Cb(
    y_1_packed,
    NUM_STATES_ANCHOR,
    [1] * NUM_WELLS,
    [NUM_DVR] * NUM_WELLS,
    [NUM_ORBITALS] * NUM_WELLS,
)

# complex64 bulk state for the mixed-precision CMF run (see H_DTYPE).
C_init = jnp.array(C_anchor[1:2], dtype=jnp.complex64)
bs_init = [jnp.array(b, dtype=jnp.complex64) for b in bs_anchor]


# ── Integrator and propagation ───────────────────────────────────────────────

DT = 1e-2
NUM_NEWTON = 5
MIN_SOLVER_TOL = 1e-5
MAX_SOLVER_TOL = 1e-2
MAX_SOLVER_ITER = 20

# ── Configurations (qubit basis index map) ───────────────────────────────────

CONFIGS = iter_bitstrings(NUM_WELLS, 2)  # 16 bitstrings; index 1 = |1000>, 2 = |0100>


# ── Parametric schedule (for QOCT) ───────────────────────────────────────────
#
# Same erf-edged top-hat as the module-level `envelope` (line ~128), but with
# `tau_ramp` and `tau_hold` exposed as JAX-traced parameters. Used by
# `propagate(tau_ramp, tau_hold)` below, which is jit-stable across optimizer
# updates of these scalars.

TAU_RAMP_DEFAULT = float(TAU_RAMP)
TAU_HOLD_DEFAULT = float(TAU_HOLD)

# Worst-case schedule duration → pinned `NUM_STEPS_MAX` so `lax.scan` length
# is a Python int (no recompile per optimizer step). Inside the scan, real
# integrator calls are gated by `t < 2·tau_ramp + tau_hold` via `lax.cond`,
# so slack steps are no-ops.
TAU_RAMP_MAX = 5.0 / DELTA_2
TAU_HOLD_MAX = float(np.pi) / DELTA_2
T_TOTAL_MAX = 2.0 * TAU_RAMP_MAX + TAU_HOLD_MAX
NUM_STEPS_MAX = int(np.ceil(T_TOTAL_MAX / DT))


def _envelope_param(t, tau_ramp, tau_hold):
    erf_scale = 6.0 / tau_ramp
    up_mid = 0.5 * tau_ramp
    down_mid = tau_ramp + tau_hold + 0.5 * tau_ramp
    return 0.5 * (
        erf(erf_scale * (t - up_mid)) - erf(erf_scale * (t - down_mid))
    )


def _x_of_t_ghz_param(t, tau_ramp, tau_hold):
    return x_star_1_ghz + (x_star_2_ghz - x_star_1_ghz) * _envelope_param(
        t, tau_ramp, tau_hold
    )


def _make_h_s_param(s, tau_ramp, tau_hold):
    centre = WELL_CENTERS[s]
    x_s = X_S[s]
    T_s = T_S[s]

    def h_s_t(t):
        x_ghz = _x_of_t_ghz_param(t, tau_ramp, tau_hold)
        a = (x_ghz[s] / TYPICAL_FREQ_GHZ).astype(H_DTYPE)
        b = (x_ghz[NUM_WELLS + s] / TYPICAL_FREQ_GHZ).astype(H_DTYPE)
        dx = x_s - centre
        v = 0.5 * a ** 2 * dx ** 2 + (b * a ** 2 / 3.0) * dx ** 4
        return T_s + jnp.diag(v)

    return h_s_t


Y_INIT = pack_Cb_to_y(C_init, bs_init)


def _propagate_window(tau_ramp, tau_hold, total_time):
    """VMF scan of `Y_INIT` over the parametric erf-top-hat schedule,
    integrating only while `t < total_time`; slack steps are no-ops via
    `lax.cond`. Shared body of `propagate` / `propagate_rampup`.
    """
    h_list_param = [
        _make_h_s_param(s, tau_ramp, tau_hold) for s in range(NUM_WELLS)
    ]
    mctdh_local = MCTDHF_n_species_DVR(
        h_list=h_list_param,
        g_dict=G_DICT,
        num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORBITALS] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=1,
        imag_time=False,
    )
    integrator_local = get_vmf_integrator(
        mctdh_local,
        DT,
        num_newton=NUM_NEWTON,
        min_solver_tol=MIN_SOLVER_TOL,
        max_solver_tol=MAX_SOLVER_TOL,
        max_solver_iter=MAX_SOLVER_ITER,
        precondition=True,
    )
    # integrator_local = get_cmf_half_step_integrator(
    #     mctdh_local,
    #     DT,
    #     c_propagator="diagonalize",
    #     b_propagator="diagonalize",
    #     # Mixed precision: complex64 bulk, orbital expm promoted to x64.
    #     b_kwargs={"orbital_x64": True},
    # )

    @jax.checkpoint  # rematerialize forward during backward — bounds memory
    def step(carry, _):
        t, y = carry
        new_carry = jax.lax.cond(
            t < total_time,
            lambda: integrator_local(t, y),
            lambda: (t + DT, y),
        )
        return new_carry, None

    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    (_, y_T), _ = jax.lax.scan(step, (t0, Y_INIT), None, length=NUM_STEPS_MAX)
    return y_T


@jax.jit
def propagate(tau_ramp, tau_hold):
    """Full 1→2→1 schedule: scan `NUM_STEPS_MAX` steps, integrator active
    while `t < 2·tau_ramp + tau_hold`. Returns the final packed `y_T`.
    See `_propagate_window` for the CMF / autodiff rationale.
    """
    return _propagate_window(tau_ramp, tau_hold, 2.0 * tau_ramp + tau_hold)


@jax.jit
def propagate_rampup(tau_ramp, tau_hold):
    """Ramp-up only: integrator active while `t < tau_ramp` (≈1/3 of the
    full schedule). A cheaper trajectory for an initial AD-vs-FD gradient
    check on the CMF path — not a physically complete swap.
    """
    return _propagate_window(tau_ramp, tau_hold, tau_ramp)


# ── Baseline: the fixed 1→2→1 full-swap propagation ──────────────────────────

if __name__ == "__main__":
    num_steps = int(np.ceil(T_TOTAL / DT))
    save_every = max(1, num_steps // 100)

    integrator = get_vmf_integrator(
        mctdhf,
        DT,
        num_newton=NUM_NEWTON,
        min_solver_tol=MIN_SOLVER_TOL,
        max_solver_tol=MAX_SOLVER_TOL,
        max_solver_iter=MAX_SOLVER_ITER,
    )

    print(
        f"Schedule: T_total={T_TOTAL:.4f}, tau_ramp={TAU_RAMP:.4f}, "
        f"tau_hold={TAU_HOLD:.4f}, Delta_2={DELTA_2:.4f}"
    )
    print(
        f"VMF: dt={DT}, num_steps={num_steps}, num_newton={NUM_NEWTON}, "
        f"solver_tol=[{MIN_SOLVER_TOL},{MAX_SOLVER_TOL}], "
        f"max_solver_iter={MAX_SOLVER_ITER}",
        flush=True,
    )

    t0 = time.perf_counter()
    ts, ys = mctdhf.integrate(
        integrator=integrator,
        num_steps=num_steps,
        t_init=0.0,
        normalize=False,
        C_init=C_init,
        bs_init_list=bs_init,
        noise_bs=0.0,
        noise_C=0.0,
        save_every=save_every,
    )
    jax.block_until_ready(ys)
    wall = time.perf_counter() - t0
    print(f"Propagation finished in {wall:.1f}s ({len(ts)} saved snapshots)")

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def _project_onto_anchor1_qubit_basis(C_t, bs_t):
        """Returns shape-(16,) complex overlaps <phi_n(anchor 1) | psi(t)>."""
        overlaps = np.empty(len(CONFIGS), dtype=np.complex128)
        for k, n in enumerate(CONFIGS):
            u_list = [bs_init[s][:, int(n[s])] for s in range(NUM_WELLS)]
            overlaps[k] = complex(hartree_overlap(C_t[0], bs_t, u_list))
        return overlaps

    energies_t = np.empty(len(ts))
    bitstring_overlaps = np.empty((len(ts), len(CONFIGS)), dtype=np.complex128)
    for i, (t, y) in enumerate(zip(ts, ys)):
        energies_t[i] = float(np.real(mctdhf.calculate_energy(t, y)[0]))
        C_t, bs_t = mctdhf._y_to_Cb(y)
        bitstring_overlaps[i] = _project_onto_anchor1_qubit_basis(C_t, bs_t)

    probs = np.abs(bitstring_overlaps) ** 2
    leakage = 1.0 - probs.sum(axis=1)

    print(
        f"Initial |<|1000>|psi(0)>|^2 = {probs[0, 1]:.4f}, "
        f"leakage(0) = {leakage[0]:.2e}"
    )
    print(
        f"Final   |<|0100>|psi(T)>|^2 = {probs[-1, 2]:.4f}, "
        f"|<|1000>|psi(T)>|^2 = {probs[-1, 1]:.4f}, "
        f"leakage(T) = {leakage[-1]:.2e}"
    )
    print(f"Energy drift |E(T) - E(0)| = {abs(energies_t[-1] - energies_t[0]):.2e}")

    # ── Save ─────────────────────────────────────────────────────────────────

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, "RTE_trajectories", GEOM_TAG)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rte_step_1to2to1_{stamp}.npz")
    np.savez(
        out_path,
        ts=np.asarray(ts),
        ys=np.asarray(ys),
        x_star_1=np.asarray(x_star_1_ghz),
        x_star_2=np.asarray(x_star_2_ghz),
        tau_ramp=TAU_RAMP,
        tau_hold=TAU_HOLD,
        delta_2=DELTA_2,
        dt=DT,
        num_steps=num_steps,
        wall=wall,
        energies_t=energies_t,
        bitstring_overlaps=bitstring_overlaps,
        leakage=leakage,
        configs=np.array(CONFIGS),
    )
    print(f"Saved trajectory to {out_path}")
