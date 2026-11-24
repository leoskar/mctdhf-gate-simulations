"""Two-well FCI on the grid: Hamiltonian, configurations and TD-FCI propagation.

Exact (FCI-on-the-grid) real-time propagation of an isolated two-well √iSWAP,
on the full product wavefunction ``psi[i, j]`` over two 64-point sine-DVR grids
(4096-dim, no orbital truncation).  grid_lib.py and run_dvr_convergence.py
(App. B 3) use its N-independent constants and configurations.

  * The four-well operating points cannot be reduced to two wells: their
    near-degeneracy (DELTA_2) depends on the Coulomb shifts from all four
    wells.  The two-well (alpha, beta) configuration is therefore optimized
    separately (two_well_config_geom_2um.npz).

  * The qubit subspace is the span of the lowest four eigenstates of the
    interacting grid Hamiltonian at the idle configuration, not single-well
    product states: the Coulomb energy (≈35 against motional gaps ~1)
    dresses the eigenstates heavily.  Ordering: psi0=|00>, psi1=|10>,
    psi2=|01>, psi3=|11>.

  * Gate protocol: erf-edged top-hat ramp idle -> resonant -> idle (same form
    as rte_step.py); near the resonant hold the two single-excitation states
    exchange (√iSWAP at tau_hold ≈ pi/(2*DELTA)).

  * Propagator: SIL / Krylov (short iterative Lanczos), as
    `mctdhf/integrators.py:_sil_solve_C`, midpoint H(t+dt/2), complex128.

A configuration is the pair (x_idle, x_res), each a 4-vector
[alpha_0, alpha_1, beta_0, beta_1] in GHz. `make_well_potential` is the single
source of truth for the time-dependent potential, shared by the FCI Hvp here
and the MCTDH `h_list` (via `make_h_list`).
"""

from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf

import pint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))   # geometry_config
sys.path.insert(0, HERE)

from mctdhf import SineDVR
from geometry_config import TWO_WELL_CENTERS_UM, GEOM_TAG


# ── Physical constants (mirror rte_step.py) ───────────────────────────────────

NUM_DVR = 64
NUM_WELLS = 2
NUM_ORBITALS = 4  # MCTDH orbital budget per well (production setting)
TYPICAL_FREQ_GHZ = 10.0

_ureg = pint.UnitRegistry()
_omega = 2 * np.pi * TYPICAL_FREQ_GHZ * _ureg.GHz
_length_unit = np.sqrt(_ureg.hbar / (_ureg.m_e * _omega))
LENGTH_UNIT_UM = _length_unit.to(_ureg.micrometer).magnitude
KAPPA = (
    _length_unit * _ureg.m_e * _ureg.e ** 2
    / (4 * _ureg.pi * _ureg.epsilon_0 * _ureg.hbar ** 2)
).to(_ureg.dimensionless).magnitude

# Two adjacent wells = the inner adjacent pair of the 4-well chain, at the active
# geometry's spacing (from geometry_config). Centres are an arbitrary x-origin
# choice (local potentials + separation-only Coulomb).
WELL_CENTERS = jnp.asarray(TWO_WELL_CENTERS_UM / LENGTH_UNIT_UM)  # float64
WELL_WIDTHS = np.array([1.0, 1.0]) / LENGTH_UNIT_UM


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

_zero_pot = lambda x: jnp.zeros_like(x)
T_S = [jnp.asarray(DVR_BASES[s].get_h(potential=_zero_pot)) for s in range(NUM_WELLS)]
X_S = [jnp.asarray(DVR_BASES[s].get_grid()) for s in range(NUM_WELLS)]
U_01 = jnp.asarray(DVR_BASES[0].get_g_cross_species(DVR_BASES[1], _coulomb))


# ── Configuration (two_well_config_<GEOM_TAG>.npz) ────────────────────────────
#
# x = [alpha_0, alpha_1, beta_0, beta_1] in GHz. Idle: detuned (localized
# single-excitation eigenstates define the computational basis). Resonant:
# identical wells (the swap point). The hand-picked fallbacks are only used if
# two_well_config.npz is absent.
_CONFIG_PATH = os.path.join(HERE, f"two_well_config_{GEOM_TAG}.npz")
if os.path.isfile(_CONFIG_PATH):
    _cfg = np.load(_CONFIG_PATH)
    X_IDLE_DEFAULT = jnp.asarray(_cfg["x_idle"], dtype=jnp.float64)
    X_RES_DEFAULT = jnp.asarray(_cfg["x_res"], dtype=jnp.float64)
else:
    import warnings
    warnings.warn(
        f"two_well_config_{GEOM_TAG}.npz not found; falling back to hand-picked "
        f"X_IDLE/X_RES of the 1 um geometry, which are wrong for this "
        f"geometry.",
        stacklevel=2,
    )
    X_IDLE_DEFAULT = jnp.array([9.0, 11.0, 0.25, 0.25], dtype=jnp.float64)
    X_RES_DEFAULT = jnp.array([10.0, 10.0, 0.25, 0.25], dtype=jnp.float64)


def _well_v(s, x4, t_arr=None):
    """Diagonal potential on grid s for config 4-vector x4 = [a0,a1,b0,b1] GHz."""
    a = x4[s] / TYPICAL_FREQ_GHZ
    b = x4[2 + s] / TYPICAL_FREQ_GHZ
    dx = X_S[s] - WELL_CENTERS[s]
    return 0.5 * a ** 2 * dx ** 2 + (b * a ** 2 / 3.0) * dx ** 4


# ── Schedule (erf-edged top-hat, mirrors rte_step.py) ──────────────────────────


def _envelope(t, tau_ramp, tau_hold):
    erf_scale = 6.0 / tau_ramp
    up_mid = 0.5 * tau_ramp
    down_mid = tau_ramp + tau_hold + 0.5 * tau_ramp
    return 0.5 * (erf(erf_scale * (t - up_mid)) - erf(erf_scale * (t - down_mid)))


def total_time(tau_ramp, tau_hold):
    return 2.0 * tau_ramp + tau_hold


def make_well_potential(s, x_idle, x_res, tau_ramp, tau_hold):
    """v_s(t): diagonal potential on grid s, ramping x_idle -> x_res -> x_idle.
    Single source of truth, shared by the FCI Hvp and the MCTDH h_list."""
    a_i, a_r = x_idle[s] / TYPICAL_FREQ_GHZ, x_res[s] / TYPICAL_FREQ_GHZ
    b_i, b_r = x_idle[2 + s] / TYPICAL_FREQ_GHZ, x_res[2 + s] / TYPICAL_FREQ_GHZ
    dx = X_S[s] - WELL_CENTERS[s]

    def v_of_t(t):
        e = _envelope(t, tau_ramp, tau_hold)
        a = a_i + (a_r - a_i) * e
        b = b_i + (b_r - b_i) * e
        return 0.5 * a ** 2 * dx ** 2 + (b * a ** 2 / 3.0) * dx ** 4

    return v_of_t


def make_h_list(x_idle, x_res, tau_ramp, tau_hold):
    """Time-dependent single-well Hamiltonians h_s(t) = T_s + diag(v_s(t)) for
    the MCTDH solver (list of callables)."""
    vs = [make_well_potential(s, x_idle, x_res, tau_ramp, tau_hold) for s in range(NUM_WELLS)]

    def make_h(s):
        T_s = T_S[s]
        v = vs[s]
        return lambda t: T_s + jnp.diag(v(t))

    return [make_h(s) for s in range(NUM_WELLS)]


# ── Grid Hamiltonian (matrix-vector product and dense static builder) ──────────


def make_hvp(x_idle, x_res, tau_ramp, tau_hold):
    """Time-dependent H(t) psi on the (NUM_DVR, NUM_DVR) product grid."""
    v0 = make_well_potential(0, x_idle, x_res, tau_ramp, tau_hold)
    v1 = make_well_potential(1, x_idle, x_res, tau_ramp, tau_hold)
    T0, T1 = T_S[0], T_S[1]

    def hvp(t, psi):
        return (
            T0 @ psi
            + psi @ T1
            + v0(t)[:, None] * psi
            + v1(t)[None, :] * psi
            + U_01 * psi
        )

    return hvp


def static_H_dense(x4):
    """Dense (NUM_DVR^2, NUM_DVR^2) Hamiltonian for the static config x4."""
    N = NUM_DVR
    I = np.eye(N)
    T0 = np.asarray(T_S[0])
    T1 = np.asarray(T_S[1])
    U = np.asarray(U_01)
    v0 = np.asarray(_well_v(0, x4))
    v1 = np.asarray(_well_v(1, x4))
    return (
        np.kron(T0, I)
        + np.kron(I, T1)
        + np.kron(np.diag(v0), I)
        + np.kron(I, np.diag(v1))
        + np.diag(U.ravel())
    )


def eigenbasis(x4, k=6):
    """Lowest k eigenstates of the interacting grid H at static config x4.

    Returns (eigvals (k,), eigvecs (k, NUM_DVR, NUM_DVR) complex128). The
    lowest 4 are the computational basis: psi0=|00>, psi1=|10>, psi2=|01>,
    psi3=|11>.
    """
    H = static_H_dense(x4)
    w, V = np.linalg.eigh(H)
    vecs = V[:, :k].T.reshape(k, NUM_DVR, NUM_DVR)
    return w[:k], jnp.asarray(vecs, dtype=jnp.complex128)


def swap_gap(x_res):
    """DELTA = E2 - E1 at the resonant config (the swap splitting)."""
    w, _ = eigenbasis(x_res, k=3)
    return float(w[2] - w[1])


# ── SIL / Krylov propagator (mirrors integrators.py:_sil_solve_C, RTE only) ────


def _sil_propagate_single(apply_H, c_flat, dt, krylov_dim):
    """exp(-i dt H) c via short iterative Lanczos with full re-orthogonalization."""
    dim = c_flat.shape[0]
    norm_c = jnp.linalg.norm(c_flat)
    safe_norm = jnp.where(norm_c > 0, norm_c, 1.0)
    v0 = c_flat / safe_norm

    vec_dtype = c_flat.dtype
    real_dtype = c_flat.real.dtype
    V_init = jnp.zeros((krylov_dim, dim), dtype=vec_dtype)

    def lanczos_step(carry, j):
        V, v_prev, v_curr, beta_prev = carry
        V = V.at[j].set(v_curr)
        w = apply_H(v_curr).astype(vec_dtype)
        alpha = jnp.real(jnp.dot(v_curr.conj(), w)).astype(real_dtype)
        w = w - alpha * v_curr - beta_prev * v_prev
        overlaps = jnp.dot(V.conj(), w)
        w = w - jnp.dot(overlaps, V)
        beta = jnp.sqrt(jnp.real(jnp.dot(w.conj(), w))).astype(real_dtype)
        safe_beta = jnp.where(beta > 1e-15, beta, 1.0)
        v_next = w / safe_beta
        return (V, v_curr, v_next, beta), (alpha, beta)

    init = (V_init, jnp.zeros_like(v0), v0, jnp.array(0.0, dtype=real_dtype))
    (V_final, _, _, _), (alphas, betas) = jax.lax.scan(
        lanczos_step, init, jnp.arange(krylov_dim), length=krylov_dim
    )

    T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
    expT = jax.scipy.linalg.expm(-1j * dt * T)
    e1 = jnp.zeros(krylov_dim, dtype=expT.dtype).at[0].set(1.0)
    coeffs = expT @ e1
    c_new = safe_norm * jnp.dot(coeffs, V_final)
    return jnp.where(norm_c > 0, c_new, c_flat)


def make_sil_step(hvp, dt, krylov_dim=12):
    """Single SIL step (t, psi) -> psi(t+dt) with midpoint H(t+dt/2)."""
    shape = (NUM_DVR, NUM_DVR)

    def step(t, psi):
        apply_H = lambda v: hvp(t + 0.5 * dt, v.reshape(shape)).reshape(-1)
        c_new = _sil_propagate_single(apply_H, psi.reshape(-1), dt, krylov_dim)
        return c_new.reshape(shape)

    return step


# ── Gate-fidelity helpers ────────────────────────────────────────────────────


def gate_matrix(basis, psi_final):
    """M[i, j] = <basis_i | psi_j(T)>, shape (K, K)."""
    return jnp.einsum("inm,jnm->ij", jnp.conj(basis), psi_final)


def avg_gate_fidelity(M):
    """F = (Tr(M M^dag) + |Tr M|^2) / [K(K+1)]."""
    K = M.shape[0]
    return float(
        (jnp.real(jnp.trace(M @ M.conj().T)) + jnp.abs(jnp.trace(M)) ** 2)
        / (K * (K + 1))
    )


# ── Driver ─────────────────────────────────────────────────────────────────────


def run_fci(
    x_idle=X_IDLE_DEFAULT,
    x_res=X_RES_DEFAULT,
    tau_ramp=2.0,
    tau_hold=None,
    dt=2.5e-3,
    krylov_dim=12,
):
    """Exact-grid propagation of the K=4 idle-config eigenstates through the
    erf-ramp gate. tau_hold defaults to pi/(2*DELTA) (√iSWAP)."""
    delta = swap_gap(x_res)
    if tau_hold is None:
        tau_hold = float(np.pi) / (2.0 * delta)

    _, basis = eigenbasis(x_idle, k=4)  # (4, N, N) — initial states & projection basis
    hvp = make_hvp(x_idle, x_res, tau_ramp, tau_hold)
    step = make_sil_step(hvp, dt, krylov_dim)

    T_tot = total_time(tau_ramp, tau_hold)
    num_steps = int(np.ceil(T_tot / dt))
    batched_step = jax.vmap(step, in_axes=(None, 0))

    IDX_10, IDX_01 = 1, 2

    def scan_body(carry, _):
        t, psi_b = carry
        psi_b = batched_step(t, psi_b)
        t_new = t + dt
        swap = psi_b[IDX_10]  # the |10> -> trajectory
        ov = jnp.einsum("knm,nm->k", jnp.conj(basis), swap)  # (4,) overlaps
        p_sub = jnp.abs(ov) ** 2
        leak = 1.0 - jnp.sum(p_sub)
        Hs = hvp(t_new, swap)
        E = jnp.real(jnp.vdot(swap, Hs))
        nrm = jnp.real(jnp.vdot(swap, swap))
        return (t_new, psi_b), (t_new, p_sub[IDX_10], p_sub[IDX_01], leak, E, nrm)

    (t_final, psi_final), (ts, p10, p01, leak, Es, norms) = jax.lax.scan(
        scan_body, (jnp.asarray(0.0), basis), None, length=num_steps
    )

    M = gate_matrix(basis, psi_final)
    return dict(
        M=np.asarray(M),
        F=avg_gate_fidelity(M),
        delta=delta,
        ts=np.asarray(ts),
        p10=np.asarray(p10),
        p01=np.asarray(p01),
        leakage=np.asarray(leak),
        energies=np.asarray(Es),
        norms=np.asarray(norms),
        dt=dt,
        num_steps=num_steps,
        tau_ramp=tau_ramp,
        tau_hold=tau_hold,
        T_total=float(T_tot),
        x_idle=np.asarray(x_idle),
        x_res=np.asarray(x_res),
    )


if __name__ == "__main__":
    print(f"KAPPA={KAPPA:.2f}  LENGTH_UNIT_UM={LENGTH_UNIT_UM:.5f}")
    delta = swap_gap(X_RES_DEFAULT)
    print(f"resonant DELTA (E2-E1) = {delta:.5f}  -> √iSWAP tau_hold = {np.pi/(2*delta):.3f}")
    for tau_ramp in (1.0, 2.0, 4.0):
        res = run_fci(tau_ramp=tau_ramp, dt=5e-3)
        print(
            f"tau_ramp={tau_ramp:4.1f}  steps={res['num_steps']:6d}  F={res['F']:.4f}  "
            f"final p10={res['p10'][-1]:.3f} p01={res['p01'][-1]:.3f} "
            f"max_leak={res['leakage'].max():.3f} "
            f"norm_drift={abs(res['norms'][-1]-1):.1e}"
        )
