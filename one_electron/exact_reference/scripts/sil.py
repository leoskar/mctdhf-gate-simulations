"""Short-iterative-Lanczos propagator on the 64^4 joint grid.

Scheme: exponential midpoint.  Each step freezes H at t + dt/2, builds a fresh
Krylov space of dimension m = 12 with FULL re-orthogonalisation, and applies
exp(-i dt (H - E0)) through the small tridiagonal.

WHY THE STEP IS SET BY THE PULSE, NOT THE PROPAGATOR.  The full spectral width
of H is ~3.5e4, but that is the high-k DVR kinetic tail, which carries no weight
in any physical state; Krylov methods sample only where psi lives, so SIL
converges on the state's own energy spread sigma_E ~ 0.66.  The step is therefore
chosen to resolve the erf ramps -- see step_schedule, which reads the ramp
windows off the envelope derivative instead of hard-coding them.

WHERE THE ENERGY SHIFT IS APPLIED, and why it matters at x32.  A uniform shift
E0 is exactly a global phase, so it cannot touch any relative phase or fidelity.
It is applied to the 12x12 TRIDIAGONAL, in float64, never to H and never to the
state: the accumulated dynamical phase <H>*T ~ 2e5 rad is then never represented
in float32 at all, while the float32 arithmetic is confined to the matvec and the
Krylov vectors.  Folding -E0 into the diagonal of H instead would have made every
Vd entry ~-80 with a 6e-6 ULP, which is strictly worse.

The per-step Krylov residual |beta_m (e^{-i dt T})_{m,0}| is returned so a run
reports how well its own step converged rather than asserting it.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from exact_common import NUM_WELLS, cdot, hpsi

M_KRYLOV = int(__import__("os").environ.get("EXACT_M", "6"))
# Relative floor below which a Lanczos beta is roundoff, not structure.  Scaled
# by max|alpha| (~78 here), so the absolute floor is ~1e-9.
BREAK_TOL = float(__import__("os").environ.get("SIL_BREAK_TOL", "1e-11"))
# Re-orthogonalisation passes.  beta spans five orders within a single step
# (3.6e-4 -> 56), which is exactly where one classical Gram-Schmidt pass loses
# orthogonality; DGKS's rule is "twice is enough", and whether it is needed here
# is a measurement, not a preference.
REORTH = int(__import__("os").environ.get("SIL_REORTH", "1"))


def sil_step(apply_H, v, dt, m, e_shift, cdtype):
    """One exp(-i dt (H - E0)) |v> with H frozen; returns (v_new, err_est).

    STRUCTURED AFTER mctdhf.integrators._sil_solve_C, deliberately: the Krylov
    basis is a preallocated (m, dim) array carried through a lax.scan and
    re-orthogonalised with a single matmul, rather than a Python list in an
    unrolled loop.  The two are mathematically identical -- diag_vs_codebase_sil
    pins them against each other and against a dense expm at 1e-14 -- but the
    stacked form is what makes m = 12 affordable: the unrolled version asked XLA
    for a single 6 GiB allocation, against 3.2 GB for the basis itself.

    apply_H is a callable so the Lanczos is decoupled from the Hamiltonian,
    which is what lets the 256-dim CI operator drive this same function.
    """
    shape = v.shape
    vf = v.reshape(-1)
    dim = vf.shape[0]
    nrm = jnp.sqrt(jnp.real(cdot(vf, vf)))
    v0 = vf / nrm.astype(cdtype)
    V_init = jnp.zeros((m, dim), dtype=cdtype)

    def lanczos_step(carry, j):
        V, v_prev, v_curr, beta_prev, scale = carry
        V = V.at[j].set(v_curr)
        w = apply_H(v_curr.reshape(shape)).reshape(-1)
        alpha = jnp.real(cdot(v_curr, w))
        w = w - alpha.astype(cdtype) * v_curr - beta_prev.astype(cdtype) * v_prev
        # Full re-orthogonalisation against the whole basis in one matmul.  Rows
        # beyond j are still zero, so their overlaps vanish and no masking is
        # needed.  Accumulated in complex128 whatever the state dtype, for the
        # same reason cdot is (see exact_common.cdot).
        ov = jnp.matmul(V.conj(), w, preferred_element_type=jnp.complex128)
        w = w - jnp.matmul(ov.astype(cdtype), V)
        beta = jnp.sqrt(jnp.real(cdot(w, w)))
        # HAPPY BREAKDOWN.  A beta at roundoff level means the Krylov space has
        # closed; normalising it would manufacture a unit vector of pure
        # rounding noise, spectrally flat and therefore sampling the full grid
        # bandwidth.  Zeroing it makes T block-diagonal, so e^{-i dt T} e_0 has
        # no support beyond the live block and the noise contributes nothing.
        scale = jnp.maximum(scale, jnp.abs(alpha))
        live = beta > BREAK_TOL * scale
        beta = jnp.where(live, beta, 0.0)
        v_next = jnp.where(live, w / jnp.where(live, beta, 1.0).astype(cdtype),
                           jnp.zeros_like(w))
        return (V, v_curr, v_next, beta, scale), (alpha, beta)

    init = (V_init, jnp.zeros_like(v0), v0, jnp.asarray(0.0, jnp.float64),
            jnp.asarray(0.0, jnp.float64))
    (V_fin, _, _, _, _), (al, be) = jax.lax.scan(
        lanczos_step, init, jnp.arange(m), length=m)

    al = al.astype(jnp.float64)
    be = be.astype(jnp.float64)
    T = (jnp.diag(al) + jnp.diag(be[:m - 1], 1) + jnp.diag(be[:m - 1], -1)
         - e_shift * jnp.eye(m, dtype=jnp.float64))
    wv, Z = jnp.linalg.eigh(T)
    col0 = Z @ (jnp.exp(-1j * dt * wv) * Z[0, :])
    v_new = nrm.astype(cdtype) * jnp.matmul(col0.astype(cdtype), V_fin)
    err = nrm * jnp.abs(be[m - 1] * col0[m - 1])
    return v_new.reshape(shape), jnp.real(err)


def make_segment(h_fns, Vd, dt, n_steps, e_shift, cdtype, rdtype, m=None):
    """Jitted scan of n_steps midpoint-SIL steps at fixed dt."""
    m = m or M_KRYLOV
    dt = jnp.asarray(dt, jnp.float64)

    def body(carry, _):
        t, v, emax = carry
        tm = t + 0.5 * dt
        hs = [h_fns[s](tm).astype(rdtype) for s in range(NUM_WELLS)]
        v2, e = sil_step(lambda x: hpsi(hs, Vd, x), v, dt, m, e_shift, cdtype)
        return (t + dt, v2, jnp.maximum(emax, e)), None

    @jax.jit
    def run(t0, v):
        (tf, vf, emax), _ = jax.lax.scan(
            body, (jnp.asarray(t0, jnp.float64), v, jnp.asarray(0.0, jnp.float64)),
            None, length=n_steps)
        return tf, vf, emax

    return run


# ── step schedule ────────────────────────────────────────────────────────────

def envelope_deriv(t, tau_ramp, tau_hold):
    """d/dt of the erf top-hat used by the MCTDH h_s(t)."""
    k = 6.0 / tau_ramp
    m1 = 0.5 * tau_ramp
    m2 = tau_ramp + tau_hold + 0.5 * tau_ramp
    return (k / np.sqrt(np.pi)) * (np.exp(-(k * (t - m1)) ** 2)
                                   - np.exp(-(k * (t - m2)) ** 2))


def step_schedule(taus_per_well, dt_fine=0.5, dt_coarse=5.0, env_tol=1e-7):
    """Segment the leg into (t0, t1, dt, n_steps) with dt set by the RAMPS.

    A step is 'fine' wherever a COARSE step would move the envelope by more than
    env_tol -- i.e. the ramp windows are read off |de/dt| rather than assumed to
    be [0, tau_ramp].  With the erf top-hat at +/-3 sigma on the nominal ramp,
    the true window is ~ +/-0.6 tau_ramp about each ramp midpoint, which a
    hard-coded [0, tau_ramp] would clip.

    Boundaries are then adjusted so every segment holds an integer number of
    steps and the segment ends land exactly on the leg endpoints.
    """
    T = max(2 * tr + th for tr, th in taus_per_well.values())
    grid = np.linspace(0.0, T, 20001)
    fine = np.zeros_like(grid, dtype=bool)
    for (tr, th) in taus_per_well.values():
        fine |= np.abs(envelope_deriv(grid, tr, th)) * dt_coarse > env_tol
    # a well that has already finished its own leg sits at idle -- flat, coarse
    edges = [0]
    for i in range(1, len(grid)):
        if fine[i] != fine[i - 1]:
            edges.append(i)
    edges.append(len(grid) - 1)

    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        t0, t1 = float(grid[a]), float(grid[b])
        if t1 - t0 <= 0:
            continue
        dt_t = dt_fine if fine[a] else dt_coarse
        n = int(np.ceil((t1 - t0) / dt_t))
        segs.append((t0, t1, (t1 - t0) / n, n))
    # snap the last segment end onto T exactly
    t0, t1, dt_t, n = segs[-1]
    segs[-1] = (t0, T, (T - t0) / n, n)
    return segs, T


def describe(segs):
    tot = sum(n for *_, n in segs)
    return (f"{len(segs)} segments, {tot} steps  "
            + "  ".join(f"[{a:.1f},{b:.1f}) dt={d:.3f} n={n}" for a, b, d, n in segs))
