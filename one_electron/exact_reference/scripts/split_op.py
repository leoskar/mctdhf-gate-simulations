"""Strang split-operator propagator on the 64^4 joint grid.

REPLACES the SIL scheme (sil.py), on measurement.  In the
two-well reduction against a dense-eigh exact answer, at EQUAL COST:

    SIL  m=12, 960 matvec-equivalents  ->  infidelity 5.5e-01
    Strang,    640 matvec-equivalents  ->  infidelity 2.0e-09

SIL was not converging badly, it was not converging: the grid Hamiltonian has
spectral spread 1307 in two wells (3.5e4 in four), so dt*dE ~ 654 against m = 12
and Krylov was never in range.  Split-operator does not have to resolve that
bandwidth -- it lives entirely inside h_s, which is diagonalised EXACTLY within
each step.

    exp(-i dt H) ~ exp(-i dt V/2) [kron_s exp(-i dt h_s)] exp(-i dt V/2)

WHAT IS SHARED, which the per-state Krylov basis could not be: exp(-i dt h_s) is
four 64x64 matrices per step, identical for every one of the sixteen states -- so
the ENSEMBLE propagates as ONE array of shape (16, 64, 64, 64, 64) and the
propagator is built once per step, not sixteen times.  Be honest about what that
buys: the operator construction is genuinely shared but was never the bottleneck
(four 64x64 eigh ~ 0.5 ms against ~1 matvec of contractions), and the
contractions are irreducibly proportional to state DATA, so total FLOPs are
unchanged.  What batching really removes is structural -- one job and one compile
instead of a sixteen-task array -- and, because there is no Krylov basis, the
whole ensemble is 4.3 GB at x64 / 2.1 GB at x32 and fits on the 8 GB card in a
single pass.  Under SIL that was impossible: 3.2 GB of Krylov basis PER STATE
forced one state at a time.
Each is built from a Hermitian eigendecomposition (exact, stable) rather than a
general expm.  A step therefore costs four axis contractions -- one matvec -- and
needs NO Krylov basis, so the memory ceiling that capped m at 8 on the 8 GB card
is gone entirely.

TIME DEPENDENCE.  h_s is sampled at the step midpoint, which keeps the scheme
second order in dt overall; the splitting error is also second order, so a single
uniform dt controls both.  Unlike SIL there is no reason to vary dt across the
ramp: Strang's error is per-step and roughly uniform in time, so the segmented
fine/coarse schedule is dropped in favour of one dt chosen by halving.

ORDER.  Strang is second order in the state error, which appears as FOURTH order
in infidelity -- the 16x-per-halving seen in the two-well test.  ORDER=4 selects
the Yoshida triple-jump composition (three Strang steps, w1 = 1/(2 - 2^(1/3))),
which costs 3 matvecs per step and is worth it only if the step count at order 2
becomes the binding constraint.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

# _SUBS / _SUBS_B are shared with exact_common.hpsi so the propagator and the
# matvec can never disagree about the ensemble axis -- which is exactly what
# happened when only one of them knew about it.
from exact_common import NUM_WELLS, _SUBS, _SUBS_B

ORDER = int(os.environ.get("SPLIT_ORDER", "2"))
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = 1.0 - 2.0 * _W1


def _axis_props(h_fns, t, dt, rdtype, cdtype):
    """exp(-i dt h_s(t)) for every well, via Hermitian eigendecomposition.

    ALWAYS BUILT IN DOUBLE, cast to the state dtype only at the end.  This is
    the x32 hazard Strang has and SIL did not.  h_s has eigenvalues to ~147, so
    a float32 eigendecomposition carries dl ~ 1e-7 * 147 = 1.5e-5 absolute,
    giving a phase error dt*dl ~ 1.9e-6 rad per step -- which is SYSTEMATIC, not
    random, so over 1902 steps it accumulates coherently to ~3.6e-3 rad and
    would sink the precision gate for a reason that has nothing to do with the
    state representation.  Four 64x64 eigendecompositions cost ~0.5 ms against
    ~0.1 s of contractions, so double precision here is free.

    The same argument does not apply to the state itself: the states are what
    the memory and the bandwidth are spent on, and their rounding is random
    rather than coherent.
    """
    out = []
    for s in range(NUM_WELLS):
        h = h_fns[s](t).astype(jnp.float64)
        w, U = jnp.linalg.eigh(h)
        E = (U.astype(jnp.complex128)
             * jnp.exp(-1j * dt.astype(jnp.float64) * w))  @ U.conj().T.astype(jnp.complex128)
        out.append(E.astype(cdtype))
    return out


def _strang(h_fns, Vd, v, t, dt, rdtype, cdtype):
    """One exp(-i dt V/2) [kron_s exp(-i dt h_s)] exp(-i dt V/2)."""
    # loop-invariant (no t dependence), so XLA hoists it out of the scan; built
    # in double for the same coherent-phase reason as the axis propagators
    half = jnp.exp(-0.5j * dt.astype(jnp.float64)
                   * Vd.astype(jnp.float64)).astype(cdtype)
    subs = _SUBS_B if v.ndim == NUM_WELLS + 1 else _SUBS
    v = half * v                       # Vd broadcasts over the leading axis
    for s, E in enumerate(_axis_props(h_fns, t + 0.5 * dt, dt, rdtype, cdtype)):
        v = jnp.einsum(subs[s], E, v, preferred_element_type=cdtype)
    return half * v


def _step(h_fns, Vd, v, t, dt, rdtype, cdtype):
    if ORDER == 2:
        return _strang(h_fns, Vd, v, t, dt, rdtype, cdtype)
    if ORDER == 4:
        v = _strang(h_fns, Vd, v, t, _W1 * dt, rdtype, cdtype)
        v = _strang(h_fns, Vd, v, t + _W1 * dt, _W0 * dt, rdtype, cdtype)
        return _strang(h_fns, Vd, v, t + (_W1 + _W0) * dt, _W1 * dt,
                       rdtype, cdtype)
    raise SystemExit(f"SPLIT_ORDER must be 2 or 4, got {ORDER}")


def make_segment(h_fns, Vd, dt, n_steps, e_shift, cdtype, rdtype, m=None):
    """Jitted scan of n_steps Strang steps at fixed dt.

    Signature matches sil.make_segment so the drivers can switch propagator
    without changes; `m` is accepted and ignored (there is no Krylov space).
    The uniform shift enters as a pure global phase per step, exactly as it does
    in the SIL tridiagonal -- it never touches H or the state.
    """
    dt = jnp.asarray(dt, jnp.float64)
    ph = jnp.exp(1j * dt * e_shift).astype(cdtype)

    def body(carry, _):
        t, v, emax = carry
        v2 = ph * _step(h_fns, Vd, v, t, dt, rdtype, cdtype)
        return (t + dt, v2, emax), None

    @jax.jit
    def run(t0, v):
        (tf, vf, emax), _ = jax.lax.scan(
            body, (jnp.asarray(t0, jnp.float64), v,
                   jnp.asarray(0.0, jnp.float64)), None, length=n_steps)
        return tf, vf, emax

    return run
