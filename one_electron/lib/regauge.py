"""Exact gauge re-orthonormalisation of the MCTDH orbitals.

MANDATORY with the CMF half-step integrator; a no-op-in-spirit with VMF.

WHY. `b_propagator` propagates a FROZEN-projector substep db/dt = A_s b whose
generator is non-Hermitian (asymmetry ratio ~0.5 under the (1-P)(h+W)D^-1
gauge). exp(-i dt A_s) is therefore NOT unitary, so b^H b = I is only a WEAK
invariant: preserved on the manifold, not off it. The violation is O(dt^2) per
step and accumulates LINEARLY in T, which is why it is invisible on short
screens and fatal on long legs. Measured on the w2w3 leg (1065 t.u., 2dt):

                        drift        ortho      b singular values
    without regauge     9.91e-01     1.19       0.011 .. 1.807
    with    regauge     4.91e-08     6.66e-16   1.000

||C|| was exactly 1.0000 in BOTH cases -- the CI substep (dense eigh) is exact.
Every failure of this kind is orbital.

WHAT IT IS. MCTDH has an exact gauge freedom: (b_s T_s, C x_s T_s^-1) represents
the same Psi. Taking T_s from the QR factorisation returns the orbitals to the
orthonormal gauge in which the projector EOM was derived, WITHOUT touching the
physical state (verified: Psi changes by 8.9e-16, norms unchanged). It costs
nothing measurable (9.03 vs 9.09 s/t.u.).

WHAT IT IS NOT. This is not the solver's built-in normalize/orthonormalize
modes. On a real checkpoint those change Psi by 2.6% and force the norms to
1.0000000 -- destroying the very diagnostic that reveals the problem.
`ortonorm_b` replaces b by Q WITHOUT compensating C by R; `ortonorm_C`
orthonormalises the 4/16 STATES against each other, which is right for
imaginary time and would destroy the mutual overlaps that ARE the gate in real
time. Both correctly default to off in real time. Do not substitute them.

CAVEAT. Projecting out gauge drift removes a warning light, so the norm must
still be REPORTED, never forced. Residual drift after this fix is genuine
integration error -- the EOM conserves <Psi|Psi> exactly in the orthonormal
gauge.

run_direct_sa_gauge.make_regauge is the same function.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def make_regauge(solver):
    """b_s = Q_s R_s -> (Q_s, R_s x_s C), with the positive-diagonal convention.

    The sign fix makes Q, R unique, so the map is deterministic and the run
    reproducible. Apply once per checkpoint chunk, not once per step.
    """
    @jax.jit
    def regauge(y):
        C, bs = solver._y_to_Cb(y)
        bs_new = []
        for s, b in enumerate(bs):
            Q, R = jnp.linalg.qr(b)
            sgn = jnp.sign(jnp.real(jnp.diag(R)))
            sgn = jnp.where(sgn == 0, 1.0, sgn).astype(R.dtype)
            Q = Q * sgn[None, :]          # Q' = Q S
            R = sgn[:, None] * R          # R' = S R, so Q'R' = QR = b exactly
            bs_new.append(Q)
            # phi_m = sum_k R'[k,m] chi_k  =>  C'_k = sum_m R'[k,m] C_m
            C = jnp.moveaxis(
                jnp.tensordot(R, jnp.moveaxis(C, s + 1, 0), axes=([1], [0])),
                0, s + 1)
        return solver._Cb_to_y(C, bs_new)
    return regauge
