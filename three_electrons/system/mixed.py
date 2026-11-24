"""Mixed-precision layer solves for the ITE relaxation: float64 residual,
float32 Krylov solve.

Why pure complex64 cannot work here, and why that is not a verdict on
single precision
---------------------------------------------------------------------

Every register state sits at ~2587 in energy while the quantities the gate
needs are 1.8-4.6 (detunings), 2.5e-2 (2J) and 2.8e-4 (static ZZ). The
float32 ulp at 2587 is 2.4e-4 — the size of ZZ. Every layer RHS is a
near-cancellation of ~2587-scale terms (the projector ``1 - P`` removes the
large diagonal only AFTER it has been computed), so a float32 RHS carries
an absolute noise floor of that order. A relaxation driven by that RHS has
no fixed point finer than the floor: it random-walks there (measured: d01
error 1.2e-3 at 3000 iterations growing to 1.4e-3 at 12000).

A plain c64 run has a second, independent problem: with the x64 setting
``solver_tol = 1e-12``, which float32 BiCGSTAB can never reach, every solve
runs to ``max_solver_iter = 400``, so c64 is SLOWER than x64 (0.106 against
0.053 s/iteration).

The fix: defect correction
--------------------------

Write the CN step for the INCREMENT rather than the new state:

    x1 = x0 + d,     (I - dt/2 L) d = dt L x0

This is algebraically identical to ``_cn_linear``'s
``(I - dt/2 L) x1 = (I + dt/2 L) x0``. Now split the precisions:

* the right-hand side ``dt L x0`` — the RESIDUAL — is formed in float64,
  from float64 frozen operators;
* the linear solve for ``d`` runs in float32, with the frozen operators
  cast down, to a RELATIVE tolerance float32 can actually meet (1e-5).

At a fixed point ``L x0 = 0`` exactly (in float64), so ``d = 0`` and the
iteration stops exactly there: **the fixed point is the float64 one**. The
float32 solve only sets how fast we get there, and its relative error
multiplies ``d``, which vanishes at convergence. This is classical
mixed-precision iterative refinement, applied once per CMF substep.

What stays float64: the integral transforms, the frozen operators (built
once per step), the residual, the state itself, QR, the dense top block and
its ``eigh`` (256 x 256 at n_s = 4). What drops to float32: every BiCGSTAB
matvec, which is where the per-step iteration count multiplies the cost.

``cn_defect`` is the generic kernel; ``B_half`` / ``b_full`` are the SPF
and orbital layer solves of ``register.relax`` rebuilt around it. They
follow ``_solve_Bs_subcycled`` (one SPF substep) and ``_cn_solve_bs`` term
for term.

``lo=False`` keeps the defect form but solves in float64 too (the ``x64d``
precision). That is NOT the same as plain CN: plain CN solves for ``x1``, so
its BiCGSTAB error ``tol*|x1|`` does not vanish at convergence and biases
the fixed point — which is why it has to run at ``tol = 1e-12``. The defect
form's error is ``tol*|d|`` and vanishes, so its tolerance can be LOOSE
(1e-2 .. 1e-4) at no cost in the fixed point: it can only change how many
outer iterations convergence takes. That makes the inner tolerance a pure
speed knob, on CPU as well as GPU.
"""

import jax
import jax.numpy as jnp


def lower(tree):
    """Cast every complex128 leaf to complex64 and float64 to float32.

    Integer leaves (determinant / orbital index tables) pass through.
    """
    def cast(x):
        if not hasattr(x, "dtype"):
            return x
        if x.dtype == jnp.complex128:
            return x.astype(jnp.complex64)
        if x.dtype == jnp.float64:
            return x.astype(jnp.float32)
        return x
    return jax.tree_util.tree_map(cast, tree)


def cn_defect(rhs_hi, rhs_lo, x0, dt, tol, maxiter, lo_dtype=jnp.complex64):
    """CN step ``x1 = x0 + d`` with ``(I - dt/2 L_lo) d = dt L_hi x0``.

    ``rhs_hi`` and ``rhs_lo`` are the same linear map, built from float64
    and float32 frozen operators respectively. The residual — the only
    place the answer is decided — is float64; BiCGSTAB runs in float32 on
    the increment, from the explicit-Euler guess ``d = dt L x0``.
    """
    shape = x0.shape
    # Trace-time guard: one float64 leaf left in the frozen operators would
    # silently promote every matvec back to complex128 and the "mixed" arm
    # would measure x64 under another name.
    probe = jax.eval_shape(rhs_lo, jax.ShapeDtypeStruct(shape, lo_dtype))
    if probe.dtype != lo_dtype:
        raise TypeError(f"low-precision RHS returns {probe.dtype}, "
                        f"expected {lo_dtype}")
    r = dt * rhs_hi(x0)                       # float64 residual

    def matvec(d_flat):
        d = d_flat.reshape(shape)
        return (d - (dt / 2) * rhs_lo(d)).reshape(-1)

    r_lo = r.astype(lo_dtype).reshape(-1)
    d_flat, _ = jax.scipy.sparse.linalg.bicgstab(
        matvec, r_lo, x0=r_lo, tol=tol, maxiter=maxiter)
    return x0 + d_flat.reshape(shape).astype(x0.dtype)


def B_half(ml, A, Bs, bs, rho_invs, rho_pairs, dt, tol, maxiter, lo=True):
    """SPF layer over ``dt``: one frozen-projector substep, mixed CN.

    Same structure as ``integrators_ml._solve_Bs_subcycled`` with
    ``n_spf_substeps = 1``: projector and partner SPFs frozen at ``Bs``,
    node densities frozen for the whole step.
    """
    h_current = ml._get_h_current(0.0)
    h_MO, g_MO = ml._ml_integrals(h_current, bs)
    Ops = ml.dCI.ml_species_ops(h_MO, g_MO)
    w = ml.state_weights
    es = [ml.dCI.spf_transition_ops(s, Bs[s]) for s in range(ml.num_species)]

    out = list(Bs)
    for s in range(ml.num_species):
        if ml.spf_is_full[s]:
            continue
        W = ml.dCI.ml_meanfield_W(A, Bs, g_MO, s, es=es, weights=w,
                                  rho_pairs=rho_pairs)
        frozen = (Ops[s], rho_invs[s], W, Bs[s])
        frozen_lo = lower(frozen) if lo else frozen

        def rhs_hi(X, _s=s, _f=frozen):
            return ml._rhs_B_linear(X, _s, *_f)

        def rhs_lo(X, _s=s, _f=frozen_lo):
            return ml._rhs_B_linear(X, _s, *_f)

        out[s] = cn_defect(rhs_hi, rhs_lo, Bs[s], dt, tol, maxiter,
                           jnp.complex64 if lo else Bs[s].dtype)
    return [ml.ortonorm_B(B) for B in out]


def b_full(ml, A, Bs, bs, dt, tol, maxiter, lo=True):
    """Orbital layer over ``dt``: one frozen-projector substep, mixed CN.

    Same structure as ``integrators._cn_solve_bs`` via
    ``_solve_bs_subcycled`` with ``n_orbital_substeps = 1``.
    """
    h_current = ml._get_h_current(0.0)
    es = [ml.dCI.spf_transition_ops(s, Bs[s]) for s in range(ml.num_species)]
    D_invs, ds = ml._compute_full_RDMs(A, Bs, es)
    g2 = ml._precompute_frozen_g2(bs)

    out = list(bs)
    for s in range(ml.num_species):
        frozen = (h_current[s], D_invs[s], ds, bs, g2)
        frozen_lo = lower(frozen) if lo else frozen

        def rhs_hi(x, _s=s, _f=frozen):
            return ml._rhs_b_linear(x, _s, *_f)

        def rhs_lo(x, _s=s, _f=frozen_lo):
            return ml._rhs_b_linear(x, _s, *_f)

        out[s] = cn_defect(rhs_hi, rhs_lo, bs[s], dt, tol, maxiter,
                           jnp.complex64 if lo else bs[s].dtype)
    return [ml.ortonorm_b(b) for b in out]
