"""The 16 register configurations as EIGENSTATES of the ML-projected H.

The problem this solves
-----------------------

The computational states should be eigenstates. State-averaged imaginary
time converges to the LOWEST ``num_states`` eigenvectors, and in the full
four-well problem the 16 computational configurations are not the lowest 16:

    levels per well below |1111>:  [33, 15, 11, 5]
    states needed to contain all 16 configurations:  269
    intruders below |1111>:                          253

Propagating 269 state-averaged roots of a ``D^4``-dimensional CI problem is
not viable.  A register built from Hartree product configurations instead,
projected onto a shared orbital set and Lowdin-repaired, has states that are
99.71-99.999 % pure, i.e. NOT eigenstates.

Why the ML structure removes the obstacle
-----------------------------------------

The intruder count is a property of ITERATIVE, ENERGY-ORDERED eigensolvers.
It is the number of roots you must chase from the bottom before the ones you
want appear. It is not a property of the states.

ML-MCTDHF's top-layer Hamiltonian is ``prod(n_s) x prod(n_s)`` — 81 at
``n_s = 3``, 256 at ``n_s = 4``, 1296 at ``n_s = 6`` — and ``_diag_solve_C``
already builds it densely for the ``diagonalize`` propagator. A dense
Hermitian matrix of that size can be diagonalised COMPLETELY. Once every
root is in hand, root selection is a MATCHING problem rather than a counting
problem: take the 16 eigenvectors with the largest overlap onto the target
configurations. The 253 intruders are still there, are still below |1111>,
and simply do not have to be traversed.

The object that makes this possible is the same dense top-block
diagonalisation the real-time integrator uses.

What these states are, stated precisely
---------------------------------------

They are exact eigenvectors of H PROJECTED onto the ML variational manifold
``span{ phi^0_i (x) phi^1_j (x) phi^2_k (x) phi^3_l }``, with the SPFs
``phi`` and the orbitals variationally optimised for the 16-state average.
They are not eigenstates of the untruncated H. Three consequences:

* **``M(0) = 1`` is exact by construction**, not by repair. They are
  eigenvectors of a Hermitian matrix, hence orthonormal to machine
  precision. A Hartree-product register needs a Lowdin step because truncating to a
  shared orbital set breaks ``<0|1> = 0`` at the 1e-3 level; here the SPF
  columns are orthonormal by construction and the CI layer is an
  eigenvector basis.
* **They are stationary under the idle Hamiltonian within the manifold**,
  so there is no initial dephasing and no "99.71 % pure" contamination to
  carry into the gate.
* **The residual error moves into the ansatz**, where it is controlled by
  ``n_s`` and can be converged, rather than sitting in the state definition
  where it could not be.

⚠⚠ ``n_s = 2`` IS A TRAP FOR A FIDELITY CALCULATION
---------------------------------------------------

At ``n_s = 2`` the top space is exactly 16-dimensional: it IS the
computational subspace, the matching is trivial (every root is wanted), and
the relaxation cannot pick a wrong one. It is also **unable to represent
leakage**, the entire remaining loss (9.3e-03) of the Hartree-product
register. Population that should leave the computational subspace has
nowhere to go except into a rotation of the time-dependent SPFs, i.e. it is
aliased into the logical basis itself rather than showing up as lost
population. A gate fidelity computed at ``n_s = 2`` would be flattered by
construction.

Use ``n_s = 2`` only to reproduce the logical manifold exactly. For anything
that reports a fidelity, ``n_s >= 3`` — at which point the roots are no
longer all wanted, and the matching below is what selects them.

Root following, the gauge, and why the weights are not uniform
--------------------------------------------------------------

This is the one-electron-per-well procedure of the paper
(Secs. S3 A 1 and S3 A 2) transplanted one
layer up. The correspondence is exact:

    1e per well                     3e per well, ML
    ---------------------------     -----------------------------------
    orbital  phi^s_i(x)             SPF  phi^s_j = sum_I B_s[I,j] |Phi^s_I>
    CI tensor C[a, i_0..i_3]        top tensor A[a, j_0..j_3]
    natural-ORBITAL gauge           natural-SPF gauge
    P^(k)_n = |C^(k)_n|^2           P^(k)_n = |A^(k)_n|^2
    diagonalise C each ITE step     diagonalise the top block each step

At one electron per well the SPF layer is absent and the two coincide; at
three electrons per well the SPFs absorb the intra-well correlation that
forced the Hartree-product construction in standard MCTDHF.
So the ML register is built the one-electron way.

Two things have to be right for the labelling to mean anything.

**The gauge.** The paper fixes the residual rotation freedom by diagonalising
the state-averaged one-particle density and working in its natural basis;
here that is ``rotate_to_natural_spfs``, which diagonalises the node density
``rho^s``. Only in that gauge does "SPF index k of well s" mean "well level
k", which is what makes ``P = |A_n|^2`` readable at all.

**The weights — and this is a trap.** The natural basis orders by
OCCUPATION, so it is well defined only when the occupations are separated.
With UNIFORM weights over the 16 register states the logical |0> and |1> are
each occupied in exactly 8 of them, so for every well

    rho^s = diag(1/2, 1/2, 0, ...)      EXACTLY degenerate

and ``eigh`` returns an arbitrary mixture of |0> and |1>. The gauge is not
merely ill-conditioned, it is undefined. Doing so
gives correct energies and orthonormality (1.1e-15) with meaningless
configuration purities (mean 0.078). The one-electron procedure avoids this
because its weights are NOT uniform: 1 for the ground state, 1/2 for the four singles,
0 above. Its non-uniform weights are what make its natural gauge well
defined — the gauge and the weighting are one prescription, not two.

The two roles are separated here, because they want different things:

* ``sa_weights`` set which states the SPFs and orbitals are optimised for
  (Eq. (7) of the paper). For a gate we want all 16 described equally, so the
  default is UNIFORM.
* ``gauge_weights`` only label the SPF axes. The rotation is an exact
  transform — it leaves Psi, the energy and every overlap unchanged — so
  using a different weighting here changes nothing physical. The default is
  geometric, ``w_n = 2^-|n|``, which gives ``rho^s = diag(2/3, 1/3, ...)``
  for every well by construction: level 0 above level 1 with a gap of 1/3,
  independent of K.

This split is the reason the top-layer roots stay exact eigenvectors of the
projected H whatever the weights: the top block is the Hamiltonian, and no
weight enters it.

**Matching.** With the gauge pinned to a physical criterion, the one-hot
product configurations are a FIXED reference and matching is against them
directly, resolved by a Hungarian assignment so two targets cannot claim the
same root (``one_electron/lib/utils.match_configs`` does the same
assignment for the same reason; greedy per-state argmax collapses near an
avoided crossing). Matching against the previous iteration instead
removes the reference that defines the labels: each step's
drift is infinitesimal, so root following reports 1.000000 while the basis
rotates out from under it. That is available as ``match="prev"`` for
comparison, and it is not the default.
"""

import time
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment

from mctdhf.integrators import _solve_bs_subcycled
from mctdhf.integrators_ml import _ml_options, _solve_Bs_subcycled

from . import fourwell as fw


def iter_bitstrings(K=4, n_levels=2):
    """Configurations ordered by total excitation, then species-major.

    The same order as ``one_electron/lib/utils.iter_bitstrings``, which
    is the column order the gate matrix and its reference assume. Getting
    this wrong permutes the gate matrix silently.
    """
    from itertools import product
    configs = list(product(range(n_levels), repeat=K))
    return sorted(configs, key=lambda c: (sum(c), c))


def config_reference(num_states, n_spf, configs):
    """One-hot top tensors for the target configurations.

    ``ref[a]`` puts well ``w`` in SPF index ``configs[a][w]``, i.e. it is the
    product configuration expressed in the CURRENT SPF basis.
    """
    A = np.zeros((num_states,) + (n_spf,) * fw.NUM_WELLS, dtype=np.complex128)
    for a, bits in enumerate(configs):
        A[(a,) + tuple(bits)] = 1.0
    return A


def excitation_weights(configs, ratio=0.5):
    """``w_n = ratio ** |n|``, normalised. The gauge-fixing weights.

    For a two-level register this puts every well's node density at
    ``diag(1/(1+ratio), ratio/(1+ratio), 0, ...)`` — exactly ``(2/3, 1/3)``
    at the default — for ANY number of wells, since summing ``ratio^|n|``
    over the configurations with ``n_s`` fixed factorises. Level 0 therefore
    sits above level 1 by a fixed, K-independent margin, which is what makes
    the natural-SPF ordering a meaningful label. See the module docstring.
    """
    w = np.array([ratio ** sum(c) for c in configs], dtype=np.float64)
    return w / w.sum()


def paper_weights(configs):
    """The one-electron weights (Sec. S3 A 1): 1 ground, 1/2 singles, 0 above.

    Kept for comparison; as ``sa_weights`` it
    optimises the SPFs for five of the sixteen states only.
    """
    w = np.array([1.0 if sum(c) == 0 else (0.5 if sum(c) == 1 else 0.0)
                  for c in configs], dtype=np.float64)
    return w / w.sum()


def regauge_natural(ml, A, Bs, weights):
    """``rotate_to_natural_spfs`` with the weights supplied explicitly.

    Same transform as ``MLMCTDHF_n_species.rotate_to_natural_spfs``, which
    takes its weights from ``ml.state_weights``; here the labelling weights
    are decoupled from the state-averaging weights (module docstring), so
    they have to be passed in. ``bs`` is untouched — this rotates the SPF
    layer only.

    The conjugation is load-bearing and follows the
    solver: ``rho^s`` is stored with the CONJUGATED index first, so the
    transform is built from ``conj(U)``.
    """
    Bs_new = []
    for s in range(ml.num_species):
        rho = ml.dCI._compute_rho_ci(A, s, weights)
        _evals, U = jnp.linalg.eigh(rho)
        U = U[:, ::-1]                       # descending occupations
        Bs_new.append(Bs[s] @ jnp.conj(U))
        A = jnp.tensordot(A, U, axes=([s + 1], [0]))
        A = jnp.moveaxis(A, -1, s + 1)
    return A, Bs_new


def node_occupations(ml, A, weights):
    """``rho^s`` eigenvalues per well, descending. Diagnostic."""
    return [np.asarray(jnp.real(jnp.linalg.eigvalsh(
        ml.dCI._compute_rho_ci(A, s, weights)))[::-1])
        for s in range(ml.num_species)]


def idle_cost(purity, configs):
    """``L_idle`` of the paper, Eq. (S13), from the configuration
    populations.

    ``L_idle = sum_n w_n [1 - P^(k_n)_n]`` with ``w_n = 3`` for |0000> and
    the singles, ``1`` for the six doubles, ``0.3`` for the triples and
    |1111> (Sec. S3 A 2). At one electron per well the
    optimised idle point reaches ``L_idle = 0.030`` with a minimum
    population of ``0.994`` (Table S1), so this is directly
    comparable — the register here is built the same way, one layer up.

    Note what it does NOT measure: ``P`` is a population inside the ML
    manifold, so leakage out of the manifold is scored by the node-density
    tail instead (see :func:`node_occupations`).
    """
    w = {0: 3.0, 1: 3.0, 2: 1.0, 3: 0.3, 4: 0.3}
    weights = np.array([w[sum(c)] for c in configs])
    return float(np.sum(weights * (1.0 - np.asarray(purity))))


def additivity(evals, configs):
    """Detunings, additivity residual and a collapse verdict.

    A four-qubit register is additive: ``E(n) = E(0) + sum_s n_s * Delta_s``
    up to the static ZZ terms, which are the whole point of the gate and are
    ~1e-5 here. So fitting ``Delta_s`` from the four single excitations and
    looking at what the other eleven configurations do is a strong test of
    whether the thing that relaxed is a register at all.

    ⚠ It is the test that PURITY CANNOT DO. Purity is ``|<config|A_a>|^2``
    with the configuration expressed in the CURRENT SPF basis, so it asks
    only whether ``A_a`` is a product state in whatever basis relaxed — it is
    self-referential. At ``dt = 2e-2`` the SPF layer collapses and purity
    reads 0.99996 with ``L_idle = 1.5e-4``, BETTER than the healthy
    ``dt = 2e-3`` solution, while |0000> has fallen to energy rank 68 of 81
    and two of the four detunings have gone NEGATIVE. Additivity catches
    that immediately; purity and ``L_idle`` endorse it.

    Returns ``(detunings, residual, healthy)``.
    """
    idx = {tuple(c): a for a, c in enumerate(configs)}
    K = len(configs[0])
    zero = tuple([0] * K)
    e = np.asarray(evals)
    e0 = e[idx[zero]]
    det = np.array([e[idx[tuple(1 if i == s else 0 for i in range(K))]] - e0
                    for s in range(K)])
    resid = max(abs(e[idx[c]] - e0 - float(np.dot(det, c))) for c in
                (tuple(c) for c in configs))
    healthy = bool(det.min() > 0.0 and np.argmin(e) == idx[zero])
    return det, float(resid), healthy


def dense_top(ml, top_ops, a_shape):
    """The top-layer Hamiltonian as a dense matrix.

    Identical in construction to ``_diag_solve_C``'s: apply the top operator
    to the identity in SPF-product space. This is the whole reason the
    intruder problem dissolves — the matrix is small enough to exist.
    """
    dim = 1
    for n in a_shape[1:]:
        dim *= int(n)
    eye = jnp.eye(dim, dtype=jnp.complex128).reshape((dim,) + a_shape[1:])
    H = ml.dCI.apply_ci_ops(top_ops, eye).reshape(dim, dim)
    return 0.5 * (H + H.conj().T)


def select_roots(H_top, A_prev, num_states):
    """Complete diagonalisation, then Hungarian matching onto ``A_prev``.

    The eigendecomposition and the overlap contraction stay ON DEVICE; only
    the ``(num_states, dim)`` overlap matrix and the eigenvalues come back to
    the host, because ``linear_sum_assignment`` is SciPy. Pulling the whole
    ``dim x dim`` eigenvector matrix across every iteration would move 27 MB
    per iteration at ``n_s = 6`` for no reason.

    Returns ``(A_new, evals_selected, overlap, rank)`` where ``rank`` is each
    selected root's position in the ENERGY ordering — the direct measure of
    how many intruders an energy-ordered solver would have had to pass.
    """
    evals, evecs, F = _eig_and_overlap(H_top, A_prev, num_states)
    F_h = np.asarray(F)
    rows, cols = linear_sum_assignment(-F_h)
    cols = cols[np.argsort(rows)]

    A_new = evecs[:, jnp.asarray(cols)].T
    evals_h = np.asarray(evals)[cols]
    return (A_new, evals_h, F_h[np.arange(num_states), cols], cols)


@partial(jax.jit, static_argnums=2)
def _eig_and_overlap(H_top, A_prev, num_states):
    dim = H_top.shape[0]
    evals, evecs = jnp.linalg.eigh(H_top)
    prev = A_prev.reshape(num_states, dim)
    prev = prev / jnp.linalg.norm(prev, axis=1, keepdims=True)
    F = jnp.abs(prev.conj() @ evecs) ** 2
    return evals, evecs, F


def relax(ml, A, Bs, bs, steps, dt, configs, verbose=True, report_every=50,
          B_propagator="diagonalize", b_propagator="cn",
          gauge_w=None, match="fixed", regauge=True,
          B_kwargs=None, b_kwargs=None, tol=1e-11, split="basic",
          precision="x64", mixed_tol=1e-5, mixed_maxiter=200):
    """State-averaged multilayer MCSCF, the 1e-per-well way (module docstring).

    One iteration:

    0. rotate the SPF layer into its NATURAL basis under ``gauge_w``, so
       that SPF index ``k`` of well ``s`` means well level ``k``;
    1. build the top-layer operators at the current (B, b) and diagonalise
       the top block COMPLETELY;
    2. select the 16 roots by Hungarian assignment onto the one-hot product
       configurations, which in that gauge are a FIXED reference;
    3. with those states fixed, take one imaginary-time step of the SPF
       layer and one of the orbital layer.

    Step 3 reuses the ML integrator's own layer solves, so the SPFs and
    orbitals relax under exactly the machinery that will later propagate
    them. Step 0 is exact — it changes no physical quantity — and is the
    gauge fix of Sec. S3 B 1; step 2 is only meaningful because of it.

    ``gauge_w`` defaults to ``excitation_weights(configs)``. Passing
    ``ml.state_weights`` reproduces ``rotate_to_natural_spfs`` exactly, and
    with uniform state weights that is the degenerate case the module
    docstring warns about.

    ⚠ The two array halves are each wrapped in ONE ``jax.jit`` built once,
    outside the loop. An earlier version called the layer solves directly
    from the Python loop; ``_expm_dense_linear`` dispatches its ``vmap`` and
    ``expm`` eagerly, so every iteration traced and compiled fresh programs
    and the compilation cache grew until LLVM could not allocate — the run
    died at iteration ~200 with "LLVM compilation error: Cannot allocate
    memory" while 40 GB was free. The production integrators avoid this by
    jitting the whole step; so does this.

    The eigendecomposition and the assignment stay on the host between the
    two jitted halves: ``linear_sum_assignment`` is SciPy, and the matrix is
    small enough that the round trip costs nothing.

    ``precision="mixed"`` replaces both CN layer solves by the defect-
    correction form of :mod:`mixed` — float64 residual, float32 BiCGSTAB on
    the increment to relative ``mixed_tol`` — which leaves the fixed point
    exactly the float64 one. ``"x64d"`` is the same defect form solved in
    float64. ``cn`` on both layers only.
    """
    if precision not in ("x64", "mixed", "x64d"):
        raise ValueError(f"precision must be 'x64', 'mixed' or 'x64d'; "
                         f"got {precision!r}")
    if precision != "x64" and (B_propagator, b_propagator) != ("cn", "cn"):
        raise ValueError(f"precision={precision!r} is implemented for "
                         "B=cn b=cn only")
    # ⚠ The sub-solver option NAMES are per-propagator (`_B_KWARG_KEYS`), so
    # the defaults have to be chosen after the propagator is known. An
    # earlier version passed `{"orbital_x64": True}` unconditionally, which
    # is a `diagonalize`-only key — every other B propagator raised on
    # validation, so `B_propagator="cn"` was unreachable from here.
    B_kw = dict(B_kwargs or {})
    b_kw = dict(b_kwargs or {})
    for prop, kw in ((B_propagator, B_kw), (b_propagator, b_kw)):
        if prop == "diagonalize":
            kw.setdefault("orbital_x64", True)
        elif prop == "cn":
            kw.setdefault("solver_tol", 1e-12)
            kw.setdefault("max_solver_iter", 400)
    _a_opts, B_opts, b_opts = _ml_options(
        "diagonalize", B_propagator, b_propagator, None, B_kw, b_kw)

    a_shape = ml.a_shape
    b_shapes = ml.b_shapes
    num_states = ml.num_states

    if gauge_w is None:
        gauge_w = excitation_weights(configs)
    gauge_w = jnp.asarray(np.asarray(gauge_w, dtype=np.float64))

    # NOT named `regauge`: that is the boolean parameter, and a def would
    # rebind it, making `--no-regauge` silently do nothing.
    @jax.jit
    def _regauge(A, Bs):
        return regauge_natural(ml, A, Bs, gauge_w)

    @jax.jit
    def top_matrix(Bs, bs):
        h_current = ml._get_h_current(0.0)
        h_MO, g_MO = ml._ml_integrals(h_current, bs)
        es = [ml.dCI.spf_transition_ops(s, Bs[s])
              for s in range(ml.num_species)]
        top_ops = ml.dCI.ml_top_ops(h_MO, g_MO, Bs, es=es)
        return dense_top(ml, top_ops, a_shape)

    def _B_half(A, Bs, bs, rho_invs, rho_pairs, step):
        """One SPF solve over `step`, with the integrals taken at `bs`."""
        h_current = ml._get_h_current(0.0)
        h_MO, g_MO = ml._ml_integrals(h_current, bs)
        Ops = ml.dCI.ml_species_ops(h_MO, g_MO)
        Bs_new = _solve_Bs_subcycled(ml, B_propagator, Bs, A, g_MO, Ops,
                                     rho_invs, rho_pairs, step, B_opts)
        return [ml.ortonorm_B(B) for B in Bs_new]

    def _b_full(A, Bs, bs):
        h_current = ml._get_h_current(0.0)
        es = [ml.dCI.spf_transition_ops(s, Bs[s])
              for s in range(ml.num_species)]
        D_invs, ds = ml._compute_full_RDMs(A, Bs, es)
        bs_new = _solve_bs_subcycled(ml, b_propagator, bs, h_current, D_invs,
                                     ds, dt, b_shapes, b_opts)
        return [ml.ortonorm_b(b) for b in bs_new]

    if precision in ("mixed", "x64d"):
        from . import mixed as _mx
        _lo = precision == "mixed"

        def _B_half(A, Bs, bs, rho_invs, rho_pairs, step):  # noqa: F811
            return _mx.B_half(ml, A, Bs, bs, rho_invs, rho_pairs, step,
                              mixed_tol, mixed_maxiter, lo=_lo)

        def _b_full(A, Bs, bs):  # noqa: F811
            return _mx.b_full(ml, A, Bs, bs, dt, mixed_tol, mixed_maxiter,
                              lo=_lo)

    @jax.jit
    def layer_step(A, Bs, bs):
        # The top tensor is REPLACED, not propagated, so there is no A-half
        # to take: the only splitting freedom an ITE iteration has is
        # between the SPF and orbital layers. 'strang' is B/2 · b · B/2 with
        # the second half-step taken at the NEW orbitals; 'basic' is B · b.
        rho_invs, rho_pairs = ml._compute_node_densities(A)
        if split == "strang":
            Bs_h = _B_half(A, Bs, bs, rho_invs, rho_pairs, dt / 2)
            bs_new = _b_full(A, Bs_h, bs)
            Bs_new = _B_half(A, Bs_h, bs_new, rho_invs, rho_pairs, dt / 2)
            return Bs_new, bs_new
        Bs_new = _B_half(A, Bs, bs, rho_invs, rho_pairs, dt)
        return Bs_new, _b_full(A, Bs_new, bs)

    ref = jnp.asarray(config_reference(num_states, ml.num_spfs[0], configs))
    A = ref
    e_prev = None
    history = []
    evals = rank = None
    label = "target" if match == "fixed" else "prev"
    t_prev = time.perf_counter()

    for it in range(steps):
        if regauge:
            A, Bs = _regauge(A, Bs)

        H_top = top_matrix(Bs, bs)
        target = ref if match == "fixed" else A
        A_flat, evals, ovl, rank = select_roots(H_top, target, num_states)
        A = A_flat.reshape(a_shape)

        Bs, bs = layer_step(A, Bs, bs)

        e_sum = float(np.sum(evals))
        # `select_roots` pulls the eigenvalues to the host, so the iteration
        # is already synchronous here and the wall time is a real one.
        now = time.perf_counter()
        history.append(dict(it=it, e_sum=e_sum, min_overlap=float(ovl.min()),
                            max_rank=int(rank.max()), t=now - t_prev))
        t_prev = now
        if verbose and (it + 1) % report_every == 0:
            d = "" if e_prev is None else f"  dE = {e_sum - e_prev:+.3e}"
            print(f"    it {it+1:5d}  sum E = {e_sum:.8f}{d}  "
                  f"min |<{label}|root>|^2 = {ovl.min():.6f}  "
                  f"highest root index = {rank.max()}", flush=True)
        # An unstable propagator (explicit scheme past its stability limit)
        # runs away to inf/nan, and `abs(nan - nan) < tol` is False, so the
        # loop would otherwise sit out the whole step budget producing
        # nothing. Non-finite is unambiguous and cannot false-positive.
        if not np.isfinite(e_sum):
            if verbose:
                print(f"    it {it+1:5d}  DIVERGED (sum E = {e_sum})",
                      flush=True)
            break
        if e_prev is not None and abs(e_sum - e_prev) < tol:
            break
        e_prev = e_sum

    if regauge:
        A, Bs = _regauge(A, Bs)
    return A, Bs, bs, evals, rank, history


def diagnose(ml, A, Bs, bs, configs, evals, rank, verbose=True,
             gauge_w=None):
    """Purity, orthonormality, the intruder count and the node densities.

    The node-density spectra are the diagnostic that caught the degenerate
    gauge. Under UNIFORM weights a clean two-level register gives
    ``(1/2, 1/2, 0, ...)`` for every well: the entries beyond the second are
    the population that has left the logical manifold, and they are the
    honest leakage measure at this stage. The gauge-weighted spectrum is
    printed alongside, where the same register gives ``(2/3, 1/3, 0, ...)``
    and the 0/1 ordering is the one the labels were read in.
    """
    num_states = ml.num_states
    dim = int(np.prod(A.shape[1:]))
    A_flat = np.asarray(A).reshape(num_states, dim)

    ref = config_reference(num_states, ml.num_spfs[0], configs)
    ref = ref.reshape(num_states, dim)
    purity = np.abs(np.sum(ref.conj() * A_flat, axis=1)) ** 2

    S = A_flat.conj() @ A_flat.T
    orth = float(np.max(np.abs(S - np.eye(num_states))))

    uni = np.full(num_states, 1.0 / num_states)
    occ_uni = node_occupations(ml, A, jnp.asarray(uni))
    if gauge_w is None:
        gauge_w = excitation_weights(configs)
    occ_gauge = node_occupations(ml, A, jnp.asarray(np.asarray(gauge_w)))
    leak = np.array([float(o[2:].sum()) if len(o) > 2 else 0.0
                     for o in occ_uni])

    if verbose:
        print(f"\n  top-space dimension            {dim}")
        print(f"  max |<a|b> - delta_ab|         {orth:.3e}")
        print(f"  configuration purity, min      {purity.min():.6f}")
        print(f"  configuration purity, mean     {purity.mean():.6f}")
        print(f"  L_idle (paper Eq. (S13))       "
              f"{idle_cost(purity, configs):.6f}   [1e optimum 0.030]")
        print(f"  highest selected root index    {int(rank.max())} of {dim}")
        print(f"  roots skipped below the top    "
              f"{int(rank.max()) + 1 - num_states}")
        det, resid, healthy = additivity(evals, configs)
        print(f"  detunings per well             "
              + " ".join(f"{d:.6f}" for d in det))
        print(f"  additivity residual (ZZ scale) {resid:.3e}")
        print(f"  register verdict               "
              f"{'HEALTHY' if healthy else '*** COLLAPSED ***'}")
        print("\n  node density rho^s (uniform weights; ideal 0.5 0.5 0 ...)")
        for s, o in enumerate(occ_uni):
            print(f"    well {s}: " + " ".join(f"{v:9.6f}" for v in o)
                  + f"   outside logical = {leak[s]:.3e}")
        print("  node density rho^s (gauge weights; ideal 0.667 0.333 0 ...)")
        for s, o in enumerate(occ_gauge):
            print(f"    well {s}: " + " ".join(f"{v:9.6f}" for v in o)
                  + f"   0/1 gap = {o[0] - o[1]:.6f}")
        print("\n  config    E                 rank   purity")
        idx = np.argsort(evals)
        for a in np.argsort(rank):
            bits = "".join(str(b) for b in configs[a])
            print(f"   |{bits}>   {evals[a]:.8f}   {rank[a]:5d}   "
                  f"{purity[a]:.6f}")
    det, resid, healthy = additivity(evals, configs)
    return dict(purity=purity, orth=orth, rank=rank, evals=evals, dim=dim,
                l_idle=idle_cost(purity, configs), detunings=det,
                additivity=resid, healthy=healthy,
                occ_uniform=np.array(occ_uni), occ_gauge=np.array(occ_gauge),
                leakage=leak)
