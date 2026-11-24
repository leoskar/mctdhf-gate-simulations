"""Memory-light determinant overlap for the norb-26 rung, installed at run
time; the library is not edited.

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.overlap_patch validate 24   # GPU
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.overlap_patch validate 16   # CPU smoke

**Why.** ``MCTDHF_n_species._det_overlap_matrices`` builds the minors
``S_IJ = det(O[occ_I, occ_J])`` by forming dense ``(D, D, norb, norb)`` Mask
and Val tensors and extracting with ``nonzero``: at norb 26 (D = 2600) that is
~73 GB complex. The CPU dies; on the 8 GB GPU XLA fuses it, but the peak
leaves the memory pool holding the card, and three norb-26 relaxations failed
loading later kernels (CUDA_ERROR_OUT_OF_MEMORY).

**The replacement** gathers the same 3x3 matrices directly,
``mat[I, J, k, l] = O[occ_I[k], occ_J[l]]`` with ``occ_I`` the ascending
occupied orbitals of determinant I: the row-major order in which the
library's ``nonzero(Mask)`` extracts them. ``jnp.linalg.det`` then sees the
same input as the library. Peak ~1 GB at norb 26.

**Checks**:

* ``validate`` (norb 24): library vs gather on overlaps between DIFFERENT
  orbital sets (embedded vs relaxed, relaxed vs a randomly mixed set,
  embedded vs mixed), the determinant-overlap matrices and the full 4x4
  ``calculate_overlap``; required <= 1e-14 (smoke test at norb 16 on the
  CPU first). Not <y|y>: with identical
  orbitals the minors are trivial and a mis-indexed gather would pass.
  -> results/overlap_patch_check<N>.json; exit status 1 on failure.
* :func:`spot_check` (norb 26, inside the relax and gate jobs): ~1000 random
  (I, J) minors against ``numpy.linalg.det(O[np.ix_(occ_I, occ_J)])`` with
  the actual 26-orbital O between two different states; required <= 1e-12.
"""
import os
import sys
import time

from . import common as C  # noqa: I100  (x64 + redirections first)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from opt_einsum import contract  # noqa: E402

from mctdhf.MCTDHF_n_species import MCTDHF_n_species  # noqa: E402

L, EC, fw = C.L, C.EC, C.fw
LIBRARY = MCTDHF_n_species._det_overlap_matrices
TOL_LIB, TOL_SPOT = 1e-14, 1e-12


def occ_table(I_s):
    """(D, norb) occupation table -> (D, n_elec) ascending occupied
    orbital indices (row-major ``nonzero``, the library's extraction order)."""
    I_s = np.asarray(I_s).astype(bool)
    n = int(I_s[0].sum())
    assert np.all(I_s.sum(axis=1) == n), "ragged occupation table"
    return np.nonzero(I_s)[1].reshape(I_s.shape[0], n)


def det_overlap_gather(self, bs1, bs2):
    """Drop-in for ``MCTDHF_n_species._det_overlap_matrices``."""
    S_matrices = []
    for s in range(self.num_species):
        occ = occ_table(self.dCI.I[s])
        O = contract("ki, kj -> ij", bs1[s].conj(), bs2[s], backend="jax")
        mat = O[occ[:, None, :, None], occ[None, :, None, :]]
        S_matrices.append(jnp.linalg.det(mat))
    return S_matrices


def install():
    MCTDHF_n_species._det_overlap_matrices = det_overlap_gather
    print("  [overlap] _det_overlap_matrices -> orbital_convergence.overlap_patch gather",
          flush=True)


def uninstall():
    MCTDHF_n_species._det_overlap_matrices = LIBRARY


def mixed(ml, y, seed=0, scale=0.3):
    """``y`` with every orbital set replaced by an orthonormalised random
    mixture ``qr(b + scale R)``: a genuinely different orbital set (generic
    O, non-trivial minors) with A and B unchanged."""
    A, Bs, bs = ml._y_to_ABb(y)
    rng = np.random.default_rng(seed)
    out = []
    for b in bs:
        b = np.asarray(b)
        R = rng.standard_normal(b.shape) + 1j * rng.standard_normal(b.shape)
        q, r = np.linalg.qr(b + scale * R)
        out.append(jnp.asarray(q * np.sign(np.real(np.diag(r)))))
    return ml._ABb_to_y(A, list(Bs), out)


def spot_check(ml, y1, y2, label, n=1000, seed=1, tol=TOL_SPOT):
    """~``n`` random (I, J) minors per species of the gather against
    ``numpy.linalg.det(O[np.ix_(occ_I, occ_J)])``. Raises on failure."""
    _, _, bs1 = ml._y_to_ABb(y1)
    _, _, bs2 = ml._y_to_ABb(y2)
    S = jax.jit(lambda a, b: det_overlap_gather(ml, a, b))(list(bs1),
                                                          list(bs2))
    rng = np.random.default_rng(seed)
    rec = {}
    for s in range(ml.num_species):
        occ = occ_table(ml.dCI.I[s])
        D = occ.shape[0]
        O = np.asarray(bs1[s]).conj().T @ np.asarray(bs2[s])
        Ss = np.asarray(S[s])
        if D * D <= n:
            pairs = [(i, j) for i in range(D) for j in range(D)]
        else:
            pairs = list(zip(rng.integers(0, D, n), rng.integers(0, D, n)))
        ref = np.array([np.linalg.det(O[np.ix_(occ[i], occ[j])])
                        for i, j in pairs])
        got = np.array([Ss[i, j] for i, j in pairs])
        err = float(np.max(np.abs(got - ref)))
        rec[f"w{s}"] = dict(D=D, n_pairs=len(pairs), max_err=err,
                            max_abs_minor=float(np.max(np.abs(ref))),
                            median_abs_minor=float(np.median(np.abs(ref))),
                            O_dist_from_identity=float(np.max(np.abs(
                                O - np.eye(*O.shape)))))
        print(f"  [spot {label}] w{s}: D {D}, {len(pairs)} minors, max err "
              f"{err:.1e}, |minor| max {rec[f'w{s}']['max_abs_minor']:.2e} "
              f"median {rec[f'w{s}']['median_abs_minor']:.2e}, "
              f"max|O - 1| {rec[f'w{s}']['O_dist_from_identity']:.2e}",
              flush=True)
    worst = max(r["max_err"] for r in rec.values())
    ok = worst <= tol
    print(f"  [spot {label}] worst {worst:.1e} (tol {tol:.0e}) -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(f"OVERLAP SPOT CHECK FAILED ({label}): {worst:.2e}")
    return dict(label=label, worst=worst, tol=tol, species=rec)


def validate(N):
    """norb N (> 14): library vs gather on different orbital sets. The
    reference check is N = 24 on the GPU, where the library fits; N = 16 on
    the CPU is a smoke test."""
    from gates_80pt import mf80
    leg = "w1w2"
    assert N > 14
    t0 = time.time()
    hi = mf80.h_idle(leg)
    lay14, lay = mf80.layout(14), mf80.layout(N)
    ml14 = L.make_ml(leg, hi, lay14)
    y14, _ = mf80.load_state(leg, 14, ml14)
    ml_lib = L.make_ml(leg, hi, lay)          # traced with the library
    ml_gat = L.make_ml(leg, hi, lay)          # traced with the gather
    y_rel, _ = mf80.load_state(leg, N, ml_lib)    # orbital_convergence n24 (num_dvr 80)
    y_emb = EC.embed_orbitals(leg, ml14, y14, lay)
    y_mix = mixed(ml_lib, y_rel)
    pairs = dict(emb_vs_relaxed=(y_emb, y_rel), relaxed_vs_mixed=(y_rel, y_mix),
                 emb_vs_mixed=(y_emb, y_mix))
    S_lib = jax.jit(lambda a, b: LIBRARY(ml_lib, a, b))
    S_gat = jax.jit(lambda a, b: det_overlap_gather(ml_gat, a, b))
    out = dict(norb=N, tol=TOL_LIB, pairs={})
    worst = 0.0
    for name, (y1, y2) in pairs.items():
        _, _, b1 = ml_lib._y_to_ABb(y1)
        _, _, b2 = ml_lib._y_to_ABb(y2)
        sg = [np.asarray(x) for x in S_gat(list(b1), list(b2))]
        install()
        Mg = np.asarray(ml_gat.calculate_overlap(y1, y2))
        uninstall()
        sl = [np.asarray(x) for x in S_lib(list(b1), list(b2))]
        Ml = np.asarray(ml_lib.calculate_overlap(y1, y2))
        dS = [float(np.max(np.abs(a - b))) for a, b in zip(sg, sl)]
        offd = [float(np.max(np.abs(x - np.diag(np.diag(x))))) for x in sl]
        dM = float(np.max(np.abs(Mg - Ml)))
        bitwise = all(np.array_equal(a, b) for a, b in zip(sg, sl)) \
            and np.array_equal(Mg, Ml)
        worst = max(worst, dM, *dS)
        out["pairs"][name] = dict(dS=dS, max_offdiag_minor=offd, dM=dM,
                                  bitwise=bool(bitwise),
                                  M_abs=np.abs(Ml).tolist())
        print(f"  [validate] {name}: max|S_gather - S_lib| per well "
              + " ".join(f"{v:.1e}" for v in dS)
              + f"; max|M_gather - M_lib| {dM:.1e}; bitwise {bitwise}; "
              f"max off-diagonal |S_lib| per well "
              + " ".join(f"{v:.2e}" for v in offd), flush=True)
    install()
    out["spot"] = spot_check(ml_gat, y_emb, y_rel, f"n{N} emb vs relaxed")
    uninstall()
    out["worst"] = worst
    out["passed"] = bool(worst <= TOL_LIB)
    out["wall"] = time.time() - t0
    C.save_json(C.assert_writable(os.path.join(
        C.RESULTS, f"overlap_patch_check{N}.json")), out)
    print(f"  [validate] worst {worst:.1e} (tol {TOL_LIB:.0e}) -> "
          f"{'PASS' if out['passed'] else 'FAIL'}", flush=True)
    if not out["passed"]:
        raise SystemExit(f"OVERLAP PATCH VALIDATION FAILED: {worst:.2e}")


if __name__ == "__main__":
    C.require_grid(80)
    L.device_banner()
    assert len(sys.argv) == 3 and sys.argv[1] == "validate", sys.argv
    validate(int(sys.argv[2]))
