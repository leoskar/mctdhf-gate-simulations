"""S2: grid and box convergence of the |11> - |30>/|03>/|40>/|04> detunings.

    python -m grid_convergence.grid_check

The central number is a ~1e-3 difference between ~8 hbar w0 excitation
energies, so the single-well 3e FCI is redone on other DVRs at the three swept
gate points (x_target), both active wells, bare:

  (n, W) = (64, 1 um)   the production grid; must reproduce S1 to <= 1e-10
           (96, 1 um), (128, 1 um)   grid density
           (90, 1.4 um)              box width at fixed spacing
                                     (1/65 um = 1.4/91 um)

The well on an n-point sine DVR over [c - W/2, c + W/2]:
  * kinetic matrix from mctdhf.sine_dvr.SineDVR (what fourwell uses),
  * V = 0.5 a^2 (x-c)^2 + (b a^2/3)(x-c)^4 (fourwell._v_well's formula),
  * intra kernel = fourwell.veff evaluated on the new grid, in the same 2-D
    DVR form (SineDVR.get_g(fourwell._veff_interaction), as g_dict builds
    the (w, w) entry).

FCI is matrix-free (C(128, 3) = 341 376 determinants): a determinant vector
c is expanded to the antisymmetric tensor X[p,q,r] (= sgn c_sorted), the
one-body part applied as (h x 1 x 1 + 1 x h x 1 + 1 x 1 x h) X — by
antisymmetry the last two are -T^(213) and -T^(321) of T = (h x 1 x 1) X, so
one n^4 matmul per matvec — and read back at the sorted triples; the
interaction is diagonal in the DVR determinants. Lanczos (eigsh) for the
lowest 8 levels.

Output: results/grid_check.json.
"""
import itertools
import os
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from . import common as K
from mctdhf.sine_dvr import SineDVR
from system import fourwell as fw

GRIDS = [(64, 1.0), (96, 1.0), (128, 1.0), (90, 1.4)]
K_LEV = 8
PAIRS = [(3, 0), (0, 3), (4, 0), (0, 4)]
ACCEPT = 1e-4


class BoxWell:
    def __init__(self, w, n, W_um):
        c = float(fw.WELL_CENTERS[w])
        half = 0.5 * W_um / fw.LENGTH_UNIT_UM
        self.dvr = SineDVR(x_min=c - half, x_max=c + half, num_points=n)
        self.x = np.asarray(self.dvr.grid, dtype=np.float64)
        self.dx = self.x - c
        self.T = np.real(np.asarray(self.dvr.get_h(potential=lambda x: 0.0 * x),
                                    dtype=np.float64))
        g = np.asarray(self.dvr.get_g(interaction=fw._veff_interaction),
                       dtype=np.float64)
        self.n = n
        self.dets = np.array(list(itertools.combinations(range(n), 3)))
        p, q, r = self.dets.T
        self.g_diag = g[p, q] + g[p, r] + g[q, r]
        self.D = len(self.dets)

    def h(self, a, b):
        v = 0.5 * a ** 2 * self.dx ** 2 + (b * a ** 2 / 3.0) * self.dx ** 4
        return self.T + np.diag(v)

    def _expand(self, c):
        n = self.n
        X = np.zeros((n, n, n))
        p, q, r = self.dets.T
        X[p, q, r] = c
        X[q, r, p] = c
        X[r, p, q] = c
        X[q, p, r] = -c
        X[p, r, q] = -c
        X[r, q, p] = -c
        return X

    def levels(self, a, b, k=K_LEV):
        E, V, res = self.solve(a, b, k)
        n = self.n
        # density within 0.1 um of the walls (ground state)
        dens = np.zeros(n)
        for s in range(3):
            dens += np.bincount(self.dets[:, s], weights=V[:, 0] ** 2, minlength=n)
        return E, res, dens

    def one_rdm(self, c):
        """gamma[p, q] = (1/2) sum_rs X[p,r,s] X[q,r,s] (trace 3) in the DVR."""
        X = self._expand(c)
        n = self.n
        Xm = X.reshape(n, n * n)
        return 0.5 * Xm @ Xm.T

    def solve(self, a, b, k=K_LEV):
        h = self.h(a, b)
        n = self.n
        p, q, r = self.dets.T

        def mv(c):
            c = np.ravel(c)
            X = self._expand(c)
            T = (h @ X.reshape(n, n * n)).reshape(n, n, n)
            Y = T - T.transpose(1, 0, 2) - T.transpose(2, 1, 0)
            return Y[p, q, r] + self.g_diag * c

        H = LinearOperator((self.D, self.D), matvec=mv, dtype=np.float64)
        E, V = eigsh(H, k=k, which="SA", tol=1e-12, ncv=48,
                     v0=np.ones(self.D) / np.sqrt(self.D))
        o = np.argsort(E)
        E, V = E[o], V[:, o]
        res = max(float(np.linalg.norm(mv(V[:, m]) - E[m] * V[:, m]))
                  for m in range(k))
        return E, V, res


def main():
    t00 = time.time()
    from gate_timing.timing import FCIWell
    fci64 = FCIWell()
    out = {"grids": [list(g) for g in GRIDS], "legs": {}}
    for leg, (i, j) in K.LEGS3.items():
        _, xg = K.x_scaled(leg)
        L_ = {}
        for (n, Wum) in GRIDS:
            per = {}
            for w in (i, j):
                t0 = time.time()
                bw = BoxWell(w, n, Wum)
                E, res, dens = bw.levels(xg[w], xg[4 + w])
                e = E - E[0]
                edge = float(dens[np.abs(bw.dx) > 0.5 * Wum / fw.LENGTH_UNIT_UM
                                  - 0.1 / fw.LENGTH_UNIT_UM].sum() / 3.0)
                per[w] = dict(E=E.tolist(), e=e.tolist(), resid=res,
                              A3e=float(E[3] - 2 * E[1] + E[0]),
                              A3e_4=float(E[4] - 2 * E[1] + E[0]),
                              wall_density=edge, D=bw.D, dx_um=float(
                                  (bw.x[1] - bw.x[0]) * fw.LENGTH_UNIT_UM))
                if (n, Wum) == (64, 1.0):
                    E64, _V, r64 = K.well_levels(fci64, w, xg[w], xg[4 + w], k=K_LEV)
                    per[w]["vs_FCIWell"] = float(np.max(np.abs(E - E64)))
                print(f"  {leg} w{w} (n {n}, W {Wum} um, D {bw.D}): e1..e5 "
                      + " ".join(f"{v:.6f}" for v in e[1:6])
                      + f"  A3e {per[w]['A3e']:+.6f}  resid {res:.1e}"
                      + (f"  vs FCIWell {per[w]['vs_FCIWell']:.1e}"
                         if "vs_FCIWell" in per[w] else "")
                      + f"  wall dens {edge:.1e}  ({time.time() - t0:.0f} s)",
                      flush=True)
            Ei, Ej = np.array(per[i]["E"]), np.array(per[j]["E"])
            det = {f"{m}{nn}": float((Ei[m] - Ei[1]) + (Ej[nn] - Ej[1]))
                   for m, nn in PAIRS}
            L_[f"{n}_{Wum}"] = dict(wells=per, pair_detuning=det)
        ref = L_["64_1.0"]["pair_detuning"]
        for key, r in L_.items():
            r["pair_shift_vs_64"] = {p: r["pair_detuning"][p] - ref[p] for p in ref}
            r["max_shift"] = max(abs(v) for v in r["pair_shift_vs_64"].values())
        out["legs"][leg] = L_
        print(f"\n  {leg} pair detunings E(mn) - E(11):", flush=True)
        print("    grid          " + "".join(f"{'|' + p + '>':>12}" for p in ref)
              + "   max|shift|", flush=True)
        for key, r in L_.items():
            print(f"    {key:<13s} " + "".join(f"{r['pair_detuning'][p]:>+12.6f}"
                                         for p in ref)
                  + f"   {r['max_shift']:.1e}", flush=True)
    worst64 = max(out["legs"][l]["64_1.0"]["wells"][w]["vs_FCIWell"]
                  for l in out["legs"] for w in out["legs"][l]["64_1.0"]["wells"])
    worst = max(r["max_shift"] for l in out["legs"].values() for r in l.values())
    out["reproduce_64"] = dict(worst=worst64, passed=worst64 <= 1e-10)
    out["acceptance"] = dict(worst_shift=worst, threshold=ACCEPT,
                             passed=worst <= ACCEPT)
    print(f"\n  (64, 1 um) vs FCIWell: worst {worst64:.1e} "
          f"({'PASS' if worst64 <= 1e-10 else 'FAIL'})", flush=True)
    print(f"  acceptance (every pair detuning moves <= {ACCEPT:g}): worst "
          f"{worst:.1e} -> {'PASS' if worst <= ACCEPT else 'FINDING'}", flush=True)
    K.save_json(os.path.join(K.RESULTS, "grid_check.json"), out)
    print(f"\n  wall {time.time() - t00:.0f} s", flush=True)


if __name__ == "__main__":
    main()
