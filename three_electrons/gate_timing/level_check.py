"""Is the MF gate's |11> leak a near-resonance with |30>/|03>? Two tests.

    level_check.py levels            # test 1: level positions vs norb
    level_check.py project           # test 2: where |11>'s lost weight goes

**levels.** Per swept leg and active well, at the gate point (env = 1) and at
idle: exact single-well FCI (``timing.FCIWell``, 64-point sine DVR,
C(64, 3) = 41664 determinants), then CI in the best N orbitals for the qubit
levels — the natural orbitals of the 2/3 |0> + 1/3 |1> ensemble (the gate's
own per-well node weights) — for N = 6 ... 22. The N-orbital determinants are
embedded exactly in the FCI space (3x3 minors of the orbital matrix), so the
truncated CI is ``T^T H T``. Reports levels 0..5 and the pair detunings
E(mn) - E(11) = (E_m - E_1)_i + (E_n - E_1)_j for m + n = 3 (|30>, |03>)
and |40>/|04>, |20>/|02> — against the FCI limit. Bare wells (no inter-well
Hartree field; the dressed shift is ~ +-0.005, physics.json).
This mimics, not reproduces, MCTDHF: its orbitals are variational and time
dependent. It shows where levels 3-4 sit in the N-orbital space built for
levels 0/1.

**project.** Final states of gate runs at the w1w2 representative point
(tau_ramp 28.3, tau_hold 105.4): the norb-14 sweep point, norb 12/14/16/18
at n_s 6. At T the Hamiltonian is back at idle, so each final pair state is
expanded in products of idle single-well FCI eigenstates |m>_i |n>_j
(m, n = 0..7), each projected into the run's own orbital space. The
spectators are traced as the state's own (single) SPF. The weight of the
|11> column on |30>, |03>, ... says where the population went.

Outputs: ``results/levelcheck/{levels,project}.json`` and the log.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

from . import mf_lib as L
from .timing import FCIWell, h_well
from system import fourwell as fw

OUT = os.path.join(L.RESULTS_DIR, "levelcheck")
LEGS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3)}
NS = list(range(6, 23))
NLEV = 8


def fci_levels(fci, h, k=NLEV):
    from scipy.sparse.linalg import eigsh
    H = fci.hamiltonian(h)
    E, V = eigsh(H, k=k, which="SA", tol=1e-12, ncv=48,
                 v0=np.ones(fci.D) / np.sqrt(fci.D))
    o = np.argsort(E)
    E, V = E[o], V[:, o]
    res = float(np.max(np.linalg.norm(H @ V - V * E, axis=0)))
    return H, E, V, res


def one_rdm(fci, c):
    """D[p, q] = <c| a+_p a_q |c> for a real FCI vector (DVR basis)."""
    n = fci.n
    D = np.zeros((n, n))
    w2 = c ** 2
    for a in range(3):
        D[np.diag_indices(n)] += np.bincount(fci.dets[:, a], weights=w2,
                                             minlength=n)
    # hop entries: <rows| a+_q a_p |cols> = sign
    np.add.at(D, (fci.q, fci.p), fci.sign * c[fci.rows] * c[fci.cols])
    return D


def minors(U, occ_rows, occ_cols, chunk=4096):
    """T[P, K] = det(U[occ_rows[P], occ_cols[K]]) for 3-particle dets."""
    out = np.empty((len(occ_rows), len(occ_cols)), dtype=np.result_type(U, float))
    for s in range(0, len(occ_rows), chunk):
        r = occ_rows[s:s + chunk]
        sub = U[r[:, None, :, None], occ_cols[None, :, None, :]]
        out[s:s + chunk] = np.linalg.det(sub)
    return out


def truncated_ci(fci, H, U, N, k=6):
    occ = np.array(list(itertools.combinations(range(N), 3)))
    T = minors(U, fci.dets, occ)
    HN = T.T @ (H @ T)
    S = T.T @ T
    assert np.max(np.abs(S - np.eye(len(occ)))) < 1e-10
    return np.linalg.eigvalsh(0.5 * (HN + HN.T))[:k]


def x_scaled(leg):
    xi, xg = L.x_vectors(leg)
    return xi / fw.TYPICAL_FREQ_GHZ, xg / fw.TYPICAL_FREQ_GHZ


def levels(a):
    fci = FCIWell()
    res = {}
    for leg, (i, j) in LEGS.items():
        xi, xg = x_scaled(leg)
        res[leg] = {}
        for where, x in (("gate", xg), ("idle", xi)):
            per = {}
            for w in (i, j):
                t0 = time.time()
                H, E, V, r = fci_levels(fci, h_well(w, x[w], x[4 + w]))
                D = (2 / 3) * one_rdm(fci, V[:, 0]) + (1 / 3) * one_rdm(fci, V[:, 1])
                occ, U = np.linalg.eigh(D)
                o = np.argsort(occ)[::-1]
                occ, U = occ[o], U[:, o]
                trunc = {N: truncated_ci(fci, H, U, N).tolist() for N in NS}
                per[w] = dict(E_fci=E.tolist(), resid=r, no_occ=occ[:24].tolist(),
                              E_N=trunc)
                print(f"  [{leg} {where}] w{w}: FCI levels "
                      + " ".join(f"{e - E[0]:.4f}" for e in E[:6])
                      + f" (rel. to E0; resid {r:.1e}; {time.time() - t0:.0f} s)",
                      flush=True)
            res[leg][where] = per
            # pair detunings E(mn) - E(11), FCI and per N
            print(f"\n  {leg} {where}: pair detuning E(mn) - E(11)", flush=True)
            pairs = [(3, 0), (0, 3), (4, 0), (0, 4), (2, 0), (0, 2)]
            print("    N   " + "".join(f"{f'|{m}{n}>':>10}" for m, n in pairs),
                  flush=True)

            def det(Ei, Ej):
                return [(Ei[m] - Ei[1]) + (Ej[n] - Ej[1]) for m, n in pairs]
            for N in NS:
                d = det(per[i]["E_N"][N], per[j]["E_N"][N])
                print(f"    {N:<4d}" + "".join(f"{v:>+10.4f}" for v in d), flush=True)
            d = det(per[i]["E_fci"], per[j]["E_fci"])
            print("    FCI " + "".join(f"{v:>+10.4f}" for v in d) + "\n", flush=True)
            res[leg][where]["pairs"] = [f"{m}{n}" for m, n in pairs]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "levels.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


# ── test 2 ────────────────────────────────────────────────────────────────

RUNS = [  # (label, file relative to results/w1w2, norb, n_s)
    ("sweep norb14 n_s4",
     "diagonalize_mix1e-4_dt0.01_tol1e-10_rg0.1_s16o4_rnstep/pt_tr28p3_th105p4.npz",
     14, 4),
    ("norb12 n_s6", "exactcheck/gate_ladder_act12s6_tr28.3_th105.4.npz", 12, 6),
    ("norb14 n_s6", "exactcheck/gate_act14s6_relaxed_tr28.3_th105.4.npz", 14, 6),
    ("norb16 n_s6", "exactcheck/gate_act16s6_relaxed_tr28.3_th105.4.npz", 16, 6),
    ("norb18 n_s6", "exactcheck/gate_act18s6_relaxed_tr28.3_th105.4.npz", 18, 6),
]


def project(a):
    leg, (i, j) = "w1w2", LEGS["w1w2"]
    fci = FCIWell()
    xi, _ = x_scaled(leg)
    chi = {}
    for w in (i, j):
        _H, E, V, r = fci_levels(fci, h_well(w, xi[w], xi[4 + w]))
        chi[w] = V
        print(f"  idle w{w} FCI levels rel. E0: "
              + " ".join(f"{e - E[0]:.4f}" for e in E) + f" (resid {r:.1e})",
              flush=True)
    h_idle = fw.h_static(xi)
    out = {}
    for label, rel, norb, ns in RUNS:
        path = os.path.join(L.RESULTS_DIR, leg, rel)
        if not os.path.exists(path):
            print(f"  {label}: {rel} missing, skipped", flush=True)
            continue
        z = np.load(path)
        lay = L.Layout(norb_active=norb, nspf_active=ns)
        ml = L.make_ml(leg, h_idle, lay)
        res = {}
        for tag, y in (("final", z["y"]), ("initial", z["y0"] if "y0" in z.files
                                            else None)):
            if y is None:
                continue
            A, Bs, bs = ml._y_to_ABb(y)
            A = np.asarray(A)
            ovl = {}
            for w in (i, j):
                b = np.asarray(bs[w])                            # (64, norb)
                occ = np.array([np.flatnonzero(d) for d in np.asarray(ml.dCI.I[w])])
                T = minors(b, fci.dets, occ)                     # (41664, D_w)
                c = T.conj().T @ chi[w]                          # <Phi_I|chi_m>
                ovl[w] = c.conj().T @ np.asarray(Bs[w])          # <chi_m|phi_k>
            # A axes: (state, w0, w1, w2, w3); spectators have one SPF
            sl = [slice(None)] + [slice(None) if s in (i, j) else 0
                                  for s in range(fw.NUM_WELLS)]
            Ap = A[tuple(sl)]                                    # (4, n_i, n_j)
            amp = np.einsum("akl,mk,nl->amn", Ap, ovl[i], ovl[j])
            wgt = np.abs(amp) ** 2
            res[tag] = wgt
        out[label] = {t: v.tolist() for t, v in res.items()}
        wf = res["final"]
        print(f"\n  {label}: final pair-state weights |<m|_i <n|_j psi_a>|^2 "
              f"(idle FCI products, m,n = 0..{NLEV - 1})", flush=True)
        for a_, lab in enumerate(L.PAIR_LABELS):
            top = sorted(((wf[a_, m, n], m, n) for m in range(NLEV)
                          for n in range(NLEV)), reverse=True)[:6]
            print(f"    column {lab}: total {wf[a_].sum():.4f}; "
                  + ", ".join(f"|{m}{n}> {v:.4f}" for v, m, n in top), flush=True)
        if "initial" in res:
            wi = res["initial"]
            print("    (t = 0: |11> column on |11> "
                  f"{wi[3, 1, 1]:.4f}, total {wi[3].sum():.4f})", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "project.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["levels", "project"])
    a = ap.parse_args()
    t0 = time.time()
    {"levels": levels, "project": project}[a.which](a)
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
