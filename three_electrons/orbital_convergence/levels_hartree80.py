"""The leak-partner levels of Fig. 7 with the inter-well
Hartree interaction, w1w2 gate point, 80 points (CPU).

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.levels_hartree80 run

Isolated-well levels (levels_own80.py) leave this out. Here, per active well w (w1, w2):

* H_w = h_well(w) + v_H^spec, v_H^spec = sum_{s in 0,3} G_ws n_s, where n_s is
  the exact isolated-well FCI ground-state density of spectator s at the gate
  point (fixed);
* "hartree" pair levels (first order, the diagonal <mn|H|mn> in products of
  the spectator-dressed single-well levels):
  E_mn - E_11 = (e1_m - e1_1) + (e2_n - e2_1) + (J_mn - J_11),
  J_mn = n1_m . G_12 . n2_n (no inter-well exchange: the wells are species);
* "frozen" (``timing.isolated_study``'s dressed convention): each active well
  in the field of the spectators plus the partner's ground-state density; the
  pair level is the sum of the two single-well levels.

Orbital sets per N: "proxy" (the N best natural orbitals of the exact
2/3 |0> + 1/3 |1> ensemble of the dressed H_w, N = 6..26), "proxy_bareNO"
(levelcheck80's isolated-well natural orbitals, dressed H), "gp" (levels_own80.py's gate-point
ML-MCTDHF orbitals, rungs 6..26). Sorted levels.

Checks: with the Hartree terms off the code reproduces levelcheck80/levels.json
(proxy_bareNO, all N) and levels_own80/summary.json (exact and gp levels, all
rungs) to <= 1e-9; FCI residuals <= 1e-8; densities sum to 3 (1e-10);
T^+T = 1 (1e-10). -> results/levels_hartree80/summary.json

**Result: the first-order "hartree" pairs are INVALID.** The
partner's field has off-diagonal elements up to ~80 Delta between the
spectator-dressed levels (``scf``), so first order fails (|40>/|04> -0.4
Delta off). The partner field is state-independent to 0.005 Delta, so
"frozen" equals the self-consistent Hartree of ``scf`` to 1e-3 Delta: use
"frozen". The "proxy" set's natural orbitals come from the spectator-only
H, which leaves out that field. "proxy_bareNO" converges faster in the
frozen H.

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.levels_hartree80 scf
"""
import argparse
import itertools
import os
import time

from . import common as C  # noqa: I100  (x64 + redirections first)

import numpy as np  # noqa: E402

from . import levels_own80 as LO  # noqa: E402

L, fw = C.L, C.fw
LEG = "w1w2"
W1, W2 = L.LEGS[LEG]                               # (1, 2)
SPEC = (0, 3)
NS_PROXY = list(range(6, 27))
RUNGS = LO.RUNGS
OUT = os.path.join(C.RESULTS, "levels_hartree80")
PAIRS = [(3, 0), (0, 3), (4, 0), (0, 4), (2, 0), (0, 2)]
SHOW = ("30", "03", "40", "04")
NLEV = 8
TOL_CHECK = 1e-9


def dens_of(fci, X):
    """Grid occupations of each column of X (full-space FCI vectors)."""
    X = np.atleast_2d(X.T).T
    out = np.zeros((X.shape[1], fci.n))
    for k in range(X.shape[1]):
        w2 = np.abs(X[:, k]) ** 2
        for a in range(3):
            out[k] += np.bincount(fci.dets[:, a], weights=w2, minlength=fci.n)
    return out


def embed(fci, U):
    from gate_timing.level_check import minors
    N = U.shape[1]
    occ = np.array(list(itertools.combinations(range(N), 3)))
    T = minors(U, fci.dets, occ)
    serr = float(np.max(np.abs(T.conj().T @ T - np.eye(len(occ)))))
    assert serr < 1e-10, serr
    return T, serr


def ci(T, H, V):
    """Lowest NLEV levels of H in span(T), their full-space vectors, and their
    largest overlaps with the exact eigenvectors V of the same H."""
    HN = T.conj().T @ (H @ T)
    e, w = np.linalg.eigh(0.5 * (HN + HN.conj().T))
    e, w = e[:NLEV], w[:, :NLEV]
    X = T @ w
    ov = np.abs(V.T @ X) ** 2
    return e, X, dict(label=np.argmax(ov, axis=0).tolist(),
                      label_weight=np.max(ov, axis=0).tolist())


def pairs_plain(E1, E2):
    return {f"{m}{n}": (E1[m] - E1[1]) + (E2[n] - E2[1]) for m, n in PAIRS}


def pairs_hartree(E1, E2, J):
    return {f"{m}{n}": (E1[m] - E1[1]) + (E2[n] - E2[1]) + (J[m, n] - J[1, 1])
            for m, n in PAIRS}


def over(d, Delta):
    return {k: float(v / Delta) for k, v in d.items()}


def run(a):
    from gate_timing import level_check as LCK
    from gate_timing.timing import FCIWell, h_well, hartree, _V_pair
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    out_path = C.assert_writable(os.path.join(OUT, "summary.json"))
    Delta = C.delta80(LEG)
    fci = FCIWell()
    assert fci.n == 80, fci.n
    _xi, xg = L.x_vectors(LEG)
    x = xg / fw.TYPICAL_FREQ_GHZ
    out = {"leg": LEG, "Delta_80": Delta, "x_ghz": xg.tolist(), "checks": {},
           "model": "inter-well Hartree", "exact": {}, "sets": {}}
    worst_fci, worst_n = 0.0, 0.0

    # ── spectators: exact isolated ground-state densities ─────────────────
    nspec = {}
    for s in SPEC:
        _H, E, V, r = LCK.fci_levels(fci, h_well(s, x[s], x[4 + s]), k=2)
        nspec[s] = dens_of(fci, V[:, :1])[0]
        worst_fci = max(worst_fci, r)
        worst_n = max(worst_n, abs(nspec[s].sum() - 3))
    # the alternative: HF spectators of the gate-point norb-14 ML state
    d14 = np.load(LO.gp_file(14))
    nspec_ml = {s: np.sum(np.abs(np.asarray(d14[f"b{s}"])) ** 2, axis=1)
                for s in SPEC}
    for s in SPEC:
        worst_n = max(worst_n, abs(nspec_ml[s].sum() - 3))

    # ── active wells: bare / dressed Hamiltonians, exact levels ────────────
    G12 = np.asarray(_V_pair(W1, W2))
    Hb, Hd, Vb, Vd, Ed, Eb, vH, vH_ml = {}, {}, {}, {}, {}, {}, {}, {}
    for w in (W1, W2):
        vH[w] = hartree(w, nspec)
        vH_ml[w] = hartree(w, nspec_ml)
        Hb[w], Eb[w], Vb[w], r1 = LCK.fci_levels(fci, h_well(w, x[w], x[4 + w]))
        Hd[w], Ed[w], Vd[w], r2 = LCK.fci_levels(
            fci, h_well(w, x[w], x[4 + w], vH[w]))
        worst_fci = max(worst_fci, r1, r2)
        print(f"  [v_H^spec] w{w}: mean {vH[w].mean():.5f}, spread over the "
              f"grid {np.ptp(vH[w]):.5f}; ML-spectator variant differs by "
              f"max {np.max(np.abs(vH_ml[w] - vH[w])):.2e}", flush=True)
        print(f"  [fci] w{w}: bare excitations "
              + " ".join(f"{e - Eb[w][0]:.5f}" for e in Eb[w][1:6])
              + "\n         dressed           "
              + " ".join(f"{e - Ed[w][0]:.5f}" for e in Ed[w][1:6]), flush=True)
    nd = {w: dens_of(fci, Vd[w]) for w in (W1, W2)}
    worst_n = max(worst_n, float(np.max(np.abs(nd[W1].sum(1) - 3))),
                  float(np.max(np.abs(nd[W2].sum(1) - 3))))
    Jex = nd[W1] @ G12 @ nd[W2].T

    # frozen: the partner's ground density added to the field
    def frozen_H(w, partner_n0):
        nn = dict(nspec)
        nn[W2 if w == W1 else W1] = partner_n0
        return fci.hamiltonian(h_well(w, x[w], x[4 + w], hartree(w, nn)))
    Hf = {W1: frozen_H(W1, nd[W2][0]), W2: frozen_H(W2, nd[W1][0])}
    Ef = {}
    for w in (W1, W2):
        from scipy.sparse.linalg import eigsh
        e, V = eigsh(Hf[w], k=NLEV, which="SA", tol=1e-12, ncv=48,
                     v0=np.ones(fci.D) / np.sqrt(fci.D))
        Ef[w] = np.sort(e)
    # the ML-spectator variant of the exact "hartree" result
    Eml, ndml = {}, {}
    for w in (W1, W2):
        _H, Eml[w], V, r = LCK.fci_levels(fci, h_well(w, x[w], x[4 + w], vH_ml[w]))
        ndml[w] = dens_of(fci, V)
        worst_fci = max(worst_fci, r)
    out["exact"] = dict(
        bare=over(pairs_plain(Eb[W1], Eb[W2]), Delta),
        hartree=over(pairs_hartree(Ed[W1], Ed[W2], Jex), Delta),
        frozen=over(pairs_plain(Ef[W1], Ef[W2]), Delta),
        hartree_ml_spectators=over(pairs_hartree(
            Eml[W1], Eml[W2], ndml[W1] @ G12 @ ndml[W2].T), Delta),
        J_over_Delta=((Jex - Jex[1, 1]) / Delta)[:5, :5].tolist(),
        E_bare={str(w): Eb[w].tolist() for w in (W1, W2)},
        E_dressed={str(w): Ed[w].tolist() for w in (W1, W2)})
    for k in ("bare", "hartree", "frozen", "hartree_ml_spectators"):
        print(f"  [exact {k:>21s}] " + " ".join(
            f"|{p}> {out['exact'][k][p]:+.4f}" for p in SHOW), flush=True)

    # ── check: Hartree off reproduces the earlier readouts ────────────────
    ref_lc = C.load_json(os.path.join(C.RESULTS, "levelcheck80", "levels.json")
                         )[LEG]["gate"]
    ref_own = C.load_json(os.path.join(C.RESULTS, "levels_own80", "summary.json"))
    chk = 0.0
    for w in (W1, W2):
        chk = max(chk, float(np.max(np.abs(
            Eb[w] - np.asarray(ref_own["exact"][str(w)])))))

    # ── the orbital sets ───────────────────────────────────────────────────
    def no_of(V):
        Dm = (2 / 3) * LCK.one_rdm(fci, V[:, 0]) + (1 / 3) * LCK.one_rdm(fci, V[:, 1])
        occ, U = np.linalg.eigh(Dm)
        return U[:, np.argsort(occ)[::-1]].astype(np.complex128)
    U_dr = {w: no_of(Vd[w]) for w in (W1, W2)}
    U_bare = {w: no_of(Vb[w]) for w in (W1, W2)}

    def gp_b(N):
        d = np.load(LO.gp_file(N))
        assert int(d["num_dvr"]) == 80
        return {w: np.asarray(d[f"b{w}"]) for w in (W1, W2)}

    sets = [("proxy", NS_PROXY, lambda N: {w: U_dr[w][:, :N] for w in (W1, W2)}),
            ("proxy_bareNO", NS_PROXY,
             lambda N: {w: U_bare[w][:, :N] for w in (W1, W2)}),
            ("gp", RUNGS, gp_b)]
    print(f"\n  (E - E11)/Delta_80, Delta_80 = {Delta:.6e}; 'hartree' pairs",
          flush=True)
    for name, Ns, orbs in sets:
        recs = {}
        for N in Ns:
            U = orbs(N)
            lev, X, Ts = {}, {}, {}
            for w in (W1, W2):
                assert U[w].shape == (80, N), U[w].shape
                T, serr = embed(fci, U[w])
                Ts[w] = T
                e, X[w], lab = ci(T, Hd[w], Vd[w])
                lev[w] = dict(E=e.tolist(), S_err=serr, **lab)
                if name == "proxy_bareNO":          # check vs levelcheck80
                    eb, _X, _l = ci(T, Hb[w], Vb[w])
                    r = np.asarray(ref_lc[str(w)]["E_N"][str(N)])
                    chk = max(chk, float(np.max(np.abs(eb[:len(r)] - r))))
                if name == "gp":                    # check vs levels_own80
                    eb, _X, _l = ci(T, Hb[w], Vb[w])
                    r = np.asarray(ref_own["rungs"][str(N)]["gp"]["levels"][str(w)]["E"])
                    chk = max(chk, float(np.max(np.abs(eb - r))))
                del T          # kept in Ts for the frozen variant
            n1, n2 = dens_of(fci, X[W1]), dens_of(fci, X[W2])
            worst_n = max(worst_n, float(np.max(np.abs(n1.sum(1) - 3))),
                          float(np.max(np.abs(n2.sum(1) - 3))))
            J = n1 @ G12 @ n2.T
            E1, E2 = np.asarray(lev[W1]["E"]), np.asarray(lev[W2]["E"])
            # frozen: the partner's ground density of the same set
            fz = {}
            for w, part in ((W1, n2[0]), (W2, n1[0])):
                T = Ts.pop(w)
                HN = T.conj().T @ (frozen_H(w, part) @ T)
                fz[w] = np.linalg.eigvalsh(0.5 * (HN + HN.conj().T))[:NLEV]
                del T
            recs[str(N)] = dict(
                levels={str(w): lev[w] for w in (W1, W2)},
                hartree=over(pairs_hartree(E1, E2, J), Delta),
                no_J=over(pairs_plain(E1, E2), Delta),
                frozen=over(pairs_plain(fz[W1], fz[W2]), Delta))
            print(f"  {name:<12s} {N:<3d} " + "".join(
                f"{recs[str(N)]['hartree'][p]:>+10.4f}" for p in SHOW)
                + f"   [{time.time() - t0:.0f} s]", flush=True)
        out["sets"][name] = recs
        out["checks"]["hartree_off_reproduces"] = dict(worst=chk)
        C.save_json(out_path, out)
    print("  exact        " + "".join(f"{out['exact']['hartree'][p]:>+10.4f}"
                                     for p in SHOW), flush=True)
    ok = (chk <= TOL_CHECK and worst_fci <= 1e-8 and worst_n <= 1e-10)
    out["checks"].update(hartree_off_reproduces=dict(worst=chk, tol=TOL_CHECK),
                         fci_resid=worst_fci, density_sum=worst_n, passed=ok)
    C.save_json(out_path, out)
    print(f"  [checks] Hartree off vs earlier readouts {chk:.1e} (tol "
          f"{TOL_CHECK:.0e}); FCI resid {worst_fci:.1e}; density sums "
          f"{worst_n:.1e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"  wrote {out_path} ({time.time() - t0:.0f} s)", flush=True)
    if not ok:
        raise SystemExit("CHECK FAILED (stop and report)")


def scf(a):
    """Exact reference only: self-consistent Hartree per pair configuration.

    For |mn>: w1 is FCI in the field of the spectators plus w2's density in its
    level n, w2 in the field of the spectators plus w1's density in its level
    m (sorted roots), iterated to |dE| < 1e-10; E_mn = e1_m + e2_n - n1.G.n2
    (the inter-well direct term counted once). Also: the partner-field
    matrices <m|G n_partner|m'> in the spectator-dressed levels, which show
    whether the first-order "hartree" pairs of ``run`` are reliable."""
    from scipy.sparse.linalg import eigsh
    from gate_timing import level_check as LCK
    from gate_timing.timing import FCIWell, h_well, hartree, _V_pair
    t0 = time.time()
    out_path = C.assert_writable(os.path.join(OUT, "scf.json"))
    os.makedirs(OUT, exist_ok=True)
    Delta = C.delta80(LEG)
    fci = FCIWell()
    assert fci.n == 80, fci.n
    x = L.x_vectors(LEG)[1] / fw.TYPICAL_FREQ_GHZ
    nspec = {}
    for s in SPEC:
        _H, _E, V, _r = LCK.fci_levels(fci, h_well(s, x[s], x[4 + s]), k=2)
        nspec[s] = dens_of(fci, V[:, :1])[0]
    G12 = np.asarray(_V_pair(W1, W2))

    def solve(w, partner_n, k, v0=None):
        nn = dict(nspec)
        if partner_n is not None:
            nn[W2 if w == W1 else W1] = partner_n
        H = fci.hamiltonian(h_well(w, x[w], x[4 + w], hartree(w, nn)))
        if v0 is None:
            v0 = np.ones(fci.D) / np.sqrt(fci.D)
        E, V = eigsh(H, k=k, which="SA", tol=1e-12, ncv=48, v0=v0)
        o = np.argsort(E)
        E, V = E[o], V[:, o]
        r = float(np.max(np.linalg.norm(H @ V - V * E, axis=0)))
        assert r < 1e-8, r
        return E, V

    Es, Vs, nd = {}, {}, {}
    for w in (W1, W2):
        Es[w], Vs[w] = solve(w, None, NLEV)
        nd[w] = dens_of(fci, Vs[w])

    # partner-field matrices in the spectator-dressed levels (units of Delta)
    out = {"leg": LEG, "Delta_80": Delta, "field": {}, "pairs": {}}
    for w, other in ((W1, W2), (W2, W1)):
        G = G12 if w == W1 else G12.T
        for n in (0, 1):
            v = G @ nd[other][n]
            M = np.zeros((5, 5))
            for i in range(5):
                for j in range(5):
                    w2 = Vs[w][:, i] * Vs[w][:, j]
                    t = np.zeros(fci.n)
                    for aa in range(3):
                        t += np.bincount(fci.dets[:, aa], weights=w2,
                                         minlength=fci.n)
                    M[i, j] = t @ v
            Md = (M - np.diag(np.full(5, M[1, 1]))) / Delta
            out["field"][f"w{w}_partner{n}"] = Md.tolist()
            print(f"  [field] w{w} in w{other}'s level-{n} field: <m|v|m'> - "
                  f"<1|v|1>, units of Delta (levels 0-4)", flush=True)
            for i in range(5):
                print("     " + " ".join(f"{Md[i, j]:+9.4f}" for j in range(5)),
                      flush=True)
        g = (Es[w][1:5] - Es[w][:4]) / Delta
        print(f"  [gaps] w{w} spectator-dressed level gaps / Delta: "
              + " ".join(f"{v:.3f}" for v in g), flush=True)

    configs = [(1, 1)] + PAIRS
    E_cfg = {}
    for m, n in configs:
        n1, n2 = nd[W1][m], nd[W2][n]
        v1 = v2 = None
        hist = []
        for it in range(40):
            E1, V1 = solve(W1, n2, max(m, n) + 4, v1)
            v1 = V1[:, m]
            n1 = dens_of(fci, V1[:, m:m + 1])[0]
            E2, V2 = solve(W2, n1, max(m, n) + 4, v2)
            v2 = V2[:, n]
            n2 = dens_of(fci, V2[:, n:n + 1])[0]
            E = E1[m] + E2[n] - n1 @ G12 @ n2
            hist.append(float(E))
            if it and abs(hist[-1] - hist[-2]) < 1e-10:
                break
        ov1 = float(abs(Vs[W1][:, m] @ V1[:, m]) ** 2)
        ov2 = float(abs(Vs[W2][:, n] @ V2[:, n]) ** 2)
        E_cfg[(m, n)] = dict(E=hist[-1], E_oneshot=hist[0], iters=len(hist),
                             converged=len(hist) < 40, overlap_w1=ov1,
                             overlap_w2=ov2)
        print(f"  [scf] |{m}{n}>: {len(hist)} it, dE last "
              f"{hist[-1] - hist[-2] if len(hist) > 1 else 0:+.1e}, overlap "
              f"with the spectator-dressed levels {ov1:.4f} / {ov2:.4f} "
              f"[{time.time() - t0:.0f} s]", flush=True)
    E11 = E_cfg[(1, 1)]
    for m, n in PAIRS:
        r = E_cfg[(m, n)]
        out["pairs"][f"{m}{n}"] = dict(
            scf=(r["E"] - E11["E"]) / Delta,
            oneshot=(r["E_oneshot"] - E11["E_oneshot"]) / Delta, **r)
    out["E11"] = E11
    C.save_json(out_path, out)
    print("  scf      " + " ".join(f"|{p}> {out['pairs'][p]['scf']:+.4f}"
                                   for p in SHOW), flush=True)
    print("  one-shot " + " ".join(f"|{p}> {out['pairs'][p]['oneshot']:+.4f}"
                                   for p in SHOW), flush=True)
    ok = all(r["converged"] for r in E_cfg.values())
    print(f"  wrote {out_path} ({time.time() - t0:.0f} s); all converged: {ok}",
          flush=True)
    if not ok:
        raise SystemExit("SCF did not converge (stop and report)")


def frozen_no(a):
    """The levelcheck80 levels redone consistently in the frozen-field H: the N best natural
    orbitals of the exact 2/3 |0> + 1/3 |1> ensemble of H_w + v_H(spectators
    + the partner's ground density), levels of that H, N = 6..26.
    -> results/levels_hartree80/frozen_no.json"""
    from scipy.sparse.linalg import eigsh
    from gate_timing import level_check as LCK
    from gate_timing.timing import FCIWell, h_well, hartree
    t0 = time.time()
    out_path = C.assert_writable(os.path.join(OUT, "frozen_no.json"))
    Delta = C.delta80(LEG)
    fci = FCIWell()
    assert fci.n == 80, fci.n
    x = L.x_vectors(LEG)[1] / fw.TYPICAL_FREQ_GHZ
    nspec = {}
    for s in SPEC:
        _H, _E, V, _r = LCK.fci_levels(fci, h_well(s, x[s], x[4 + s]), k=2)
        nspec[s] = dens_of(fci, V[:, :1])[0]

    def H_of(w, partner=None):
        nn = dict(nspec)
        if partner is not None:
            nn[W2 if w == W1 else W1] = partner
        return fci.hamiltonian(h_well(w, x[w], x[4 + w], hartree(w, nn)))

    def exact(H):
        E, V = eigsh(H, k=NLEV, which="SA", tol=1e-12, ncv=48,
                     v0=np.ones(fci.D) / np.sqrt(fci.D))
        o = np.argsort(E)
        r = float(np.max(np.linalg.norm(H @ V[:, o] - V[:, o] * E[o], axis=0)))
        assert r < 1e-8, r
        return E[o], V[:, o]
    n0 = {w: dens_of(fci, exact(H_of(w))[1][:, :1])[0] for w in (W1, W2)}
    Hf = {W1: H_of(W1, n0[W2]), W2: H_of(W2, n0[W1])}
    ex, U = {}, {}
    for w in (W1, W2):
        ex[w] = exact(Hf[w])
        V = ex[w][1]
        Dm = (2 / 3) * LCK.one_rdm(fci, V[:, 0]) + (1 / 3) * LCK.one_rdm(fci, V[:, 1])
        occ, Uw = np.linalg.eigh(Dm)
        U[w] = Uw[:, np.argsort(occ)[::-1]].astype(np.complex128)
    out = {"leg": LEG, "Delta_80": Delta, "exact": over(
        pairs_plain(ex[W1][0], ex[W2][0]), Delta), "N": {}}
    print("  exact frozen " + " ".join(f"|{p}> {out['exact'][p]:+.4f}"
                                       for p in SHOW), flush=True)
    for N in NS_PROXY:
        lev = {}
        for w in (W1, W2):
            T, serr = embed(fci, U[w][:, :N])
            e, _X, lab = ci(T, Hf[w], ex[w][1])
            lev[w] = dict(E=e.tolist(), S_err=serr, **lab)
            del T
        out["N"][str(N)] = dict(levels={str(w): lev[w] for w in (W1, W2)},
                                pair_over_Delta=over(pairs_plain(
                                    np.asarray(lev[W1]["E"]),
                                    np.asarray(lev[W2]["E"])), Delta))
        C.save_json(out_path, out)
        print(f"  frozen-NO {N:<3d} " + "".join(
            f"{out['N'][str(N)]['pair_over_Delta'][p]:>+10.4f}" for p in SHOW)
            + f"   labels w1 {lev[W1]['label'][:5]} [{time.time() - t0:.0f} s]",
            flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "scf", "frozen_no"])
    a = p.parse_args(argv)
    C.require_grid(80)
    t0 = time.time()
    {"run": run, "scf": scf, "frozen_no": frozen_no}[a.cmd](a)
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
