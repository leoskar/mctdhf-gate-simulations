"""The leak-partner levels of Fig. 7 in ML-MCTDHF's own relaxed
gate-point orbitals, w1w2, 80 points (an earlier study; ramp80.py gives Fig. 7(b)).

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.levels_own80 relax --norb 6 8 10
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.levels_own80 relax --norb 26 \
        --overlap gather
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.levels_own80 readout

**relax.** The four MF pair states (weights 1, 1/2, 1/2, 1/4; spectators
Hartree-Fock), relaxed at the static GATE-POINT Hamiltonian, x =
``mf_lib.x_vectors("w1w2")[1]`` (the gate's own target).

Start. The first route started
from the idle MF state and relaxed w1's excited SPF onto bare level 2-3
(discarded, ``results/levels_own80/discarded_idle_start/``). Now:

* norb 14: gates_80pt's gate-point register ``oppoint80_w1w2_state.npz`` (16
  states, norb 14 / n_s 4 everywhere, relaxed at this x). Active wells: its
  orbitals and its NATURAL SPFs (node density at its own weights), reordered
  so the SPFs labelled bare level 0 and 1 come first; spectators: the idle MF
  norb-14 state's Hartree-Fock orbitals (the spectator potential is the same
  at idle and gate). A: the four pair configurations one-hot on those SPFs.
* every other rung: the relaxed gate-point norb-14 state, projected (N < 14,
  ``exact_check.project_active``) or embedded (N > 14,
  ``exact_check.embed_orbitals``), as the idle ladder was built.

``exact_check.relax``'s settings (x64d, dt 2e-2, chunk 50, eps 1e-6, inner
tol 1e-4, cap 40000) with ``gates_80pt.mf80.relax_chunk``'s chunk-end stop on
every rung; both hard-wire the idle H, so the loop is re-implemented here
with H as an argument. Checks: converged; M(0) - 1 < 1e-10; |00> root
population on its configuration > 0.99 (the |10>/|01> roots hybridise at the
gate point, 0.55/0.45 in the gates_80pt state, and |11> mixes with its partners,
so their populations are recorded, not asserted); the LEVEL LABELS: in each
active well the two most occupied natural SPFs must carry bare levels 0 and
1 of the run's own orbital space (Hungarian on overlaps, each >= 0.9),
before (norb 14 start) and after every relaxation; then the energy order of
sum w_i E_i over the gate-point rungs present.
``--overlap gather`` (norb 24/26, CPU): ``orbital_convergence.overlap_patch`` with its spot
checks. -> results/levels_own80/gp_w1w2_n<N>.npz
``--tag _gpu`` (norb 26 on the GPU, racing the CPU
run): the same relaxation written to gp_w1w2_n26_gpu.npz (and the tagged
spot-check and label records), so the two runs never write the same file;
the energy check and the readout take the untagged file, else the _gpu one.

**readout.** ``gate_timing.level_check``'s truncated CI with the orbitals
swapped: exact single-well FCI at the gate point (``timing.FCIWell``, 80
points) for H; the N-orbital determinants embedded by the 3x3 minors of the
run's active-well orbitals b (80 x N, complex), H_N = T^+ H T. Per rung, per
orbital set (``idle``: the gates' start states; ``gp``: the relax output) and
active well: sorted levels E_0..E_7, each level's largest overlap with the
exact eigenvectors, and the principal angles between the two orbital sets;
then the pair detunings (E - E_11)/Delta_80. Checks first: (1) with the
proxy's natural orbitals the code reproduces levelcheck80/levels.json at
N = 13 and 26 (<= 1e-9); (2) on the norb-14 gate-point state the solver's own
species operator (``dressed80.species_ops``) gives the same levels
(<= 1e-8). -> results/levels_own80/summary.json
"""
import argparse
import itertools
import os
import time

from . import common as C  # noqa: I100  (x64 + redirections first)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gates_80pt import mf80  # noqa: E402

L, fw = C.L, C.fw
LEG = "w1w2"
WELLS = L.LEGS[LEG]                               # (1, 2)
RUNGS = [6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24, 26]
OUT = os.path.join(C.RESULTS, "levels_own80")
PAIRS = [(3, 0), (0, 3), (4, 0), (0, 4), (2, 0), (0, 2)]
NLEV = 8
TOL_CHUNK, CAP = 1e-10, 40000


def h_gate():
    _xi, xg = L.x_vectors(LEG)
    return fw.h_static(xg / fw.TYPICAL_FREQ_GHZ)


def gp_path(N, tag=""):
    return C.assert_writable(os.path.join(OUT, f"gp_{LEG}_n{N}{tag}.npz"))


def gp_file(N):
    """The gate-point state of rung N: the untagged file, else the _gpu one."""
    for tag in ("", "_gpu"):
        p = os.path.join(OUT, f"gp_{LEG}_n{N}{tag}.npz")
        if os.path.exists(p):
            return p
    return None


def make_ite(lay, inner_tol):
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt
    cfgs = L.pair_configs(LEG)
    return MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(), num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_mctdhf_orbitals=lay.norbs(LEG),
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_spfs=lay.nspfs(LEG), relax_configs=cfgs,
        relax_weights=list(L.RELAX_WEIGHTS), all_configs=cfgs, dt=2e-2,
        precision="x64d", inner_tol=inner_tol, chunk=50, eps=1e-6)


def relax_at(h, lay, y_start, inner_tol):
    """``gates_80pt.mf80.relax_chunk`` with the Hamiltonian ``h`` as an argument."""
    from mctdhf.MLMCTDHF_n_species import make_regauge_ml
    t0 = time.time()
    ite = make_ite(lay, inner_tol)
    ml = L.make_ml(LEG, h, lay)
    E0 = np.real(np.asarray(ml.calculate_energy(0.0, y_start)))
    A, Bs, bs = ml._y_to_ABb(y_start)
    state, steps, e_prev, converged, dch = (A, list(Bs), list(bs)), 0, None, \
        False, np.nan
    while steps < CAP:
        state, e, info = ite.integrate(ite.chunk, h, state)
        steps += len(e)
        if not np.all(np.isfinite(e)):
            break
        if e_prev is not None:
            dch = e[-1] - e_prev
            if abs(dch) < TOL_CHUNK:
                converged = True
                break
        e_prev = e[-1]
    print(f"  [relax-gp] {steps} it, converged {converged} (last chunk-end "
          f"dE {dch:+.2e}, last-iteration dE {e[-1] - e[-2]:+.2e}), "
          f"{time.time() - t0:.0f} s", flush=True)
    hh = ite.as_h(h)
    A, Bs, bs = state
    Bs_rg, _evals, evecs, ovl = ite._front_match(A, Bs, bs, hh)
    cols, pop = ite._assign(ovl)
    A = evecs[:, jnp.asarray(cols)].T.reshape(ml.a_shape)
    y = make_regauge_ml(ml)(ml._ABb_to_y(A, list(Bs_rg), list(bs)))
    E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    err = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y))
                              - np.eye(4))))
    print("  [relax-gp] E start (gate H)  " + " ".join(f"{v:.8f}" for v in E0))
    print("  [relax-gp] E relaxed         " + " ".join(f"{v:.8f}" for v in E)
          + f"   M(0)-1 {err:.1e}  match pop "
          + " ".join(f"{v:.6f}" for v in np.asarray(pop)), flush=True)
    assert converged and np.all(np.isfinite(E)), "relaxation failed"
    assert float(np.asarray(pop)[0]) > 0.99, f"|00> root not matched: {pop}"
    assert err < 1e-10, err
    return ml, y, ite, dict(steps=steps, match_pop=np.asarray(pop),
                            E_start=E0, M0_err=err, last_chunk_dE=float(dch))


def spf_labels(ite, h, st, nlev=6):
    """Natural SPFs (``ITE_opt``'s gauge and weights) of the active wells
    against the bare single-well levels of the state's own orbital space
    (``dressed80.species_ops``). Returns {well: (labels, O, occ)}, O[level,
    spf] = |<level|spf>|^2, labels[spf] = Hungarian level."""
    from scipy.optimize import linear_sum_assignment
    from . import dressed80 as D80
    A, Bs, bs = st
    _A, Bs_rg, _e, _v, occ = ite._analyse_dev(A, Bs, bs, ite.as_h(h))
    Ops = D80.species_ops(ite, bs, h)
    out = {}
    for s in WELLS:
        _ev, V = np.linalg.eigh(0.5 * (Ops[s] + Ops[s].conj().T))
        O = np.abs(V[:, :nlev].conj().T @ np.asarray(Bs_rg[s])) ** 2
        _r, c = linear_sum_assignment(-O.T)
        out[s] = ([int(v) for v in c], O, np.asarray(occ[s]))
    return out


def check_labels(ite, h, st, tag, tol=0.9):
    lab = spf_labels(ite, h, st)
    ok = True
    rec = {}
    for s, (c, O, occ) in lab.items():
        print(f"  [labels {tag}] w{s}: natural SPF -> level {c}, occupations "
              + " ".join(f"{v:.2e}" for v in occ), flush=True)
        for k in range(O.shape[1]):
            print(f"      SPF {k}: |<level|SPF>|^2 levels 0-5 "
                  + " ".join(f"{v:.4f}" for v in O[:, k]), flush=True)
        good = c[:2] == [0, 1] and O[0, 0] >= tol and O[1, 1] >= tol
        ok = ok and good
        rec[str(s)] = dict(labels=c, overlaps=O.tolist(), occ=occ.tolist(),
                           passed=bool(good))
    print(f"  [labels {tag}] -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok, rec


def gate_start14(h):
    """The norb-14 MF start from gates_80pt's gate-point register (docstring)."""
    lay = mf80.layout(14)
    p = os.path.join(C.G80_RESULTS, f"oppoint80_{LEG}_state.npz")
    d = np.load(p)
    assert int(d["num_dvr"]) == 80, p
    assert np.array_equal(np.asarray(d["x_ghz"], dtype=np.float64),
                          L.x_vectors(LEG)[1]), "oppoint x is not the gate x"
    A16 = np.asarray(d["A"])
    w16 = np.asarray(d["relax_weights"], dtype=np.float64)
    assert A16.shape == (16, 4, 4, 4, 4)
    idle = np.load(C.mf_state_path(LEG, 14))
    assert int(idle["num_dvr"]) == 80
    bs = [np.asarray(idle["b0"]), np.asarray(d["b1"]), np.asarray(d["b2"]),
          np.asarray(idle["b3"])]
    Bs = [np.asarray(idle["B0"]), None, None, np.asarray(idle["B3"])]
    for s in WELLS:
        rho = L._node_density(A16, s + 1, w16)
        ev, V = np.linalg.eigh(rho)
        o = np.argsort(ev)[::-1]
        ev, V = ev[o], V[:, o]
        B = np.asarray(d[f"B{s}"]) @ V                    # natural SPFs
        A_n = L._move(A16, V.conj(), s + 1)
        # exact: the expansion on this mode is unchanged
        dC = np.max(np.abs(np.tensordot(A_n, B, axes=([s + 1], [1]))
                           - np.tensordot(A16, np.asarray(d[f"B{s}"]),
                                          axes=([s + 1], [1]))))
        r2 = L._node_density(A_n, s + 1, w16)
        off = np.max(np.abs(r2 - np.diag(np.diag(r2))))
        print(f"  [start] w{s}: register natural occupations "
              + " ".join(f"{v:.3e}" for v in ev)
              + f"  (expansion change {dC:.1e}, off-diagonal {off:.1e})",
              flush=True)
        assert dC < 1e-12 and off < 1e-12
        Bs[s] = B
    ite = make_ite(lay, 1e-4)
    # order: the SPFs labelled level 0 and 1 first, the rest in natural order
    from . import dressed80 as D80
    Ops = D80.species_ops(ite, bs, h)
    for s in WELLS:
        _ev, V = np.linalg.eigh(0.5 * (Ops[s] + Ops[s].conj().T))
        O = np.abs(V[:, :6].conj().T @ Bs[s]) ** 2
        k0, k1 = int(np.argmax(O[0])), int(np.argmax(O[1]))
        assert k0 != k1, (s, O)
        order = [k0, k1] + [k for k in range(4) if k not in (k0, k1)]
        Bs[s] = Bs[s][:, order]
        print(f"  [start] w{s}: natural SPFs {order} -> slots 0..3 "
              f"(level-0 overlap {O[0, k0]:.4f}, level-1 {O[1, k1]:.4f})",
              flush=True)
    A = np.zeros((4,) + tuple(lay.nspfs(LEG)), dtype=np.complex128)
    i, j = WELLS
    for a, (m, n) in enumerate(L.PAIR_BITS):
        idx = [a, 0, 0, 0, 0]
        idx[i + 1], idx[j + 1] = m, n
        A[tuple(idx)] = 1.0
    ml = L.make_ml(LEG, h, lay)
    y = ml._ABb_to_y(jnp.asarray(A), [jnp.asarray(B, dtype=jnp.complex128)
                                      for B in Bs],
                     [jnp.asarray(b, dtype=jnp.complex128) for b in bs])
    M0 = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y))
                             - np.eye(4))))
    assert M0 < 1e-10, M0
    ok, _rec = check_labels(ite, h, ml._y_to_ABb(y), "start n14")
    if not ok:
        raise SystemExit("start label check FAILED (stop and report)")
    return ml, y


def weighted_E(path):
    d = np.load(path)
    return float(C.W4 @ np.real(np.asarray(d["E"])))


def require_energy_order(tol=1e-9):
    rows = [(N, weighted_E(gp_file(N))) for N in RUNGS if gp_file(N)]
    print("  [echeck-gp] norb  weighted E (gate H)  step", flush=True)
    bad = []
    for k, (N, E) in enumerate(rows):
        step = np.nan if k == 0 else E - rows[k - 1][1]
        if k and step > tol:
            bad.append((rows[k - 1][0], N, step))
        print(f"  [echeck-gp] {N:>4d}  {E:.10f}  {step:+.3e}"
              + ("   <-- INCREASES" if k and step > tol else ""), flush=True)
    if bad:
        raise SystemExit("ENERGY CHECK VIOLATED (stop and report): " + "; ".join(
            f"norb {a} -> {b}: {s:+.3e}" for a, b, s in bad))


def cmd_relax(a):
    os.makedirs(OUT, exist_ok=True)
    OP = None
    if a.overlap == "gather":
        assert len(a.norb) == 1 and a.norb[0] in (24, 26), \
            "--overlap gather is for one norb-24 or norb-26 rung"
        from . import overlap_patch as OP
        OP.install()
    h = h_gate()
    for N in a.norb:
        assert N in RUNGS, N
        out = gp_path(N, a.tag)
        t0 = time.time()
        lay = mf80.layout(N)
        print(f"\n=== gate-point relax {LEG} norb {lay.norbs(LEG)} n_s "
              f"{lay.nspfs(LEG)} ===", flush=True)
        if N == 14:
            ml0, y0 = gate_start14(h)
            how = "gates_80pt oppoint80 register: natural SPFs, HF spectators"
        else:
            lay14 = mf80.layout(14)
            ml14 = L.make_ml(LEG, h, lay14)
            d14 = np.load(gp_path(14))
            assert int(d14["num_dvr"]) == 80
            y14 = ml14._ABb_to_y(jnp.asarray(d14["A"]),
                                 [jnp.asarray(d14[f"B{s}"]) for s in range(4)],
                                 [jnp.asarray(d14[f"b{s}"]) for s in range(4)])
            if N < 14:
                _, y0 = C.EC.project_active(LEG, lay14, lay, src=(ml14, y14))
                how = "gate-point n14, projected (exact_check.project_active)"
            else:
                # embed_orbitals checks the embedding by the energy at the
                # idle H of BOTH states, so it needs an idle-H ml_small
                ml14i = L.make_ml(LEG, mf80.h_idle(LEG), lay14)
                y0 = C.EC.embed_orbitals(LEG, ml14i, y14, lay)
                how = "gate-point n14, embedded (exact_check.embed_orbitals)"
            ml0 = L.make_ml(LEG, h, lay)
        spots = []
        spot_path = C.assert_writable(os.path.join(
            OUT, f"overlap_spot_gp_n{N}{a.tag}.json"))
        if OP:
            spots.append(OP.spot_check(ml0, y0, OP.mixed(ml0, y0),
                                       f"n{N} start vs mixed"))
            C.save_json(spot_path, spots)
        ml, y, ite, extra = relax_at(h, lay, y0, a.inner_tol)
        if OP:
            spots.append(OP.spot_check(ml, y0, y, f"n{N} start vs relaxed"))
            C.save_json(spot_path, spots)
            extra["overlap"] = "gather (orbital_convergence.overlap_patch)"
            extra["overlap_spot_worst"] = max(r["worst"] for r in spots)
        E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
        print("  [relax-gp] excitations |10> |01> |11> rel. |00>: "
              + " ".join(f"{v - E[0]:.5f}" for v in E[1:])
              + "   (exact single-well level 1: 4.03697 / 4.03864)",
              flush=True)
        ok, labels = check_labels(ite, h, ml._y_to_ABb(y), f"relaxed n{N}")
        extra["labels_passed"] = ok
        extra["start"] = how
        pops = ml.spf_populations(y)
        for s in WELLS:
            print(f"  [relax-gp] w{s} SPF populations "
                  + " ".join(f"{float(v):.2e}" for v in np.asarray(pops[s])),
                  flush=True)
        mf80.save_state(out, ml, y, source=how + " + x64d relax at the gate "
                        "point (chunk-end rule)", leg=LEG, norb=N,
                        x_ghz=L.x_vectors(LEG)[1], inner_tol=a.inner_tol,
                        wall=time.time() - t0, **extra)
        C.save_json(C.assert_writable(os.path.join(
            OUT, f"labels_gp_n{N}{a.tag}.json")), labels)
        print(f"  saved -> {out}  ({time.time() - t0:.0f} s)", flush=True)
        if not ok:
            raise SystemExit(f"label check FAILED at norb {N} (stop and "
                             "report; the state is saved for inspection)")
        require_energy_order()


# ── readout ───────────────────────────────────────────────────────────────

def truncated_ci(fci, H, U, V):
    """Levels of H in the 3-particle determinants of the orbitals U (n x N),
    and each level's largest overlap with the exact eigenvectors V."""
    from gate_timing.level_check import minors
    N = U.shape[1]
    occ = np.array(list(itertools.combinations(range(N), 3)))
    T = minors(U, fci.dets, occ)
    S = T.conj().T @ T
    serr = float(np.max(np.abs(S - np.eye(len(occ)))))
    assert serr < 1e-10, serr
    HN = T.conj().T @ (H @ T)
    e, w = np.linalg.eigh(0.5 * (HN + HN.conj().T))
    e, w = e[:NLEV], w[:, :NLEV]
    ov = np.abs(V.T @ (T @ w)) ** 2                 # (exact level, CI level)
    return e, ov, serr


def pair_dets(Ei, Ej):
    return {f"{m}{n}": (Ei[m] - Ei[1]) + (Ej[n] - Ej[1]) for m, n in PAIRS}


def cmd_readout(a):
    from gate_timing import level_check as LCK
    from gate_timing.timing import FCIWell, h_well
    from . import dressed80 as D80
    t0 = time.time()
    Delta = C.delta80(LEG)
    fci = FCIWell()
    assert fci.n == 80, fci.n
    _xi, xg = L.x_vectors(LEG)
    x = xg / fw.TYPICAL_FREQ_GHZ
    exact = {}
    for w in WELLS:
        H, E, V, r = LCK.fci_levels(fci, h_well(w, x[w], x[4 + w]))
        exact[w] = (H, E, V)
        print(f"  [fci] w{w} gate point: levels rel. E0 "
              + " ".join(f"{e - E[0]:.5f}" for e in E) + f" (resid {r:.1e})",
              flush=True)
    out = {"leg": LEG, "Delta_80": Delta, "x_ghz": xg.tolist(), "checks": {},
           "exact": {}, "rungs": {}}
    for w in WELLS:
        out["exact"][str(w)] = exact[w][1].tolist()

    # check 1: the proxy's orbitals reproduce levelcheck80
    ref = C.load_json(os.path.join(C.RESULTS, "levelcheck80", "levels.json"))
    worst = 0.0
    for w in WELLS:
        H, E, V = exact[w]
        Dm = (2 / 3) * LCK.one_rdm(fci, V[:, 0]) + (1 / 3) * LCK.one_rdm(fci, V[:, 1])
        occ, U = np.linalg.eigh(Dm)
        U = U[:, np.argsort(occ)[::-1]]
        for N in (13, 26):
            e, _ov, _s = truncated_ci(fci, H, U[:, :N].astype(np.complex128), V)
            r = np.asarray(ref[LEG]["gate"][str(w)]["E_N"][str(N)])
            d = float(np.max(np.abs(e[:len(r)] - r)))
            worst = max(worst, d)
            print(f"  [check 1] w{w} N {N}: proxy orbitals vs levels.json "
                  f"max diff {d:.1e}", flush=True)
    ok1 = worst <= 1e-9
    out["checks"]["proxy_reproduced"] = dict(worst=worst, passed=ok1)
    print(f"  [check 1] worst {worst:.1e} -> {'PASS' if ok1 else 'FAIL'}",
          flush=True)
    if not ok1:
        C.save_json(C.assert_writable(os.path.join(OUT, "summary.json")), out)
        raise SystemExit("readout check 1 FAILED (stop and report)")

    # check 2: the solver's own species operator, norb-14 gate-point state
    d14 = np.load(os.path.join(OUT, f"gp_{LEG}_n14.npz"))
    bs14 = [np.asarray(d14[f"b{s}"]) for s in range(4)]
    ite = make_ite(mf80.layout(14), 1e-4)
    Ops = D80.species_ops(ite, bs14, h_gate())
    worst = 0.0
    for w in WELLS:
        H, E, V = exact[w]
        e, _ov, _s = truncated_ci(fci, H, bs14[w], V)
        e2 = np.linalg.eigvalsh(0.5 * (Ops[w] + Ops[w].conj().T))[:NLEV]
        d = float(np.max(np.abs(e - e2)))
        worst = max(worst, d)
        print(f"  [check 2] w{w} n14 gate-point orbitals: FCIWell route vs "
              f"ml_species_ops max diff {d:.1e}", flush=True)
    ok2 = worst <= 1e-8
    out["checks"]["species_ops_route"] = dict(worst=worst, passed=ok2)
    print(f"  [check 2] worst {worst:.1e} -> {'PASS' if ok2 else 'FAIL'}",
          flush=True)
    if not ok2:
        C.save_json(C.assert_writable(os.path.join(OUT, "summary.json")), out)
        raise SystemExit("readout check 2 FAILED (stop and report)")

    # the rungs
    print(f"\n  (E - E11)/Delta_80, Delta_80 = {Delta:.6e}", flush=True)
    print("  N   set  " + "".join(f"{'|' + k + '>':>10}" for k in
                                  ("30", "03", "40", "04")), flush=True)
    for N in RUNGS:
        files = {"idle": C.mf_state_path(LEG, N),
                 "gp": gp_file(N)}
        rec, bset = {}, {}
        for kind, p in files.items():
            d = np.load(p)
            assert int(d["num_dvr"]) == 80, p
            bset[kind] = {w: np.asarray(d[f"b{w}"]) for w in WELLS}
            lev = {}
            for w in WELLS:
                H, E, V = exact[w]
                b = bset[kind][w]
                assert b.shape == (80, N), (p, b.shape)
                e, ov, serr = truncated_ci(fci, H, b, V)
                lev[w] = dict(E=e.tolist(), label=np.argmax(ov, axis=0).tolist(),
                              label_weight=np.max(ov, axis=0).tolist(),
                              S_err=serr)
            pdet = pair_dets(lev[WELLS[0]]["E"], lev[WELLS[1]]["E"])
            rec[kind] = dict(source=os.path.relpath(p, C.HERE), levels={
                str(w): lev[w] for w in WELLS},
                pair_over_Delta={k: v / Delta for k, v in pdet.items()},
                E_states=np.real(np.asarray(d["E"])).tolist())
            print(f"  {N:<3d} {kind:<4s} " + "".join(
                f"{rec[kind]['pair_over_Delta'][k]:>+10.4f}"
                for k in ("30", "03", "40", "04")), flush=True)
        ang = {}
        for w in WELLS:
            sv = np.linalg.svd(bset["idle"][w].conj().T @ bset["gp"][w],
                               compute_uv=False)
            ang[str(w)] = dict(min_cos=float(sv.min()),
                               max_sin=float(np.sqrt(max(0.0, 1 - sv.min() ** 2))))
        rec["principal_angles_idle_vs_gp"] = ang
        out["rungs"][str(N)] = rec
    ex = pair_dets(out["exact"][str(WELLS[0])], out["exact"][str(WELLS[1])])
    out["exact_pair_over_Delta"] = {k: v / Delta for k, v in ex.items()}
    print("  FCI      " + "".join(f"{out['exact_pair_over_Delta'][k]:>+10.4f}"
                                  for k in ("30", "03", "40", "04")), flush=True)
    C.save_json(C.assert_writable(os.path.join(OUT, "summary.json")), out)
    print(f"  wrote {os.path.join(OUT, 'summary.json')} "
          f"({time.time() - t0:.0f} s)", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["relax", "readout", "echeck"])
    p.add_argument("--norb", type=int, nargs="+", default=[])
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--overlap", choices=["library", "gather"],
                   default="library")
    p.add_argument("--tag", choices=["", "_gpu"], default="")
    a = p.parse_args(argv)
    C.require_grid(80)
    L.device_banner()
    t0 = time.time()
    {"relax": cmd_relax, "readout": cmd_readout,
     "echeck": lambda _a: require_energy_order()}[a.cmd](a)
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
