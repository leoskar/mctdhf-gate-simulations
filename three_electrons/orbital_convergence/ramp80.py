"""Fig. 7(b): the leak-partner levels in the orbitals the
w1w2 theta = pi/2 gate actually has during its ramp-up and early hold, 80 points.

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ramp80 prop --norb 6 8 10    # GPU
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ramp80 prop --norb 26 --overlap gather
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ramp80 readout --norb 6      # CPU

**prop.** The theta = pi/2 gate of each ladder rung (``ladder80 --pulse p3``:
tau_ramp 9.4, tau_hold 126.1, MF
layout, ``Integ()`` defaults), rerun from the gate file's own ``y0`` with the
same envelope, Hamiltonian and ``mf_lib.Propagator`` (the step times come
from the global step index, so stopping early leaves the trajectory
unchanged), stopped at T_STOP. The states at SAVE are written FIRST
(``results/ramp80/ramp_w1w2_n<N>.npz``: the active-well orbitals b1, b2 and
the full y per time). Then the check: the retention at every 2 t.u. up to
T_STOP must reproduce the gate file's ``ret_t`` within TOL_RET = 1e-4, a
guard against setup errors, which move it by >= 1e-2 at t = 2, 4 (norb 6 differed by 1.0e-7, mixed-precision noise).

**readout.** Per saved time, ``levels_hartree80``'s machinery on the run's
active-well orbitals: sorted levels and pair detunings (E - E_11)/Delta_80
for the isolated wells ("bare") and with the inter-well Hartree field
("frozen": spectators' exact ground densities plus the partner's ground
density in the same orbital space; equal to self-consistent Hartree within
1e-3 Delta at the exact level, levels_hartree80.py); labels against the exact
eigenvectors of the same H; principal angles against the levels_own80.py gate-point
orbitals and the idle orbitals. -> results/ramp80/levels_n<N>.json
"""
import argparse
import dataclasses
import json
import os
import time

from . import common as C  # noqa: I100  (x64 + redirections first)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gates_80pt import mf80  # noqa: E402

L, fw = C.L, C.fw
LEG = "w1w2"
W1, W2 = L.LEGS[LEG]
TAU_RAMP, TAU_HOLD = 9.4, 126.1
SAVE = [4.7, 9.4, 14.4, 19.4, 29.4, 39.4]
T_STOP = SAVE[-1]
CHECK = [float(t) for t in np.arange(2.0, T_STOP, 2.0)]
RUNGS = [6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24, 26]
OUT = os.path.join(C.RESULTS, "ramp80")
TOL_RET = 1e-4


def gate_file(N):
    if N == 22:
        return os.path.join(C.G80_RESULTS, f"mf80_leg_{LEG}_n22.npz")
    return os.path.join(C.RESULTS, f"ladderp3_{LEG}_n{N}.npz")


def ramp_path(N):
    return C.assert_writable(os.path.join(OUT, f"ramp_{LEG}_n{N}.npz"))


def cmd_prop(a):
    os.makedirs(OUT, exist_ok=True)
    OP = None
    if a.overlap == "gather":
        assert a.norb == [26], "--overlap gather is for norb 26 (as its gate)"
        from . import overlap_patch as OP
        OP.install()
    L.device_banner()
    for N in a.norb:
        assert N in RUNGS, N
        out = ramp_path(N)
        ref = np.load(gate_file(N))
        lay = mf80.layout(N)
        integ = L.Integ()
        assert (float(ref["tau_ramp"]), float(ref["tau_hold"])) == (TAU_RAMP, TAU_HOLD)
        assert str(ref["kind"]) == "full" and str(ref["status"]) == "ok"
        assert int(ref["num_dvr"]) == 80
        assert json.loads(str(ref["layout"])) == dataclasses.asdict(lay), N
        assert json.loads(str(ref["integ"])) == integ.as_dict(), N
        env, _T = L.make_env(TAU_RAMP, TAU_HOLD, "full")
        ml = L.make_ml(LEG, L.h_list_for(LEG, env), lay)
        y0 = jnp.asarray(ref["y0"])
        print(f"\n=== ramp {LEG} norb {lay.norbs(LEG)} n_s {lay.nspfs(LEG)}: "
              f"0 -> {T_STOP} (gate file {os.path.basename(gate_file(N))}) ===",
              flush=True)
        t0 = time.time()
        P = L.Propagator(ml, integ)
        y, status, sn, sps, tset = P.run(y0, 0.0, T_STOP,
                                         snaps=sorted(set(SAVE) | set(CHECK)),
                                         probe=L.norm_probe(ml))
        print(f"  status {status}  {sps * 1e3:.2f} ms/step", flush=True)
        # save first, numpy only (no new kernels before the state is on
        # disk: at norb 26 a later kernel load can run out of memory); then unpack and resave
        meta = dict(times=np.array(SAVE),
                    y_t=np.stack([np.asarray(sn[t]) for t in SAVE]),
                    y0=np.asarray(y0), leg=LEG, norb=N, status=status,
                    sec_per_step=sps, tau_ramp=TAU_RAMP, tau_hold=TAU_HOLD,
                    layout=json.dumps(dataclasses.asdict(lay)),
                    integ=json.dumps(integ.as_dict()), num_dvr=fw.NUM_DVR,
                    gate_file=gate_file(N), wall=time.time() - t0)
        L.save_npz(out, **meta)
        bs = {t: ml._y_to_ABb(jnp.asarray(sn[t]))[2] for t in SAVE}
        L.save_npz(out, **meta, **{
            f"b{w}_t": np.stack([np.asarray(bs[t][w]) for t in SAVE])
            for w in range(4)})
        print(f"  saved -> {out}  ({time.time() - t0:.0f} s)", flush=True)
        # the check against the gate's own retention record
        rts = list(np.round(np.asarray(ref["ret_ts"]), 6))
        worst = 0.0
        for t in CHECK:
            r_new = np.asarray(L.gate_matrix(ml, y0, jnp.asarray(sn[t]))[3])
            r_ref = np.asarray(ref["ret_t"])[rts.index(round(t, 6))]
            worst = max(worst, float(np.max(np.abs(r_new - r_ref))))
        r94 = np.asarray(L.gate_matrix(ml, y0, jnp.asarray(sn[9.4]))[3])
        ok = status == "ok" and worst <= TOL_RET
        C.save_json(C.assert_writable(os.path.join(OUT, f"check_n{N}.json")),
                    dict(norb=N, status=status, retention_vs_gate_worst=worst,
                         tol=TOL_RET, n_checked=len(CHECK),
                         retention_at_9p4=r94.tolist(),
                         passed=ok))
        print(f"  [check] retention at t = 2 ... {CHECK[-1]:g} vs the gate "
              f"file: worst {worst:.1e} (tol {TOL_RET:.0e}) -> "
              f"{'PASS' if ok else 'FAIL'}; retention at 9.4 "
              + " ".join(f"{v:.5f}" for v in r94), flush=True)
        if not ok:
            raise SystemExit(f"ramp check FAILED at norb {N} (stop and report)")


def cmd_readout(a):
    from scipy.sparse.linalg import eigsh
    from gate_timing import level_check as LCK
    from gate_timing.timing import FCIWell, h_well, hartree
    from . import levels_hartree80 as LH
    from . import levels_own80 as LO
    t0 = time.time()
    Delta = C.delta80(LEG)
    fci = FCIWell()
    assert fci.n == 80, fci.n
    x = L.x_vectors(LEG)[1] / fw.TYPICAL_FREQ_GHZ
    nspec = {}
    for s in LH.SPEC:
        _H, _E, V, _r = LCK.fci_levels(fci, h_well(s, x[s], x[4 + s]), k=2)
        nspec[s] = LH.dens_of(fci, V[:, :1])[0]

    def H_of(w, partner=None):
        nn = dict(nspec)
        if partner is not None:
            nn[W2 if w == W1 else W1] = partner
        return fci.hamiltonian(h_well(w, x[w], x[4 + w], hartree(w, nn)))

    def exact(H):
        E, V = eigsh(H, k=LH.NLEV, which="SA", tol=1e-12, ncv=48,
                     v0=np.ones(fci.D) / np.sqrt(fci.D))
        o = np.argsort(E)
        r = float(np.max(np.linalg.norm(H @ V[:, o] - V[:, o] * E[o], axis=0)))
        assert r < 1e-8, r
        return E[o], V[:, o]

    Hb = {w: fci.hamiltonian(h_well(w, x[w], x[4 + w])) for w in (W1, W2)}
    Hd = {w: H_of(w) for w in (W1, W2)}
    ex = {"bare": {}, "dress": {}, "frozen": {}}
    for w in (W1, W2):
        ex["bare"][w] = exact(Hb[w])
        ex["dress"][w] = exact(Hd[w])
    nd0 = {w: LH.dens_of(fci, ex["dress"][w][1][:, :1])[0] for w in (W1, W2)}
    Hf_ex = {W1: H_of(W1, nd0[W2]), W2: H_of(W2, nd0[W1])}
    for w in (W1, W2):
        ex["frozen"][w] = exact(Hf_ex[w])
    out_ex = {k: LH.over(LH.pairs_plain(ex[k][W1][0], ex[k][W2][0]), Delta)
              for k in ("bare", "frozen")}
    print("  exact bare   " + " ".join(f"|{p}> {out_ex['bare'][p]:+.4f}"
                                       for p in LH.SHOW), flush=True)
    print("  exact frozen " + " ".join(f"|{p}> {out_ex['frozen'][p]:+.4f}"
                                       for p in LH.SHOW), flush=True)

    for N in a.norb:
        d = np.load(os.path.join(OUT, f"ramp_{LEG}_n{N}.npz"))
        assert int(d["num_dvr"]) == 80
        times = [float(t) for t in d["times"]]
        b_gp = {w: np.asarray(np.load(LO.gp_file(N))[f"b{w}"]) for w in (W1, W2)}
        b_id = {w: np.asarray(np.load(C.mf_state_path(LEG, N))[f"b{w}"])
                for w in (W1, W2)}
        rec = {"norb": N, "Delta_80": Delta, "exact": out_ex, "times": {}}
        for it, t in enumerate(times):
            U = {w: np.asarray(d[f"b{w}_t"][it]) for w in (W1, W2)}
            Ts, lev, X = {}, {"bare": {}, "frozen": {}}, {}
            ang = {}
            for w in (W1, W2):
                assert U[w].shape == (80, N), U[w].shape
                Ts[w], serr = LH.embed(fci, U[w])
                e, _X, lab = LH.ci(Ts[w], Hb[w], ex["bare"][w][1])
                lev["bare"][w] = dict(E=e.tolist(), S_err=serr, **lab)
                _e, X[w], _l = LH.ci(Ts[w], Hd[w], ex["dress"][w][1])
                for nm, bref in (("gp", b_gp[w]), ("idle", b_id[w])):
                    sv = np.linalg.svd(bref.conj().T @ U[w], compute_uv=False)
                    ang[f"w{w}_{nm}_max_sin"] = float(
                        np.sqrt(max(0.0, 1 - sv.min() ** 2)))
                    ang[f"w{w}_{nm}_mean_sin2"] = float(np.mean(1 - sv ** 2))
            n0 = {w: LH.dens_of(fci, X[w][:, :1])[0] for w in (W1, W2)}
            for w in (W1, W2):
                part = n0[W2] if w == W1 else n0[W1]
                e, _X, lab = LH.ci(Ts[w], H_of(w, part), ex["frozen"][w][1])
                lev["frozen"][w] = dict(E=e.tolist(), **lab)
            del Ts
            r = {"angles": ang}
            for k in ("bare", "frozen"):
                r[k] = dict(levels={str(w): lev[k][w] for w in (W1, W2)},
                            pair_over_Delta=LH.over(LH.pairs_plain(
                                np.asarray(lev[k][W1]["E"]),
                                np.asarray(lev[k][W2]["E"])), Delta))
            rec["times"][f"{t:g}"] = r
            print(f"  n{N:<3d} t {t:5.1f}  bare " + "".join(
                f"{r['bare']['pair_over_Delta'][p]:>+9.3f}" for p in LH.SHOW)
                + "  frozen " + "".join(
                f"{r['frozen']['pair_over_Delta'][p]:>+9.3f}" for p in LH.SHOW)
                + f"  | sin vs gp {ang[f'w{W1}_gp_mean_sin2']:.1e} "
                f"[{time.time() - t0:.0f} s]", flush=True)
            C.save_json(C.assert_writable(
                os.path.join(OUT, f"levels_n{N}.json")), rec)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prop", "readout"])
    p.add_argument("--norb", type=int, nargs="+", required=True)
    p.add_argument("--overlap", choices=["library", "gather"], default="library")
    a = p.parse_args(argv)
    C.require_grid(80)
    t0 = time.time()
    {"prop": cmd_prop, "readout": cmd_readout}[a.cmd](a)
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
