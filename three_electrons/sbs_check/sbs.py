"""SBS-count check of the three-electron sqrt(iSWAP) gates at the paper's
settings: 80-point grid, actives norb 22, HF spectators (norb 3, n_s 1), the
theta = pi/2 schedules (gates_80pt/results/pulses80.json), ``Integ()`` defaults. Only the
number of SBS (= SPFs, ``nspf_active``) on the two active wells changes.

    FOURWELL_NUM_DVR=80 python -m sbs_check.sbs relax --leg w1w2 --nspf 6
    FOURWELL_NUM_DVR=80 python -m sbs_check.sbs pilot --leg w1w2 --nspf 6 --T 6
    FOURWELL_NUM_DVR=80 python -m sbs_check.sbs gate  --leg w1w2 --nspf 6

**relax**: the gates_80pt four-SBS state (results/mfstates80_<leg>_n22.npz, read
only) embedded exactly into n_s SBS (``exact_check.embed_full``) and relaxed
at idle in the MF layout, x64d, inner tol 1e-4, RELAX_WEIGHTS, with the
chunk-end stopping rule (``gates_80pt.mf80.relax_chunk``). The state is saved
BEFORE any diagnostics (norb-26 kernel-load OOM precedent); the checks then
go to results/relax_<leg>_s<n>.json: |<embedded_i|relaxed_j>|, energies
before/after, the weighted energy, M(0) - 1, SBS populations.

**gate**: ``gates_80pt.mf80.cmd_gate`` itself, called through ``mf80.main`` with
four module-level patches and nothing else changed:
``mf80.layout`` (nspf_active = n_s), ``common.mf_state_path`` (this
directory's state files), ``common.RESULTS`` (output here) and
``mf80.delta80`` (still reads gates_80pt/results/oppoints80.json, which
``common.RESULTS`` would otherwise redirect away from).

**pilot**: the first T t.u. of the same pulse with the same solver,
integrator and Propagator as ``cmd_gate`` (which can only run a full
pulse): ms/step, peak GPU memory, norms. At n_s 4 it starts from the gates_80pt
state and its column retention at t = 2, 4, ... is compared with the
baseline gate's ``ret_t``.
"""
import argparse
import dataclasses
import json
import os
import time

from gates_80pt import common as T  # noqa: I100  (x64 + gates_80pt redirections first)
from gates_80pt import mf80

import jax
import jax.numpy as jnp
import numpy as np

L, EC, fw = T.L, T.EC, T.fw
W = L.WELLS
assert fw.NUM_DVR == 80, f"fourwell.NUM_DVR = {fw.NUM_DVR}, need 80"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
G80_RESULTS = T.RESULTS          # read only
NORB = 22

# Redirect every module-level output path of the imported packages into
# this directory (gates_80pt.common pointed them into gates_80pt/results).
L.RESULTS_DIR = os.path.join(RESULTS, "_mf_lib")
L.MF_STATE_DIR = os.path.join(RESULTS, "_mf_states")


def _ec_out_dir(leg):
    d = os.path.join(RESULTS, "_exact_check", leg)
    os.makedirs(d, exist_ok=True)
    return d


EC.out_dir = _ec_out_dir

# The pulses the paper reports (task statement), checked against the file.
PULSES = {"w0w1": (12.6, 98.3), "w1w2": (9.4, 126.1), "w2w3": (8.0, 131.6)}


def pulse(leg):
    p = T.load_json(os.path.join(G80_RESULTS, "pulses80.json"))[leg]
    assert (p["tau_ramp"], p["tau_hold"]) == PULSES[leg], (leg, p)
    return p["tau_ramp"], p["tau_hold"]


def layout(norb, nspf):
    return dataclasses.replace(L.Layout(), norb_active=norb, nspf_active=nspf)


SUFFIX = ""        # --out-suffix: a diagnostic run's own state file


def state_path(leg, norb, nspf):
    if nspf == 4:   # the baseline, read only. Not via common.mf_state_path:
        # that reads common.RESULTS at call time, which patch() redirects.
        return os.path.join(G80_RESULTS, f"mfstates80_{leg}_n{norb}.npz")
    return os.path.join(RESULTS, f"mfstates80_{leg}_n{norb}_s{nspf}{SUFFIX}.npz")


def log_chunks():
    """Print the summed energy at every ITE chunk end (observation only:
    wraps ``MLMCTDHF_n_species_ITE_opt.integrate``, arithmetic unchanged)."""
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt as C
    orig, st = C.integrate, {"n": 0, "prev": None, "t": time.time()}

    def integrate(self, *args, **kw):
        out = orig(self, *args, **kw)
        e = np.asarray(out[1])
        st["n"] += len(e)
        d = np.nan if st["prev"] is None else e[-1] - st["prev"]
        st["prev"] = e[-1]
        print(f"    [chunk] it {st['n']:6d}  E {e[-1]:.12f}  chunk-end dE "
              f"{d:+.2e}  last-it dE {e[-1] - e[-2]:+.2e}  "
              f"{time.time() - st['t']:.0f} s", flush=True)
        return out
    C.integrate = integrate


def delta80(leg):
    r = T.load_json(os.path.join(G80_RESULTS, "oppoints80.json")).get(leg, {})
    return r.get("final", {}).get("Delta") if r.get("status") == "ok" else None


def patch(nspf):
    """Point cmd_gate at n_s = nspf and at this directory; nothing else."""
    mf80.layout = lambda norb: layout(norb, nspf)
    T.mf_state_path = lambda leg, norb: state_path(leg, norb, nspf)
    T.RESULTS = RESULTS
    mf80.delta80 = delta80


def _peak_gib():
    s = jax.devices()[0].memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 2 ** 30


def _wavg(E):
    w = np.asarray(L.RELAX_WEIGHTS) / np.sum(L.RELAX_WEIGHTS)
    return float(np.dot(w, E))


def cmd_relax(a):
    leg, n = a.leg, a.nspf
    assert n > 4
    t0 = time.time()
    hi = mf80.h_idle(leg)
    lay4, lay_n = layout(NORB, 4), layout(NORB, n)
    ml4 = L.make_ml(leg, hi, lay4)
    patch(4)
    y4, M0_4 = mf80.load_state(leg, NORB, ml4)
    E_parent_file = np.asarray(np.load(state_path(leg, NORB, 4))["E"])
    y_emb = EC.embed_full(leg, ml4, y4, lay_n)
    ml_n = L.make_ml(leg, hi, lay_n)
    print(f"  [relax] {leg} n_s {lay_n.nspfs(leg)}, chunk-end rule, x64d, "
          f"inner tol {a.inner_tol}", flush=True)
    ml_r, y_r = mf80.relax_chunk(leg, lay_n, y_emb, ml_n, a.inner_tol)
    out = state_path(leg, NORB, n)
    # save first: diagnostics after the save (kernel-load OOM precedent)
    mf80.save_state(out, ml_r, y_r,
                    source=f"embed_full s4 -> s{n} + x64d relax_chunk",
                    parent=os.path.basename(state_path(leg, NORB, 4)),
                    inner_tol=a.inner_tol, stop_rule="chunk-end")
    print(f"  saved -> {out}  ({time.time() - t0:.0f} s)", flush=True)

    E_emb = np.real(np.asarray(ml_n.calculate_energy(0.0, y_emb)))
    E_rel = np.real(np.asarray(ml_r.calculate_energy(0.0, y_r)))
    S = np.asarray(ml_r.calculate_overlap(y_emb, y_r))
    absS = np.abs(S)
    M0 = float(np.max(np.abs(np.asarray(ml_r.calculate_overlap(y_r, y_r))
                             - np.eye(4))))
    pops = [np.asarray(p) for p in ml_r.spf_populations(y_r)]
    pops_emb = [np.asarray(p) for p in ml_n.spf_populations(y_emb)]
    order_ok = bool(np.all(np.argmax(absS, axis=1) == np.arange(4)))
    chk = dict(
        leg=leg, nspf=n, parent=os.path.basename(state_path(leg, NORB, 4)),
        E_parent_file=E_parent_file, E_embedded=E_emb, E_relaxed=E_rel,
        dE=E_rel - E_emb, Ebar_embedded=_wavg(E_emb), Ebar_relaxed=_wavg(E_rel),
        dEbar=_wavg(E_rel) - _wavg(E_emb),
        abs_overlap=absS, diag_overlap=np.diag(absS),
        max_offdiag_overlap=float(np.max(absS - np.diag(np.diag(absS)))),
        order_ok=order_ok, M0_err=M0, M0_err_parent=M0_4,
        spf_pops={f"w{s}": pops[s] for s in L.LEGS[leg]},
        spf_pops_embedded={f"w{s}": pops_emb[s] for s in L.LEGS[leg]},
        wall=time.time() - t0)
    chk["passed"] = dict(
        overlap=bool(np.min(np.diag(absS)) > 0.99 and order_ok),
        Ebar_not_up=bool(chk["dEbar"] <= 1e-10),
        E_small=bool(np.max(np.abs(chk["dE"])) < 1e-3),
        M0=bool(M0 < 1e-10))
    print("  [check] E embedded " + " ".join(f"{v:.8f}" for v in E_emb))
    print("  [check] E relaxed  " + " ".join(f"{v:.8f}" for v in E_rel))
    print("  [check] dE         " + " ".join(f"{v:+.3e}" for v in chk["dE"])
          + f"   dEbar {chk['dEbar']:+.3e}")
    print("  [check] |<emb_i|rel_j>|:")
    for r in absS:
        print("           " + " ".join(f"{v:.8f}" for v in r))
    print(f"  [check] order ok {order_ok}; M(0)-1 {M0:.2e}")
    for s in L.LEGS[leg]:
        print(f"  [check] w{s} SBS populations relaxed  "
              + " ".join(f"{float(v):.3e}" for v in pops[s]))
        print(f"  [check] w{s} SBS populations embedded "
              + " ".join(f"{float(v):.3e}" for v in pops_emb[s]))
    print(f"  [check] passed {chk['passed']}", flush=True)
    T.save_json(os.path.join(RESULTS, f"relax_{leg}_s{n}{SUFFIX}.json"), chk)
    assert all(chk["passed"].values()), f"relaxed-state check FAILED: {chk['passed']}"


class _Stop(Exception):
    pass


def plateau_path(leg, n, it):
    return os.path.join(RESULTS, f"mfstates80_{leg}_n{NORB}_s{n}_it{it}.npz")


def plateau_log_path(leg, n):
    return os.path.join(RESULTS, f"plateau_{leg}_s{n}_log.json")


def _embedded(leg, n):
    """The gates_80pt four-SBS state embedded exactly into n SBS (as cmd_relax)."""
    hi = mf80.h_idle(leg)
    ml4 = L.make_ml(leg, hi, layout(NORB, 4))
    patch(4)
    y4, M0_4 = mf80.load_state(leg, NORB, ml4)
    y_emb = EC.embed_full(leg, ml4, y4, layout(NORB, n))
    return hi, y_emb, M0_4


def cmd_plateau(a):
    """Fixed-length relaxation for a leg whose six-SBS run never meets the
    1e-10 chunk-end rule (w0w1). ``mf80.relax_chunk`` runs unchanged, with all
    its ITE settings (x64d, dt 2e-2, chunk 50, RELAX_WEIGHTS, eps 1e-6, inner
    tol), from the same exact embedding as cmd_relax. A class-level wrapper on
    ``integrate`` observes every chunk end and leaves the arithmetic alone.
    At each chunk end in [--save-from, --max-it] it saves the state after
    exactly the post-processing relax_chunk applies to its return value:
    the top block is re-diagonalised, the roots are matched by identity and
    the state is regauged. The saved copy is
    results/mfstates80_<leg>_n22_s<n>_it<k>.npz; the raw ITE state continues
    untouched. At --max-it the wrapper raises, which stops relax_chunk. The
    chunk log (with the per-iteration energies) is rewritten to
    results/plateau_<leg>_s<n>_log.json after every chunk."""
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt as C
    from mctdhf.MLMCTDHF_n_species import make_regauge_ml
    leg, n = a.leg, a.nspf
    assert n > 4
    t0 = time.time()
    hi, y_emb, _ = _embedded(leg, n)
    lay_n = layout(NORB, n)
    ml_n = L.make_ml(leg, hi, lay_n)
    regauge = make_regauge_ml(ml_n)
    parent = os.path.basename(state_path(leg, NORB, 4))
    print(f"  [plateau] {leg} n_s {lay_n.nspfs(leg)}: mf80.relax_chunk, x64d, "
          f"inner tol {a.inner_tol}, stopped at {a.max_it} it; states saved at "
          f"chunk ends {a.save_from}..{a.max_it}", flush=True)
    orig, st = C.integrate, {"n": 0, "prev": None, "log": []}

    def integrate(self, *args, **kw):
        out = orig(self, *args, **kw)
        e = np.asarray(out[1])
        st["n"] += len(e)
        k = st["n"]
        d = np.nan if st["prev"] is None else e[-1] - st["prev"]
        st["prev"] = e[-1]
        rec = dict(it=k, E=float(e[-1]), dE_chunk=float(d),
                   dE_last=float(e[-1] - e[-2]), t=time.time() - t0,
                   e_iter=e)
        print(f"    [chunk] it {k:6d}  E {e[-1]:.12f}  chunk-end dE {d:+.2e}  "
              f"last-it dE {e[-1] - e[-2]:+.2e}  {rec['t']:.0f} s", flush=True)
        if a.save_from <= k <= a.max_it:
            A, Bs, bs = out[0]
            Bs_rg, evals, evecs, ovl = self._front_match(A, Bs, bs,
                                                         self.as_h(hi))
            cols, pop = self._assign(ovl)
            A = evecs[:, jnp.asarray(cols)].T.reshape(ml_n.a_shape)
            y = regauge(ml_n._ABb_to_y(A, list(Bs_rg), list(bs)))
            path = plateau_path(leg, n, k)
            mf80.save_state(path, ml_n, y,
                            source=f"embed_full s4 -> s{n} + x64d relax_chunk, "
                                   f"stopped at chunk end {k}",
                            parent=parent, inner_tol=a.inner_tol,
                            stop_rule=f"fixed length ({k} it)", it=k,
                            match_pop=np.asarray(pop))
            rec.update(saved=os.path.basename(path),
                       match_pop=np.asarray(pop))
            print(f"      saved -> {os.path.basename(path)}  match pop "
                  + " ".join(f"{float(v):.6f}" for v in np.asarray(pop)),
                  flush=True)
        st["log"].append(rec)
        T.save_json(plateau_log_path(leg, n), st["log"])
        if k >= a.max_it:
            raise _Stop
        return out

    C.integrate = integrate
    try:
        mf80.relax_chunk(leg, lay_n, y_emb, ml_n, a.inner_tol)
        print("  [plateau] relax_chunk returned before --max-it (1e-10 rule "
              "met or non-finite energy)", flush=True)
    except _Stop:
        print(f"  [plateau] stopped at {st['n']} it ({time.time() - t0:.0f} s)",
              flush=True)
    finally:
        C.integrate = orig


def cmd_plateau_check(a):
    """The cmd_relax checks for the plateau states at --its A B: overlap with
    the embedded parent, the energy change of each state, M(0) - 1, the SBS
    populations, plus |<A_i|B_j>|. No energy criterion: |11> is selected by
    matching and is not among the lowest roots, so no variational bound
    holds for it or for the weighted sum."""
    leg, n = a.leg, a.nspf
    hi, y_emb, M0_4 = _embedded(leg, n)
    ml = L.make_ml(leg, hi, layout(NORB, n))
    E_emb = np.real(np.asarray(ml.calculate_energy(0.0, y_emb)))
    pops_emb = [np.asarray(p) for p in ml.spf_populations(y_emb)]
    ys, out = {}, dict(leg=leg, nspf=n, its=a.its, E_embedded=E_emb,
                       Ebar_embedded=_wavg(E_emb), M0_err_parent=M0_4,
                       spf_pops_embedded={f"w{s}": pops_emb[s]
                                          for s in L.LEGS[leg]})
    for it in a.its:
        d = np.load(plateau_path(leg, n, it))
        assert int(d["num_dvr"]) == fw.NUM_DVR
        y = ml._ABb_to_y(jnp.asarray(d["A"]),
                         [jnp.asarray(d[f"B{s}"]) for s in W],
                         [jnp.asarray(d[f"b{s}"]) for s in W])
        ys[it] = y
        E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
        absS = np.abs(np.asarray(ml.calculate_overlap(y_emb, y)))
        M0 = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y))
                                 - np.eye(4))))
        pops = [np.asarray(p) for p in ml.spf_populations(y)]
        order_ok = bool(np.all(np.argmax(absS, axis=1) == np.arange(4)))
        c = dict(it=it, E_relaxed=E, dE=E - E_emb, Ebar_relaxed=_wavg(E),
                 dEbar=_wavg(E) - _wavg(E_emb), abs_overlap=absS,
                 diag_overlap=np.diag(absS),
                 max_offdiag_overlap=float(np.max(absS - np.diag(np.diag(absS)))),
                 order_ok=order_ok, M0_err=M0, match_pop=d["match_pop"],
                 spf_pops={f"w{s}": pops[s] for s in L.LEGS[leg]})
        c["passed"] = dict(
            overlap=bool(np.min(np.diag(absS)) > 0.99 and order_ok),
            E_small=bool(np.max(np.abs(c["dE"])) < 1e-3),
            M0=bool(M0 < 1e-10))
        out[f"it{it}"] = c
        print(f"  [check it {it}] E relaxed " + " ".join(f"{v:.8f}" for v in E))
        print(f"  [check it {it}] dE " + " ".join(f"{v:+.3e}" for v in c["dE"])
              + f"   dEbar {c['dEbar']:+.3e}")
        print(f"  [check it {it}] |<emb_i|rel_j>|:")
        for r in absS:
            print("           " + " ".join(f"{v:.8f}" for v in r))
        print(f"  [check it {it}] order ok {order_ok}; M(0)-1 {M0:.2e}")
        for s in L.LEGS[leg]:
            print(f"  [check it {it}] w{s} SBS populations "
                  + " ".join(f"{float(v):.3e}" for v in pops[s]))
        print(f"  [check it {it}] passed {c['passed']}", flush=True)
    if len(a.its) == 2:
        i, j = a.its
        S = np.abs(np.asarray(ml.calculate_overlap(ys[i], ys[j])))
        out["abs_overlap_AB"] = S
        print(f"  [check] |<it{i}_a|it{j}_b>|:")
        for r in S:
            print("           " + " ".join(f"{v:.10f}" for v in r))
    tag = "_".join(f"it{i}" for i in a.its)
    T.save_json(os.path.join(RESULTS, f"plateau_check_{leg}_s{n}_{tag}.json"),
                out)


def cmd_pilot(a):
    leg, n = a.leg, a.nspf
    patch(n)
    tr, th = pulse(leg)
    lay = mf80.layout(NORB)
    env, T_tot = L.make_env(tr, th, "full")
    ml = L.make_ml(leg, L.h_list_for(leg, env), lay)
    y0, M0 = mf80.load_state(leg, NORB, ml)
    integ = L.Integ()
    L.device_banner()
    print(f"\n=== pilot leg {leg} norb {lay.norbs(leg)} n_s {lay.nspfs(leg)} "
          f"tau_ramp {tr} tau_hold {th} (T {T_tot}), first {a.T} t.u. ===\n"
          f"  integ {integ.as_dict()}", flush=True)
    snaps = [round(t, 1) for t in np.arange(2.0, a.T + 1e-9, 2.0)]
    t0 = time.time()
    P = L.Propagator(ml, integ)
    y, status, sn, sps, tsetup = P.run(y0, 0.0, a.T, snaps=snaps,
                                       probe=L.norm_probe(ml),
                                       report_every=10)
    ts = np.array(sorted(sn))
    ret_t = np.array([L.gate_matrix(ml, y0, jnp.asarray(sn[t]))[3] for t in ts])
    nrm_t = np.array([np.abs(np.diag(np.asarray(ml.calculate_overlap(
        jnp.asarray(sn[t]), jnp.asarray(sn[t]))))) for t in ts])
    out = dict(leg=leg, nspf=n, T_pilot=a.T, status=status, ms_per_step=sps * 1e3,
               setup_s=tsetup, peak_GiB=_peak_gib(), M0_err=M0,
               max_norm_dev=float(np.max(np.abs(nrm_t - 1.0))), ret_ts=ts,
               ret_t=ret_t, wall=time.time() - t0)
    steps = {lg: round((sum(PULSES[lg]) + PULSES[lg][0]) / integ.dt)
             for lg in PULSES}
    out["projected_gate_h"] = {lg: k * sps / 3600 for lg, k in steps.items()}
    out["projected_total_h"] = float(sum(out["projected_gate_h"].values()))
    if n == 4:
        g = np.load(os.path.join(G80_RESULTS, f"mf80_leg_{leg}_n22.npz"))
        idx = [int(np.argmin(np.abs(g["ret_ts"] - t))) for t in ts]
        assert np.allclose(g["ret_ts"][idx], ts)
        out["baseline_ret_t"] = g["ret_t"][idx]
        out["max_dret_vs_baseline"] = float(np.max(np.abs(
            g["ret_t"][idx] - ret_t)))
    for k, v in out.items():
        print(f"  {k:<22} {v}", flush=True)
    T.save_json(os.path.join(RESULTS, f"pilot_{leg}_s{n}.json"), out)


def cmd_gate(a):
    leg, n = a.leg, a.nspf
    patch(n)
    tr, th = pulse(leg)
    print(f"  [sbs] gates_80pt.mf80.cmd_gate at nspf_active {n}; state "
          f"{state_path(leg, NORB, n)}; output dir {T.RESULTS}", flush=True)
    argv = ["gate", "--leg", leg, "--norb", str(NORB), "--kind", "full",
            "--tau-ramp", repr(tr), "--tau-hold", repr(th),
            "--out", f"mf80_leg_{leg}_n{NORB}_s{n}{SUFFIX}"]
    print(f"  [sbs] mf80.main({argv})", flush=True)
    mf80.main(argv)
    print(f"  [sbs] peak GPU memory {_peak_gib():.2f} GiB", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["relax", "pilot", "gate", "plateau",
                                   "plateau-check"])
    p.add_argument("--leg", default="w1w2", choices=list(PULSES))
    p.add_argument("--nspf", type=int, default=6)
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--T", type=float, default=6.0)
    p.add_argument("--log-chunks", action="store_true")
    p.add_argument("--out-suffix", default="")
    p.add_argument("--max-it", type=int, default=1200)
    p.add_argument("--save-from", type=int, default=700)
    p.add_argument("--its", type=int, nargs="+", default=[])
    a = p.parse_args()
    global SUFFIX
    SUFFIX = a.out_suffix
    if a.log_chunks:
        log_chunks()
    T.require_grid(80)
    {"relax": cmd_relax, "pilot": cmd_pilot, "gate": cmd_gate,
     "plateau": cmd_plateau, "plateau-check": cmd_plateau_check}[a.cmd](a)


if __name__ == "__main__":
    main()
