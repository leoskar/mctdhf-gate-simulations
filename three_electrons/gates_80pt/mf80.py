"""The mean-field (Hartree–Fock spectator) track on the 80-point grid.

    FOURWELL_NUM_DVR=80 python -m gates_80pt.mf80 relax --leg w1w2 --norb 14 18 20 22
    FOURWELL_NUM_DVR=80 python -m gates_80pt.mf80 idle  --leg w1w2 --norb 14
    FOURWELL_NUM_DVR=80 python -m gates_80pt.mf80 gate  --leg w1w2 --norb 22 \
        --tau-ramp 28.3 --tau-hold 105.4 --out mf80_ladder_w1w2_n22
    FOURWELL_NUM_DVR=80 python -m gates_80pt.mf80 gate  --leg w1w2 --norb 22 \
        --tau-ramp 9.4 --hold-after 10 --kind up --out g4_w1w2_n22_s16o4
    FOURWELL_NUM_DVR=80 python -m gates_80pt.mf80 gate  --leg w1w2 --norb 22 \
        --tau-ramp 9.4 --hold-after 10 --kind up --dt 5e-3 --nsub-spf 8 \
        --nsub-orb 2 --out g4_w1w2_n22_dt5e-3_s8o2
    FOURWELL_NUM_DVR=80 python -m gates_80pt.mf80 gate  --leg w1w2 --norb 22 \
        --tau-ramp 9.4 --hold-after 10 --kind up --renorm none \
        --out g4_w1w2_n22_s16o4_rnnone
    python -m gates_80pt.mf80 compare --leg w1w2 --norb 22 --ref g4_..._s32o8 \
        --arm g4_..._s16o4

**relax**. norb 14: ``mf_lib.projected_states`` of the 80-point idle
register (``mf_lib.IDLE_PATH`` redirected by ``gates_80pt.common``) onto
Hartree–Fock spectators, then ``mf_lib.relax_mf_states`` in the MF layout at
idle, x64d, inner tol 1e-4 (exact_check's ladder value). norb > 14:
the norb-14 states with the active orbitals embedded exactly
(``exact_check.embed_orbitals``) and relaxed in x64d (``exact_check.relax``),
as exact_check's ``ladder`` mode. Saved as results/mfstates80_<leg>_n<N>.npz.

**idle** (check): the four states propagated at the
constant idle Hamiltonian for T t.u. with ``Integ()`` defaults; 1 - F̄(identity),
retention, max off-diagonal |M|.

**gate**: one MF gate from the saved states, ``Integ()`` defaults
unless --dt/--nsub-spf/--nsub-orb/--renorm are given (omitted: dt 1e-2,
renorm step); ``kind full`` = the genuine erf ramp-hold-ramp, ``kind up`` =
ramp-up + plateau to tau_ramp + hold_after.
Column retention every 2 t.u.; metrics as ``exact_check.run_gate``
(``mf_lib.gate_metrics``) with Delta_80 from results/oppoints80.json for the
enhancement. Saved as results/<out>.npz.

**compare** (the integration checks of Sec. S6 B, files ``g4_*``): the
reference gate infidelity 1 - F̄(C, 1),
C_ij = <psi_ref,i|psi_j> normalised (gate_timing integ_study.state_diff), the
conditional-phase error, norm drift, ms/step.
"""
import argparse
import dataclasses
import json
import os
import time

from . import common as T  # noqa: I100  (x64 + redirections first)

import jax.numpy as jnp
import numpy as np

L, EC, fw = T.L, T.EC, T.fw
W = L.WELLS


def layout(norb):
    return dataclasses.replace(L.Layout(), norb_active=norb)


def h_idle(leg):
    return fw.h_static(L.x_vectors(leg)[0] / fw.TYPICAL_FREQ_GHZ)


def save_state(path, ml, y, **extra):
    A, Bs, bs = ml._y_to_ABb(y)
    E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    L.save_npz(path, A=np.asarray(A), **{f"B{s}": np.asarray(Bs[s]) for s in W},
               **{f"b{s}": np.asarray(bs[s]) for s in W}, E=E,
               num_dvr=fw.NUM_DVR, **extra)


def load_state(leg, norb, ml):
    path = T.mf_state_path(leg, norb)
    d = np.load(path)
    assert int(d["num_dvr"]) == fw.NUM_DVR, (path, int(d["num_dvr"]))
    y = ml._ABb_to_y(jnp.asarray(d["A"]), [jnp.asarray(d[f"B{s}"]) for s in W],
                     [jnp.asarray(d[f"b{s}"]) for s in W])
    M0 = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y))
                             - np.eye(4))))
    print(f"  [state] {os.path.basename(path)}: M(0) - 1 = {M0:.2e}",
          flush=True)
    assert M0 < 1e-10
    return y, M0


def check_idle_register():
    d = np.load(T.REGISTER80)
    assert int(d["num_dvr"]) == fw.NUM_DVR
    assert (int(d["norb"]), int(d["n_spf"])) == (14, 4)


def cmd_relax(a):
    check_idle_register()
    leg = a.leg
    hi = h_idle(leg)
    for N in sorted(a.norb):
        t0 = time.time()
        lay = layout(N)
        out = T.mf_state_path(leg, N)
        if N == 14:
            y, info = L.relax_mf_states(leg, lay, dt=2e-2,
                                        inner_tol=a.inner_tol, tol=1e-10,
                                        max_steps=40000, precision="x64d",
                                        out_tag="_x64d")
            src = L.relaxed_path(leg, lay, "_x64d")
            d = dict(np.load(src))
            d["num_dvr"] = fw.NUM_DVR
            L.save_npz(out, **d)
            print(f"  copied {src} -> {out}", flush=True)
            continue
        lay14 = layout(14)
        ml14 = L.make_ml(leg, hi, lay14)
        y14, _ = load_state(leg, 14, ml14)
        y0 = EC.embed_orbitals(leg, ml14, y14, lay)
        ml_b = L.make_ml(leg, hi, lay)
        if a.chunk_rule:
            ml_r, y_r = relax_chunk(leg, lay, y0, ml_b, a.inner_tol)
        else:
            ml_r, y_r = EC.relax(leg, lay, y0, ml_b, a.inner_tol,
                                 precision="x64d", prefix="g80_")
        pops = ml_r.spf_populations(y_r)
        for s in L.LEGS[leg]:
            print(f"  [relax] w{s} SPF populations "
                  + " ".join(f"{float(v):.2e}" for v in np.asarray(pops[s])),
                  flush=True)
        save_state(out, ml_r, y_r, source="embed n14 + x64d relax",
                   inner_tol=a.inner_tol)
        print(f"  saved -> {out}  ({time.time() - t0:.0f} s)", flush=True)


def relax_chunk(leg, lay, y_start, ml_start, inner_tol, max_steps=40000):
    """``exact_check.relax`` with the CHUNK-END stopping rule: the summed
    energy changes by < 1e-10 between consecutive 50-iteration chunk ends
    (routed.relax_register's rule, adopted for the
    period-2 cycle that w0w1 settles into: consecutive iterations alternate
    by ~1e-7 while chunk ends agree to 1e-13, so the per-iteration rule never
    fires). Everything else as exact_check.relax, x64d."""
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt
    from mctdhf.MLMCTDHF_n_species import make_regauge_ml
    t0 = time.time()
    hi = h_idle(leg)
    cfgs = L.pair_configs(leg)
    ite = MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(), num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_mctdhf_orbitals=lay.norbs(leg),
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_spfs=lay.nspfs(leg), relax_configs=cfgs,
        relax_weights=list(L.RELAX_WEIGHTS), all_configs=cfgs, dt=2e-2,
        precision="x64d", inner_tol=inner_tol, chunk=50, eps=1e-6)
    ml = L.make_ml(leg, hi, lay)
    E0 = np.real(np.asarray(ml_start.calculate_energy(0.0, y_start)))
    A, Bs, bs = ml._y_to_ABb(y_start)
    state, steps, e_prev, converged = (A, list(Bs), list(bs)), 0, None, False
    while steps < max_steps:
        state, e, info = ite.integrate(ite.chunk, hi, state)
        steps += len(e)
        if not np.all(np.isfinite(e)):
            break
        dch = np.nan if e_prev is None else e[-1] - e_prev
        if e_prev is not None and abs(dch) < 1e-10:
            converged = True
            break
        e_prev = e[-1]
    print(f"  [relax-chunk] {steps} it, converged {converged} (last chunk-end "
          f"dE {dch:+.2e}, last-iteration dE {e[-1] - e[-2]:+.2e}), "
          f"{time.time() - t0:.0f} s", flush=True)
    h = ite.as_h(hi)
    A, Bs, bs = state
    Bs_rg, evals, evecs, ovl = ite._front_match(A, Bs, bs, h)
    cols, pop = ite._assign(ovl)
    A = evecs[:, jnp.asarray(cols)].T.reshape(ml.a_shape)
    y = make_regauge_ml(ml)(ml._ABb_to_y(A, list(Bs_rg), list(bs)))
    E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    err = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y))
                              - np.eye(4))))
    print("  [relax-chunk] E start   " + " ".join(f"{v:.8f}" for v in E0))
    print("  [relax-chunk] E relaxed " + " ".join(f"{v:.8f}" for v in E)
          + f"   M(0)-1 {err:.1e}  match pop "
          + " ".join(f"{v:.6f}" for v in np.asarray(pop)), flush=True)
    assert converged and np.all(np.isfinite(E)), "relaxation failed"
    assert np.min(np.asarray(pop)) > 0.99, f"root matching failed: {pop}"
    assert err < 1e-10
    return ml, y


def delta80(leg):
    path = os.path.join(T.RESULTS, "oppoints80.json")
    if not os.path.exists(path):
        return None
    r = T.load_json(path).get(leg, {})
    return r.get("final", {}).get("Delta") if r.get("status") == "ok" else None


def integ_of(a):
    # --dt / --renorm omitted: the Integ defaults (dt 1e-2, renorm "step")
    kw = {k: v for k, v in (("dt", a.dt), ("renorm", a.renorm))
          if v is not None}
    return L.Integ(nsub_spf=a.nsub_spf, nsub_orb=a.nsub_orb, **kw)


def cmd_idle(a):
    leg, lay = a.leg, layout(a.norb[0])
    env, _ = L.make_env(1.0, 1.0, "idle")
    ml = L.make_ml(leg, L.h_list_for(leg, env), lay)
    y0, M0 = load_state(leg, a.norb[0], ml)
    integ = integ_of(a)
    L.device_banner()
    print(f"=== idle control leg {leg} norb {lay.norbs(leg)} T {a.T} ===\n"
          f"  integ {integ.as_dict()}", flush=True)
    P = L.Propagator(ml, integ)
    y, status, _, sps, tcomp = P.run(y0, 0.0, a.T, probe=L.norm_probe(ml))
    print(f"  status {status}  {sps * 1e3:.2f} ms/step (setup+first block "
          f"{tcomp:.0f} s)", flush=True)
    m = L.gate_metrics(ml, y0, y, None)
    L.print_metrics(m)
    loss_id = 1.0 - m["F_identity"]
    offd = float(np.max(np.abs(m["M"] - np.diag(np.diag(m["M"])))))
    print(f"  IDLE: 1 - Fbar(identity) = {loss_id:.3e}   max offdiag |M| "
          f"{offd:.3e}", flush=True)
    pops = m.pop("spf_pops")
    path = os.path.join(T.RESULTS, f"mf80_idle_{leg}_n{a.norb[0]}.npz")
    L.save_npz(path, **m, T=a.T, status=status, sec_per_step=sps,
               loss_identity=loss_id, max_offdiag=offd, M0_err=M0,
               spf_pops=np.concatenate([np.asarray(p) for p in pops]),
               integ=json.dumps(integ.as_dict()), y=np.asarray(y))
    print(f"  saved -> {path}", flush=True)


def cmd_gate(a):
    leg, N = a.leg, a.norb[0]
    lay = layout(N)
    if a.kind == "full":
        env, T_tot = L.make_env(a.tau_ramp, a.tau_hold, "full")
        T_eff = a.tau_ramp + a.tau_hold
    else:
        env, _ = L.make_env(a.tau_ramp, 0.0, "up")
        T_tot = round(a.tau_ramp + a.hold_after, 6)
        T_eff = None
    ml = L.make_ml(leg, L.h_list_for(leg, env), lay)
    y0, M0 = load_state(leg, N, ml)
    integ = integ_of(a)
    L.device_banner()
    print(f"\n=== gate ({a.out}) leg {leg} norb {lay.norbs(leg)} n_s "
          f"{lay.nspfs(leg)} kind {a.kind} tau_ramp {a.tau_ramp} tau_hold "
          f"{a.tau_hold} T {T_tot} ===\n  integ {integ.as_dict()}", flush=True)
    snaps = [round(t, 1) for t in np.arange(2.0, T_tot - 1e-9, 2.0)] + [T_tot]
    t0 = time.time()
    P = L.Propagator(ml, integ)
    y, status, sn, sps, tsetup = P.run(y0, 0.0, T_tot, snaps=snaps,
                                       probe=L.norm_probe(ml))
    print(f"  status {status}  {sps * 1e3:.2f} ms/step (setup + first block "
          f"{tsetup:.0f} s)", flush=True)
    ts = np.array(sorted(sn))
    ret_t = np.array([L.gate_matrix(ml, y0, jnp.asarray(sn[t]))[3] for t in ts])
    nrm_t = np.array([np.abs(np.diag(np.asarray(ml.calculate_overlap(
        jnp.asarray(sn[t]), jnp.asarray(sn[t]))))) for t in ts])
    print("  retention vs t (|00> |10> |01> |11>):", flush=True)
    for t, r in zip(ts, ret_t):
        if abs(t % 10) < 1e-9 or t == ts[-1]:
            print(f"    t {t:7.1f}  " + " ".join(f"{v:.5f}" for v in r),
                  flush=True)
    m = L.gate_metrics(ml, y0, y, T_eff, delta80(leg) if T_eff else None,
                       fid=(a.kind == "full"))
    L.print_metrics(m)
    pops = m.pop("spf_pops")
    max_norm_dev = float(np.max(np.abs(nrm_t - 1.0)))
    print(f"  max|norm - 1| over the run {max_norm_dev:.2e}; M(0) - 1 "
          f"{M0:.2e}; wall {time.time() - t0:.0f} s", flush=True)
    path = os.path.join(T.RESULTS, f"{a.out}.npz")
    L.save_npz(path, **m, spf_pops=np.concatenate([np.asarray(p) for p in pops]),
               layout=json.dumps(dataclasses.asdict(lay)),
               integ=json.dumps(integ.as_dict()), leg=leg, norb=N,
               kind=a.kind, tau_ramp=a.tau_ramp, tau_hold=a.tau_hold,
               T=T_tot, T_eff=T_eff if T_eff else np.nan, status=status,
               sec_per_step=sps, setup_s=tsetup, ret_t=ret_t, ret_ts=ts,
               norm_t=nrm_t, max_norm_dev=max_norm_dev, M0_err=M0,
               Delta80=delta80(leg) or np.nan, num_dvr=fw.NUM_DVR,
               y=np.asarray(y), y0=np.asarray(y0), wall=time.time() - t0)
    print(f"  saved -> {path}", flush=True)


def cmd_compare(a):
    leg, lay = a.leg, layout(a.norb[0])
    ml = L.make_ml(leg, h_idle(leg), lay)
    ref = np.load(os.path.join(T.RESULTS, f"{a.ref}.npz"))
    arm = np.load(os.path.join(T.RESULTS, f"{a.arm}.npz"))
    yr, ya = jnp.asarray(ref["y"]), jnp.asarray(arm["y"])
    nr = np.sqrt(np.abs(np.diag(np.asarray(ml.calculate_overlap(yr, yr)))))
    na = np.sqrt(np.abs(np.diag(np.asarray(ml.calculate_overlap(ya, ya)))))
    C = np.asarray(ml.calculate_overlap(yr, ya)) / np.outer(nr, na)
    F_ref, _ = L.fbar(L.M_NULL, C)
    ph = np.angle(np.diag(C))
    cond = float(np.angle(np.exp(1j * (ph[0] + ph[3] - ph[1] - ph[2]))))
    Sn = np.diag(C)
    out = dict(ref=a.ref, arm=a.arm, gate_infid=float(1 - F_ref), dcond=cond,
               infid_phase_blind=float(np.max(1 - np.abs(Sn))),
               norm_drift_arm=float(arm["max_norm_dev"]),
               norm_drift_ref=float(ref["max_norm_dev"]),
               ms_arm=float(arm["sec_per_step"]) * 1e3,
               ms_ref=float(ref["sec_per_step"]) * 1e3,
               status=(str(arm["status"]), str(ref["status"])))
    out["passed"] = bool(out["gate_infid"] <= 1e-4)
    for k, v in out.items():
        print(f"  {k:<20} {v}", flush=True)
    T.save_json(os.path.join(T.RESULTS, f"compare_{a.arm}_vs_{a.ref}.json"),
                out)
    if not out["passed"]:
        raise SystemExit(f"integration check FAILED: 1 - Fbar(C, 1) = {out['gate_infid']:.2e}"
                         f" > 1e-4")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["relax", "idle", "gate", "compare"])
    p.add_argument("--leg", default="w1w2", choices=list(T.LEGS3))
    p.add_argument("--norb", type=int, nargs="+", default=[14])
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--chunk-rule", action="store_true",
                   help="relax: chunk-end stopping rule (period-2 cycles)")
    p.add_argument("--T", type=float, default=162.0)
    p.add_argument("--kind", default="full", choices=["full", "up"])
    p.add_argument("--tau-ramp", type=float, default=28.3)
    p.add_argument("--tau-hold", type=float, default=105.4)
    p.add_argument("--hold-after", type=float, default=10.0)
    p.add_argument("--nsub-spf", type=int, default=16)
    p.add_argument("--nsub-orb", type=int, default=4)
    p.add_argument("--dt", type=float, default=None,
                   help="CMF step; omitted = the Integ default (1e-2)")
    p.add_argument("--renorm", default=None, choices=["step", "none"],
                   help="per-step Löwdin renormalisation; omitted = the "
                        "Integ default (step)")
    p.add_argument("--out", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--arm", default=None)
    a = p.parse_args(argv)
    T.require_grid(80)
    {"relax": cmd_relax, "idle": cmd_idle, "gate": cmd_gate,
     "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
