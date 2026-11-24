"""MF gate sweep over (tau_ramp, tau_hold) by exact checkpoint reuse.

    sweep.py relax   --leg w1w2                  # once per leg (done: results/mf_states)
    sweep.py segA    --leg w1w2 --tau-ramp 28.3 --grid
    sweep.py segB    --leg w1w2 --tau-ramp 28.3 --grid
    sweep.py point   --leg w1w2 --tau-ramp 28.3 --tau-hold 105.4
    sweep.py idle    --leg w1w2 --T 162
    sweep.py collect --leg w1w2

The integrator/precision defaults come from integ_study.py (Sec. S6 B);
``--grid`` takes the holds for ``--tau-ramp`` from grids.json (``--holds``
gives them explicitly). Submit through Slurm from the package root, e.g.

    sbatch -J mfsw-segA slurm/gpu.sbatch \
        -m gate_timing.sweep segA --leg w1w2 --tau-ramp 28.3 --grid

The initial states (= the anchors of M) are the relaxed MF pair states in
``results/mf_states`` (``Layout.init = "relaxed"``), with the real-time
state-averaging weights they were relaxed with (1, 1/2, 1/2, 1/4).

The segmented scheme of Sec. S5 D, with Delta = tau_ramp:

* **segA**, once per (leg, tau_ramp): ``env_up`` (ramp-up, then an exact
  plateau) from 0 to the largest hold, a checkpoint at every grid hold
  ``t_c = tau_hold``. ``env_up`` equals the genuine envelope to machine
  precision for t <= t_c, so the checkpoint IS the genuine leg's state.
* **segB**, once per (tau_ramp, tau_hold): restart at t_c with the genuine
  envelope and run to ``T = 2 tau_ramp + tau_hold`` — 2 tau_ramp per point,
  whatever the hold.
* **point** is the unsegmented leg, for the segmentation identity check.
* **idle** propagates the four MF states at the constant idle Hamiltonian:
  how stationary are the HF-projected states?

segB refuses to run if ANY of the discretisation (``mf_lib.Integ``), the
budget (``mf_lib.Layout``) or the control vectors differ from segA's, and
asserts the checkpoint time equals t_c to 1e-9. The step time is ``k * dt``
from the global step index, so there is no accumulated-``t`` drift to patch.
"""
import argparse
import dataclasses
import glob
import json
import os
import time

import numpy as np
import jax.numpy as jnp

from . import mf_lib as L


def _tag(v):
    return f"{v:g}".replace(".", "p")


def integ_from(a):
    return L.Integ(dt=a.dt, a_prop=a.a_prop, krylov=a.krylov,
                   precision=a.precision, cn_tol=a.cn_tol,
                   inner_tol=a.inner_tol, max_iter=a.max_iter,
                   nsub_spf=a.nsub_spf, nsub_orb=a.nsub_orb,
                   regauge_t=a.regauge_t, B_prop=a.B_prop, b_prop=a.b_prop,
              dop_sub=a.dop_sub, renorm=a.renorm)


def layout_from(a):
    return L.Layout(norb_spec=a.norb_spec, nspf_spec=a.nspf_spec,
                    init=a.init, weights=a.weights, init_tag=a.init_tag)


def run_dir(a, integ, layout):
    d = os.path.join(L.RESULTS_DIR, a.leg,
                     integ.tag() + layout.tag() + a.out_tag)
    os.makedirs(d, exist_ok=True)
    return d


def static_delta(leg):
    try:
        g = L.load_grids()
        return float(g["legs"][leg]["Delta"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def grid_holds(leg, tau_ramp):
    """The grid holds for ``tau_ramp`` from ``grids.json`` (Step 1)."""
    g = L.load_grids()["legs"][leg]
    holds = [float(p["tau_hold"]) for p in g["points"]
             if abs(float(p["tau_ramp"]) - tau_ramp) < 1e-9]
    if not holds:
        raise SystemExit(f"no grid points for {leg} at tau_ramp={tau_ramp}; "
                         f"grid ramps are "
                         f"{sorted({p['tau_ramp'] for p in g['points']})}")
    return sorted(holds)


def holds_from(a):
    if a.grid:
        return grid_holds(a.leg, a.tau_ramp)
    if not a.holds:
        raise SystemExit("give --holds or --grid")
    return sorted(float(v) for v in a.holds.split(","))


def t_check(tau_hold, tau_ramp, delta_frac, block):
    """Checkpoint time of a hold: ``t_c = tau_hold + tau_ramp - Delta`` with
    ``Delta = delta_frac * tau_ramp``, snapped DOWN to the block grid (which
    only enlarges Delta). ``delta_frac = 1`` gives ``t_c = tau_hold``.

    The segmented split is exact iff ``env_up == env`` on ``[0, t_c]``; the
    difference there is ``0.5 erfc(3 + 6 Delta / tau_ramp)``, identically
    zero in float64 from ``Delta / tau_ramp ~ 0.48`` (``seg_a`` re-checks it
    on the step grid).
    """
    if delta_frac == 1.0:
        return tau_hold
    k = int(np.floor((tau_hold + (1.0 - delta_frac) * tau_ramp) / block + 1e-9))
    return round(k * block, 10)


def check_split_exact(tau_ramp, tau_hold, t_c, dt):
    """max |env_up - env| on [0, t_c]; must be 0. The difference is
    0.5 erfc(-6 (t - t_dn) / tau_ramp), monotone in t, so its maximum is at
    t_c; a grid at dt/32 (finer than every SPF/orbital substep) is checked
    as well."""
    up, _ = L.make_env(tau_ramp, tau_hold, "up")
    full, _ = L.make_env(tau_ramp, tau_hold, "full")
    n = int(round(t_c / dt)) * 32
    t = jnp.concatenate([jnp.arange(0, n + 1) * (dt / 32), jnp.array([t_c])])
    return float(jnp.max(jnp.abs(up(t) - full(t))))


def setup(a, env):
    integ, layout = integ_from(a), layout_from(a)
    hs = L.h_list_for(a.leg, env)
    ml = L.make_ml(a.leg, hs, layout, ci_sparse=not a.dense_ci)
    y0, info = L.mf_initial_states(a.leg, ml, layout, verbose=True)
    return integ, layout, ml, y0, info


def meta(a, integ, layout):
    xi, xg = L.x_vectors(a.leg)
    return dict(leg=a.leg, integ=json.dumps(integ.as_dict(), sort_keys=True),
                layout=json.dumps(dataclasses.asdict(layout), sort_keys=True),
                x_idle=xi, x_target=xg, dense_ci=bool(a.dense_ci),
                integrator="midpoint")


def banner(a, integ, layout, what):
    L.device_banner()
    print(f"=== {what}  leg {a.leg}  norb {layout.norbs(a.leg)} "
          f"n_s {layout.nspfs(a.leg)} ===", flush=True)
    print(f"  integ {integ.as_dict()}  every {integ.every} "
          f"(block {integ.block_t:g} t.u.)", flush=True)


def save_point(path, ml, y0, y, a, integ, layout, tau_ramp, tau_hold, status,
               sps, extra):
    T_eff = tau_ramp + tau_hold
    m = L.gate_metrics(ml, y0, y, T_eff, static_delta(a.leg))
    L.print_metrics(m)
    pops = m.pop("spf_pops")
    L.save_npz(path, **m, **meta(a, integ, layout),
               spf_pops=np.concatenate([np.asarray(p) for p in pops]),
               spf_pops_split=np.cumsum([len(p) for p in pops])[:-1],
               tau_ramp=tau_ramp, tau_hold=tau_hold, T=2 * tau_ramp + tau_hold,
               T_eff=T_eff, status=status, sec_per_step=sps,
               y=np.asarray(y), **extra)
    print(f"  saved -> {path}", flush=True)


# ── segA ──────────────────────────────────────────────────────────────────

def seg_a(a):
    holds = holds_from(a)
    env, _ = L.make_env(a.tau_ramp, max(holds), "up")
    integ, layout, ml, y0, info = setup(a, env)
    banner(a, integ, layout, f"segA tau_ramp={a.tau_ramp}")
    t_cs = [t_check(th, a.tau_ramp, a.delta_frac, integ.block_t)
            for th in holds]
    for th, tc in zip(holds, t_cs):
        err = check_split_exact(a.tau_ramp, th, tc, integ.dt)
        assert err == 0.0, (f"env_up != env on [0, t_c={tc}] for tau_hold "
                            f"{th}: {err:.3e} (delta_frac {a.delta_frac})")
    T_A = max(t_cs)
    print(f"  delta_frac {a.delta_frac}: Delta/tau_ramp = "
          + " ".join(f"{(th + a.tau_ramp - tc) / a.tau_ramp:.3f}"
                     for th, tc in zip(holds, t_cs))
          + "  (env_up == env on [0, t_c] exactly, checked)", flush=True)
    print(f"  env_up 0 -> {T_A}, checkpoints at {t_cs} (holds {holds})",
          flush=True)
    P = L.Propagator(ml, integ)
    y, status, snaps, sps, tcomp = P.run(y0, 0.0, T_A, snaps=t_cs,
                                         probe=L.norm_probe(ml))
    print(f"  status {status}  {sps * 1e3:.2f} ms/step (compile {tcomp:.1f} s)",
          flush=True)
    ts = np.array(sorted(snaps))
    ys = np.array([snaps[t] for t in ts])
    path = os.path.join(run_dir(a, integ, layout),
                        f"segA_tr{_tag(a.tau_ramp)}.npz")
    L.save_npz(path, ts=ts, ys=ys, y0=np.asarray(y0), tau_ramp=a.tau_ramp,
               delta_frac=a.delta_frac, holds=np.asarray(holds),
               T_A=T_A, status=status, sec_per_step=sps,
               M0_err=info["M0_err"], **meta(a, integ, layout))
    print(f"  saved -> {path}", flush=True)


# ── segB ──────────────────────────────────────────────────────────────────

def seg_b(a):
    integ, layout = integ_from(a), layout_from(a)
    d = run_dir(a, integ, layout)
    src = a.sega or os.path.join(d, f"segA_tr{_tag(a.tau_ramp)}.npz")
    z = np.load(src)
    assert str(z["status"]) == "ok", f"segA did not finish: {z['status']}"
    mine = meta(a, integ, layout)
    mismatch = []
    for k in ("leg", "integ", "layout", "integrator"):
        if str(z[k]) != str(mine[k]):
            mismatch.append(f"{k}: segA={str(z[k])} segB={mine[k]}")
    for k in ("x_idle", "x_target"):
        if not np.array_equal(z[k], mine[k]):
            mismatch.append(f"{k} differs")
    if bool(z["dense_ci"]) != mine["dense_ci"]:
        mismatch.append("dense_ci differs")
    if abs(float(z["tau_ramp"]) - a.tau_ramp) > 0:
        mismatch.append("tau_ramp differs")
    d_A = float(z["delta_frac"]) if "delta_frac" in z.files else 1.0
    if d_A != a.delta_frac:
        mismatch.append(f"delta_frac: segA={d_A} segB={a.delta_frac}")
    assert not mismatch, ("segA and segB disagree on the discretisation, so "
                          "the checkpoint is NOT the genuine leg's state:\n  "
                          + "\n  ".join(mismatch))
    ts, ys = z["ts"], z["ys"]
    holds = holds_from(a)
    for th in holds:
        t_c = t_check(th, a.tau_ramp, a.delta_frac, integ.block_t)
        k = int(np.argmin(np.abs(ts - t_c)))
        assert abs(float(ts[k]) - t_c) < 1e-9, (
            f"no checkpoint at t_c = {t_c} (nearest {float(ts[k])})")
        env, T = L.make_env(a.tau_ramp, th, "full")
        hs = L.h_list_for(a.leg, env)
        ml = L.make_ml(a.leg, hs, layout, ci_sparse=not a.dense_ci)
        y0, _ = L.mf_initial_states(a.leg, ml, layout, verbose=False)
        assert np.array_equal(np.asarray(y0), z["y0"]), \
            "anchors differ from segA's"
        if th == holds[0]:
            banner(a, integ, layout, f"segB tau_ramp={a.tau_ramp}")
        print(f"\n  --- tau_hold {th}: t_c {t_c} -> T {T}  (T_eff "
              f"{a.tau_ramp + th}) ---", flush=True)
        P = L.Propagator(ml, integ)
        y, status, _, sps, tcomp = P.run(jnp.asarray(ys[k]), t_c, T,
                                         verbose=a.verbose)
        print(f"  status {status}  {sps * 1e3:.2f} ms/step "
              f"(compile {tcomp:.1f} s)", flush=True)
        save_point(os.path.join(d, f"pt_tr{_tag(a.tau_ramp)}_th{_tag(th)}.npz"),
                   ml, y0, y, a, integ, layout, a.tau_ramp, th, status, sps,
                   dict(kind="segB", t_c=t_c, delta_frac=a.delta_frac,
                        segA=os.path.basename(src)))


# ── point / idle ──────────────────────────────────────────────────────────

def point(a):
    env, T = L.make_env(a.tau_ramp, a.tau_hold, "full")
    integ, layout, ml, y0, info = setup(a, env)
    banner(a, integ, layout, f"point tau_ramp={a.tau_ramp} "
           f"tau_hold={a.tau_hold} T={T}")
    snaps = [float(v) for v in a.snaps.split(",")] if a.snaps else []
    P = L.Propagator(ml, integ)
    y, status, sn, sps, tcomp = P.run(y0, 0.0, T, snaps=snaps,
                                      probe=L.norm_probe(ml))
    print(f"  status {status}  {sps * 1e3:.2f} ms/step (compile {tcomp:.1f} s)",
          flush=True)
    d = run_dir(a, integ, layout)
    ts = np.array(sorted(sn))
    save_point(os.path.join(d, f"point_tr{_tag(a.tau_ramp)}"
                               f"_th{_tag(a.tau_hold)}.npz"),
               ml, y0, y, a, integ, layout, a.tau_ramp, a.tau_hold, status, sps,
               dict(kind="point", snap_ts=ts,
                    snap_ys=np.array([sn[t] for t in ts]) if len(ts) else
                    np.zeros(0)))


def idle(a):
    env, _ = L.make_env(1.0, 1.0, "idle")
    integ, layout, ml, y0, info = setup(a, env)
    banner(a, integ, layout, f"idle control T={a.T}")
    P = L.Propagator(ml, integ)
    y, status, _, sps, tcomp = P.run(y0, 0.0, a.T, probe=L.norm_probe(ml))
    print(f"  status {status}  {sps * 1e3:.2f} ms/step", flush=True)
    m = L.gate_metrics(ml, y0, y, None)
    L.print_metrics(m)
    loss_id = 1.0 - m["F_identity"]
    offd = float(np.max(np.abs(m["M"] - np.diag(np.diag(m["M"])))))
    print(f"  IDLE: 1 - Fbar(identity) = {loss_id:.3e}   max offdiag |M| "
          f"{offd:.3e}   -> {'OK' if loss_id <= 1e-4 else 'EXCEEDS 1e-4'}",
          flush=True)
    pops = m.pop("spf_pops")
    path = os.path.join(run_dir(a, integ, layout), f"idle_T{_tag(a.T)}.npz")
    L.save_npz(path, **m, **meta(a, integ, layout), T=a.T, status=status,
               sec_per_step=sps, loss_identity=loss_id, max_offdiag=offd,
               spf_pops=np.concatenate([np.asarray(p) for p in pops]),
               y=np.asarray(y), info=json.dumps(info, default=float))
    print(f"  saved -> {path}", flush=True)


# ── relax ─────────────────────────────────────────────────────────────────

def relax(a):
    L.device_banner()
    layout = layout_from(a)
    print(f"=== relax MF pair states  leg {a.leg}  norb {layout.norbs(a.leg)} "
          f"n_s {layout.nspfs(a.leg)}  weights {L.RELAX_WEIGHTS} ===",
          flush=True)
    L.relax_mf_states(a.leg, layout, dt=a.ite_dt, inner_tol=a.ite_inner_tol,
                      tol=a.ite_tol, max_steps=a.ite_max_steps,
                      precision=a.ite_precision, out_tag=a.relax_tag)


# ── collect ───────────────────────────────────────────────────────────────

def collect(a):
    integ, layout = integ_from(a), layout_from(a)
    d = run_dir(a, integ, layout)
    pts = sorted(glob.glob(os.path.join(d, "pt_tr*_th*.npz")))
    if not pts:
        print(f"  no points in {d}", flush=True)
        return
    rows = []
    for p in pts:
        z = np.load(p)
        rows.append((float(z["tau_ramp"]), float(z["tau_hold"]),
                     float(z["T_eff"]), float(z["F"]), float(z["Fxc"]),
                     float(z["F_identity"]), float(np.min(z["retention"])),
                     float(np.max(z["retention"])), float(z["theta"]),
                     float(z["Delta_eff"]), float(z["transfer"]),
                     str(z["status"])))
    rows.sort()
    print(f"=== map {a.leg} ({len(rows)} points) in {d} ===", flush=True)
    print(f"  {'tau_r':>7} {'tau_h':>8} {'T_eff':>8} {'Fbar':>10} {'1-Fbar':>9} "
          f"{'F_xc':>9} {'F_id':>9} {'min ret':>9} {'theta':>8} "
          f"{'Delta_eff':>11} status", flush=True)
    for r in rows:
        flag = "  RET>1" if r[7] > 1 + 1e-9 else ""
        print(f"  {r[0]:>7.2f} {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>10.7f} "
              f"{1 - r[3]:>9.2e} {r[4]:>9.6f} {r[5]:>9.6f} {r[6]:>9.6f} "
              f"{r[8]:>8.5f} {r[9]:>11.5e} {r[11]}{flag}", flush=True)
    ok = [r for r in rows if r[7] <= 1 + 1e-9 and r[11] == "ok"]
    best = max(ok, key=lambda r: r[3]) if ok else None
    if best:
        print(f"\n  BEST tau_ramp={best[0]} tau_hold={best[1]} "
              f"Fbar={best[3]:.7f} (loss {1 - best[3]:.3e})", flush=True)
    arr = np.array([r[:11] for r in rows])
    out = os.path.join(L.RESULTS_DIR, a.leg, f"map_{a.leg}_{os.path.basename(d)}.npz")
    L.save_npz(out, rows=arr, columns=np.array(
        ["tau_ramp", "tau_hold", "T_eff", "F", "Fxc", "F_identity", "ret_min",
         "ret_max", "theta", "Delta_eff", "transfer"]),
        status=np.array([r[11] for r in rows]),
        best=np.array(best[:11] if best else []), run_dir=d)
    print(f"  saved -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("segA", "segB", "point", "idle", "collect", "relax"):
        p = sub.add_parser(name)
        p.add_argument("--leg", required=True, choices=list(L.LEGS))
        p.add_argument("--dt", type=float, default=1e-2)
        p.add_argument("--a-prop", default="diagonalize",
                       choices=["diagonalize", "sil", "cn"])
        p.add_argument("--krylov", type=int, default=10)
        p.add_argument("--precision", default="mixed", choices=["x64", "mixed"])
        p.add_argument("--cn-tol", type=float, default=1e-10)
        p.add_argument("--inner-tol", type=float, default=1e-4)
        p.add_argument("--max-iter", type=int, default=200)
        p.add_argument("--nsub-spf", type=int, default=16)
        p.add_argument("--nsub-orb", type=int, default=4)
        p.add_argument("--regauge-t", type=float, default=0.1)
        p.add_argument("--B-prop", default="",
                       choices=["", "cn", "cn_mixed", "diagonalize", "dop"])
        p.add_argument("--b-prop", default="",
                       choices=["", "cn", "cn_mixed", "diagonalize", "dop"])
        p.add_argument("--dop-sub", type=int, default=4)
        p.add_argument("--renorm", default="step", choices=["none", "step"])
        p.add_argument("--norb-spec", type=int, default=3)
        p.add_argument("--nspf-spec", type=int, default=1)
        p.add_argument("--init", default="relaxed",
                       choices=["relaxed", "projected"])
        p.add_argument("--weights", default="relax", choices=["relax", "equal"])
        p.add_argument("--init-tag", default="")
        p.add_argument("--dense-ci", action="store_true")
        p.add_argument("--out-tag", default="")
        if name in ("segA", "segB", "point"):
            p.add_argument("--tau-ramp", type=float, required=True)
        if name in ("segA", "segB"):
            p.add_argument("--holds", default=None)
            p.add_argument("--delta-frac", type=float, default=1.0,
                           help="Delta/tau_ramp: segB restarts at t_c = "
                                "tau_hold + tau_ramp - Delta (1 = t_c = "
                                "tau_hold; >= ~0.48 is exact)")
            p.add_argument("--grid", action="store_true",
                           help="take the holds for --tau-ramp from grids.json")
        if name == "segB":
            p.add_argument("--sega", default=None)
            p.add_argument("--verbose", action="store_true")
        if name == "point":
            p.add_argument("--tau-hold", type=float, required=True)
            p.add_argument("--snaps", default=None)
        if name == "idle":
            p.add_argument("--T", type=float, required=True)
        if name == "relax":
            p.add_argument("--ite-dt", type=float, default=2e-2)
            p.add_argument("--ite-inner-tol", type=float, default=1e-2)
            p.add_argument("--ite-tol", type=float, default=1e-10)
            p.add_argument("--ite-max-steps", type=int, default=40000)
            p.add_argument("--ite-precision", default="mixed",
                           choices=["mixed", "x64d"])
            p.add_argument("--relax-tag", default="",
                           help="suffix of the saved file (the default, "
                                "untagged file is what the sweep loads)")
    a = ap.parse_args(argv)
    t0 = time.time()
    {"segA": seg_a, "segB": seg_b, "point": point, "idle": idle,
     "collect": collect, "relax": relax}[a.cmd](a)
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
