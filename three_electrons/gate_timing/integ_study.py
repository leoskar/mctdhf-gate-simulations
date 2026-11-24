"""Real-time integrator study for the MF gate legs.

Midpoint CMF only. Every arm starts from the SAME MF initial states
(``mf_lib.mf_initial_states``), propagates one window, and saves its final
state and gate matrix; ``compare`` scores a group of arms against its
reference.

    integ_study.py timing  --leg w1w2 --tau-ramp 6 [--tops ...] [--precs ...]
    integ_study.py run     --group G --leg w0w1 --window up --tau-ramp 6 [integ]
    integ_study.py ladder  --group G --leg w0w1 --window up --tau-ramp 6 \\
                           --dts 2e-2,1e-2,5e-3 [integ]
    integ_study.py compare --group G --ref <arm name>

Windows (``--window``):

``up``    the leading edge: ``env_up`` from 0 to ``--t-end`` (default
          tau_ramp). The integration error is generated in
          the ramp edges (``system/fourwell.make_env_up``), so this is the
          stress window.
``full``  the genuine ramp-hold-ramp gate, 0 -> 2 tau_ramp + tau_hold.
``idle``  constant idle Hamiltonian, 0 -> ``--t-end``.

Scored against the reference arm: ``|dFbar|`` (full windows), ``max|d|M||``
(the modulus of the gate matrix, phase-blind), ``max|dM|`` (phase-sensitive),
retention and norm drift; ``compare`` also fits the order between
consecutive dt rungs.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from . import mf_lib as L
from .sweep import _tag

STUDY_DIR = os.path.join(L.RESULTS_DIR, "study")


def add_integ_args(p):
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
    p.add_argument("--init", default="relaxed", choices=["relaxed", "projected"])
    p.add_argument("--weights", default="relax", choices=["relax", "equal"])


def layout_of(a):
    return L.Layout(norb_spec=a.norb_spec, nspf_spec=a.nspf_spec,
                    init=a.init, weights=a.weights)


def integ_of(a, **over):
    kw = dict(dt=a.dt, a_prop=a.a_prop, krylov=a.krylov,
              precision=a.precision, cn_tol=a.cn_tol, inner_tol=a.inner_tol,
              max_iter=a.max_iter, nsub_spf=a.nsub_spf, nsub_orb=a.nsub_orb,
              regauge_t=a.regauge_t, B_prop=a.B_prop, b_prop=a.b_prop,
              dop_sub=a.dop_sub, renorm=a.renorm)
    kw.update(over)
    return L.Integ(**kw)


def window(a):
    """``(env, t_end, T_eff)`` for the chosen window."""
    if a.window == "up":
        env, _ = L.make_env(a.tau_ramp, 1.0, "up")
        return env, (a.t_end or a.tau_ramp), None
    if a.window == "full":
        env, T = L.make_env(a.tau_ramp, a.tau_hold, "full")
        return env, T, a.tau_ramp + a.tau_hold
    if a.window == "idle":
        env, _ = L.make_env(1.0, 1.0, "idle")
        return env, a.t_end, None
    raise ValueError(a.window)


def run_one(a, integ, name=None, ml_y0=None):
    layout = layout_of(a)
    env, t_end, T_eff = window(a)
    if ml_y0 is None:
        ml = L.make_ml(a.leg, L.h_list_for(a.leg, env), layout)
        y0, _ = L.mf_initial_states(a.leg, ml, layout, verbose=False)
    else:
        ml, y0 = ml_y0
    name = name or integ.tag()
    print(f"\n  --- arm {name}: {a.leg} {a.window} tau_r={a.tau_ramp} "
          f"t_end={t_end} ---", flush=True)
    P = L.Propagator(ml, integ)
    wall = time.time()
    y, status, _, sps, tcomp = P.run(y0, 0.0, t_end, probe=L.norm_probe(ml),
                                     verbose=a.verbose)
    wall = time.time() - wall
    m = L.gate_metrics(ml, y0, y, T_eff, None, fid=(a.window != "up"))
    L.print_metrics(m)
    print(f"  status {status}  {sps * 1e3:.3f} ms/step  compile {tcomp:.1f} s"
          f"  wall {wall:.0f} s", flush=True)
    pops = m.pop("spf_pops")
    d = os.path.join(STUDY_DIR, a.group)
    path = os.path.join(d, f"{name}.npz")
    L.save_npz(path, **m, spf_pops=np.concatenate([np.asarray(p) for p in pops]),
               y=np.asarray(y), status=status, sec_per_step=sps,
               compile_s=tcomp, wall=wall, leg=a.leg, window=a.window,
               tau_ramp=a.tau_ramp, tau_hold=a.tau_hold or np.nan,
               t_end=t_end, integ=json.dumps(integ.as_dict(), sort_keys=True),
               backend=__import__("jax").default_backend())
    print(f"  saved -> {path}", flush=True)
    return ml, y0


# ── commands ──────────────────────────────────────────────────────────────

def cmd_run(a):
    L.device_banner()
    run_one(a, integ_of(a), a.name)


def cmd_ladder(a):
    L.device_banner()
    dts = [float(v) for v in a.dts.split(",")]
    cache = None
    for dt in dts:
        integ = integ_of(a, dt=dt)
        cache = run_one(a, integ, None, cache)


def cmd_variants(a):
    """Several discretisations in one process, given as JSON overrides:
    ``--variants '[{"cn_tol":1e-8},{"nsub_spf":2,"nsub_orb":2}]'``."""
    L.device_banner()
    cache = None
    for over in json.loads(a.variants):
        cache = run_one(a, integ_of(a, **over), None, cache)


def cmd_timing(a):
    """s/step per (top propagator, precision): ramp window, then plateau.

    Short windows on the w1w2 (or given) leg, ``env_up`` with the given
    tau_ramp: ``[0, tau_r]`` (H moving) and ``[tau_r, tau_r + plateau]``
    (H frozen at the target, the hold the sweep spends most of its time in).
    """
    import jax
    L.device_banner()
    layout = layout_of(a)
    env, _ = L.make_env(a.tau_ramp, 1.0, "up")
    ml = L.make_ml(a.leg, L.h_list_for(a.leg, env), layout)
    y0, _ = L.mf_initial_states(a.leg, ml, layout, verbose=True)
    rows = []
    for top in a.tops.split(","):
        a_prop, k = (("sil", int(top[3:])) if top.startswith("sil")
                     else (top, a.krylov))
        for prec in a.precs.split(","):
            integ = integ_of(a, a_prop=a_prop, krylov=k, precision=prec)
            try:
                P = L.Propagator(ml, integ)
                y1, st1, _, s_ramp, tcomp = P.run(y0, 0.0, a.tau_ramp,
                                                  verbose=False)
                y2, st2, _, s_plat, _ = P.run(y1, a.tau_ramp,
                                              a.tau_ramp + a.plateau,
                                              verbose=False)
                n = np.abs(np.diag(np.asarray(ml.calculate_overlap(y2, y2))))
                row = dict(top=top, prec=prec, ramp_ms=s_ramp * 1e3,
                           plateau_ms=s_plat * 1e3, compile_s=tcomp,
                           drift=float(np.max(np.abs(n - 1))),
                           status=f"{st1}/{st2}")
            except Exception as e:                       # noqa: BLE001
                row = dict(top=top, prec=prec, error=repr(e)[:200])
            print("  " + json.dumps(row), flush=True)
            rows.append(row)
    path = os.path.join(STUDY_DIR, "timing",
                        f"timing_{a.leg}_{jax.default_backend()}"
                        f"_dt{_tag(a.dt)}{a.out_tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(dict(leg=a.leg, dt=a.dt, tau_ramp=a.tau_ramp,
                       plateau=a.plateau, backend=jax.default_backend(),
                       device=str(jax.devices()[0]), rows=rows), fh, indent=1)
    print(f"  saved -> {path}", flush=True)


def load_arm(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def state_diff(ml, arm, ref):
    """Per-state overlap of the arm's final state with the reference's:
    ``infid = max_a 1 - |<ref_a|y_a>| / (|ref_a| |y_a|)`` (phase-blind) and
    ``dpsi = max_a | <ref_a|y_a> / (|ref_a| |y_a|) - 1 |`` (keeps the phase).
    After a ramp |M| against the t=0 anchors can be ~0 and carries nothing.
    """
    import jax.numpy as jnp
    yr, ya = jnp.asarray(ref["y"]), jnp.asarray(arm["y"])
    S = np.diag(np.asarray(ml.calculate_overlap(yr, ya)))
    nr = np.sqrt(np.abs(np.diag(np.asarray(ml.calculate_overlap(yr, yr)))))
    na = np.sqrt(np.abs(np.diag(np.asarray(ml.calculate_overlap(ya, ya)))))
    Sn = S / (nr * na)
    # The gate fidelity between the two propagations: the full 4x4
    # C_ij = <ref_i|y_j> (normalised), scored by the same F̄ as the gates,
    # against the identity (mf_lib.fbar: local-Z on both sides + global phase
    # maximised out). It keeps amplitudes, cross-state terms and the
    # conditional phase; only single-qubit Z phases are removed.
    C = np.asarray(ml.calculate_overlap(yr, ya)) / np.outer(nr, na)
    F_ref, _ = L.fbar(L.M_NULL, C)
    ph = np.angle(np.diag(C))
    cond = float(np.angle(np.exp(1j * (ph[0] + ph[3] - ph[1] - ph[2]))))
    return dict(infid=float(np.max(1 - np.abs(Sn))),
                dpsi=float(np.max(np.abs(Sn - 1))),
                gate_infid=float(1 - F_ref), dcond=cond)


def diff(arm, ref):
    out = dict(
        dM=float(np.max(np.abs(arm["M"] - ref["M"]))),
        dabsM=float(np.max(np.abs(np.abs(arm["M"]) - np.abs(ref["M"])))),
        norm_drift=float(np.max(np.abs(arm["norms"] - 1))),
        ret_min=float(np.min(arm["retention"])),
        ms=float(arm["sec_per_step"]) * 1e3)
    if "F" in arm and "F" in ref:
        out["dF"] = float(abs(arm["F"] - ref["F"]))
        out["F"] = float(arm["F"])
    return out


def cmd_compare(a):
    d = os.path.join(STUDY_DIR, a.group)
    ref = load_arm(os.path.join(d, a.ref + ".npz"))
    leg = str(ref["leg"])
    lay = layout_of(a)
    ml = L.make_ml(leg, L.h_list_for(leg, L.make_env(1.0, 1.0, "idle")[0]), lay)
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        name = os.path.basename(p)[:-4]
        if name == a.ref:
            continue
        arm = load_arm(p)
        r = diff(arm, ref)
        r.update(state_diff(ml, arm, ref))
        r["name"] = name
        r["dt"] = json.loads(str(arm["integ"]))["dt"]
        rows.append(r)
    rows.sort(key=lambda r: (-r["dt"], r["name"]))
    print(f"=== {a.group}: {len(rows)} arms against {a.ref} ===")
    hdr = (f"  {'arm':<58} {'1-Fbar(ref)':>11} {'dphi_cond':>10} "
           f"{'1-|ovl|':>9} {'|dpsi|':>9} {'max|dM|':>9} "
           f"{'max|d|M||':>10} {'|dF|':>9} {'norm drift':>10} {'ms/step':>8} "
           f"{'ord(dpsi)':>9}")
    print(hdr)
    prev = {}
    for r in rows:
        # order against the previous rung of the SAME series (name minus dt)
        key = r["name"].replace(f"_dt{r['dt']:g}", "_dt*")
        order = ""
        p = prev.get(key)
        if p is not None and p["dt"] != r["dt"] and r["dpsi"] > 0:
            order = f"{np.log(p['dpsi'] / r['dpsi']) / np.log(p['dt'] / r['dt']):.2f}"
        r["order_dpsi"] = order
        dF = f"{r['dF']:.2e}" if "dF" in r else "-"
        print(f"  {r['name']:<58} {r['gate_infid']:>11.2e} {r['dcond']:>+10.2e} "
              f"{r['infid']:>9.2e} {r['dpsi']:>9.2e} "
              f"{r['dM']:>9.2e} {r['dabsM']:>10.2e} {dF:>9} "
              f"{r['norm_drift']:>10.2e} {r['ms']:>8.2f} {order:>9}")
        prev[key] = r
    out = os.path.join(d, f"compare_vs_{a.ref}.json")
    with open(out, "w") as fh:
        json.dump(dict(ref=a.ref, rows=rows), fh, indent=1)
    print(f"  saved -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "ladder", "variants", "timing"):
        p = sub.add_parser(name)
        p.add_argument("--leg", required=True, choices=list(L.LEGS))
        p.add_argument("--tau-ramp", type=float, required=True)
        p.add_argument("--verbose", action="store_true")
        add_integ_args(p)
        if name != "timing":
            p.add_argument("--group", required=True)
            p.add_argument("--window", default="up",
                           choices=["up", "full", "idle"])
            p.add_argument("--tau-hold", type=float, default=None)
            p.add_argument("--t-end", type=float, default=None)
        if name == "run":
            p.add_argument("--name", default=None)
        if name == "ladder":
            p.add_argument("--dts", required=True)
        if name == "variants":
            p.add_argument("--variants", required=True)
        if name == "timing":
            p.add_argument("--tops", default="diagonalize,sil6,sil10,cn")
            p.add_argument("--precs", default="x64,mixed")
            p.add_argument("--plateau", type=float, default=2.0)
            p.add_argument("--out-tag", default="")
    p = sub.add_parser("compare")
    p.add_argument("--group", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--norb-spec", type=int, default=3)
    p.add_argument("--nspf-spec", type=int, default=1)
    p.add_argument("--init", default="relaxed", choices=["relaxed", "projected"])
    p.add_argument("--weights", default="relax", choices=["relax", "equal"])
    a = ap.parse_args(argv)
    {"run": cmd_run, "ladder": cmd_ladder, "variants": cmd_variants,
     "timing": cmd_timing, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
