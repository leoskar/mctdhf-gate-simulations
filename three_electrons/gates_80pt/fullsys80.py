"""The full-system track on the 80-point grid: all four wells correlated
(n_s 4 everywhere), the 16-state register, per-well orbital budgets.

    FOURWELL_NUM_DVR=80 python -m gates_80pt.fullsys80 relax --leg w1w2 --nstar 20
    FOURWELL_NUM_DVR=80 python -m gates_80pt.fullsys80 pilot --leg w1w2 --nstar 20
    FOURWELL_NUM_DVR=80 python -m gates_80pt.fullsys80 leg   --leg w1w2 --nstar 20

**relax**. The converged norb-16 80-point idle register (n_s 4 on all
wells; after register80's budget16 check: spectators at norb 16),
the leg's two active wells' orbitals embedded exactly into norb N*
(``register80.embed_orbitals_register`` restricted to those wells), then the
16 states relaxed together in x64d at their own weights 2^-n at idle
(``routed.relax_register``'s recipe: chunk-end sum-E change < 1e-10, top
block re-diagonalised and roots matched by identity, exact regauge). Saved
as results/register80_<leg>_n<N*>s16.npz.

**pilot**. 2 t.u. of the leg pulse from the relaxed register if it
exists, else from the exact embedding (identical cost; M(0) = 1 either way).
status, M(0) - 1, max|norm - 1|, ms/step, XLA setup time.

**leg**. One 16-state propagation at the leg's theta = pi/2 schedule
(results/pulses80.json), checkpointed every 10 t.u. (resumable). Scores
(routed.py helpers, imported): F4 of the spectators-|00> corner against
sqrt(iSWAP) and theta; F16 against sqrt(iSWAP)(i,j) x I maximised over local
Z on all four qubits (``fxc_4q``); column retention of all 16 inputs and the
minimum; the pair-|11> retention by spectator state. Saved as
results/fs80_leg_<leg>_n<N*>s16.npz.

Anchors are phase-fixed (dominant top-tensor element real positive) before
any scoring; the configuration order is the solver's (``register_configs``),
mapped to bit order by ``routed.bit_perm``.
"""
import argparse
import json
import os
import time

from . import common as T  # noqa: I100  (x64 + redirections first)
from . import register80 as R

import jax
import jax.numpy as jnp
import numpy as np

from mctdhf.MLMCTDHF_n_species import MLMCTDHF_n_species_DVR, make_regauge_ml
from routed_gates.routed import (M16_of, bit_perm, corner4, fxc_4q, pair_gate16,
                                 strip_localZ, theta_of, SQI)

L, fw = T.L, T.fw
NSPF, NSTATES = 4, 16
# After the failed budget16 check of register80, the full-system base is the
# converged norb-16 idle register and the spectators carry norb 16, not 14.
NSPEC = 16
BASE = os.path.join(T.RESULTS, "register80_idle_norb16.npz")
WIN, SNAP = 10.0, 0.1
PULSES = os.path.join(T.RESULTS, "pulses80.json")


def budgets(leg, nstar):
    return [nstar if w in T.LEGS3[leg] else NSPEC for w in range(4)]


def reg_path(leg, nstar):
    return os.path.join(T.RESULTS, f"register80_{leg}_n{nstar}s{NSPEC}.npz")


def load_idle():
    d = np.load(BASE)
    assert int(d["num_dvr"]) == fw.NUM_DVR
    assert (int(d["norb"]), int(d["n_spf"])) == (NSPEC, NSPF)
    cfgs = [tuple(int(x) for x in c) for c in d["configs"]]
    w = np.asarray(d["relax_weights"], dtype=np.float64)
    st = (jnp.asarray(d["A"]), [jnp.asarray(d[f"B{s}"]) for s in range(4)],
          [jnp.asarray(d[f"b{s}"]) for s in range(4)])
    return st, cfgs, w / w.sum(), np.asarray(d["x_ghz"], dtype=float)


def embed(st, leg, nstar):
    """Active wells' orbitals NSPEC -> N*, exact; the spectators untouched."""
    A, Bs, bs = st
    if nstar == NSPEC:
        return st
    _A, Bn, bn = R.embed_orbitals_register(st, NSPEC, nstar)
    Bs, bs = list(Bs), list(bs)
    for w in T.LEGS3[leg]:
        Bs[w], bs[w] = Bn[w], bn[w]
    return (A, Bs, bs)


def make_ml(leg, nstar, h_list, weights, norbs=None):
    ml = MLMCTDHF_n_species_DVR(
        h_list=h_list, g_dict=fw.g_dict(),
        num_particles=[fw.NUM_PARTICLES] * 4,
        num_mctdhf_orbitals=norbs or budgets(leg, nstar),
        num_spatial_orbitals=[fw.NUM_DVR] * 4,
        num_spfs=[NSPF] * 4, num_states=NSTATES, imag_time=False, eps=1e-6,
        e_shift=0.0, state_weights=weights, ci_sparse=True)
    assert ml.e_shift == 0.0
    ml.state_weights = jnp.asarray(weights)
    return ml


def h_idle_list(x_ghz):
    xi = x_ghz / fw.TYPICAL_FREQ_GHZ
    return list(fw.h_ramp(xi, xi, L.make_env(1.0, 1.0, "idle")[0]))


def h_leg_list(leg, tau_ramp, tau_hold):
    xi, xg = L.x_vectors(leg)
    env, _ = L.make_env(tau_ramp, tau_hold, "full")
    return list(fw.h_ramp(xi / fw.TYPICAL_FREQ_GHZ, xg / fw.TYPICAL_FREQ_GHZ,
                          env))


def cmd_relax(a):
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt
    t0 = time.time()
    leg, N = a.leg, a.nstar
    st, cfgs, w, x = load_idle()
    h_idle = fw.h_static(x / fw.TYPICAL_FREQ_GHZ)
    ite = MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(), num_particles=[fw.NUM_PARTICLES] * 4,
        num_mctdhf_orbitals=budgets(leg, N),
        num_spatial_orbitals=[fw.NUM_DVR] * 4, num_spfs=[NSPF] * 4,
        relax_configs=cfgs, relax_weights=list(w), all_configs=cfgs,
        dt=2e-2, precision="x64d", inner_tol=a.inner_tol, chunk=50, eps=1e-6)
    ml = make_ml(leg, N, h_idle_list(x), w)
    st = embed(st, leg, N)
    A, Bs, bs = st
    E0 = np.real(np.asarray(ml.calculate_energy(0.0, ml._ABb_to_y(A, Bs, bs))))
    st14, _, _, _ = load_idle()
    ml14 = make_ml(leg, NSPEC, h_idle_list(x), w)
    E14 = np.real(np.asarray(ml14.calculate_energy(
        0.0, ml14._ABb_to_y(*st14))))
    print(f"  [embed] budgets {budgets(leg, N)}: max|dE| vs norb {NSPEC} "
          f"{np.max(np.abs(E0 - E14)):.2e}", flush=True)
    assert np.max(np.abs(E0 - E14)) < 1e-8
    state, steps, e_prev, converged, fallback = st, 0, None, False, []
    while steps < a.max_steps:
        state, e, info = ite.integrate(ite.chunk, h_idle, state)
        fallback += [steps // ite.chunk] if info["fallback_chunks"] else []
        steps += len(e)
        if not np.all(np.isfinite(e)):
            break
        dch = np.nan if e_prev is None else e[-1] - e_prev
        if steps % 500 == 0 or (e_prev is not None and abs(dch) < 1e-10):
            print(f"    it {steps:6d}  sum E = {e[-1]:.10f}  dE(chunk) = "
                  f"{dch:+.3e}  ({time.time() - t0:.0f} s)", flush=True)
        if e_prev is not None and abs(dch) < 1e-10:
            converged = True
            break
        e_prev = e[-1]
    h = ite.as_h(h_idle)
    A, Bs, bs = state
    Bs_rg, evals, evecs, ovl = ite._front_match(A, Bs, bs, h)
    cols, pop = ite._assign(ovl)
    A = evecs[:, jnp.asarray(cols)].T.reshape(ml.a_shape)
    y = make_regauge_ml(ml)(ml._ABb_to_y(A, list(Bs_rg), list(bs)))
    E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    err = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y))
                              - np.eye(NSTATES))))
    an = ite.analyse(ml._y_to_ABb(y), h_idle)
    print(f"  [relax] converged {converged} after {steps} it "
          f"({time.time() - t0:.0f} s), fallbacks {fallback}", flush=True)
    print("  [relax] match pop " + " ".join(f"{v:.6f}" for v in np.asarray(pop)),
          flush=True)
    print("  [relax] E - E_embedded " + " ".join(f"{v:+.2e}" for v in E - E0)
          + f"   weighted dE {np.dot(w, E - E0):+.3e}   M(0)-1 {err:.1e}",
          flush=True)
    print("  [relax] detunings " + " ".join(f"{v:.6f}" for v in an["detunings"])
          + f"  healthy {an['healthy']}  min P "
          f"{np.diag(an['F_matched']).min():.5f}", flush=True)
    A, Bs, bs = ml._y_to_ABb(y)
    path = reg_path(leg, N)
    L.save_npz(path, A=np.asarray(A),
               **{f"B{s}": np.asarray(Bs[s]) for s in range(4)},
               **{f"b{s}": np.asarray(bs[s]) for s in range(4)},
               E=E, E_embedded=E0, converged=bool(converged), steps=int(steps),
               inner_tol=a.inner_tol, match_pop=np.asarray(pop),
               configs=np.asarray(cfgs), relax_weights=w,
               detunings=np.asarray(an["detunings"]),
               F_matched=np.asarray(an["F_matched"]),
               budgets=np.asarray(budgets(leg, N)), x_ghz=x,
               num_dvr=fw.NUM_DVR, wall=time.time() - t0)
    print(f"  -> {path}", flush=True)
    assert converged and np.all(np.isfinite(E)), "relaxation failed"
    assert np.min(np.asarray(pop)) > 0.99, f"root matching failed: {pop}"
    assert err < 1e-10, f"relaxed M(0) != 1: {err:.3e}"


def phase_fixed(st, cfgs):
    A = np.asarray(st[0]).copy()
    ph = []
    for a, c in enumerate(cfgs):
        k = np.unravel_index(np.argmax(np.abs(A[a])), A[a].shape)
        assert tuple(int(v) for v in k) == c, (
            f"row {a}: dominant SPF index {k} != config {c}")
        ph.append(float(np.degrees(np.angle(A[a][k]))))
        A[a] *= np.exp(-1j * np.angle(A[a][k]))
    return (A, st[1], st[2]), ph


def start_state(leg, N, allow_embedded):
    path = reg_path(leg, N)
    st0, cfgs, w, x = load_idle()
    if os.path.exists(path):
        d = np.load(path)
        assert int(d["num_dvr"]) == fw.NUM_DVR
        assert list(d["budgets"]) == budgets(leg, N)
        st = (np.asarray(d["A"]), [np.asarray(d[f"B{s}"]) for s in range(4)],
              [np.asarray(d[f"b{s}"]) for s in range(4)])
        src = path
    elif allow_embedded:
        st, src = embed(st0, leg, N), "embedded (unrelaxed)"
    else:
        raise FileNotFoundError(f"{path}: run `relax` first")
    st, ph = phase_fixed(st, cfgs)
    return st, cfgs, w, src, ph


def norms_of(ml, y):
    return np.abs(np.diag(np.asarray(ml.calculate_overlap(y, y))))


def _mem():
    try:
        s = jax.devices()[0].memory_stats() or {}
        return s.get("peak_bytes_in_use", 0) / 2 ** 30
    except Exception:  # noqa: BLE001
        return float("nan")


def cmd_pilot(a):
    leg, N = a.leg, a.nstar
    norbs = None
    if a.all:
        # costing run: every well at N* (exact embedding of the base)
        st0, cfgs, w, _x = load_idle()
        _A, Bn, bn = R.embed_orbitals_register(st0, NSPEC, N)
        st, _ph = phase_fixed((st0[0], Bn, bn), cfgs)
        src, norbs = f"embedded, all wells norb {N}", [N] * 4
    else:
        st, cfgs, w, src, _ = start_state(leg, N, allow_embedded=True)
    tr, th = a.tau_ramp, a.tau_hold
    L.device_banner()
    t0 = time.time()
    ml = make_ml(leg, N, h_leg_list(leg, tr, th), w, norbs)
    y0 = ml._ABb_to_y(jnp.asarray(st[0]), [jnp.asarray(B) for B in st[1]],
                      [jnp.asarray(b) for b in st[2]])
    M0 = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y0, y0))
                             - np.eye(NSTATES))))
    t_build = time.time() - t0
    prop = L.Propagator(ml, L.Integ())
    y, status, _, sps, setup = prop.run(y0, 0.0, a.T, report_every=5)
    n = norms_of(ml, y)
    out = dict(leg=leg, budgets=norbs or budgets(leg, N), source=src,
               status=status,
               M0_err=M0, max_norm_dev=float(np.max(np.abs(n - 1))),
               ms_per_step=sps * 1e3, setup_first_block_s=setup,
               build_s=t_build, peak_GiB=_mem(), T=a.T, pulse=[tr, th],
               wall=time.time() - t0)
    for k, v in out.items():
        print(f"  {k:<22} {v}", flush=True)
    ok = (status == "ok" and M0 <= 1e-10 and out["max_norm_dev"] <= 1e-9)
    print(f"  PILOT {'PASS' if ok else 'FAIL'}", flush=True)
    tag = f"all{N}" if a.all else f"n{N}s{NSPEC}"
    T.save_json(os.path.join(T.RESULTS, f"pilot_fs80_{leg}_{tag}.json"),
                dict(out, passed=ok))


def score16(M16, leg):
    """The scores of a 16-state leg."""
    pair = T.LEGS3[leg]
    c4 = corner4(M16, pair)
    F4, fxc4 = L.fbar(L.M_REF, c4)
    th = theta_of(c4)
    U = pair_gate16(SQI, pair)
    fxc = fxc_4q(U, M16)
    nm = float(np.vdot(M16, M16).real)
    F16 = (nm + fxc * float(np.vdot(U, U).real) * nm) / (16 * 17)
    surv = np.sum(np.abs(M16) ** 2, axis=0)
    # pair-|11> retention by spectator state: column with the pair in |11>
    # and the spectators in (s1, s2); weight inside the computational
    # subspace with the spectators unchanged (rows of the same corner)
    i, j = pair
    spec = [w for w in range(4) if w not in pair]
    by_spec = {}
    for s1 in (0, 1):
        for s2 in (0, 1):
            def idx(bi, bj):
                bits = [0] * 4
                bits[i], bits[j] = bi, bj
                bits[spec[0]], bits[spec[1]] = s1, s2
                return (bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3]
            col = idx(1, 1)
            rows = [idx(0, 0), idx(1, 0), idx(0, 1), idx(1, 1)]
            by_spec[f"{s1}{s2}"] = dict(
                ret11_corner=float(np.sum(np.abs(M16[rows, col]) ** 2)),
                survival=float(surv[col]),
                p11_to_11=float(abs(M16[col, col]) ** 2))
    _, zp, rb, ra = strip_localZ(c4, SQI)
    ret4 = np.sum(np.abs(c4) ** 2, axis=0)
    return dict(corner=c4, F4=float(F4), Fxc4=float(fxc4), theta=float(th),
                theta_over_pi2=float(th / (np.pi / 2)), F16=float(F16),
                Fxc16=float(fxc), norm2_M16=nm, survival=surv,
                min_survival=float(surv.min()), corner_retention=ret4,
                ret11_by_spectator=by_spec, zframe=zp,
                raw_overlap=float(rb), stripped_overlap=float(ra))


def cmd_leg(a):
    leg, N = a.leg, a.nstar
    pulses = T.load_json(PULSES)
    tr, th = pulses[leg]["tau_ramp"], pulses[leg]["tau_hold"]
    Ttot = round(2 * tr + th, 6)
    st, cfgs, w, src, ph = start_state(leg, N, allow_embedded=False)
    L.device_banner()
    print(f"=== full-system leg {leg} budgets {budgets(leg, N)} n_s 4, 16 "
          f"states, pulse tau_ramp {tr} tau_hold {th} T {Ttot} ===\n"
          f"  start {src}; anchor phases before fix (deg) "
          + " ".join(f"{v:.1f}" for v in ph), flush=True)
    t0 = time.time()
    ml = make_ml(leg, N, h_leg_list(leg, tr, th), w)
    y0 = ml._ABb_to_y(jnp.asarray(st[0]), [jnp.asarray(B) for B in st[1]],
                      [jnp.asarray(b) for b in st[2]])
    M0 = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y0, y0))
                             - np.eye(NSTATES))))
    print(f"  M(0) - 1 = {M0:.2e}", flush=True)
    assert M0 < 1e-10
    integ = L.Integ()
    prop = L.Propagator(ml, integ)
    ck = os.path.join(T.RESULTS, f"fs80_leg_{leg}_n{N}s{NSPEC}.ckpt.npz")
    meta = dict(leg=leg, budgets=budgets(leg, N), pulse=[tr, th],
                integ=integ.as_dict(), source=src)
    y, t_cur, log = y0, 0.0, []
    if os.path.exists(ck):
        d = np.load(ck)
        assert json.loads(str(d["meta"])) == meta, "checkpoint settings differ"
        y, t_cur = jnp.asarray(d["y"]), float(d["t"])
        log = json.loads(str(d["log"]))
        print(f"  [resume] from t = {t_cur}", flush=True)
    P = bit_perm(cfgs)
    sps_all = []
    while t_cur < Ttot - 1e-9:
        t1 = round(min(t_cur + WIN, Ttot), 6)
        nblk = round((t1 - t_cur) / SNAP)
        y, status, _, s, setup = prop.run(y, t_cur, t1, verbose=False,
                                          report_every=max(1, nblk))
        if status != "ok":
            raise SystemExit(f"LEG FAILED {status} in ({t_cur}, {t1}]")
        n = norms_of(ml, y)
        M16 = M16_of(ml, y0, y, P)
        c4 = corner4(M16, T.LEGS3[leg])
        info = dict(t=t1, max_norm_dev=float(np.max(np.abs(n - 1))),
                    ms_per_step=float(s * 1e3), setup_s=float(setup),
                    theta=theta_of(c4),
                    corner_ret=np.sum(np.abs(c4) ** 2, 0).tolist())
        log.append(info)
        sps_all.append(s)
        print(f"    t {t1:7.1f}/{Ttot}  {s * 1e3:7.1f} ms/step  max|n-1| "
              f"{info['max_norm_dev']:.2e}  theta/(pi/2) "
              f"{info['theta'] / (np.pi / 2):.4f}  corner ret "
              + " ".join(f"{v:.4f}" for v in info["corner_ret"]), flush=True)
        L.save_npz(ck, y=np.asarray(y), t=t1, log=json.dumps(log),
                   meta=json.dumps(meta))
        t_cur = t1
    M16 = M16_of(ml, y0, y, P)
    sc = score16(M16, leg)
    print(f"  F4 (spectators |00>) = {sc['F4']:.6f}  theta/(pi/2) = "
          f"{sc['theta_over_pi2']:.4f}  corner retention "
          + " ".join(f"{v:.4f}" for v in sc["corner_retention"]), flush=True)
    print(f"  F16 = {sc['F16']:.6f}  (F_xc16 {sc['Fxc16']:.6f}, ||M16||^2 "
          f"{sc['norm2_M16']:.4f})  min survival {sc['min_survival']:.4f}",
          flush=True)
    print("  survival (bit order) " + " ".join(f"{v:.4f}" for v in sc["survival"]),
          flush=True)
    for k, v in sc["ret11_by_spectator"].items():
        print(f"  spectators |{k}>: pair-|11> column survival "
              f"{v['survival']:.4f}, in its corner {v['ret11_corner']:.4f}, "
              f"|11>->|11> {v['p11_to_11']:.4f}", flush=True)
    max_norm = max(r["max_norm_dev"] for r in log)
    out = os.path.join(T.RESULTS, f"fs80_leg_{leg}_n{N}s{NSPEC}.npz")
    L.save_npz(out, M16=M16, configs=np.asarray(cfgs),
               **{k: v for k, v in sc.items() if k != "ret11_by_spectator"},
               ret11_by_spectator=json.dumps(sc["ret11_by_spectator"]),
               log=json.dumps(log), meta=json.dumps(meta), T=Ttot,
               tau_ramp=tr, tau_hold=th, budgets=np.asarray(budgets(leg, N)),
               ms_per_step=float(np.nanmedian(sps_all)) * 1e3 if sps_all
               else np.nan, max_norm_dev=max_norm, M0_err=M0,
               num_dvr=fw.NUM_DVR, y=np.asarray(y), wall=time.time() - t0)
    print(f"  -> {out}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["relax", "pilot", "leg"])
    p.add_argument("--leg", default="w1w2", choices=list(T.LEGS3))
    p.add_argument("--nstar", type=int, default=20)
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--tau-ramp", type=float, default=9.4)
    p.add_argument("--tau-hold", type=float, default=126.2)
    p.add_argument("--all", action="store_true",
                   help="pilot only: every well at --nstar (costing run)")
    a = p.parse_args(argv)
    T.require_grid(80)
    {"relax": cmd_relax, "pilot": cmd_pilot, "leg": cmd_leg}[a.cmd](a)


if __name__ == "__main__":
    main()
