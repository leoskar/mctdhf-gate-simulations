"""MF orbital ladders on 80 points, norb 6 ... 26 (Fig. 7(a): ``--pulse p3``).

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ladder80 relax  --norb 24 16
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ladder80 gate   --norb 24
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ladder80 echeck
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ladder80 relax --leg w0w1 --norb 6 8
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.ladder80 gate --leg w0w1 --pulse p3 \
        --norb 6

``--leg`` (default w1w2) and ``--pulse``: ``ladder`` (default; tau_ramp 28.3,
tau_hold 105.4, w1w2 only) or ``p3``, the leg's theta = pi/2 gate (:data:`GATE`,
the filled points of Fig. 6), checked against the norb-22 gate of
gates_80pt, ``gates_80pt/results/mf80_leg_<leg>_n22.npz``, which is the
ladder's norb-22 rung; output results/ladderp3_<leg>_n<N>.npz.

**relax.** Source: the relaxed norb-14 MF states of gates_80pt
(gates_80pt/results/mfstates80_<leg>_n14.npz). norb > 14: the active orbitals
embedded exactly (``exact_check.embed_orbitals``), as ``gates_80pt.mf80
relax`` does; norb < 14: projected onto the top-N active natural
orbitals with the top 4 natural SPFs kept (``exact_check.project_active``
with ``src=``), as exact_check's ``ladder`` mode does. Both then
relaxed at the idle H in x64d by ``exact_check.relax`` (per-iteration stop,
dE < 1e-10). If that stop never fires (w0w1 shows a period-2 cycle), the
relaxation is redone from the same start with ``gates_80pt.mf80.relax_chunk``
(chunk-end rule) and the file records it. ``--rule chunk`` goes straight
to the chunk-end rule (for w0w1, where the per-iteration stop never fires and
only burns the 40000-iteration cap). Checks (asserted inside the two
relax functions): converged, root matching > 0.99, M(0) - 1 < 1e-10; then
:func:`energy_order` over every rung present, those of gates_80pt included.
Saved as results/mfstates80_<leg>_n<N>.npz (``num_dvr`` = 80).

**gate.** ``gates_80pt.mf80.cmd_gate`` itself (kind full, ``Integ()`` defaults,
τ_ramp 28.3, τ_hold 105.4), with its output directory, state path and
Delta_80 redirected by ``orbital_convergence.common`` / here. Output
results/ladder80_w1w2_n<N>.npz, with the keys of gates_80pt's gate files. Accept:
status ok, M(0) - 1 < 1e-10, max|norm - 1| <= 1e-10; otherwise ONE rerun
with ``Integ(precision="x64")`` -> ladder80_w1w2_n<N>_x64.npz.

**echeck.** The energy-order check alone (all rungs present).

``--overlap gather`` (norb 26; also norb 24 for the
CPU relaxations of w0w1 and w2w3): installs
:mod:`orbital_convergence.overlap_patch` (the library's dense determinant overlap does not
fit at norb 26 on either device, nor at norb 24 on the CPU) and runs its spot checks on the actual 26-orbital overlap
between two different states: relax, embedded vs a randomly mixed orbital
set before relaxing and embedded vs relaxed after; gate, relaxed vs mixed
before propagating. Records -> results/overlap_spot_n<N>_<cmd>.json for
w1w2 with the ladder pulse, results/overlap_spot_<leg>_n<N>_relax.json and
overlap_spot_<leg>_<pulse>_n<N>_gate.json otherwise.
"""
import argparse
import os
import time
import types

from . import common as C  # noqa: I100  (x64 + redirections first)

import numpy as np  # noqa: E402

from gates_80pt import mf80  # noqa: E402

L, EC, fw = C.L, C.EC, C.fw
LEG = "w1w2"                    # the default leg
TAU_RAMP, TAU_HOLD = 28.3, 105.4
# each leg's theta = pi/2 gate (tau_ramp, tau_hold): the filled points of Fig. 6
GATE = {"w0w1": (12.6, 98.3), "w1w2": (9.4, 126.1), "w2w3": (8.0, 131.6)}


def pulse(leg, kind):
    """(tau_ramp, tau_hold, output prefix)."""
    if kind == "ladder":
        assert leg == "w1w2", "the ladder pulse is defined for w1w2 only"
        return TAU_RAMP, TAU_HOLD, "ladder80"
    tr, th = GATE[leg]
    ref = np.load(os.path.join(C.G80_RESULTS, f"mf80_leg_{leg}_n22.npz"))
    assert (float(ref["tau_ramp"]), float(ref["tau_hold"])) == (tr, th), \
        (leg, float(ref["tau_ramp"]), float(ref["tau_hold"]))
    assert str(ref["kind"]) == "full" and str(ref["status"]) == "ok"
    return tr, th, "ladderp3"

mf80.delta80 = C.delta80        # Delta_80 from gates_80pt/results/oppoints80.json


def weighted_E(path):
    d = np.load(path)
    assert int(d["num_dvr"]) == 80, (path, int(d["num_dvr"]))
    return float(C.W4 @ np.real(np.asarray(d["E"])))


def energy_order(leg=LEG, tol=1e-9, verbose=True):
    """Σ w_i E_i (weights 1, ½, ½, ¼, normalised) must not increase with
    norb over every rung present (those of gates_80pt included)."""
    rows = []
    for N in range(4, 40):
        p = C.mf_state_path(leg, N)
        if os.path.exists(p):
            rows.append((N, weighted_E(p), p))
    bad = []
    if verbose:
        print("  [echeck] norb  weighted E          step from previous rung",
              flush=True)
    for k, (N, E, p) in enumerate(rows):
        step = np.nan if k == 0 else E - rows[k - 1][1]
        flag = ""
        if k and step > tol:
            bad.append((rows[k - 1][0], N, step))
            flag = "   <-- INCREASES"
        if verbose:
            src = "gates_80pt" if "/gates_80pt/" in p else "orbital_convergence"
            print(f"  [echeck] {N:>4d}  {E:.10f}  {step:+.3e}  ({src}){flag}",
                  flush=True)
    return rows, bad


def require_energy_order(leg=LEG):
    _, bad = energy_order(leg)
    if bad:
        raise SystemExit("ENERGY CHECK VIOLATED (a different stationary point; "
                         "stop and report): " + "; ".join(
                             f"norb {a} -> {b}: {s:+.3e}" for a, b, s in bad))


def _overlap(a):
    if a.overlap == "library":
        return None
    assert len(a.norb) == 1 and a.norb[0] in (24, 26), \
        "--overlap gather is for one norb-24 or norb-26 rung"
    from . import overlap_patch as OP
    OP.install()
    return OP


def _spot_path(a):
    N = a.norb[0]
    if a.leg == "w1w2" and a.pulse == "ladder":     # the unsuffixed names
        name = f"overlap_spot_n{N}_{a.cmd}.json"
    else:
        tag = a.leg if a.cmd == "relax" else f"{a.leg}_{a.pulse}"
        name = f"overlap_spot_{tag}_n{N}_{a.cmd}.json"
    return C.assert_writable(os.path.join(C.RESULTS, name))


def _save_spots(a, spots):
    C.save_json(_spot_path(a), spots)


def cmd_relax(a):
    OP = _overlap(a)
    spots = []
    LEG = a.leg
    hi = mf80.h_idle(LEG)
    lay14 = mf80.layout(14)
    ml14 = L.make_ml(LEG, hi, lay14)
    y14, _ = mf80.load_state(LEG, 14, ml14)     # gates_80pt n14, asserts num_dvr
    for N in a.norb:
        assert N not in C.G80_RUNGS[LEG], f"norb {N} is a gates_80pt rung"
        assert 6 <= N <= 26, "norb outside the ladder (6-26)"
        out = C.assert_writable(C.mf_state_path(LEG, N))
        t0 = time.time()
        lay = mf80.layout(N)
        print(f"\n=== relax {LEG} norb {lay.norbs(LEG)} n_s {lay.nspfs(LEG)} "
              f"===", flush=True)
        if N > 14:
            y0 = EC.embed_orbitals(LEG, ml14, y14, lay)
            how = "embed n14 (exact_check.embed_orbitals)"
        else:
            _, y0 = EC.project_active(LEG, lay14, lay, src=(ml14, y14))
            how = "project n14 (exact_check.project_active)"
        ml_b = L.make_ml(LEG, hi, lay)
        if OP:
            spots.append(OP.spot_check(ml_b, y0, OP.mixed(ml_b, y0),
                                       f"n{N} embedded vs mixed"))
            _save_spots(a, spots)
        rule = "per-iteration (exact_check.relax)"
        try:
            if a.rule == "chunk":
                raise AssertionError("converged False (--rule chunk: "
                                     "per-iteration stop skipped)")
            ml_r, y_r = EC.relax(LEG, lay, y0, ml_b, a.inner_tol,
                                 precision="x64d", prefix="oc_")
            z = np.load(os.path.join(EC.out_dir(LEG),
                                     f"relaxed_oc_{EC.tag(lay)}.npz"))
            extra = dict(steps=int(z["steps"]), match_pop=z["match_pop"])
        except AssertionError as e:
            if "converged False" not in str(e):
                raise
            print(f"  [relax] per-iteration stop never fired ({e}); redoing "
                  f"from the same start with the chunk-end rule", flush=True)
            ml_r, y_r = mf80.relax_chunk(LEG, lay, y0, ml_b, a.inner_tol)
            rule = "chunk-end (gates_80pt.mf80.relax_chunk)"
            extra = {}
        if OP:
            spots.append(OP.spot_check(ml_r, y0, y_r,
                                       f"n{N} embedded vs relaxed"))
            _save_spots(a, spots)
            extra["overlap"] = "gather (orbital_convergence.overlap_patch)"
            extra["overlap_spot_worst"] = max(r["worst"] for r in spots)
        pops = ml_r.spf_populations(y_r)
        for s in L.LEGS[LEG]:
            print(f"  [relax] w{s} SPF populations "
                  + " ".join(f"{float(v):.2e}" for v in np.asarray(pops[s])),
                  flush=True)
        M0 = float(np.max(np.abs(np.asarray(ml_r.calculate_overlap(y_r, y_r))
                                 - np.eye(4))))
        assert M0 < 1e-10, M0
        mf80.save_state(out, ml_r, y_r, source=f"{how} + x64d relax",
                        stop_rule=rule, inner_tol=a.inner_tol, M0_err=M0,
                        wall=time.time() - t0, **extra)
        print(f"  saved -> {out}  ({time.time() - t0:.0f} s, stop rule "
              f"{rule})", flush=True)
        require_energy_order(LEG)


def _accept(path):
    z = np.load(path)
    ok = (str(z["status"]) == "ok" and float(z["M0_err"]) < 1e-10
          and float(z["max_norm_dev"]) <= 1e-10)
    return ok, str(z["status"]), float(z["M0_err"]), float(z["max_norm_dev"])


def cmd_gate(a):
    LEG = a.leg
    tau_ramp, tau_hold, prefix = pulse(LEG, a.pulse)
    require_energy_order(LEG)
    OP = _overlap(a)
    if OP:
        N = a.norb[0]
        ml = L.make_ml(LEG, mf80.h_idle(LEG), mf80.layout(N))
        y, _ = mf80.load_state(LEG, N, ml)
        _save_spots(a, [OP.spot_check(ml, y, OP.mixed(ml, y),
                                      f"n{N} relaxed vs mixed")])
    for N in a.norb:
        out = f"{prefix}_{LEG}_n{N}"
        C.assert_writable(os.path.join(C.RESULTS, out + ".npz"))
        g = types.SimpleNamespace(leg=LEG, norb=[N], kind="full",
                                  tau_ramp=tau_ramp, tau_hold=tau_hold,
                                  nsub_spf=16, nsub_orb=4, out=out)
        assert mf80.integ_of(g) == L.Integ(), "not the Integ() defaults"
        failed = None
        try:
            mf80.cmd_gate(g)
            ok, st, M0, nd = _accept(os.path.join(C.RESULTS, out + ".npz"))
            print(f"  [accept] {out}: status {st}, M(0)-1 {M0:.1e}, "
                  f"max|norm-1| {nd:.1e} -> {'ACCEPTED' if ok else 'FAILED'}",
                  flush=True)
            if not ok:
                failed = f"status {st}, max|norm-1| {nd:.1e}"
        except Exception as e:  # a numerical failure inside the propagation
            if "RESOURCE_EXHAUSTED" in str(e) or "OUT_OF_MEMORY" in str(e):
                raise           # memory, not numerics: no x64 rerun
            failed = f"{type(e).__name__}: {e}"
            print(f"  [accept] {out}: FAILED with {failed}", flush=True)
        if failed:
            print(f"\n  rerunning {out} ONCE with Integ(precision='x64')",
                  flush=True)
            g.out = out + "_x64"
            orig = mf80.integ_of
            mf80.integ_of = lambda _a: L.Integ(precision="x64")
            try:
                mf80.cmd_gate(g)
            finally:
                mf80.integ_of = orig
            ok, st, M0, nd = _accept(os.path.join(C.RESULTS, g.out + ".npz"))
            print(f"  [accept] {g.out}: status {st}, M(0)-1 {M0:.1e}, "
                  f"max|norm-1| {nd:.1e} -> {'ACCEPTED' if ok else 'FAILED'}",
                  flush=True)


def cmd_echeck(a):
    require_energy_order(a.leg)
    print("  [echeck] PASS", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["relax", "gate", "echeck"])
    p.add_argument("--norb", type=int, nargs="+", default=[])
    p.add_argument("--leg", default=LEG, choices=sorted(GATE))
    p.add_argument("--pulse", default="ladder", choices=["ladder", "p3"])
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--rule", default="auto", choices=["auto", "chunk"])
    p.add_argument("--overlap", choices=["library", "gather"],
                   default="library")
    a = p.parse_args(argv)
    C.require_grid(80)
    L.device_banner()
    t0 = time.time()
    {"relax": cmd_relax, "gate": cmd_gate, "echeck": cmd_echeck}[a.cmd](a)
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
