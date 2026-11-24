"""Hold scans at norb 22, MF layout, 80 points, for w1w2, w0w1 and w2w3
(Fig. 6).

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.holdscan80 segA
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.holdscan80 repro      # segB gate hold + check
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.holdscan80 segB       # the grid holds
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.holdscan80 segB --holds 169.2 178.2
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.holdscan80 collect
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.holdscan80 segA --leg w0w1

``--leg`` defaults to w1w2, whose summary files carry no leg suffix.

``gate_timing.sweep`` unchanged (segA: ramp-up + exact plateau with a checkpoint
at every hold; segB: restart at t_c with the genuine envelope), wrapped:

* ``sweep.layout_from`` -> its Layout with ``norb_active = 22`` (it fixes 14);
* the norb-22 state ``gates_80pt/results/mfstates80_<leg>_n22.npz`` is copied to
  ``mf_lib.relaxed_path(leg, layout)`` under orbital_convergence's ``MF_STATE_DIR``
  (``num_dvr`` asserted 80 on the way);
* ``sweep.static_delta`` -> Delta_80 (gates_80pt/results/oppoints80.json);
* ``mf_lib.RESULTS_DIR`` -> orbital_convergence/results/holdscan, so the run directory is
  orbital_convergence/results/holdscan/<leg>/<Integ tag><Layout tag>/.

Holds: the 13 grid holds for the leg's τ_ramp in gate_timing/grids.json plus
the hold of the leg's theta = pi/2 gate in gates_80pt (the gate hold; w1w2
126.1, not a grid hold; w0w1 98.3 and w2w3 131.6 are grid holds). ``--delta-frac 0.5``: segB restarts at t_c = τ_hold + 0.5 τ_ramp,
where env_up == env exactly (segA asserts it); the 64-point sweeps used the
same.

**repro** runs segB for the gate hold alone, then requires F and retention[3]
to reproduce gates_80pt/results/mf80_leg_<leg>_n22.npz to <= 1e-6 (exit status 1
otherwise, so the dependent segB jobs never start). **segB** without
``--holds`` runs every hold but the gate hold; give ``--holds`` to split the
holds over jobs (JAX's memory pool grows across holds in one process).
"""
import argparse
import dataclasses
import os
import sys
import time

from . import common as C  # noqa: I100  (x64 + redirections first)

import numpy as np  # noqa: E402

from gate_timing import sweep as S  # noqa: E402

L, fw = C.L, C.fw
NORB = 22
# leg -> (tau_ramp, gate hold, first and last grid hold)
LEGS = {"w1w2": (9.4, 126.1, 70.5, 178.2),
        "w0w1": (12.6, 98.3, 53.7, 143.0),
        "w2w3": (8.0, 131.6, 75.4, 187.8)}
L.RESULTS_DIR = os.path.join(C.RESULTS, "holdscan")

_layout_from = S.layout_from


def layout_from(a):
    return dataclasses.replace(_layout_from(a), norb_active=NORB)


S.layout_from = layout_from
S.static_delta = C.delta80


def leg_holds(leg):
    """(tau_ramp, gate hold, grid holds, all holds) of a leg."""
    tau_ramp, hold_g, first, last = LEGS[leg]
    grid = S.grid_holds(leg, tau_ramp)
    assert len(grid) == 13 and grid[0] == first and grid[-1] == last, grid
    return tau_ramp, hold_g, grid, sorted(set(grid) | {hold_g})


def common_args(leg):
    return ["--leg", leg, "--tau-ramp", f"{LEGS[leg][0]}", "--delta-frac",
            "0.5"]


def _sweep_args():
    """sweep.py's layout arguments at their defaults (as every call here
    passes them)."""
    return argparse.Namespace(norb_spec=3, nspf_spec=1, init="relaxed",
                              weights="relax", init_tag="")


def stage_state(leg):
    """The norb-22 relaxed states where ``mf_initial_states`` looks."""
    lay = layout_from(_sweep_args())
    assert lay == dataclasses.replace(L.Layout(), norb_active=NORB), lay
    # the gates_80pt file for every leg (for w1w2 this is C.mf_state_path's choice)
    src = os.path.join(C.G80_RESULTS, f"mfstates80_{leg}_n{NORB}.npz")
    assert "/gates_80pt/" in src, src
    d = dict(np.load(src))
    assert int(d["num_dvr"]) == 80, (src, int(d["num_dvr"]))
    dst = C.assert_writable(L.relaxed_path(leg, lay))
    if os.path.exists(dst):
        e = np.load(dst)
        same = all(np.array_equal(e[k], d[k]) for k in
                   ["A", "E"] + [f"{x}{s}" for x in "Bb" for s in L.WELLS])
        assert same, f"{dst} exists and differs from {src}"
        print(f"  [stage] {dst} present, identical to {src}", flush=True)
    else:
        L.save_npz(dst, **d)
        print(f"  [stage] copied {src} -> {dst}", flush=True)
    return lay


def run_dir(leg):
    return os.path.join(L.RESULTS_DIR, leg,
                        L.Integ().tag() + layout_from(_sweep_args()).tag())


def repro_json(leg):
    """w1w2's file has no leg suffix."""
    name = ("holdscan_repro.json" if leg == "w1w2"
            else f"holdscan_repro_{leg}.json")
    return C.assert_writable(os.path.join(C.RESULTS, name))


def repro_check(leg):
    tag = S._tag
    tau_ramp, hold_g = LEGS[leg][:2]
    pt = os.path.join(run_dir(leg), f"pt_tr{tag(tau_ramp)}_th{tag(hold_g)}.npz")
    ref = np.load(os.path.join(C.G80_RESULTS, f"mf80_leg_{leg}_n22.npz"))
    assert float(ref["tau_ramp"]) == tau_ramp \
        and float(ref["tau_hold"]) == hold_g, (float(ref["tau_ramp"]),
                                           float(ref["tau_hold"]))
    z = np.load(pt)
    dF = float(z["F"]) - float(ref["F"])
    dr = float(z["retention"][3]) - float(ref["retention"][3])
    dth = float(z["theta"]) - float(ref["theta"])
    ok = abs(dF) <= 1e-6 and abs(dr) <= 1e-6
    print(f"\n=== reproduction check, {leg} τ_hold {hold_g} vs gates_80pt ===\n"
          f"  F            {float(z['F']):.9f}  ref {float(ref['F']):.9f}  "
          f"diff {dF:+.2e}\n"
          f"  retention[3] {float(z['retention'][3]):.9f}  ref "
          f"{float(ref['retention'][3]):.9f}  diff {dr:+.2e}\n"
          f"  theta        diff {dth:+.2e};  max|M - M_ref| "
          f"{float(np.max(np.abs(z['M'] - ref['M']))):.2e}\n"
          f"  -> {'PASS' if ok else 'FAIL'}", flush=True)
    C.save_json(repro_json(leg),
                dict(point=pt, F=float(z["F"]), F_ref=float(ref["F"]), dF=dF,
                     ret11=float(z["retention"][3]),
                     ret11_ref=float(ref["retention"][3]), dret11=dr,
                     dtheta=dth, passed=ok))
    if not ok:
        raise SystemExit("REPRODUCTION CHECK FAILED: the wiring differs from "
                         "the gates_80pt gate")


def holds_arg(holds):
    return ["--holds", ",".join(f"{h:g}" for h in holds)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["segA", "repro", "segB", "check",
                                      "collect"])
    ap.add_argument("--leg", default="w1w2", choices=sorted(LEGS))
    ap.add_argument("--holds", type=float, nargs="+", default=None,
                    help="segB: a subset of the grid holds (a fresh process "
                    "for holds a previous segB job did not reach)")
    a = ap.parse_args(argv)
    C.require_grid(80)
    leg = a.leg
    tau_ramp, hold_g, grid_holds, all_holds = leg_holds(leg)
    common = common_args(leg)
    stage_state(leg)
    print(f"  leg {leg}  tau_ramp {tau_ramp}  gate hold {hold_g}\n"
          f"  holds {all_holds}\n  run dir {run_dir(leg)}", flush=True)
    if a.which == "segA":
        S.main(["segA", *common, *holds_arg(all_holds)])
    elif a.which == "repro":
        S.main(["segB", *common, *holds_arg([hold_g])])
        repro_check(leg)
    elif a.which == "segB":
        holds = ([h for h in grid_holds if h != hold_g] if a.holds is None
                 else a.holds)
        assert set(holds) <= set(grid_holds), holds
        S.main(["segB", *common, *holds_arg(holds)])
    elif a.which == "check":
        repro_check(leg)
    else:
        S.main(["collect", "--leg", leg])


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1:])
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
