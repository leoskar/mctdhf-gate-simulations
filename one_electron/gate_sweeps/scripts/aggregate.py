"""Read-only aggregation of the sqiSWAP re-sweep.

Per leg: the point table sorted by Fbar, the Fbar map over (tau_ramp, tau_hold),
the argmax, and the bracket check --

  * the argmax must be INTERIOR in both axes (not on the first/last swept ramp,
    not on the first/last hold of its own ramp row);
  * each ramp row's OWN peak must not sit at a hold-grid edge.

Also the per-point sanity gates: finite=True and per-column subspace retention
<= 1 (retention > 1 means the propagation is unstable and the point is not
physics -- it must be discarded, not read).

Writes nothing outside gate_sweeps/; pass --json to dump a machine-readable
summary next to the tables.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from grid import GRID, EXPECTED_POINTS

RET_TOL = 1e-6          # retention > 1 + RET_TOL is unstable, not rounding

# calibrations before this sweep, for comparison
CURRENT = {"w0w1": (60, 89.3, 0.99523), "w1w2": (30, 116.8, 0.99923),
           "w2w3": (30, 115.2, 0.99808), "w0w3": (10, 2111.3, 0.8380)}


def load_leg(leg):
    d = os.path.join(GS_DIR, "results", f"{leg}_sqiswap_mf", "sweep_pts")
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "pt_tr*_th*.npz"))):
        z = np.load(p, allow_pickle=True)
        sub = np.asarray(z["subspace"], dtype=float)
        rows.append(dict(tr=float(z["tau_ramp"]), th=float(z["tau_hold"]),
                         Fbar=float(z["Fbar"]), Fxc=float(z["F_xc"]),
                         transfer=float(z["transfer"]), sub=sub,
                         s11=float(sub[3]), ret_max=float(sub.max()),
                         finite=bool(z["finite"]),
                         T_ns=float(z["T_total"]) * 15.9155e-3, path=p))
    return rows


def bracket_check(leg, rows, ok):
    """Returns (violations, argmax_row)."""
    v = []
    ramps = [r for r, _, _ in GRID[leg]["rows"]]
    # Holds actually measured for each ramp = coarse grid UNION fine pass. Using
    # the coarse grid alone would hide a fine-pass point from the row-peak test
    # and could call a refined interior peak an "edge".
    holds_of = {r: sorted({round(h, 6) for h in hs}
                          | {round(x["th"], 6) for x in ok
                             if abs(x["tr"] - r) < 1e-9})
                for r, _, hs in GRID[leg]["rows"]}
    if not ok:
        return ["no usable points"], None
    best = max(ok, key=lambda r: r["Fbar"])
    # axis 1: the argmax ramp must not be the first or last swept ramp
    if abs(best["tr"] - ramps[0]) < 1e-9 or abs(best["tr"] - ramps[-1]) < 1e-9:
        v.append(f"argmax ramp {best['tr']:g} is at a RAMP-GRID EDGE "
                 f"(swept {ramps[0]:g}..{ramps[-1]:g})")
    # axis 2: the argmax hold must not be the first or last hold of its row
    hr = holds_of[best["tr"]]
    if abs(best["th"] - hr[0]) < 1e-9 or abs(best["th"] - hr[-1]) < 1e-9:
        v.append(f"argmax hold {best['th']:g} is at a HOLD-GRID EDGE for "
                 f"tau_ramp={best['tr']:g} ({hr[0]:g}..{hr[-1]:g})")
    # every ramp row's own peak must be interior in hold
    for r in ramps:
        row = [x for x in ok if abs(x["tr"] - r) < 1e-9]
        if not row:
            v.append(f"tau_ramp={r:g}: NO points")
            continue
        rb = max(row, key=lambda x: x["Fbar"])
        hs = holds_of[r]
        if abs(rb["th"] - hs[0]) < 1e-9 or abs(rb["th"] - hs[-1]) < 1e-9:
            v.append(f"tau_ramp={r:g}: row peak at hold edge "
                     f"{rb['th']:g} (Fbar {rb['Fbar']:.5f}, row "
                     f"{hs[0]:g}..{hs[-1]:g})")
    return v, best


def fmap(leg, rows):
    ramps = [r for r, _, _ in GRID[leg]["rows"]]
    coarse = {r: set(np.round(h, 6)) for r, _, h in GRID[leg]["rows"]}
    holds = {r: sorted({round(h, 6) for h in hs}
                       | {round(x["th"], 6) for x in rows
                          if abs(x["tr"] - r) < 1e-9})
             for r, _, hs in GRID[leg]["rows"]}
    by = {(r["tr"], r["th"]): r for r in rows}
    out = ["  Fbar map (rows = tau_ramp, cols = that ramp's holds, low -> high;"
           " * = row peak, + = fine-pass point)"]
    for r in ramps:
        cells = []
        row = [by.get((r, h)) for h in holds[r]]
        vals = [x["Fbar"] if x else -1 for x in row]
        pk = int(np.argmax(vals)) if max(vals) > -1 else -1
        for i, x in enumerate(row):
            if x is None:
                cells.append("  ---- "); continue
            mark = "*" if i == pk else ("+" if holds[r][i] not in coarse[r] else " ")
            cells.append(mark + f"{x['Fbar']:.4f}")
        out.append(f"   tr={r:6.1f} | " + " ".join(cells))
        out.append(f"     holds   | " + " ".join(f"{h:6.1f}" for h in holds[r])
                   + (" " * 0))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="w0w1,w1w2,w2w3,w0w3")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    summary, any_viol = {}, False
    for leg in args.legs.split(","):
        rows = load_leg(leg)
        bad_fin = [r for r in rows if not r["finite"]]
        bad_ret = [r for r in rows if r["ret_max"] > 1 + RET_TOL]
        ok = [r for r in rows if r["finite"] and r["ret_max"] <= 1 + RET_TOL]
        print(f"\n{'='*78}\n=== {leg}  ({len(rows)}/{EXPECTED_POINTS[leg]} points "
              f"present, {len(ok)} usable) ===\n{'='*78}")
        if len(rows) < EXPECTED_POINTS[leg]:
            print(f"  !! INCOMPLETE: {EXPECTED_POINTS[leg]-len(rows)} coarse "
                  f"points missing")
        elif len(rows) > EXPECTED_POINTS[leg]:
            print(f"  ({len(rows)-EXPECTED_POINTS[leg]} fine-pass points beyond "
                  f"the {EXPECTED_POINTS[leg]}-point coarse grid)")
        for r in bad_fin:
            print(f"  !! NON-FINITE  tr={r['tr']:g} th={r['th']:g}  -> DISCARDED")
        for r in bad_ret:
            print(f"  !! RETENTION {r['ret_max']:.4f} > 1 (UNSTABLE, not physics) "
                  f"tr={r['tr']:g} th={r['th']:g}  -> DISCARDED")
        any_viol |= bool(bad_fin or bad_ret)
        if not ok:
            continue
        print("\n  top 10 by Fbar:")
        print("   tau_ramp  tau_hold     Fbar      F_xc   transfer  |11>sub  T(ns)")
        for r in sorted(ok, key=lambda x: -x["Fbar"])[:10]:
            print(f"   {r['tr']:7.1f}  {r['th']:8.1f}   {r['Fbar']:.5f}  "
                  f"{r['Fxc']:.5f}   {r['transfer']:.4f}   {r['s11']:.4f}  "
                  f"{r['T_ns']:6.2f}")
        print()
        print(fmap(leg, ok))
        viol, best = bracket_check(leg, rows, ok)
        print(f"\n  ARGMAX: tau_ramp={best['tr']:g}  tau_hold={best['th']:g}  "
              f"Fbar={best['Fbar']:.5f}  T={best['T_ns']:.2f} ns")
        cr, ch, cf = CURRENT[leg]
        print(f"  current calibration: ({cr:g}, {ch:g}) Fbar {cf:.5f}   "
              f"-> dF = {best['Fbar']-cf:+.5f},  d(tau_ramp) = "
              f"{best['tr']-cr:+g},  d(tau_hold) = {best['th']-ch:+g}")
        if viol:
            any_viol = True
            print("  !! BRACKET CHECK FAILED:")
            for x in viol:
                print(f"     !! {x}")
        else:
            print("  bracket check: PASS (argmax interior in both axes; "
                  "no ramp row peaks at a hold edge)")
        summary[leg] = dict(n=len(rows), n_ok=len(ok),
                            argmax=dict(tau_ramp=best["tr"], tau_hold=best["th"],
                                        Fbar=best["Fbar"], F_xc=best["Fxc"],
                                        transfer=best["transfer"],
                                        s11=best["s11"], T_ns=best["T_ns"]),
                            violations=viol,
                            discarded=[dict(tr=r["tr"], th=r["th"],
                                            ret=r["ret_max"], finite=r["finite"])
                                       for r in bad_fin + bad_ret])
    print(f"\n{'='*78}")
    print("OVERALL: " + ("VIOLATIONS PRESENT — see !! lines above"
                         if any_viol else "all gates PASS"))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
