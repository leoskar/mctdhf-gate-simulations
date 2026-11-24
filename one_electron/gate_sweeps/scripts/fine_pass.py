"""Build the fine-pass segment-B manifests from the coarse sweep argmax.

At the ARGMAX RAMP ONLY (the ramp axis is flat to <1e-3), add holds at
tau_hold_opt + m*step_fine for m = -2,-1,+1,+2. These reuse that ramp's existing
segment-A checkpoint, so only segment B runs.

Guards: the new hold must be > 0, on the 0.1 t.u. grid, and <= that ramp's
th_max (segA only checkpointed that far, so a larger hold has no t_c).
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from grid import GRID, CPU_LEGS, GPU_LEGS

M = (-2, -1, 1, 2)


def measured(leg):
    """{tau_ramp: {tau_hold: Fbar}} over every usable point on disk."""
    d = os.path.join(GS_DIR, "results", f"{leg}_sqiswap_mf", "sweep_pts")
    out = {}
    for p in glob.glob(os.path.join(d, "pt_tr*_th*.npz")):
        z = np.load(p, allow_pickle=True)
        sub = np.asarray(z["subspace"], dtype=float)
        if not bool(z["finite"]) or sub.max() > 1 + 1e-6:
            continue
        out.setdefault(round(float(z["tau_ramp"]), 6), {})[
            round(float(z["tau_hold"]), 6)] = float(z["Fbar"])
    return out


def top_rows(leg, n):
    """The n ramp rows with the highest MEASURED peak, each with that peak.

    The ranking on the measured peak is itself subject to the hold-grid
    phase error this pass exists to remove, so the
    ranking can move once the refined points land.
    """
    m = measured(leg)
    rows = [(r, max(d, key=d.get), max(d.values())) for r, d in m.items() if d]
    rows.sort(key=lambda x: -x[2])
    return rows[:n], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="w0w1,w1w2,w2w3,w0w3")
    ap.add_argument("--top", default="w0w1=3,w1w2=3,w2w3=3,w0w3=2",
                    help="how many top-peak ramp rows to refine, per leg")
    ap.add_argument("--ramps", default="",
                    help="refine these EXPLICIT ramp rows instead of the "
                         "top-peak ones, e.g. 'w1w2=8.9,w2w3=7.0'. Use when the "
                         "selection criterion is not fidelity (e.g. gate speed).")
    args = ap.parse_args()
    want = args.legs.split(",")
    ntop = {k: int(v) for k, v in (x.split("=") for x in args.top.split(","))}
    explicit = {}
    for x in filter(None, args.ramps.split(",")):
        k, v = x.split("=")
        explicit.setdefault(k, []).append(float(v))
    cpu, gpu, report = [], [], []
    for leg, g in GRID.items():
        if leg not in want:
            continue
        if leg in explicit:
            m = measured(leg)
            rows = []
            for r in explicit[leg]:
                r = round(r, 6)
                assert r in m, f"{leg}: no measured points at tau_ramp={r:g}"
                rows.append((r, max(m[r], key=m[r].get), max(m[r].values())))
        else:
            rows, m = top_rows(leg, ntop.get(leg, 3))
        if not rows:
            print(f"{leg}: no usable points — skipped"); continue
        thmax_of = {r: tm for r, tm, _ in g["rows"]}
        step = g["step_fine"]
        how = ("EXPLICIT ramp rows" if leg in explicit
               else f"the top {len(rows)} ramp rows")
        print(f"\n{leg}: refining {how}, step_fine={step:g}")
        for rank, (tr, th, f) in enumerate(rows, 1):
            thmax = thmax_of[tr]
            # Every hold ALREADY measured for this ramp -- coarse grid or an
            # earlier fine batch. Reuse is per-ramp: the same hold on a
            # different ramp is a different point.
            done = set(m.get(tr, {}))
            new = []
            for k in M:
                h = round(th + k * step, 6)
                if h <= 0:
                    print(f"   {leg} tr={tr:g}: skip {k:+d} -> {h:g} <= 0"); continue
                if h > thmax + 1e-9:
                    print(f"   {leg} tr={tr:g}: skip {k:+d} -> {h:g} > th_max "
                          f"{thmax:g} (no segA checkpoint)"); continue
                assert abs(round(h * 10) - h * 10) < 1e-6, f"{leg}: {h} off grid"
                if h in done:
                    print(f"   {leg} tr={tr:g}: skip {k:+d} -> {h:g} already measured")
                    continue
                new.append(h)
            print(f"   #{rank} tr={tr:6g}  peak {f:.5f} @ {th:g}  -> {len(new)} new: "
                  + " ".join(f"{h:g}" for h in new))
            report.append((leg, tr, th, f, new))
            tgt = gpu if leg in GPU_LEGS else cpu
            for h in new:
                tgt.append(f"{leg} {tr:g} {h:g}")
    for name, rows in [("manifest_segB_fine_cpu.txt", cpu),
                       ("manifest_segB_fine_gpu.txt", gpu)]:
        with open(os.path.join(HERE, name), "w") as fh:
            fh.write("\n".join(rows) + ("\n" if rows else ""))
        print(f"{name}: {len(rows)} lines")


if __name__ == "__main__":
    main()
