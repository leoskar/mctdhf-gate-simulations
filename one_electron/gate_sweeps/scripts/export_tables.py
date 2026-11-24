"""Export the sweep fidelities as flat CSV tables, one pair per leg.

    results/<leg>_sqiswap_mf/coarse_sweep_fidelities.csv
    results/<leg>_sqiswap_mf/fine_sweep_fidelities.csv

The files have no comment or metadata lines (a header row, then data), so
`pd.read_csv` and `np.genfromtxt(..., names=True, delimiter=",")` read them
directly.

"coarse" = the 270-point grid of grid.py. "fine" = every refinement point
added on top of it. The split is by membership in that grid, so the two
files are disjoint and their union is the complete data set; concatenate them
for a full (tau_ramp, tau_hold) map.

Rows are sorted by (tau_ramp, tau_hold) so each ramp row is contiguous and
already in hold order -- ready to plot as a line per ramp without re-sorting.

Read-only with respect to the propagation artifacts; writes only the CSVs.
"""
from __future__ import annotations
import csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from grid import GRID

TU_TO_NS = 15.9155e-3

COLUMNS = [
    "tau_ramp", "tau_hold", "T_total_tu", "T_total_ns",
    "Fbar", "F_xc", "transfer",
    "sub_00", "sub_10", "sub_01", "sub_11", "retention_max",
    "doubles_00", "doubles_10", "doubles_01", "doubles_11", "doubles_max",
    "finite", "wall_s",
]


def rows_for(leg):
    d = os.path.join(GS_DIR, "results", f"{leg}_sqiswap_mf", "sweep_pts")
    out = []
    for p in sorted(glob.glob(os.path.join(d, "pt_tr*_th*.npz"))):
        z = np.load(p, allow_pickle=True)
        sub = np.asarray(z["subspace"], dtype=float)
        dbl = (np.abs(np.asarray(z["doubles"])) ** 2).sum(axis=0)  # per input col
        T = float(z["T_total"])
        out.append({
            "tau_ramp": float(z["tau_ramp"]), "tau_hold": float(z["tau_hold"]),
            "T_total_tu": T, "T_total_ns": T * TU_TO_NS,
            "Fbar": float(z["Fbar"]), "F_xc": float(z["F_xc"]),
            "transfer": float(z["transfer"]),
            "sub_00": sub[0], "sub_10": sub[1], "sub_01": sub[2], "sub_11": sub[3],
            "retention_max": float(sub.max()),
            "doubles_00": dbl[0], "doubles_10": dbl[1],
            "doubles_01": dbl[2], "doubles_11": dbl[3],
            "doubles_max": float(dbl.max()),
            "finite": int(bool(z["finite"])),
            "wall_s": float(z["wall"]),
        })
    return out


def fmt(v):
    if isinstance(v, int):
        return str(v)
    if abs(v) >= 1e-4 or v == 0.0:
        return f"{v:.10g}"
    return f"{v:.6e}"


def main():
    grand = 0
    for leg, g in GRID.items():
        coarse_keys = {(round(r, 6), round(h, 6))
                       for r, _, hs in g["rows"] for h in hs}
        rows = rows_for(leg)
        if not rows:
            print(f"{leg}: no points — skipped"); continue
        split = {"coarse": [], "fine": []}
        for r in rows:
            k = (round(r["tau_ramp"], 6), round(r["tau_hold"], 6))
            split["coarse" if k in coarse_keys else "fine"].append(r)
        outdir = os.path.join(GS_DIR, "results", f"{leg}_sqiswap_mf")
        assert outdir.startswith(os.path.join(GS_DIR, "results"))
        for tag, rs in split.items():
            rs.sort(key=lambda x: (x["tau_ramp"], x["tau_hold"]))
            f = os.path.join(outdir, f"{tag}_sweep_fidelities.csv")
            with open(f, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(COLUMNS)
                for r in rs:
                    w.writerow([fmt(r[c]) for c in COLUMNS])
            print(f"  {os.path.relpath(f, GS_DIR)}: {len(rs)} rows")
            grand += len(rs)
        # integrity: the two files must partition the data set exactly
        assert len(split["coarse"]) + len(split["fine"]) == len(rows)
        assert len(split["coarse"]) <= len(coarse_keys)
    print(f"total rows written: {grand}")


if __name__ == "__main__":
    main()
