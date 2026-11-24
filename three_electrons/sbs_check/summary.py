"""Table of the SBS check from the saved files (numpy only, no solver).

    python sbs_check/summary.py

Rows: the gates_80pt four-SBS gates (gates_80pt/results/mf80_leg_<leg>_n22.npz) and
this directory's results/mf80_leg_<leg>_n22_s<n>.npz; then the differences
from four SBS, then the relaxed-state checks (results/relax_*.json).
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIER1 = os.path.join(HERE, "..", "gates_80pt", "results")
RES = os.path.join(HERE, "results")
LEGS = ("w0w1", "w1w2", "w2w3")
BASE_JOBS = {"w0w1": 6823, "w1w2": 6820, "w2w3": 6822}


PLATEAU = {"w0w1": (750, 1000)}   # sbs.py plateau states A, B (gates _it<k>)


def gate(leg, n, tag=""):
    p = (os.path.join(TIER1, f"mf80_leg_{leg}_n22.npz") if n == 4
         else os.path.join(RES, f"mf80_leg_{leg}_n22_s{n}{tag}.npz"))
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return dict(F=float(d["F"]), ret=np.asarray(d["retention"]),
                dth=float(d["theta"]) / (np.pi / 2) - 1, theta=float(d["theta"]),
                norm=float(d["max_norm_dev"]), ms=float(d["sec_per_step"]) * 1e3,
                wall=float(d["wall"]), status=str(d["status"]),
                Fxc=float(d["Fxc"]), integ=json.loads(str(d["integ"])),
                layout=json.loads(str(d["layout"])))


def main():
    rows = {}
    print("| gate | n_s | F̄₄ | \\|11⟩ ret | θ/(π/2) − 1 | ret \\|00⟩ \\|10⟩ \\|01⟩ "
          "| max\\|norm − 1\\| | ms/step | wall | status |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for leg in LEGS:
        for n in (4, 6, 8):
            g = gate(leg, n)
            if g is None:
                continue
            rows[leg, n] = g
            r = g["ret"]
            print(f"| {leg} | {n} | {g['F']:.5f} | {r[3]:.4f} | {g['dth']:+.3%} "
                  f"| {r[0]:.6f} {r[1]:.5f} {r[2]:.5f} | {g['norm']:.1e} "
                  f"| {g['ms']:.1f} | {g['wall'] / 60:.0f} min | {g['status']} |")
    print("\n| gate | n_s | ΔF̄₄ | Δ\\|11⟩ ret | Δθ (rad) | Δ(θ/(π/2)) |")
    print("|---|---|---|---|---|---|")
    for leg in LEGS:
        b = rows.get((leg, 4))
        for n in (6, 8):
            g = rows.get((leg, n))
            if g is None or b is None:
                continue
            print(f"| {leg} | {n} − 4 | {g['F'] - b['F']:+.2e} "
                  f"| {g['ret'][3] - b['ret'][3]:+.4f} "
                  f"| {g['theta'] - b['theta']:+.2e} "
                  f"| {g['dth'] - b['dth']:+.3%} |")
        if (leg, 6) in rows and (leg, 8) in rows:
            g6, g8 = rows[leg, 6], rows[leg, 8]
            print(f"| {leg} | 8 − 6 | {g8['F'] - g6['F']:+.2e} "
                  f"| {g8['ret'][3] - g6['ret'][3]:+.4f} "
                  f"| {g8['theta'] - g6['theta']:+.2e} "
                  f"| {g8['dth'] - g6['dth']:+.3%} |")
    for (leg, n), g in rows.items():
        if n != 4:
            b = rows.get((leg, 4))
            same = b is not None and g["integ"] == b["integ"]
            lay_diff = {k: (b["layout"][k], v) for k, v in g["layout"].items()
                        if b and b["layout"][k] != v}
            print(f"  {leg} s{n}: integ identical to baseline {same}; "
                  f"layout differences {lay_diff}")
    print()
    for p in sorted(glob.glob(os.path.join(RES, "relax_*.json"))):
        c = json.load(open(p))
        print(f"{os.path.basename(p)}: diag|<emb|rel>| "
              + " ".join(f"{v:.8f}" for v in c["diag_overlap"])
              + f", max offdiag {c['max_offdiag_overlap']:.1e}, order "
              f"{c['order_ok']}; dE " + " ".join(f"{v:+.2e}" for v in c["dE"])
              + f"; dEbar {c['dEbar']:+.2e}; M0-1 {c['M0_err']:.1e}; "
              f"passed {c['passed']}")
        for w, pp in c["spf_pops"].items():
            print(f"    {w} SBS pops " + " ".join(f"{v:.2e}" for v in pp))
    for leg, its in PLATEAU.items():
        b = rows.get((leg, 4))
        gs = {k: gate(leg, 6, f"_it{k}") for k in its}
        print(f"\n{leg} six-SBS plateau gates (states at ITE iteration k)")
        for k, g in gs.items():
            if g is None:
                print(f"| {leg} | 6, it {k} | not finished |")
                continue
            r = g["ret"]
            print(f"| {leg} | 6, it {k} | {g['F']:.5f} | {r[3]:.4f} | {g['dth']:+.3%} "
                  f"| {r[0]:.6f} {r[1]:.5f} {r[2]:.5f} | {g['norm']:.1e} "
                  f"| {g['ms']:.1f} | {g['wall'] / 60:.0f} min | {g['status']} "
                  f"| integ = baseline {g['integ'] == b['integ']} |")
        pairs = [(f"it{k} − 4", gs[k], b) for k in its]
        if all(gs.values()):
            pairs.append((f"it{its[0]} − it{its[1]}", gs[its[0]], gs[its[1]]))
        for name, g, h in pairs:
            if g is None:
                continue
            print(f"| {leg} | {name} | {g['F'] - h['F']:+.2e} "
                  f"| {g['ret'][3] - h['ret'][3]:+.4f} "
                  f"| {g['theta'] - h['theta']:+.2e} | {g['dth'] - h['dth']:+.3%} "
                  "| ret " + " ".join(f"{v:+.1e}" for v in g['ret'] - h['ret'])
                  + f" | F_xc {g['Fxc'] - h['Fxc']:+.2e} |")
    for p in sorted(glob.glob(os.path.join(RES, "pilot_*.json"))):
        c = json.load(open(p))
        print(f"{os.path.basename(p)}: {c['status']} {c['ms_per_step']:.1f} "
              f"ms/step, peak {c['peak_GiB']:.2f} GiB, max|n-1| "
              f"{c['max_norm_dev']:.1e}, projected 3 gates "
              f"{c['projected_total_h']:.1f} h"
              + (f", max|dret| vs baseline {c['max_dret_vs_baseline']:.1e}"
                 if "max_dret_vs_baseline" in c else ""))


if __name__ == "__main__":
    main()
