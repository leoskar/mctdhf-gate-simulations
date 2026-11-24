"""2x2 grid: one subplot per leg, coarse+fine sqiSWAP re-sweep fidelities.

Reads <leg>/coarse_sweep_fidelities.csv and <leg>/fine_sweep_fidelities.csv
for each of the four legs in results/. Per subplot: x = tau_hold, y = Fbar,
one "o-" line per tau_ramp (coarse sweep), with the matching fine-sweep
points overplotted as "x" markers in the same color (matched by tau_ramp).

Usage: python scripts/plot_resweep.py [--out FILE]
"""
from __future__ import annotations
import argparse, csv, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GS_DIR = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(GS_DIR, "results")

LEGS = ["w0w1_sqiswap_mf", "w1w2_sqiswap_mf", "w2w3_sqiswap_mf", "w0w3_sqiswap_mf"]


def load_csv(path):
    """Returns dict: tau_ramp -> list of (tau_hold, Fbar), hold-sorted."""
    by_ramp = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tr = float(row["tau_ramp"])
            th = float(row["tau_hold"])
            fbar = float(row["Fbar"])
            by_ramp.setdefault(tr, []).append((th, fbar))
    for tr in by_ramp:
        by_ramp[tr].sort()
    return by_ramp


def plot_leg(ax, leg):
    coarse = load_csv(os.path.join(RESULTS, leg, "coarse_sweep_fidelities.csv"))
    fine = load_csv(os.path.join(RESULTS, leg, "fine_sweep_fidelities.csv"))

    ramps = sorted(coarse)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {tr: color_cycle[i % len(color_cycle)] for i, tr in enumerate(ramps)}

    for tr in ramps:
        pts = coarse[tr]
        holds = [p[0] for p in pts]
        fbars = [p[1] for p in pts]
        ax.plot(holds, fbars, "o-", color=colors[tr], label=f"tau_ramp={tr:g}")

    for tr, pts in fine.items():
        color = colors.get(tr)
        if color is None:
            # fine-sweep ramp not present in coarse sweep: assign it a new color
            color = color_cycle[len(colors) % len(color_cycle)]
            colors[tr] = color
        holds = [p[0] for p in pts]
        fbars = [p[1] for p in pts]
        ax.plot(holds, fbars, "x", color=color)

    ax.set_xlabel("tau_hold")
    ax.set_ylabel("Fbar")
    ax.set_title(leg)
    ax.legend(fontsize="small")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(GS_DIR, "results", "resweep_fbar_grid.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, leg in zip(axes.flat, LEGS):
        plot_leg(ax, leg)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
