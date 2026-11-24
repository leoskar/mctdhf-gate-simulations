"""Fig. S1: mean-field sweep fidelities of the four gates on a 2x2 grid
(layout of gate_sweeps/scripts/plot_resweep.py), with x-axis either gate time
2*tau_ramp + tau_hold (default) or tau_hold alone. Styled for the paper: full
text width of the JCP reprint layout (510 pt), Times via LaTeX/mathptmx.

Usage: python figures/1e_mf_sweep.py [--xaxis {gate,hold}] [--out FILE]
(--xaxis gate, the default, is the paper's figure; the script also writes a
zoomed variant.)
"""
from __future__ import annotations
import argparse, os, sys
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GS = os.path.abspath(os.path.join(HERE, "..", "one_electron", "gate_sweeps"))
RESULTS = os.path.join(GS, "results")
sys.path.insert(0, os.path.join(GS, "scripts"))
from plot_resweep import LEGS, load_csv

TU_TO_NS = 15.9155e-3
TEXTWIDTH_IN = 510.0 / 72.27  # revtex4-1 jcp reprint \textwidth

TITLES = {
    "w0w1_sqiswap_mf": r"$w_0w_1$",
    "w1w2_sqiswap_mf": r"$w_1w_2$",
    "w2w3_sqiswap_mf": r"$w_2w_3$",
    "w0w3_sqiswap_mf": r"$w_0w_3$",
}
XLABELS = {"gate": r"$t_{\mathrm{gate}}$ (ns)", "hold": r"$t_{\mathrm{hold}}$ (ns)"}
# legend placement per x-axis and leg, chosen to clear the data at the shared y-limits
LEGEND = {
    "gate": {
        "w0w1_sqiswap_mf": dict(loc="lower right", ncol=2),
        "w1w2_sqiswap_mf": dict(loc="lower right", ncol=2),
        "w2w3_sqiswap_mf": dict(loc="lower right", ncol=2),
        "w0w3_sqiswap_mf": dict(loc="center right", ncol=1, bbox_to_anchor=(1.0, 0.45)),
    },
    "hold": {
        "w0w1_sqiswap_mf": dict(loc="lower left", ncol=2),
        "w1w2_sqiswap_mf": dict(loc="lower right", ncol=2),
        "w2w3_sqiswap_mf": dict(loc="lower right", ncol=2),
        "w0w3_sqiswap_mf": dict(loc="center right", ncol=1, bbox_to_anchor=(1.0, 0.45)),
    },
}

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{mathptmx}",
    "font.family": "serif",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 8,
    "lines.linewidth": 1.0, "lines.markersize": 2.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
})


def plot_leg(ax, leg, zoom = False, xaxis = "gate"):
    coarse = load_csv(os.path.join(RESULTS, leg, "coarse_sweep_fidelities.csv"))
    fine = load_csv(os.path.join(RESULTS, leg, "fine_sweep_fidelities.csv"))

    ramps = sorted(coarse)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {tr: color_cycle[i % len(color_cycle)] for i, tr in enumerate(ramps)}

    ramp_weight = 2 if xaxis == "gate" else 0  # x = (ramp_weight*tau_ramp + tau_hold)
    f_max = 0.0
    ramp_max = 0
    hold_max = 0
    gate_max = 0

    for tr in ramps:
        pts = coarse[tr]
        xs = [(ramp_weight * tr + th) * TU_TO_NS for th, _ in pts]
        fbars = [fb for _, fb in pts]
        ax.plot(xs, fbars, "o-", color=colors[tr], label=f"{tr * TU_TO_NS:.2f}")
        if np.max(fbars) > f_max:
            f_max = np.max(fbars)
            ramp_max = tr
            hold_max = pts[np.argmax(fbars)][0]
            gate_max = 2 * ramp_max + hold_max

    for tr, pts in fine.items():
        color = colors.get(tr)
        if color is None:
            # fine-sweep ramp not present in coarse sweep: assign it a new color
            color = color_cycle[len(colors) % len(color_cycle)]
            colors[tr] = color
        xs = [(ramp_weight * tr + th) * TU_TO_NS for th, _ in pts]
        fbars = [fb for _, fb in pts]
        ax.plot(xs, fbars, "x", color=color)

        if np.max(fbars) > f_max:
            f_max = np.max(fbars)
            ramp_max = tr
            hold_max = pts[np.argmax(fbars)][0]
            gate_max = 2 * ramp_max + hold_max

    ax.set_xlabel(XLABELS[xaxis])
    ax.set_ylabel(r"$\bar{F}_4$")
    ax.grid(True)
    ax.set_title(TITLES[leg])
    ax.legend(title=r"$t_{\mathrm{ramp}}$ (ns)", columnspacing=1.0,
              handlelength=1.5, handletextpad=0.4, borderpad=0.4,
              **LEGEND[xaxis][leg])

    if zoom:
        ax.set_ylim(f_max - 1e-3, f_max + 1e-3)
        print(f"{leg}: max Fbar={f_max:.6f} at tau_ramp={ramp_max*TU_TO_NS:g} ns, tau_hold={hold_max*TU_TO_NS:g} ns, gate_time={gate_max*TU_TO_NS:g} ns")


def make_grid(zoom, xaxis="gate"):
    # shared y-limits in the full view; the zoomed view centres each panel on its own max
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 0.72 * TEXTWIDTH_IN),
                             sharey=not zoom)
    for i, (ax, leg) in enumerate(zip(axes.flat, LEGS)):
        plot_leg(ax, leg, zoom, xaxis)
        first_col = i % 2 == 0
        if not zoom and not first_col:
            ax.set_ylabel("")
        ax.text(-0.14 if first_col or zoom else -0.04, 1.04, f"({'abcd'[i]})",
                transform=ax.transAxes, va="bottom", ha="left")
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=1.0)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xaxis", choices=("gate", "hold"), default="gate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, "1e_mf_sweep" if args.xaxis == "gate"
                                else "1e_mf_sweep_holdtime")

    fig = make_grid(False, args.xaxis)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)

    fig = make_grid(True, args.xaxis)
    fig.savefig(args.out + "_zoomed.pdf")
    fig.savefig(args.out + "_zoomed.png", dpi=300)
    print(f"wrote {args.out}(_zoomed).{{pdf,png}}")


if __name__ == "__main__":
    main()
