"""Shared style for the Sec. IV.B figures (3e_gate, 3e_convergence).

Type: Times via usetex + mathptmx, the manuscript's text font
(main.tex loads mathptmx), at the sizes of the paper's 1e figure
(mf_fig_wide.py): labels 9 pt, ticks 8 pt, legends 9 pt, panel labels
10 pt.  Colours are matplotlib's
default cycle (C0, C1, ...), as in Fig. 4.  Each leg also carries its own
marker shape, so identity never rests on colour alone (print, CVD).
"""
import os

import matplotlib as mpl

mpl.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "three_electrons"))
MF = os.path.join(SIMS, "gate_timing")
RES = os.path.join(MF, "results")
LEAK = os.path.join(SIMS, "grid_convergence", "results")

SWEEP_SUB = "diagonalize_mix1e-4_dt0.01_tol1e-10_rg0.1_s16o4_rnstep"
LEGS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3)}
LEG_TEX = {leg: r"$w_%s w_%s$" % (leg[1], leg[3]) for leg in LEGS}
COL = {"w0w1": "C0", "w1w2": "C1", "w2w3": "C2"}
MRK = {"w0w1": "o", "w1w2": "s", "w2w3": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
WINDOW = "#e9e8e4"                                    # exchange-window shade

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{braket}",
    "font.size": 10, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
    "lines.linewidth": 1.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "figure.dpi": 400, "savefig.bbox": None,
})


def tidy(ax, grid=True):
    """Light grid behind, no top/right spines."""
    if grid:
        ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


FS_PANEL = 10                                         # panel labels, pt


def panel_label(ax, k, x=-0.05, y=1.0):
    ax.text(x, y, "(%s)" % k, transform=ax.transAxes, fontsize=FS_PANEL,
            ha="right", va="bottom")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"))
    print("wrote", os.path.join(HERE, stem + ".{pdf,png}"))
