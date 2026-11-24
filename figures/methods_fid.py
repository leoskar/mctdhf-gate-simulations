"""Which channel does the matrix-multiplication simplification get wrong? (Fig. 5)


Layout:

  * BOTH panels plot the INFIDELITY, so every bar starts from a common zero and
    the error segments can be compared directly across bars.
  * Panel (b) is grouped BY CIRCUIT, matching (a), so one shaded band can
    enclose the same circuit in both panels -- the figure-level banding
    error_channels uses.  Nesting is circuit > CALCULATION > spectator block,
    and each calculation's four-block group in (b) is centred on the SAME x as
    its single bar in (a), so the panels align column by column and the
    calculation is named by a tick in both.

Within each group of four, left to right: spectator block |00>, |01>, |10>, |11>,
labelled 0-3 by the binary value of the spectator pair.

The decomposition, the gauge optimisation and the colours come from
`one_electron/lib/error_channels.py`.
"""
import os, importlib.util
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ONE = os.path.abspath(os.path.join(HERE, "..", "one_electron"))
FPI = os.path.join(ONE, "lib", "error_channels.py")
DATA = os.path.join(ONE, "routed_gates", "results")   # prodmats_*.npz (impl_locals.py)

_spec = importlib.util.spec_from_file_location("fpi", FPI)
fpi = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(fpi)
except SystemExit:
    pass
decompose, block_budgets = fpi.decompose, fpi.block_budgets

C_INTER, C_INT = "#d1495b", "#eb8f34"
C_ROUTE, C_LEAK = "#8a8884", "#cfcecb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
EDGE = dict(edgecolor="#949494", linewidth=0.5)          # AIP: min 0.5 pt

TERMS = [("inter", C_INTER, "Collective"), ("internal", C_INT, "Internal"),
         ("misrouting", C_ROUTE, "Misrouting"), ("leakage", C_LEAK, "Leakage")]
ARMS = [("Serial", "prodmats_best_serial.npz"),
        ("Parallel", "prodmats_best_parallel.npz"),
        ("Synchronized", "prodmats_sync_parallel.npz")]
VARIANTS = [("M_prop", "Prop."), ("M_prod_16", r"16$\times$16"),
            ("M_prod_c4", r"4$\times$4")]
BLK = [(0, 0), (0, 1), (1, 0), (1, 1)]
# Spectator-block labels are two stacked bits, so they fit under each bar.

mpl.rcParams.update({
    # Times via mathptmx, the manuscript's text font
    "text.usetex": True, "font.family": "serif", "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{braket}",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 10,
    "figure.dpi": 400, "savefig.bbox": None,
})

fig = plt.figure(figsize=(6.69, 4.0))   # AIP: <=6.69 wide, <=8.25 deep
# The gap between the panels now carries (a)'s calculation ticks AND the
# circuit names; the legend sits above the figure, so `top` leaves room for it.
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.383,
                      top=0.935, bottom=0.131, left=0.078, right=0.997)
ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# Circuit c owns [c-0.5, c+0.5] in BOTH panels, so a band spans the same x
# range in each and the two groupings line up.
data = {lab: np.load(os.path.join(DATA, f), allow_pickle=True)
        for lab, f in ARMS}

# Shared geometry: calculation m of circuit c sits at XM(c, m) in BOTH panels.
MPITCH, WA = 0.315, 0.28


def XM(c, m):
    return c + (m - 1) * MPITCH


def stack(a, xc, w, vals):
    """Stacked bar carrying ONE outline around the whole stack.

    A 0.5 pt edge on each segment is a large fraction of a narrow bar, and a
    channel thinner than its own outline renders as a dark line rather than as
    area -- which visually inflates it.  The segments are drawn flat and the
    outline is added once; the colour changes still separate the channels.
    """
    bot = 0.0
    for name, col, _ in TERMS:
        a.bar(xc, vals[name], w, bottom=bot, color=col, zorder=3,
              edgecolor="none", linewidth=0)
        bot += vals[name]
    a.add_patch(plt.Rectangle((xc - w / 2, 0.0), w, bot, facecolor="none",
                              zorder=4, **EDGE))
    return bot


# ── (a) whole register, one bar per calculation ──────────────────────────────
for c, (lab, _f) in enumerate(ARMS):
    for m, (key, tag) in enumerate(VARIANTS):
        b = decompose(np.asarray(data[lab][key]))["budget"]
        xc = XM(c, m)
        bot = stack(ax, XM(c, m), WA, b)
        ax.text(xc, bot + 0.004, f"{1 - b['realized']:.3f}", ha="center",
                va="bottom", fontsize=10, color=INK)
ax.set_xticks([XM(c, m) for c in range(3) for m in range(3)])
ax.set_xticklabels([t for _ in range(3) for _, t in VARIANTS])
ax.set_ylabel(r"$1-\bar F_{16}$")
ax.set_ylim(0, 0.240)

# ── (b) same circuits, resolved by spectator block ───────────────────────────
# A gap between bars keeps the stacked labels readable.
WB, BW = 0.070, 0.058       # 10.3 pt pitch, 8.6 pt bar -> 1.7 pt gap
for c, (lab, _f) in enumerate(ARMS):
    for m, (key, _tag) in enumerate(VARIANTS):
        bb = block_budgets(np.asarray(data[lab][key]))
        for bi, blk in enumerate(BLK):
            xc = XM(c, m) + (bi - 1.5) * WB
            stack(axb, xc, BW, bb[blk])
# block tick row: a rotated ket under each bar
for c in range(3):
    for m in range(3):
        for bi, blk in enumerate(BLK):
            axb.annotate(r"$\ket{%d%d}$" % blk,
                         xy=(XM(c, m) + (bi - 1.5) * WB, 0.0),
                         xycoords=("data", "axes fraction"),
                         textcoords="offset points", xytext=(0, -2),
                         ha="center", va="top", rotation=90, fontsize=8,
                         color=INK, annotation_clip=False)
axb.set_xticks([XM(c, m) for c in range(3) for m in range(3)])
axb.set_xticklabels([t for _ in range(3) for _, t in VARIANTS])
axb.set_ylabel(r"$1-\bar F_4$ in block")
axb.set_ylim(0, 0.235)

for a in (ax, axb):
    a.set_xlim(-0.5, 2.5)
    a.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        a.spines[sp].set_zorder(6)
    a.tick_params(axis="x", length=0, pad=2)

# (b)'s ticks must clear the block digit row; set this AFTER the shared loop
# above, which resets pad for both axes.
axb.tick_params(axis="x", pad=20)

for a, lab in ((ax, "a"), (axb, "b")):
    a.text(-0.02, 1.02, r"(%s)" % lab, transform=a.transAxes,
           fontsize=10, va="bottom", ha="right")

# ── one band per circuit, spanning BOTH panels (error_channels's construction) ─
ax.patch.set_visible(False)
axb.patch.set_visible(False)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
y_lab = min(inv.transform(t.get_window_extent(rend))[0][1]
            for t in axb.get_xticklabels())
y_top = ax.get_position().y1
for c in (0, 2):
    (x0, _), (x1, _) = inv.transform(
        axb.transData.transform([(c - 0.5, 0.0), (c + 0.5, 0.0)]))
    fig.add_artist(plt.Rectangle((x0, y_lab - 0.0132), x1 - x0,
                                 y_top - y_lab + 0.0132, facecolor=GRID,
                                 alpha=0.35, edgecolor="none", zorder=0))
# circuit names sit in the gap below (a)'s tick labels
y_a = min(inv.transform(t.get_window_extent(rend))[0][1]
          for t in ax.get_xticklabels())
y_mid = y_a - 0.65 * (y_a - axb.get_position().y1)
for c, (lab, _f) in enumerate(ARMS):
    (xm, _) = inv.transform(axb.transData.transform([(c, 0.0)]))[0]
    fig.text(xm, y_mid, r"\textbf{%s}" % lab, ha="center", va="center",
             fontsize=10, color=INK)
# ── legend ───────────────────────────────────────────────────────────────────
# Above the panels, so it needs no measured anchor: `top` is set to leave room.
def _blank():
    return plt.Rectangle((0, 0), 1, 1, fc="none", ec="none")


_seg = {n: (plt.Rectangle((0, 0), 1, 1, fc=c, **EDGE), t)
        for n, c, t in TERMS}
_h, _l = [_blank()], ["Phase error:"]
for n in ("inter", "internal"):
    _h.append(_seg[n][0]); _l.append(_seg[n][1])
_h.append(_blank()); _l.append("Population error:")
for n in ("misrouting", "leakage"):
    _h.append(_seg[n][0]); _l.append(_seg[n][1])
fig.legend(_h, _l, frameon=False, ncol=6, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure,
           handlelength=0.9, handleheight=0.9, columnspacing=0.6,
           handletextpad=0.35, borderaxespad=0.0, labelspacing=0.3, fontsize=10)

for a in (ax, axb):
    for p in a.patches:
        p.set_snap(False)

for f in ("methods_fid.pdf", "methods_fid.png"):
    fig.savefig(os.path.join(HERE, f))
print("wrote", os.path.join(HERE, "methods_fid.pdf"))
