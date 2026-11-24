"""Fig. 7 -- orbital convergence of the gate and its leak partners, 80 points.

    python figures/3e_convergence.py

Run on the default grid (FOURWELL_NUM_DVR unset): every number is read from
files.  Two stacked panels on a shared x = norb (orbitals per active well),
the rungs 6, 8, 10, 12, 13, 14, 16, ..., 26.  x label "Number of orbitals
(M_sigma)", the manuscript's symbol.

(a) Fbar_4 (solid, filled markers) and |11> retention (dashed, open
    markers) of the three theta = pi/2 gates (w0w1 tau_ramp 12.6 /
    tau_hold 98.3, w1w2 9.4 / 126.1, w2w3 8.0 / 131.6), MF layout, n_s 4
    (orbital_convergence ladderp3_<leg>_n<N>, norb 22 from gates_80pt
    mf80_leg_<leg>_n22).  Leg colours and markers as Fig. 6, keyed by a
    legend at the lower left of (a); a second legend at the upper right
    keys the line styles.
(b) w1w2 only.  Pair detunings (E - E_11)/Delta_80 of |30>, |03>, |40>,
    |04>, read in the orbitals the w1w2 gate has at the end of its ramp
    (t = tau_ramp = 9.4, the start of the hold):
    `orbital_convergence/ramp80.py` (results/ramp80/levels_n<N>.json, "frozen" at
    t = 9.4).  Each active well is FCI in its norb orbitals, in the
    gate-point Hamiltonian plus the Hartree field of the spectators and of
    the partner well's ground state (the "frozen" field of
    `orbital_convergence/levels_hartree80.py`).  Levels
    labelled by CHARACTER: at each orbital count, the sorted levels 3 and
    4 of each active well are assigned to the 80-orbital level they
    overlap most with (`label`, `label_weight` of levels_n<N>.json,
    "frozen", t = 9.4).  |30>, |03> are then the doubly excited (second
    centre-of-mass quantum) state at every orbital count and |40>, |04>
    the internal excitation.  Up to 13 orbitals the two are exchanged
    with respect to the energy order; at 13 they are mixed (weights 0.60
    and 0.54).  Dotted: the same model in all 80 orbitals
    of the grid, labelled "M_sigma = 80" (NOT "exact": the
    model leaves out inter-well correlation, and only the orbital count
    differs from the curves).  Linear y axis from 0 (= |11>); the lines of
    values above the range run off the top edge, unmarked;
    the values are in the caption.

The ramp reruns are the w1w2 gates of (a): each reproduces its gate file's
|11> retention to <= 1e-4 (orbital_convergence check_n<N>.json, asserted).  The 12 -> 13
band (one strip through both panels and the gap between them, unlabelled)
marks the drop.  ``3e_convergence_numbers.md`` lists
everything drawn.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import (ScaledTranslation,  # noqa: E402
                                   blended_transform_factory)

assert "FOURWELL_NUM_DVR" not in os.environ, "run on the default grid"
RES80 = os.path.join(S.SIMS, "gates_80pt", "results")
RES_OC = os.path.join(S.SIMS, "orbital_convergence", "results")
RAMP = os.path.join(RES_OC, "ramp80")
LEG = "w1w2"
GATE = (9.4, 126.1)
T_READ = "9.4"                                  # end of the ramp
T_HOLD = ["9.4", "14.4", "19.4", "29.4", "39.4"]  # first 30 t.u. of the hold
NORB = [6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24, 26]
PAIRS = ("30", "03", "40", "04")
DELTA = json.load(open(os.path.join(RES80, "oppoints80.json")))[LEG]["final"]["Delta"]
assert abs(DELTA - 1.175341e-2) < 5e-9, DELTA
NUMBERS = []

# ── (a) ladders of the theta = pi/2 gates; stored values (orbital_convergence/ladder80.py)
GATES = {"w0w1": (12.6, 98.3), "w1w2": GATE, "w2w3": (8.0, 131.6)}
REC = {"w0w1": {6: (0.99691, 0.9957), 8: (0.99322, 0.9812),
                10: (0.97443, 0.9208), 12: (0.95502, 0.8960),
                13: (0.79988, 0.3910), 14: (0.68803, 0.1486),
                16: (0.68133, 0.1346), 18: (0.68093, 0.1337),
                20: (0.67476, 0.1205), 22: (0.66334, 0.0972),
                24: (0.66133, 0.0931), 26: (0.66090, 0.0923)},
       "w1w2": {6: (0.99892, 0.9996), 8: (0.99846, 0.9943),
                10: (0.98499, 0.9455), 12: (0.97308, 0.9097),
                13: (0.79529, 0.4214), 14: (0.75274, 0.3094),
                16: (0.73155, 0.2580), 18: (0.73085, 0.2564),
                20: (0.71704, 0.2230), 22: (0.70827, 0.2037),
                24: (0.70795, 0.2036), 26: (0.70648, 0.1998)},
       "w2w3": {6: (0.99879, 0.9989), 8: (0.99947, 0.9980),
                10: (0.98833, 0.9582), 12: (0.96933, 0.8938),
                13: (0.81834, 0.4767), 14: (0.77519, 0.3585),
                16: (0.76309, 0.3274), 18: (0.75068, 0.2956),
                20: (0.74002, 0.2693), 22: (0.72675, 0.2372),
                24: (0.72571, 0.2348), 26: (0.72468, 0.2323)}}


def gate_path(leg, n):
    if n == 22:
        return os.path.join(RES80, f"mf80_leg_{leg}_n22.npz")
    return os.path.join(RES_OC, f"ladderp3_{leg}_n{n}.npz")


RET = {leg: {} for leg in S.LEGS}
FB = {leg: {} for leg in S.LEGS}
for leg in S.LEGS:
    assert sorted(REC[leg]) == NORB
    for n in NORB:
        p = gate_path(leg, n)
        with np.load(p, allow_pickle=True) as d:
            assert str(d["status"]) == "ok" and str(d["leg"]) == leg, p
            assert (float(d["tau_ramp"]), float(d["tau_hold"])) == GATES[leg], p
            assert int(d["num_dvr"]) == 80 and int(d["norb"]) == n, p
            assert str(d["kind"]) == "full", p
            assert float(d["max_norm_dev"]) <= 1e-10, p
            F, P = float(d["F"]), float(np.asarray(d["retention"])[3])
        assert abs(F - REC[leg][n][0]) < 6e-6, (leg, n, F)
        assert abs(P - REC[leg][n][1]) < 6e-5, (leg, n, P)
        RET[leg][n] = P
        FB[leg][n] = F
        NUMBERS.append(("a", f"{leg}, norb {n}: |11> retention", P,
                        os.path.relpath(p, S.SIMS)))
# the norb-22 rungs are Fig. 6's gates
for leg, r in {"w0w1": 0.0972, "w1w2": 0.2037, "w2w3": 0.2372}.items():
    assert abs(RET[leg][22] - r) < 5e-5, leg

# ── (b) levels in the gate's own orbitals (orbital_convergence/ramp80.py) ────────────
# stored values at t = 9.4
REC_LEV = {6: (10.29, 10.48, 269.4, 269.8), 8: (3.15, 3.43, 55.39, 55.69),
           10: (0.979, 1.190, 9.069, 9.304), 12: (1.150, 1.363, 4.478, 4.694),
           13: (1.336, 1.537, 1.477, 1.700), 14: (1.181, 1.385, 1.409, 1.629),
           16: (1.112, 1.319, 1.404, 1.625), 18: (1.103, 1.308, 1.403, 1.624),
           20: (1.092, 1.296, 1.403, 1.624), 22: (0.993, 1.200, 1.402, 1.620),
           24: (0.971, 1.179, 1.401, 1.620), 26: (0.953, 1.160, 1.401, 1.620)}
H80 = json.load(open(os.path.join(RES_OC, "levels_hartree80", "summary.json")))
assert H80["checks"]["passed"]
# the same model in all 80 orbitals (stored under "exact" in the json)
LIM = {k: H80["exact"]["frozen"][k] for k in PAIRS}
for k, v in zip(PAIRS, (0.8240, 1.0289, 1.4013, 1.6188)):
    assert abs(LIM[k] - v) < 5e-5, (k, LIM[k])
    # frozen field = self-consistent Hartree within 1e-3 Delta (levels_hartree80)
LEV = {k: [] for k in PAIRS}
DRIFT = {"all": 0.0, "13+": 0.0}
WEIGHT = {}                                 # norb -> smallest label weight
EXCHANGED = []                              # norb with levels 3, 4 exchanged


def by_character(fr):
    """Sorted pair key -> key of the 80-orbital level it overlaps most with."""
    out, w = {}, 1.0
    for well, fmt in (("1", "%d0"), ("2", "0%d")):
        lab = fr["levels"][well]["label"]
        assert lab[:3] == [0, 1, 2] and sorted(lab[3:5]) == [3, 4], lab
        for i in (3, 4):
            out[fmt % i] = fmt % lab[i]
            w = min(w, fr["levels"][well]["label_weight"][i])
    assert sorted(out.values()) == sorted(PAIRS), out
    return out, w


for n in NORB:
    chk = json.load(open(os.path.join(RAMP, f"check_n{n}.json")))
    assert chk["passed"] and chk["status"] == "ok" and chk["norb"] == n
    assert chk["retention_vs_gate_worst"] <= 1e-4, (n, chk)
    rec = json.load(open(os.path.join(RAMP, f"levels_n{n}.json")))
    assert rec["norb"] == n and abs(rec["Delta_80"] - DELTA) < 1e-15
    for k in PAIRS:
        assert abs(rec["exact"]["frozen"][k] - LIM[k]) < 1e-9, (n, k)
    assert sorted(rec["times"], key=float) == ["4.7"] + T_HOLD, rec["times"]
    r = rec["times"][T_READ]["frozen"]["pair_over_Delta"]
    char, WEIGHT[n] = by_character(rec["times"][T_READ]["frozen"])
    if char["30"] == "40":
        EXCHANGED.append(n)
    for t in T_HOLD:                        # same assignment through the hold
        assert by_character(rec["times"][t]["frozen"])[0] == char or n == 13, (n, t)
    for k, v_rec in zip(PAIRS, REC_LEV[n]):   # k: sorted key, as REC_LEV
        tol = 6e-3 if abs(v_rec) < 10 else 6e-2   # quoted to 3-4 figures
        assert abs(r[k] - v_rec) < tol, (n, k, r[k], v_rec)
        LEV[char[k]].append(r[k])
        NUMBERS.append(("b", f"norb {n}: |{char[k]}> by character (sorted "
                        f"|{k}>) (E-E11)/Delta at t = 9.4",
                        r[k], f"orbital_convergence/results/ramp80/levels_n{n}.json"))
        hold = [rec["times"][t]["frozen"]["pair_over_Delta"][k]
                for t in T_HOLD]
        for key in ("all", "13+")[:1 + (n >= 13)]:
            DRIFT[key] = max(DRIFT[key], max(hold) - min(hold))
LEV = {k: np.array(v) for k, v in LEV.items()}
assert all(len(v) == len(NORB) for v in LEV.values())
# exchanged up to 13 orbitals; a clear assignment everywhere but at 13
assert EXCHANGED == [6, 8, 10, 12, 13], EXCHANGED
assert all(WEIGHT[n] > 0.95 for n in NORB if n >= 14), WEIGHT
assert 0.5 < WEIGHT[13] < 0.61 and WEIGHT[12] > 0.97 and WEIGHT[10] > 0.97
for k in PAIRS:
    NUMBERS.append(("b", f"M_sigma = 80: |{k}> (E-E11)/Delta", LIM[k],
                    "orbital_convergence/results/levels_hartree80/summary.json exact.frozen"))
# caption: over the first 30 t.u. of the hold, <= 0.14 Delta (<= 0.09 from 13)
assert DRIFT["all"] < 0.145 and DRIFT["13+"] < 0.09, DRIFT
# the story the caption tells
i12, i13 = NORB.index(12), NORB.index(13)
# the doubly excited state falls monotonically; the internal one is flat
assert LEV["30"][i12] > 4.4 and LEV["30"][i13] < 1.5
assert all(np.all(np.diff(LEV[k]) < 0) for k in ("30", "03"))
assert all(abs(LEV[k][j] - LIM[k]) < 0.01 for k in ("40", "04")
           for j in range(NORB.index(16), len(NORB)))

# ═════════════════════════════════════════════════════════════════ drawing ══
C_30, C_40 = "C4", "C5"            # mirror partners share a colour
XL = (5.3, 26.7)
DASH = (0, (3, 1.5))
DOT = (0, (1, 1.5))
MS = 3.0


def band(fig, a, b):
    """The 12-13 strip, continuous from (b)'s bottom spine to (a)'s top.

    Drawn on the figure, behind transparent axes, so it also crosses the
    gap between the panels.  x in data units (shared axis), y in figure
    units."""
    for ax in (a, b):
        ax.patch.set_alpha(0)
    tr = blended_transform_factory(b.transData, fig.transFigure)
    y0, y1 = b.get_position().y0, a.get_position().y1
    fig.patches.append(Rectangle((12, y0), 1, y1 - y0, transform=tr,
                                 color=S.WINDOW, lw=0, zorder=-1))


fig = plt.figure(figsize=(3.37, 4.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.28,
                      left=0.155, right=0.975, top=0.935, bottom=0.11)
a = fig.add_subplot(gs[0])
b = fig.add_subplot(gs[1], sharex=a)

# (a)
for leg in S.LEGS:
    a.plot(NORB, [FB[leg][n] for n in NORB], color=S.COL[leg],
           marker=S.MRK[leg], ms=MS, lw=0.9, zorder=4)
    a.plot(NORB, [RET[leg][n] for n in NORB], color=S.COL[leg],
           marker=S.MRK[leg], ms=MS, mfc="white", mew=0.7, lw=0.9,
           ls=DASH, zorder=4)
a.set_ylim(0, 1.05)
a.set_yticks([0, 0.5, 1])
a.set_ylabel(r"$\bar F_4$, $\ket{11}$ retention", labelpad=4)

# (b): linear from 0 (= |11>); off-scale points unmarked, their lines clipped
sty = {"30": ("o", C_30, C_30, "-"), "03": ("o", C_30, "white", DASH),
       "40": ("s", C_40, C_40, "-"), "04": ("s", C_40, "white", DASH)}
YL = (0, 5.5)
OFF = {}                                   # norb -> off-scale series
for k, (mk, ec, fc, ls) in sty.items():
    assert LEV[k].min() > 0, k             # no partner below |11>
    top = 2 * (k in ("30", "03"))          # the doubly excited state on top
    b.plot(NORB, LEV[k], color=ec, lw=0.8, ls=ls, zorder=2 + top)
    on = LEV[k] < YL[1]
    b.plot(np.array(NORB)[on], LEV[k][on], mk, ms=MS, mec=ec, mfc=fc,
           mew=0.7, zorder=3 + top)
    for n in np.array(NORB)[~on]:
        OFF.setdefault(int(n), []).append(k)
    b.axhline(LIM[k], color=ec, lw=0.6, ls=DOT, zorder=1)
assert OFF == {6: ["30", "03", "40", "04"], 8: ["30", "03"],
               10: ["30", "03"]}, OFF       # the caption lists these
b.set_ylim(*YL)
b.set_yticks(range(0, 6))
b.set_ylabel(r"$(E_{mn}-E_{11})/\Delta$", labelpad=4)
# column-major fill: |30> |03> / |40> |04> on rows, M_sigma = 80 below |30>
H = {k: Line2D([], [], color=sty[k][1], marker=sty[k][0], ms=MS,
               mfc=sty[k][2], mew=0.7, lw=0.8, ls=sty[k][3],
               label=r"$\ket{%s}$" % k) for k in sty}
H["lim"] = Line2D([], [], color=S.INK2, lw=0.6, ls=DOT,
                  label=r"$M_\sigma=80$")
b.legend(handles=[H[k] for k in ("30", "40", "lim", "03", "04")],
         loc="upper right", ncol=2, handlelength=1.8,
         handletextpad=0.4, columnspacing=1.0, labelspacing=0.3)
# (b) is w1w2 only, while (a) carries all three gates: a heading over the
# panel
b.text(0.5, 1.03, S.LEG_TEX[LEG] + " gate", transform=b.transAxes,
       fontsize=S.mpl.rcParams["legend.fontsize"], ha="center", va="bottom")

for ax in (a, b):
    S.tidy(ax)
    ax.set_xlim(*XL)
    ax.set_xticks(range(6, 27, 2))
a.tick_params(labelbottom=False)
b.set_xlabel(r"Number of orbitals ($M_\sigma$)", labelpad=1)
# panel labels placed as Fig. 6's relative to the y labels:
# the left edge FIG6_GAP_IN left of the leftmost y label (Fig. 6's (a),
# measured at labelpad 4), and the same height above the axes in inches
# (0.05 of Fig. 6's 1.36-in panels), since the panel heights differ here.
# A fixed axes fraction, as in Fig. 6, would sit too far out: the tick
# labels here are narrower, so the y labels are closer to the axes.
FIG6_LIFT_IN = 0.05 * (0.915 - 0.095) * 4.0 / (2 + 0.42)
FIG6_GAP_IN = 0.031
fig.draw_without_rendering()
x_in = min(ax.yaxis.label.get_window_extent().x0
           for ax in (a, b)) / fig.dpi - FIG6_GAP_IN
for ax, k in zip((a, b), "ab"):
    tr = (blended_transform_factory(fig.dpi_scale_trans, ax.transAxes)
          + ScaledTranslation(0, FIG6_LIFT_IN, fig.dpi_scale_trans))
    ax.text(x_in, 1.0, "(%s)" % k, transform=tr, fontsize=S.FS_PANEL,
            ha="left", va="bottom")

# the gate legend keys (a) only, so it sits in (a), clear of the data.
# Both legends keep matplotlib's default frame
leg_gate = a.legend(handles=[Line2D([], [], color=S.COL[l], lw=0.9,
                         marker=S.MRK[l], ms=MS, label=S.LEG_TEX[l])
                         for l in S.LEGS],
         loc="lower left", ncol=1, handlelength=1.8,
         handletextpad=0.4, labelspacing=0.3)
a.add_artist(leg_gate)
a.legend(handles=[Line2D([], [], color=S.INK2, lw=0.9, marker="o", ms=MS,
                         label=r"$\bar F_4$"),
                  Line2D([], [], color=S.INK2, lw=0.9, marker="o", ms=MS,
                         mfc="white", mew=0.7, ls=DASH,
                         label=r"$\ket{11}$ retention")],
         loc="upper right", bbox_to_anchor=(1.02, 1.04), ncol=2,
         handlelength=1.5, handletextpad=0.35, columnspacing=0.8,
         labelspacing=0.3, borderpad=0.3)
band(fig, a, b)                     # after the axis limits are final
S.save(fig, "3e_convergence")

with open(os.path.join(HERE, "3e_convergence_numbers.md"), "w") as fh:
    fh.write("| panel | quantity | value | source |\n|---|---|---|---|\n")
    for p, w, v, s in NUMBERS:
        fh.write(f"| {p} | {w} | {v:.4f} | {s} |\n")
    fh.write(f"\nDelta_80 (w1w2) = {DELTA:.6e} "
             "(gates_80pt/results/oppoints80.json w1w2.final.Delta)\n")
    fh.write("(a) pulses: each leg's theta = pi/2 gate (tau_ramp, tau_hold) = "
             + ", ".join(f"{l} {GATES[l]}" for l in S.LEGS)
             + "; MF layout, n_s 4.\n")
    fh.write("(b) levels at t = tau_ramp = 9.4 (start of the hold); largest "
             f"change of any plotted level over t = 9.4 ... 39.4: "
             f"{DRIFT['all']:.4f} Delta (norb >= 13: {DRIFT['13+']:.4f}).\n")
    fh.write("(b) off the plotted range (0-5.5 Delta), lines clipped, unmarked: "
             + "; ".join(f"norb {n}: " + ", ".join(
                 f"|{k}> {LEV[k][NORB.index(n)]:.1f}" for k in ks)
                 for n, ks in OFF.items()) + " Delta.\n")
    fh.write("(b) labels by character: largest overlap with the 80-orbital "
             "levels; exchanged with respect to the energy order at norb "
             + ", ".join(map(str, EXCHANGED)) + ". Smallest label weight: "
             + ", ".join(f"norb {n}: {WEIGHT[n]:.3f}" for n in NORB) + ".\n")
