"""Fig. 6 -- the three-electron qubit has no gate point, 80-point grid.

    python figures/3e_gate.py

Run on the default grid (FOURWELL_NUM_DVR unset; the 3e numbers are read
from 80-point files).  Writes ``3e_gate.{pdf,png}``.

Type from style.py: Times (mathptmx, the manuscript's text font) at the
sizes of the paper's 1e figure (mf_fig_wide).  Width 3.37 in (AIP reprint column 246 pt
= 3.40 in), included at natural size.

Data: one hold scan per leg (orbital_convergence, norb 22, MF; tau_ramp 12.6 / 9.4 / 8.0
for w0w1 / w1w2 / w2w3).  Each scan's theta = pi/2 point is the gates_80pt gate
(asserted) and is drawn filled in both panels; every other point is open.

(a) 1 - F4 (average gate fidelity against sqrt(iSWAP), up to local Z;
    mf_lib.fbar) against t_hold in ns, styled as the 1e figure
    mf_fig_wide.py (t_hold = tau_hold * time unit 1/omega_0,
    dashed rule at the gate hold) so the two read side by side: at 1e the
    infidelity has a sharp minimum at the calibrated hold, at 3e it has none
    near the gate.  Linear axis.
(b) |11> retention against theta / (pi/2), dotted rule at theta = pi/2.

Every number is checked against stored values before plotting (asserts
below); ``3e_gate_numbers.md`` lists everything drawn.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import style as S  # noqa: E402  (sets rcParams)

assert "FOURWELL_NUM_DVR" not in os.environ, "run on the default grid"
sys.path[:0] = [S.SIMS]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerLine2D  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gate_timing import mf_lib  # noqa: E402
from system import fourwell as fw  # noqa: E402

RES80 = os.path.join(S.SIMS, "gates_80pt", "results")
RES_OC = os.path.join(S.SIMS, "orbital_convergence", "results")

NUMBERS = []                                 # (panel, quantity, value, source)


def rec(panel, what, val, src):
    NUMBERS.append((panel, what, val, src))


# ── time unit: the same 1/omega_0 as the one-electron simulations ──────────
# Both measure time in 1/omega_0, omega_0 = 2 pi * 10 GHz: here
# fourwell.TIME_UNIT_PS, for one electron geometry_config's hbar / (hbar omega)
# with the same TYPICAL_FREQ_GHZ (checked in its source, so that this script
# does not need pint).
assert fw.TYPICAL_FREQ_GHZ == 10.0
assert abs(fw.TIME_UNIT_PS - 1e3 / (2 * np.pi * 10.0)) < 1e-12
GEO_1E = os.path.join(HERE, "..", "one_electron", "lib", "geometry_config.py")
_src = open(GEO_1E).read()
for pat in (r"\nTYPICAL_FREQ_GHZ = 10\.0\n",
            r"\n_omega = 2 \* np\.pi \* TYPICAL_FREQ_GHZ \* _ureg\.GHz\n",
            r"\n_energy_unit = _ureg\.hbar \* _omega\n",
            r"\n_time_unit = _ureg\.hbar / _energy_unit\n"):
    assert re.search(pat, _src), ("1e time unit changed?", pat)
NS = fw.TIME_UNIT_PS * 1e-3                  # t.u. -> ns, 0.0159155

# ── the hold scans (orbital_convergence/holdscan80.py) ────────────────
# leg -> (summary file, tau_ramp, rows, gate hold = gates_80pt's leg at theta = pi/2)
SCAN = {"w0w1": ("holdscan_summary_w0w1.json", 12.6, 13, 98.3),
        "w1w2": ("holdscan_summary.json", 9.4, 14, 126.1),
        "w2w3": ("holdscan_summary_w2w3.json", 8.0, 13, 131.6)}
hold, igate = {}, {}         # hold[leg] columns: tau_hold, theta/(pi/2), P11, F4
for leg, (fn, tr, nrow, hold_g) in SCAN.items():
    HS = json.load(open(os.path.join(RES_OC, fn)))
    assert len(HS["rows"]) == nrow, (leg, len(HS["rows"]))
    h = []
    for r in HS["rows"]:
        assert r["status"] == "ok"
        with np.load(os.path.join(S.SIMS, r["file"]), allow_pickle=True) as d:
            assert str(d["status"]) == "ok" and str(d["leg"]) == leg
            assert float(d["tau_ramp"]) == tr
            assert float(d["tau_hold"]) == r["tau_hold"]
            assert json.loads(str(d["layout"]))["norb_active"] == 22
            th = float(d["theta"]) / (np.pi / 2)
            p11 = float(np.asarray(d["retention"])[3])
            F4 = float(d["F"])
            if r["tau_hold"] == hold_g:     # F4 is fbar(sqrt(iSWAP), M), as drawn
                F_re = mf_lib.fbar(mf_lib.M_REF, np.asarray(d["M"]))[0]
                assert abs(F_re - F4) < 1e-9, (leg, F_re, F4)
        assert abs(th - r["theta_rel"]) < 1e-12 \
            and abs(p11 - r["ret"][3]) < 1e-12 and abs(F4 - r["F"]) < 1e-12
        h.append((r["tau_hold"], th, p11, F4))
    hold[leg] = np.array(sorted(h, key=lambda t: t[1]))
    # theta rises with the hold, so the theta order is the hold order
    assert np.all(np.diff(hold[leg][:, 0]) > 0), leg
    igate[leg] = int(np.nonzero(hold[leg][:, 0] == hold_g)[0][0])
i126 = igate["w1w2"]
assert abs(hold["w1w2"][i126, 1] - 0.9998) < 5e-5 \
    and abs(hold["w1w2"][i126, 2] - 0.2037) < 5e-5
imin = int(np.argmin(hold["w1w2"][:, 2]))
assert hold["w1w2"][imin, 0] == 115.3 \
    and abs(hold["w1w2"][imin, 2] - 0.179) < 5e-4

# ── the MF legs at theta = pi/2 (gates_80pt) = each scan's filled point ───────
REC_LEG = {"w0w1": 0.0972, "w1w2": 0.2037, "w2w3": 0.2372}
legpt = {}
for leg in S.LEGS:
    p = os.path.join(RES80, f"mf80_leg_{leg}_n22.npz")
    with np.load(p, allow_pickle=True) as d:
        assert str(d["status"]) == "ok" and int(d["num_dvr"]) == 80
        assert int(d["norb"]) == 22 and str(d["kind"]) == "full"
        th = float(d["theta"]) / (np.pi / 2)
        p11 = float(np.asarray(d["retention"])[3])
    assert abs(th - 1) < 5e-4 and abs(p11 - REC_LEG[leg]) < 5e-5, (leg, th, p11)
    legpt[leg] = (th, p11)
# each scan's theta = pi/2 point is that leg (orbital_convergence repro checks,
# <= 1e-6 on F and P11)
for leg in S.LEGS:
    assert abs(hold[leg][igate[leg], 1] - legpt[leg][0]) < 1e-7, leg
    assert abs(hold[leg][igate[leg], 2] - legpt[leg][1]) < 1e-7, leg

# ── F4 against the stored values (to 1e-5) ──────────────────────
REC_F = {  # leg: (F4 at pi/2, F4 at the shortest hold, (tau_hold, F4) at min)
    "w0w1": (0.66334, 0.79556, (120.7, 0.63443)),
    "w1w2": (0.70827, 0.78775, (133.3, 0.70818)),
    "w2w3": (0.72675, 0.79645, (140.9, 0.72647))}
for leg, (f_gate, f_short, (h_min, f_min)) in REC_F.items():
    H = hold[leg]
    assert abs(H[igate[leg], 3] - f_gate) < 1e-5, (leg, H[igate[leg], 3])
    assert abs(H[0, 3] - f_short) < 1e-5, (leg, H[0, 3])
    k = int(np.argmin(H[:, 3]))
    assert H[k, 0] == h_min and abs(H[k, 3] - f_min) < 1e-5, (leg, H[k])
    # what panel (a) shows: lowest infidelity at the SHORTEST hold, and the
    # infidelity at the gate is above it (no minimum near theta = pi/2)
    assert int(np.argmax(H[:, 3])) == 0, leg
    assert k >= igate[leg], leg

# ── the gate holds (theta = pi/2) in ns ─────────────────────────────────────
REC_NS = {"w0w1": (98.3, 1.56449), "w1w2": (126.1, 2.00694),
          "w2w3": (131.6, 2.09448)}
for leg, (h, ns) in REC_NS.items():
    assert hold[leg][igate[leg], 0] == h
    assert abs(h * NS - ns) < 1e-5, (leg, h * NS)

for leg, (fn, tr, _n, hold_g) in SCAN.items():
    for th_, t_, p11, F4 in hold[leg]:
        rec("6a,b", f"hold scan {leg} tr{tr} th{th_:g}: t_hold (ns), "
            "theta/(pi/2), P11, 1-F4",
            (th_ * NS, t_, p11, 1 - F4),
            f"orbital_convergence/results/{fn}" + (" (= gates_80pt leg, filled)"
                                     if th_ == hold_g else ""))


# ═════════════════════════════════════════════════════════════════ figure ═══
MS, MS_GATE = 3.2, 4.2
DASH = (0, (3, 2))                    # mf_fig_wide's gate-hold rule


def draw_scan(ax, x, y, leg):
    """Open markers joined by a line; the theta = pi/2 point filled."""
    c, mk, k = S.COL[leg], S.MRK[leg], igate[leg]
    ax.plot(x, y, color=c, lw=0.8, marker=mk, ms=MS, mfc="white", mec=c,
            mew=0.6, zorder=3)
    ax.plot(x[k], y[k], ls="", marker=mk, ms=MS_GATE, mfc=c, mec=c, mew=0.6,
            zorder=4)


def draw_a(ax):
    xlim = (0.75, 3.1)
    for leg in S.LEGS:
        H, c = hold[leg], S.COL[leg]
        x = H[:, 0] * NS
        assert xlim[0] < x.min() and x.max() < xlim[1], leg
        ax.axvline(x[igate[leg]], color=c, lw=0.7, ls=DASH, zorder=1)
        draw_scan(ax, x, 1 - H[:, 3], leg)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.18, 0.40)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel(r"$t_{\rm hold}$  (ns)", labelpad=1)
    ax.set_ylabel(r"$1-\bar F_4$", labelpad=4)
    # full grid, as (b)
    S.tidy(ax)


def draw_b(ax):
    ax.axvline(1.0, color=S.INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    for leg in S.LEGS:
        draw_scan(ax, hold[leg][:, 1], hold[leg][:, 2], leg)
    xlim = (0.55, 1.45)           # the scans span 0.589 ... 1.405
    for leg in S.LEGS:            # widen only if a scan needs it
        xlim = (min(xlim[0], hold[leg][0, 1] - 0.03),
                max(xlim[1], hold[leg][-1, 1] + 0.03))
    ax.set_xlim(*xlim)
    # from zero (a population); 0.7 clears the highest point (0.63)
    assert max(hold[leg][:, 2].max() for leg in S.LEGS) < 0.68
    ax.set_ylim(0, 0.7)
    ax.set_xlabel(r"$\theta/(\pi/2)$", labelpad=1)
    ax.set_ylabel(r"$\ket{11}$ retention", labelpad=4)
    S.tidy(ax)


class _LoweredLine(HandlerLine2D):
    """Legend handle moved down by ``dy`` points.  matplotlib centres the
    handle on a box that the subscripted labels (w_0 w_1) push up, so the
    stock handle sits near the top of the letters; lowered, it runs through
    their middle, as in the 1e figure's legend."""

    def __init__(self, dy, **kw):
        super().__init__(**kw)
        self.dy = dy

    def create_artists(self, *args, **kw):
        arts = super().create_artists(*args, **kw)
        for a in arts:
            a.set_ydata(np.asarray(a.get_ydata(), float) - self.dy)
        return arts


def render(stem):
    # single column, two stacked panels, each with its own x axis
    fig = plt.figure(figsize=(3.37, 4.0))
    gs = fig.add_gridspec(2, 1, top=0.915, bottom=0.095, left=0.16,
                          right=0.97, hspace=0.42)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    draw_a(axa)
    draw_b(axb)
    for ax, k in ((axa, "a"), (axb, "b")):
        ax.text(-0.13, 1.05, "(%s)" % k, transform=ax.transAxes,
                fontsize=S.FS_PANEL, ha="right", va="bottom")
    # one key for both panels: each leg as drawn (open markers on a line).
    # The filled theta = pi/2 point is left to the caption:
    # in (b) it sits on the theta/(pi/2) = 1 rule, in (a) on its dashed rule.
    # handles exactly as the plotted series (draw_scan): same line width,
    # marker size and edge width
    hs = [Line2D([], [], color=S.COL[l], lw=0.8, marker=S.MRK[l], ms=MS,
                 mfc="white", mec=S.COL[l], mew=0.6, label=S.LEG_TEX[l])
          for l in S.LEGS]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False,
               handler_map={Line2D: _LoweredLine(1.0)},
               handlelength=1.5, handletextpad=0.3, columnspacing=1.2,
               borderpad=0.0, bbox_to_anchor=(0.56, 1.0),
               bbox_transform=fig.transFigure)
    S.save(fig, stem)
    plt.close(fig)


render("3e_gate")

# ── provenance table ─────────────────────────────────────────────────────────
with open(os.path.join(HERE, "3e_gate_numbers.md"), "w") as fh:
    fh.write("Fig. 6: hold scans on 80 points, norb 22, MF spectators.\n\n")
    fh.write(f"t_hold = tau_hold * {NS:.7f} ns (fourwell.TIME_UNIT_PS = "
             "1/omega_0, omega_0 = 2 pi 10 GHz; the one-electron unit).\n\n")
    fh.write("| panel | quantity | value | source |\n|---|---|---|---|\n")
    for p, w, v, s in NUMBERS:
        vs = "(" + ", ".join("%.5f" % x for x in v) + ")"
        fh.write(f"| {p} | {w} | {vs} | {s} |\n")
    fh.write("\nAsserted F4 (recorded values, 1e-5) and gate holds:\n")
    for leg, (f_gate, f_short, (h_min, f_min)) in REC_F.items():
        H, k = hold[leg], igate[leg]
        km = int(np.argmin(H[:, 3]))
        fh.write(f"- {leg}: theta = pi/2 at tau_hold {H[k, 0]:g} = "
                 f"{H[k, 0] * NS:.5f} ns, F4 {H[k, 3]:.5f}; shortest hold "
                 f"{H[0, 0]:g} F4 {H[0, 3]:.5f}; min F4 {H[km, 3]:.5f} at "
                 f"{H[km, 0]:g} (theta/(pi/2) {H[km, 1]:.4f})\n")
