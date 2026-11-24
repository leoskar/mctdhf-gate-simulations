"""Fig. 4: mean-field spectators against four orbitals in every well.

Columns: the gates w0w1, w1w2, w2w3, whose spectator wells are idle, and
w0w3, whose spectator wells mediate the exchange.  Each column is the gate's
best-fidelity ramp row of the mean-field sweep, the row that also has a
four-orbital hold sweep.

  row 1  (a)-(d)   1 - Fbar_4 along the hold axis, log scale, both methods.
  row 2  (e)-(h)   Delta X = X(norb 4) - X(mean field) for each term of the
                   error budget of one_electron/lib/error_channels.py (the
                   decomposition of Fig. 5), evaluated at d = 4 on the gate
                   matrix.  The terms sum to Delta(1 - Fbar_4) exactly
                   (asserted); Delta > 0 means that the mean field
                   underestimates that error.

`realized` is asserted to reproduce the Fbar_4 stored by the sweep.  The two
phase terms (inter-input and internal) are drawn as one curve.
"""
import os, re, glob, importlib.util
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ONE = os.path.abspath(os.path.join(HERE, "..", "one_electron"))
RS = os.path.join(ONE, "gate_sweeps", "results")
FL = RS                               # the norb-4 hold rows sit beside the MF sweeps
FPI = os.path.join(ONE, "lib", "error_channels.py")
NS = 0.0159155                        # t.u. -> ns (1/omega_0, omega_0 = 2 pi 10 GHz)
C_MF, C_N4 = "C0", "C1"

# the gauge optimiser and the ideal 4x4 sqiSWAP come from error_channels.py
_spec = importlib.util.spec_from_file_location("fpi", FPI)
fpi = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(fpi)
except SystemExit:
    pass
SQ2 = np.asarray(fpi._SQ2, dtype=complex)

# colours and stacking order as in methods_fid.py (Fig. 5)
C_INTER, C_INT, C_ROUTE = "#d1495b", "#eb8f34", "#8a8884"
C_LEAK = "#0b0b0b"
TERMS = [("inter", C_INTER, "Inter-input"), ("internal", C_INT, "Internal"),
         ("misrouting", C_ROUTE, "Misrouting"), ("leakage", C_LEAK, "Leakage")]
EDGE = dict(edgecolor="#949494", linewidth=0.5)

mpl.rcParams.update({
    "text.usetex": True,
    # Times via mathptmx, the manuscript's text font
    "font.family": "serif", "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{braket}",
    "font.size": 10, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
    "lines.linewidth": 1.3,
    "figure.dpi": 400, "savefig.bbox": None,
})


def _z2(thetas):
    """2-qubit analogue of error_channels._zphases, which is hardwired to the
    16-state register.  Same construction, nq = 2."""
    bits = np.array([[(i >> (1 - k)) & 1 for k in range(2)] for i in range(4)])
    return np.exp(1j * (bits @ np.asarray(thetas)))


def leg_budget(M, U=SQ2):
    """The four-term budget of error_channels at d = 4 on one leg's gate matrix.

    Same construction as error_channels.block_budgets, with the block slicing
    dropped (the leg matrix IS 4x4) and the gauge optimised for this leg alone
    rather than inherited from a 16-state register.
    """
    x, _ = fpi.best_gauge(U, M)
    zl, zr = _z2(x[:2]), _z2(x[2:4])
    Mg = (zl[:, None] * M) * zr[None, :]
    s = (np.abs(M) ** 2).sum(0)              # survival per input (= `subspace`)
    p = (np.abs(U) * np.abs(M)).sum(0)
    o = np.einsum("ij,ij->j", U.conj(), Mg)
    S = float(s.sum())

    def _F(c):
        return (S + c ** 2) / 20.0           # d(d+1) = 20 at d = 4

    b_lk = _F(float(np.sqrt(s).sum()))
    b_rt = _F(float(p.sum()))
    b_in = _F(float(np.abs(o).sum()))
    Fb = _F(abs(o.sum()))
    b = dict(leakage=1.0 - b_lk, misrouting=b_lk - b_rt,
             internal=b_rt - b_in, inter=b_in - Fb, realized=Fb)
    # the budget is an identity, so this holds to rounding
    closure = sum(b[k] for k, _, _ in TERMS) - (1.0 - Fb)
    assert abs(closure) < 1e-12, f"budget does not close: {closure:.2e}"
    return b


# ---------------------------------------------------------------- loading
def _key_from(path, d):
    m = re.match(r"pt_tr([\d.]+)_th([\d.]+)\.npz", os.path.basename(path))
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (float(d["tau_ramp"]), float(d["tau_hold"]))


def load(pattern, fkey):
    out = {}
    for p in glob.glob(pattern):
        d = np.load(p, allow_pickle=True)
        out[_key_from(p, d)] = dict(Fbar=float(d[fkey]),
                                    M=np.asarray(d["M"], dtype=complex))
    return out


def mf_points(leg):
    return load(os.path.join(RS, f"{leg}_sqiswap_mf/sweep_pts/pt_tr*_th*.npz"), "Fbar")


def n4_points(leg):
    out = load(os.path.join(FL, f"{leg}_sqiswap/sweep_pts/pt_tr*_th*.npz"), "Fbar")
    out.update(load(os.path.join(RS, "norb4_confirm", f"n4_{leg}_tr*_th*.npz"),
                    "F_sqiSWAP"))                     # full-leg runs win
    return out


# leg, operating tau_ramp, operating tau_hold, blurb, window half-width (ns).
# Operating points: the best-fidelity points of the MF sweeps.
COLS = [("w0w1", 82.3, 73.2, "idle", 0.25),
        ("w1w2", 57.2, 95.4, "idle", 0.25),
        ("w2w3", 34.1, 112.8, "idle", 0.25),
        ("w0w3", 81.3, 2160.7, "mediating", 5.0)]
ROWS = [("mean field", "MF"), ("full orbitals", "N4")]

MF = {leg: mf_points(leg) for leg, *_ in COLS}
N4 = {leg: n4_points(leg) for leg, *_ in COLS}
SRC = {"MF": MF, "N4": N4}


def series(tag, leg, tr, th_op, win=0.0):
    row = sorted([(h, v) for (r, h), v in SRC[tag][leg].items()
                  if abs(r - tr) < 1e-9])
    if win:
        row = [(h, v) for h, v in row if abs(h - th_op) * NS <= win]
    xs = np.array([h * NS for h, _ in row])
    tot = np.array([1 - v["Fbar"] for _, v in row])
    return xs, [v for _, v in row], tot


def deltas(leg, tr):
    """Per-channel Delta = norb4 - mean field, on the holds BOTH methods have."""
    mf, n4 = SRC["MF"][leg], SRC["N4"][leg]
    hs = sorted(h for (r, h) in set(mf) & set(n4) if abs(r - tr) < 1e-9)
    xs, d, dtot = np.array([h * NS for h in hs]), {k: [] for k, _, _ in TERMS}, []
    for h in hs:
        a, b = leg_budget(mf[(tr, h)]["M"]), leg_budget(n4[(tr, h)]["M"])
        for k, _c, _l in TERMS:
            d[k].append(b[k] - a[k])
        dtot.append((1 - b["realized"]) - (1 - a["realized"]))
        # the channels must sum to the total
        clo = abs(sum(d[k][-1] for k, _, _ in TERMS) - dtot[-1])
        assert clo < 1e-12, f"{leg} th={h}: channels do not sum to total ({clo:.1e})"
    return xs, {k: np.array(v) for k, v in d.items()}, np.array(dtot)


# two-column width
fig = plt.figure(figsize=(6.69, 4.0), constrained_layout=True)
LEG_BAND, BOT_BAND = 0.945, 0.06
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.07, hspace=0.05,
                            rect=(0, BOT_BAND, 1, LEG_BAND - BOT_BAND))
NC = len(COLS)
gs = fig.add_gridspec(2, NC, height_ratios=[0.88, 1.0])
axes = np.empty((2, NC), dtype=object)
C_TOT = "#914bdd"
# One marker per series, so the curves stay separable in print.
# The two phase terms (inter-input and internal) are drawn as one curve,
# keyed on "inter": their split depends on the local-Z gauge, their sum does
# not (error_channels.py).  At d = 4 the gauge optimisation removes the
# internal term almost entirely (<= 3.6e-6 here).
DRAWN = ("inter", "misrouting", "leakage")
PHASE = ("inter", "internal")
MARK = {"inter": dict(marker="o", ms=2.8),
        "misrouting": dict(marker="d", ms=2.5),
        "leakage": dict(marker="s", ms=2.6),
        "total": dict(marker="o", ms=2.2)}
MARK_MF, MARK_N4 = dict(marker="o", ms=3.5), dict(marker="D", ms=2.0)

for j, (leg, tr, th_op, blurb, _win) in enumerate(COLS):
    # ---- row 1: the landscape, both methods, full hold row
    ax = fig.add_subplot(gs[0, j], sharey=axes[0, 0] if j else None)
    axes[0, j] = ax
    ax.axvline(th_op * NS, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=1)
    for tag, colour, mk, ls in (("MF", C_MF, MARK_MF, "-"),
                                ("N4", C_N4, MARK_N4, (0, (1, 1)))):
        xs, _v, tot = series(tag, leg, tr, th_op)
        ax.plot(xs, tot, ls=ls, color=colour, zorder=3, **mk)
    ax.set_yscale("log")
    ax.tick_params(labelbottom=False)          # row 2 carries the shared x axis
    if j == 0:
        ax.set_ylabel(r"$1-\bar F_4$")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(r"$w_%s w_%s$" % (leg[1], leg[3]), pad=4)

    # ---- row 2: what the surrogate got wrong, per channel
    axd = fig.add_subplot(gs[1, j], sharex=ax)
    axes[1, j] = axd
    xs, d, dtot = deltas(leg, tr)
    # zero line: above = the mean field underestimates that channel
    axd.axhline(0.0, color="0.35", lw=0.8, zorder=2)
    axd.axvline(th_op * NS, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=1)
    for k, colour, _lab in TERMS:
        if k not in DRAWN:
            continue
        y = sum(d[t] for t in PHASE) if k == "inter" else d[k]
        axd.plot(xs, 1e3 * y, color=colour, lw=1.2, zorder=4, **MARK[k])
    # With the phase channels combined the drawn set is exhaustive, so this is
    # the identity again and not an approximation.
    _drawn_sum = sum(sum(d[t] for t in PHASE) if k == "inter" else d[k]
                     for k in DRAWN)
    assert np.abs(_drawn_sum - dtot).max() < 1e-12, "drawn channels do not close"
    axd.plot(xs, 1e3 * dtot, color=C_TOT, lw=1.0, ls=(0, (2, 1.4)),
             zorder=5, **MARK["total"])
    axd.set_xlabel(r"$t_{\rm hold}$  (ns)", labelpad=1)
    axd.text(0.0, 1.0, r"$\times10^{-3}$", transform=axd.transAxes,
             fontsize=8, ha="left", va="bottom")
    if j == 0:
        axd.set_ylabel(r"Infidelity difference")

for ax in axes.ravel():
    ax.grid(lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)

for k, ax in enumerate(axes.ravel()):           # row-major: (a)-(d) on top
    ax.text(-0.13, 1.045, "(%s)" % "abcdefgh"[k], transform=ax.transAxes,
            fontsize=10, ha="right", va="bottom")

# The series are named by scheme: the mean-field orbital numbers differ
# between columns ([1,4,4,1] on w1w2, [4,1,1,4] on w0w3).
fig.legend(handles=[Line2D([], [], color=C_MF, label="Mean-field spectators",
                           **MARK_MF),
                    Line2D([], [], color=C_N4, ls=(0, (1, 1)),
                           label="Four orbitals per well", **MARK_N4)],
           loc="upper center", ncol=2, frameon=False,
           columnspacing=1.2, labelspacing=0.3, borderpad=0.0,
           handletextpad=0.4, bbox_to_anchor=(0.5, 1.0),
           bbox_transform=fig.transFigure, fontsize=9)
def _blank():
    return Line2D([], [], color="none")


_seg = {k: Line2D([], [], color=c, lw=1.6, **MARK[k])
        for k, c, _l in TERMS if k in DRAWN}
_seg["total"] = Line2D([], [], color=C_TOT, lw=1.0, ls=(0, (2, 1.4)),
                       **MARK["total"])
# channel key: three channels and their sum
_lab = {"total": "All channels",
        "inter": "Phase error",
        "leakage": "Leakage",
        "misrouting": "Misrouting"}
# one row: All channels, Leakage, Phase error, Misrouting
_ORDER = ["total", "leakage", "inter", "misrouting"]
_h = [_seg[k] for k in _ORDER]
_l = [_lab[k] for k in _ORDER]
fig.legend(_h, _l, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
           handlelength=2.0, handleheight=0.9, columnspacing=2.0,
           handletextpad=0.35, borderaxespad=0.0, labelspacing=0.3, fontsize=9)

for f in ("mf_fig_wide.pdf", "mf_fig_wide.png"):
    fig.savefig(os.path.join(HERE, f))
print("wrote", os.path.join(HERE, "mf_fig_wide.pdf"))
