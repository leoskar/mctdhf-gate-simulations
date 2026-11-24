"""Stronger parallel-route constraint: both legs share the SAME (tau_ramp, tau_hold).

The two legs were never sampled at the same tau_ramp, so this needs a 2D
interpolation of Fbar(tau_ramp, tau_hold) per leg. Two independent schemes are
built and compared:

  (a) "sheared bilinear": at each sampled tau_ramp, a piecewise-linear F(tau_hold)
      from the merged coarse+fine hold scan; then piecewise-linear across tau_ramp.
  (b) scipy LinearNDInterpolator on the scattered points (axes rescaled to unit
      span so the Delaunay triangulation is not degenerate).

Usage: python scripts/opt_shared_pulse.py
"""
from __future__ import annotations
import csv, os
import numpy as np
from scipy.interpolate import LinearNDInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.abspath(os.path.join(HERE, "..")), "results")
TU_TO_NS = 15.9155e-3
LEGS = {"w0w1_sqiswap_mf": "w0w1", "w2w3_sqiswap_mf": "w2w3"}


def load(leg):
    """{tau_ramp: (holds sorted, Fbars)} merged over coarse+fine."""
    per_ramp = {}
    for fname in ("coarse_sweep_fidelities.csv", "fine_sweep_fidelities.csv"):
        for r in csv.DictReader(open(os.path.join(RESULTS, leg, fname))):
            if int(r["finite"]) != 1:
                continue
            per_ramp.setdefault(float(r["tau_ramp"]), {})[float(r["tau_hold"])] = float(r["Fbar"])
    out = {}
    for tr, d in per_ramp.items():
        hs = np.array(sorted(d))
        out[tr] = (hs, np.array([d[h] for h in hs]))
    return out


def sheared_bilinear(data, ramps, R, H):
    """F on meshgrid (R,H); NaN wherever the hold falls outside a bracketing ramp's scan."""
    cols = []
    for tr in ramps:
        hs, fs = data[tr]
        col = np.interp(H, hs, fs)
        cols.append(np.where((H >= hs[0]) & (H <= hs[-1]), col, np.nan))
    cols = np.array(cols)  # (n_ramp, ...)
    idx = np.clip(np.searchsorted(ramps, R), 1, len(ramps) - 1)
    r0, r1 = ramps[idx - 1], ramps[idx]
    w = (R - r0) / (r1 - r0)
    lo = np.take_along_axis(cols, (idx - 1)[None], 0)[0]
    hi = np.take_along_axis(cols, idx[None], 0)[0]
    F = lo + w * (hi - lo)
    return np.where((R >= ramps[0]) & (R <= ramps[-1]), F, np.nan)


def peak_holds(data, ramps):
    """tau_hold of the best Fbar at each sampled tau_ramp (the ridge)."""
    return np.array([data[tr][0][np.argmax(data[tr][1])] for tr in ramps])


def ridge_bilinear(data, ramps, R, H):
    """As sheared_bilinear, but the hold coordinate is measured relative to the
    per-ramp fidelity ridge, so interpolation runs ALONG the ridge instead of
    across it."""
    pk = peak_holds(data, ramps)
    ridge = np.interp(R, ramps, pk)
    delta = H - ridge
    cols = []
    for tr, p in zip(ramps, pk):
        hs, fs = data[tr]
        h = p + delta
        col = np.interp(h, hs, fs)
        cols.append(np.where((h >= hs[0]) & (h <= hs[-1]), col, np.nan))
    cols = np.array(cols)
    idx = np.clip(np.searchsorted(ramps, R), 1, len(ramps) - 1)
    r0, r1 = ramps[idx - 1], ramps[idx]
    w = (R - r0) / (r1 - r0)
    lo = np.take_along_axis(cols, (idx - 1)[None], 0)[0]
    hi = np.take_along_axis(cols, idx[None], 0)[0]
    return np.where((R >= ramps[0]) & (R <= ramps[-1]), lo + w * (hi - lo), np.nan)


def loo_check(data, ramps, kind):
    """Leave-one-out over interior ramp nodes: predict a held-out ramp's whole
    hold scan from its neighbours, report the error at that ramp's own optimum."""
    errs = []
    for k in range(1, len(ramps) - 1):
        keep = np.delete(ramps, k)
        sub = {tr: data[tr] for tr in keep}
        hs, fs = data[ramps[k]]
        R = np.full(hs.shape, ramps[k])
        pred = (ridge_bilinear(sub, keep, R, hs) if kind == "ridge"
                else sheared_bilinear(sub, keep, R, hs))
        ok = np.isfinite(pred)
        if not ok.any():
            continue
        # error at the true optimum of the held-out scan, and peak-vs-peak error
        j = int(np.argmax(fs))
        e_true = pred[j] - fs[j] if ok[j] else np.nan
        e_peak = np.nanmax(pred[ok]) - fs.max()
        errs.append((ramps[k], e_true, e_peak))
    return errs


def scattered(data):
    pts, vals = [], []
    for tr, (hs, fs) in data.items():
        for h, f in zip(hs, fs):
            pts.append((tr, h)); vals.append(f)
    pts = np.array(pts); vals = np.array(vals)
    scale = np.ptp(pts, axis=0)
    return LinearNDInterpolator(pts / scale, vals), scale


def main():
    data = {leg: load(leg) for leg in LEGS}
    ramps = {leg: np.array(sorted(d)) for leg, d in data.items()}

    r_lo = max(r[0] for r in ramps.values())
    r_hi = min(r[-1] for r in ramps.values())
    h_lo = max(min(hs[0] for hs, _ in d.values()) for d in data.values())
    h_hi = min(max(hs[-1] for hs, _ in d.values()) for d in data.values())
    print(f"shared domain: tau_ramp [{r_lo:g}, {r_hi:g}]  tau_hold [{h_lo:g}, {h_hi:g}] tu")

    rg = np.linspace(r_lo, r_hi, 1401)
    hg = np.linspace(h_lo, h_hi, 2001)
    R, H = np.meshgrid(rg, hg, indexing="ij")

    for leg in LEGS:
        for kind in ("ridge", "flat"):
            errs = loo_check(data[leg], ramps[leg], kind)
            emax = max(abs(e) for _, _, e in errs)
            print(f"  LOO {LEGS[leg]:5s} {kind:5s}: peak-vs-peak err " +
                  " ".join(f"{r:g}:{e:+.1e}" for r, _, e in errs) + f"   |max|={emax:.1e}")

    for label, method in (("ridge-aligned", "c"), ("sheared bilinear", "a"), ("LinearND", "b")):
        Fs = {}
        for leg in LEGS:
            if method == "c":
                Fs[leg] = ridge_bilinear(data[leg], ramps[leg], R, H)
            elif method == "a":
                Fs[leg] = sheared_bilinear(data[leg], ramps[leg], R, H)
            else:
                itp, scale = scattered(data[leg])
                Fs[leg] = itp(np.stack([R / scale[0], H / scale[1]], -1))
        A, B = Fs["w0w1_sqiswap_mf"], Fs["w2w3_sqiswap_mf"]
        avg = 0.5 * (A + B)
        avg = np.where(np.isfinite(A) & np.isfinite(B), avg, -np.inf)
        i, j = np.unravel_index(np.argmax(avg), avg.shape)
        tr, th = rg[i], hg[j]
        T = th + 2 * tr
        print(f"\n[{label}]  tau_ramp={tr:.2f} tu ({tr*TU_TO_NS:.4f} ns)  "
              f"tau_hold={th:.2f} tu ({th*TU_TO_NS:.4f} ns)  t_gate={T:.2f} tu ({T*TU_TO_NS:.4f} ns)")
        print(f"    w0w1 Fbar={A[i,j]:.6f}   w2w3 Fbar={B[i,j]:.6f}   avg={avg[i,j]:.6f}   "
              f"product={A[i,j]*B[i,j]:.6f}")
        # what each leg could reach alone, same interpolant, same domain
        for leg, F in Fs.items():
            k = np.nanargmax(np.where(np.isfinite(F), F, -np.inf))
            ii, jj = np.unravel_index(k, F.shape)
            print(f"    unconstrained {LEGS[leg]}: Fbar={F[ii,jj]:.6f} at ramp={rg[ii]:.2f} hold={hg[jj]:.2f}")
        np.savez(os.path.join(RESULTS, f"shared_pulse_map_{method}.npz"),
                 rg=rg, hg=hg, A=A, B=B, tr=tr, th=th)


if __name__ == "__main__":
    main()
