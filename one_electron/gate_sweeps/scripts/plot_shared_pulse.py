"""Profile of the shared-pulse constraint: best average Fbar as a function of the
shared tau_ramp (tau_hold optimised at each ramp).

Usage: python scripts/plot_shared_pulse.py
"""
from __future__ import annotations
import os, sys
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.abspath(os.path.join(HERE, "..")), "results")
sys.path.insert(0, HERE)
from opt_shared_pulse import LEGS, TU_TO_NS, load, ridge_bilinear

data = {leg: load(leg) for leg in LEGS}
ramps = {leg: np.array(sorted(d)) for leg, d in data.items()}

rg = np.linspace(max(r[0] for r in ramps.values()), min(r[-1] for r in ramps.values()), 701)
hg = np.linspace(4.2, 169.0, 2001)
R, H = np.meshgrid(rg, hg, indexing="ij")
F = {leg: ridge_bilinear(data[leg], ramps[leg], R, H) for leg in LEGS}
A, B = F["w0w1_sqiswap_mf"], F["w2w3_sqiswap_mf"]
avg = np.where(np.isfinite(A) & np.isfinite(B), 0.5 * (A + B), -np.inf)

j = np.argmax(avg, axis=1)                      # best shared hold at each shared ramp
i_all = np.arange(len(rg))
best_avg, best_hold = avg[i_all, j], hg[j]
fa, fb = A[i_all, j], B[i_all, j]
k = int(np.argmax(best_avg))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
                              gridspec_kw=dict(height_ratios=[2, 1]))
ax.plot(rg * TU_TO_NS, fa, "C0-", label=r"w0w1 $\bar F_4$")
ax.plot(rg * TU_TO_NS, fb, "C1-", label=r"w2w3 $\bar F_4$")
ax.plot(rg * TU_TO_NS, best_avg, "k--", lw=1.8, label="average (shared pulse)")
ax.axhline(0.996697, color="0.6", lw=1, ls="-.",
           label=r"average, same $t_{\rm gate}$ only (0.99670)")
ax.plot([rg[k] * TU_TO_NS], [best_avg[k]], "k*", ms=13,
        label=rf"optimum $\bar F$={best_avg[k]:.4f}")
ax.set_ylabel(r"$\bar F_4$"); ax.grid(True, alpha=0.4)
ax.set_ylim(0.985, 0.9985)
ax.legend(fontsize="small", loc="lower right")
ax.set_title(r"Shared pulse: both legs at the same $(\tau_{\rm ramp}, \tau_{\rm hold})$")

ax2.plot(rg * TU_TO_NS, best_hold * TU_TO_NS, "k-", label=r"optimal shared $\tau_{\rm hold}$")
for leg, c in zip(LEGS, ["C0", "C1"]):
    pk = np.array([data[leg][tr][0][np.argmax(data[leg][tr][1])] for tr in ramps[leg]])
    ax2.plot(ramps[leg] * TU_TO_NS, pk * TU_TO_NS, c + "o--", ms=4,
             label=rf"{LEGS[leg]} own ridge")
ax2.axvline(rg[k] * TU_TO_NS, color="0.5", lw=1, ls=":")
ax2.set_xlabel(r"shared $\tau_{\rm ramp}$ [ns]")
ax2.set_ylabel(r"$\tau_{\rm hold}$ [ns]")
ax2.grid(True, alpha=0.4); ax2.legend(fontsize="small")
fig.tight_layout()
out = os.path.join(RESULTS, "shared_pulse_profile")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=160)

print(f"optimum: tau_ramp={rg[k]:.2f} tu ({rg[k]*TU_TO_NS:.4f} ns)  "
      f"tau_hold={best_hold[k]:.2f} tu ({best_hold[k]*TU_TO_NS:.4f} ns)  "
      f"t_gate={best_hold[k]+2*rg[k]:.2f} tu ({(best_hold[k]+2*rg[k])*TU_TO_NS:.4f} ns)")
print(f"  w0w1={fa[k]:.6f}  w2w3={fb[k]:.6f}  avg={best_avg[k]:.6f}  product={fa[k]*fb[k]:.6f}")
print("\n  shared-ramp profile (hold optimised at each ramp):")
for r in (18.3, 26.6, 34.1, 38.8, 45.0, 50.0, 56.5, 57.9, 65.0, 70.0, 82.3):
    m = int(np.argmin(abs(rg - r)))
    print(f"    ramp={rg[m]:6.2f}  hold={best_hold[m]:6.2f}  t_gate={best_hold[m]+2*rg[m]:6.2f}  "
          f"w0w1={fa[m]:.6f} w2w3={fb[m]:.6f} avg={best_avg[m]:.6f}")
print(f"wrote {out}.png / .pdf")
