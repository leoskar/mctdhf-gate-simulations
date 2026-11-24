"""Re-score the K=16 register matrices with phase-fixed anchors.

WHY.  `confirm_k16.py` scored against `run_route_sa.basis_states()` WITHOUT the
anchor-phase normalisation (`lib/common.py:phase_fix_anchors`,
applied by `run_arm_decomp.py`).  The 16 anchors are separately computed
eigenstates carrying independent arbitrary phases, and under Psi_i -> e^{i th_i}
Psi_i a gate matrix transforms as M -> diag(e^{-i th}) M diag(e^{+i th}).

`run_route_sa.fxc_4q`'s gauge is PRODUCT-FORM ONLY -- `_zphases` builds
exp(i * bits @ thetas) from 4+4 generators -- so the non-product part of a
16-phase diagonal cannot be absorbed and lands in F_bar16.  Two-qubit fidelities
ARE exactly invariant to the same phases, which is why the (0,3)/pair corner F4
extracted from each K=16 run reproduces its K=4 confirmation to 2e-6 while
F_bar16 reads far too low.

NO PROPAGATION IS REPEATED.  The full 16x16 M is stored in every
results/k16_confirm/k16_*.npz, so the fix is a similarity transform applied
post-hoc, then a re-score with the same fxc_4q + F_bar16 formula.

INDEX ORDER.  The stored M was already permuted from CONFIGS order to plain bit
order inside confirm_k16.score16 (`M = P @ M @ P.T`).  The phases come out of
phase_fix_anchors in CONFIGS order, so they are scattered through the same
permutation before use:  pb[bit[i]] = pc[i].

Writes k16_*_rescored.npz next to the originals; the originals are left intact.
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GS_DIR = os.path.abspath(os.path.join(HERE, ".."))
LIB = os.path.abspath(os.path.join(GS_DIR, "..", "lib"))   # one_electron/lib
sys.path.insert(0, LIB)
sys.path.insert(0, HERE)

# GATE_LEG only selects which resonance anchor qoct_gate_leg binds; the 16 basis
# states and their phases are leg-independent, so any valid value works here.
os.environ.setdefault("GATE_LEG", "w0w1")

import run_route_sa as B                      # basis_states, fxc_4q, CONFIGS
from common import phase_fix_anchors, anchor_phase_report

_S2 = 1.0 / np.sqrt(2.0)
SQI4 = np.array([[1, 0, 0, 0], [0, _S2, 1j * _S2, 0],
                 [0, 1j * _S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)
PAIRS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3), "w0w3": (0, 3)}


def target_sqiswap(pair):
    """sqiSWAP on `pair` (x) identity elsewhere, plain bit order."""
    a, b = pair
    U = np.zeros((16, 16), complex)
    for i in range(16):
        q = [(i >> (3 - k)) & 1 for k in range(4)]
        for li in range(4):
            qa, qb = (li >> 1) & 1, li & 1
            amp = SQI4[2 * q[a] + q[b], 2 * qa + qb]
            if amp != 0:
                o = list(q)
                o[a], o[b] = qa, qb
                U[sum(bit << (3 - k) for k, bit in enumerate(o)), i] = amp
    return U


def fbar16(U, M):
    fxc = B.fxc_4q(U, M)
    d = 16
    return (np.vdot(M, M).real
            + fxc * np.vdot(U, U).real * np.vdot(M, M).real) / (d * (d + 1)), fxc


def fbar4_localZ(U, M, n=2, restarts=40, seed=0):
    """Gauge-corrected average fidelity on the 4x4 corner (same formula, d=4)."""
    from scipy.optimize import minimize
    d = 2 ** n
    bits = np.array([[(i >> (n - 1 - k)) & 1 for k in range(n)] for i in range(d)])
    Uc = U.conj().T

    def neg(x):
        zl = np.exp(1j * (bits @ x[:n]))
        zr = np.exp(1j * (bits @ x[n:2 * n]))
        return -abs(np.trace(Uc @ ((zl[:, None] * M) * zr[None, :]))) ** 2

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        r = minimize(neg, rng.uniform(0, 2 * np.pi, 2 * n), method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-14, maxiter=8000))
        if best is None or r.fun < best.fun:
            best = r
    return (np.vdot(M, M).real + (-best.fun)) / (d * (d + 1))


def main():
    C16, _ = B.basis_states()
    print("anchor phases BEFORE fix (deg):",
          np.round(anchor_phase_report(C16), 3))
    _, pc = phase_fix_anchors(C16)            # pc in CONFIGS order
    bit = np.array([(c[0] << 3) | (c[1] << 2) | (c[2] << 1) | c[3]
                    for c in B.CONFIGS])
    pb = np.empty(16, dtype=complex)
    pb[bit] = pc                              # scatter into bit order
    print("distinct phases:", len(np.unique(np.round(np.angle(pc), 6))))

    d = os.path.join(GS_DIR, "results", "k16_confirm")
    files = sorted(glob.glob(os.path.join(d, "k16_*.npz")))
    files = [f for f in files if "_rescored" not in f]
    # confirm_k16.py has since been patched to apply the fix itself; a file it
    # wrote already carries the marker and must NOT be transformed again (the
    # phases are +-1, so a second application would silently undo the first).
    kept = []
    for f in files:
        if bool(np.load(f, allow_pickle=True).files.__contains__(
                "anchors_phase_fixed")):
            print(f"  skip (already phase-fixed at source): "
                  f"{os.path.basename(f)}")
        else:
            kept.append(f)
    files = kept
    if not files:
        raise SystemExit(f"no k16_*.npz in {d}")

    print(f"\n{'leg':6s} {'(tr,th)':22s} {'F16 stored':>11s} {'F16 fixed':>10s} "
          f"{'shift':>9s} {'corner F4':>10s} {'corner shift':>12s}")
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        leg = str(z["leg"])
        pair = tuple(int(x) for x in z["pair"])
        M = np.asarray(z["M"])
        # M -> diag(conj(p)) M diag(p)
        Mf = (pb.conj()[:, None] * M) * pb[None, :]
        U = target_sqiswap(pair)
        F16, fxc = fbar16(U, Mf)
        spect = tuple(s for s in range(4) if s not in pair)
        sub = [i for i in range(16)
               if all(((i >> (3 - s)) & 1) == 0 for s in spect)]
        Mlog = Mf[np.ix_(sub, sub)]
        F4 = fbar4_localZ(SQI4, Mlog)
        surv = (np.abs(Mf) ** 2).sum(axis=0)
        tag = f"({float(z['tau_ramp']):g},{float(z['tau_hold']):g})"
        print(f"{leg:6s} {tag:22s} {float(z['Fbar16']):11.6f} {F16:10.6f} "
              f"{F16-float(z['Fbar16']):+9.4f} {F4:10.6f} "
              f"{F4-float(z['F_corner4']):+12.2e}")
        out = f.replace(".npz", "_rescored.npz")
        np.savez(out, M=Mf, M_logical=Mlog, Fbar16=F16, F_xc16=fxc,
                 F_corner4=F4, survival=surv, phases_bitorder=pb,
                 tau_ramp=z["tau_ramp"], tau_hold=z["tau_hold"], leg=leg,
                 pair=np.array(pair), T_total=z["T_total"], norb=z["norb"],
                 dt=z["dt"], Fbar16_unfixed=z["Fbar16"])
        rows.append((leg, tag, F16, F4))
    print("\nwrote *_rescored.npz alongside the originals (originals untouched)")


if __name__ == "__main__":
    main()
