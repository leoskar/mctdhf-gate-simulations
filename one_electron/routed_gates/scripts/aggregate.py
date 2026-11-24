"""Print the results of the routed circuits and the direct gate.

Pure reading and arithmetic -- no propagation, no re-scoring.  Everything it
prints is either stored in an .npz or derived from a stored matrix by the same
formula the drivers used.

THE DECOMPOSITION.  With d = 16,

    F_bar16 = [ Tr(MM^H) + F_xc * Tr(UU^H) * Tr(MM^H) ] / (d(d+1))
            = ||M||^2 (1 + 16 F_xc) / 272

so a departure between the propagated arm and its product model splits cleanly
into a POPULATION channel (||M||^2, i.e. leakage out of the computational
subspace -- gauge invariant, no local frame can recover it) and a COHERENT PHASE
channel (F_xc, the local-Z-optimised overlap).

    PULSE_SET=best python aggregate.py            # one set
    PULSE_SET=all  python aggregate.py            # best, sync, fast, direct
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RG_DIR = os.path.abspath(os.path.join(HERE, ".."))
LIB = os.path.abspath(os.path.join(RG_DIR, "..", "lib"))            # one_electron/lib
GS = os.path.abspath(os.path.join(RG_DIR, "..", "gate_sweeps"))     # K=16 direct gate
NS = 0.0159155                                  # t.u. -> ns

_S2 = 1.0 / np.sqrt(2.0)
SQI = np.array([[1, 0, 0, 0], [0, _S2, 1j * _S2, 0],
                [0, 1j * _S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)


def decomp(M):
    """(||M||^2, F_xc, F_bar16) -- F_xc back-solved from the stored F_bar16
    would hide a bug, so it is recomputed with the shared scoring code."""
    sys.path.insert(0, LIB)
    import run_route_sa as B
    U = B.target_sqiswap_03()
    nm = float(np.vdot(M, M).real)
    fx = float(B.fxc_4q(U, M))
    return nm, fx, nm * (1 + 16 * fx) / 272.0


def leak(nm, d=16):
    return 1.0 - nm / d


def show_set(setname):
    out = os.path.join(RG_DIR, "results", f"arms_{setname}")
    print(f"\n{'='*78}\n  PULSE SET: {setname}   ({out})\n{'='*78}")

    legs = os.path.join(out, "legs16.npz")
    if os.path.exists(legs):
        S = dict(np.load(legs))
        print("\n-- per-leg 16x16 matrices (Stage 2a) --")
        print(f"  {'leg':6s} {'T (t.u.)':>9s} {'T (ns)':>7s} {'||M16||^2':>10s} "
              f"{'leak':>7s} {'drift':>9s} {'ortho':>9s}")
        for k in ("w0w1", "w1w2", "w2w3", "par"):
            if f"M16_{k}" not in S:
                continue
            M = S[f"M16_{k}"]
            nm = float(np.vdot(M, M).real)
            T = float(S[f"T_{k}"])
            dr, orth = S[f"diag_{k}"]
            print(f"  {k:6s} {T:9.1f} {T*NS:7.2f} {nm:10.5f} {leak(nm):7.2%} "
                  f"{dr:9.2e} {orth:9.2e}")
        print(f"\n  {'leg/pair':10s} {'F_bar4 corner':>13s} {'||corner||^2':>13s} "
              f"{'z frame (rad)':>40s}")
        from qoct_gate_leg import avg_gate_fidelity_up_to_localZ as F4
        for k, pr in (("w0w1", (0, 1)), ("w1w2", (1, 2)), ("w2w3", (2, 3)),
                      ("par", (0, 1)), ("par", (2, 3))):
            key = f"corner_{k}_{pr[0]}{pr[1]}"
            if key not in S:
                continue
            c4 = S[key]
            print(f"  {k+str(pr):10s} {float(F4(SQI, c4)):13.6f} "
                  f"{float(np.linalg.norm(c4)**2):13.5f} "
                  f"{str(np.round(S[f'z_{k}_{pr[0]}{pr[1]}'], 3)):>40s}")

    for arm in ("serial", "parallel"):
        fs = sorted(glob.glob(os.path.join(out, f"arm_{arm}_*.npz")))
        fs = [f for f in fs if not os.path.basename(f).startswith("INVALID")]
        if not fs:
            print(f"\n-- {arm}: not run yet --")
            continue
        if len(fs) > 1:
            print(f"\n-- {arm}: {len(fs)} runs at different steps, all reported --")
        for f in fs:
            _one_arm(f, arm)


def _one_arm(f, arm):
        z = dict(np.load(f, allow_pickle=True))
        T = float(z["T_total"])
        print(f"\n-- {arm} arm ({os.path.basename(f)}) --")
        dtd = float(z["dt_div"]) if "dt_div" in z else 0.1
        step = {0.1: "10dt", 0.25: "4dt", 0.5: "2dt"}.get(dtd, f"dt_div={dtd:g}")
        print(f"  {int(z['n_legs'])} legs, {int(z['n_layers'])} layers, "
              f"{int(z.get('n_zframes', 0))} virtual-Z frame updates, "
              f"T = {T:.1f} t.u. = {T*NS:.2f} ns   [step: {step}]")
        rows = [("FULL PROPAGATION", z["M"]),
                ("product, implemented locals", z["M_prod_impl"]),
                ("product, ideal locals", z["M_prod_idl"])]
        print(f"  {'':30s} {'F_bar16':>9s} {'||M||^2':>9s} {'leak':>8s} "
              f"{'F_xc':>9s}")
        base = None
        for tag, M in rows:
            nm, fx, fb = decomp(M)
            if base is None:
                base = (nm, fx, fb)
            print(f"  {tag:30s} {fb:9.6f} {nm:9.5f} {leak(nm):8.2%} {fx:9.6f}")
        for tag, M in rows[1:]:
            nm, fx, fb = decomp(M)
            print(f"    {tag} MINUS propagation: dF_bar16 = {fb-base[2]:+.6f}  "
                  f"[population d||M||^2 = {nm-base[0]:+.5f}, "
                  f"coherent dF_xc = {fx-base[1]:+.6f}]")
        Ml = z["M_logical"]
        from qoct_gate_leg import avg_gate_fidelity_up_to_localZ as F4
        print(f"  (0,3) logical corner F_bar4 = {float(F4(SQI, Ml)):.6f}   "
              f"survival per col = {np.round((np.abs(Ml)**2).sum(0), 4)}")
        surv = (np.abs(z["M"]) ** 2).sum(axis=0)
        print(f"  16-column survival: min {surv.min():.4f}  max {surv.max():.4f}"
              f"  mean {surv.mean():.4f}   (>1 anywhere = instability)")


def show_direct():
    print(f"\n{'='*78}\n  DIRECT ARM (comparison)\n{'='*78}")
    d = os.path.join(GS, "results")
    f = os.path.join(d, "k16_confirm", "k16_w0w3_tr81.3_th2160.7_rescored.npz")
    raw = os.path.join(d, "k16_confirm", "k16_w0w3_tr81.3_th2160.7.npz")
    if not os.path.exists(f) and not os.path.exists(raw):
        print("  w0w3 K=16 register run not found.")
        return
    z = dict(np.load(f if os.path.exists(f) else raw, allow_pickle=True))
    M = z["M"]
    nm, fx, fb = decomp(M)
    T = float(z["T_total"])
    print(f"  w0w3 (81.3, 2160.7)  T = {T:.1f} t.u. = {T*NS:.2f} ns")
    print(f"  F_bar16 = {fb:.6f}   ||M||^2 = {nm:.5f}  leak {leak(nm):.2%}   "
          f"F_xc = {fx:.6f}")
    print(f"  (0,3) corner F_bar4 = {float(z['F_corner4']):.6f}")
    surv = (np.abs(M) ** 2).sum(axis=0)
    print(f"  16-column survival: min {surv.min():.4f} max {surv.max():.4f}")


if __name__ == "__main__":
    import jax
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, LIB)
    os.environ.setdefault("GATE_LEG", "w0w1")
    want = os.environ.get("PULSE_SET", "all")
    for s in (("best", "sync", "fast") if want == "all" else (want,)):
        show_set(s)
    show_direct()
