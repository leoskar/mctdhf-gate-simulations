"""Score an exact run and compare it against MCTDH (or against another exact run).

Two comparisons, both reported:

  ELEMENTWISE on the 16x16 M, and SCORED (F_bar16, F_xc, ||M||^2, corner F_bar4).
  Fidelities are gauge-corrected and therefore blind to things the raw matrix is
  not, so a run that agrees on F_bar16 but not on M has still told you something.

THE UNIFORM SHIFT IS UNDONE EXPLICITLY.  An x32 run propagates exp(-i dt (H-E0)),
so its matrix carries a known global factor exp(+i E0 T); it is divided out from
the stored e_shift and T rather than being fitted away, and the residual global
phase is then reported as a free-standing number.  Fitting the phase instead
would hide exactly the failure mode the precision gate exists to detect.

Env: TAG_A (required), TAG_B or MCTDH=<set>:<leg> / <set>:arm:<name>[:<tag>].
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_common as ec

PAIRS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3), "w0w3": (0, 3)}


def load_exact(tag, which="primary"):
    d = np.load(os.path.join(ec.RESULTS, tag, "M.npz"))
    meta = json.load(open(os.path.join(ec.RESULTS, tag, "meta.json")))
    M = d[f"M_{which}_cfg"]
    M = ec.to_bit_order(M) * np.exp(-1j * meta["e_shift"] * meta["T"])
    return M, meta


def load_mctdh(spec):
    """MCTDH=<set>:<leg>       -> the stored per-leg 16x16 from legs16.npz
       MCTDH=<set>:arm:<name>[:<legs_tag>]
                               -> the stored ARM matrix, newest arm_<name>_dt<x>_
                                  [<legs_tag>_]<stamp>.npz; no tag = untagged only
       MCTDH=k16:<leg>         -> gate_sweeps K=16 register confirmation

    All are already in bit-order (the MCTDH scorers permute before storing), so
    no further permutation is applied here."""
    parts = spec.split(":")
    pset = parts[0]
    if pset == "k16":
        # GAUGE, not a detail: confirm_k16.py wrote the early files BEFORE it
        # applied phase_fix_anchors at source, so their M sits in the UNFIXED
        # anchor gauge while the exact side phase-fixes before expanding.
        # rescore_k16.py applies the closing transform; its output is the
        # like-for-like matrix.  Picking the raw file instead would shift
        # F_bar16 by ~0.4 -- thousands of times the ansatz error being measured.
        leg = parts[1]
        d = os.path.join(ec.GS, "results", "k16_confirm")
        tr, th = (ec.TAU_W0W3 if leg == "w0w3"
                  else ec.PULSE_SETS[os.environ.get("PULSE_SET", "best")][leg])
        raw = os.path.join(d, f"k16_{leg}_tr{tr:g}_th{th:g}.npz")
        assert os.path.exists(raw), f"no K=16 confirmation at {raw}"
        z = np.load(raw, allow_pickle=True)
        if "anchors_phase_fixed" not in z.files:
            res = raw.replace(".npz", "_rescored.npz")
            assert os.path.exists(res), (
                f"{os.path.basename(raw)} is in the unfixed anchor gauge and has "
                f"no _rescored companion -- run rescore_k16.py")
            z, raw = np.load(res, allow_pickle=True), res
        print(f"  MCTDH K=16: {os.path.basename(raw)}  stored Fbar16 "
              f"{float(z['Fbar16']):.6f}  norb {list(z['norb'])}  "
              f"dt {float(z['dt']):g}", flush=True)
        return np.asarray(z["M"]), dict(source=raw, key="M",
                                        T=float(z["T_total"]))
    if parts[1] == "arm":
        # Match the frame tag EXACTLY: "newest arm_<name>_*" picks a tagged
        # re-run (e.g. _symz_ sorts after the date stamp) for an untagged spec.
        import glob, re
        tag = parts[3] if len(parts) > 3 else ""
        sfx = f"_{re.escape(tag)}" if tag else ""
        rx = re.compile(rf"arm_{re.escape(parts[2])}_dt[0-9.]+{sfx}_\d{{8}}_\d{{6}}\.npz")
        pat = os.path.join(ec.RW, "results", f"arms_{pset}",
                           f"arm_{parts[2]}_*.npz")
        cands = [f for f in sorted(glob.glob(pat))
                 if rx.fullmatch(os.path.basename(f))]
        assert cands, f"no MCTDH arm matching {rx.pattern} in {os.path.dirname(pat)}"
        f = cands[-1]
        d = np.load(f)
        print(f"  MCTDH arm: {os.path.basename(f)}  stored Fbar16 "
              f"{float(d['Fbar16']):.6f}", flush=True)
        return np.asarray(d["M"]), dict(source=f, key="M", T=float(d["T_total"]))
    leg = parts[1]
    f = os.path.join(ec.RW, "results", f"arms_{pset}", "legs16.npz")
    d = np.load(f)
    return np.asarray(d[f"M16_{leg}"]), dict(source=f, key=f"M16_{leg}",
                                             T=float(d[f"T_{leg}"]))


def report(name, M, pair=None):
    F, fxc, Mlog = ec.fbar16(M)
    print(f"  {name:22s} F_bar16 {F:.8f}   F_xc {fxc:.6f}   "
          f"||M||^2 {np.linalg.norm(M)**2:.6f} (ideal 16)", flush=True)
    if pair is not None and tuple(pair) != (0, 3):
        # the leg against its OWN target (confirm_k16 convention); the line
        # above keeps the (0,3) target every earlier number was scored with
        Fp, xp, _ = ec.fbar16(M, pair)
        print(f"  {'':22s} vs sqiSWAP{pair}: F_bar16 {Fp:.8f}   F_xc {xp:.6f}",
              flush=True)
    if pair is not None:
        c4 = ec.corner4(M, pair)
        print(f"  {'':22s} pair {pair} F_bar4 {ec.fbar4(ec.SQI, c4):.8f}   "
              f"||corner||^2 {np.linalg.norm(c4)**2:.6f} (ideal 4)", flush=True)
    else:
        print(f"  {'':22s} (0,3) logical corner F_bar4 "
              f"{ec.fbar4(ec.SQI, Mlog):.8f}   ||corner||^2 "
              f"{np.linalg.norm(Mlog)**2:.6f} (ideal 4)", flush=True)
        print(f"  {'':22s} per-column survival: "
              f"{np.round((np.abs(M)**2).sum(0), 4)}", flush=True)
    return F, fxc


def main():
    tag_a = os.environ["TAG_A"]
    which = os.environ.get("WHICH", "primary")
    A, meta = load_exact(tag_a, which)
    # arms carry "arm", legs carry "leg"; PAIRS only resolves for a single leg,
    # and for an arm the meaningful 4x4 is the (0,3) LOGICAL corner that fbar16
    # already returns.
    leg = meta.get("leg", meta.get("arm", "?"))
    pair = PAIRS.get(leg)

    print(f"=== compare_M  A = {tag_a} ({which} anchors) ===", flush=True)
    scheme = meta.get("propagator", f"sil m={meta.get('m', '?')}")
    if scheme == "strang":
        scheme += f" order {os.environ.get('SPLIT_ORDER', '?')}"
    print(f"  T = {meta['T']}  steps {meta['total_steps']}  {scheme}  "
          f"dt {meta.get('dt_fine')}  e_shift {meta['e_shift']:.6f}  "
          f"prec {meta['prec']}", flush=True)
    dn_max = max(s["dnorm"] for s in meta["summary"])
    line = f"  worst |1-norm| {dn_max:.2e}"
    if any("sil" in s for s in meta["summary"]):
        line += f"   worst SIL residual {max(s['sil'] for s in meta['summary']):.2e}"
    print(line, flush=True)

    ec.banner("scored")
    FA, xA = report("exact " + meta["prec"], A, pair)

    if os.environ.get("TAG_B"):
        B, mb = load_exact(os.environ["TAG_B"], os.environ.get("WHICH_B", which))
        label = os.environ["TAG_B"]
        FB, xB = report("exact " + mb["prec"], B, pair)
    elif os.environ.get("MCTDH"):
        B, mb = load_mctdh(os.environ["MCTDH"])
        label = os.environ["MCTDH"]
        FB, xB = report("MCTDH norb=4", B, pair)
        assert abs(mb["T"] - meta["T"]) < 1e-3, (
            f"durations differ: exact T={meta['T']} vs MCTDH T={mb['T']} "
            f"-- these are not the same schedule")
    else:
        return 0

    ec.banner(f"elementwise, A vs {label}")
    d = np.abs(A - B)
    print(f"  max|dM| {d.max():.3e}   rms|dM| {np.sqrt((d**2).mean()):.3e}   "
          f"max|A| {np.abs(A).max():.4f}", flush=True)
    # residual global phase, measured (not removed)
    ip = np.vdot(B, A)
    print(f"  residual global phase arg<B|A> = {np.angle(ip):+.3e} rad   "
          f"|<B|A>|/(||A|| ||B||) = "
          f"{abs(ip)/(np.linalg.norm(A)*np.linalg.norm(B)):.10f}", flush=True)
    Aal = A * np.exp(-1j * np.angle(ip))
    print(f"  after global-phase alignment: max|dM| "
          f"{np.abs(Aal - B).max():.3e}", flush=True)
    print(f"  dF_bar16 = {FA - FB:+.3e}    dF_xc = {xA - xB:+.3e}    "
          f"d||M||^2 = {np.linalg.norm(A)**2 - np.linalg.norm(B)**2:+.3e}",
          flush=True)
    if pair is not None and tuple(pair) != (0, 3):
        print(f"  vs sqiSWAP{pair}: dF_bar16 = "
              f"{ec.fbar16(A, pair)[0] - ec.fbar16(B, pair)[0]:+.3e}", flush=True)
    np.savez(os.path.join(ec.RESULTS, f"compare_{tag_a}_vs_"
                          f"{label.replace(':', '-')}.npz"),
             A=A, B=B, dF=FA - FB, dfxc=xA - xB,
             max_dM=d.max(), phase=np.angle(ip))
    return 0


if __name__ == "__main__":
    sys.exit(main())
