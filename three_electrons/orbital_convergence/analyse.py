"""Tables from the result files (numpy only; no solver, no grid).

    python -m orbital_convergence.analyse ladder     # w1w2 ladder: table, cliff, convergence
    python -m orbital_convergence.analyse holdscan   # w1w2: table beside the 64-point sweep
    python -m orbital_convergence.analyse holdscan w0w1   # (also w2w3)
    python -m orbital_convergence.analyse all

Writes results/ladder_summary.json and results/holdscan_summary.json
(w1w2; results/holdscan_summary_<leg>.json for w0w1 and w2w3) and prints
the markdown tables.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.dirname(HERE)
RES_OC = os.path.join(HERE, "results")
RES80 = os.path.join(PS, "gates_80pt", "results")
MF64 = os.path.join(PS, "gate_timing", "results", "w1w2")
SW64 = os.path.join(MF64, "diagonalize_mix1e-4_dt0.01_tol1e-10_rg0.1_s16o4_rnstep")
G80_RUNGS = (14, 18, 20, 22)
NEW_RUNGS = (6, 8, 10, 12, 13, 16, 24, 26)


def ladder80_file(N):
    if N in G80_RUNGS:
        return os.path.join(RES80, f"mf80_ladder_w1w2_n{N}.npz")
    return os.path.join(RES_OC, f"ladder80_w1w2_n{N}.npz")


def ladder64_file(N):
    if N == 14:
        return os.path.join(SW64, "pt_tr28p3_th105p4.npz")
    if N == 22:
        return os.path.join(PS, "grid_convergence", "results", "w1w2",
                            "gate_ladder_act22s4_tr28.3_th105.4.npz")
    return os.path.join(MF64, "exactcheck",
                        f"gate_ladder_act{N}s4_tr28.3_th105.4.npz")


def row_of(path):
    z = np.load(path)
    th = float(z["theta"])
    r = dict(F=float(z["F"]), ret11=float(z["retention"][3]),
             ret=[float(v) for v in z["retention"]],
             dtheta=th / (np.pi / 2) - 1, Fxc=float(z["Fxc"]),
             ms=float(z["sec_per_step"]) * 1e3, status=str(z["status"]),
             file=os.path.relpath(path, PS))
    for k in ("max_norm_dev", "M0_err", "wall"):
        if k in z.files:
            r[k] = float(z[k])
    return r


def aitken(x0, x1, x2):
    d1, d2 = x1 - x0, x2 - x1
    return x2 - d2 ** 2 / (d2 - d1)


def ladder():
    rows = {}
    for N in sorted(G80_RUNGS + NEW_RUNGS):
        p = ladder80_file(N)
        r = dict(N=N, r80=row_of(p) if os.path.exists(p) else None)
        px = os.path.join(RES_OC, f"ladder80_w1w2_n{N}_x64.npz")
        if os.path.exists(px):
            r["r80_x64"] = row_of(px)
        p64 = ladder64_file(N)
        r["r64"] = row_of(p64) if os.path.exists(p64) else None
        rows[N] = r
    print("| norb | F̄₄ 80 | F̄₄ 64 | \\|11⟩ ret 80 | \\|11⟩ ret 64 | "
          "θ/(π/2) − 1 (80) | θ/(π/2) − 1 (64) | F_xc 80 | F_xc 64 | "
          "ms/step 80 | max\\|norm − 1\\| 80 | source 80 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for N, r in rows.items():
        a, b = r["r80"], r["r64"]
        f = (lambda d, k, fmt: format(d[k], fmt) if d else "—")
        print(f"| {N} | {f(a, 'F', '.5f')} | {f(b, 'F', '.5f')} | "
              f"{f(a, 'ret11', '.4f')} | {f(b, 'ret11', '.4f')} | "
              f"{(format(a['dtheta'] * 100, '+.2f') + ' %') if a else '—'} | "
              f"{(format(b['dtheta'] * 100, '+.2f') + ' %') if b else '—'} | "
              f"{f(a, 'Fxc', '.4f')} | {f(b, 'Fxc', '.4f')} | "
              f"{f(a, 'ms', '.1f')} | "
              f"{f(a, 'max_norm_dev', '.1e') if a and 'max_norm_dev' in a else '—'} | "
              f"{'gates_80pt' if N in G80_RUNGS else 'orbital_convergence'} |")
    # flags: theta differs from the existing -4.0 % by > 0.5 points
    ref = np.mean([rows[N]["r80"]["dtheta"] for N in G80_RUNGS])
    flags = [N for N, r in rows.items() if r["r80"]
             and abs(r["r80"]["dtheta"] - ref) > 0.005]
    print(f"\n  θ reference (mean over the gates_80pt rungs) {ref * 100:+.2f} %; rungs differing "
          f"by > 0.5 points: {flags or 'none'}")
    # the cliff
    have = [N for N, r in rows.items() if r["r80"]]
    cliff = next((N for N in have if rows[N]["r80"]["F"] < 0.9), None)
    below = [N for N in have if N < (cliff or 99)]
    print(f"  cliff: first rung with F̄₄ < 0.9 = {cliff} "
          f"(previous rung present: {below[-1] if below else None})")
    # convergence over the 2-orbital steps 14 -> 26
    conv = {}
    seq = [N for N in range(14, 27, 2) if rows.get(N) and rows[N]["r80"]]
    for q in ("F", "ret11"):
        x = {N: rows[N]["r80"][q] for N in seq}
        steps = {f"{N}->{N + 2}": x[N + 2] - x[N] for N in seq if N + 2 in x}
        sv = list(steps.values())
        ratios = {f"{list(steps)[k + 1]} / {list(steps)[k]}":
                  sv[k + 1] / sv[k] for k in range(len(sv) - 1)}
        rv = list(ratios.values())
        c = dict(values=x, steps=steps, ratios=ratios)
        if len(rv) >= 2 and 26 in x:
            r1, r2 = rv[-2], rv[-1]
            c["criterion"] = dict(last_two=[r1, r2],
                                  both_below_1=bool(r1 < 1 and r2 < 1),
                                  differ=abs(r2 - r1))
            if r1 < 1 and r2 < 1 and abs(r2 - r1) <= 0.15:
                lim_a = aitken(x[22], x[24], x[26])
                lim_b = aitken(x[20], x[22], x[24])
                c["limit"] = dict(from_22_24_26=lim_a, from_20_22_24=lim_b,
                                  uncertainty=abs(lim_a - lim_b))
            else:
                c["limit"] = "not converged by norb 26"
        conv[q] = c
        print(f"\n  {q}: values " + ", ".join(f"{N}: {v:.5f}" for N, v in x.items()))
        print("    steps  " + ", ".join(f"{k} {v:+.5f}" for k, v in steps.items()))
        print("    ratios " + ", ".join(f"{k} = {v:.3f}" for k, v in ratios.items()))
        if "limit" in c:
            print(f"    limit: {c['limit']}")
    out = dict(rows={str(N): r for N, r in rows.items()}, theta_ref=ref,
               theta_flags=flags, cliff=cliff, convergence=conv)
    with open(os.path.join(RES_OC, "ladder_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def _tag(v):
    return f"{v:g}".replace(".", "p")


HOLDSCAN_RAMP = {"w1w2": 9.4, "w0w1": 12.6, "w2w3": 8.0}


def holdscan(leg="w1w2"):
    tr = _tag(HOLDSCAN_RAMP[leg])
    sw64 = os.path.join(PS, "gate_timing", "results", leg,
                        os.path.basename(SW64))
    d = glob.glob(os.path.join(RES_OC, "holdscan", leg, "*_act22s4"))
    assert len(d) == 1, d
    pts = sorted(glob.glob(os.path.join(d[0], f"pt_tr{tr}_th*.npz")),
                 key=lambda p: float(np.load(p)["tau_hold"]))
    rows = []
    print("| τ_hold | T_eff | θ/(π/2) | \\|11⟩ ret | ret \\|00⟩ \\|10⟩ \\|01⟩ | F̄₄ | "
          "F_xc | 64 pts: θ/(π/2) | \\|11⟩ ret | F̄₄ | F_xc |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in pts:
        z = np.load(p)
        th = float(z["tau_hold"])
        r = dict(tau_hold=th, T_eff=float(z["T_eff"]),
                 theta_rel=float(z["theta"]) / (np.pi / 2),
                 ret=[float(v) for v in z["retention"]], F=float(z["F"]),
                 Fxc=float(z["Fxc"]), status=str(z["status"]),
                 ms=float(z["sec_per_step"]) * 1e3,
                 file=os.path.relpath(p, PS))
        p64 = os.path.join(sw64, f"pt_tr{tr}_th{_tag(th)}.npz")
        if os.path.exists(p64):
            y = np.load(p64)
            r["r64"] = dict(theta_rel=float(y["theta"]) / (np.pi / 2),
                            ret=[float(v) for v in y["retention"]],
                            F=float(y["F"]), Fxc=float(y["Fxc"]),
                            file=os.path.relpath(p64, PS))
        rows.append(r)
        b = r.get("r64")
        s64 = (f"{b['theta_rel']:.4f} | {b['ret'][3]:.4f} | {b['F']:.5f} | "
               f"{b['Fxc']:.4f}" if b else "— | — | — | —")
        print(f"| {th:g} | {r['T_eff']:.1f} | {r['theta_rel']:.4f} | "
              f"{r['ret'][3]:.4f} | " + " ".join(f"{v:.5f}" for v in r["ret"][:3])
              + f" | {r['F']:.5f} | {r['Fxc']:.4f} | {s64} |")
    imin = int(np.argmin([r["ret"][3] for r in rows]))
    print(f"\n|11> retention minimum: {rows[imin]['ret'][3]:.4f} at τ_hold "
          f"{rows[imin]['tau_hold']:g} (θ/(π/2) {rows[imin]['theta_rel']:.4f})"
          f";  ms/step {min(r['ms'] for r in rows):.1f}–"
          f"{max(r['ms'] for r in rows):.1f}")
    r64 = [r for r in rows if "r64" in r]
    if r64:
        j = int(np.argmin([r["r64"]["ret"][3] for r in r64]))
        print(f"64 points: minimum {r64[j]['r64']['ret'][3]:.4f} at τ_hold "
              f"{r64[j]['tau_hold']:g} (θ/(π/2) "
              f"{r64[j]['r64']['theta_rel']:.4f})")
    name = ("holdscan_summary.json" if leg == "w1w2"
            else f"holdscan_summary_{leg}.json")
    out = dict(rows=rows, run_dir=os.path.relpath(d[0], PS))
    if leg != "w1w2":       # the w1w2 file has no leg keys
        out.update(leg=leg, tau_ramp=HOLDSCAN_RAMP[leg],
                   ret11_min=dict(tau_hold=rows[imin]["tau_hold"],
                                  theta_rel=rows[imin]["theta_rel"],
                                  ret11=rows[imin]["ret"][3]))
    with open(os.path.join(RES_OC, name), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ladder", "all"):
        ladder()
    if which in ("holdscan", "all"):
        holdscan(*sys.argv[2:3])
