"""Single-well spectroscopy and the centre-of-mass (COM) check.

    python -m grid_convergence.spectra

Points (bare isolated wells, the 64-point sine DVR of fourwell):
  * idle, all four wells (x_idle of mf_lib.x_vectors);
  * the 3e gate points of w0w1, w1w2, w2w3, w0w3 (each leg's x_target),
    both active wells;
  * the 1e gate points (Table S2; full precision from
    the 1e operating_points), both active wells.

Per (point, well):
  * 3e exact FCI, levels E0..E9 and eigenvectors (timing.FCIWell,
    level_check.fci_levels, k = 10);
  * 1e levels E0..E5 of the 64x64 one-body h_well (the same Hamiltonian);
  * D = sum_i (x_i - c_w), diagonal in the DVR determinant basis
    (d(det) = sum over occupied k of (x_k - c_w)); D_mn and (D^2)_mn for
    m, n < 10, and from them
      f1  = |D_10|^2 / (<0|D^2|0> - D_00^2)       (Kohn: 1 in a parabola)
      w_n = |D_n1|^2 / (<1|D^2|1> - D_11^2)       (harmonic: 1/3 on level 0,
                                                    2/3 on the 2nd COM quantum)
      |D_20|^2, |D_21|^2;  E2/E1 (excitation energies) against sqrt 3;
      A_3e = E3 - 2E1 + E0 and A_3e(4) = E4 - 2E1 + E0;  A_1e = E2 - 2E1 + E0.
Per leg: Delta_3e from gate_timing/grids.json; Delta_1e = pi/(2 T0) with the
one-electron T0 of Table S5 (in t.u.).

Check: the 3e levels must reproduce levels.json (<leg>.gate.<well>.E_fci) and
physics.json (iso.idle_bare, and iso.paths env = 1 bare for w0w3) to <= 1e-9.
Output: results/spectra.json and printed tables.
"""
import json
import os
import time

import numpy as np

from . import common as K
from gate_timing import mf_lib as L
from gate_timing.timing import FCIWell, h_well
from system import fourwell as fw

LEGS4 = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3), "w0w3": (0, 3)}
# 1e gate points, GHz: (alpha_i, alpha_j, beta_i, beta_j), 1e operating points
X1E_GHZ = {"w0w1": (11.745547, 11.747714, 1.322604, 1.537676),
           "w1w2": (12.959169, 12.959169, 1.976507, 2.0),
           "w2w3": (13.971952, 14.368281, 1.921143, 1.254329),
           "w0w3": (7.917293, 8.230025, 0.263383, 0.070993)}
T0_1E = {"w0w1": 129.2, "w1w2": 142.9, "w2w3": 154.2, "w0w3": 2100.9}
K_LEV = 10


def analyse(fci, w, a, b):
    E, V, r = K.well_levels(fci, w, a, b, k=K_LEV)
    c = float(fw.WELL_CENTERS[w])
    xk = fw.X_S[w] - c
    d = xk[fci.dets].sum(axis=1)
    D = V.T @ (d[:, None] * V)
    D2 = V.T @ ((d ** 2)[:, None] * V)
    e = E - E[0]
    f1 = D[1, 0] ** 2 / (D2[0, 0] - D[0, 0] ** 2)
    var1 = D2[1, 1] - D[1, 1] ** 2
    w_n = D[:, 1] ** 2 / var1
    e1 = np.linalg.eigvalsh(h_well(w, a, b))[:6]
    return dict(
        alpha=a, beta=b, E=E, resid=r, e=e, E1e=e1,
        D=D, D2=D2, f1=f1, w_n=w_n, w_n_sum=float(w_n.sum()),
        D20_sq=D[2, 0] ** 2, D21_sq=D[2, 1] ** 2, D10_sq=D[1, 0] ** 2,
        E2_over_E1=e[2] / e[1], A3e=E[3] - 2 * E[1] + E[0],
        A3e_4=E[4] - 2 * E[1] + E[0], A1e=e1[2] - 2 * e1[1] + e1[0],
        d01=e[1], d12=e[2] - e[1], d23=e[3] - e[2], d34=e[4] - e[3])


def gate_against_refs(res):
    """max |E - E_ref| against levels.json and physics.json."""
    mf = K.MF_RESULTS
    lv = json.load(open(os.path.join(mf, "levelcheck", "levels.json")))
    ph = json.load(open(os.path.join(mf, "timing", "physics.json")))["iso"]
    diffs = []
    for leg in ("w0w1", "w1w2", "w2w3"):
        for w in LEGS4[leg]:
            ref = np.array(lv[leg]["gate"][str(w)]["E_fci"])
            got = np.asarray(res["gate3e"][leg][w]["E"])[:len(ref)]
            diffs.append((f"levels.json {leg} gate w{w}", float(np.max(np.abs(got - ref)))))
            ref = np.array(lv[leg]["idle"][str(w)]["E_fci"])
            got = np.asarray(res["idle"][w]["E"])[:len(ref)]
            diffs.append((f"levels.json {leg} idle w{w}", float(np.max(np.abs(got - ref)))))
    for w in range(fw.NUM_WELLS):
        ref = np.array(ph["idle_bare"][str(w)]["E"])
        got = np.asarray(res["idle"][w]["E"])[:len(ref)]
        diffs.append((f"physics.json idle_bare w{w}", float(np.max(np.abs(got - ref)))))
    for w in LEGS4["w0w3"]:
        rows = [r for r in ph["paths"]["w0w3"][str(w)]
                if r["kind"] == "bare" and abs(r["env"] - 1.0) < 1e-12]
        ref = np.array(rows[0]["E"])
        got = np.asarray(res["gate3e"]["w0w3"][w]["E"])[:len(ref)]
        diffs.append((f"physics.json paths w0w3 env1 bare w{w}",
                      float(np.max(np.abs(got - ref)))))
    return diffs


def fmt_well(tag, r):
    return (f"  {tag:<22s} e1 {r['e'][1]:.5f}  E2/E1 {r['E2_over_E1']:.4f}  "
            f"f1 {r['f1']:.6f}  w_n " + " ".join(f"{v:.4f}" for v in r["w_n"])
            + f"  |D20|^2 {r['D20_sq']:.1e} |D21|^2 {r['D21_sq']:.1e}  "
            f"A3e {r['A3e']:+.5f} A3e(4) {r['A3e_4']:+.5f}  A1e {r['A1e']:+.5f}")


def main():
    t0 = time.time()
    fci = FCIWell()
    xi, _ = K.x_scaled("w1w2")
    res = {"idle": {}, "gate3e": {}, "gate1e": {}, "legs": {}}
    for w in range(fw.NUM_WELLS):
        res["idle"][w] = analyse(fci, w, xi[w], xi[4 + w])
        print(fmt_well(f"idle w{w}", res["idle"][w]), flush=True)
    grids = L.load_grids()
    for leg, (i, j) in LEGS4.items():
        _, xg = K.x_scaled(leg)
        res["gate3e"][leg] = {w: analyse(fci, w, xg[w], xg[4 + w]) for w in (i, j)}
        ai, aj, bi, bj = (v / fw.TYPICAL_FREQ_GHZ for v in X1E_GHZ[leg])
        res["gate1e"][leg] = {i: analyse(fci, i, ai, bi), j: analyse(fci, j, aj, bj)}
        for w in (i, j):
            print(fmt_well(f"{leg} 3e gate w{w}", res["gate3e"][leg][w]), flush=True)
        for w in (i, j):
            print(fmt_well(f"{leg} 1e gate w{w}", res["gate1e"][leg][w]), flush=True)
        D3 = float(grids["legs"][leg]["Delta"])
        D1 = np.pi / (2 * T0_1E[leg])
        g3, g1 = res["gate3e"][leg], res["gate1e"][leg]
        res["legs"][leg] = dict(
            wells=(i, j), Delta_3e=D3, Delta_1e=D1, T0_1e=T0_1E[leg],
            A1e={w: g1[w]["A1e"] for w in (i, j)},
            A3e={w: g3[w]["A3e"] for w in (i, j)},
            A3e_4={w: g3[w]["A3e_4"] for w in (i, j)},
            A1e_over_D={w: g1[w]["A1e"] / D1 for w in (i, j)},
            A3e_over_D={w: g3[w]["A3e"] / D3 for w in (i, j)},
            A3e_4_over_D={w: g3[w]["A3e_4"] / D3 for w in (i, j)})
    diffs = gate_against_refs(res)
    print("\n  gate against the stored FCI levels:", flush=True)
    for lab, d in diffs:
        print(f"    {lab:<40s} max|dE| {d:.2e}", flush=True)
    worst = max(d for _, d in diffs)
    res["gate"] = dict(diffs=diffs, worst=worst, passed=worst <= 1e-9)
    print(f"  GATE: {'PASS' if worst <= 1e-9 else 'FAIL'} (worst {worst:.2e})",
          flush=True)

    print("\n| leg | Δ_1e | A_1e (w_i, w_j) | A_1e/Δ_1e | Δ_3e | A_3e = E3−2E1+E0 "
          "(w_i, w_j) | A_3e/Δ_3e | E4−2E1+E0 (w_i, w_j) | (E4−2E1+E0)/Δ_3e |")
    print("|---|---|---|---|---|---|---|---|---|")
    for leg, r in res["legs"].items():
        i, j = r["wells"]
        print(f"| {leg} | {r['Delta_1e']:.4e} | {r['A1e'][i]:+.4f}, {r['A1e'][j]:+.4f} | "
              f"{r['A1e_over_D'][i]:+.0f}, {r['A1e_over_D'][j]:+.0f} | "
              f"{r['Delta_3e']:.4e} | {r['A3e'][i]:+.5f}, {r['A3e'][j]:+.5f} | "
              f"{r['A3e_over_D'][i]:+.2f}, {r['A3e_over_D'][j]:+.2f} | "
              f"{r['A3e_4'][i]:+.5f}, {r['A3e_4'][j]:+.5f} | "
              f"{r['A3e_4_over_D'][i]:+.2f}, {r['A3e_4_over_D'][j]:+.2f} |")
    K.save_json(os.path.join(K.RESULTS, "spectra.json"),
                json.loads(json.dumps(res, default=lambda o: o.tolist()
                                      if isinstance(o, np.ndarray) else float(o))))
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
