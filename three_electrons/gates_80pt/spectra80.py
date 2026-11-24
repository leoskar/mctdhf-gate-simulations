"""80-point single-well spectra: an FCI check and the centre-of-mass table.

    FOURWELL_NUM_DVR=80  python -m gates_80pt.spectra80

FCI check: single-well 3e FCI (``timing.FCIWell`` with n = 80, ``level_check.
fci_levels``) at the w1w2 gate point, well 1, must reproduce grid_convergence's
grid ladder at n = 80 to <= 1e-6:
e1 4.036971, e2 7.087706, e3 8.085080, e4 8.091519.

Table: ``grid_convergence.spectra.analyse`` (unchanged; it reads fourwell's grid at
call time) on 80 points at idle (all four wells) and at the three swept gate
points (both active wells): f1, w_n, e2/e1, A_3e = E3 - 2E1 + E0; plus the
bare pair detunings E(mn) - E(11) at each gate point. A/Delta is filled in
by ``oppoints80.py``. Output: results/spectra80.json.
"""
import time

import numpy as np

from . import common as C
from grid_convergence import spectra as S
from gate_timing.timing import FCIWell, h_well
from gate_timing.level_check import fci_levels
from system import fourwell as fw

FCI_REF = {1: 4.036971, 2: 7.087706, 3: 8.085080, 4: 8.091519}
SPEC_JSON = "spectra80.json"


def pair_detunings(gi, gj):
    """E(mn) - E(11) from the two wells' levels (bare, additive)."""
    ei = np.asarray(gi["E"]) - gi["E"][0]
    ej = np.asarray(gj["E"]) - gj["E"][0]
    e11 = ei[1] + ej[1]
    out = {}
    for m, n in ((3, 0), (0, 3), (4, 0), (0, 4), (2, 0), (0, 2), (2, 1), (1, 2)):
        out[f"|{m}{n}>"] = float(ei[m] + ej[n] - e11)
    return out


def main():
    t0 = time.time()
    dx = C.require_grid(80)
    fci = FCIWell(n=fw.NUM_DVR)
    print(f"  FCIWell: n = {fci.n}, D = {fci.D}", flush=True)

    # ── FCI check ──
    _, xg = C.x_scaled("w1w2")
    _, E, V, r = fci_levels(fci, h_well(1, xg[1], xg[5]), k=8)
    e = E - E[0]
    diffs = {k: float(e[k] - v) for k, v in FCI_REF.items()}
    worst = max(abs(v) for v in diffs.values())
    print("  FCI check: w1w2 gate point, well 1, n = 80: e1..e4 = "
          + " ".join(f"{e[k]:.7f}" for k in range(1, 5))
          + f"  (resid {r:.1e})", flush=True)
    print("  FCI check: minus grid_convergence ladder: "
          + " ".join(f"{diffs[k]:+.1e}" for k in range(1, 5)), flush=True)
    fci_pass = worst <= 1e-6
    print(f"  FCI check: {'PASS' if fci_pass else 'FAIL'} (worst {worst:.1e},"
          f" required 1e-6)", flush=True)
    out = {"grid": {"n": fw.NUM_DVR, "dx_nm": dx},
           "fci_check": dict(e=e[:5].tolist(), ref=FCI_REF, diffs=diffs, worst=worst,
                      resid=r, passed=fci_pass)}
    if not fci_pass:
        C.save_json(f"{C.RESULTS}/{SPEC_JSON}", out)
        raise SystemExit("FCI check failed: stopping before the table")

    # ── centre-of-mass table ──
    xi, _ = C.x_scaled("w1w2")
    out["idle"] = {}
    for w in range(fw.NUM_WELLS):
        out["idle"][w] = S.analyse(fci, w, xi[w], xi[4 + w])
        print(S.fmt_well(f"idle w{w}", out["idle"][w]), flush=True)
    out["gate3e"], out["legs"] = {}, {}
    for leg, (i, j) in C.LEGS3.items():
        _, xg = C.x_scaled(leg)
        g = {w: S.analyse(fci, w, xg[w], xg[4 + w]) for w in (i, j)}
        out["gate3e"][leg] = g
        for w in (i, j):
            print(S.fmt_well(f"{leg} gate w{w}", g[w]), flush=True)
        pdet = pair_detunings(g[i], g[j])
        out["legs"][leg] = dict(
            wells=(i, j), A3e={w: g[w]["A3e"] for w in (i, j)},
            A3e_4={w: g[w]["A3e_4"] for w in (i, j)},
            f1={w: g[w]["f1"] for w in (i, j)},
            E2_over_E1={w: g[w]["E2_over_E1"] for w in (i, j)},
            pair_detunings=pdet)
        print(f"  {leg} pair detunings E(mn)-E(11): "
              + "  ".join(f"{k} {v:+.6f}" for k, v in pdet.items()), flush=True)

    print("\n| point | well | e₁ | e₂/e₁ | f1 | w₀ | w on 2nd quantum (level) "
          "| A_3e = E3−2E1+E0 | E4−2E1+E0 |")
    print("|---|---|---|---|---|---|---|---|---|")

    def row(tag, w, r):
        wn = np.asarray(r["w_n"])
        k = int(np.argmax(wn[2:]) + 2)
        print(f"| {tag} | w{w} | {r['e'][1]:.4f} | {r['E2_over_E1']:.3f} | "
              f"{r['f1']:.4f} | {wn[0]:.4f} | {wn[k]:.4f} ({k}) | "
              f"{r['A3e']:+.5f} | {r['A3e_4']:+.5f} |")
    for w in range(fw.NUM_WELLS):
        row("idle", w, out["idle"][w])
    for leg, (i, j) in C.LEGS3.items():
        for w in (i, j):
            row(f"{leg} gate", w, out["gate3e"][leg][w])
    out["wall"] = time.time() - t0
    import json
    C.save_json(f"{C.RESULTS}/{SPEC_JSON}",
                json.loads(json.dumps(out, default=lambda o: o.tolist()
                                      if isinstance(o, np.ndarray) else float(o))))
    print(f"\n  saved -> results/{SPEC_JSON}   wall {out['wall']:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
