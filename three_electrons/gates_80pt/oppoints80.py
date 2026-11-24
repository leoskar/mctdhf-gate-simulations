"""The three operating points on the 80-point grid (GPU).

    FOURWELL_NUM_DVR=80  python -m gates_80pt.oppoints80 [--legs w0w1 w1w2 w2w3]

At each leg's 64-point target x (``mf_lib.target_path``'s ``best_x``; x is
grid-independent) the 16-state register (norb 14 / n_s 4 everywhere, weights
2^-n) is relaxed with the operating_points machinery (``timing.ml_solver``'s settings:
dt 2e-2, mixed, inner tol 1e-2), two candidates, the lower healthy sum E over
the relaxed set wins:

* ``cold5to16`` — ``operating_points.common.cold_state_opt`` (5-state cold relaxation
  from isolated wells at iso_dt 2e-3, then the 16-state one);
* ``seeded``    — the 80-point idle register (register80.py), relaxed at the target x.

Both to |dE| < 1e-9 (cap 25 000). The winner is then polished in x64d
(chunk-to-chunk weighted-E change < 1e-10, min 500, cap 5000); Delta is
reported before and after the polish.

Delta = |gap| * 2 sqrt(p_A (1 - p_A)) from the matched singles pair
(``timing.delta_leg``'s construction, block from the target file's
settings), plus the p_B / normalised-p / 2x2 variants.

Check: keep x if p_A in [0.45, 0.65]; otherwise the leg is flagged STOP (it
would need a new beta resonance search).

Outputs: results/oppoints80.json, results/oppoint80_<leg>_state.npz, and
A/Delta filled into results/spectra80.json.
"""
import argparse
import json
import os
import time

from . import common as T  # noqa: I100  (x64, redirections)
from . import register80 as R  # noqa: F401  (redirects operating_points's cache)
from operating_points import common as C

import numpy as np

PA_LO, PA_HI = 0.45, 0.65
OUT_JSON = os.path.join(T.RESULTS, "oppoints80.json")


def target(leg):
    with np.load(T.L.target_path(leg), allow_pickle=False) as d:
        x = np.asarray(d["best_x"], dtype=float)
        settings = json.loads(str(d["settings"]))
    a, b = settings["block"]
    assert tuple(settings["active"]) == T.LEGS3[leg]
    return x, (int(a), int(b))


def delta_of(sv, state, h, block):
    """``timing.delta_leg``'s numbers for ``state``."""
    a, b = block
    an = sv.analyse(state, h)
    E, F = an["E_matched"], an["F_matched"]
    gap = float(E[b] - E[a])
    pA, pB = float(F[a, a]), float(F[b, b])
    pn = pA / (F[a, a] + F[a, b])
    _A, _B, evals, evecs, _occ = sv._analyse_dev(*state, sv.as_h(h))
    evals, evecs = np.asarray(evals), np.asarray(evecs)
    rk = an["ranks_all"]
    V = evecs[np.ix_(sv._flat_all[[a, b]], rk[[a, b]])]
    w, Q = np.linalg.eigh(V.conj().T @ V)
    U = V @ Q @ np.diag(w ** -0.5) @ Q.conj().T
    Heff = U @ np.diag(evals[rk[[a, b]]]) @ U.conj().T
    return an, dict(
        block=[a, b], E_A=float(E[a]), E_B=float(E[b]), gap=gap, pA=pA,
        pB=pB, block_weight_A=float(F[a, a] + F[a, b]),
        block_weight_B=float(F[b, b] + F[b, a]),
        Delta=float(abs(gap) * 2 * np.sqrt(pA * (1 - pA))),
        Delta_pB=float(abs(gap) * 2 * np.sqrt(pB * (1 - pB))),
        Delta_pnorm=float(abs(gap) * 2 * np.sqrt(pn * (1 - pn))),
        Delta_2x2=float(2 * abs(Heff[0, 1])),
        eps_2x2=float(np.real(Heff[1, 1] - Heff[0, 0])),
        detunings=list(map(float, an["detunings"])),
        healthy=bool(an["healthy"]),
        minP=float(np.diag(F).min()),
        sum_E=float(an["e_relax"]),
        weighted_E=float(np.dot(sv.relax_weights, E)))


def run_leg(leg, args, sv, svx, seed_state):
    t0 = time.perf_counter()
    x, block = target(leg)
    h = C.h_of(x)
    print(f"\n=== {leg}: x = {np.round(x, 6)}  block {block} "
          f"({C.CONFIGS[block[0]]}, {C.CONFIGS[block[1]]}) ===", flush=True)
    cands = {}
    t1 = time.perf_counter()
    st, _e, info = C.cold_state_opt(sv, x, 14, tol=args.tol)
    an, d = delta_of(sv, st, h, block)
    cands["cold5to16"] = (st, d, info, time.perf_counter() - t1)
    t1 = time.perf_counter()
    st, _e, info = sv.relax_to_tol(h, seed_state, tol=args.tol,
                                   max_steps=args.cap)
    an, d = delta_of(sv, st, h, block)
    cands["seeded"] = (st, d, info, time.perf_counter() - t1)
    for k, (st, d, info, wall) in cands.items():
        print(f"  [{k}] {info['steps']} it, converged {info['converged']}, "
              f"{wall:.0f} s: sum E {d['sum_E']:.8f}  healthy {d['healthy']}  "
              f"det " + " ".join(f"{v:.6f}" for v in d["detunings"])
              + f"  gap {d['gap']:+.6e} p_A {d['pA']:.4f}  Delta "
              f"{d['Delta']:.6e}", flush=True)
    healthy = {k: v for k, v in cands.items() if v[1]["healthy"]
               and np.isfinite(v[1]["sum_E"])}
    if not healthy:
        return dict(leg=leg, status="NO HEALTHY CANDIDATE")
    best = min(healthy, key=lambda k: healthy[k][1]["sum_E"])
    print(f"  lower sum E: {best} (" + ", ".join(
        f"{k} {v[1]['sum_E'] - healthy[best][1]['sum_E']:+.3e}"
        for k, v in cands.items()) + ")", flush=True)
    d_mixed = cands[best][1]
    st, pol = R.polish(svx, cands[best][0], h, 1e-10, args.polish_cap, 500)
    an, d = delta_of(svx, st, h, block)
    st_rg = an["state_regauged"]
    ref_pA, ref_D = T.REF64["pA"][leg], T.REF64["Delta"][leg]
    keep = PA_LO <= d["pA"] <= PA_HI
    print(f"  final (x64d): gap {d['gap']:+.6e}  p_A {d['pA']:.4f} (64: "
          f"{ref_pA})  Delta {d['Delta']:.6e} (64: {ref_D:.5e}, "
          f"{d['Delta'] / ref_D - 1:+.2%})  2x2 {d['Delta_2x2']:.6e}  "
          f"block weight {d['block_weight_A']:.4f}  healthy {d['healthy']}  "
          f"min P {d['minP']:.4f}", flush=True)
    print(f"  check {leg}: p_A {d['pA']:.4f} {'in' if keep else 'OUTSIDE'} "
          f"[{PA_LO}, {PA_HI}] -> {'KEEP x' if keep else 'STOP'}",
          flush=True)
    path = os.path.join(T.RESULTS, f"oppoint80_{leg}_state.npz")
    labels = np.array(["".join(map(str, c)) for c in C.CONFIGS])
    T.L.save_npz(path, A=np.asarray(st_rg[0]),
                 **{f"B{s}": np.asarray(st_rg[1][s]) for s in range(4)},
                 **{f"b{s}": np.asarray(st_rg[2][s]) for s in range(4)},
                 x_ghz=x, configs=np.array(C.CONFIGS), config_labels=labels,
                 relax_weights=np.asarray(svx.relax_weights),
                 E_matched=np.asarray(an["E_matched"]),
                 F_matched=np.asarray(an["F_matched"]),
                 detunings=np.asarray(an["detunings"]), block=np.array(block),
                 num_dvr=T.fw.NUM_DVR)
    return dict(
        leg=leg, x_ghz=x.tolist(), status="ok", keep=bool(keep),
        chosen=best, candidates={k: dict(v[1], steps=int(v[2]["steps"]),
                                         converged=bool(v[2]["converged"]),
                                         wall=v[3])
                                 for k, v in cands.items()},
        mixed=d_mixed, polish=pol, final=d, ref64=dict(pA=ref_pA, Delta=ref_D),
        Delta_shift_rel=d["Delta"] / ref_D - 1, state=path,
        wall=time.perf_counter() - t0)


def fill_spectra(res):
    path = os.path.join(T.RESULTS, "spectra80.json")
    if not os.path.exists(path):
        print("  spectra80.json missing: A/Delta not filled", flush=True)
        return
    sp = T.load_json(path)
    for leg, r in res.items():
        if r.get("status") != "ok":
            continue
        D = r["final"]["Delta"]
        L = sp["legs"][leg]
        L["Delta_80"] = D
        L["A3e_over_D"] = {w: v / D for w, v in L["A3e"].items()}
        L["A3e_4_over_D"] = {w: v / D for w, v in L["A3e_4"].items()}
        L["pair_detunings_over_D"] = {k: v / D for k, v in
                                      L["pair_detunings"].items()}
        print(f"  {leg}: Delta_80 {D:.5e}  A_3e/Delta " + ", ".join(
            f"w{w} {v:.3f}" for w, v in L["A3e_over_D"].items()), flush=True)
    T.save_json(path, sp)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--legs", nargs="+", default=list(T.LEGS3))
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--cap", type=int, default=25000)
    p.add_argument("--polish-cap", type=int, default=5000)
    args = p.parse_args(argv)
    T.require_grid(80)
    T.L.device_banner()
    rs = C.relax_set("16")
    sv = C.make_solver(14, 4, dt=0.02, precision="mixed", inner_tol=0.01,
                       chunk=50, eps=1e-6, **rs)
    svx = C.make_solver(14, 4, precision="x64d", inner_tol=1e-12, **rs)
    d = np.load(T.REGISTER80)
    assert int(d["num_dvr"]) == T.fw.NUM_DVR
    seed = C.load_state(T.REGISTER80)
    res = T.load_json(OUT_JSON) if os.path.exists(OUT_JSON) else {}
    for leg in args.legs:
        try:
            res[leg] = run_leg(leg, args, sv, svx, seed)
        except Exception as exc:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[leg] = dict(leg=leg, status=f"FAILED: {exc!r}")
        T.save_json(OUT_JSON, res)
    fill_spectra(res)
    print("\n| leg | gap | p_A (64) | Δ_80 | Δ_64 | shift | keep |")
    print("|---|---|---|---|---|---|---|")
    for leg, r in res.items():
        if r.get("status") != "ok":
            print(f"| {leg} | {r.get('status')} |")
            continue
        f = r["final"]
        print(f"| {leg} | {f['gap']:+.5e} | {f['pA']:.4f} ({r['ref64']['pA']}) "
              f"| {f['Delta']:.5e} | {r['ref64']['Delta']:.5e} | "
              f"{r['Delta_shift_rel']:+.2%} | {r['keep']} |")


if __name__ == "__main__":
    main()
