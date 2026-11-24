"""Initial states for the real-time propagation: the 16 register eigenstates
at the ML idle optimum, strictly converged. GPU, through Slurm:

    sbatch -J prep-idle slurm/gpu.sbatch \
        -m operating_points.prepare_states

Protocol: the states are prepared ONLY at the idle point
and then propagated to the target points.

What it does
------------

All 16 register configurations relaxed together at weights 2^-n by
excitation number n (``C.relax_set("16")``, the set the targets were
optimised with), norb 14, n_s 4, at the idle optimum's x.

1. Candidates, mixed precision, each to ``|dE| < --cand-tol`` (1e-9):
   ``warm`` (the saved idle SA16 state, re-relaxed),
   ``cold16`` (isolated wells, bare SPFs, one-hot A) and ``cold5to16``
   (``C.cold_state_opt``). Their weighted sum E on the SAME functional
   decides: the warm idle state is known to keep well 0 on a stationary
   point ~1.9e-3 above the cold one.
2. The lowest candidate polished in ``x64d`` with inner tolerance 1e-12
   (deterministic, no float32 increment) until the weighted sum E changes by
   less than ``--tol`` across a whole chunk of 50 iterations (not between
   two iterations: a slow mode passes that test while it still moves,
   see common.py), capped at ``--polish-cap``. Detunings and populations are
   logged every 500 iterations so the drift is visible.
3. Saved in the natural gauge of the 2^-n weights, row ``a`` of ``A`` being
   the eigenstate matched to ``CONFIGS[a]``, with energies, the 16x16
   populations, occupations and the convergence record:
   ``results/idle_states_sa16_ns4_norb14.npz`` (+ ``.json`` summary).
"""

import argparse
import json
import os
import time

from operating_points import common as C  # noqa: I100  (sets x64 first)
from operating_points import validate as V

import jax.numpy as jnp
import numpy as np

IDLE_STEM = "ml_20260923_idle_stage1_singles_ground"
LOSS_NAME = "stage1_singles_ground"
NORB, NSPF = 14, 4


def _weighted_e(sv, an):
    return float(np.dot(sv.relax_weights, an["E_matched"]))


def _summary(label, sv, st, h):
    an = sv.analyse(st, h)
    ew = _weighted_e(sv, an)
    P = np.diag(an["F_matched"])
    print(f"  [{label}] weighted E {ew:.10f}  sum E {an['e_relax']:.8f}  "
          f"detunings " + " ".join(f"{v:.6f}" for v in an["detunings"])
          + f"  min P {P.min():.5f}  healthy {an['healthy']}", flush=True)
    return an, ew


def _orthonormality(st):
    """Largest deviation from orthonormality of the A rows, SPF columns and
    orbital columns."""
    A, Bs, bs = (np.asarray(st[0]), [np.asarray(b) for b in st[1]],
                 [np.asarray(b) for b in st[2]])
    a = A.reshape(A.shape[0], -1)
    dev = dict(A=float(np.abs(a.conj() @ a.T - np.eye(len(a))).max()))
    dev["B"] = float(max(np.abs(B.conj().T @ B - np.eye(B.shape[1])).max()
                         for B in Bs))
    dev["b"] = float(max(np.abs(b.conj().T @ b - np.eye(b.shape[1])).max()
                         for b in bs))
    return dev


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cand-tol", type=float, default=1e-9)
    p.add_argument("--cand-cap", type=int, default=25000)
    p.add_argument("--tol", type=float, default=1e-10,
                   help="weighted-E change across one 50-it chunk (x64d)")
    p.add_argument("--polish-cap", type=int, default=10000)
    p.add_argument("--min-polish", type=int, default=1000)
    p.add_argument("--out", default="idle_states_sa16_ns4_norb14")
    args = p.parse_args(argv)
    t_start = time.perf_counter()

    with np.load(os.path.join(C.RESULTS_DIR,
                              IDLE_STEM + "_best_state.npz")) as d:
        x = np.asarray(d["x_ghz"], dtype=float)
    h = C.h_of(x)
    rs = C.relax_set("16")
    print(f"idle x = {np.round(x, 8)}\nweights (2^-n) "
          f"{np.round(rs['relax_weights'], 4)}", flush=True)

    sv = C.make_solver(NORB, NSPF, **rs)
    cands, record = {}, {}

    # 1. candidates, mixed precision
    def run(label, st0):
        t0 = time.perf_counter()
        st, _e, info = sv.relax_to_tol(h, st0, tol=args.cand_tol,
                                       max_steps=args.cand_cap)
        an, ew = _summary(f"{label}: {info['steps']} it, converged "
                          f"{info['converged']}, "
                          f"{time.perf_counter() - t0:.0f} s", sv, st, h)
        cands[label] = (st, ew, an)
        record[label] = dict(weighted_E=ew, steps=info["steps"],
                             converged=bool(info["converged"]),
                             detunings=list(map(float, an["detunings"])),
                             healthy=bool(an["healthy"]),
                             wall=time.perf_counter() - t0)

    warm_path = os.path.join(C.RESULTS_DIR,
                             IDLE_STEM + "_sa16_ns4_state.npz")
    run("warm", C.load_state(warm_path))
    bs_iso, _p = C.isolated_orbitals(x, NORB)
    run("cold16", sv.default_state(bs_iso, h))
    t0 = time.perf_counter()
    st, _e, info = C.cold_state_opt(sv, x, NORB, tol=args.cand_tol)
    an, ew = _summary(f"cold5to16: {info['steps']} it "
                      f"({info['steps_5state']} 5-state), converged "
                      f"{info['converged']}, {time.perf_counter() - t0:.0f} s",
                      sv, st, h)
    cands["cold5to16"] = (st, ew, an)
    record["cold5to16"] = dict(weighted_E=ew, steps=info["steps"],
                               converged=bool(info["converged"]),
                               detunings=list(map(float, an["detunings"])),
                               healthy=bool(an["healthy"]),
                               wall=time.perf_counter() - t0)

    healthy = {k: v for k, v in cands.items() if v[2]["healthy"]}
    best = min(healthy, key=lambda k: healthy[k][1])
    print(f"  lowest weighted E: {best} ("
          + ", ".join(f"{k} {v[1] - healthy[best][1]:+.3e}"
                      for k, v in cands.items()) + ")", flush=True)

    # 2. polish in x64d
    svx = C.make_solver(NORB, NSPF, precision="x64d", inner_tol=1e-12, **rs)
    st = cands[best][0]
    t0 = time.perf_counter()
    prev = None
    steps, converged, hist = 0, False, []
    while steps < args.polish_cap:
        st, e, _info = svx.integrate(svx.chunk, h, st)
        steps += len(e)
        an = svx.analyse(st, h)
        ew = _weighted_e(svx, an)
        dE = None if prev is None else ew - prev
        prev = ew
        if steps % 500 == 0:
            hist.append(dict(it=steps, weighted_E=ew,
                             detunings=list(map(float, an["detunings"])),
                             P=np.diag(an["F_matched"]).tolist()))
            print(f"    x64d it {steps:6d}  weighted E {ew:.12f}  dE/chunk "
                  f"{dE if dE is None else f'{dE:+.2e}'}  det "
                  + " ".join(f"{v:.8f}" for v in an["detunings"]),
                  flush=True)
        if not np.isfinite(ew):
            break
        if dE is not None and steps >= args.min_polish and abs(dE) < args.tol:
            converged = True
            break
    polish = dict(steps=steps, converged=converged, last_dE_chunk=dE,
                  wall=time.perf_counter() - t0, history=hist)
    print(f"  polish: {steps} it, converged {converged} (last dE/chunk "
          f"{dE:+.2e}), {polish['wall']:.0f} s", flush=True)

    # 3. report and save, natural gauge
    an, out = V._full_report("final (x64d)", LOSS_NAME, C.load_loss(LOSS_NAME),
                             svx, st, h)
    A_rg, Bs_rg, bs = an["state_regauged"]
    st_rg = (A_rg, Bs_rg, bs)
    ortho = _orthonormality(st_rg)
    print(f"  orthonormality: {ortho}", flush=True)
    drift = None
    if len(hist) >= 3:
        drift = float(np.abs(np.array(hist[-1]["detunings"])
                             - np.array(hist[-3]["detunings"])).max())
        print(f"  detuning drift over the last 1000 it: {drift:.2e}",
              flush=True)

    path = os.path.join(C.RESULTS_DIR, args.out + ".npz")
    labels = np.array(["".join(map(str, c)) for c in C.CONFIGS])
    C.save_state(path, st_rg, x_ghz=x, configs=np.array(C.CONFIGS),
                 config_labels=labels,
                 relax_weights=np.asarray(svx.relax_weights),
                 E_matched=np.asarray(an["E_matched"]),
                 F_matched=np.asarray(an["F_matched"]),
                 detunings=np.asarray(an["detunings"]),
                 occupations=np.asarray(an["occupations"]),
                 norb=NORB, n_spf=NSPF, dt=svx.dt, eps=svx.eps,
                 start=best, idle_stem=IDLE_STEM)
    summary = dict(x_ghz=x.tolist(), candidates=record, chosen=best,
                   polish=polish, final=out, orthonormality=ortho,
                   detuning_drift_last_1000=drift,
                   wall=time.perf_counter() - t_start)
    with open(os.path.join(C.RESULTS_DIR, args.out + ".json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"  saved {path}", flush=True)
    ok = converged and out["healthy"] and max(ortho.values()) < 1e-10
    print(f"PREPARE {'PASS' if ok else 'CHECK'}", flush=True)


if __name__ == "__main__":
    main()
