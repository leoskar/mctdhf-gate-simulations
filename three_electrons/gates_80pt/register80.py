"""The 16-state idle register on the 80-point grid (GPU).

    FOURWELL_NUM_DVR=80  python -m gates_80pt.register80 prepare
    FOURWELL_NUM_DVR=80  python -m gates_80pt.register80 budget16

``prepare`` follows ``operating_points/prepare_states.py`` at the ML idle optimum's x
(``x_ghz`` of the 64-point idle file; x is grid-independent), norb 14 / n_s 4
on all wells, the 16 configurations relaxed together at weights 2^-n:

1. candidates in mixed precision, each to |dE| < 1e-9 (cap 25 000):
   ``cold16`` (isolated wells at iso_dt 2e-3 -> bare SPFs, one-hot A) and
   ``cold5to16`` (``operating_points.common.cold_state_opt``: 5-state cold relaxation,
   then the 16-state one), both at the ITE dt 2e-2 of prepare_states; and
   ``cold5to16_dt1e-2``, the same at dt 1e-2 — the second SETTING for the
   sum-E detector. prepare_states' ``warm`` candidate (the
   saved 64-point SA16 state) is NOT available on this grid and is dropped.
2. the lowest healthy weighted sum E polished in x64d (inner tol 1e-12)
   until the weighted sum E changes by < 1e-10 across a 50-iteration chunk
   (min 1000, cap 10 000), as prepare_states.
3. report (idle loss, min population, detunings) and save
   ``results/register80_idle.npz`` in the 64-point file's format.

``budget16``: the saved register with every well's orbitals embedded
exactly into norb 16 (empty orbitals appended; determinants mapped by their
occupied orbitals; A unchanged), relaxed in x64d at norb 16, detunings
compared with norb 14 (required ~1e-5).

``operating_points.common``'s isolated-orbital cache is keyed by (norb, dt, x) but not
by the grid, so CACHE_DIR is redirected to ``results/cache80`` before any
call; RESULTS_DIR too.
"""
import argparse
import json
import os
import time

from . import common as T  # noqa: I100  (x64, redirections)
from operating_points import common as C
from operating_points import validate as V

import jax.numpy as jnp
import numpy as np

from mctdhf.SemiDirectCI_n_species import SemiDirectCI_n_species

C.CACHE_DIR = os.path.join(T.RESULTS, "cache80")
C.RESULTS_DIR = os.path.join(T.RESULTS, "_operating_points")

LOSS_NAME = "stage1_singles_ground"
NORB, NSPF = 14, 4
OUT = T.REGISTER80


def idle_x():
    with np.load(T.OLD_IDLE_64) as d:
        x = np.asarray(d["x_ghz"], dtype=float)
    return x


def weighted_e(sv, an):
    return float(np.dot(sv.relax_weights, an["E_matched"]))


def summary(label, sv, st, h):
    an = sv.analyse(st, h)
    ew = weighted_e(sv, an)
    P = np.diag(an["F_matched"])
    print(f"  [{label}] weighted E {ew:.10f}  sum E {an['e_relax']:.8f}  "
          f"detunings " + " ".join(f"{v:.6f}" for v in an["detunings"])
          + f"  min P {P.min():.5f}  healthy {an['healthy']}", flush=True)
    return an, ew


def orthonormality(st):
    A, Bs, bs = (np.asarray(st[0]), [np.asarray(b) for b in st[1]],
                 [np.asarray(b) for b in st[2]])
    a = A.reshape(A.shape[0], -1)
    dev = dict(A=float(np.abs(a.conj() @ a.T - np.eye(len(a))).max()))
    dev["B"] = float(max(np.abs(B.conj().T @ B - np.eye(B.shape[1])).max()
                         for B in Bs))
    dev["b"] = float(max(np.abs(b.conj().T @ b - np.eye(b.shape[1])).max()
                         for b in bs))
    return dev


def polish(svx, st, h, tol, cap, min_steps):
    """prepare_states' x64d polish: stop when the weighted sum E changes by
    < tol across one whole chunk."""
    t0 = time.perf_counter()
    prev, dE, steps, converged, hist = None, None, 0, False, []
    while steps < cap:
        st, e, _info = svx.integrate(svx.chunk, h, st)
        steps += len(e)
        an = svx.analyse(st, h)
        ew = weighted_e(svx, an)
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
        if dE is not None and steps >= min_steps and abs(dE) < tol:
            converged = True
            break
    info = dict(steps=steps, converged=converged, last_dE_chunk=dE,
                wall=time.perf_counter() - t0, history=hist)
    print(f"  polish: {steps} it, converged {converged} (last dE/chunk "
          f"{dE:+.2e}), {info['wall']:.0f} s", flush=True)
    return st, info


def save_register(path, svx, st_rg, an, x, norb, extra):
    labels = np.array(["".join(map(str, c)) for c in C.CONFIGS])
    T.L.save_npz(path, A=np.asarray(st_rg[0]),
                 **{f"B{s}": np.asarray(st_rg[1][s]) for s in range(4)},
                 **{f"b{s}": np.asarray(st_rg[2][s]) for s in range(4)},
                 x_ghz=x, configs=np.array(C.CONFIGS), config_labels=labels,
                 relax_weights=np.asarray(svx.relax_weights),
                 E_matched=np.asarray(an["E_matched"]),
                 F_matched=np.asarray(an["F_matched"]),
                 detunings=np.asarray(an["detunings"]),
                 occupations=np.asarray(an["occupations"]),
                 norb=norb,
                 n_spf=NSPF, dt=svx.dt, eps=svx.eps,
                 num_dvr=T.fw.NUM_DVR, **extra)


def prepare(args):
    t_start = time.perf_counter()
    x = idle_x()
    h = C.h_of(x)
    rs = C.relax_set("16")
    print(f"idle x = {np.round(x, 8)}\nweights (2^-n) "
          f"{np.round(rs['relax_weights'], 4)}", flush=True)
    sv = C.make_solver(NORB, NSPF, **rs)
    sv1 = C.make_solver(NORB, NSPF, dt=1e-2, **rs)
    cands, record = {}, {}

    def keep(label, st, info, t0, solver):
        an, ew = summary(f"{label}: {info['steps']} it, converged "
                         f"{info['converged']}, {time.perf_counter() - t0:.0f} s",
                         solver, st, h)
        cands[label] = (st, ew, an)
        record[label] = dict(weighted_E=ew, sum_E=an["e_relax"],
                             steps=int(info["steps"]),
                             converged=bool(info["converged"]),
                             detunings=list(map(float, an["detunings"])),
                             minP=float(np.diag(an["F_matched"]).min()),
                             healthy=bool(an["healthy"]),
                             wall=time.perf_counter() - t0)

    t0 = time.perf_counter()
    bs_iso, iso_path = C.isolated_orbitals(x, NORB)       # iso_dt 2e-3
    print(f"  isolated orbitals: {iso_path}", flush=True)
    assert bs_iso[0].shape == (T.fw.NUM_DVR, NORB), bs_iso[0].shape
    st, _e, info = sv.relax_to_tol(h, sv.default_state(bs_iso, h),
                                   tol=args.cand_tol, max_steps=args.cand_cap)
    keep("cold16", st, info, t0, sv)
    for label, solver in (("cold5to16", sv), ("cold5to16_dt1e-2", sv1)):
        t0 = time.perf_counter()
        st, _e, info = C.cold_state_opt(solver, x, NORB, tol=args.cand_tol)
        keep(label, st, info, t0, solver)

    healthy = {k: v for k, v in cands.items() if v[2]["healthy"]}
    if not healthy:
        raise SystemExit("no healthy candidate")
    best = min(healthy, key=lambda k: healthy[k][1])
    print(f"  lowest weighted E: {best} ("
          + ", ".join(f"{k} {v[1] - healthy[best][1]:+.3e}"
                      for k, v in cands.items()) + ")", flush=True)

    svx = C.make_solver(NORB, NSPF, precision="x64d", inner_tol=1e-12, **rs)
    st, pol = polish(svx, cands[best][0], h, args.tol, args.polish_cap,
                     args.min_polish)
    an, out = V._full_report("final (x64d)", LOSS_NAME, C.load_loss(LOSS_NAME),
                             svx, st, h)
    ew_final = weighted_e(svx, an)
    st_rg = an["state_regauged"]
    ortho = orthonormality(st_rg)
    P = np.diag(an["F_matched"])
    print(f"  orthonormality: {ortho}", flush=True)
    det64 = np.asarray(T.REF64["detunings"])
    det = np.asarray(an["detunings"])
    print(f"  80-point: idle loss {out['loss']:.6f}  min P {P.min():.5f}  "
          f"detunings " + " ".join(f"{v:.6f}" for v in det), flush=True)
    print(f"  64-point: idle loss {T.REF64['idle_loss']:.4f}  min P "
          f"{T.REF64['idle_minP']:.4f}  detunings "
          + " ".join(f"{v:.6f}" for v in det64), flush=True)
    print("  shift 80 - 64: detunings " + " ".join(
        f"{v:+.2e}" for v in det - det64), flush=True)
    # the three detectors
    sumE = {k: v["weighted_E"] for k, v in record.items()}
    sumE["x64d polish"] = ew_final
    print("  detector 1, weighted sum E by setting (minus the x64d polish): "
          + ", ".join(f"{k} {v - ew_final:+.3e}" for k, v in sumE.items()),
          flush=True)
    print(f"  detector 2, healthy: {an['healthy']}", flush=True)
    print(f"  detector 3, purity (min P): {P.min():.6f}", flush=True)

    save_register(OUT, svx, st_rg, an, x, NORB,
                  dict(start=best, loss=out["loss"]))
    summ = dict(x_ghz=x.tolist(), candidates=record, chosen=best,
                polish=pol, final=out, weighted_E_final=ew_final,
                minP=float(P.min()), orthonormality=ortho,
                shift_detunings=(det - det64).tolist(),
                detector_sumE=sumE, num_dvr=T.fw.NUM_DVR,
                wall=time.perf_counter() - t_start)
    T.save_json(OUT[:-4] + ".json", summ)
    print(f"  saved {OUT}", flush=True)
    ok = (pol["converged"] and out["healthy"] and max(ortho.values()) < 1e-10)
    print(f"PREPARE {'PASS' if ok else 'CHECK'}  wall "
          f"{time.perf_counter() - t_start:.0f} s", flush=True)


def embed_orbitals_register(st, norb_old, norb_new, N=3):
    """Every well: b -> [b, orthonormal complement]; B rows scattered to the
    determinant with the same occupied orbitals; A unchanged. Exact."""
    A, Bs, bs = st
    occ_old = [tuple(np.flatnonzero(d)) for d in np.asarray(
        SemiDirectCI_n_species.get_slater_dets(norb_old, N))]
    new_idx = {tuple(np.flatnonzero(d)): i for i, d in enumerate(np.asarray(
        SemiDirectCI_n_species.get_slater_dets(norb_new, N)))}
    nBs, nbs = [], []
    for B, b in zip(Bs, bs):
        B, b = np.asarray(B), np.asarray(b)
        U = np.linalg.svd(b, full_matrices=True)[0]
        comp = U[:, norb_old:norb_new]
        comp -= b @ (b.conj().T @ comp)
        comp, _ = np.linalg.qr(comp)
        nbs.append(jnp.asarray(np.concatenate([b, comp], axis=1)))
        Bn = np.zeros((len(new_idx), B.shape[1]), dtype=B.dtype)
        for i_old, occ in enumerate(occ_old):
            Bn[new_idx[occ]] = B[i_old]
        nBs.append(jnp.asarray(Bn))
    return (jnp.asarray(A), nBs, nbs)


def budget16(args):
    t_start = time.perf_counter()
    d = np.load(OUT)
    assert int(d["num_dvr"]) == T.fw.NUM_DVR
    x = np.asarray(d["x_ghz"], dtype=float)
    h = C.h_of(x)
    rs = C.relax_set("16")
    st14 = C.load_state(OUT)
    sv14 = C.make_solver(NORB, NSPF, precision="x64d", inner_tol=1e-12, **rs)
    an14, ew14 = summary("norb 14 (saved)", sv14, st14, h)
    N = args.norb
    sv = C.make_solver(N, NSPF, precision="x64d", inner_tol=1e-12, **rs)
    st = embed_orbitals_register(st14, NORB, N)
    an0, ew0 = summary(f"norb {N} embedded", sv, st, h)
    print(f"  embedding: weighted E change {ew0 - ew14:+.2e} (exact: ~0)",
          flush=True)
    assert abs(ew0 - ew14) < 1e-8
    st, pol = polish(sv, st, h, args.tol, args.polish_cap, args.min_polish)
    an, out = V._full_report(f"norb {N} (x64d)", LOSS_NAME,
                             C.load_loss(LOSS_NAME), sv, st, h)
    ew = weighted_e(sv, an)
    dd = np.asarray(an["detunings"]) - np.asarray(an14["detunings"])
    print(f"  norb {N} - norb 14: weighted E {ew - ew14:+.3e}, detunings "
          + " ".join(f"{v:+.2e}" for v in dd)
          + f"  (max {np.max(np.abs(dd)):.1e}; required ~1e-5)", flush=True)
    st_rg = an["state_regauged"]
    path = os.path.join(T.RESULTS, f"register80_idle_norb{N}.npz")
    save_register(path, sv, st_rg, an, x, N, dict(loss=out["loss"]))
    T.save_json(path[:-4] + ".json", dict(
        norb=N, weighted_E=ew, weighted_E_14=ew14,
        detunings=list(map(float, an["detunings"])),
        detunings_14=list(map(float, an14["detunings"])),
        diff=dd.tolist(), max_diff=float(np.max(np.abs(dd))), polish=pol,
        final=out, wall=time.perf_counter() - t_start))
    print(f"  saved {path}  wall {time.perf_counter() - t_start:.0f} s",
          flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "budget16"])
    p.add_argument("--cand-tol", type=float, default=1e-9)
    p.add_argument("--cand-cap", type=int, default=25000)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--polish-cap", type=int, default=10000)
    p.add_argument("--min-polish", type=int, default=1000)
    p.add_argument("--norb", type=int, default=16)
    args = p.parse_args(argv)
    T.require_grid(80)
    T.L.device_banner()
    {"prepare": prepare, "budget16": budget16}[args.mode](args)


if __name__ == "__main__":
    main()
