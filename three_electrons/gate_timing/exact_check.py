"""Exactness check of the MF gate's |11> leakage: is it the SPF truncation?

At norb 6 an active well has C(6, 3) = 20 determinants, so ML-MCTDHF with
n_s = 20 on the active wells IS standard MCTDHF (``num_spfs = num_dets``).
Two jobs, same gate point and integrator as the sweep
(``mf_lib.Integ`` defaults):

    exact_check.py ns4  --leg w1w2 --tau-ramp 28.3 --tau-hold 105.4
        relax the MF pair states at [3,6,6,3] / n_s 4 (start: the relaxed
        norb-14 states projected onto the top-6 active natural orbitals),
        run (a) n_s 4 from them, then (b) n_s 20 from the SAME states embedded
        exactly (B_s completed to a unitary, A zero-padded): identical
        initial state and Hamiltonian, only the SPF truncation differs.
    exact_check.py ns20 --leg w1w2 --tau-ramp 28.3 --tau-hold 105.4
        (c) relax at n_s 20 (exact MCTDHF [3,6,6,3]) and run from its own
        relaxed states — a standard-MCTDHF MF gate.

The norb-14 / n_s 4 counterpart is the sweep point itself
(``results/<leg>/<sweep dir>/pt_tr*_th*.npz``). Every run also records the
column retention every 2 t.u., to see WHEN |11> loses its weight.
Outputs: ``results/<leg>/exactcheck/``.
"""
import argparse
import dataclasses
import json
import os
import time

import numpy as np
import jax.numpy as jnp

from . import mf_lib as L
from system import fourwell as fw

W = L.WELLS


def out_dir(leg):
    d = os.path.join(L.RESULTS_DIR, leg, "exactcheck")
    os.makedirs(d, exist_ok=True)
    return d


def project_active(leg, lay14, lay6, verbose=True, src=None, wells=None):
    """The relaxed norb-14 states, active wells projected onto their top
    ``lay6.norb_active`` natural orbitals (N-th compound matrix, as
    ``mf_lib.projected_states`` does for the spectators), top
    ``lay6.nspf_active`` natural SPFs kept, Löwdin across the 4 states."""
    h_idle = fw.h_static(L.x_vectors(leg)[0] / fw.TYPICAL_FREQ_GHZ)
    if src is None:
        ml14 = L.make_ml(leg, h_idle, lay14)
        y14, _ = L.mf_initial_states(leg, ml14, lay14, verbose=verbose)
    else:
        ml14, y14 = src
    ml6 = L.make_ml(leg, h_idle, lay6)
    A, Bs, bs = ml14._y_to_ABb(y14)
    A, Bs, bs = np.asarray(A), [np.asarray(B) for B in Bs], [np.asarray(b) for b in bs]
    w = lay14.state_weights()
    Ds = [np.asarray(D) for D in ml14.dCI.ml_1p_RDMs(
        jnp.asarray(A), [jnp.asarray(B) for B in Bs], weights=jnp.asarray(w))]
    newB, newb = list(Bs), list(bs)
    for s in (wells or L.LEGS[leg]):
        m = lay6.norbs(leg)[s]
        n_keep = lay6.nspfs(leg)[s]
        old_occ = np.array([np.flatnonzero(d) for d in np.asarray(ml14.dCI.I[s])])
        new_occ = np.array([np.flatnonzero(d) for d in np.asarray(ml6.dCI.I[s])])
        occ_ev, V = np.linalg.eigh(Ds[s])
        order = np.argsort(occ_ev)[::-1]
        occ_ev, V = occ_ev[order], V[:, order]
        best = None
        for label, U in (("V", V), ("conjV", V.conj())):
            R = U.conj().T[:m, :]
            sub = R[new_occ[:, None, :, None], old_occ[None, :, None, :]]
            Bp = np.linalg.det(sub) @ Bs[s]               # (D_new, n_old)
            Ap = L._move(A, Bp.T, s + 1)
            ret = float(np.diag(L._gram(Ap)).real.mean())
            if best is None or ret > best[0]:
                best = (ret, label, U, Ap)
        ret, label, U, Ap = best
        rho = L._node_density(Ap, s + 1, w)
        ev, Vs = np.linalg.eigh(rho)
        Vs = Vs[:, np.argsort(ev)[::-1][:n_keep]]
        A = L._move(Ap, Vs.conj(), s + 1)
        newB[s] = Vs
        newb[s] = bs[s] @ U[:, :m]
        if verbose:
            print(f"  [proj] w{s}: natural occupations "
                  + " ".join(f"{v:.3e}" for v in occ_ev[:m + 1])
                  + f"  retained {ret:.6f} ({label})", flush=True)
    S = L._gram(A)
    ev, U = np.linalg.eigh(S)
    A = np.einsum("ba,b...->a...", U @ np.diag(ev ** -0.5) @ U.conj().T, A)
    y = ml6._ABb_to_y(jnp.asarray(A, dtype=jnp.complex128),
                      [jnp.asarray(B, dtype=jnp.complex128) for B in newB],
                      [jnp.asarray(b, dtype=jnp.complex128) for b in newb])
    return ml6, y


def relax(leg, lay6, y_start, ml_start, inner_tol, verbose=True,
          precision="mixed", prefix=""):
    """``mf_lib.relax_mf_states`` from ``y_start`` instead of the HF
    projection (which needs the idle file's active budget)."""
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt
    from mctdhf.MLMCTDHF_n_species import make_regauge_ml
    xi, _ = L.x_vectors(leg)
    h_idle = fw.h_static(xi / fw.TYPICAL_FREQ_GHZ)
    cfgs = L.pair_configs(leg)
    ite = MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(), num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_mctdhf_orbitals=lay6.norbs(leg),
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_spfs=lay6.nspfs(leg), relax_configs=cfgs,
        relax_weights=list(L.RELAX_WEIGHTS), all_configs=cfgs, dt=2e-2,
        precision=precision, inner_tol=inner_tol, chunk=50, eps=1e-6)
    ml = L.make_ml(leg, h_idle, lay6)
    E0 = np.real(np.asarray(ml_start.calculate_energy(0.0, y_start)))
    A, Bs, bs = ml._y_to_ABb(y_start)
    t0 = time.time()
    state, e_sum, info = ite.relax_to_tol(h_idle, (A, list(Bs), list(bs)),
                                          tol=1e-10, max_steps=40000,
                                          verbose=False)
    h = ite.as_h(h_idle)
    A, Bs, bs = state
    Bs_rg, evals, evecs, ovl = ite._front_match(A, Bs, bs, h)
    cols, pop = ite._assign(ovl)
    A = evecs[:, jnp.asarray(cols)].T.reshape(ml.a_shape)
    y = make_regauge_ml(ml)(ml._ABb_to_y(A, list(Bs_rg), list(bs)))
    E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    err = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y)) - np.eye(4))))
    assert info["converged"] and np.all(np.isfinite(E)), (
        f"relaxation failed: converged {info['converged']}, E {E}")
    assert np.min(np.asarray(pop)) > 0.99, f"root matching failed: {pop}"
    assert err < 1e-10, f"relaxed M(0) != 1: {err:.3e}"
    print(f"  [relax] ({precision}) n_s {lay6.nspfs(leg)}: converged {info['converged']} "
          f"after {info['steps']} it ({time.time() - t0:.0f} s), fallbacks "
          f"{info['fallback_chunks']}, match pop "
          + " ".join(f"{v:.6f}" for v in np.asarray(pop)), flush=True)
    print("  [relax] E start   " + " ".join(f"{v:.8f}" for v in E0), flush=True)
    print("  [relax] E relaxed " + " ".join(f"{v:.8f}" for v in E)
          + f"   M(0)-1 {err:.1e}", flush=True)
    A, Bs, bs = ml._y_to_ABb(y)
    L.save_npz(os.path.join(out_dir(leg), f"relaxed_{prefix}{tag(lay6)}.npz"),
               A=np.asarray(A), **{f"B{s}": np.asarray(Bs[s]) for s in W},
               **{f"b{s}": np.asarray(bs[s]) for s in W}, E=E, E_start=E0,
               converged=bool(info["converged"]), steps=int(info["steps"]),
               inner_tol=inner_tol, match_pop=np.asarray(pop))
    return ml, y


def embed_full(leg, ml_small, y_small, lay_full, wells=None):
    """The same state with the active SPF bases enlarged to
    ``lay_full.nspf_active`` (all determinants, or fewer): B_s -> [B_s,
    orthonormal complement], A zero-padded. Exact."""
    h_idle = fw.h_static(L.x_vectors(leg)[0] / fw.TYPICAL_FREQ_GHZ)
    ml = L.make_ml(leg, h_idle, lay_full)
    A, Bs, bs = ml_small._y_to_ABb(y_small)
    A, Bs, bs = np.asarray(A), [np.asarray(B) for B in Bs], [np.asarray(b) for b in bs]
    for s in (wells or L.LEGS[leg]):
        B = Bs[s]
        n, D = B.shape[1], lay_full.nspfs(leg)[s]
        U = np.linalg.svd(B, full_matrices=True)[0]
        comp = U[:, n:D]
        comp -= B @ (B.conj().T @ comp)
        comp, _ = np.linalg.qr(comp)
        Bs[s] = np.concatenate([B, comp], axis=1)
        pad = [(0, 0)] * A.ndim
        pad[s + 1] = (0, D - n)
        A = np.pad(A, pad)
    y = ml._ABb_to_y(jnp.asarray(A), [jnp.asarray(B) for B in Bs],
                     [jnp.asarray(b) for b in bs])
    if np.prod([float(d) for d in ml.num_dets]) <= 1e7:
        dC = float(np.max(np.abs(np.asarray(ml.expand_C(*ml._y_to_ABb(y)[:2]))
                                 - np.asarray(ml_small.expand_C(
                                     *ml_small._y_to_ABb(y_small)[:2])))))
        print(f"  [embed] n_s {lay_full.nspfs(leg)}: max|dC| vs the n_s "
              f"{ml_small.num_spfs} state = {dC:.2e}", flush=True)
        assert dC < 1e-12
    else:
        # the full CI tensor is too large to form: compare the energies
        # (ml_small is built at the idle h by every caller)
        E_s = np.real(np.asarray(ml_small.calculate_energy(0.0, y_small)))
        E_b = np.real(np.asarray(ml.calculate_energy(0.0, y)))
        M0 = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y))
                                 - np.eye(4))))
        print(f"  [embed] n_s {lay_full.nspfs(leg)}: max|dE| vs the n_s "
              f"{ml_small.num_spfs} state {np.max(np.abs(E_b - E_s)):.2e}, "
              f"M(0)-1 {M0:.1e}", flush=True)
        assert np.max(np.abs(E_b - E_s)) < 1e-8 and M0 < 1e-10
    return y


def load_relaxed(leg, lay, ml, prefix=""):
    """A state saved by ``relax`` (results/<leg>/exactcheck/relaxed_*.npz)."""
    d = np.load(os.path.join(out_dir(leg), f"relaxed_{prefix}{tag(lay)}.npz"))
    return ml._ABb_to_y(jnp.asarray(d["A"]),
                        [jnp.asarray(d[f"B{s}"]) for s in W],
                        [jnp.asarray(d[f"b{s}"]) for s in W])


def embed_orbitals(leg, ml_small, y_small, lay_big, wells=None):
    """The same state with the active orbital sets enlarged to
    ``lay_big.norb_active``: b_s -> [b_s, orthonormal complement in the DVR
    space], each determinant mapped to the one with the same occupied
    orbitals (new orbitals appended, so ordering and signs are unchanged),
    B_s scattered accordingly, A unchanged. Exact; checked by the energy."""
    h_idle = fw.h_static(L.x_vectors(leg)[0] / fw.TYPICAL_FREQ_GHZ)
    ml = L.make_ml(leg, h_idle, lay_big)
    A, Bs, bs = ml_small._y_to_ABb(y_small)
    A, Bs, bs = np.asarray(A), [np.asarray(B) for B in Bs], [np.asarray(b) for b in bs]
    for s in (wells or L.LEGS[leg]):
        b = bs[s]
        m_old, m_new = b.shape[1], lay_big.norbs(leg)[s]
        U = np.linalg.svd(b, full_matrices=True)[0]
        comp = U[:, m_old:m_new]
        comp -= b @ (b.conj().T @ comp)
        comp, _ = np.linalg.qr(comp)
        bs[s] = np.concatenate([b, comp], axis=1)
        old = [tuple(np.flatnonzero(d)) for d in np.asarray(ml_small.dCI.I[s])]
        new = {tuple(np.flatnonzero(d)): i
               for i, d in enumerate(np.asarray(ml.dCI.I[s]))}
        Bn = np.zeros((len(new), Bs[s].shape[1]), dtype=Bs[s].dtype)
        for i_old, occ in enumerate(old):
            Bn[new[occ]] = Bs[s][i_old]
        Bs[s] = Bn
    y = ml._ABb_to_y(jnp.asarray(A), [jnp.asarray(B) for B in Bs],
                     [jnp.asarray(b) for b in bs])
    # ml_small must be built at the idle h (the caller's is)
    E_s = np.real(np.asarray(ml_small.calculate_energy(0.0, y_small)))
    E_b = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    M0 = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y)) - np.eye(4))))
    print(f"  [embed-orb] norb {lay_big.norbs(leg)}: max|dE| vs the small "
          f"state {np.max(np.abs(E_b - E_s)):.2e}, M(0)-1 {M0:.1e}", flush=True)
    assert np.max(np.abs(E_b - E_s)) < 1e-8 and M0 < 1e-10
    return y


def tag(lay):
    return f"act{lay.norb_active}s{lay.nspf_active}"


def run_gate(leg, lay, y0, tau_ramp, tau_hold, name, integ=None):
    env, T = L.make_env(tau_ramp, tau_hold, "full")
    ml = L.make_ml(leg, L.h_list_for(leg, env), lay)
    integ = integ or L.Integ()
    print(f"\n=== gate ({name}) leg {leg} norb {lay.norbs(leg)} n_s "
          f"{lay.nspfs(leg)} tau_ramp {tau_ramp} tau_hold {tau_hold} T {T} ===",
          flush=True)
    print(f"  integ {integ.as_dict()}", flush=True)
    snaps = [round(t, 1) for t in np.arange(2.0, T - 1e-9, 2.0)] + [T]
    P = L.Propagator(ml, integ)
    y, status, sn, sps, tcomp = P.run(y0, 0.0, T, snaps=snaps,
                                      probe=L.norm_probe(ml))
    print(f"  status {status}  {sps * 1e3:.2f} ms/step (compile {tcomp:.1f} s)",
          flush=True)
    ts = np.array(sorted(sn))
    ret_t = np.array([L.gate_matrix(ml, y0, jnp.asarray(sn[t]))[3] for t in ts])
    print("  retention vs t (|00> |10> |01> |11>):", flush=True)
    for t, r in zip(ts, ret_t):
        if abs(t % 10) < 1e-9 or t == ts[-1] or t <= tau_ramp + 2:
            print(f"    t {t:7.1f}  " + " ".join(f"{v:.5f}" for v in r),
                  flush=True)
    m = L.gate_metrics(ml, y0, y, tau_ramp + tau_hold,
                       float(L.load_grids()["legs"][leg]["Delta"]))
    L.print_metrics(m)
    pops = m.pop("spf_pops")
    path = os.path.join(out_dir(leg), f"gate_{name}_tr{tau_ramp:g}_th{tau_hold:g}.npz")
    L.save_npz(path, **m, spf_pops=np.concatenate([np.asarray(p) for p in pops]),
               layout=json.dumps(dataclasses.asdict(lay)),
               integ=json.dumps(integ.as_dict()), tau_ramp=tau_ramp,
               tau_hold=tau_hold, T=T, status=status, sec_per_step=sps,
               ret_t=ret_t, ret_ts=ts, y=np.asarray(y), y0=np.asarray(y0))
    print(f"  saved -> {path}", flush=True)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ns4", "ns20", "spf", "spfrelax",
                                      "norb", "ladder", "full",
                                      "fullbudget"])
    ap.add_argument("--nspf", type=int, default=8,
                    help="spf mode: target n_s on the active wells")
    ap.add_argument("--leg", default="w1w2", choices=["w0w1", "w1w2", "w2w3"])
    ap.add_argument("--tau-ramp", type=float, default=28.3)
    ap.add_argument("--tau-hold", type=float, default=105.4)
    ap.add_argument("--norb", type=int, default=6)
    ap.add_argument("--relax-inner-tol", type=float, default=1e-4)
    a = ap.parse_args()
    t0 = time.time()
    L.device_banner()
    lay14 = L.Layout()
    if a.which == "fullbudget":
        # NOT mean field at (norb, n_s) = (a.norb, a.nspf) on ALL wells.
        # Source: the relaxed full-register (14, 4) states
        # (relaxed_full_act14s4.npz). Every well: norb < 14 by natural-
        # orbital projection (keeping the top n_s natural SPFs), norb > 14 by
        # exact embedding of empty orbitals; n_s > 4 by exact embedding of
        # empty SPFs. Relaxed in x64d at the target budget, then the gate.
        h_idle = fw.h_static(L.x_vectors(a.leg)[0] / fw.TYPICAL_FREQ_GHZ)
        l14 = dataclasses.replace(lay14, norb_spec=14, nspf_spec=4)
        ml14 = L.make_ml(a.leg, h_idle, l14)
        y14 = load_relaxed(a.leg, l14, ml14, prefix="full_")
        N, n = a.norb, a.nspf
        lay_t = dataclasses.replace(l14, norb_active=N, nspf_active=n,
                                    norb_spec=N, nspf_spec=n)
        if N < 14:
            _, y0 = project_active(a.leg, l14, lay_t, src=(ml14, y14), wells=W)
        else:
            lay_o = dataclasses.replace(l14, norb_active=N, norb_spec=N)
            y0, ml_o = y14, ml14
            if N > 14:
                y0 = embed_orbitals(a.leg, ml14, y14, lay_o, wells=W)
                ml_o = L.make_ml(a.leg, h_idle, lay_o)
            if n > 4:
                y0 = embed_full(a.leg, ml_o, y0, lay_t, wells=W)
        ml_t = L.make_ml(a.leg, h_idle, lay_t)
        ml_r, y_r = relax(a.leg, lay_t, y0, ml_t, a.relax_inner_tol,
                          precision="x64d", prefix=f"full_all{N}s{n}_")
        pops = ml_r.spf_populations(y_r)
        for s in W:
            print(f"  [relax] w{s} SPF populations "
                  + " ".join(f"{float(v):.2e}" for v in np.asarray(pops[s])),
                  flush=True)
        name = f"full_all{N}s{n}"
        st = run_gate(a.leg, lay_t, y_r, a.tau_ramp, a.tau_hold, name)
        if st != "ok":
            print(f"\n  mixed-precision gate failed ({st}); rerunning ONCE "
                  f"with x64 SPF/orbital solves", flush=True)
            run_gate(a.leg, lay_t, y_r, a.tau_ramp, a.tau_hold, name + "_x64",
                     integ=L.Integ(precision="x64"))
        print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
        return
    if a.which == "full":
        # NOT mean field: norb 14 / n_s 4 on ALL four wells (correlated
        # spectators). The four pair states (spectators in |0>) are the
        # matching rows of the operating_points idle 16-state set (same budget),
        # Löwdin-orthonormalised, relaxed together in x64d with the MF
        # weights (1, 1/2, 1/2, 1/4), then the same gate.
        lay_f = dataclasses.replace(lay14, norb_spec=14, nspf_spec=4)
        h_idle = fw.h_static(L.x_vectors(a.leg)[0] / fw.TYPICAL_FREQ_GHZ)
        ml_f = L.make_ml(a.leg, h_idle, lay_f)
        A16, B16, b16, configs, norb_i, nspf_i = L._load_idle()
        assert (norb_i, nspf_i) == (14, 4)
        rows = [int(np.nonzero((configs == np.array(c)).all(axis=1))[0][0])
                for c in L.pair_configs(a.leg)]
        A = np.asarray(A16[rows])
        S = L._gram(A)
        ev, U = np.linalg.eigh(S)
        A = np.einsum("ba,b...->a...", U @ np.diag(ev ** -0.5) @ U.conj().T, A)
        print(f"  [full] idle rows {rows}, max|S - 1| before Löwdin "
              f"{np.max(np.abs(S - np.eye(4))):.1e}", flush=True)
        y0 = ml_f._ABb_to_y(jnp.asarray(A, dtype=jnp.complex128),
                            [jnp.asarray(B, dtype=jnp.complex128) for B in B16],
                            [jnp.asarray(b, dtype=jnp.complex128) for b in b16])
        ml_r, y_r = relax(a.leg, lay_f, y0, ml_f, a.relax_inner_tol,
                          precision="x64d", prefix="full_")
        pops = ml_r.spf_populations(y_r)
        for s in W:
            print(f"  [relax] w{s} SPF populations "
                  + " ".join(f"{float(v):.2e}" for v in np.asarray(pops[s])),
                  flush=True)
        name = "full_all14s4"
        st = run_gate(a.leg, lay_f, y_r, a.tau_ramp, a.tau_hold, name)
        if st != "ok":
            print(f"\n  mixed-precision gate failed ({st}); rerunning ONCE "
                  f"with x64 SPF/orbital solves", flush=True)
            run_gate(a.leg, lay_f, y_r, a.tau_ramp, a.tau_hold, name + "_x64",
                     integ=L.Integ(precision="x64"))
        print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
        return
    if a.which == "ladder":
        # (norb, n_s) ladder: source = the relaxed norb-14 states at the same
        # n_s (n_s 4: the sweep's mf_states; else relaxed_act14s<n> from the
        # spfrelax check); norb < 14 by natural-orbital projection, > 14 by
        # exact embedding; relaxed in x64d; gate. Files prefixed "ladder_".
        h_idle = fw.h_static(L.x_vectors(a.leg)[0] / fw.TYPICAL_FREQ_GHZ)
        lay_s = dataclasses.replace(lay14, nspf_active=a.nspf)
        ml_s = L.make_ml(a.leg, h_idle, lay_s)
        if a.nspf == lay14.nspf_active:
            y_s, _ = L.mf_initial_states(a.leg, ml_s, lay_s)
        else:
            y_s = load_relaxed(a.leg, lay_s, ml_s)
        lay_b = dataclasses.replace(lay_s, norb_active=a.norb)
        if a.norb > lay14.norb_active:
            y0 = embed_orbitals(a.leg, ml_s, y_s, lay_b)
        elif a.norb < lay14.norb_active:
            _, y0 = project_active(a.leg, lay_s, lay_b, src=(ml_s, y_s))
        else:
            y0 = y_s
        ml_b = L.make_ml(a.leg, h_idle, lay_b)
        ml_r, y_r = relax(a.leg, lay_b, y0, ml_b, a.relax_inner_tol,
                          precision="x64d", prefix="ladder_")
        pops = ml_r.spf_populations(y_r)
        for s in L.LEGS[a.leg]:
            print(f"  [relax] w{s} SPF populations "
                  + " ".join(f"{float(v):.2e}" for v in np.asarray(pops[s])),
                  flush=True)
        name = f"ladder_{tag(lay_b)}"
        st = run_gate(a.leg, lay_b, y_r, a.tau_ramp, a.tau_hold, name)
        if st != "ok":
            print(f"\n  mixed-precision gate failed ({st}); rerunning ONCE "
                  f"with x64 SPF/orbital solves", flush=True)
            run_gate(a.leg, lay_b, y_r, a.tau_ramp, a.tau_hold, name + "_x64",
                     integ=L.Integ(precision="x64"))
        print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
        return
    if a.which == "norb":
        # norb ladder at a fixed SPF budget: the relaxed norb-14 states at
        # n_s = a.nspf (from the spfrelax check) embedded into norb = a.norb
        # (empty extra orbitals), relaxed there in x64d, then the gate.
        h_idle = fw.h_static(L.x_vectors(a.leg)[0] / fw.TYPICAL_FREQ_GHZ)
        lay_s = dataclasses.replace(lay14, nspf_active=a.nspf)
        ml_s = L.make_ml(a.leg, h_idle, lay_s)
        y_s = load_relaxed(a.leg, lay_s, ml_s)
        lay_b = dataclasses.replace(lay_s, norb_active=a.norb)
        y_emb = embed_orbitals(a.leg, ml_s, y_s, lay_b)
        ml_b = L.make_ml(a.leg, h_idle, lay_b)
        ml_r, y_r = relax(a.leg, lay_b, y_emb, ml_b, a.relax_inner_tol,
                          precision="x64d")
        name = f"{tag(lay_b)}_relaxed"
        st = run_gate(a.leg, lay_b, y_r, a.tau_ramp, a.tau_hold, name)
        if st != "ok":
            print(f"\n  mixed-precision gate failed ({st}); rerunning ONCE "
                  f"with x64 SPF/orbital solves", flush=True)
            run_gate(a.leg, lay_b, y_r, a.tau_ramp, a.tau_hold, name + "_x64",
                     integ=L.Integ(precision="x64"))
        print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
        return
    if a.which == "spfrelax":
        # n_s-4 relaxed states embedded (empty extra SPFs), RELAXED at the
        # larger n_s in x64d (a float32 inner solve would amplify rounding
        # through the empty SPFs' regularised 1/eps density inverse, which
        # is what made the "spf" runs 6672/6673 diverge), then the gate.
        assert a.norb == lay14.norb_active and a.nspf > lay14.nspf_active
        h_idle = fw.h_static(L.x_vectors(a.leg)[0] / fw.TYPICAL_FREQ_GHZ)
        ml14 = L.make_ml(a.leg, h_idle, lay14)
        y14, _ = L.mf_initial_states(a.leg, ml14, lay14)
        lay_n = dataclasses.replace(lay14, nspf_active=a.nspf)
        y_emb = embed_full(a.leg, ml14, y14, lay_n)
        ml_n = L.make_ml(a.leg, h_idle, lay_n)
        ml_r, y_r = relax(a.leg, lay_n, y_emb, ml_n, a.relax_inner_tol,
                          precision="x64d")
        pops = ml_r.spf_populations(y_r)
        for s in L.LEGS[a.leg]:
            print(f"  [relax] w{s} SPF populations "
                  + " ".join(f"{float(v):.2e}" for v in np.asarray(pops[s])),
                  flush=True)
        name = f"{tag(lay_n)}_relaxed"
        st = run_gate(a.leg, lay_n, y_r, a.tau_ramp, a.tau_hold, name)
        if st != "ok":
            print(f"\n  mixed-precision gate failed ({st}); rerunning ONCE "
                  f"with x64 SPF/orbital solves (cn, tol 1e-10) — a check "
                  f"run, not the sweep setting", flush=True)
            run_gate(a.leg, lay_n, y_r, a.tau_ramp, a.tau_hold, name + "_x64",
                     integ=L.Integ(precision="x64"))
        print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
        return
    if a.which == "spf":
        # the sweep's own relaxed norb-14 / n_s-4 states, embedded exactly
        # into n_s = a.nspf; the n_s-4 counterpart is the sweep point
        assert a.norb == lay14.norb_active and a.nspf > lay14.nspf_active
        h_idle = fw.h_static(L.x_vectors(a.leg)[0] / fw.TYPICAL_FREQ_GHZ)
        ml14 = L.make_ml(a.leg, h_idle, lay14)
        y14, _ = L.mf_initial_states(a.leg, ml14, lay14)
        lay_n = dataclasses.replace(lay14, nspf_active=a.nspf)
        yn = embed_full(a.leg, ml14, y14, lay_n)
        run_gate(a.leg, lay_n, yn, a.tau_ramp, a.tau_hold,
                 f"{tag(lay_n)}_from_s4")
        print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
        return
    D = {6: 20, 7: 35}[a.norb]
    lay_s = dataclasses.replace(lay14, norb_active=a.norb, nspf_active=4)
    lay_f = dataclasses.replace(lay14, norb_active=a.norb, nspf_active=D)
    if a.which == "ns4":
        ml_p, y_p = project_active(a.leg, lay14, lay_s)
        ml4, y4 = relax(a.leg, lay_s, y_p, ml_p, a.relax_inner_tol)
        run_gate(a.leg, lay_s, y4, a.tau_ramp, a.tau_hold, f"{tag(lay_s)}")
        y20 = embed_full(a.leg, ml4, y4, lay_f)
        run_gate(a.leg, lay_f, y20, a.tau_ramp, a.tau_hold,
                 f"{tag(lay_f)}_from_s4")
    else:
        ml_p, y_p = project_active(a.leg, lay14, lay_f)
        mlf, yf = relax(a.leg, lay_f, y_p, ml_p, a.relax_inner_tol)
        run_gate(a.leg, lay_f, yf, a.tau_ramp, a.tau_hold, f"{tag(lay_f)}")
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
