"""ML-MCTDHF dressed pair levels at the gate point (an earlier study).

    FOURWELL_NUM_DVR=80 python -m orbital_convergence.dressed80 r1 --leg w1w2
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.dressed80 variant --leg w1w2 --kind eps8
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.dressed80 variant --leg w1w2 --kind random
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.dressed80 variant --leg w1w2 --kind x64d
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.dressed80 ext --leg w1w2
    FOURWELL_NUM_DVR=80 python -m orbital_convergence.dressed80 variant --leg w1w2 \
        --base ext --kind x64d

**ext**: r1's start and
settings, with the six leak partners of the leg added to the relaxed set at
2^-(m+n), spectators in |0>; the register weights keep their 2^-n ratios. The
partners' A rows start one-hot on the SBS carrying their bare levels.
Variants of it are named ``ext_<kind>`` and compared with it on all six
partners.

Full system (active wells norb 22 / n_s 6, spectators norb 16 / n_s 4), the
16 register states at the weights 2^-n of Sec. S3 B, relaxed with
``MLMCTDHF_n_species_ITE_opt`` at the leg's static gate-point Hamiltonian.

**r1** starts from ``gates_80pt/results/oppoint80_<leg>_state.npz`` (the gates_80pt
gate-point relaxation, norb 14 / n_s 4 everywhere):

1. the saved state is re-analysed here and must reproduce its ``E_matched``
   (<= 1e-8): same Hamiltonian, same configurations, same weights;
2. orbitals embedded exactly (14 -> 22 active, 14 -> 16 spectators;
   ``register80.embed_orbitals_register``'s construction, re-implemented
   here so nothing of gates_80pt is imported with side effects); the top block
   restricted to the original SBS must reproduce the norb-14 spectrum
   (<= 1e-8);
3. two SBS added to each active well: the lowest bare single-well
   eigenvectors at the gate point outside span(B) ("bare start"); A is zero
   on them.

**variant** restarts from r1's state with one change: ``eps8`` (eps 1e-8),
``random`` (every SBS of an active well with occupation < 1e-4 replaced by a
random orthonormal vector, A zeroed on it) or ``x64d`` (precision x64d,
inner tol 1e-12).

**Readout** (every run): natural-SBS occupations; overlaps with the bare
eigenvectors of each active well in the same orbital space
(``ml_species_ops`` at the final orbitals); SBS -> level labels (Hungarian);
the diabatic dressed detunings (<mn|H|mn> - <11|H|11>) / Delta_80 of the
top-layer Hamiltonian in the natural-SBS product basis, spectators in |0>;
the bare detunings in the same orbital space and the exact ones
(``spectra80.json``); the adiabatic populations of the roots on those
configurations. Written to results/dressed80/<leg>_<run>.{npz,json}.
"""
import argparse
import json
import os
import sys
import time

from . import common as C  # noqa: I100  (x64 + redirections first)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from mctdhf.MLMCTDHF_n_species_ITE_opt import (  # noqa: E402
    MLMCTDHF_n_species_ITE_opt, register_configs, _additivity, _top_matrix)

fw = C.fw
OUT = os.path.join(C.RESULTS, "dressed80")
LEGS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3)}
NORB_ACT, NORB_SPEC, NS_ACT, NS_SPEC = 22, 16, 6, 4
NORB_OP, NS_OP = 14, 4                       # oppoint80_<leg>_state.npz
CONFIGS = register_configs(4, 2)             # register order
WEIGHTS = [0.5 ** sum(c) for c in CONFIGS]   # 2^-n
PAIRS = [(0, 0), (1, 0), (0, 1), (1, 1),
         (2, 0), (0, 2), (3, 0), (0, 3), (4, 0), (0, 4)]
PARTNERS = PAIRS[4:]
NLEV = 10                                    # bare levels for the overlaps
TOL, CAP, CHUNK = 1e-9, 20000, 50
CKPT_EVERY = 20                              # chunks


def layout(leg):
    act = LEGS[leg]
    norbs = [NORB_ACT if w in act else NORB_SPEC for w in range(4)]
    nspfs = [NS_ACT if w in act else NS_SPEC for w in range(4)]
    return norbs, nspfs


def ext_configs(leg):
    """ext: the 16 register states at their 2^-n weights plus the six leak
    partners of the leg, spectators in |0>, at 2^-(m+n) (excitation quanta;
    for the register states quanta = excited wells, so their weights are
    unchanged). The partners' configuration indices are natural-SBS indices,
    which equal the level indices only if the natural order is the level
    order; ``cmd_ext`` checks that at the start and ``readout`` at the end."""
    i, j = LEGS[leg]
    cfgs, wts = list(CONFIGS), list(WEIGHTS)
    for (m, n) in PARTNERS:
        c = [0, 0, 0, 0]
        c[i], c[j] = m, n
        cfgs.append(tuple(c))
        wts.append(0.5 ** (m + n))
    return cfgs, wts


def solver(norbs, nspfs, precision="mixed", eps=1e-6, leg=None):
    """``leg`` given: the extended relaxed set of ext (``ext_configs``)."""
    cfgs, wts = ext_configs(leg) if leg else (CONFIGS, WEIGHTS)
    return MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(), num_particles=[fw.NUM_PARTICLES] * 4,
        num_mctdhf_orbitals=list(norbs),
        num_spatial_orbitals=[fw.NUM_DVR] * 4, num_spfs=list(nspfs),
        relax_configs=cfgs, relax_weights=wts, all_configs=CONFIGS,
        dt=2e-2, precision=precision,
        inner_tol=1e-2 if precision == "mixed" else 1e-12, chunk=CHUNK,
        eps=eps)


def gate_point(leg):
    o = C.load_json(os.path.join(C.G80_RESULTS, "oppoints80.json"))[leg]
    assert o["status"] == "ok", o["status"]
    x = np.asarray(o["x_ghz"], dtype=np.float64)
    return x, fw.h_static(x / fw.TYPICAL_FREQ_GHZ), float(o["final"]["Delta"])


def load_oppoint(leg, x):
    p = os.path.join(C.G80_RESULTS, f"oppoint80_{leg}_state.npz")
    d = np.load(p)
    assert int(d["num_dvr"]) == 80, p
    assert np.array_equal(np.asarray(d["x_ghz"], dtype=np.float64), x), \
        "oppoint state is not at oppoints80.json's x"
    assert [tuple(int(v) for v in c) for c in d["configs"]] == CONFIGS
    w = np.asarray(WEIGHTS) / np.sum(WEIGHTS)
    assert np.allclose(d["relax_weights"], w, rtol=0, atol=1e-15)
    st = (jnp.asarray(d["A"]), [jnp.asarray(d[f"B{s}"]) for s in range(4)],
          [jnp.asarray(d[f"b{s}"]) for s in range(4)])
    assert st[0].shape == (16,) + (NS_OP,) * 4
    return st, d, p


def species_ops(ite, bs, h):
    """``Op_s`` (bare single-well Hamiltonian in determinant space) at
    ``bs``, exactly as ``ITE_opt.default_state`` builds it."""
    from opt_einsum import contract
    ml, h = ite.ml, ite.as_h(h)
    bs = [jnp.asarray(b, dtype=jnp.complex128) for b in bs]
    h_MO = []
    for s in range(4):
        h_1 = contract("jm, ij -> im", bs[s], h[s], backend="jax")
        h_MO.append(contract("in, im -> nm", bs[s].conj(), h_1,
                             backend="jax"))
    g_MO, _ = ml._transform_all_g(bs, full=False)
    return [np.asarray(o) for o in ml.dCI.ml_species_ops(h_MO, g_MO)]


def bare_levels(Op, k=NLEV):
    e, v = np.linalg.eigh(0.5 * (Op + Op.conj().T))
    return e[:k], v[:, :k]


def embed_orbitals(st, small, big):
    """b -> [b, orthonormal complement]; B rows scattered to the determinant
    with the same occupied orbitals (new orbitals appended, so order and
    signs are unchanged); A unchanged. Exact."""
    A, Bs, bs = st
    nBs, nbs = [], []
    for s in range(4):
        B, b = np.asarray(Bs[s]), np.asarray(bs[s])
        m_old = b.shape[1]
        m_new = int(np.asarray(big.ml.dCI.I[s]).shape[1])   # occupation rows
        if m_new == m_old:
            nBs.append(B)
            nbs.append(b)
            continue
        U = np.linalg.svd(b, full_matrices=True)[0]
        comp = U[:, m_old:m_new]
        comp -= b @ (b.conj().T @ comp)
        comp, _ = np.linalg.qr(comp)
        nbs.append(np.concatenate([b, comp], axis=1))
        old = [tuple(np.flatnonzero(r)) for r in np.asarray(small.ml.dCI.I[s])]
        new = {tuple(np.flatnonzero(r)): i
               for i, r in enumerate(np.asarray(big.ml.dCI.I[s]))}
        Bn = np.zeros((len(new), B.shape[1]), dtype=B.dtype)
        for i_old, occ in enumerate(old):
            Bn[new[occ]] = B[i_old]
        nBs.append(Bn)
    return np.asarray(A), nBs, nbs


def pad_A(A, nspfs):
    out = np.zeros((A.shape[0],) + tuple(nspfs), dtype=np.complex128)
    out[(slice(None),) + tuple(slice(0, n) for n in A.shape[1:])] = A
    return out


def add_spfs_bare(B, V, n_new):
    """Columns of V (energy order) projected out of span(B) and of each
    other; the first ``n_new`` with residual norm > 0.3. Returns the new B
    and the bare levels used."""
    cols, used = [np.asarray(B)], []
    for k in range(V.shape[1]):
        Q = np.concatenate(cols, axis=1)
        r = V[:, k] - Q @ (Q.conj().T @ V[:, k])
        nr = np.linalg.norm(r)
        if nr > 0.3:
            cols.append((r / nr)[:, None])
            used.append((k, float(nr)))
        if len(used) == n_new:
            break
    assert len(used) == n_new, used
    return np.concatenate(cols, axis=1), used


def replace_spfs_random(B, idx, rng):
    keep = [j for j in range(B.shape[1]) if j not in idx]
    Q = B[:, keep]
    R = rng.normal(size=(B.shape[0], len(idx))) \
        + 1j * rng.normal(size=(B.shape[0], len(idx)))
    R -= Q @ (Q.conj().T @ R)
    R, _ = np.linalg.qr(R)
    Bn = B.copy()
    Bn[:, idx] = R
    return Bn


def out_path(leg, run, ext):
    os.makedirs(OUT, exist_ok=True)
    return C.assert_writable(os.path.join(OUT, f"{leg}_{run}.{ext}"))


def save_state(path, st, **meta):
    A, Bs, bs = st
    C.L.save_npz(path, A=np.asarray(A),
                 **{f"B{s}": np.asarray(Bs[s]) for s in range(4)},
                 **{f"b{s}": np.asarray(bs[s]) for s in range(4)},
                 num_dvr=fw.NUM_DVR, **meta)


def load_state(path):
    d = np.load(path)
    assert int(d["num_dvr"]) == 80, path
    return (jnp.asarray(d["A"]), [jnp.asarray(d[f"B{s}"]) for s in range(4)],
            [jnp.asarray(d[f"b{s}"]) for s in range(4)]), d


# ── relaxation (relax_to_tol's loop, with timing and checkpoints) ─────────

ORDER_EVERY = 4                              # chunks, ext runs


def natural_labels(ite, st, h, leg):
    """SBS -> level (Hungarian on bare overlaps) of the active wells in the
    natural gauge of ``ite``'s weights."""
    _A, Bs_rg, _e, _v, _o = ite._analyse_dev(*st, ite.as_h(h))
    Ops = species_ops(ite, st[2], h)
    out = {}
    for s in LEGS[leg]:
        _e, V = bare_levels(Ops[s])
        O = np.abs(V.conj().T @ np.asarray(Bs_rg[s])) ** 2
        _r, c = linear_sum_assignment(-O.T)
        out[s] = [int(v) for v in c]
    return out


def relax(ite, h, st, ckpt, order_check=None):
    """``order_check`` (ext runs): stop as soon as the natural order of the
    active wells' SBS is no longer the level order, since the partners'
    weights would then sit on the wrong roots."""
    t0, steps, e_all, converged = time.time(), 0, [], False
    k, order_broken = 0, None
    while steps < CAP:
        st, e, info = ite.integrate(CHUNK, h, st)
        e_all.append(e)
        steps += len(e)
        k += 1
        if not np.all(np.isfinite(e)):
            print("  NON-FINITE energy: stop", flush=True)
            break
        dt_ = time.time() - t0
        dE = float(e[-1] - e[-2])
        print(f"    it {steps:6d}  sum E {e[-1]:.10f}  dE {dE:+.3e}  "
              f"fallback {bool(info['fallback_chunks'])}  "
              f"{dt_ / steps:.3f} s/it  ({dt_ / 3600:.2f} h)", flush=True)
        if abs(dE) < TOL:
            converged = True
            break
        if order_check is not None and k % ORDER_EVERY == 0:
            lab = order_check(st)
            if any(v[:5] != [0, 1, 2, 3, 4] for v in lab.values()):
                order_broken = dict(steps=steps, labels=lab)
                print(f"  NATURAL ORDER BROKEN at it {steps}: {lab}; stop",
                      flush=True)
                break
        if k % CKPT_EVERY == 0:
            save_state(ckpt, st, steps=steps)
    e_sum = np.concatenate(e_all) if e_all else np.zeros(0)
    return st, dict(converged=converged, steps=steps, wall=time.time() - t0,
                    order_broken=order_broken,
                    s_per_it=(time.time() - t0) / max(steps, 1),
                    e_last=float(e_sum[-1]) if len(e_sum) else np.nan)


# ── readout ───────────────────────────────────────────────────────────────

def readout(ite, st, h, leg, Delta):
    i, j = LEGS[leg]
    A, Bs, bs = st
    A_rg, Bs_rg, evals, evecs, occ = ite._analyse_dev(A, Bs, bs, ite.as_h(h))
    evals, evecs = np.asarray(evals), np.asarray(evecs)
    occ = [np.asarray(o) for o in occ]
    Ops = species_ops(ite, bs, h)
    lab, inv, ovl, ebare = {}, {}, {}, {}
    for s in (i, j):
        e, V = bare_levels(Ops[s])
        O = np.abs(V.conj().T @ np.asarray(Bs_rg[s])) ** 2   # level x SBS
        r, c = linear_sum_assignment(-O.T)                  # SBS -> level
        assert list(r) == list(range(O.shape[1]))
        lab[s] = [int(v) for v in c]
        inv[s] = {lv: q for q, lv in enumerate(lab[s])}
        ovl[s], ebare[s] = O, e
    H = (evecs * evals) @ evecs.conj().T                   # top, natural SBS
    nspfs = ite.num_spfs

    def flat(m, n):
        cfg = [0, 0, 0, 0]
        cfg[i], cfg[j] = inv[i][m], inv[j][n]
        return int(np.ravel_multi_index(tuple(cfg), nspfs))

    sp = C.load_json(os.path.join(C.G80_RESULTS, "spectra80.json"))
    pd = sp["legs"][leg]["pair_detunings"]
    Ed = {p: float(np.real(H[flat(*p), flat(*p)])) for p in PAIRS
          if p[0] in inv[i] and p[1] in inv[j]}
    missing = [(s, lv) for s in (i, j) for lv in range(5) if lv not in inv[s]]
    ok = [p for p in PAIRS if p[0] in inv[i] and p[1] in inv[j]]
    rows = {}
    for (m, n) in PARTNERS:
        if (m, n) not in ok or (1, 1) not in ok:
            rows[f"{m}{n}"] = dict(dressed=float("nan"), missing=True)
            continue
        J = flat(m, n)
        pops = np.abs(evecs[J, :]) ** 2
        k = int(np.argmax(pops))
        rows[f"{m}{n}"] = dict(
            dressed=(Ed[(m, n)] - Ed[(1, 1)]) / Delta,
            bare_space=((ebare[i][m] - ebare[i][1])
                        + (ebare[j][n] - ebare[j][1])) / Delta,
            bare_exact=pd[f"|{m}{n}>"] / Delta,
            sbs=[inv[i][m], inv[j][n]],
            overlap=[float(ovl[i][m, inv[i][m]]), float(ovl[j][n, inv[j][n]])],
            adiabatic_max_pop=float(pops[k]),
            adiabatic_root_pop11=float(np.abs(evecs[flat(1, 1), k]) ** 2))
    # ITE_opt.analyse stacks the occupations of all species into one array,
    # which fails for unequal n_s; match() + _additivity give the same E.
    # Matched on the 16 register configurations whatever the relaxed set.
    mt = ite.match(st, h, configs=CONFIGS)
    _det, _res, healthy = _additivity(mt["evals"], CONFIGS)
    extra = {}
    if len(ite.relax_configs) > len(CONFIGS):     # ext: partner roots
        mr = ite.match(st, h)
        extra = dict(
            partner_E_matched=np.asarray(mr["evals"][16:]).tolist(),
            partner_pop_relax=np.asarray(mr["overlaps"][16:]).tolist())
    return dict(
        **extra,
        natural_order_is_level_order=all(lab[s][:5] == [0, 1, 2, 3, 4]
                                         for s in (i, j)),
        leg=leg, Delta_80=Delta, pairs=rows, missing_levels=missing,
        diabatic_E={f"{m}{n}": v for (m, n), v in Ed.items()},
        labels={f"w{s}": lab[s] for s in (i, j)},
        level_overlap={f"w{s}": [float(ovl[s][lv, inv[s][lv]]) if lv in inv[s]
                                 else 0.0 for lv in range(5)]
                       for s in (i, j)},
        occupations={f"w{s}": occ[s].tolist() for s in range(4)},
        overlaps={f"w{s}": ovl[s].tolist() for s in (i, j)},
        qubit_levels_are_top2=all(lab[s][:2] == [0, 1] for s in (i, j)),
        E_matched=np.asarray(mt["evals"]).tolist(),
        pop_relax=np.asarray(mt["overlaps"]).tolist(),
        healthy=bool(healthy), e_relax=float(np.sum(mt["evals"])))


def print_readout(r, title):
    print(f"\n=== readout: {title} ===", flush=True)
    for s, o in r["occupations"].items():
        print(f"  occ {s}: " + " ".join(f"{v:.2e}" for v in o), flush=True)
    for s, l in r["labels"].items():
        print(f"  SBS -> level {s}: {l}  (overlap of levels 0-4: "
              + " ".join(f"{v:.3f}" for v in r["level_overlap"][s]) + ")",
              flush=True)
    if r["missing_levels"]:
        print(f"  !! levels without an SBS (well, level): "
              f"{r['missing_levels']}", flush=True)
    print(f"  qubit levels are the top-2 SBS: {r['qubit_levels_are_top2']}; "
          f"healthy {r['healthy']}; min pop_relax "
          f"{min(r['pop_relax']):.4f}; natural order = level order (0-4) "
          f"{r['natural_order_is_level_order']}", flush=True)
    if "partner_pop_relax" in r:
        print("  partner roots, pop on their configuration: "
              + " ".join(f"|{k}> {p:.3f}" for k, p in
                         zip(r["pairs"], r["partner_pop_relax"])), flush=True)
    print("  pair  dressed/Δ  bare(space)/Δ  bare(exact)/Δ  overlaps   "
          "adiabatic: max pop, its |11> pop", flush=True)
    for k, v in r["pairs"].items():
        if v.get("missing"):
            print(f"  |{k}>  (level not represented)", flush=True)
            continue
        print(f"  |{k}>  {v['dressed']:+9.4f}  {v['bare_space']:+9.4f}  "
              f"{v['bare_exact']:+9.4f}  {v['overlap'][0]:.3f} "
              f"{v['overlap'][1]:.3f}   {v['adiabatic_max_pop']:.3f} "
              f"{v['adiabatic_root_pop11']:.3f}", flush=True)


# ── commands ──────────────────────────────────────────────────────────────

def prepare(leg, big):
    """Steps 1-3 of r1's start (module docstring) with both checks.
    Returns the state, h, Delta and the metadata."""
    x, h, Delta = gate_point(leg)
    st0, d0, src = load_oppoint(leg, x)
    small = solver([NORB_OP] * 4, [NS_OP] * 4)
    an0 = small.analyse(st0, h)
    dE = float(np.max(np.abs(np.asarray(an0["E_matched"]) - d0["E_matched"])))
    print(f"  [check 1] {os.path.basename(src)} re-analysed: max|E_matched "
          f"- saved| = {dE:.2e}", flush=True)
    assert dE < 1e-8, dE
    E_all_small = np.sort(np.asarray(an0["E_all"]))

    nspfs = big.num_spfs
    A, Bs, bs = embed_orbitals(st0, small, big)
    Ops = species_ops(big, bs, h)
    used = {}
    for s in LEGS[leg]:
        _e, V = bare_levels(Ops[s])
        Bs[s], used[s] = add_spfs_bare(Bs[s], V, NS_ACT - NS_OP)
        print(f"  [spf] well {s}: added bare levels "
              + ", ".join(f"{k} (residual {r:.3f})" for k, r in used[s]),
              flush=True)
    A = pad_A(A, nspfs)
    st = (jnp.asarray(A), [jnp.asarray(B) for B in Bs],
          [jnp.asarray(b) for b in bs])
    H = np.asarray(_top_matrix(big.ml, st[1], st[2], big.as_h(h),
                               big.a_shape))
    sub = np.ravel_multi_index(
        np.indices((NS_OP,) * 4).reshape(4, -1), nspfs)
    ev = np.sort(np.linalg.eigvalsh(H[np.ix_(sub, sub)]))
    dS = float(np.max(np.abs(ev - E_all_small)))
    print(f"  [check 2] embedded top block on the original SBS vs norb 14: "
          f"max|dE| = {dS:.2e}", flush=True)
    assert dS < 1e-8, dS
    return st, h, Delta, dict(start=os.path.relpath(src, C.HERE),
                              added=str(used), check1=dE, check2=dS)


def cmd_r1(a):
    leg = a.leg
    big = solver(*layout(leg))
    st, h, Delta, meta = prepare(leg, big)
    r0 = readout(big, st, h, leg, Delta)
    print_readout(r0, "start (embedded, bare-start SBS, unrelaxed)")
    run(a, big, h, st, leg, Delta, "r1", meta, r0)


def cmd_ext(a):
    """Option B: r1's start, plus one row of A per leak partner, one-hot on
    the SBS that carries its bare levels (Hungarian on the bare overlaps of
    the start's SBS columns), spectators on SBS 0."""
    leg = a.leg
    i, j = LEGS[leg]
    ite = solver(*layout(leg), leg=leg)
    st, h, Delta, meta = prepare(leg, ite)
    A, Bs, bs = (np.array(st[0]), [np.array(B) for B in st[1]],
                 st[2])
    Ops = species_ops(ite, bs, h)
    if a.clean:
        # Clean start: the two qubit SBS of each active well kept, the other
        # four replaced by bare levels 2-5 projected out of them, A zeroed
        # there. In the outer wells r1's start carries levels 3 and 4 as
        # mixtures, and the natural order then comes out 0 1 2 4 3.
        for s in (i, j):
            _e, V = bare_levels(Ops[s])
            Bs[s], used = add_spfs_bare(Bs[s][:, :2], V[:, 2:], NS_ACT - 2)
            sl = [slice(None)] * 5
            sl[1 + s] = slice(2, None)
            A[tuple(sl)] = 0.0
            print(f"  [clean] well {s}: SBS 2-5 = bare levels "
                  + ", ".join(f"{k + 2} (residual {r:.3f})" for k, r in used),
                  flush=True)
        meta.update(clean_start=True)
    inv = {}
    for s in (i, j):
        _e, V = bare_levels(Ops[s])
        O = np.abs(V.conj().T @ Bs[s]) ** 2
        r, c = linear_sum_assignment(-O.T)                  # SBS -> level
        inv[s] = {int(lv): int(q) for q, lv in zip(r, c)}
        print(f"  [ext] well {s}: start SBS -> level {[int(v) for v in c]}",
              flush=True)
    rows = []
    for (m, n) in PARTNERS:
        R = np.zeros(A.shape[1:], dtype=A.dtype)
        cfg = [0, 0, 0, 0]
        cfg[i], cfg[j] = inv[i][m], inv[j][n]
        R[tuple(cfg)] = 1.0
        rows.append(R)
    A = np.concatenate([A, np.stack(rows)], axis=0)
    assert A.shape[0] == len(ite.relax_configs)
    st = (jnp.asarray(A), [jnp.asarray(B) for B in Bs], st[2])
    r0 = readout(ite, st, h, leg, Delta)
    print_readout(r0, "start of ext (embedded, bare-start SBS, partner rows)")
    # The partner configurations are given as natural-SBS indices = levels;
    # if the start's natural order is not the level order, the weights
    # would land on the wrong roots from the first iteration.
    assert r0["natural_order_is_level_order"], r0["labels"]
    meta.update(relax_configs=[list(c) for c in ite.relax_configs],
                relax_weights=ite.relax_weights.tolist())
    name = a.name or "ext"
    run(a, ite, h, st, leg, Delta, name, meta, r0,
        base="ext" if name != "ext" else None,
        order_check=lambda x: natural_labels(ite, x, h, leg))


def cmd_variant(a):
    leg = a.leg
    x, h, Delta = gate_point(leg)
    base = a.base
    st, _d = load_state(out_path(leg, base, "npz"))
    norbs, nspfs = layout(leg)
    ext = leg if base == "ext" else None
    name = a.kind if base == "r1" else f"{base}_{a.kind}"
    if a.kind == "eps8":
        ite = solver(norbs, nspfs, eps=1e-8, leg=ext)
    elif a.kind == "x64d":
        ite = solver(norbs, nspfs, precision="x64d", leg=ext)
    elif a.kind == "random":
        ite = solver(norbs, nspfs, leg=ext)
        A_rg, Bs_rg, _e, _v, _o = ite._analyse_dev(*st, ite.as_h(h))
        bs = st[2]
        r1 = C.load_json(out_path(leg, base, "json"))
        A_rg, Bs_rg = np.array(A_rg), [np.array(B) for B in Bs_rg]
        rng = np.random.default_rng(20261021)
        for s in LEGS[leg]:
            occ = np.asarray(r1["occupations"][f"w{s}"])
            idx = [q for q in range(nspfs[s]) if occ[q] < 1e-4]
            print(f"  [random] well {s}: replacing SBS {idx} (occupations "
                  + " ".join(f"{occ[q]:.1e}" for q in idx) + ")", flush=True)
            Bs_rg[s] = replace_spfs_random(Bs_rg[s], idx, rng)
            sl = [slice(None)] * 5
            sl[1 + s] = idx
            A_rg[tuple(sl)] = 0.0
        st = (jnp.asarray(A_rg), [jnp.asarray(B) for B in Bs_rg],
              [jnp.asarray(b) for b in bs])
    else:
        raise SystemExit(a.kind)
    r0 = readout(ite, st, h, leg, Delta)
    print_readout(r0, f"start of {name}")
    run(a, ite, h, st, leg, Delta, name, dict(start=base), r0, base=base,
        order_check=(lambda x: natural_labels(ite, x, h, leg)) if ext
        else None)


def run(a, ite, h, st, leg, Delta, name, meta, r0, base=None,
        order_check=None):
    ckpt = out_path(leg, name + ".ckpt", "npz")
    print(f"\n  relaxing {name}: {ite.num_spfs} SBS, norbs "
          f"{[int(np.asarray(b).shape[1]) for b in st[2]]}, precision "
          f"{ite.precision}, eps {ite.eps}", flush=True)
    st, info = relax(ite, h, st, ckpt, order_check)
    print(f"  {name}: converged {info['converged']}, {info['steps']} it, "
          f"{info['s_per_it']:.3f} s/it, {info['wall'] / 3600:.2f} h",
          flush=True)
    save_state(out_path(leg, name, "npz"), st, run=name,
               x_ghz=gate_point(leg)[0])
    r = readout(ite, st, h, leg, Delta)
    print_readout(r, name)
    r.update(run=name, info=info, meta=meta, start_readout=r0)
    if base is not None:
        r1 = C.load_json(out_path(leg, base, "json"))
        diff = {k: r["pairs"][k]["dressed"] - r1["pairs"][k]["dressed"]
                for k in r["pairs"]}
        # every level 0-4 carried by an SBS with bare overlap > 0.5, in both
        # runs (the rank order of the ~1e-7-occupied SBS may differ)
        same = all(min(rr["level_overlap"][w]) > 0.5
                   for rr in (r, r1) for w in rr["level_overlap"])
        r["vs_base_label_order_same"] = r["labels"] == r1["labels"]
        worst = max(abs(diff[k]) for k in ("20", "02", "40", "04"))
        worst_all = max(abs(v) for v in diff.values())
        # The criterion is on |20>/|40>; for ext every column
        # comes from the run, so all six partners are held to it.
        crit = worst if base == "r1" else worst_all
        r["vs_base"] = dict(base=base, diff=diff, labels_same=same,
                            worst_2040=worst, worst_all=worst_all,
                            passed=bool(same and np.isfinite(crit)
                                        and crit < 1e-3))
        print(f"\n  vs {base}: " + " ".join(f"|{k}> {v:+.2e}" for k, v in
                                           diff.items())
              + f"; labels same {same}; worst |20>/|40> {worst:.2e} Δ, "
              f"worst all {worst_all:.2e} Δ -> "
              f"{'PASS' if r['vs_base']['passed'] else 'FAIL'}", flush=True)
    C.save_json(out_path(leg, name, "json"), r)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"  saved -> {out_path(leg, name, 'json')}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["r1", "ext", "variant"])
    ap.add_argument("--leg", default="w1w2", choices=sorted(LEGS))
    ap.add_argument("--kind", choices=["eps8", "random", "x64d"])
    ap.add_argument("--base", default="r1", choices=["r1", "ext"])
    ap.add_argument("--clean", action="store_true",
                    help="ext: clean start (qubit SBS + bare levels 2-5)")
    ap.add_argument("--name", help="ext: run name (default ext); any other "
                    "name is compared with the leg's ext run")
    a = ap.parse_args(argv)
    C.require_grid(80)
    C.L.device_banner()
    {"r1": cmd_r1, "ext": cmd_ext, "variant": cmd_variant}[a.which](a)


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1:])
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)
