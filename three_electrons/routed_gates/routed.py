"""Routed sqrt(iSWAP)(0,3) with ML-MCTDHF, 3 e-/well: sequential and
synchronized circuits.  The paper uses the modes ``pulses``, ``relax`` and
``check`` (see three_electrons/README.md); the other modes are not part of
the paper.

    python -m routed_gates.routed pulses            # pulses from the MF maps
    python -m routed_gates.routed relax             # x64d register
    python -m routed_gates.routed check             # no propagation (CPU)
    python -m routed_gates.routed pilot             # 2 t.u. idle + 2 t.u. leg
    python -m routed_gates.routed idle --T 30       # |M_ii(t)| at idle
    python -m routed_gates.routed leg --key w1w2    # one leg, 16x16 matrix
    python -m routed_gates.routed arm --arm serial  # a full arm

Built on the one-electron code ``one_electron/routed_gates/`` (driver
``scripts/run_arm_decomp.py``, conventions ``one_electron/lib/common.py``,
scoring ``one_electron/lib/run_route_sa.py``); the functions taken over
unchanged are marked.
What differs, and why:

* **Register.** The 16-state idle set of ``operating_points`` (norb 14 / n_s 4 on all
  wells, exact eigenvectors of the ML-projected idle H, relaxed with
  ``relax_weights``), re-relaxed in x64d by ``relax`` into
  ``results/register_x64d.npz``, propagated with those weights. The anchors are the
  same 16 states at t = 0, each phase-fixed so its dominant top-tensor
  element is real positive (the analogue of ``phase_fix_anchors``: product
  phases additive over bits).
* **Pulses**: each leg at its second-shortest
  sweep ramp, the hold linearly interpolated along that ramp to
  theta = pi/2. Synchronized slot = equal T, own ramps: w2w3 at its
  sequential pulse, w0w1 interpolated along its own pi/2 curve (linear in
  tau_ramp between the bracketing sweep ramps) to the same T, so neither
  pair idles. At identical (tau_ramp, tau_hold) no point has theta = pi/2 on
  both pairs.
* **Single-qubit layers are many-body operators on ONE well.** With 3 e-
  per well, |1> is the centre-of-mass excitation, which no one-body
  rotation reaches. The one-body ``U = 1 - QQ^H + Q u Q^H`` is lifted one layer:
  P_s = the well's |0>/|1> as 3-particle vectors (natural SPFs of the
  cleanest anchors), projected into the CURRENT orbitals' determinant space
  (compound minors of b_s(t)^H b_s(0)), Lowdin-orthonormalised;
  ``B_s <- U B_s``. A, the other wells and all orbitals are untouched. The
  projection retention is logged at every layer. For N = 1 this is exactly
  the one-body construction restricted to the current orbital span.
* **Integrator**: ``mf_lib.Integ()`` defaults (the sweep's). A failed leg
  stops the run; nothing is changed to get past it.
"""
import argparse
import json
import os
import time

import numpy as np

from gate_timing import mf_lib as L  # noqa: E402  (switches x64 on)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mctdhf.MLMCTDHF_n_species import MLMCTDHF_n_species_DVR  # noqa: E402
from system import fourwell as fw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PULSES_PATH = os.path.join(RES, "pulses.json")
LEGS3 = ("w0w1", "w1w2", "w2w3")
PAIRS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3)}
NORB, NSPF, NSTATES = 14, 4, 16
WIN = 10.0                    # checkpoint window inside a leg, t.u.
SNAP = 0.1                    # the block grid (Integ().block_t)
# The published schedules (Table S6); compute_pulses() must reproduce them.
EXPECTED = {"w0w1": (12.6, 98.3), "w1w2": (9.4, 126.2), "w2w3": (8.0, 131.6),
            "sync_w0w1": (32.5, 82.6), "sync_w2w3": (8.0, 131.6)}


# ── pulses ────────────────────────────────────────────────────────────────

def _map(leg):
    integ = L.Integ()
    p = os.path.join(L.RESULTS_DIR, leg, f"map_{leg}_{integ.tag()}.npz")
    z = np.load(p)
    c = list(z["columns"])
    R = z["rows"]
    out = []
    for tr in sorted(set(R[:, c.index("tau_ramp")])):
        r = R[R[:, c.index("tau_ramp")] == tr]
        r = r[np.argsort(r[:, c.index("tau_hold")])]
        out.append((float(tr), r[:, c.index("tau_hold")],
                    r[:, c.index("theta")]))
    return out, p


def _cross(h, th):
    """The hold at which theta = pi/2, by linear interpolation; exactly one
    sign change is required."""
    d = th - np.pi / 2
    k = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    assert len(k) == 1, f"{len(k)} crossings of pi/2 on one ramp"
    k = int(k[0])
    return float(h[k] - d[k] * (h[k + 1] - h[k]) / (d[k + 1] - d[k]))


def _theta_at(ramps, tr, th):
    """Map theta at (tr, th): linear along the hold grid of each ramp, then
    linear in tau_ramp between the bracketing ramps."""
    trs = [r[0] for r in ramps]
    vals = [float(np.interp(th, r[1], r[2])) for r in ramps]
    return float(np.interp(tr, trs, vals))


def compute_pulses(verbose=True):
    curves, maps, sources = {}, {}, {}
    for leg in LEGS3:
        ramps, src = _map(leg)
        maps[leg], sources[leg] = ramps, src
        curves[leg] = [(tr, _cross(h, th)) for tr, h, th in ramps]
    seq = {}
    for leg in LEGS3:
        tr, h = curves[leg][1]                         # second-shortest ramp
        seq[leg] = (round(tr, 1), round(h, 1))
    tr23, th23 = seq["w2w3"]
    T = round(2 * tr23 + th23, 6)
    T01 = [(tr, 2 * tr + h) for tr, h in curves["w0w1"]]
    k = [i for i in range(len(T01) - 1) if (T01[i][1] - T) * (T01[i + 1][1] - T) <= 0]
    assert len(k) == 1, f"equal-T bracket for w0w1 not unique: {k}"
    (ta, Ta), (tb, Tb) = T01[k[0]], T01[k[0] + 1]
    tr01 = ta + (T - Ta) / (Tb - Ta) * (tb - ta)
    tr01 = round(tr01, 1)
    th01 = round(T - 2 * tr01, 1)
    assert abs(2 * tr01 + th01 - T) < 1e-9
    p = {"seq": {k_: list(v) for k_, v in seq.items()},
         "sync": {"w0w1": [tr01, th01], "w2w3": [tr23, th23]},
         "pi2_curves": {k_: [list(v) for v in c] for k_, c in curves.items()},
         "maps": sources}
    pred = {}
    for leg in LEGS3:
        pred[leg] = _theta_at(maps[leg], *seq[leg]) / (np.pi / 2)
    pred["sync_w0w1"] = _theta_at(maps["w0w1"], tr01, th01) / (np.pi / 2)
    p["predicted_theta_over_pi2"] = pred
    got = {**{k_: tuple(v) for k_, v in seq.items()},
           "sync_w0w1": (tr01, th01), "sync_w2w3": (tr23, th23)}
    assert got == EXPECTED, f"pulses {got} != approved {EXPECTED}"
    if verbose:
        for leg in LEGS3:
            tr, h = seq[leg]
            print(f"  {leg}: tau_ramp {tr}, tau_hold {h}, T {2 * tr + h:.1f}; "
                  f"map theta/(pi/2) at the snapped pulse {pred[leg]:.4f}",
                  flush=True)
        print(f"  sync slot T {T}: w0w1 ({tr01}, {th01}) [bracket ramps {ta}, "
              f"{tb}; map theta/(pi/2) {pred['sync_w0w1']:.4f}], w2w3 "
              f"({tr23}, {th23})", flush=True)
    return p


def load_pulses():
    p = compute_pulses(verbose=False)
    assert os.path.exists(PULSES_PATH), f"{PULSES_PATH} missing: run `pulses`"
    with open(PULSES_PATH) as fh:
        q = json.load(fh)
    assert q["seq"] == p["seq"] and q["sync"] == p["sync"], (
        "pulses.json differs from the maps")
    return p


def leg_specs(p):
    """key -> [(leg, tau_ramp, tau_hold), ...] (two entries = concurrent)."""
    s = {k: [(k, *p["seq"][k])] for k in LEGS3}
    s["sync"] = [("w0w1", *p["sync"]["w0w1"]), ("w2w3", *p["sync"]["w2w3"])]
    return s


def leg_T(spec):
    Ts = {round(2 * tr + th, 6) for _, tr, th in spec}
    assert len(Ts) == 1, f"concurrent legs of unequal length {Ts}"
    T = Ts.pop()
    n = round(T / SNAP)
    assert abs(n * SNAP - T) < 1e-9, f"T = {T} not on the {SNAP} grid"
    return T


def h_list_for(spec):
    xi = L.x_vectors(spec[0][0])[0] / fw.TYPICAL_FREQ_GHZ
    hs = list(fw.h_ramp(xi, xi, L.make_env(1.0, 1.0, "idle")[0]))
    for leg, tr, th in spec:
        xg = L.x_vectors(leg)[1] / fw.TYPICAL_FREQ_GHZ
        env, _ = L.make_env(tr, th)
        hl = fw.h_ramp(xi, xg, env)
        for w in L.LEGS[leg]:
            hs[w] = hl[w]
    return hs


def h_idle_list():
    xi = L.x_vectors("w1w2")[0] / fw.TYPICAL_FREQ_GHZ
    return list(fw.h_ramp(xi, xi, L.make_env(1.0, 1.0, "idle")[0]))


def probe_h(h_list, t):
    return np.array([float(jnp.sum(jnp.abs(h(jnp.asarray(t))))) for h in h_list])


def check_fingerprints(specs):
    fp = {k: probe_h(h_list_for(s), 0.35 * leg_T(s)) for k, s in specs.items()}
    ks = sorted(fp)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            assert not np.allclose(fp[ks[i]], fp[ks[j]], rtol=0, atol=1e-9), (
                f"legs {ks[i]} and {ks[j]} have identical Hamiltonians")
    return fp


# ── register, anchors, solver ─────────────────────────────────────────────

REGISTER_PATH = os.path.join(RES, "register_x64d.npz")


def relax_register(inner_tol=1e-4):
    """Re-relax the operating_points 16-state idle register in x64d at its own
    weights (pilot runs: the operating_points set, relaxed in mixed
    precision, loses 5.4e-4 of |M_ii| in 2 t.u. of real-time idle). Same
    recipe as ``exact_check.relax``: ITE_opt from the given state, top block
    re-diagonalised and roots matched by identity, exact regauge."""
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt
    from mctdhf.MLMCTDHF_n_species import make_regauge_ml
    d = np.load(L.IDLE_PATH)
    cfgs = [tuple(int(x) for x in c) for c in d["configs"]]
    w = np.asarray(d["relax_weights"], dtype=np.float64)
    xi = L.x_vectors("w1w2")[0] / fw.TYPICAL_FREQ_GHZ
    h_idle = fw.h_static(xi)
    ite = MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(), num_particles=[fw.NUM_PARTICLES] * 4,
        num_mctdhf_orbitals=[NORB] * 4, num_spatial_orbitals=[fw.NUM_DVR] * 4,
        num_spfs=[NSPF] * 4, relax_configs=cfgs, relax_weights=list(w),
        all_configs=cfgs, dt=2e-2, precision="x64d", inner_tol=inner_tol,
        chunk=50, eps=1e-6)
    ml = make_ml16(h_idle_list(), w / w.sum())
    A = jnp.asarray(d["A"], dtype=jnp.complex128)
    Bs = [jnp.asarray(d[f"B{s}"], dtype=jnp.complex128) for s in range(4)]
    bs = [jnp.asarray(d[f"b{s}"], dtype=jnp.complex128) for s in range(4)]
    E0 = np.real(np.asarray(ml.calculate_energy(0.0, ml._ABb_to_y(A, Bs, bs))))
    t0 = time.time()
    # Stopping rule: the summed energy changes by
    # < 1e-10 between consecutive CHUNK ends (50 iterations), the rule
    # operating_points's polish used. ITE_opt.relax_to_tol's per-iteration rule cannot
    # be met here: a pilot settled into a period-2 cycle (consecutive
    # iterations alternate by 5.2e-7, chunk ends agree to 1e-10).
    state, steps, fallback, e_prev, converged = (A, Bs, bs), 0, [], None, False
    info = {}
    while steps < 40000:
        state, e, info = ite.integrate(ite.chunk, h_idle, state)
        fallback += [steps // ite.chunk] if info["fallback_chunks"] else []
        steps += len(e)
        if not np.all(np.isfinite(e)):
            break
        dch = np.nan if e_prev is None else e[-1] - e_prev
        print(f"    it {steps:6d}  sum E = {e[-1]:.10f}  dE(chunk) = {dch:+.3e}"
              f"  dE(last it) = {e[-1] - e[-2]:+.3e}", flush=True)
        if e_prev is not None and abs(dch) < 1e-10:
            converged = True
            break
        e_prev = e[-1]
    info = dict(info, converged=converged, steps=steps, fallback_chunks=fallback)
    h = ite.as_h(h_idle)
    A, Bs, bs = state
    Bs_rg, evals, evecs, ovl = ite._front_match(A, Bs, bs, h)
    cols, pop = ite._assign(ovl)
    A = evecs[:, jnp.asarray(cols)].T.reshape(ml.a_shape)
    y = make_regauge_ml(ml)(ml._ABb_to_y(A, list(Bs_rg), list(bs)))
    E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    err = float(np.max(np.abs(np.asarray(ml.calculate_overlap(y, y)) - np.eye(16))))
    print(f"  [relax] converged {info['converged']} after {info['steps']} it "
          f"({time.time() - t0:.0f} s), fallbacks {info['fallback_chunks']}",
          flush=True)
    print("  [relax] match pop " + " ".join(f"{v:.6f}" for v in np.asarray(pop)),
          flush=True)
    print("  [relax] E - E_start " + " ".join(f"{v:+.2e}" for v in E - E0)
          + f"   M(0)-1 {err:.1e}", flush=True)
    A, Bs, bs = ml._y_to_ABb(y)
    L.save_npz(REGISTER_PATH, A=np.asarray(A),
               **{f"B{s}": np.asarray(Bs[s]) for s in range(4)},
               **{f"b{s}": np.asarray(bs[s]) for s in range(4)},
               E=E, E_start=E0, converged=bool(info["converged"]),
               steps=int(info["steps"]), inner_tol=inner_tol,
               match_pop=np.asarray(pop), source=L.IDLE_PATH,
               configs=np.asarray(cfgs), relax_weights=w)
    print(f"  -> {REGISTER_PATH}", flush=True)
    # saved first (always keep the state), judged after
    assert info["converged"] and np.all(np.isfinite(E)), (
        f"relaxation failed: converged {info['converged']}, E {E}")
    assert np.min(np.asarray(pop)) > 0.99, f"root matching failed: {pop}"
    assert err < 1e-10, f"relaxed M(0) != 1: {err:.3e}"


def load_register():
    """The x64d-relaxed register (``relax`` mode); configs and weights are
    the operating_points idle file's."""
    d0 = np.load(L.IDLE_PATH)
    assert os.path.exists(REGISTER_PATH), f"{REGISTER_PATH} missing: run `relax`"
    d = np.load(REGISTER_PATH)
    assert np.array_equal(d["configs"], d0["configs"])
    assert np.array_equal(d["relax_weights"], d0["relax_weights"])
    A = np.asarray(d["A"]).copy()
    Bs = [np.asarray(d[f"B{s}"]) for s in range(4)]
    bs = [np.asarray(d[f"b{s}"]) for s in range(4)]
    configs = [tuple(int(x) for x in c) for c in d["configs"]]
    w = np.asarray(d["relax_weights"], dtype=np.float64)
    assert (int(d0["norb"]), int(d0["n_spf"])) == (NORB, NSPF)
    assert A.shape == (NSTATES,) + (NSPF,) * 4
    ph_before = []
    for a, c in enumerate(configs):
        k = np.unravel_index(np.argmax(np.abs(A[a])), A[a].shape)
        assert tuple(int(x) for x in k) == c, (
            f"row {a}: dominant SPF index {k} != config {c}")
        ph_before.append(float(np.degrees(np.angle(A[a][k]))))
        A[a] *= np.exp(-1j * np.angle(A[a][k]))
    return dict(A=A, Bs=Bs, bs=bs, configs=configs, weights=w / w.sum(),
                phases_before_deg=ph_before)


def make_ml16(h_list, weights):
    ml = MLMCTDHF_n_species_DVR(
        h_list=h_list, g_dict=fw.g_dict(),
        num_particles=[fw.NUM_PARTICLES] * 4,
        num_mctdhf_orbitals=[NORB] * 4,
        num_spatial_orbitals=[fw.NUM_DVR] * 4,
        num_spfs=[NSPF] * 4, num_states=NSTATES, imag_time=False, eps=1e-6,
        e_shift=0.0, state_weights=weights, ci_sparse=True)
    assert ml.e_shift == 0.0
    ml.state_weights = jnp.asarray(weights)     # float64, as L.make_ml
    return ml


def y_of(ml, A, Bs, bs):
    return ml._ABb_to_y(jnp.asarray(A, dtype=jnp.complex128),
                        [jnp.asarray(B, dtype=jnp.complex128) for B in Bs],
                        [jnp.asarray(b, dtype=jnp.complex128) for b in bs])


def bit_perm(configs):
    bit = np.array([(c[0] << 3) | (c[1] << 2) | (c[2] << 1) | c[3]
                    for c in configs])
    P = np.zeros((16, 16))
    for q in range(16):
        P[bit[q], q] = 1.0
    return P


def M16_of(ml, y0, y, P):
    """``M[i, j] = <anchor_i|psi_j>`` in BIT ORDER (q0 most significant)."""
    M = np.asarray(ml.calculate_overlap(y0, y))
    return P @ M @ P.T


def norms_of(ml, y):
    return np.abs(np.diag(np.asarray(ml.calculate_overlap(y, y))))


# ── qubit states and the many-body single-well layer ──────────────────────

def det_occ(ml, s):
    return np.array([np.flatnonzero(d) for d in np.asarray(ml.dCI.I[s])])


def det_overlap(b1, b2, occ):
    """``S[K, I] = <Phi_K(b1)|Phi_I(b2)> = det((b1^H b2)[occ_K, occ_I])``,
    the same minors as ``_det_overlap_matrices``."""
    O = np.asarray(b1).conj().T @ np.asarray(b2)
    sub = O[occ[:, None, :, None], occ[None, :, None, :]]
    return np.linalg.det(sub)


def qubit_states(reg):
    """P_s = (|0>_s, |1>_s) as determinant-space vectors in the t = 0
    orbitals: the natural SPF of well s in the cleanest anchors (all other
    wells in |0>), phase-aligned to B_s columns 0/1, QR. Plus the state
    dependence over all 16 anchors (as ``qubit_basis16`` of run_arm_decomp)."""
    A, Bs, cfg = reg["A"], reg["Bs"], reg["configs"]

    def nat(a, s):
        rho = L._node_density(A[a:a + 1], s + 1, np.ones(1))
        ev, V = np.linalg.eigh(rho)
        return V[:, -1], ev[-1] / ev.sum()

    P, info = [], []
    for s in range(4):
        zero = (0, 0, 0, 0)
        one = tuple(1 if k == s else 0 for k in range(4))
        cols, occs = [], []
        for n, c in enumerate((zero, one)):
            v, o = nat(cfg.index(c), s)
            v = v * np.exp(-1j * np.angle(v[n]))
            cols.append(Bs[s] @ v)
            occs.append(o)
        Q, _ = np.linalg.qr(np.stack(cols, 1))
        for n in range(2):
            Q[:, n] *= np.exp(-1j * np.angle(np.vdot(Bs[s][:, n], Q[:, n])))
        dev = 0.0
        for a, c in enumerate(cfg):
            v, _ = nat(a, s)
            q = Bs[s] @ v
            dev = max(dev, float(np.sqrt(max(0.0, 1 - abs(np.vdot(Q[:, c[s]], q)) ** 2))))
        P.append(Q)
        info.append(dict(nat_occ=occs, overlap_spf=[abs(np.vdot(Bs[s][:, n], Q[:, n]))
                                                     for n in range(2)],
                         state_dependence=dev))
    return P, info


def apply_layer(ml, y, assign, P0, b0, occ):
    """Many-body single-well layers ``{well: 2x2}``: ``B_s <- U B_s`` with
    ``U = 1 - P P^H + P u P^H`` on the CURRENT determinant space. Returns
    the new y and, per well, the singular values^2 of the projected P (the
    retention of |0>/|1> in the current orbital space)."""
    A, Bs, bs = ml._y_to_ABb(y)
    Bs = list(Bs)
    ret = {}
    for s, u in assign.items():
        S = det_overlap(bs[s], b0[s], occ[s])
        Pt = S @ P0[s]
        sv = np.linalg.svd(Pt, compute_uv=False)
        ret[s] = (sv ** 2).tolist()
        G = Pt.conj().T @ Pt
        ev, V = np.linalg.eigh(G)
        Ph = Pt @ (V @ np.diag(ev ** -0.5) @ V.conj().T)
        U = np.eye(Ph.shape[0], dtype=complex) - Ph @ Ph.conj().T \
            + Ph @ np.asarray(u) @ Ph.conj().T
        Bs[s] = jnp.asarray(U) @ Bs[s]
    return ml._ABb_to_y(A, Bs, list(bs)), ret


def onebody_Q(ml, reg, s, conj=False):
    """The orbitals of the one-body construction for well s: the orbital that
    empties and the one that fills between |0>_s and |1>_s (eigenvectors of
    D(|1_s>) - D(|0>) at its extreme eigenvalues), in the DVR basis."""
    A, Bs, bs, cfg = reg["A"], reg["Bs"], reg["bs"], reg["configs"]
    one = tuple(1 if k == s else 0 for k in range(4))
    D = []
    for c in ((0, 0, 0, 0), one):
        a = cfg.index(c)
        Ds = ml.dCI.ml_1p_RDMs(jnp.asarray(A[a:a + 1]),
                               [jnp.asarray(B) for B in Bs],
                               weights=jnp.ones(1))
        D.append(np.asarray(Ds[s]))
    ev, V = np.linalg.eigh(D[1] - D[0])
    if conj:
        V = V.conj()
    Q = np.stack([bs[s] @ V[:, 0], bs[s] @ V[:, -1]], 1)
    Q, _ = np.linalg.qr(Q)
    return Q, ev


def apply_layer_onebody(ml, y, assign, Q):
    A, Bs, bs = ml._y_to_ABb(y)
    bs = list(bs)
    for s, u in assign.items():
        U = np.eye(fw.NUM_DVR, dtype=complex) - Q[s] @ Q[s].conj().T \
            + Q[s] @ np.asarray(u) @ Q[s].conj().T
        bs[s] = jnp.asarray(U) @ bs[s]
    return ml._ABb_to_y(A, list(Bs), bs)


# ── unchanged from one_electron/lib/common.py ──────────────────────

_S2 = 1.0 / np.sqrt(2.0)
SQI = np.array([[1, 0, 0, 0], [0, _S2, 1j * _S2, 0],
                [0, 1j * _S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
HAD = np.array([[1, 1], [1, -1]], dtype=complex) * _S2


def _z4(a, b):
    return np.diag([1.0, np.exp(1j * a), np.exp(1j * b), np.exp(1j * (a + b))])


def strip_localZ(M, M_ref=SQI, ntries=40, seed=0):
    from scipy.optimize import minimize
    M = np.asarray(M, dtype=complex)
    nrm = np.linalg.norm(M_ref) * np.linalg.norm(M)

    def negov(p):
        return -abs(np.vdot(M_ref, _z4(p[0], p[1]) @ M @ _z4(p[2], p[3]))) / nrm

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(ntries):
        r = minimize(negov, rng.uniform(0, 2 * np.pi, 4), method="BFGS",
                     options=dict(maxiter=5000, gtol=1e-14))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    Ms = _z4(p[0], p[1]) @ M @ _z4(p[2], p[3])
    return Ms, p, abs(np.vdot(M_ref, M)) / nrm, -best.fun


def schmidt(M):
    M = np.asarray(M, dtype=complex)
    return np.linalg.svd(M.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4),
                         compute_uv=False)


# ── unchanged from one_electron/lib/run_route_sa.py ────────────────

def target_sqiswap_03():
    U = np.zeros((16, 16), complex)
    for i in range(16):
        q = [(i >> (3 - k)) & 1 for k in range(4)]
        for li in range(4):
            q0o, q3o = (li >> 1) & 1, li & 1
            amp = SQI[2*q[0] + q[3], 2*q0o + q3o]
            if amp != 0:
                o = (q0o << 3) | (q[1] << 2) | (q[2] << 1) | q3o
                U[o, i] = amp
    return U


def _zphases(thetas):
    bits = np.array([[(i >> (3 - k)) & 1 for k in range(4)] for i in range(16)])
    return np.exp(1j * (bits @ np.asarray(thetas)))


def fxc_4q(U_ref, M, restarts=40, seed=0):
    from scipy.optimize import minimize
    if not np.all(np.isfinite(M)):
        return np.nan
    nref = np.vdot(U_ref, U_ref).real; nm = np.vdot(M, M).real
    Uc = U_ref.conj().T

    def neg(x):
        zl = _zphases(x[:4]); zr = _zphases(x[4:8])
        t = np.trace(Uc @ ((zl[:, None] * M) * zr[None, :]))
        return -(abs(t) ** 2) / (nref * nm)
    rng = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        r = minimize(neg, rng.uniform(0, 2*np.pi, 8), method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-11, maxiter=8000))
        if best is None or r.fun < best.fun:
            best = r
    return float(-best.fun)


# ── unchanged from one_electron/routed_gates/scripts/run_arm_decomp.py ──

def corner4(M16, pair):
    a, b = pair
    idx = []
    for j in range(4):
        na, nb = j & 1, (j >> 1) & 1
        bits = [0, 0, 0, 0]
        bits[a], bits[b] = na, nb
        idx.append((bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3])
    return M16[np.ix_(idx, idx)]


SDG = np.diag([1.0, -1j]).astype(complex)        # S^dagger
BLOCK_U2 = HAD @ SDG                             # the per-qubit half of B's layer


def block_layer(pair):
    return {pair[0]: BLOCK_U2.copy(), pair[1]: BLOCK_U2.copy()}


def check_block():
    Lk = np.kron(BLOCK_U2, BLOCK_U2)
    B = SQI @ Lk
    return float(np.abs(B @ B @ B - np.exp(-1j * np.pi / 4) * SWAP).max())


def zframe(p, pair, side):
    a, b = (p[0], p[1]) if side == "after" else (p[2], p[3])
    return {pair[0]: np.diag([1.0, np.exp(1j * a)]).astype(complex),
            pair[1]: np.diag([1.0, np.exp(1j * b)]).astype(complex)}


def layer16_ideal(assign):
    U = np.eye(1, dtype=complex)
    for s in range(4):
        U = np.kron(U, assign.get(s, np.eye(2, dtype=complex)))
    return U


def swap_block(key, pair, z):
    return [("layer", block_layer(pair)),
            ("zframe", zframe(z, pair, "before")),
            ("leg", key),
            ("zframe", zframe(z, pair, "after"))]


def entangler(key, pair, z):
    return [("zframe", zframe(z, pair, "before")),
            ("leg", key),
            ("zframe", zframe(z, pair, "after"))]


def par_block(z01, z23, key="sync"):
    """One concurrent B on (0,1) and (2,3) -- one four-well layer (as
    ``par_block`` of the one-electron code; the slot key is "sync" here)."""
    return [("layer", {**block_layer((0, 1)), **block_layer((2, 3))}),
            ("zframe", {**zframe(z01, (0, 1), "before"),
                        **zframe(z23, (2, 3), "before")}),
            ("leg", key),
            ("zframe", {**zframe(z01, (0, 1), "after"),
                        **zframe(z23, (2, 3), "after")})]


def coalesce(sched):
    out = []
    for kind, val in sched:
        if kind in ("layer", "zframe") and out and out[-1][0] == "local":
            merged = dict(out[-1][1])
            for w, u in val.items():
                merged[w] = u @ merged[w] if w in merged else u
            out[-1] = ("local", merged)
        elif kind in ("layer", "zframe"):
            out.append(("local", dict(val)))
        else:
            out.append((kind, val))
    return out


def serial_schedule(z):
    sched = []
    for key, pair in (("w2w3", (2, 3)), ("w1w2", (1, 2))):
        for _ in range(3):
            sched += swap_block(key, pair, z[key])
    sched += entangler("w0w1", (0, 1), z["w0w1"])
    for key, pair in (("w1w2", (1, 2)), ("w2w3", (2, 3))):
        for _ in range(3):
            sched += swap_block(key, pair, z[key])
    return sched


def parallel_schedule(z01, z23, zent):
    sched = []
    for _ in range(3):
        sched += par_block(z01, z23)
    sched += entangler("w1w2", (1, 2), zent)
    for _ in range(3):
        sched += par_block(z01, z23)
    return sched


def _score_matrix(M):
    U = target_sqiswap_03()
    fxc = fxc_4q(U, M)
    d = 16
    Fbar = (np.vdot(M, M).real
            + fxc * np.vdot(U, U).real * np.vdot(M, M).real) / (d * (d + 1))
    sub = [i for i in range(16) if ((i >> 2) & 1) == 0 and ((i >> 1) & 1) == 0]
    return Fbar, fxc, M[np.ix_(sub, sub)]

# ── end of the unchanged parts ────────────────────────────────────────────


def pair_gate16(G4, pair):
    """A 4x4 pair gate (basis index n_a + 2 n_b) embedded in bit order."""
    a, b = pair
    U = np.zeros((16, 16), complex)
    for i in range(16):
        bits = [(i >> (3 - k)) & 1 for k in range(4)]
        li = bits[a] + 2 * bits[b]
        for lo in range(4):
            if G4[lo, li] != 0:
                ob = list(bits)
                ob[a], ob[b] = lo & 1, (lo >> 1) & 1
                o = (ob[0] << 3) | (ob[1] << 2) | (ob[2] << 1) | ob[3]
                U[o, i] += G4[lo, li]
    return U


IDEAL_LEG = {"w0w1": pair_gate16(SQI, (0, 1)), "w1w2": pair_gate16(SQI, (1, 2)),
             "w2w3": pair_gate16(SQI, (2, 3)),
             "sync": pair_gate16(SQI, (0, 1)) @ pair_gate16(SQI, (2, 3))}


def schedule_for(arm, z):
    if arm == "serial":
        return serial_schedule(z)
    if arm == "sync":
        return parallel_schedule(z["sync01"], z["sync23"], z["w1w2"])
    raise ValueError(arm)


ARM_LEGS = {"serial": ("w0w1", "w1w2", "w2w3"), "sync": ("sync", "w1w2")}


def raw_fbar(M_ideal, M):
    d = M.shape[0]
    return float((np.vdot(M, M).real
                  + abs(np.trace(M_ideal.conj().T @ M)) ** 2) / (d * (d + 1)))


def theta_of(c4):
    Mn = c4 / np.sqrt(np.sum(np.abs(c4) ** 2, axis=0))[None, :]
    return float(2.0 * np.arctan2(abs(Mn[2, 1]), abs(Mn[1, 1])))


# ── propagation with resumable windows ────────────────────────────────────

class State:
    """One atomically rewritten checkpoint per run: y, the schedule position,
    the time reached inside the current leg, and the log."""

    def __init__(self, path, meta):
        self.path, self.meta = path, meta

    def load(self):
        if not os.path.exists(self.path):
            return None
        d = np.load(self.path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        assert meta == self.meta, (
            f"{self.path} was written with different settings:\n{meta}\n"
            f"vs\n{self.meta}")
        return dict(y=jnp.asarray(d["y"]), step=int(d["step"]),
                    t_leg=float(d["t_leg"]), log=json.loads(str(d["log"])))

    def save(self, y, step, t_leg, log):
        L.save_npz(self.path, y=np.asarray(y), step=step, t_leg=t_leg,
                   log=json.dumps(log), meta=json.dumps(self.meta))


def run_leg(ml, prop, y, T, t_start, label, on_window):
    """Propagate one leg from ``t_start`` to ``T`` in WIN windows on the
    global step grid; ``on_window(t, y, info)`` after each. Status not ok
    -> SystemExit (nothing is scored from a failed leg)."""
    t0 = t_start
    sps = []
    while t0 < T - 1e-9:
        t1 = round(min(t0 + WIN, T), 6)
        nblk = round((t1 - t0) / SNAP)
        y, status, _, s, _ = prop.run(y, t0, t1, verbose=False,
                                      report_every=max(1, nblk))
        if status != "ok":
            raise SystemExit(f"LEG FAILED [{label}] {status} in window "
                             f"({t0}, {t1}]; stopping, settings unchanged")
        n = norms_of(ml, y)
        info = dict(t=t1, max_norm_dev=float(np.max(np.abs(n - 1))),
                    ms_per_step=float(s * 1e3))
        sps.append(s)
        print(f"    [{label}] t = {t1:8.1f}/{T}  {s * 1e3:7.1f} ms/step  "
              f"max|n-1| {info['max_norm_dev']:.2e}", flush=True)
        on_window(t1, y, info)
        t0 = t1
    return y, (float(np.nanmedian(sps)) if sps else np.nan)


class Legs:
    """Lazily built solver + propagator per leg key (setup is minutes on
    the GPU, so a leg met again reuses its compiled step)."""

    def __init__(self, specs, weights, integ):
        self.specs, self.w, self.integ, self.c = specs, weights, integ, {}

    def get(self, key):
        if key not in self.c:
            t = time.time()
            ml = make_ml16(h_list_for(self.specs[key]), self.w)
            self.c[key] = (ml, L.Propagator(ml, self.integ))
            print(f"  [setup] {key}: {self.specs[key]}, T {leg_T(self.specs[key])} "
                  f"({time.time() - t:.0f} s)", flush=True)
        return self.c[key]


# ── modes ─────────────────────────────────────────────────────────────────

def mode_relax(a):
    L.device_banner()
    relax_register()


def mode_idle(a):
    """Idle diagnostic: the register at idle for
    ``--T`` t.u., |M_ii(t)| per state every 0.5 t.u. -- bounded wobble or a
    secular loss, and which states carry it."""
    p, specs, reg = _setup()
    L.device_banner()
    integ = L.Integ()
    P = bit_perm(reg["configs"])
    ml = make_ml16(h_idle_list(), reg["weights"])
    y0 = y_of(ml, reg["A"], reg["Bs"], reg["bs"])
    prop = L.Propagator(ml, integ)
    ts, D, off, y, t0 = [], [], [], y0, 0.0
    lab = ["".join(map(str, c)) for c in reg["configs"]]
    while t0 < a.T - 1e-9:
        t1 = round(t0 + 0.5, 6)
        y, status, _, sps, _ = prop.run(y, t0, t1, verbose=False, report_every=5)
        if status != "ok":
            raise SystemExit(f"IDLE: {status}")
        M = np.asarray(ml.calculate_overlap(y0, y))        # config order
        d = np.abs(np.diag(M))
        ts.append(t1)
        D.append(d)
        off.append(float(np.max(np.abs(M - np.diag(np.diag(M))))))
        k = int(np.argmin(d))
        print(f"  t = {t1:6.1f}  min|M_ii| {d[k]:.6f} ({lab[k]})  mean "
              f"{d.mean():.6f}  max offdiag {off[-1]:.2e}  {sps * 1e3:.0f} ms/step",
              flush=True)
        if round(t1 / 5.0, 6) == round(t1 / 5.0):
            print("    1-|M_ii|: " + " ".join(f"{l}:{1 - v:.1e}"
                                              for l, v in zip(lab, d)), flush=True)
        t0 = t1
    L.save_npz(os.path.join(RES, "idle_diag.npz"), t=np.asarray(ts),
               absMii=np.asarray(D), max_offdiag=np.asarray(off),
               configs=np.asarray(reg["configs"]), integ=json.dumps(integ.as_dict()))
    print(f"  -> {os.path.join(RES, 'idle_diag.npz')}", flush=True)


def mode_pulses(a):
    p = compute_pulses()
    os.makedirs(RES, exist_ok=True)
    with open(PULSES_PATH, "w") as fh:
        json.dump(p, fh, indent=1)
    print(f"  -> {PULSES_PATH}", flush=True)


def _setup(verbose=True):
    p = load_pulses()
    specs = leg_specs(p)
    reg = load_register()
    if verbose:
        print("  anchor dominant-element phases before the fix (deg): "
              + " ".join(f"{v:.0f}" for v in reg["phases_before_deg"]), flush=True)
    return p, specs, reg


def mode_check(a):
    p, specs, reg = _setup()
    integ = L.Integ()
    print(f"  integ {integ.tag()}", flush=True)
    ml = make_ml16(h_idle_list(), reg["weights"])
    y0 = y_of(ml, reg["A"], reg["Bs"], reg["bs"])
    P = bit_perm(reg["configs"])
    occ = [det_occ(ml, s) for s in range(4)]
    b0 = reg["bs"]
    fails = []

    def gate(name, ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
        if not ok:
            fails.append(name)

    M0 = M16_of(ml, y0, y0, P)
    e = float(np.max(np.abs(M0 - np.eye(16))))
    gate("M(0) = 1", e < 1e-10, f"max|M(0) - 1| = {e:.2e}")

    r = check_block()
    gate("B^3 = e^{-i pi/4} SWAP", r < 1e-12, f"residual {r:.2e}")

    # minors against the library's determinant overlaps (species 0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(NORB, NORB)) + 1j * rng.normal(size=(NORB, NORB))
    Uq, _ = np.linalg.qr(X)
    b_rot = b0[0] @ Uq
    S_np = det_overlap(b_rot, b0[0], occ[0])
    S_lib = np.asarray(ml._det_overlap_matrices(
        [jnp.asarray(b_rot)] + [jnp.asarray(b) for b in b0[1:]],
        [jnp.asarray(b) for b in b0])[0])
    e = float(np.max(np.abs(S_np - S_lib)))
    gate("minors vs _det_overlap_matrices", e < 1e-12, f"max|dS| = {e:.2e}")

    # ideal composition and counts
    U03 = target_sqiswap_03()
    zero = np.zeros(4)
    z0 = {k: zero for k in ("w0w1", "w1w2", "w2w3", "sync01", "sync23")}
    for arm, n_legs_exp, T_exp in (("serial", 13, 1879.1), ("sync", 7, 1030.6)):
        sched = schedule_for(arm, z0)
        n_legs = sum(1 for k, _ in sched if k == "leg")
        n_lay = sum(1 for k, _ in sched if k == "layer")
        T_tot = sum(leg_T(specs[v]) for k, v in sched if k == "leg")
        Mi = np.eye(16, dtype=complex)
        for kind, val in coalesce(sched):
            Mi = (layer16_ideal(val) if kind == "local" else IDEAL_LEG[val]) @ Mi
        ov = abs(np.vdot(U03, Mi)) / (np.linalg.norm(U03) * np.linalg.norm(Mi))
        gate(f"{arm}: ideal composition = sqrt(iSWAP)(0,3) up to global phase",
             abs(ov - 1) < 1e-12, f"|<U,P>|/norms = {ov:.15f}")
        gate(f"{arm}: counts and duration",
             n_legs == n_legs_exp and abs(T_tot - T_exp) < 1e-6,
             f"{n_legs} legs, {n_lay} layers, T = {T_tot:.1f} t.u.")

    fp = check_fingerprints(specs)
    gate("distinct leg Hamiltonians", True,
         "; ".join(f"{k} {np.round(v, 3).tolist()}" for k, v in fp.items()))

    # qubit states
    P0, qinfo = qubit_states(reg)
    for s, q in enumerate(qinfo):
        print(f"  well {s} qubit states: natural-SPF occ {q['nat_occ'][0]:.6f}/"
              f"{q['nat_occ'][1]:.6f}, |<B[:,n]|P[:,n]>| "
              f"{q['overlap_spf'][0]:.6f}/{q['overlap_spf'][1]:.6f}, "
              f"state dependence {q['state_dependence']:.1e}", flush=True)

    # layers at t = 0
    X2 = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    yI, ret = apply_layer(ml, y0, {s: I2 for s in range(4)}, P0, b0, occ)
    e = float(np.max(np.abs(np.asarray(yI) - np.asarray(y0))))
    gate("identity layer", e < 1e-12, f"max|dy| = {e:.2e}, retention "
         f"{min(min(v) for v in ret.values()):.15f}")
    out = {}
    for s in range(4):
        yx, _ = apply_layer(ml, y0, {s: X2}, P0, b0, occ)
        yxx, _ = apply_layer(ml, yx, {s: X2}, P0, b0, occ)
        e = float(np.max(np.abs(np.asarray(yxx) - np.asarray(y0))))
        Lx = M16_of(ml, y0, yx, P)
        flip = 1 << (3 - s)
        ok_bits = all(int(np.argmax(np.abs(Lx[:, j]))) == (j ^ flip)
                      for j in range(16))
        gate(f"X on well {s}: X.X = 1 and bit order", e < 1e-12 and ok_bits,
             f"max|dy| {e:.2e}; dominant output = input XOR {flip:04b}: {ok_bits}")
        yh, _ = apply_layer(ml, y0, {s: HAD}, P0, b0, occ)
        Lh = M16_of(ml, y0, yh, P)
        Fh = raw_fbar(layer16_ideal({s: HAD}), Lh)
        # the one-body construction, best of conventions and signs
        best = None
        for cj in (False, True):
            Q, ev = onebody_Q(ml, reg, s, conj=cj)
            for sg in (1.0, -1.0):
                Qs = Q * np.array([1.0, sg])[None, :]
                y1 = apply_layer_onebody(ml, y0, {s: HAD}, {s: Qs})
                F1 = raw_fbar(layer16_ideal({s: HAD}), M16_of(ml, y0, y1, P))
                if best is None or F1 > best[0]:
                    best = (F1, cj, sg, ev)
        out[f"H_w{s}"] = dict(F_manybody=Fh, norm2_manybody=float(np.linalg.norm(Lh) ** 2),
                              F_onebody=best[0], dD_ev=best[3].tolist())
        print(f"  H on well {s}: many-body F = {Fh:.6f} (||L||^2 "
              f"{np.linalg.norm(Lh) ** 2:.5f});  one-body F = {best[0]:.6f} "
              f"(conj {best[1]}, sign {best[2]:+.0f}; dD eigenvalues "
              f"{best[3][0]:+.4f} ... {best[3][-1]:+.4f})", flush=True)
    for pair in ((0, 1), (1, 2), (2, 3)):
        yb, _ = apply_layer(ml, y0, block_layer(pair), P0, b0, occ)
        Lb = M16_of(ml, y0, yb, P)
        Fb = raw_fbar(layer16_ideal(block_layer(pair)), Lb)
        sch = schmidt(corner4(Lb, pair))
        print(f"  block layer H.Sdg on {pair}: many-body F = {Fb:.6f}, corner "
              f"Schmidt {np.round(sch, 5).tolist()}", flush=True)
        out[f"block_{pair[0]}{pair[1]}"] = dict(F=Fb, schmidt=sch.tolist())
    L.save_npz(os.path.join(RES, "check.npz"), summary=json.dumps(out),
               fails=json.dumps(fails))
    if fails:
        raise SystemExit(f"CHECK FAILED: {fails}")
    print("\n  all checks PASS", flush=True)


def _mem():
    try:
        st = jax.devices()[0].memory_stats() or {}
        return f"peak device memory {st.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB"
    except Exception:  # noqa: BLE001
        return "memory stats n/a"


def mode_pilot(a):
    p, specs, reg = _setup()
    L.device_banner()
    integ = L.Integ()
    P = bit_perm(reg["configs"])
    out = {}
    for label, hl, T in (("idle", h_idle_list(), 2.0),
                         ("w1w2", h_list_for(specs["w1w2"]), 2.0)):
        t = time.time()
        ml = make_ml16(hl, reg["weights"])
        y0 = y_of(ml, reg["A"], reg["Bs"], reg["bs"])
        prop = L.Propagator(ml, integ)
        y, status, _, sps, setup = prop.run(y0, 0.0, T, verbose=True,
                                            report_every=5)
        M = M16_of(ml, y0, y, P)
        n = norms_of(ml, y)
        out[label] = dict(status=status, ms_per_step=sps * 1e3, setup_s=setup,
                          wall=time.time() - t, min_absMii=float(np.min(np.abs(np.diag(M)))),
                          max_offdiag=float(np.max(np.abs(M - np.diag(np.diag(M))))),
                          max_norm_dev=float(np.max(np.abs(n - 1))))
        print(f"  [{label}] {status}: {sps * 1e3:.1f} ms/step, setup+first "
              f"block {setup:.0f} s, min|M_ii| {out[label]['min_absMii']:.3e}"
              f", max offdiag {out[label]['max_offdiag']:.2e}, max|n-1| "
              f"{out[label]['max_norm_dev']:.2e}; {_mem()}", flush=True)
    dev = jax.devices()[0].platform
    L.save_npz(os.path.join(RES, f"pilot_{dev}.npz"), summary=json.dumps(out))
    d = 1 - out["idle"]["min_absMii"]
    if d > 1e-6:
        raise SystemExit(f"PILOT: idle drift 1 - min|M_ii| = {d:.2e} > 1e-6")


def mode_leg(a):
    p, specs, reg = _setup()
    L.device_banner()
    integ = L.Integ()
    key = a.key
    spec = specs[key]
    T = leg_T(spec)
    P = bit_perm(reg["configs"])
    d = os.path.join(RES, "legs16")
    os.makedirs(d, exist_ok=True)
    legs = Legs(specs, reg["weights"], integ)
    ml, prop = legs.get(key)
    y0 = y_of(ml, reg["A"], reg["Bs"], reg["bs"])
    st = State(os.path.join(d, f"{key}.state.npz"),
               dict(key=key, spec=[list(s) for s in spec], integ=integ.as_dict()))
    s0 = st.load()
    y, t_start, log = y0, 0.0, []
    if s0 is not None:
        y, t_start, log = s0["y"], s0["t_leg"], s0["log"]
        print(f"  [resume] {key} from t = {t_start}", flush=True)
    t = time.time()

    def on_window(t1, yw, info):
        log.append(info)
        st.save(yw, 0, t1, log)

    y, sps = run_leg(ml, prop, y, T, t_start, key, on_window)
    M16 = M16_of(ml, y0, y, P)
    pairs = [PAIRS[key]] if key != "sync" else [(0, 1), (2, 3)]
    store = dict(M16=M16, T=T, spec=json.dumps(spec), integ=json.dumps(integ.as_dict()),
                 norms=norms_of(ml, y), ms_per_step=sps * 1e3,
                 log=json.dumps(log), wall=time.time() - t)
    pred = p["predicted_theta_over_pi2"]
    for pr in pairs:
        c4 = corner4(M16, pr)
        _, zp, rb, ra = strip_localZ(c4, SQI)
        F4 = L.fbar(L.M_REF, c4)[0]
        th = theta_of(c4) / (np.pi / 2)
        lab = f"{pr[0]}{pr[1]}"
        k_pred = key if key != "sync" else ("sync_w0w1" if pr == (0, 1) else "w2w3")
        store.update({f"z_{lab}": zp, f"corner_{lab}": c4, f"F4_{lab}": F4,
                      f"theta_{lab}": th})
        flag = "" if abs(th - 1) <= 0.02 else "   !! theta off pi/2 by > 2 %"
        print(f"  [{key}] pair {pr}: F_bar4 = {F4:.6f}, theta/(pi/2) = {th:.4f} "
              f"(map {pred[k_pred]:.4f}), ||corner||^2 = "
              f"{np.linalg.norm(c4) ** 2:.5f}, col retention "
              f"{np.round(np.sum(np.abs(c4) ** 2, 0), 4).tolist()}, raw overlap "
              f"{rb:.4f} -> {ra:.6f} after Z-strip{flag}", flush=True)
    print(f"  [{key}] ||M16||^2 = {np.linalg.norm(M16) ** 2:.4f} (ideal 16), "
          f"min column survival {np.min(np.sum(np.abs(M16) ** 2, 0)):.4f}",
          flush=True)
    dst = os.path.join(d, f"{key}.npz")
    L.save_npz(dst, **store)
    print(f"  -> {dst}", flush=True)


def mode_arm(a):
    p, specs, reg = _setup()
    L.device_banner()
    integ = L.Integ()
    arm = a.arm
    d = os.path.join(RES, "legs16")
    store = {}
    for k in ARM_LEGS[arm]:
        f = os.path.join(d, f"{k}.npz")
        assert os.path.exists(f), f"{f} missing: run `leg --key {k}` first"
        z = np.load(f)
        assert json.loads(str(z["spec"])) == [list(s) for s in specs[k]]
        assert json.loads(str(z["integ"])) == integ.as_dict()
        store[k] = z
    z = {}
    for k in ARM_LEGS[arm]:
        if k == "sync":
            z["sync01"], z["sync23"] = store[k]["z_01"], store[k]["z_23"]
        else:
            a_, b_ = PAIRS[k]
            z[k] = store[k][f"z_{a_}{b_}"]
    for k, v in z.items():
        print(f"  z frame [{k}] = {np.round(v, 4).tolist()} rad", flush=True)
    check_fingerprints(specs)

    raw = schedule_for(arm, z)
    n_legs = sum(1 for k, _ in raw if k == "leg")
    n_lay = sum(1 for k, _ in raw if k == "layer")
    sched = coalesce(raw)
    T_tot = sum(leg_T(specs[v]) for k, v in sched if k == "leg")
    print(f"  {arm}: {n_legs} legs, {n_lay} local layers, merged into "
          f"{sum(1 for k, _ in sched if k == 'local')} rotations, T = "
          f"{T_tot:.1f} t.u.", flush=True)

    legs = Legs(specs, reg["weights"], integ)
    ml_any, _ = legs.get(sched[1][1] if sched[0][0] == "local" else sched[0][1])
    P = bit_perm(reg["configs"])
    occ = [det_occ(ml_any, s) for s in range(4)]
    b0 = reg["bs"]
    P0, qinfo = qubit_states(reg)
    y0 = y_of(ml_any, reg["A"], reg["Bs"], reg["bs"])

    os.makedirs(os.path.join(RES, "arms"), exist_ok=True)
    st = State(os.path.join(RES, "arms", f"{arm}.state.npz"),
               dict(arm=arm, specs={k: [list(s) for s in v] for k, v in specs.items()},
                    integ=integ.as_dict(),
                    z={k: np.asarray(v).tolist() for k, v in z.items()}))
    s0 = st.load()
    y, start, t_leg, log = y0, 0, 0.0, []
    if s0 is not None:
        y, start, t_leg, log = s0["y"], s0["step"], s0["t_leg"], s0["log"]
        print(f"  [resume] {arm} at schedule step {start + 1}/{len(sched)}, "
              f"t in leg {t_leg}", flush=True)

    # the product model from the same schedule (layers implemented at t = 0)
    M_impl = np.eye(16, dtype=complex)
    M_idl = np.eye(16, dtype=complex)
    for kind, val in sched:
        if kind == "local":
            yl, _ = apply_layer(ml_any, y0, val, P0, b0, occ)
            M_impl = M16_of(ml_any, y0, yl, P) @ M_impl
            M_idl = layer16_ideal(val) @ M_idl
        else:
            M_impl = store[val]["M16"] @ M_impl
            M_idl = store[val]["M16"] @ M_idl

    t_all = time.time()
    for i, (kind, val) in enumerate(sched):
        if i < start:
            continue
        if kind == "local":
            y, ret = apply_layer(ml_any, y, val, P0, b0, occ)
            n = norms_of(ml_any, y)
            log.append(dict(step=i + 1, kind="local", wells=sorted(val),
                            retention={str(s): v for s, v in ret.items()},
                            max_norm_dev=float(np.max(np.abs(n - 1)))))
            print(f"  step {i + 1}/{len(sched)} local on wells {sorted(val)}: "
                  f"|0>/|1> retention " + ", ".join(
                      f"w{s} {min(v):.6f}" for s, v in ret.items())
                  + f"; max|n-1| {log[-1]['max_norm_dev']:.2e}", flush=True)
            st.save(y, i + 1, 0.0, log)
        else:
            ml, prop = legs.get(val)
            T = leg_T(specs[val])
            t0 = t_leg if i == start else 0.0
            print(f"  step {i + 1}/{len(sched)} leg {val} (T {T}, from t {t0})",
                  flush=True)

            def on_window(t1, yw, info, i=i, val=val):
                log.append(dict(step=i + 1, kind="leg", key=val, **info))
                st.save(yw, i if t1 < T - 1e-9 else i + 1,
                        t1 if t1 < T - 1e-9 else 0.0, log)

            y, _ = run_leg(ml, prop, y, T, t0, f"{arm} {i + 1}/{len(sched)} {val}",
                           on_window)
        t_leg = 0.0

    M = M16_of(ml_any, y0, y, P)
    surv = np.sum(np.abs(M) ** 2, axis=0)
    assert np.all(np.isfinite(M)), "non-finite final M"
    Fb, fxc, Mlog = _score_matrix(M)
    Fp, xp, Lp = _score_matrix(M_impl)
    Fi, xi_, Li = _score_matrix(M_idl)
    F4 = L.fbar(L.M_REF, Mlog)[0]
    print(f"\n=== RESULT: {arm} arm ===", flush=True)
    print(f"  FULL PROPAGATION      F_bar16 = {Fb:.6f}   F_xc = {fxc:.6f}   "
          f"||M||^2 = {np.linalg.norm(M) ** 2:.4f} (ideal 16)", flush=True)
    print(f"  product, impl. layers F_bar16 = {Fp:.6f}   F_xc = {xp:.6f}   "
          f"||M||^2 = {np.linalg.norm(M_impl) ** 2:.4f}", flush=True)
    print(f"  product, ideal layers F_bar16 = {Fi:.6f}   F_xc = {xi_:.6f}",
          flush=True)
    print(f"  (0,3) corner F_bar4: full {F4:.6f}, product "
          f"{L.fbar(L.M_REF, Lp)[0]:.6f}", flush=True)
    print(f"  column survival min/mean/max {surv.min():.4f}/{surv.mean():.4f}/"
          f"{surv.max():.4f}" + ("   !! survival > 1" if surv.max() > 1 + 1e-9
                                 else ""), flush=True)
    print(f"  T = {T_tot:.1f} t.u.; wall {(time.time() - t_all) / 3600:.2f} h",
          flush=True)
    dst = os.path.join(RES, "arms", f"arm_{arm}.npz")
    L.save_npz(dst, M=M, Fbar16=Fb, F_xc=fxc, M_logical=Mlog, F4_corner=F4,
               survival=surv, M_prod_impl=M_impl, M_prod_idl=M_idl,
               Fbar_prod_impl=Fp, Fbar_prod_idl=Fi, T_total=T_tot,
               n_legs=n_legs, n_layers=n_lay, log=json.dumps(log),
               z=json.dumps({k: np.asarray(v).tolist() for k, v in z.items()}),
               specs=json.dumps({k: [list(s) for s in v] for k, v in specs.items()}),
               integ=json.dumps(integ.as_dict()), configs=np.array(reg["configs"]))
    print(f"  -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pulses", "relax", "check", "pilot", "idle", "leg", "arm"])
    ap.add_argument("--key", choices=["w0w1", "w1w2", "w2w3", "sync"])
    ap.add_argument("--arm", choices=["serial", "sync"])
    ap.add_argument("--T", type=float, default=30.0)
    a = ap.parse_args()
    t0 = time.time()
    {"pulses": mode_pulses, "relax": mode_relax, "check": mode_check, "pilot": mode_pilot, "idle": mode_idle,
     "leg": mode_leg, "arm": mode_arm}[a.mode](a)
    print(f"\n  wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
