"""Gate timing for 3 e-/well: hold, ramp window, excursion bound, sweep grid.

    python -m gate_timing.timing                    # everything (CPU is enough)
    python -m gate_timing.timing --grid-only        # rebuild grids from physics.json
    python -m gate_timing.timing --grid-only --block 0.2

The rules of Secs. S5 A-C, applied to the three-electron operating points
with the change of Sec. S5 F.  Everything is computed from the stored
operating points: no fitted parameter and no real-time propagation.

What is computed
----------------

**Delta (hold).** Each gate's 16-state best state
(``operating_points/results/ml_sa16_20260924_target_stage{2..5}_superpos_<leg>``) is
re-analysed with ``MLMCTDHF_n_species_ITE_opt.analyse`` at its ``best_x``.
The matched singles partners (``settings["block"]``, e.g. |1000>/|0100>) are
used, never the energy order.  The operating points sit at a 55/45 mixing, so
the raw gap over-estimates the splitting; in the two-level picture
``gap = sqrt(eps^2 + Delta^2)`` and ``2 sqrt(p(1-p)) = Delta/gap``, hence
``Delta = gap * 2 sqrt(p(1-p))`` with ``p = F_matched[A, A]``.  As a
cross-check the 2x2 effective Hamiltonian is rebuilt from the top-block
eigenvector amplitudes (Loewdin-orthonormalised) and ``2|H_01|`` reported.
``T_0 = pi/(2 Delta)``; only k = 0 counts.

**omega_min (ramp lower bound).**  With three electrons the anharmonicity
changes sign (Delta12 < Delta01), so the spacing that must not be driven is
``min(Delta01, Delta12)``.  Each active well is solved alone along the ramp
path ``x(env) = x_idle + env (x_target - x_idle)``, env = 0, 0.1, ..., 1, the
path ``fourwell.h_ramp`` interpolates, by exact FCI in its full 64-point DVR
(:class:`FCIWell`).  Single-species SA-3 MCTDHF at norb 14 (:class:`IsoWell`)
costs ~0.15 s/step x ~1500-2000 steps per point on CPU, so it is run at the
idle point only, as a check of the FCI (``mctdhf_check``: agreement
<= 3.4e-4).  That the idle end sets the bound is checked.

Two dressings are reported:

* ``bare``: the isolated well.  This sets the grid.
* ``dressed``: the same well plus the frozen Hartree potential of the other
  three wells' ground-state densities (their own bare isolated solutions at
  the same env; spectators at idle).  Needed because (i) the bare well is
  parity-symmetric and ``dH/d env`` is even, so ``<0|dH|1>`` vanishes there
  and the excursion diagnostic would miss |00> -> |10> transitions; (ii) it is
  what the four-well detunings should be compared with.

At idle a third, ``dressed_ml``, takes the Hartree field from the
ML idle state's |0000> row instead.

**Excursion diagnostic.**  ``tau_ramp >> 3.385 max |<m| dH/d env |n>| / dE_mn^2``
over m in {0, 1} (the populated qubit levels) and n <= 5,
3.385 = 6/sqrt(pi) the peak slope of the erf edge of ``fourwell.make_envelope``
in units of 1/tau_ramp.  ``dH/d env`` is the one-body ``dV/d env`` in the MO
basis with g = 0 (in the DVR: the diagonal occupation sum, applied to the
FCI vectors; ``IsoWell`` does the same through ``get_sigma``).  Also
reported with the envelope profile, ``f(env) = 3.385 exp(-erfinv(2 env-1)^2)``
instead of its peak.  Diagnostic only: the grid follows the rules above.

**Grid.**  alpha = 1, so ``tau_hold = T_eff - tau_ramp``.
``dT = sqrt(10 eps)/Delta``, eps = 1e-2; ``T_eff = T_0 + k dT/3``, k = -6..6.
Ramp: 6 geometric points in ``[10/omega_min, 1/Delta]``.  ``tau_ramp`` is
snapped up onto the block grid (rounding to the nearest can break the lower
bound), ``T_eff`` to the nearest block; points with ``tau_hold < 2`` are
dropped.  Refine step ``dT(1e-3)/3``.

Verification
------------
* ``analyse`` on the idle state reproduces the stored detunings and
  ``E_matched`` (``idle_states_sa16_ns4_norb14.npz``);
* ``analyse`` at each ``best_x`` reproduces the stored energies/fidelities
  of the best evaluation (``ml_sa16_..._<leg>.npz``, row argmin(losses)),
  hence the gap;
* the isolated-well FCI reproduces reference spacings of a 64-point FCI
  at the one-electron idle point ``fourwell.X_IDLE`` (REF_EXACT_X_IDLE);
* the isolated Delta01 at idle is compared with the four-well detunings
  1.712 / 2.540 / 3.167 / 4.413.

Outputs: ``results/timing/physics.json`` (+ ``physics.npz`` arrays),
``results/timing/tables.md`` and ``gate_timing/grids.json``.
"""

import argparse
import json
import os
import sys
import time

from operating_points import common as C  # noqa: I100  (sets x64 before any array)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from opt_einsum import contract  # noqa: E402
from scipy.special import erfinv  # noqa: E402

from mctdhf.MCTDHF_n_species_ITE_opt import (  # noqa: E402
    MCTDHF_n_species_ITE_opt, _make_cmf_step)
from system import fourwell as fw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "timing")
GRIDS_JSON = os.path.join(HERE, "grids.json")

IDLE_STATE = os.path.join(C.RESULTS_DIR, "idle_states_sa16_ns4_norb14.npz")
STEM = "ml_sa16_20260924_target_stage{st}_superpos_{leg}"
# leg -> (stage, active wells). The matched pair comes from settings["block"].
LEGS = {"w0w1": (2, (0, 1)), "w1w2": (3, (1, 2)), "w2w3": (4, (2, 3)),
        "w0w3": (5, (0, 3))}
DEFERRED = ("w0w3",)          # derive, do not sweep

# Preliminary numbers (stored diagonals) and the dressed 2J of an earlier
# standard-MCTDHF three-electron calculation, for comparison.
PRELIM_DELTA = {"w0w1": 1.4159e-2, "w1w2": 1.1747e-2, "w2w3": 1.1254e-2,
                "w0w3": 1.082e-4}
REF_2J_MCTDHF = {"w0w1": 1.38280e-2, "w1w2": 1.17226e-2, "w2w3": 1.18583e-2}
# Four-well idle detunings the isolated Delta01 is cross-checked against
# (idle_states_sa16_ns4_norb14.npz "detunings").
IDLE_DET_REF = np.array([1.71196085, 2.54033981, 3.16694683, 4.41267074])
# Reference isolated N = 3 spacings (64-point FCI, Davidson) at the
# one-electron idle point fourwell.X_IDLE.
REF_EXACT_X_IDLE = {0: (1.742185, 1.268956), 1: (2.549894, 1.868854),
                        2: (3.013643, 2.178306), 3: (4.410508, 3.313768)}

ENVS = np.round(np.linspace(0.0, 1.0, 11), 10)
PEAK_SLOPE = 6.0 / np.sqrt(np.pi)          # 3.385, erf edge, units 1/tau_r
N_RAMP = 6
C_RAMP = 10.0                              # Sec. S5 B, c = 10
K_RANGE = np.arange(-6, 7)
TH_MIN = 2.0                               # stage_a_plan.TH_MIN
NORB_ML, NSPF_ML = 14, 4


def _stem(leg):
    return STEM.format(st=LEGS[leg][0], leg=leg)


# ═══════════════════════════════════════════════════════════════ Delta ═════


def _load_leg(leg):
    npz = os.path.join(C.RESULTS_DIR, _stem(leg) + ".npz")
    st_path = os.path.join(C.RESULTS_DIR, _stem(leg) + "_best_state.npz")
    with np.load(npz, allow_pickle=False) as d:
        settings = json.loads(str(d["settings"]))
        best_x = np.asarray(d["best_x"], dtype=float)
        i = int(np.argmin(d["losses"]))
        stored = dict(E=np.asarray(d["energies"][i]),
                      F=np.asarray(d["fidelities"][i]),
                      det=np.asarray(d["detunings"][i]),
                      x=np.asarray(d["xs"][i]), idx=i,
                      loss=float(d["losses"][i]))
    with np.load(st_path) as d:
        x_state = np.asarray(d["x_ghz"], dtype=float)
    assert np.array_equal(x_state, best_x), (leg, x_state, best_x)
    assert np.array_equal(stored["x"], best_x), leg
    assert tuple(settings["active"]) == LEGS[leg][1], (leg, settings["active"])
    return npz, st_path, settings, best_x, stored


def ml_solver():
    return C.make_solver(NORB_ML, NSPF_ML, **C.relax_set("16"), dt=0.02,
                         precision="mixed", inner_tol=0.01, chunk=50,
                         eps=1e-6)


def delta_leg(sv, leg):
    """Delta from the matched singles partners at the leg's best state."""
    npz, st_path, settings, best_x, stored = _load_leg(leg)
    a, b = settings["block"]
    state = C.load_state(st_path)
    h = C.h_of(best_x)
    t0 = time.perf_counter()
    an = sv.analyse(state, h)
    E, F = an["E_matched"], an["F_matched"]
    gap = float(E[b] - E[a])
    pA, pB = float(F[a, a]), float(F[b, b])
    delta = abs(gap) * 2 * np.sqrt(pA * (1 - pA))
    delta_B = abs(gap) * 2 * np.sqrt(pB * (1 - pB))
    pn = pA / (F[a, a] + F[a, b])
    delta_norm = abs(gap) * 2 * np.sqrt(pn * (1 - pn))
    # 2x2 effective Hamiltonian from the amplitudes of the two matched roots
    # on the two configurations (Loewdin), H_eff = U diag(E) U^dagger.
    _A, _B, evals, evecs, _occ = sv._analyse_dev(*state, sv.as_h(h))
    evals, evecs = np.asarray(evals), np.asarray(evecs)
    rk = an["ranks_all"]
    V = evecs[np.ix_(sv._flat_all[[a, b]], rk[[a, b]])]
    w, Q = np.linalg.eigh(V.conj().T @ V)
    U = V @ Q @ np.diag(w ** -0.5) @ Q.conj().T
    Heff = U @ np.diag(evals[rk[[a, b]]]) @ U.conj().T
    delta_2x2 = 2 * abs(Heff[0, 1])
    ver = dict(
        dE_max=float(np.max(np.abs(E - stored["E"]))),
        dF_max=float(np.max(np.abs(np.diag(F) - stored["F"]))),
        ddet_max=float(np.max(np.abs(an["detunings"] - stored["det"]))),
        gap_stored=float(stored["E"][b] - stored["E"][a]),
        gap_diff=float(gap - (stored["E"][b] - stored["E"][a])),
    )
    out = dict(
        leg=leg, active=list(LEGS[leg][1]), block=[a, b],
        best_x_ghz=best_x.tolist(), E_A=float(E[a]), E_B=float(E[b]),
        gap=gap, pA=pA, pB=pB, block_weight_A=float(F[a, a] + F[a, b]),
        Delta=float(delta), Delta_pB=float(delta_B),
        Delta_pnorm=float(delta_norm), Delta_2x2=float(delta_2x2),
        eps_2x2=float(np.real(Heff[1, 1] - Heff[0, 0])),
        detunings=an["detunings"].tolist(), healthy=bool(an["healthy"]),
        verify=ver, stored_row=stored["idx"],
        files=dict(npz=npz, best_state=st_path),
        wall=time.perf_counter() - t0,
    )
    print(f"  [{leg}] pair {a},{b}: E_A {E[a]:.9f} E_B {E[b]:.9f} gap "
          f"{gap:+.6e} p_A {pA:.5f} p_B {pB:.5f}\n"
          f"         Delta = gap*2sqrt(p(1-p)) = {delta:.6e}  (p_B "
          f"{delta_B:.6e}, p_norm {delta_norm:.6e}, 2x2 {delta_2x2:.6e})\n"
          f"         verify vs npz row {stored['idx']}: max|dE| "
          f"{ver['dE_max']:.2e} max|dF_diag| {ver['dF_max']:.2e} max|ddet| "
          f"{ver['ddet_max']:.2e} gap diff {ver['gap_diff']:.2e}  "
          f"[{out['wall']:.1f} s]", flush=True)
    return out


def verify_idle(sv):
    """analyse on the idle state must reproduce its stored detunings."""
    with np.load(IDLE_STATE) as d:
        x = np.asarray(d["x_ghz"], dtype=float)
        det_ref = np.asarray(d["detunings"])
        E_ref = np.asarray(d["E_matched"])
    st = C.load_state(IDLE_STATE)
    an = sv.analyse(st, C.h_of(x))
    out = dict(x_idle_ghz=x.tolist(), detunings=an["detunings"].tolist(),
               detunings_stored=det_ref.tolist(),
               ddet_max=float(np.max(np.abs(an["detunings"] - det_ref))),
               dE_max=float(np.max(np.abs(an["E_matched"] - E_ref))),
               healthy=bool(an["healthy"]))
    print(f"  [idle] detunings " + " ".join(f"{v:.8f}" for v in
                                           an["detunings"])
          + f"\n         vs stored max|d| {out['ddet_max']:.2e}, "
          f"max|dE_matched| {out['dE_max']:.2e}", flush=True)
    # Hartree field of the |0000> row, for the dressed_ml idle check.
    A, Bs, bs = st
    Ds = sv.ml.dCI.ml_1p_RDMs(A[:1], Bs)
    dens = [np.real(np.einsum("ip,pq,iq->i", np.conj(np.asarray(bs[s])),
                              np.asarray(Ds[s][0]), np.asarray(bs[s])))
            for s in range(fw.NUM_WELLS)]
    for s, n in enumerate(dens):
        assert abs(n.sum() - fw.NUM_PARTICLES) < 1e-8, (s, n.sum())
    return out, x / fw.TYPICAL_FREQ_GHZ, dens


# ═════════════════════════════════════════════════════ isolated wells ═════


class IsoWell:
    """One N = 3 well alone: SA-3 MCTDHF relaxation and its analysis.

    All four wells are translates of one another (same box width, same
    intra kernel, ``fourwell.g_dict``), so one solver serves every well; only
    ``h`` (traced) changes.
    """

    def __init__(self, norb, dt, chunk, max_steps, tol):
        g = fw.g_dict()[(0, 0)]
        self.sv = MCTDHF_n_species_ITE_opt(
            g_dict={(0, 0): g}, num_particles=[fw.NUM_PARTICLES],
            num_mctdhf_orbitals=[norb], num_spatial_orbitals=[fw.NUM_DVR],
            num_states=3, eps=1e-6, dtype=jnp.complex128)
        self.norb, self.dt, self.chunk = norb, dt, chunk
        self.max_steps, self.tol = max_steps, tol
        self.D = int(self.sv.num_dets[0])
        self._eig = jax.jit(self._eig_impl)
        self._step = _make_cmf_step(self.sv, dt, 20)
        sv = self.sv

        @jax.jit
        def _ortho(y):
            C_, bs_ = sv._y_to_Cb(y)
            return sv._Cb_to_y(sv.ortonorm_C(C_), [sv.ortonorm_b(bs_[0])])

        self._ortho = _ortho

    def _eig_impl(self, b, h, dV):
        sv = self.sv
        h_MO = sv.transform_h(h.astype(jnp.complex128), b)[1]
        g_MO, _ = sv._transform_all_g([b])
        eye = jnp.eye(self.D, dtype=jnp.complex128)
        H = sv.dCI.get_sigma(eye, [h_MO], g_MO).reshape(self.D, self.D).T
        H = 0.5 * (H + H.conj().T)
        E, V = jnp.linalg.eigh(H)
        V3 = V[:, :3].T                                   # (3, D) states
        dh = contract("im, i, in -> mn", b.conj(), dV.astype(jnp.complex128),
                      b, backend="jax")
        zero = {k: jnp.zeros_like(v) for k, v in g_MO.items()}
        sig = sv.dCI.get_sigma(V3, [dh], zero)
        M = V3.conj() @ sig.T                             # <m|dH|n>
        D1 = sv.dCI.get_1p_RDMs(V3)[0]                    # (3, norb, norb)
        dens = jnp.real(contract("ip, spq, iq -> si", b.conj(), D1, b,
                                 backend="jax"))
        return E[:4], M, dens

    def relax(self, h, b0=None):
        h = np.asarray(h, dtype=np.float64)
        if b0 is None:
            _e, v = np.linalg.eigh(h)
            b0 = v[:, :self.norb]
        b = jnp.asarray(b0, dtype=jnp.complex128)
        C0 = jnp.asarray(np.eye(3, self.D), dtype=jnp.complex128)
        zero_dV = jnp.zeros(h.shape[0])
        hj = (jnp.asarray(h, dtype=jnp.float64),)
        y = self.sv._Cb_to_y(C0, [b])
        t = jnp.asarray(0.0)
        E_prev, steps, converged = None, 0, False
        while steps < self.max_steps:
            # The one-step jitted CMF step of MCTDHF_n_species_ITE_opt
            # (diagonalize C, SIL orbitals) in a host loop, with the same
            # per-step orbital QR its `integrate` scan applies. Its scan
            # takes > 5 min to compile and a lot of memory.
            for _ in range(self.chunk):
                t, y = self._step(t, y, hj)
                y = self._ortho(y)
            steps += self.chunk
            b = self.sv._y_to_Cb(y)[1][0]
            E = np.asarray(self._eig(b, hj[0], zero_dV)[0])
            if E_prev is not None and np.max(np.abs(E[:3] - E_prev[:3])) \
                    < self.tol:
                converged = True
                break
            E_prev = E
        return b, steps, converged

    def analyse(self, b, h, dV):
        E, M, dens = self._eig(b, jnp.asarray(h, dtype=jnp.float64),
                               jnp.asarray(dV, dtype=jnp.float64))
        return np.asarray(E), np.asarray(M), np.asarray(dens)


class FCIWell:
    """Exact N = 3 FCI of one well in its full 64-point sine DVR.

    The norb -> 64 limit of :class:`IsoWell` (``FermiSpace(NUM_DVR, 3)`` +
    Davidson). In the DVR the
    potential and the intra kernel are diagonal, so with determinants = sorted
    grid-point triples

        H = sum_{p != q} T_pq a+_q a_p  +  sum_i (T_ii + V_i) n_i
            + sum_{i<j} g_ij n_i n_j ,

    the off-diagonal part is single hops with sign (-1)^(occupied strictly
    between p and q). ``g`` is ``fourwell.g_dict()[(w, w)]`` (the veff
    kernel), used in the same 2-D DVR form the MCTDHF solvers use. D = 41664.
    """

    def __init__(self, n=fw.NUM_DVR, N=fw.NUM_PARTICLES):
        from itertools import combinations
        assert N == 3
        dets = np.array(list(combinations(range(n), N)), dtype=np.int64)
        D = len(dets)
        look = -np.ones((n, n, n), dtype=np.int64)
        look[dets[:, 0], dets[:, 1], dets[:, 2]] = np.arange(D)
        rows, cols, ps, qs, sg = [], [], [], [], []
        for a in range(N):
            others = np.delete(dets, a, axis=1)             # (D, 2)
            p = dets[:, a]
            for q in range(n):
                ok = np.all(others != q, axis=1) & (p != q)
                d_idx = np.nonzero(ok)[0]
                new = np.sort(np.column_stack(
                    [others[d_idx], np.full(len(d_idx), q)]), axis=1)
                lo = np.minimum(p[d_idx], q)
                hi = np.maximum(p[d_idx], q)
                o = others[d_idx]
                between = ((o > lo[:, None]) & (o < hi[:, None])).sum(axis=1)
                rows.append(look[new[:, 0], new[:, 1], new[:, 2]])
                cols.append(d_idx)
                ps.append(p[d_idx])
                qs.append(np.full(len(d_idx), q))
                sg.append(1.0 - 2.0 * (between % 2))
        self.rows = np.concatenate(rows)
        self.cols = np.concatenate(cols)
        self.p = np.concatenate(ps)
        self.q = np.concatenate(qs)
        self.sign = np.concatenate(sg)
        assert np.all(self.rows >= 0)
        self.dets, self.D, self.n = dets, D, n
        g = np.asarray(fw.g_dict()[(0, 0)], dtype=np.float64)
        i, j, k = dets.T
        self.g_diag = g[i, j] + g[i, k] + g[j, k]

    def hamiltonian(self, h):
        import scipy.sparse as sp
        h = np.asarray(h, dtype=np.float64)
        off = sp.csr_matrix((self.sign * h[self.q, self.p],
                             (self.rows, self.cols)), shape=(self.D, self.D))
        diag = np.diag(h)[self.dets].sum(axis=1) + self.g_diag
        return off + sp.diags(diag)

    def solve(self, h, dV, k=6, v0=None):
        from scipy.sparse.linalg import eigsh
        H = self.hamiltonian(h)
        if v0 is None:
            v0 = np.ones(self.D) / np.sqrt(self.D)
        E, V = eigsh(H, k=k, which="SA", tol=1e-13, v0=v0, ncv=40)
        o = np.argsort(E)
        E, V = E[o], V[:, o]
        res = float(np.max(np.linalg.norm(H @ V - V * E, axis=0)))
        occ_dV = np.asarray(dV)[self.dets].sum(axis=1)
        V3 = V[:, :3]
        M = V.T @ (occ_dV[:, None] * V)                    # <m|dV|n>, k x k
        dens = np.zeros((3, self.n))
        for s in range(3):
            w2 = V3[:, s] ** 2
            for a in range(3):
                dens[s] += np.bincount(self.dets[:, a], weights=w2,
                                       minlength=self.n)
        return E, M, dens, res, V[:, 0]


def _t_kin(w):
    return np.real(np.asarray(fw.t_kin()[w]))


def h_well(w, a, b, vH=None):
    """Isolated-well h at (alpha, beta): ``fourwell.h_static``'s formula."""
    h = _t_kin(w) + np.diag(fw._v_well(w, a, b))
    return h if vH is None else h + np.diag(vH)


def dV_denv(w, x_i, x_t, env):
    """d V_w / d env along x(env) = x_i + env (x_t - x_i) (fourwell.h_ramp)."""
    dx = fw.X_S[w] - float(fw.WELL_CENTERS[w])
    a = x_i[w] + env * (x_t[w] - x_i[w])
    b = x_i[4 + w] + env * (x_t[4 + w] - x_i[4 + w])
    da, db = x_t[w] - x_i[w], x_t[4 + w] - x_i[4 + w]
    dV_da = a * dx ** 2 + (2.0 * a * b / 3.0) * dx ** 4
    dV_db = (a ** 2 / 3.0) * dx ** 4
    return dV_da * da + dV_db * db


def x_at(x_i, x_t, env):
    return x_i + env * (x_t - x_i)


def _V_pair(s, k):
    g = fw.g_dict()
    return np.asarray(g[(s, k)]) if s < k else np.asarray(g[(k, s)]).T


def hartree(s, dens):
    """Hartree potential on well s from the densities of the other wells."""
    v = np.zeros(fw.NUM_DVR)
    for k, n in dens.items():
        if k != s:
            v = v + _V_pair(s, k) @ n
    return v


# Transitions OUT OF the qubit levels 0 and 1 (the only populated ones) into
# every other computed level. At N = 3 the level spacing above |2> is small
# (D23 = 0.46 in w0 at idle), and 1 -> 3 has the parity of 0 -> 2, so the
# 0/1/2 subset alone would miss the |1>-sourced leakage.
PAIRS = [(0, n) for n in range(1, 6)] + [(1, n) for n in range(2, 6)]


def _pair_bounds(E, M, fac=PEAK_SLOPE):
    out = {}
    for m, n in PAIRS:
        if n < len(E):
            out[f"{m}{n}"] = float(fac * abs(M[m, n]) / (E[n] - E[m]) ** 2)
    return out


def isolated_study(fci, x_idle, x_targets, ml_dens):
    """Bare and dressed exact-FCI spectra along every leg's ramp path."""
    res = {"idle_bare": {}, "idle_dressed_ml": {}, "old_idle": {},
           "paths": {}}
    t0 = time.perf_counter()
    zero = np.zeros(fw.NUM_DVR)

    # ── validation against the reference FCI at fourwell.X_IDLE ────────────
    for w in range(fw.NUM_WELLS):
        h = h_well(w, fw.X_IDLE[w], fw.X_IDLE[4 + w])
        E, _M, _d, r, _v = fci.solve(h, zero)
        d01, d12 = float(E[1] - E[0]), float(E[2] - E[1])
        r01, r12 = REF_EXACT_X_IDLE[w]
        res["old_idle"][w] = dict(d01=d01, d12=d12, ref01=r01, ref12=r12,
                                  resid=r)
        print(f"  [X_IDLE, vs reference] w{w}: D01 {d01:.6f} (ref "
              f"{r01:.6f}, {d01 - r01:+.1e})  D12 {d12:.6f} ({r12:.6f}, "
              f"{d12 - r12:+.1e})  resid {r:.1e}", flush=True)

    # ── idle, bare, all four wells (spectator densities for dressing) ──────
    bare_dens = {}
    for w in range(fw.NUM_WELLS):
        h = h_well(w, x_idle[w], x_idle[4 + w])
        E, _M, dens, r, _v = fci.solve(h, zero)
        bare_dens[w] = dens[0]
        res["idle_bare"][w] = dict(E=E.tolist(), d01=float(E[1] - E[0]),
                                   d12=float(E[2] - E[1]),
                                   d23=float(E[3] - E[2]), resid=r,
                                   ref_det=float(IDLE_DET_REF[w]))
        print(f"  [idle bare] w{w}: D01 {E[1]-E[0]:.6f} D12 {E[2]-E[1]:.6f} "
              f"D23 {E[3]-E[2]:.6f}  (4-well detuning {IDLE_DET_REF[w]:.6f},"
              f" diff {E[1]-E[0]-IDLE_DET_REF[w]:+.4f})", flush=True)
    for w in range(fw.NUM_WELLS):
        vH = hartree(w, dict(enumerate(ml_dens)))
        h = h_well(w, x_idle[w], x_idle[4 + w], vH)
        E, _M, _d, r, _v = fci.solve(h, zero)
        res["idle_dressed_ml"][w] = dict(d01=float(E[1] - E[0]),
                                         d12=float(E[2] - E[1]), resid=r)
        print(f"  [idle dressed by ML |0000> Hartree] w{w}: D01 "
              f"{E[1]-E[0]:.6f} (diff {E[1]-E[0]-IDLE_DET_REF[w]:+.4f}) "
              f"D12 {E[2]-E[1]:.6f}", flush=True)

    # ── the ramp paths ─────────────────────────────────────────────────────
    prof = {float(e): (PEAK_SLOPE * np.exp(-erfinv(2 * e - 1) ** 2)
                       if 0 < e < 1 else 0.0) for e in ENVS}
    for leg, x_t in x_targets.items():
        act = LEGS[leg][1]
        spec = [w for w in range(fw.NUM_WELLS) if w not in act]
        diff = np.nonzero(np.abs(x_t - x_idle) > 1e-12)[0]
        allowed = set(act) | {4 + w for w in act}
        assert set(diff) <= allowed, (leg, diff)
        bb, dd = {}, {}
        for w in act:                                    # bare pass
            v = None                     # warm Lanczos start along the path
            for e in ENVS:
                x = x_at(x_idle, x_t, e)
                h = h_well(w, x[w], x[4 + w])
                E, M, dens, r, v = fci.solve(h, dV_denv(w, x_idle, x_t, e),
                                             v0=v)
                bb[(w, e)] = dict(E=E, M=M, dens=dens[0], resid=r, v=v)
        for w in act:          # dressed: frozen Hartree of the other wells
            other = [v for v in act if v != w][0]
            for e in ENVS:
                x = x_at(x_idle, x_t, e)
                dens = {other: bb[(other, e)]["dens"]}
                for sp in spec:
                    dens[sp] = bare_dens[sp]
                h = h_well(w, x[w], x[4 + w], hartree(w, dens))
                E, M, _dn, r, _v = fci.solve(h, dV_denv(w, x_idle, x_t, e),
                                             v0=bb[(w, e)]["v"])
                dd[(w, e)] = dict(E=E, M=M, resid=r)
        legres = {}
        for w in act:
            rows = []
            for e in ENVS:
                for kind, src in (("bare", bb), ("dressed", dd)):
                    r = src[(w, e)]
                    E, M = r["E"], r["M"]
                    rows.append(dict(
                        env=float(e), kind=kind, E=E.tolist(),
                        d01=float(E[1] - E[0]), d12=float(E[2] - E[1]),
                        d23=float(E[3] - E[2]), d34=float(E[4] - E[3]),
                        M=np.abs(M).tolist(),
                        bound_peak=_pair_bounds(E, M),
                        bound_prof=_pair_bounds(E, M, prof[float(e)]),
                        resid=float(r["resid"])))
            legres[w] = rows
            for kind in ("bare", "dressed"):
                rr = [r for r in rows if r["kind"] == kind]
                wm = [min(r["d01"], r["d12"]) for r in rr]
                print(f"  [{leg} w{w} {kind}] min(D01,D12) along env: "
                      + " ".join(f"{v:.4f}" for v in wm)
                      + f"  -> argmin env {rr[int(np.argmin(wm))]['env']:.1f}"
                      f"  max resid {max(r['resid'] for r in rr):.1e}",
                      flush=True)
                print("      |<0|dH|1>| " + " ".join(
                    f"{r['M'][0][1]:.3g}" for r in rr), flush=True)
                bp = max(max(r["bound_peak"].values()) for r in rr)
                print(f"      excursion bound (peak slope) {bp:.3f}   by pair "
                      + " ".join(f"{p}:{max(r['bound_peak'][p] for r in rr):.3g}"
                                 for p in rr[0]["bound_peak"]), flush=True)
        res["paths"][leg] = legres
    res["wall"] = time.perf_counter() - t0
    return res


def mctdhf_check(iso, fci, x_idle):
    """norb-14 SA-3 MCTDHF against the exact FCI at the idle point (run at
    idle only: ~0.1 s/step on CPU)."""
    out = {}
    zero = np.zeros(fw.NUM_DVR)
    for w in range(fw.NUM_WELLS):
        h = h_well(w, x_idle[w], x_idle[4 + w])
        t0 = time.perf_counter()
        b, n, cv = iso.relax(h)
        E, _M, _d = iso.analyse(b, h, zero)
        Ef = fci.solve(h, zero)[0]
        out[w] = dict(d01=float(E[1] - E[0]), d12=float(E[2] - E[1]),
                      d01_fci=float(Ef[1] - Ef[0]),
                      d12_fci=float(Ef[2] - Ef[1]),
                      E0=float(E[0]), E0_fci=float(Ef[0]), steps=n,
                      converged=cv, wall=time.perf_counter() - t0)
        print(f"  [norb {iso.norb} SA-3 vs FCI] w{w}: D01 {E[1]-E[0]:.6f} vs "
              f"{Ef[1]-Ef[0]:.6f}  D12 {E[2]-E[1]:.6f} vs {Ef[2]-Ef[1]:.6f}"
              f"  E0 {E[0]-Ef[0]:+.2e}  steps {n} conv {cv} "
              f"[{out[w]['wall']:.0f} s]", flush=True)
    return out


# ═════════════════════════════════════════════════════════════ summary ═════


def omega_summary(iso_res, leg, kind="bare"):
    act = LEGS[leg][1]
    best = None
    per_well = {}
    for w in act:
        rows = [r for r in iso_res["paths"][leg][w] if r["kind"] == kind]
        vals = []
        for r in rows:
            for gname in ("d01", "d12"):
                vals.append((r[gname], r["env"], gname))
        m = min(vals)
        idle = [r for r in rows if r["env"] == 0.0][0]
        gate = [r for r in rows if r["env"] == 1.0][0]
        per_well[w] = dict(min=m[0], env=m[1], gap=m[2],
                           d01_idle=idle["d01"], d12_idle=idle["d12"],
                           d01_gate=gate["d01"], d12_gate=gate["d12"])
        if best is None or m[0] < best[0]:
            best = (m[0], w, m[1], m[2])
    return dict(omega_min=best[0], well=best[1], env=best[2], gap=best[3],
                idle_binds=bool(best[2] == 0.0), per_well=per_well)


def excursion_summary(iso_res, leg):
    out = {}
    for kind in ("bare", "dressed"):
        best = dict(peak=(0.0, None), prof=(0.0, None))
        for w, rows in iso_res["paths"][leg].items():
            for r in rows:
                if r["kind"] != kind:
                    continue
                for key, fld in (("peak", "bound_peak"), ("prof", "bound_prof")):
                    for p, v in r[fld].items():
                        if v > best[key][0]:
                            best[key] = (v, dict(well=int(w), env=r["env"],
                                                 pair=p))
        out[kind] = dict(peak=best["peak"][0], peak_at=best["peak"][1],
                         prof=best["prof"][0], prof_at=best["prof"][1])
    return out


# ════════════════════════════════════════════════════════════════ grid ═════


def snap_up(v, blk):
    return np.round(np.ceil(np.asarray(v) / blk - 1e-9) * blk, 10)


def snap_near(v, blk):
    return np.round(np.round(np.asarray(v) / blk) * blk, 10)


def build_grid(leg, Delta, omega_min, block, eps=1e-2, eps_ref=1e-3):
    """Sweep grid of Sec. S5 C, alpha = 1, k = 0 only."""
    T0 = np.pi / (2 * Delta)
    dT = np.sqrt(10 * eps) / Delta
    dT_ref = np.sqrt(10 * eps_ref) / Delta
    lo, hi = C_RAMP / omega_min, 1.0 / Delta
    ramp_exact = lo * (hi / lo) ** (np.arange(N_RAMP) / (N_RAMP - 1))
    ramp = snap_up(ramp_exact, block)
    assert np.all(ramp >= lo - 1e-12)
    Teff_exact = T0 + K_RANGE * dT / 3
    Teff = snap_near(Teff_exact, block)
    pts, dropped = [], []
    for tr in ramp:
        for T in Teff:
            th = round(float(T) - float(tr), 10)
            rec = dict(tau_ramp=float(tr), tau_hold=th, T_eff=float(T),
                       T_total=round(2 * float(tr) + th, 10))
            (pts if th >= TH_MIN else dropped).append(rec)
    for p in pts:                       # every coordinate on the block grid
        for k in ("tau_ramp", "tau_hold", "T_eff", "T_total"):
            nb = p[k] / block
            assert abs(nb - round(nb)) < 1e-6, (leg, k, p[k])
    return dict(
        leg=leg, deferred=leg in DEFERRED, Delta=Delta, T0=T0,
        dT=dT, hold_step=dT / 3, refine_step=dT_ref / 3,
        refine_step_snapped=float(snap_near(dT_ref / 3, block)),
        omega_min=omega_min, ramp_lo=lo, ramp_hi=hi,
        ramp_ratio=hi / lo, ramp_factor=(hi / lo) ** (1 / (N_RAMP - 1)),
        ramp_exact=ramp_exact.tolist(), ramp=ramp.tolist(),
        ramp_max_shift=float(np.max(np.abs(ramp - ramp_exact))),
        Teff_exact=Teff_exact.tolist(), Teff=Teff.tolist(),
        Teff_max_shift=float(np.max(np.abs(Teff - Teff_exact))),
        n_points=len(pts), points=pts, dropped=dropped,
        th_max={f"{tr:g}": max(p["tau_hold"] for p in pts
                               if p["tau_ramp"] == tr)
                for tr in ramp.tolist()
                if any(p["tau_ramp"] == tr for p in pts)},
    )


# ═════════════════════════════════════════════════════════════ output ═════


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def tables_md(phys, grids):
    L = []
    idle = phys["idle"]
    L.append("### Verification\n")
    L.append(f"- idle state re-analysed: detunings "
             + " ".join(f"{v:.8f}" for v in idle["detunings"])
             + f"; max |diff| vs stored {idle['ddet_max']:.1e}, max "
             f"|dE_matched| {idle['dE_max']:.1e}.")
    for leg, d in phys["delta"].items():
        v = d["verify"]
        L.append(f"- {leg}: vs `{os.path.basename(d['files']['npz'])}` row "
                 f"{d['stored_row']}: max|dE| {v['dE_max']:.1e}, max|dF_diag|"
                 f" {v['dF_max']:.1e}, max|d det| {v['ddet_max']:.1e}, gap "
                 f"{d['gap']:+.10e} vs stored {v['gap_stored']:+.10e} "
                 f"(diff {v['gap_diff']:.1e}).")
    iso = phys["iso"]
    L.append("\n| well | FCI D01 @ X_IDLE | reference | diff | FCI D12 | "
             "reference | diff |")
    L.append("|---|---|---|---|---|---|---|")
    for w, r in iso["old_idle"].items():
        L.append(f"| w{w} | {r['d01']:.6f} | {r['ref01']:.6f} | "
                 f"{r['d01']-r['ref01']:+.1e} | {r['d12']:.6f} | "
                 f"{r['ref12']:.6f} | {r['d12']-r['ref12']:+.1e} |")
    L.append("\n| well | 4-well detuning (idle) | bare D01 | diff | "
             "dressed (ML \\|0000> Hartree) D01 | diff | bare D12 | "
             "bare D23 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for w, r in iso["idle_bare"].items():
        dm = iso["idle_dressed_ml"][w]
        L.append(f"| w{w} | {r['ref_det']:.6f} | {r['d01']:.6f} | "
                 f"{r['d01']-r['ref_det']:+.4f} ({(r['d01']/r['ref_det']-1):+.2%})"
                 f" | {dm['d01']:.6f} | {dm['d01']-r['ref_det']:+.4f} | "
                 f"{r['d12']:.6f} | {r['d23']:.6f} |")
    if "mctdhf_check" in phys:
        L.append("\n| well | norb-14 SA-3 D01 | FCI D01 | norb-14 D12 | "
                 "FCI D12 | E0 - E0(FCI) | steps | converged |")
        L.append("|---|---|---|---|---|---|---|---|")
        for w, r in phys["mctdhf_check"]["wells"].items():
            L.append(f"| w{w} | {r['d01']:.6f} | {r['d01_fci']:.6f} | "
                     f"{r['d12']:.6f} | {r['d12_fci']:.6f} | "
                     f"{r['E0']-r['E0_fci']:+.2e} | {r['steps']} | "
                     f"{r['converged']} |")
    L.append("")
    L.append("### Delta per leg\n")
    L.append("| leg | pair | gap | p_A | p_B | **Delta** | Delta (2x2) | "
             "prelim | vs prelim | 2J (MCTDHF) | vs 2J | T_0 | dT(1e-2) | "
             "step | refine step | 1/Delta |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for leg, d in phys["delta"].items():
        g = grids[leg]
        p2 = REF_2J_MCTDHF.get(leg)
        L.append(
            f"| {leg} | {d['block'][0]},{d['block'][1]} | {d['gap']:+.4e} | "
            f"{d['pA']:.4f} | {d['pB']:.4f} | **{d['Delta']:.5e}** | "
            f"{d['Delta_2x2']:.5e} | {PRELIM_DELTA[leg]:.4e} | "
            f"{d['Delta']/PRELIM_DELTA[leg]-1:+.2%} | "
            + (f"{p2:.4e} | {d['Delta']/p2-1:+.1%} | " if p2 else "— | — | ")
            + f"{g['T0']:.2f} | {g['dT']:.2f} | {g['hold_step']:.3f} | "
            f"{g['refine_step']:.3f} | {g['ramp_hi']:.2f} |")
    L.append("\n### omega_min (bare isolated wells, exact FCI in the 64-point DVR)\n")
    L.append("| leg | well | D01 idle | D12 idle | D01 gate | D12 gate | "
             "min along path (env, gap) |")
    L.append("|---|---|---|---|---|---|---|")
    for leg, o in phys["omega"].items():
        for w, pw in o["per_well"].items():
            L.append(f"| {leg} | w{w} | {pw['d01_idle']:.6f} | "
                     f"{pw['d12_idle']:.6f} | {pw['d01_gate']:.6f} | "
                     f"{pw['d12_gate']:.6f} | {pw['min']:.6f} "
                     f"({pw['env']:.1f}, {pw['gap']}) |")
    L.append("\n| leg | omega_min | binding well / gap | idle end binds? | "
             "10/omega_min | dressed omega_min |")
    L.append("|---|---|---|---|---|---|")
    for leg, o in phys["omega"].items():
        od = phys["omega_dressed"][leg]
        L.append(f"| {leg} | {o['omega_min']:.6f} | w{o['well']} "
                 f"{o['gap']} | {'yes' if o['idle_binds'] else 'NO (env ' + str(o['env']) + ')'}"
                 f" | {C_RAMP/o['omega_min']:.3f} | {od['omega_min']:.6f} "
                 f"(w{od['well']} {od['gap']}, env {od['env']:.1f}) |")
    L.append("\n### Excursion diagnostic (tau_ramp >> bound)\n")
    L.append("| leg | bare, peak slope | dressed, peak slope (well, env, "
             "pair) | dressed, envelope profile | 10/omega_min | "
             "shortest grid ramp |")
    L.append("|---|---|---|---|---|---|")
    for leg, x in phys["excursion"].items():
        b, d = x["bare"], x["dressed"]
        at = d["peak_at"]
        L.append(f"| {leg} | {b['peak']:.3g} | {d['peak']:.3g} (w{at['well']},"
                 f" {at['env']:.1f}, {at['pair']}) | {d['prof']:.3g} | "
                 f"{grids[leg]['ramp_lo']:.3f} | {grids[leg]['ramp'][0]:g} |")
    L.append("\n#### Excursion bound by transition (dressed, peak slope, "
             "max over env and both active wells)\n")
    keys = [f"{m}{n}" for m, n in PAIRS]
    L.append("| leg | " + " | ".join(f"{k[0]}->{k[1]}" for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for leg in phys["excursion"]:
        mx = {k: 0.0 for k in keys}
        for w, rows in phys["iso"]["paths"][leg].items():
            for r in rows:
                if r["kind"] == "dressed":
                    for k in keys:
                        mx[k] = max(mx[k], r["bound_peak"].get(k, 0.0))
        L.append(f"| {leg} | " + " | ".join(f"{mx[k]:.3g}" for k in keys)
                 + " |")
    for leg, g in grids.items():
        L.append(f"\n### Grid {leg}" + (" — DEFERRED (not swept)"
                                        if g["deferred"] else "") + "\n")
        L.append(f"Delta {g['Delta']:.6e}, T_0 {g['T0']:.3f}, dT "
                 f"{g['dT']:.3f}, step {g['hold_step']:.3f}, ramp "
                 f"[{g['ramp_lo']:.3f}, {g['ramp_hi']:.3f}] x"
                 f"{g['ramp_factor']:.3f}; ramp exact "
                 + ", ".join(f"{v:.3f}" for v in g["ramp_exact"])
                 + f"; max snap shift ramp {g['ramp_max_shift']:.4f}, T_eff "
                 f"{g['Teff_max_shift']:.4f}; {g['n_points']} points"
                 f" ({len(g['dropped'])} dropped, tau_hold < {TH_MIN:g}).\n")
        L.append("| tau_ramp \\ T_eff | " + " | ".join(f"{T:g}" for T in
                                                        g["Teff"]) + " |")
        L.append("|---|" + "---|" * len(g["Teff"]))
        for tr in g["ramp"]:
            cells = []
            for T in g["Teff"]:
                th = round(T - tr, 10)
                cells.append(f"{th:g}" if th >= TH_MIN else "—")
            L.append(f"| **{tr:g}** | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def grids_from_phys(phys, block, omega_kind="bare"):
    grids = {}
    for leg in LEGS:
        D = phys["delta"][leg]["Delta"]
        wm = phys["omega" if omega_kind == "bare" else "omega_dressed"][leg]
        grids[leg] = build_grid(leg, D, wm["omega_min"], block)
    return grids


def write_outputs(phys, block, omega_kind, job):
    grids = grids_from_phys(phys, block, omega_kind)
    for leg, g in grids.items():
        print(f"  [{leg}{' DEFERRED' if g['deferred'] else ''}] Delta "
              f"{g['Delta']:.6e} T0 {g['T0']:.3f} dT {g['dT']:.3f} step "
              f"{g['hold_step']:.3f} refine {g['refine_step']:.3f}\n"
              f"      ramp [{g['ramp_lo']:.3f}, {g['ramp_hi']:.3f}] exact "
              + " ".join(f"{v:.3f}" for v in g["ramp_exact"])
              + "\n      snapped " + " ".join(f"{v:g}" for v in g["ramp"])
              + f"   T_eff {g['Teff'][0]:g} .. {g['Teff'][-1]:g}"
              f"   max shift ramp {g['ramp_max_shift']:.4f} T_eff "
              f"{g['Teff_max_shift']:.4f}   {g['n_points']} points, "
              f"{len(g['dropped'])} dropped", flush=True)
    src = sorted({phys["idle"]["file"]}
                 | {v for d in phys["delta"].values()
                    for v in d["files"].values()})
    doc = dict(
        description="MF sweep grid: the rules of Secs. S5 A-C with the "
                    "three-electron change of Sec. S5 F. alpha = 1: "
                    "tau_hold = T_eff - tau_ramp; k = 0 only.",
        block=block, eps_span=1e-2, eps_refine=1e-3, k_range=[-6, 6],
        n_ramp=N_RAMP, c_ramp=C_RAMP, tau_hold_min=TH_MIN,
        omega_source=omega_kind,
        snapping="tau_ramp rounded UP (keeps >= c/omega_min), T_eff to "
                 "nearest block",
        legs={leg: dict(
            g, omega=phys["omega"][leg], omega_dressed=phys["omega_dressed"][leg],
            excursion=phys["excursion"][leg],
            Delta_detail={k: v for k, v in phys["delta"][leg].items()
                          if k not in ("files",)})
            for leg, g in grids.items()},
        provenance=dict(
            script="gate_timing/timing.py",
            physics_json=os.path.join(OUT_DIR, "physics.json"),
            source_files=src,
            method="supplementary material, Sec. S5 (gate timing)",
            created=time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    with open(GRIDS_JSON, "w") as f:
        json.dump(_jsonable(doc), f, indent=1)
    with open(os.path.join(OUT_DIR, "tables.md"), "w") as f:
        f.write(tables_md(phys, grids))
    print(f"\n  wrote {GRIDS_JSON}\n  wrote {os.path.join(OUT_DIR, 'tables.md')}",
          flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=float, default=0.1,
                   help="block grid in t.u. (dt x regauge interval)")
    p.add_argument("--omega", choices=("bare", "dressed"), default="bare")
    p.add_argument("--grid-only", action="store_true")
    p.add_argument("--norb-iso", type=int, default=14)
    p.add_argument("--iso-dt", type=float, default=2e-3)
    p.add_argument("--iso-chunk", type=int, default=500)
    p.add_argument("--iso-max-steps", type=int, default=8000,
                   help="norb-14 MCTDHF idle check; 0 skips it")
    p.add_argument("--iso-tol", type=float, default=1e-10)
    a = p.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID")
    print(f"jax {jax.__version__} {jax.devices()}  job {job}", flush=True)
    phys_path = os.path.join(OUT_DIR, "physics.json")

    if a.grid_only:
        with open(phys_path) as f:
            phys = json.load(f)
        write_outputs(phys, a.block, a.omega, job)
        return

    if os.path.exists(phys_path):
        import shutil
        shutil.copyfile(phys_path, phys_path + ".prev")
    t_all = time.perf_counter()
    print("\n=== 1. ML anchors: verification and Delta ===", flush=True)
    sv = ml_solver()
    idle, x_idle, ml_dens = verify_idle(sv)
    idle["file"] = IDLE_STATE
    delta = {leg: delta_leg(sv, leg) for leg in LEGS}
    x_targets = {leg: np.asarray(d["best_x_ghz"]) / fw.TYPICAL_FREQ_GHZ
                 for leg, d in delta.items()}

    print("\n=== 2. isolated wells along the ramp (exact FCI, 64-point "
          "DVR) ===", flush=True)
    t0 = time.perf_counter()
    fci = FCIWell()
    print(f"  FCI tables: D = {fci.D}, {len(fci.rows)} hops "
          f"[{time.perf_counter() - t0:.1f} s]", flush=True)
    iso_res = isolated_study(fci, x_idle, x_targets, ml_dens)

    phys = dict(
        idle=idle, delta=delta,
        iso=dict(method="exact FCI, 64-point sine DVR, 3 e-",
                 envs=ENVS.tolist(), **iso_res),
        omega={leg: omega_summary(iso_res, leg, "bare") for leg in LEGS},
        omega_dressed={leg: omega_summary(iso_res, leg, "dressed")
                       for leg in LEGS},
        excursion={leg: excursion_summary(iso_res, leg) for leg in LEGS},
        wall=time.perf_counter() - t_all,
    )
    with open(phys_path, "w") as f:
        json.dump(_jsonable(phys), f, indent=1)
    print(f"\n  wrote {phys_path}", flush=True)
    for leg in LEGS:
        o, od, ex = (phys["omega"][leg], phys["omega_dressed"][leg],
                     phys["excursion"][leg])
        print(f"  [{leg}] omega_min bare {o['omega_min']:.6f} (w{o['well']} "
              f"{o['gap']} env {o['env']:.1f}, idle binds {o['idle_binds']})"
              f"  dressed {od['omega_min']:.6f} (w{od['well']} {od['gap']} "
              f"env {od['env']:.1f})\n      excursion bound bare "
              f"{ex['bare']['peak']:.3g}  dressed {ex['dressed']['peak']:.3g}"
              f" at {ex['dressed']['peak_at']}  (profile "
              f"{ex['dressed']['prof']:.3g})", flush=True)

    print("\n=== 3. grids ===", flush=True)
    write_outputs(phys, a.block, a.omega, job)

    old = None
    if a.iso_max_steps == 0 and os.path.exists(phys_path + ".prev"):
        with open(phys_path + ".prev") as f:
            old = json.load(f).get("mctdhf_check")
    if old is not None:        # carry the (slow) idle check over unchanged
        phys["mctdhf_check"] = dict(old, carried_over_from=phys_path + ".prev")
        with open(phys_path, "w") as f:
            json.dump(_jsonable(phys), f, indent=1)
        write_outputs(phys, a.block, a.omega, job)
    if a.iso_max_steps > 0:
        print(f"\n=== 4. norb {a.norb_iso} SA-3 MCTDHF vs FCI"
              " at idle ===", flush=True)
        iso = IsoWell(a.norb_iso, a.iso_dt, a.iso_chunk, a.iso_max_steps,
                      a.iso_tol)
        phys["mctdhf_check"] = dict(
            norb=a.norb_iso, dt=a.iso_dt, chunk=a.iso_chunk, tol=a.iso_tol,
            max_steps=a.iso_max_steps,
            wells=mctdhf_check(iso, fci, x_idle))
        phys["wall"] = time.perf_counter() - t_all
        with open(phys_path, "w") as f:
            json.dump(_jsonable(phys), f, indent=1)
        write_outputs(phys, a.block, a.omega, job)
    print(f"\n  total wall {time.perf_counter() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
