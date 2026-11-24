"""Mean-field (Hartree–Fock spectator) ML-MCTDHF gate legs, 3 e-/well.

In short:

* the two ACTIVE wells of a leg keep the idle operating point's ML budget
  (norb 14, n_s 4 — the ``operating_points`` idle states), the two SPECTATOR wells are
  cut down to a single determinant (norb 3, n_s 1: Hartree–Fock) by default.
  The spectator budget is a parameter (``Layout``), so norb 14 / n_s 1 can be
  compared later;
* four pair states (|00>, |10>, |01>, |11> of the active pair, spectators in
  |0>) are propagated together, ``num_states = 4`` with equal weights, one
  shared set of orbitals and SPFs;
* the same four states at t = 0 are the anchors: ``M_ij = <anchor_i|psi_j(T)>``.

Everything that defines the discretisation of a run lives in
:class:`Integ`; ``sweep.py`` stores it in every output and refuses to restart a
segment from a checkpoint written with a different one.

x64 is switched on at import, before any array exists.
"""

import dataclasses
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mctdhf.MLMCTDHF_n_species import (  # noqa: E402
    MLMCTDHF_n_species_DVR, make_regauge_ml)
from mctdhf.integrators_ml import get_ml_cmf_midpoint_integrator  # noqa: E402

from system import fourwell as fw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
ML_OPT_RESULTS = os.path.abspath(os.path.join(HERE, "..", "operating_points", "results"))

IDLE_PATH = os.path.join(ML_OPT_RESULTS, "idle_states_sa16_ns4_norb14.npz")
# The resonance anchors: the 16-state reruns of the target stages (best_x).
_STAGE = {"w0w1": 2, "w1w2": 3, "w2w3": 4, "w0w3": 5}
LEGS = {"w0w1": (0, 1), "w1w2": (1, 2), "w2w3": (2, 3), "w0w3": (0, 3)}
WELLS = tuple(range(fw.NUM_WELLS))

# Basis order of M's rows and columns, (active[0], active[1]) bits, as in
# one_electron/lib/gate_fidelity.py.
PAIR_LABELS = ["|00>", "|10>", "|01>", "|11>"]
PAIR_BITS = [(0, 0), (1, 0), (0, 1), (1, 1)]
# State-averaging weights of the four pair states, PAIR_BITS order: 2^-n by
# excitation number. Non-degenerate on each active well (rho = 2/3, 1/3), so
# the natural-SPF gauge that defines |0>/|1> is fixed (operating_points relax_set "16"
# uses the same rule); equal weights would make rho = (1/2, 1/2) and the
# identity matching of ITE_opt ill-defined.
RELAX_WEIGHTS = (1.0, 0.5, 0.5, 0.25)
MF_STATE_DIR = os.path.join(RESULTS_DIR, "mf_states")


def target_path(leg):
    st = _STAGE[leg]
    return os.path.join(ML_OPT_RESULTS,
                        f"ml_sa16_20260924_target_stage{st}_superpos_{leg}.npz")


def x_vectors(leg):
    """``(x_idle, x_target)`` in GHz. Only the active entries may differ."""
    with np.load(IDLE_PATH) as d:
        xi = np.asarray(d["x_ghz"], dtype=np.float64)
    with np.load(target_path(leg), allow_pickle=True) as d:
        xg = np.asarray(d["best_x"], dtype=np.float64)
    i, j = LEGS[leg]
    for w in WELLS:
        if w not in (i, j):
            assert xg[w] == xi[w] and xg[4 + w] == xi[4 + w], (
                f"leg {leg}: target moves spectator well {w} "
                f"({xi[[w, 4 + w]]} -> {xg[[w, 4 + w]]})")
    return xi, xg


# ── the pulse ─────────────────────────────────────────────────────────────

def make_env(tau_ramp, tau_hold, kind="full"):
    """``full`` = the genuine erf ramp-hold-ramp; ``up`` = ramp-up + plateau.

    The two agree to machine precision
    for ``t <= t_c = tau_hold`` (the ramp-down erf argument is <= -9 there,
    and erf(-9) = -1 exactly in float64), which is what makes the segment-A
    checkpoint the genuine leg's state (``sweep.py`` docstring).
    """
    from jax.scipy.special import erf as jerf
    t_up = 0.5 * tau_ramp
    t_dn = 1.5 * tau_ramp + tau_hold
    if kind == "up":
        def env(t):
            return 0.5 * (1.0 + jerf(6.0 * (t - t_up) / tau_ramp))
    elif kind == "full":
        def env(t):
            return 0.5 * (jerf(6.0 * (t - t_up) / tau_ramp)
                          - jerf(6.0 * (t - t_dn) / tau_ramp))
    elif kind == "idle":
        def env(t):
            return 0.0 * t
    else:
        raise ValueError(kind)
    return env, 2.0 * tau_ramp + tau_hold


def h_list_for(leg, env):
    xi, xg = x_vectors(leg)
    return fw.h_ramp(xi / fw.TYPICAL_FREQ_GHZ, xg / fw.TYPICAL_FREQ_GHZ, env)


# ── budgets and integrator settings ───────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Layout:
    """Per-leg orbital / SPF budget: active wells vs spectators; the
    real-time state-averaging weights; and which initial states are used.

    ``init``: ``"relaxed"`` = the four pair states relaxed by ``ITE_opt`` in
    this layout at idle (``relax_mf_states``; the projected
    states fail the idle check), ``"projected"`` = the
    HF projection of the idle ML states (``projected_states``).
    ``weights``: ``"relax"`` = ``RELAX_WEIGHTS`` (the relaxation's own, so the
    relaxed states are stationary), ``"equal"`` = 1/4 each.
    """
    norb_active: int = 14
    nspf_active: int = 4
    norb_spec: int = 3
    nspf_spec: int = 1
    init: str = "relaxed"
    weights: str = "relax"
    init_tag: str = ""          # which relaxed-state file (relaxed_path tag)

    def state_weights(self):
        if self.weights == "relax":
            w = np.asarray(RELAX_WEIGHTS, dtype=np.float64)
        elif self.weights == "equal":
            w = np.ones(4)
        else:
            raise ValueError(self.weights)
        return w / w.sum()

    def tag(self, leg=None):
        """Suffix of the budget/model part of a run directory ("" = default)."""
        d = Layout()
        parts = []
        if (self.norb_spec, self.nspf_spec) != (d.norb_spec, d.nspf_spec):
            parts.append(f"spec{self.norb_spec}s{self.nspf_spec}")
        if (self.norb_active, self.nspf_active) != (d.norb_active, d.nspf_active):
            parts.append(f"act{self.norb_active}s{self.nspf_active}")
        if self.init != d.init:
            parts.append(f"init{self.init}")
        if self.weights != d.weights:
            parts.append(f"w{self.weights}")
        if self.init_tag:
            parts.append(f"st{self.init_tag.lstrip('_')}")
        return "".join("_" + p for p in parts)

    def norbs(self, leg):
        a = LEGS[leg]
        return [self.norb_active if w in a else self.norb_spec for w in WELLS]

    def nspfs(self, leg):
        a = LEGS[leg]
        return [self.nspf_active if w in a else self.nspf_spec for w in WELLS]


@dataclasses.dataclass(frozen=True)
class Integ:
    """The whole discretisation of a run. Midpoint CMF only.

    ``precision`` picks the SPF and orbital propagator: ``x64`` = ``cn``,
    ``mixed`` = ``cn_mixed`` (defect-form CN, inner solve in complex64);
    ``cn_tol`` is the x64 CN tolerance, ``inner_tol`` the mixed inner one.
    ``regauge_t`` is held fixed in TIME.
    """
    # Defaults from the integrator study (integ_study.py, Sec. S6 B): midpoint CMF, diagonalize top, cn_mixed 1e-4 on SPF and
    # orbital layers, dt 1e-2 with 16 SPF substeps (6.25e-4) and 4 orbital
    # substeps (2.5e-3) — the error is set by the substep lengths, not by
    # dt — regauge 0.1 t.u., per-step Löwdin renormalisation.
    dt: float = 1e-2
    a_prop: str = "diagonalize"
    krylov: int = 10
    precision: str = "mixed"
    cn_tol: float = 1e-10
    inner_tol: float = 1e-4
    max_iter: int = 200
    nsub_spf: int = 16
    nsub_orb: int = 4
    regauge_t: float = 0.1
    # Explicit SPF / orbital propagators; "" = derived from `precision`
    # (x64 -> cn, mixed -> cn_mixed). 'diagonalize' and 'dop' run in x64.
    B_prop: str = ""
    b_prop: str = ""
    dop_sub: int = 4
    # "none" | "step": the library integrator's renorm option (Löwdin
    # renormalisation of b_s and B_s after every step).
    renorm: str = "step"

    @property
    def every(self):
        n = int(round(self.regauge_t / self.dt))
        assert n >= 1 and abs(n * self.dt - self.regauge_t) < 1e-12, (
            f"regauge_t {self.regauge_t} is not an integer number of steps "
            f"of dt {self.dt}")
        return n

    @property
    def block_t(self):
        return self.dt * self.every

    def as_dict(self):
        return dataclasses.asdict(self)

    def tag(self):
        a = self.a_prop + (str(self.krylov) if self.a_prop == "sil" else "")
        p = ("x64" if self.precision == "x64"
             else f"mix{self.inner_tol:.0e}".replace("e-0", "e-"))
        s = f"_s{self.nsub_spf}o{self.nsub_orb}" if (
            self.nsub_spf, self.nsub_orb) != (1, 1) else ""
        props = ""
        if self.B_prop or self.b_prop:
            props = f"_B{self.B_prop or 'def'}_b{self.b_prop or 'def'}"
            if "dop" in (self.B_prop, self.b_prop):
                props += f"{self.dop_sub}"
        rn = "" if self.renorm == "none" else f"_rn{self.renorm}"
        return (f"{a}_{p}{props}_dt{self.dt:g}_tol{self.cn_tol:.0e}"
                f"_rg{self.regauge_t:g}{s}{rn}").replace("e-0", "e-")

    def integrator_kwargs(self):
        if self.a_prop == "sil":
            a_kw = {"krylov_dim": self.krylov}
        elif self.a_prop == "cn":
            a_kw = {"solver_tol": self.cn_tol, "max_solver_iter": self.max_iter}
        else:
            a_kw = {}
        if self.precision not in ("x64", "mixed"):
            raise ValueError(self.precision)
        default = "cn" if self.precision == "x64" else "cn_mixed"
        B_prop = self.B_prop or default
        b_prop = self.b_prop or default

        def kw(prop):
            if prop == "cn":
                return {"solver_tol": self.cn_tol,
                        "max_solver_iter": self.max_iter}
            if prop == "cn_mixed":
                return {"solver_tol": self.inner_tol,
                        "max_solver_iter": self.max_iter}
            if prop == "diagonalize":
                return {"orbital_x64": True}
            if prop == "dop":
                return {"n_substeps": self.dop_sub}
            raise ValueError(prop)

        B_kw = dict(kw(B_prop), n_spf_substeps=self.nsub_spf)
        b_kw = dict(kw(b_prop), n_orbital_substeps=self.nsub_orb)
        return dict(a_propagator=self.a_prop, B_propagator=B_prop,
                    b_propagator=b_prop, a_kwargs=a_kw, B_kwargs=B_kw,
                    b_kwargs=b_kw)


def make_ml(leg, h_list, layout=Layout(), num_states=4, ci_sparse=True):
    w = layout.state_weights()
    ml = MLMCTDHF_n_species_DVR(
        h_list=h_list, g_dict=fw.g_dict(),
        num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_mctdhf_orbitals=layout.norbs(leg),
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_spfs=layout.nspfs(leg),
        num_states=num_states, imag_time=False, eps=1e-6, e_shift=0.0,
        state_weights=w, ci_sparse=ci_sparse)
    # The base class stores weights in float32 (1/2.25 rounded at 3e-8);
    # ITE_opt replaces them by float64 the same way, and the relaxed states
    # are only stationary under the weights they were relaxed with. They are
    # read at trace time, so replacing them before any trace is safe.
    ml.state_weights = jnp.asarray(w)
    return ml


# ── MF initial states ─────────────────────────────────────────────────────

def _load_idle():
    d = np.load(IDLE_PATH)
    K = fw.NUM_WELLS
    return (np.asarray(d["A"]), [np.asarray(d[f"B{s}"]) for s in range(K)],
            [np.asarray(d[f"b{s}"]) for s in range(K)],
            np.asarray(d["configs"]), int(d["norb"]), int(d["n_spf"]))


def _move(A, U, ax):
    """``A[.., m, ..] <- sum_j A[.., j, ..] U[j, m]`` on axis ``ax``."""
    return np.moveaxis(np.tensordot(A, U, axes=([ax], [0])), -1, ax)


def _gram(A):
    """``S[a, b] = <A_a|A_b>`` over all non-state axes."""
    X = A.reshape(A.shape[0], -1)
    return X.conj() @ X.T


def _node_density(A, ax, w):
    """``rho[j, k] = sum_a w_a sum_rest A[a..j..] conj(A[a..k..])``.

    Ordered so that its eigenvectors ARE the natural-SPF coefficients:
    ``v^dagger rho v`` is the weight on ``sum_j v_j |j>``.
    """
    X = np.moveaxis(A, ax, 1).reshape(A.shape[0], A.shape[ax], -1)
    return np.einsum("a,ajr,akr->jk", w, X, X.conj())


_CI_IDLE = {}


def _idle_ci(norb):
    if norb not in _CI_IDLE:
        from mctdhf.SemiDirectCI_ml import SemiDirectCI_ML
        _CI_IDLE[norb] = SemiDirectCI_ML([norb] * fw.NUM_WELLS,
                                         [fw.NUM_PARTICLES] * fw.NUM_WELLS,
                                         sparse=True)
    return _CI_IDLE[norb]


def pair_configs(leg):
    """The four pair configurations (one level per well), PAIR_BITS order."""
    act = LEGS[leg]
    out = []
    for bits in PAIR_BITS:
        cfg = [0] * fw.NUM_WELLS
        cfg[act[0]], cfg[act[1]] = bits
        out.append(tuple(cfg))
    return out


def relaxed_path(leg, layout, tag=""):
    return os.path.join(MF_STATE_DIR,
                        f"mf_{leg}_n{''.join(map(str, layout.norbs(leg)))}"
                        f"_s{''.join(map(str, layout.nspfs(leg)))}{tag}.npz")


def mf_initial_states(leg, ml, layout=Layout(), verbose=True):
    """The four MF pair states (and anchors) in ``ml``'s layout: the relaxed
    ones (default) or the HF projection, per ``layout.init``."""
    if layout.init == "projected":
        return projected_states(leg, ml, layout, verbose)
    if layout.init != "relaxed":
        raise ValueError(layout.init)
    path = relaxed_path(leg, layout, layout.init_tag)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing: run `sweep.py relax --leg {leg}` first")
    d = np.load(path)
    A = jnp.asarray(d["A"])
    Bs = [jnp.asarray(d[f"B{s}"]) for s in WELLS]
    bs = [jnp.asarray(d[f"b{s}"]) for s in WELLS]
    assert A.shape == ml.a_shape, (A.shape, ml.a_shape)
    for s in WELLS:
        assert Bs[s].shape == ml.B_shapes[s] and bs[s].shape == ml.b_shapes[s]
    y0 = ml._ABb_to_y(A, Bs, bs)
    M0 = np.asarray(ml.calculate_overlap(y0, y0))
    err = float(np.max(np.abs(M0 - np.eye(4))))
    info = {"source": path, "M0_err": err,
            "relax_E": np.asarray(d["E"]).tolist()}
    if verbose:
        print(f"  [mf] relaxed states {os.path.basename(path)}: M(0) - 1 = "
              f"{err:.2e}", flush=True)
    assert err < 1e-10, f"M(0) != 1: {err:.3e}"
    return y0, info


def relax_mf_states(leg, layout=Layout(), dt=2e-2, inner_tol=1e-2,
                    tol=1e-10, max_steps=40000, verbose=True,
                    precision="mixed", out_tag=""):
    """Relax the four pair states in the MF layout at the IDLE point.

    ``MLMCTDHF_n_species_ITE_opt`` with ``relax_configs`` = the pair
    configurations and ``RELAX_WEIGHTS``, started from the HF projection.
    Afterwards the top block is re-diagonalised at the final ``(B, b)`` so
    each row is an exact eigenvector (the last ITE iteration leaves A one
    layer step behind), then an exact regauge (``make_regauge_ml``) makes
    every layer orthonormal to rounding. Saved to ``relaxed_path``.
    """
    from mctdhf.MLMCTDHF_n_species_ITE_opt import MLMCTDHF_n_species_ITE_opt
    xi, _ = x_vectors(leg)
    h_idle = fw.h_static(xi / fw.TYPICAL_FREQ_GHZ)
    cfgs = pair_configs(leg)
    ite = MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(),
        num_particles=[fw.NUM_PARTICLES] * fw.NUM_WELLS,
        num_mctdhf_orbitals=layout.norbs(leg),
        num_spatial_orbitals=[fw.NUM_DVR] * fw.NUM_WELLS,
        num_spfs=layout.nspfs(leg), relax_configs=cfgs,
        relax_weights=list(RELAX_WEIGHTS), all_configs=cfgs, dt=dt,
        precision=precision, inner_tol=inner_tol, chunk=50, eps=1e-6)

    proj = dataclasses.replace(layout, init="projected")
    ml = make_ml(leg, h_idle, proj)
    y_p, info_p = projected_states(leg, ml, proj, verbose=verbose)
    E_p = np.real(np.asarray(ml.calculate_energy(0.0, y_p)))
    state = tuple(ml._y_to_ABb(y_p))
    state = (state[0], list(state[1]), list(state[2]))

    t0 = time.time()
    state, e_sum, info = ite.relax_to_tol(h_idle, state, tol=tol,
                                          max_steps=max_steps,
                                          verbose=verbose)
    wall = time.time() - t0
    # exact eigenvectors of the final top block, matched by identity
    h = ite.as_h(h_idle)
    A, Bs, bs = state
    Bs_rg, evals, evecs, ovl = ite._front_match(A, Bs, bs, h)
    cols, pop = ite._assign(ovl)
    A = evecs[:, jnp.asarray(cols)].T.reshape(ml.a_shape)
    y = make_regauge_ml(ml)(ml._ABb_to_y(A, list(Bs_rg), list(bs)))
    E = np.real(np.asarray(ml.calculate_energy(0.0, y)))
    M0 = np.asarray(ml.calculate_overlap(y, y))
    err = float(np.max(np.abs(M0 - np.eye(4))))
    assert err < 1e-10, f"relaxed M(0) != 1: {err:.3e}"
    # overlap of the relaxed states with the projected ones
    S = np.abs(np.diag(np.asarray(ml.calculate_overlap(y_p, y))))
    A, Bs, bs = ml._y_to_ABb(y)
    path = relaxed_path(leg, layout, out_tag)
    save_npz(path, A=np.asarray(A),
             **{f"B{s}": np.asarray(Bs[s]) for s in WELLS},
             **{f"b{s}": np.asarray(bs[s]) for s in WELLS},
             E=E, E_projected=E_p, E_top=np.asarray(evals)[cols],
             match_pop=np.asarray(pop), overlap_with_projected=S,
             e_sum=e_sum, converged=bool(info["converged"]),
             steps=int(info["steps"]), fallback_chunks=np.asarray(
                 info["fallback_chunks"], dtype=int),
             relax_weights=np.asarray(RELAX_WEIGHTS), configs=np.asarray(cfgs),
             dt=dt, inner_tol=inner_tol, tol=tol, wall=wall, M0_err=err,
             precision=precision,
             x_idle=xi, layout=json.dumps(dataclasses.asdict(layout)))
    if verbose:
        print(f"  [relax] {leg}: converged {info['converged']} after "
              f"{info['steps']} it ({wall:.0f} s), fallbacks "
              f"{info['fallback_chunks']}", flush=True)
        print("  [relax] E projected " + " ".join(f"{v:.8f}" for v in E_p),
              flush=True)
        print("  [relax] E relaxed   " + " ".join(f"{v:.8f}" for v in E),
              flush=True)
        print("  [relax] |<projected|relaxed>| " + " ".join(
            f"{v:.6f}" for v in S) + f"   match pop " + " ".join(
            f"{v:.6f}" for v in np.asarray(pop)) + f"   M(0)-1 {err:.1e}",
              flush=True)
        print(f"  saved -> {path}", flush=True)
    return y, info


def projected_states(leg, ml, layout=Layout(), verbose=True):
    """The four MF pair states in ``ml``'s layout, and ``info``.

    1. The 4 rows of the idle ``A`` for the pair configurations, spectators
       in |0>.
    2. Per spectator, the 4-state-averaged 1-RDM; keep its top ``norb_spec``
       natural orbitals, ``b_s <- b_s U[:, :m]``.
    3. Project the spectator's SPFs onto the new determinant space. With the
       old orbitals ``phi_p = sum_a conj(U[p, a]) psi_a + (discarded)``, a
       determinant expands as ``|Phi_I> = sum_K det(R[occ'_K, occ_I]) |Psi_K>``
       with ``R = U^dagger[:m, :]`` — the N-th compound matrix, the same
       construction as ``make_regauge_ml``. Keep the top ``nspf_spec``
       natural SPFs of the projected state (for norb 3 there is ONE
       determinant and the SPF is it).
    4. Löwdin-orthonormalise the four states; M(0) = 1 to < 1e-10, asserted.

    Active wells keep the idle ``b`` and ``B`` unchanged; that requires the
    layout's active budget to equal the idle file's.
    """
    A16, B16, b16, configs, norb_idle, nspf_idle = _load_idle()
    act = LEGS[leg]
    assert layout.norb_active == norb_idle and layout.nspf_active == nspf_idle, (
        f"active budget {layout.norb_active}/{layout.nspf_active} differs "
        f"from the idle states' {norb_idle}/{nspf_idle}")
    rows = []
    for bits in PAIR_BITS:
        cfg = [0] * fw.NUM_WELLS
        cfg[act[0]], cfg[act[1]] = bits
        (r,) = np.nonzero((configs == np.array(cfg)).all(axis=1))[0]
        rows.append(int(r))
    A = A16[rows]                                      # (4, n0, n1, n2, n3)
    w = np.full(4, 0.25)
    S_in = _gram(A)
    info = {"rows": rows, "idle_row_orthonormality":
            float(np.max(np.abs(S_in - np.eye(4))))}

    # The idle states' own CI tables (norb_idle on EVERY well): the MF
    # solver's dCI has the spectators at norb_spec and cannot take B16.
    ci_idle = _idle_ci(norb_idle)
    old_occ = np.array([np.flatnonzero(d) for d in np.asarray(ci_idle.I[0])])
    # Spectator 1-RDMs from the UNPROJECTED rows (A is overwritten below).
    Ds = [np.asarray(D) for D in ci_idle.ml_1p_RDMs(
        jnp.asarray(A), [jnp.asarray(B) for B in B16], weights=jnp.asarray(w))]
    Bs, bs = [], []
    for s in WELLS:
        if s in act:
            Bs.append(B16[s])
            bs.append(b16[s])
            continue
        m = layout.norb_spec
        D = Ds[s]                          # 4-state-averaged 1-RDM
        occ_ev, V = np.linalg.eigh(D)
        order = np.argsort(occ_ev)[::-1]
        occ_ev, V = occ_ev[order], V[:, order]
        new_occ = np.array([np.flatnonzero(d) for d in np.asarray(
            ml.dCI.I[s])])
        assert len(new_occ) == ml.num_dets[s]
        # The RDM's index convention decides between V and conj(V) for the
        # natural-orbital coefficients; take the one that retains more
        # weight (they coincide for real states) and record both.
        best = None
        for label, U in (("V", V), ("conjV", V.conj())):
            R = U.conj().T[:m, :]                         # (m, M_old)
            sub = R[new_occ[:, None, :, None], old_occ[None, :, None, :]]
            Mc = np.linalg.det(sub)                       # (D_new, D_old)
            Bp = Mc @ B16[s]                              # (D_new, n_old)
            Ap = _move(A, Bp.T, s + 1)                    # SPF axis -> dets
            ret = float(np.diag(_gram(Ap)).real.mean())
            info[f"ret_{label}_w{s}"] = ret
            if best is None or ret > best[0]:
                best = (ret, label, U, Ap)
        ret, label, U, Ap = best
        rho = _node_density(Ap, s + 1, w)                 # (D_new, D_new)
        ev, Vs = np.linalg.eigh(rho)
        Vs = Vs[:, np.argsort(ev)[::-1][:layout.nspf_spec]]
        # A <- A x_s Vs^dagger, B_s = Vs (orthonormal columns)
        Ap = _move(Ap, Vs.conj(), s + 1)
        A = Ap
        Bs.append(Vs)
        bs.append(b16[s] @ U[:, :m])
        info[f"nat_occ_w{s}"] = occ_ev[:m + 1].tolist()
        info[f"convention_w{s}"] = label
        info[f"retained_w{s}"] = ret
        if verbose:
            print(f"  [mf] spectator w{s}: top-{m} natural occupations "
                  + " ".join(f"{v:.6f}" for v in occ_ev[:m + 1])
                  + f"  retained weight {ret:.6f} ({label})", flush=True)

    S = _gram(A)
    info["norms_before_lowdin"] = np.sqrt(np.diag(S).real).tolist()
    info["offdiag_before_lowdin"] = float(np.max(np.abs(S - np.diag(np.diag(S)))))
    ev, U = np.linalg.eigh(S)
    X = U @ np.diag(ev ** -0.5) @ U.conj().T              # S^{-1/2}
    A = np.einsum("ba,b...->a...", X, A)

    A = jnp.asarray(A, dtype=jnp.complex128)
    Bs = [jnp.asarray(B, dtype=jnp.complex128) for B in Bs]
    bs = [jnp.asarray(b, dtype=jnp.complex128) for b in bs]
    for s in WELLS:
        assert Bs[s].shape == ml.B_shapes[s] and bs[s].shape == ml.b_shapes[s]
    y0 = ml._ABb_to_y(A, Bs, bs)
    M0 = np.asarray(ml.calculate_overlap(y0, y0))
    err = float(np.max(np.abs(M0 - np.eye(4))))
    info["M0_err"] = err
    if verbose:
        print(f"  [mf] norms before Löwdin "
              + " ".join(f"{v:.6f}" for v in info["norms_before_lowdin"])
              + f", max offdiag {info['offdiag_before_lowdin']:.2e};"
              f"  M(0) - 1 = {err:.2e}", flush=True)
    assert err < 1e-10, f"M(0) != 1: {err:.3e}"
    return y0, info


# ── fidelity, as in the one-electron code ──────────────────────────────────
# _localZ and gate_fidelity_up_to_localZ as in one_electron/lib/gate_fidelity.py;
# M_REF (sqrt(iSWAP)), M_NULL and fbar (average gate fidelity). Basis order of M:
#   (|00>, |10>, |01>, |11>).

def _localZ(alpha, beta):
    """Single-qubit Z⊗Z phase matrix in basis order (|00>,|10>,|01>,|11>)."""
    return np.diag([1.0, np.exp(1j * alpha), np.exp(1j * beta),
                    np.exp(1j * (alpha + beta))])


def gate_fidelity_up_to_localZ(M_ref, M):
    """Normalised gate overlap maximised over local Z⊗Z (both sides) + global
    phase: F = max |Tr(M_ref† Z_L M Z_R)|² / (‖M_ref‖_F² ‖M‖_F²). → 1 iff the
    gates agree up to single-qubit Z's. The idle dynamical phases are exactly
    such a local Z⊗Z, so this removes that trivial gauge."""
    from scipy.optimize import minimize
    if not np.all(np.isfinite(M)):
        return np.nan
    nref = np.vdot(M_ref, M_ref).real
    nm = np.vdot(M, M).real

    def neg(x):
        t = np.trace(M_ref.conj().T @ (_localZ(x[0], x[1]) @ M @ _localZ(x[2], x[3])))
        return -(abs(t) ** 2) / (nref * nm)

    best = None
    rng = np.random.default_rng(0)
    for _ in range(30):
        r = minimize(neg, rng.uniform(0, 2 * np.pi, 4), method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-12, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return float(-best.fun)


_S2 = 1.0 / np.sqrt(2.0)
M_REF = np.array([[1, 0, 0, 0], [0, _S2, 1j*_S2, 0],
                  [0, 1j*_S2, _S2, 0], [0, 0, 0, 1]], dtype=complex)
M_NULL = np.eye(4, dtype=complex)


def fbar(M_ref, M):
    d = M_ref.shape[0]
    nm = float(np.linalg.norm(M) ** 2)
    fxc = gate_fidelity_up_to_localZ(M_ref, M)
    return (nm + fxc * float(np.linalg.norm(M_ref) ** 2) * nm) / (d * (d + 1.)), fxc

# ── end of the fidelity helpers ───────────────────────────────────────────


def gate_matrix(ml, y_anchor, y):
    """``M_raw[i, j] = <anchor_i|psi_j>``, the column-normalised ``Mn``,
    the norms and the column retention."""
    M = np.asarray(ml.calculate_overlap(y_anchor, y))
    nrm = np.abs(np.diag(np.asarray(ml.calculate_overlap(y, y))))
    Mn = M / np.sqrt(nrm)[None, :]
    return Mn, M, nrm, np.sum(np.abs(Mn) ** 2, axis=0)


def gate_metrics(ml, y_anchor, y, T_eff, Delta=None, fid=True):
    """Everything ``pt_*.npz`` stores about the final state."""
    Mn, M, nrm, ret = gate_matrix(ml, y_anchor, y)
    out = dict(M=Mn, M_raw=M, norms=nrm, retention=ret,
               retention_over_1=bool(np.any(ret > 1.0 + 1e-9)),
               spf_pops=[np.asarray(p) for p in ml.spf_populations(y)])
    theta = 2.0 * np.arctan2(abs(Mn[2, 1]), abs(Mn[1, 1]))
    out.update(theta=theta, Delta_eff=theta / T_eff if T_eff else np.nan,
               transfer=float(abs(Mn[2, 1]) ** 2))
    if Delta:
        out["enhancement"] = out["Delta_eff"] / Delta - 1.0
    if fid:
        F, fxc = fbar(M_REF, Mn)
        Fid, fxc_id = fbar(M_NULL, Mn)
        out.update(F=F, Fxc=fxc, F_identity=Fid, Fxc_identity=fxc_id)
    return out


def print_metrics(m, T_eff=None):
    print("  |M| (rows <anchor|, cols |psi(T)>):", flush=True)
    print("        " + "".join(f"{b:>12}" for b in PAIR_LABELS), flush=True)
    for r in range(4):
        print(f"  {PAIR_LABELS[r]:>6}" + "".join(
            f"{abs(m['M'][r, c]):>12.8f}" for c in range(4)), flush=True)
    print("  norms " + " ".join(f"{v:.10f}" for v in m["norms"])
          + "   retention " + " ".join(f"{v:.8f}" for v in m["retention"]),
          flush=True)
    if "F" in m:
        print(f"  Fbar(sqrt iSWAP) = {m['F']:.8f} (loss {1 - m['F']:.3e})  "
              f"F_xc = {m['Fxc']:.8f}   Fbar(identity) = {m['F_identity']:.8f}",
              flush=True)
    print(f"  theta = {m['theta']:.6f} (pi/2 {np.pi / 2:.6f}, "
          f"{m['theta'] / (np.pi / 2) - 1:+.3%})  Delta_eff = {m['Delta_eff']:.6e}"
          + (f"  enhancement {m['enhancement']:+.3%}" if "enhancement" in m
             else ""), flush=True)
    print("  last SPF pops " + " ".join(f"w{s}:{float(p[-1]):.2e}"
                                         for s, p in enumerate(m["spf_pops"])),
          flush=True)


# ── propagation ───────────────────────────────────────────────────────────

class Propagator:
    """Block-scanned midpoint CMF with an exact regauge between blocks.

    ``integ.renorm == "step"`` passes ``renorm=True`` to the library
    integrator: Löwdin renormalisation of every orbital and SPF matrix after
    each step (``integrators_ml._renormalise`` has the rationale).
    The exact regauge still runs between blocks.

    The time of step ``k`` is ``k * dt`` from the GLOBAL integer step index,
    never an accumulated float, so a segment-B restart at ``t_c`` evaluates
    the Hamiltonian at bit-identical times to the unsegmented leg.
    """

    def __init__(self, ml, integ: Integ, norm_guard=0.1):
        self.ml, self.integ = ml, integ
        self.norm_guard = norm_guard
        if integ.renorm not in ("none", "step"):
            raise ValueError(integ.renorm)
        self.step = get_ml_cmf_midpoint_integrator(
            ml, integ.dt, renorm=(integ.renorm == "step"),
            **integ.integrator_kwargs())
        self.regauge = make_regauge_ml(ml)
        dt, every = integ.dt, integ.every
        step, regauge = self.step, self.regauge

        @jax.jit
        def blk(k0, y):
            def body(i, y):
                t = (k0 + i).astype(jnp.float64) * dt
                return step(t, y)[1]
            y = jax.lax.fori_loop(0, every, body, y)
            return regauge(y)

        self._blk = blk

    def k_of(self, t):
        k = int(round(t / self.integ.dt))
        assert abs(k * self.integ.dt - t) < 1e-9, (
            f"t = {t!r} is not on the step grid (dt {self.integ.dt})")
        return k

    def run(self, y, t0, T, snaps=(), verbose=True, probe=None,
            report_every=None):
        """Propagate ``t0 -> T``. ``snaps`` = times to checkpoint at, each on
        the block grid. Returns ``(y, status, {t: y}, sec_per_step)``."""
        every = self.integ.every
        k0, k1 = self.k_of(t0), self.k_of(T)
        nblk, rem = divmod(k1 - k0, every)
        assert rem == 0, (
            f"T - t0 = {T - t0} is not an integer number of blocks "
            f"({self.integ.block_t} t.u.): snap the grid, rounding here "
            f"would silently change the pulse")
        snap_k = {}
        for ts in snaps:
            ks = self.k_of(ts)
            assert (ks - k0) % every == 0 and k0 < ks <= k1, (
                f"checkpoint t={ts} is not on the block grid in ({t0}, {T}]")
            snap_k[ks] = ts
        out = {}
        if t0 in snaps:
            out[t0] = np.asarray(y)
        if report_every is None:
            report_every = max(1, nblk // 20)
        status = "ok"
        tc = time.perf_counter()
        t_first = None
        k = k0
        for ib in range(nblk):
            y = self._blk(jnp.asarray(k, dtype=jnp.int64), y)
            k += every
            if ib == 0:
                y.block_until_ready()
                t_first = time.perf_counter()
            if k in snap_k:
                out[snap_k[k]] = np.asarray(y)
            if (ib + 1) % report_every == 0 or ib == nblk - 1:
                y_np = np.asarray(y)
                if not np.all(np.isfinite(y_np)):
                    status = f"DIVERGE@t={k * self.integ.dt:.3f}"
                    break
                # A finite state can still have collapsed (one run: the
                # norms went to 3e-7 at t = 21 with status "ok").
                nrm = np.abs(np.diag(np.asarray(
                    self.ml.calculate_overlap(y, y))))
                if np.max(np.abs(nrm - 1.0)) > self.norm_guard:
                    status = (f"NORM@t={k * self.integ.dt:.3f}"
                              f"(max|n-1|={np.max(np.abs(nrm - 1.0)):.2e})")
                    break
                if verbose:
                    el = time.perf_counter() - t_first
                    sps = el / max(1, (ib) * every)
                    msg = (f"    t = {k * self.integ.dt:10.3f}  "
                           f"blk {ib + 1}/{nblk}  {sps * 1e3:.2f} ms/step  "
                           f"ETA {(nblk - ib - 1) * every * sps / 60:.1f} min")
                    if probe is not None:
                        msg += "  " + probe(y)
                    print(msg, flush=True)
        y.block_until_ready()
        n_after = (nblk - 1) * every
        sps = ((time.perf_counter() - t_first) / n_after
               if n_after > 0 and t_first else np.nan)
        return y, status, out, sps, (t_first - tc if t_first else np.nan)


def norm_probe(ml):
    def probe(y):
        n = np.abs(np.diag(np.asarray(ml.calculate_overlap(y, y))))
        return "norms-1 " + " ".join(f"{v - 1:+.2e}" for v in n)
    return probe


def device_banner():
    d = jax.devices()[0]
    print(f"  device {d.platform}:{d.device_kind}  jax {jax.__version__}  "
          f"x64 {jax.config.jax_enable_x64}", flush=True)


def save_npz(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def load_grids():
    with open(os.path.join(HERE, "grids.json")) as fh:
        return json.load(fh)
