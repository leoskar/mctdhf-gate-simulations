"""Shared machinery for the 3 e-/well operating-point optimisation.

In short: the
ML ``ITE_opt`` (:mod:`mctdhf.MLMCTDHF_n_species_ITE_opt`) relaxes the five
weighted configurations (|0000> at 1, the singles at 1/2, by identity), then
all sixteen register configurations are matched and the 16x16 ``F_matched``
goes into the one-electron loss functions unchanged.

What lives here
---------------

* the one-electron losses, imported read-only from
  ``one_electron/operating_points/losses`` (:func:`load_loss`), with the
  sha256 of every source file they pulled in;
* copies of the parametrisation helpers of
  ``one_electron/operating_points/optimize_qubit_params.py`` and
  ``optimize_qubit_params_frozen.py`` — those modules cannot be imported,
  because they build the 1 e- solver at import;
* :func:`make_solver`, :func:`h_of`, :func:`cold_state`,
  :func:`evaluate_params` (with the wrong-fixed-point guard),
  :class:`Run` (the PyBOBYQA objective with its bookkeeping) and
  :func:`save_trajectory`.

x64 is switched on at import, before any array exists.
"""

import hashlib
import importlib
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mctdhf.MLMCTDHF_n_species_ITE_opt import (  # noqa: E402
    MLMCTDHF_n_species_ITE_opt, register_configs)

from system import fourwell as fw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
RESULTS_DIR = os.path.join(HERE, "results")
CKPT_DIR = os.path.abspath(os.path.join(HERE, "..", "ckpt"))

# The one-electron operating-point code. Read-only: its `losses` package is
# imported, nothing in it is written.
ONE_E_OP = os.path.abspath(os.path.join(HERE, "..", "..", "one_electron",
                                      "operating_points"))
ONE_E_TRAJ = os.path.join(ONE_E_OP, "ITE_trajectories")

NUM_WELLS = fw.NUM_WELLS
CONFIGS = register_configs(NUM_WELLS, 2)

# x0 = the appendix optimum of each stage.
X0_FILES = {
    "stage1_singles_ground": "new_geometry_step1_stage1_singles_ground.npz",
    "stage2_superpos_w0w1":
        "chain_frozen_20260621_140848_step2_stage2_superpos_w0w1.npz",
    "stage3_superpos_w1w2":
        "chain_frozen_20260621_140848_step3_stage3_superpos_w1w2.npz",
    "stage4_superpos_w2w3":
        "chain_frozen_20260621_140848_step4_stage4_superpos_w2w3.npz",
    "stage5_superpos_w0w3":
        "chain_frozen_20260621_140848_step5_stage5_superpos_w0w3.npz",
}

# ── optimiser settings ──────────────────────────────

RHOBEG = 0.02
RHOEND = 1e-3
MAXFUN = 400
SEEK_GLOBAL_MINIMUM = False
WARM_STEPS = 300

# ── as in one_electron/operating_points/optimize_qubit_params{,_frozen}.py

ALPHA_BOUNDS_GHZ = np.tile(np.array([5.0, 15.0]), (NUM_WELLS, 1))
BETA_BOUNDS_GHZ = np.tile(np.array([0.05, 2.0]), (NUM_WELLS, 1))
FREEZE_GAP_GHZ = 2.0
FREEZE_BETA = True


def _normalise_bounds(b, name):
    arr = np.asarray(b, dtype=float)
    if arr.shape == (2,):
        arr = np.tile(arr, (NUM_WELLS, 1))
    if arr.shape != (NUM_WELLS, 2):
        raise ValueError(
            f"{name} must have shape (2,) or ({NUM_WELLS}, 2); got {arr.shape}"
        )
    if not np.all(arr[:, 0] < arr[:, 1]):
        raise ValueError(f"{name} requires lo < hi per row; got {arr}")
    return arr


def _z_to_x(z, alpha_lo, alpha_hi, eps, num_wells=NUM_WELLS):
    """Stick-breaking: fractional [t_0..t_{K-1}, β...] → physical [α..., β...].

    Each tⱼ ∈ [0, 1] is the fraction of the *remaining* feasible α-headroom
    consumed at step j, so for any t in the unit box we get
    α_lo ≤ α₀ < α₁ < ... < α_{K-1} ≤ α_hi with each gap ≥ eps. Concretely:

        α₀ = α_lo + t₀ · (α_hi − α_lo − (K−1)·eps)
        αⱼ = α_{j−1} + eps + tⱼ · (α_hi − α_{j−1} − (K−j)·eps)   for j ≥ 1
    """
    z = np.asarray(z, dtype=float)
    K = num_wells
    alpha = np.empty(K)
    alpha[0] = alpha_lo + z[0] * (alpha_hi - alpha_lo - (K - 1) * eps)
    for j in range(1, K):
        headroom = alpha_hi - alpha[j - 1] - (K - j) * eps
        alpha[j] = alpha[j - 1] + eps + z[j] * headroom
    return np.concatenate([alpha, z[K:]])


def _x_to_z(x, alpha_lo, alpha_hi, eps, num_wells=NUM_WELLS):
    """Inverse stick-breaking: physical [α..., β...] → fractional [t..., β...].

    Inverts `_z_to_x`. For physically valid x (α ascending with gaps ≥ eps,
    α_{K−1} ≤ α_hi) the resulting tⱼ live in [0, 1]. Floating-point can put
    them slightly outside; the caller should clip to [0, 1] afterwards.
    Degenerate denominators (zero headroom) map to t = 0.
    """
    x = np.asarray(x, dtype=float)
    K = num_wells
    alpha = x[:K]
    t = np.empty(K)
    span0 = alpha_hi - alpha_lo - (K - 1) * eps
    t[0] = (alpha[0] - alpha_lo) / span0 if span0 > 0 else 0.0
    for j in range(1, K):
        headroom = alpha_hi - alpha[j - 1] - (K - j) * eps
        gap = alpha[j] - alpha[j - 1] - eps
        t[j] = gap / headroom if headroom > 1e-12 else 0.0
    return np.concatenate([t, x[K:]])


def _derive_z_bounds(alpha_b, beta_b, eps, num_wells=NUM_WELLS):
    """Build z-space bounds for the stick-breaking parametrisation.

    z = [t_0..t_{K-1}, β_0..β_{K-1}] with each tⱼ ∈ [0, 1]. The α part is
    reconstructed via `_z_to_x`, which guarantees α_lo ≤ α₀ < α₁ < ... <
    α_{K-1} ≤ α_hi by construction. Per-well α bounds are collapsed to a
    single global range [min α_lo, max α_hi] — the ordering constraint
    cannot be expressed against per-well bounds. Returns
    (lb_z, ub_z, alpha_lo, alpha_hi, warn_msg) where warn_msg is non-None
    when the input α rows are non-uniform.
    """
    alpha_b = np.asarray(alpha_b, dtype=float)
    beta_b = np.asarray(beta_b, dtype=float)
    alpha_lo = float(alpha_b[:, 0].min())
    alpha_hi = float(alpha_b[:, 1].max())

    span = alpha_hi - alpha_lo
    if span <= (num_wells - 1) * eps:
        raise ValueError(
            f"alpha range [{alpha_lo}, {alpha_hi}] too narrow for "
            f"{num_wells - 1} gaps of >= {eps} GHz each"
        )

    lb_z = np.empty(2 * num_wells)
    ub_z = np.empty(2 * num_wells)
    lb_z[:num_wells] = 0.0
    ub_z[:num_wells] = 1.0
    lb_z[num_wells:] = beta_b[:, 0]
    ub_z[num_wells:] = beta_b[:, 1]

    rows_uniform = (
        np.allclose(alpha_b[:, 0], alpha_b[0, 0])
        and np.allclose(alpha_b[:, 1], alpha_b[0, 1])
    )
    warn_msg = (
        None if rows_uniform
        else (
            f"alpha_ordering=ascending: per-well α bounds collapsed to "
            f"global range [{alpha_lo}, {alpha_hi}] GHz "
            f"(input rows were not uniform)"
        )
    )
    return lb_z, ub_z, alpha_lo, alpha_hi, warn_msg


def _active_wells_for_loss(loss_name):
    """Active (superposition) wells for a stage = (A-1, B-1).

    The stage modules expose module-level config indices `A, B` (in
    iter_bitstrings order, where config c in 1..4 is the single excitation of
    well c-1). The two superposition partners are therefore wells A-1 and B-1.
    """
    _one_e_losses()
    module = importlib.import_module(f"losses.{loss_name}")
    if not (hasattr(module, "A") and hasattr(module, "B")):
        raise ValueError(
            f"loss '{loss_name}' does not expose module-level A, B config "
            f"indices; cannot derive the active wells for freezing. The frozen "
            f"variant supports the superposition stages (2/3/4)."
        )
    a, b = int(module.A), int(module.B)
    active = sorted({a - 1, b - 1})
    if not all(0 <= w < NUM_WELLS for w in active):
        raise ValueError(
            f"loss '{loss_name}' gives active wells {active} outside "
            f"[0, {NUM_WELLS}); A={a}, B={b}."
        )
    if len(active) != 2:
        raise ValueError(
            f"expected exactly 2 distinct active wells; got {active} "
            f"(A={a}, B={b})."
        )
    return active


def _build_active_alpha_window(active, frozen_alpha, alpha_b, gap, apply_gap=True):
    """Per-active-well alpha box [lo, hi].

    When `apply_gap` (the "ascending" stages 2/3/4): for each active well w the
    lower bound is `gap` above the nearest frozen well *below* w (or the global
    lower bound if none), and the upper bound is `gap` below the nearest frozen
    well *above* w (or the global upper bound). Raises if any window is empty.

    When not `apply_gap` (the "none" stage 5): no gap/ascending constraint -- the
    active wells get their full per-well bounds and may cross the frozen middles
    to reach the long-range resonance.
    """
    active = list(active)
    if not apply_gap:
        return alpha_b[active, 0].copy(), alpha_b[active, 1].copy()
    frozen = sorted(set(range(NUM_WELLS)) - set(active))
    los, his = [], []
    for w in active:
        below = [f for f in frozen if f < w]
        above = [f for f in frozen if f > w]
        lo = alpha_b[w, 0] if not below else frozen_alpha[max(below)] + gap
        hi = alpha_b[w, 1] if not above else frozen_alpha[min(above)] - gap
        lo = max(lo, alpha_b[w, 0])
        hi = min(hi, alpha_b[w, 1])
        if lo > hi:
            raise ValueError(
                f"active well {w}: empty alpha window [{lo:.3f}, {hi:.3f}] GHz "
                f"under gap={gap} GHz and frozen alphas {np.round(frozen_alpha, 3)}. "
                f"This geometry is not supported by the frozen variant "
                f"(e.g. stage 5's non-adjacent pair); use optimize_qubit_params.py."
            )
        los.append(lo)
        his.append(hi)
    return np.array(los), np.array(his)


# ── the one-electron losses, read-only ────────────────────────────────────


def _one_e_losses():
    """The ``losses`` package of one_electron/operating_points.

    APPENDED to ``sys.path`` so nothing there can shadow this package or
    ``mctdhf``; the stage modules import only numpy/scipy and each other.
    """
    # Read-only means no __pycache__ either: an import from this Python would
    # otherwise drop .pyc files into one_electron/operating_points. Process
    # wide, and set before the first `losses` import.
    sys.dont_write_bytecode = True
    if ONE_E_OP not in sys.path:
        sys.path.append(ONE_E_OP)
    mod = importlib.import_module("losses")
    if not os.path.abspath(mod.__file__).startswith(ONE_E_OP):
        raise ImportError(f"'losses' resolved to {mod.__file__}, not the "
                          f"one-electron code in {ONE_E_OP}")
    return mod


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_loss(name):
    """``losses.load(name)`` from one_electron/operating_points, plus provenance.

    ``spec["sha256"]`` maps every ``losses.*`` source file imported so far to
    its hash, so each output records exactly which loss code produced it.
    """
    spec = _one_e_losses().load(name)
    files = {}
    for mname, m in sorted(sys.modules.items()):
        if (mname == "losses" or mname.startswith("losses.")) and \
                getattr(m, "__file__", None):
            files[os.path.relpath(m.__file__, ONE_E_OP)] = _sha256(m.__file__)
    spec["sha256"] = files
    spec["name"] = name
    return spec


def load_x0(loss_name):
    """``best_x`` (GHz) of the appendix optimum for a stage, and its path."""
    path = os.path.join(ONE_E_TRAJ, X0_FILES[loss_name])
    with np.load(path, allow_pickle=False) as d:
        x = np.asarray(d["best_x"], dtype=float)
    return x, path


def load_best_x(path):
    with np.load(path, allow_pickle=False) as d:
        return np.asarray(d["best_x"], dtype=float)


# ── solver and states ─────────────────────────────────────────────────────


def make_solver(norb, n_spf, **kwargs):
    """The four-well ML ``ITE_opt``: geometry and interaction from
    ``system/fourwell.py``, N = 3 per well."""
    return MLMCTDHF_n_species_ITE_opt(
        g_dict=fw.g_dict(),
        num_particles=[fw.NUM_PARTICLES] * NUM_WELLS,
        num_mctdhf_orbitals=[norb] * NUM_WELLS,
        num_spatial_orbitals=[fw.NUM_DVR] * NUM_WELLS,
        num_spfs=[n_spf] * NUM_WELLS,
        **kwargs)


def h_of(x_ghz):
    """Frozen one-body ``h_list`` at ``x = [alpha_0..3, beta_0..3]`` in GHz."""
    return fw.h_static(np.asarray(x_ghz, dtype=np.float64)
                       / fw.TYPICAL_FREQ_GHZ)


def _x_key(x_ghz):
    return hashlib.sha1(np.asarray(x_ghz, dtype=np.float64).tobytes()
                        ).hexdigest()[:12]


def isolated_orbitals(x_ghz, norb, iso_dt=2e-3, iso_steps=8000,
                      verbose=True):
    """Each well relaxed on its own (standard MCTDHF), at ``x``. Cached.

    ⚠ ``iso_dt`` must stay <= 2e-3: this is a standard CMF relaxation and
    lands on a collapsed fixed point above that (the dt
    trap). The ML relaxation that follows is not limited this way.
    """
    from system.check_replica import relax_isolated

    if iso_dt > 2e-3 + 1e-15:
        raise ValueError(f"iso_dt = {iso_dt} > 2e-3 collapses the isolated "
                         "wells")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR,
                        f"iso_n{norb}_dt{iso_dt:g}_{_x_key(x_ghz)}.npz")
    if os.path.exists(path):
        d = np.load(path)
        if not np.array_equal(d["x_ghz"], np.asarray(x_ghz, dtype=float)):
            raise RuntimeError(f"cache key collision in {path}")
        return [jnp.asarray(d[f"b{w}"]) for w in range(NUM_WELLS)], path
    x_dim = np.asarray(x_ghz, dtype=float) / fw.TYPICAL_FREQ_GHZ
    bs, es = [], []
    for w in range(NUM_WELLS):
        b, e = relax_isolated(w, norb, iso_steps, iso_dt, x_dim)
        bs.append(jnp.asarray(b, dtype=jnp.complex128))
        es.append(float(e))
        if verbose:
            print(f"    well {w} isolated (dt {iso_dt:g}): E = {e:.8f}",
                  flush=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, x_ghz=np.asarray(x_ghz, dtype=float), energy=np.array(es),
             iso_dt=iso_dt,
             **{f"b{w}": np.asarray(bs[w]) for w in range(NUM_WELLS)})
    os.replace(tmp, path)
    return bs, path


def cold_state(solver, x_ghz, norb, tol=1e-9, max_steps=20000,
               verbose=True):
    """Isolated wells -> ``default_state`` -> ``relax_to_tol``.

    Returns ``(state, e_sum, info)`` with ``info["iso_cache"]``.
    """
    bs, iso_path = isolated_orbitals(x_ghz, norb, verbose=verbose)
    h = h_of(x_ghz)
    state = solver.default_state(bs, h)
    state, e, info = solver.relax_to_tol(h, state, tol=tol,
                                         max_steps=max_steps)
    info["iso_cache"] = iso_path
    return state, e, info


_SOLVER5 = {}


def cold_state_opt(solver, x_ghz, norb, tol=None, verbose=True):
    """The cold candidate inside the optimisation.

    For the default 5-state relaxed set, :func:`cold_state`. For a larger set
    (``relax_set("16")``) a cold 16-state relaxation from isolated wells can
    land far too high: at stage 2's x0 sum E16 was 8 above the
    seeded state, which itself had well 0's |1> replaced by a higher state
    (detuning 6.83 against 3.41; validate seedcheck). So the
    cold candidate is the 5-state cold relaxation, then the larger set
    relaxed from its regauged SPFs and orbitals (A one-hot).
    """
    tol = COLD_TOL_OPT if tol is None else tol
    if solver.num_states <= 5:
        return cold_state(solver, x_ghz, norb, tol=tol, verbose=verbose)
    key = (norb, solver.num_spfs, solver.dt, solver.precision,
           solver.inner_tol, solver.inner_maxiter, solver.chunk, solver.eps)
    if key not in _SOLVER5:
        _SOLVER5[key] = make_solver(
            norb, solver.num_spfs[0], dt=solver.dt, precision=solver.precision,
            inner_tol=solver.inner_tol, inner_maxiter=solver.inner_maxiter,
            chunk=solver.chunk, eps=solver.eps)
    sv5 = _SOLVER5[key]
    h = h_of(x_ghz)
    st5, _e5, info5 = cold_state(sv5, x_ghz, norb, tol=tol, verbose=verbose)
    _A, Bs, bs = sv5.analyse(st5, h)["state_regauged"]
    A0 = solver.default_state(bs, h)[0]
    state, e, info = solver.relax_to_tol(h, (A0, Bs, bs), tol=tol)
    info = dict(info, steps=info["steps"] + info5["steps"],
                steps_5state=info5["steps"], iso_cache=info5["iso_cache"])
    return state, e, info


def save_state(path, state, **meta):
    A, Bs, bs = state
    np.savez(path, A=np.asarray(A),
             **{f"B{s}": np.asarray(Bs[s]) for s in range(len(Bs))},
             **{f"b{s}": np.asarray(bs[s]) for s in range(len(bs))}, **meta)


def load_state(path):
    d = np.load(path)
    K = sum(1 for k in d.files if k.startswith("B") and k[1:].isdigit())
    return (jnp.asarray(d["A"]),
            [jnp.asarray(d[f"B{s}"]) for s in range(K)],
            [jnp.asarray(d[f"b{s}"]) for s in range(K)])


# ── one evaluation ────────────────────────────────────────────────────────

DETUNING_JUMP = 0.25
# Inside the optimisation the states only need to rank (alpha, beta), so a
# cold relaxation stops at 1e-6 (detunings within ~5e-6).
# Certification of the final eigenstates keeps 1e-9 (cold_state's default).
# Chosen after the fixedpoint checks (validate.py).
COLD_TOL_OPT = 1e-6
# Warm start for the FIRST evaluation of a target stage, from the ML idle
# optimum's state (see evaluate_params' `seed`).
SEED_STEPS = 1000
# Superposition stages: each of the two superposition roots must keep this
# much weight on its pair {|A>, |B>} (``block_weight``). Below it the root has
# hybridised with a state outside the pair (an intruder), which the super
# term of the one-electron loss REWARDS, because it pulls F[A,A] toward 1/2 whatever A mixes with
# (stage 4 reached 4.25 at a block weight of ~0.63 with no
# A-B mixing at all). Stages 2/3's good points are at ~1.
BLOCK_WEIGHT_MIN = 0.95


# The relaxed (state-averaged) set. "5": |0000> + singles at 1, 1/2 (the
# solver's default). "16": every register configuration at 2^-n by
# excitation number n, non-degenerate so the natural-SPF gauge that defines
# |0>/|1> stays fixed (each well's |0> carries twice its |1>). The loss
# scores the doubles, triples and |1111> too, and the 5-state relaxation
# misdescribes them near a resonance (stage 3: 0.896 against 0.67, loss 1.18
# against 1.40). The target stages
# are rerun with "16".
def relax_set(name):
    if name == "5":
        return {}
    if name == "16":
        return dict(relax_configs=list(CONFIGS),
                    relax_weights=[0.5 ** sum(c) for c in CONFIGS])
    raise ValueError(f"relax set {name!r}: '5' or '16'")


def superpos_block(loss_name):
    """``(A, B)``, the config indices of a superposition stage's pair, from
    the loss module (``None`` for a loss without them, e.g. stage 1)."""
    _one_e_losses()
    module = importlib.import_module(f"losses.{loss_name}")
    if not (hasattr(module, "A") and hasattr(module, "B")):
        return None
    return int(module.A), int(module.B)


def _score(solver, x_ghz, spec, state, e, info, steps, cold, t0):
    h = h_of(x_ghz)
    an = solver.analyse(state, h, info)
    A, Bs, bs = state
    r = spec["compute"](C_final=A, bs_final=(Bs, bs), F=an["F_matched"],
                        energies=an["E_matched"], num_wells=NUM_WELLS)
    try:
        r_en = spec["compute"](C_final=A, bs_final=(Bs, bs),
                               F=an["F_energy"], energies=an["E_all"],
                               num_wells=NUM_WELLS)
        loss_energy = float(r_en["loss"])
    except Exception:                                  # noqa: BLE001
        loss_energy = float("nan")   # logged only, never optimised
    finite = bool(np.all(np.isfinite(e)) and np.all(np.isfinite(
        an["E_matched"])) and np.isfinite(r["loss"]))
    block_weight = None
    if spec.get("block") is not None:
        a, b = spec["block"]
        F = an["F_matched"]
        block_weight = float(min(F[a, a] + F[a, b], F[b, a] + F[b, b]))
    return dict(
        block_weight=block_weight,
        x=np.asarray(x_ghz, dtype=float).copy(), state=state,
        loss=float(r["loss"]), loss_energy=loss_energy,
        fidelities=np.asarray(r["fidelities"], dtype=float),
        matched_idx=np.asarray(r["matched_idx"], dtype=int),
        min_fid=float(np.min(r["fidelities"])),
        E_matched=an["E_matched"], F_matched=an["F_matched"],
        e_relax=an["e_relax"], detunings=an["detunings"],
        additivity=an["additivity"], healthy=an["healthy"],
        leakage=an["leakage"], occupations=an["occupations"],
        ranks_all=an["ranks_all"], ranks_relax=an["ranks_relax"],
        dE_last_chunk=info.get("dE_last_chunk", float("nan")),
        last_chunk_fallback=bool(info.get("last_chunk_fallback", False)),
        n_fallback_chunks=len(info.get("fallback_chunks", [])),
        steps=int(steps), cold=cold, finite=finite,
        wall=time.perf_counter() - t0, extra={k: v for k, v in r.items()
                                              if k not in ("loss",
                                                           "fidelities",
                                                           "matched_idx")},
    )


def guard_reasons(res, prev_detunings):
    """Why a WARM evaluation is not trusted (empty list = trusted).

    The guard targets WRONG fixed points (collapse, mislabelled or jumping
    levels), not slow convergence. A "last-chunk |dE| > 1e-5" test
    it was dropped: after a 1 % alpha step a
    300-iteration warm evaluation ends at 7.7e-6 .. 9.6e-6, so larger steps
    would trip it routinely, and every trip costs a 240 s cold relaxation
    that can itself land on a HIGHER fixed point.
    ``dE_last_chunk`` is still recorded per evaluation.
    """
    why = []
    if not res["finite"]:
        why.append("non-finite")
    if not res["healthy"]:
        why.append("unhealthy")
    if prev_detunings is not None and res["finite"]:
        rel = np.abs(res["detunings"] - prev_detunings) / np.abs(prev_detunings)
        if np.any(rel > DETUNING_JUMP):
            why.append(f"detuning jump {rel.max():.2f}")
    if res["last_chunk_fallback"]:
        why.append("root-set fallback in last chunk")
    return why


def _reject_intruder(res):
    """A superposition root hybridised with a state outside {|A>, |B>}.

    Not a wrong fixed point: the relaxed state is right, the operating point
    is not a qubit point. So no cold relaxation is tried (it would find the
    same physics); the evaluation is REJECTED with the sentinel and the
    caller keeps the previous warm state (``keep_prev``).
    """
    bw = res.get("block_weight")
    if bw is None or bw >= BLOCK_WEIGHT_MIN:
        return None
    return f"intruder: block weight {bw:.3f} < {BLOCK_WEIGHT_MIN}"


def evaluate_params(solver, x_ghz, spec, prev, norb, warm_steps=WARM_STEPS,
                    verbose=True, seed=None, seed_steps=SEED_STEPS):
    """One loss evaluation at ``x_ghz`` (GHz).

    ``prev`` is ``None`` (cold start) or the previous accepted result, whose
    ``state`` warm-starts a fixed ``warm_steps`` relaxation and whose
    ``detunings`` feed the guard.

    ⚠ The guard exists because every wrong fixed point seen CONVERGED
    and passed at least two of the three detectors. A warm result that fails :func:`guard_reasons`
    triggers a cold relaxation; of the two, the healthy one with the lower
    sum E over the relaxed configurations is kept. The relaxation is
    variational in (B, b), and every wrong fixed point seen had the
    higher sum E.

    ``seed`` (a state, used only when ``prev`` is None) adds a third
    candidate to the cold one: ``seed_steps`` fixed iterations warm-started
    from it. The target stages seed their first evaluation with the ML idle
    optimum's state, because a cold relaxation can land on a higher fixed
    point (+7.5e-3 in sum E5 at alpha_0 + 1 %). The lower
    healthy sum E5 wins, as for the guard.

    Returns the result dict (``ok`` False when no candidate was usable).
    """
    t0 = time.perf_counter()
    h = h_of(x_ghz)
    cands, reasons = [], []
    if prev is not None:
        state, e, info = solver.integrate(warm_steps, h, prev["state"])
        warm = _score(solver, x_ghz, spec, state, e, info, warm_steps,
                      False, t0)
        reasons = guard_reasons(warm, prev.get("detunings"))
        if not reasons:
            intr = _reject_intruder(warm)
            if intr:
                warm.update(ok=False, guard=[intr], keep_prev=True,
                            wall=time.perf_counter() - t0)
                return warm
            warm.update(ok=True, guard=[])
            return warm
        cands.append(warm)
        if verbose:
            print(f"      guard: {', '.join(reasons)} -> cold relaxation",
                  flush=True)
    if prev is None and seed is not None:
        state, e, info = solver.integrate(seed_steps, h, seed)
        seeded = _score(solver, x_ghz, spec, state, e, info, seed_steps,
                        False, t0)
        seeded["seeded"] = True
        cands.append(seeded)
    try:
        state, e, info = cold_state_opt(solver, x_ghz, norb,
                                        verbose=verbose)
        cold = _score(solver, x_ghz, spec, state, e, info, info["steps"],
                      True, t0)
        cands.append(cold)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        if verbose:
            print(f"      cold relaxation failed: {exc}", flush=True)
    usable = [c for c in cands if c["finite"] and c["healthy"]]
    if not usable:
        out = dict(cands[-1]) if cands else dict(x=np.asarray(x_ghz))
        out.update(ok=False, guard=reasons, wall=time.perf_counter() - t0)
        return out
    best = min(usable, key=lambda c: c["e_relax"])
    if verbose and len(cands) > 1:
        print("      candidates: " + "  ".join(
            f"{'cold' if c['cold'] else ('seeded' if c.get('seeded') else 'warm')}"
            f" ΣE5={c['e_relax']:.8f}{'' if c['healthy'] else ' (unhealthy)'}"
            for c in cands) + f"  -> keep "
            f"{'cold' if best['cold'] else ('seeded' if best.get('seeded') else 'warm')}",
            flush=True)
    intr = _reject_intruder(best)
    if intr:
        best.update(ok=False, guard=reasons + [intr], keep_prev=True,
                    wall=time.perf_counter() - t0,
                    e_relax_candidates=[c["e_relax"] for c in cands])
        return best
    best.update(ok=True, guard=reasons, wall=time.perf_counter() - t0,
                warm_rejected=bool(prev is not None and best["cold"]),
                e_relax_candidates=[c["e_relax"] for c in cands])
    return best


# ── the PyBOBYQA objective ────────────────────────────────────────────────


class Run:
    """PyBOBYQA objective over the solver variables ``z``.

    ``to_x(z)`` maps to the full 8-vector in GHz. Keeps the warm state, the
    best point and a per-evaluation record. A failed evaluation returns the
    distance-scaled sentinel of the one-electron optimiser,
    ``1e6 (1 + |z - z_best| / (ub - lb))``, and
    drops the warm state.
    """

    def __init__(self, solver, spec, norb, to_x, lb_z, ub_z,
                 warm_steps=WARM_STEPS, seed=None, log_stem=None):
        self.solver, self.spec, self.norb = solver, spec, norb
        self.to_x, self.lb_z, self.ub_z = to_x, np.asarray(lb_z), np.asarray(ub_z)
        self.warm_steps = warm_steps
        self.seed = seed      # extra candidate whenever there is no warm state
        # Incremental output, so an unfinished or crashed run can still be
        # judged: one JSON line per evaluation in <stem>.evals.jsonl, and the
        # best state rewritten to <stem>_best_state.npz whenever it improves.
        self.log_stem = log_stem
        if log_stem is not None:
            os.makedirs(os.path.dirname(log_stem), exist_ok=True)
            open(log_stem + ".evals.jsonl", "w").close()
        self.prev = None
        self.best = None
        self.best_z = None
        self.records = []
        # "search" (optimize_target's resonance search) or "bobyqa"; logged.
        self.phase = "bobyqa"

    def _log(self, rec, improved):
        if self.log_stem is None:
            return
        keys = ("loss", "loss_energy", "min_fid", "e_relax", "healthy",
                "cold", "steps", "wall", "oob", "guard", "dE_last_chunk",
                "n_fallback_chunks", "e_relax_candidates", "block_weight")
        row = {k: rec.get(k) for k in keys if k in rec}
        row.update(eval=len(self.records), x=list(map(float, rec["x"])),
                   improved=improved, phase=self.phase)
        for k in ("detunings", "fidelities", "leakage", "ranks_all"):
            if k in rec:
                row[k] = [float(v) for v in np.asarray(rec[k]).ravel()]
        with open(self.log_stem + ".evals.jsonl", "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        if improved:
            save_state(self.log_stem + "_best_state.npz", rec["state"],
                       x_ghz=rec["x"], loss=rec["loss"],
                       e_relax=rec["e_relax"])

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        x = self.to_x(z)
        i = len(self.records) + 1
        res = evaluate_params(self.solver, x, self.spec, self.prev,
                              self.norb, self.warm_steps, seed=self.seed)
        if not res["ok"]:
            dist = (np.linalg.norm((z - self.best_z) / (self.ub_z - self.lb_z))
                    if self.best_z is not None else 0.0)
            loss = 1e6 * (1.0 + dist)
            # An intruder rejection leaves the previous warm state valid.
            if not res.get("keep_prev"):
                self.prev = None
            print(f"[{i:03d}] α={np.round(x[:4], 4)} β={np.round(x[4:], 4)}  "
                  f"FAILED ({', '.join(res.get('guard', [])) or 'cold failed'})"
                  f"  sentinel={loss:.2e}  {res['wall']:.1f}s", flush=True)
            self.records.append(dict(res, loss=loss, oob=True, z=z,
                                     phase=self.phase))
            self._log(self.records[-1], False)
            return loss
        self.prev = res
        improved = self.best is None or res["loss"] < self.best["loss"]
        if improved:
            self.best, self.best_z = res, z.copy()
        worst = CONFIGS[int(np.argmin(res["fidelities"]))]
        tag = "cold" if res["cold"] else "warm"
        print(f"[{i:03d}] α={np.round(x[:4], 4)} β={np.round(x[4:], 4)}  "
              f"loss={res['loss']:.5f} (E-ord {res['loss_energy']:.5f})  "
              f"min_fid={res['min_fid']:.5f} {worst}  "
              f"ΣE5={res['e_relax']:.8f}  d="
              + " ".join(f"{v:.5f}" for v in res["detunings"])
              + (f"  bw={res['block_weight']:.4f}"
                 if res.get("block_weight") is not None else "")
              + f"  {tag}{' [guard: ' + ', '.join(res['guard']) + ']' if res['guard'] else ''}"
              f"  {res['wall']:.1f}s", flush=True)
        self.records.append(dict(res, oob=False, z=z, phase=self.phase))
        self._log(self.records[-1], improved)
        return res["loss"]


def run_bobyqa(run, z0, lb_z, ub_z, maxfun=MAXFUN, rhobeg=RHOBEG,
               rhoend=RHOEND):
    import pybobyqa   # only the optimisation needs it
    t0 = time.perf_counter()
    soln = pybobyqa.solve(run, np.asarray(z0, dtype=float),
                          bounds=(np.asarray(lb_z), np.asarray(ub_z)),
                          rhobeg=rhobeg, rhoend=rhoend, maxfun=maxfun,
                          scaling_within_bounds=True, print_progress=False,
                          seek_global_minimum=SEEK_GLOBAL_MINIMUM)
    return soln, time.perf_counter() - t0


# ── output ────────────────────────────────────────────────────────────────


def save_trajectory(log_path, run, soln, wall_total, spec, *,
                    alpha_bounds_ghz, beta_bounds_ghz, x0_ghz, x0_source,
                    alpha_ordering, solver_x_physical, settings):
    """The ``_save_trajectory`` fields of the one-electron optimiser, plus the ML per-evaluation
    diagnostics; the best state goes to ``<stem>_best_state.npz``."""
    recs = run.records
    good = [r for r in recs if not r["oob"]]
    nan16 = np.full(len(CONFIGS), np.nan)

    def col(key, fill, rows=recs):
        return np.array([r.get(key, fill) if r.get(key) is not None else fill
                         for r in rows])

    np.savez(
        log_path,
        loss_name=spec["name"],
        xs=np.stack([r["x"] for r in recs]),
        losses=col("loss", np.nan),
        min_fids=np.array([0.0 if r["oob"] else r["min_fid"] for r in recs]),
        walls=col("wall", np.nan),
        oob=np.array([r["oob"] for r in recs]),
        best_x=run.best["x"],
        best_loss=run.best["loss"],
        solver_x=soln.x, solver_fun=soln.f, solver_flag=soln.flag,
        solver_msg=soln.msg, wall_total=wall_total,
        configs=np.array(CONFIGS),
        fid_xs=np.stack([r["x"] for r in good]),
        fidelities=np.stack([r["fidelities"] for r in good]),
        matched_idx=np.stack([r["matched_idx"] for r in good]),
        energies=np.stack([r["E_matched"] for r in good]),
        alpha_bounds_ghz=np.asarray(alpha_bounds_ghz),
        beta_bounds_ghz=np.asarray(beta_bounds_ghz),
        x0_ghz=np.asarray(x0_ghz), x0_source=x0_source,
        alpha_ordering=alpha_ordering,
        solver_x_physical=np.asarray(solver_x_physical),
        seek_global_minimum=SEEK_GLOBAL_MINIMUM,
        # ML additions, one row per evaluation (NaN on failed ones)
        e_relax=np.array([r.get("e_relax", np.nan) for r in recs]),
        detunings=np.stack([np.asarray(r.get("detunings", nan16[:4]))
                            for r in recs]),
        healthy=np.array([bool(r.get("healthy", False)) for r in recs]),
        leakage=np.stack([np.asarray(r.get("leakage", nan16[:4]))
                          for r in recs]),
        ranks_all=np.stack([np.asarray(r.get("ranks_all",
                                             -np.ones(16, int)))
                            for r in recs]),
        loss_energy=np.array([r.get("loss_energy", np.nan) for r in recs]),
        steps=np.array([r.get("steps", 0) for r in recs]),
        cold=np.array([bool(r.get("cold", False)) for r in recs]),
        fallback=np.array([bool(r.get("last_chunk_fallback", False))
                           or r.get("n_fallback_chunks", 0) > 0
                           for r in recs]),
        dE_last_chunk=np.array([r.get("dE_last_chunk", np.nan)
                                for r in recs]),
        guard=np.array([";".join(r.get("guard", [])) for r in recs]),
        phase=np.array([r.get("phase", "bobyqa") for r in recs]),
        block_weight=np.array([np.nan if r.get("block_weight") is None
                               else r["block_weight"] for r in recs]),
        loss_sha256=json.dumps(spec["sha256"]),
        settings=json.dumps(settings),
    )
    stem = log_path[:-4] if log_path.endswith(".npz") else log_path
    save_state(stem + "_best_state.npz", run.best["state"],
               x_ghz=run.best["x"], loss=run.best["loss"],
               e_relax=run.best["e_relax"])
    print(f"Trajectory saved to {log_path}", flush=True)
