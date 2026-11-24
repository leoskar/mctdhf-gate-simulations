"""(tau_ramp, tau_hold) map of a gate leg by checkpoint reuse, with a frozen-H hold fit.

Every grid point is scored against three references from the same dynamics
(sqiswap, swap and identity; identity flags an adiabatic near-identity
evolution) with the average gate fidelity F̄_4 up to local Z.

Checkpoint reuse. The hold runs the frozen target-point Hamiltonian, so the
dynamics is time-translation invariant on the hold:
  segment A = ramp-up + hold, propagated once to (tau_ramp + hold_max), saving
              a full packed-y checkpoint every SNAP_DT time units;
  segment B = ramp-down only (length tau_ramp), propagated from each checkpoint
              -> one final state -> one column of M(tau_ramp, tau_hold_k).
The glued envelope equals the erf-edged _envelope_param up to the erf tails
(~1.1e-5 of the control); a built-in check compares a full leg with the
glued A+B at one point.  (sweep_sa.py, used for the paper, restarts
segment B earlier and is exact.)

Frozen-H hold fit. The hold dynamics is linear, so each gate entry is
M_ij(tau_hold) ~ sum_k A_k exp(-i omega_k tau_hold), with omega_k the gaps of
the target-point Hamiltonian between the ~6 relevant states (4 computational
+ |20>,|02>).  With the omega_k given, a coarse hold grid fixes the A_k and
gives a continuous F̄_4(tau_hold) per tau_ramp.  The fit only interpolates
between propagated points; each predicted optimum is checked by a direct
propagation.

Solver settings as in qoct_gate_leg.py (dt=5e-3, solver caps 1e-5/20,

Usage:
    python lib/sweep_map.py verify          (from one_electron/)
    python lib/sweep_map.py test    [--tau-ramp 22] [--tau-hold-max 5]
    python lib/sweep_map.py sweep   [--tau-ramps 10,22,34,46,58,70]
                                                       [--tau-hold-max 40] [--hold-stride 5]
                                                       [--consistency-point 22,12.5] [--no-consistency]
    python lib/sweep_map.py fit     <results.npz> [--omegas a,b,..|--omegas-from trace]
                                                       [--verify-optimum]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime

import jax

# VMF is latency-bound -> CPU; pin in-script BEFORE backend init.
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))          # one_electron/lib/
sys.path.insert(0, HERE)

from qoct_gate_leg import (    # loss machinery, x64 integrals, prod settings
    DT_PROD,
    G_DICT_64,
    M_REFS,
    NUM_ORB,
    PAIR_CONFIGS,
    PAIR_LABELS,
    T_S_64,
    TYPICAL_FREQ_GHZ,
    VMF_MAX_ITER,
    VMF_MIN_TOL,
    WELL_CENTERS_64,
    X_S_64,
    avg_gate_fidelity_up_to_localZ,
    identify_anchor_states,
    mctdh_overlap,
    verify_loss,
)
from rte_step import (
    MAX_SOLVER_TOL,
    NUM_DVR,
    NUM_NEWTON,
    NUM_WELLS,
    C_anchor,
    bs_anchor,
    x_star_1_ghz,
    x_star_2_ghz,
)
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import get_vmf_integrator
from gate_fidelity import gate_fidelity_up_to_localZ

assert NUM_WELLS == 4


# ── Sweep configuration ───────────────────────────────────────────────────────

# Three scoring references from the same dynamics. Identity flags an
# adiabatic near-identity evolution.
SWEEP_REFS = {
    "sqiswap": np.asarray(M_REFS["sqiswap"]),
    "swap": np.asarray(M_REFS["swap"]),
    "identity": np.eye(4, dtype=complex),
}
REF_NAMES = ["sqiswap", "swap", "identity"]

# Timestep. Default = qoct_gate_leg.DT_PROD (5e-3), overridable via the
# SWEEP_DT env var (set by --dt). Env-based ON PURPOSE: spawn workers re-import
# this module, so they read the same SWEEP_DT and stay consistent with the
# parent (a parent-only global rebind would NOT reach the children). Short
# ramps sit past the dt=5e-3 Newton stability edge (tau_ramp=10 drifted
# 2.9e-4) and need a smaller dt.
DT = float(os.environ.get("SWEEP_DT", str(DT_PROD)))

# Checkpoint / trajectory snapshot cadence: SNAP_DT must be an integer number
# of steps (exact step boundaries -> no interpolation when selecting
# checkpoints). SNAP_STEPS steps per snapshot; doubles as trajectory resolution.
SNAP_STEPS = 20
SNAP_DT = SNAP_STEPS * DT                        # 0.1 t.u. at dt=5e-3
TRAJ_POINTS_DEFAULT = 1500                       # ~points per full trajectory saved

# Default sweep box, snapped to SNAP_DT.
TAU_RAMPS_DEFAULT = [10.0, 22.0, 34.0, 46.0, 58.0, 70.0]
TAU_HOLD_MAX_DEFAULT = 40.0
HOLD_STRIDE_DEFAULT = 5.0
CONSISTENCY_POINT_DEFAULT = (22.0, 12.5)

# Doubles manifold of the active (0,1) pair: |20> = well0 in orbital 2,
# |02> = well1 in orbital 2 (determinant-index tuples, like PAIR_CONFIGS).
DOUBLES_CONFIGS = [(2, 0, 0, 0), (0, 2, 0, 0)]
DOUBLES_LABELS = ["|20>", "|02>"]

# Worker per-eval safety nets.
COMPILE_TIMEOUT = 3600.0
EVAL_TIMEOUT = 14400.0


# ── Segment schedules (frozen-plateau gluing) ─────────────────────────────────
#
# env_up  : ramp-up then EXACT plateau (the tau_hold -> inf limit of
#           _envelope_param) -> every checkpoint is valid for every tau_hold.
# env_down: ramp-down in LOCAL time t' from 0, starting from the plateau.
# The composite (A to checkpoint, then B) equals the genuine
# _envelope_param(., tau_ramp, tau_hold) up to the erf tails (~1.1e-5); the
# consistency check measures the resulting dM / dF̄.


def _envelope_up(t, tau_ramp):
    return 0.5 * (1.0 + erf((6.0 / tau_ramp) * (t - 0.5 * tau_ramp)))


def _envelope_down(t_local, tau_ramp):
    return 0.5 * (1.0 - erf((6.0 / tau_ramp) * (t_local - 0.5 * tau_ramp)))


def _make_h_s_seg(s, tau_ramp, envelope):
    """h_s(t) at float64 with a pluggable envelope (clone of
    qoct_gate_leg._make_h_s_x64; same potential, x64 integrals)."""
    centre = WELL_CENTERS_64[s]
    x_s = X_S_64[s]
    T_s = T_S_64[s]

    def h_s_t(t):
        x_ghz = x_star_1_ghz + (x_star_2_ghz - x_star_1_ghz) * envelope(t, tau_ramp)
        a = x_ghz[s] / TYPICAL_FREQ_GHZ
        b = x_ghz[NUM_WELLS + s] / TYPICAL_FREQ_GHZ
        dx = x_s - centre
        v = 0.5 * a**2 * dx**2 + (b * a**2 / 3.0) * dx**4
        return T_s + jnp.diag(v)

    return h_s_t


def _build_solver(h_list):
    return MCTDHF_n_species_DVR(
        h_list=h_list,
        g_dict=G_DICT_64,
        num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_ORB] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=1,
        imag_time=False,
    )


def _integrator(solver):
    return get_vmf_integrator(
        solver, DT, num_newton=NUM_NEWTON,
        min_solver_tol=VMF_MIN_TOL, max_solver_tol=MAX_SOLVER_TOL,
        max_solver_iter=VMF_MAX_ITER, precondition=True,
    )


# ── Reference states (anchors + doubles), identified by dominant determinant ──


def identify_doubles_states(C):
    """Map each DOUBLES_CONFIG to its bundle state index by dominant
    determinant (same discipline as identify_anchor_states; NEVER energy
    ordering). chain_step1.npz: |20>->state 5 (w 0.963), |02>->state 8 (0.917)."""
    C = np.asarray(C)
    w = np.abs(C) ** 2
    norms = w.reshape(C.shape[0], -1).sum(axis=1)
    picks = {}
    for cfg, lab in zip(DOUBLES_CONFIGS, DOUBLES_LABELS):
        w_cfg = w[(slice(None),) + cfg] / norms
        k = int(np.argmax(w_cfg))
        dom = tuple(int(d) for d in np.unravel_index(np.argmax(w[k]), w[k].shape))
        assert dom == cfg, (
            f"doubles identity check failed for {lab}: best-weight state {k} "
            f"has dominant determinant {dom}, expected {cfg}")
        assert w_cfg[k] > 0.5, (
            f"doubles {lab}: dominant-determinant weight {w_cfg[k]:.3f} <= 0.5 "
            f"in state {k}")
        picks[cfg] = (k, float(w_cfg[k]))
    return picks


def load_reference_states():
    """6 normalised reference CI tensors (4 anchors then |20>,|02>) sharing the
    idle-anchor orbital set bs0. The propagated leg returns to idle geometry, so
    the idle-anchor doubles states are the correct projection basis at T."""
    bs0 = [jnp.asarray(b, dtype=jnp.complex128) for b in bs_anchor]
    refsC, info = [], []
    for cfg, lab in zip(PAIR_CONFIGS, PAIR_LABELS):
        k, wdom = identify_anchor_states(C_anchor)[cfg]
        Ck = jnp.asarray(C_anchor[k], dtype=jnp.complex128)
        n2 = float(jnp.real(mctdh_overlap(Ck, bs0, Ck, bs0)))
        refsC.append(Ck / jnp.sqrt(n2))
        info.append((lab, k, wdom))
    for cfg, lab in zip(DOUBLES_CONFIGS, DOUBLES_LABELS):
        k, wdom = identify_doubles_states(C_anchor)[cfg]
        Ck = jnp.asarray(C_anchor[k], dtype=jnp.complex128)
        n2 = float(jnp.real(mctdh_overlap(Ck, bs0, Ck, bs0)))
        refsC.append(Ck / jnp.sqrt(n2))
        info.append((lab, k, wdom))
    return refsC, bs0, info


def _overlaps6(y_T_np, refsC, bs0, solver_shapes):
    """[<anchor_0|psi>,...,<anchor_3|psi>, <20|psi>, <02|psi>] (length 6)."""
    C_T, bs_T = solver_shapes._y_to_Cb(jnp.asarray(y_T_np))
    return np.array(
        [complex(mctdh_overlap(Cr, bs0, C_T[0], bs_T)) for Cr in refsC])


# ── Grid snapping ─────────────────────────────────────────────────────────────


def _snap(values, name=""):
    """Snap to the nearest SNAP_DT multiple; warn if anything moved."""
    arr = np.asarray(values, dtype=float)
    snapped = np.round(arr / SNAP_DT) * SNAP_DT
    moved = np.abs(snapped - arr) > 1e-9
    if np.any(moved):
        for a, s in zip(arr[moved], snapped[moved]):
            print(f"  [snap] {name} {a:g} -> {s:g} (nearest {SNAP_DT:g} multiple)")
    return snapped


def _checkpoint_index(tau_ramp, tau_hold):
    """Exact snapshot index for the state at t = tau_ramp + tau_hold (segment-A
    snapshots are at times [0, SNAP_DT, 2*SNAP_DT, ...])."""
    return int(round((tau_ramp + tau_hold) / SNAP_DT))


# ── Persistent per-state worker (spawned process) ─────────────────────────────


def _downsample_stride(n_total, target):
    return max(1, int(np.ceil(n_total / max(target, 1))))


def _worker(j, ramp_max, hold_max, traj_stem, save_traj, traj_points,
            task_q, result_q):
    """Propagate loss state j. AOT-compiles the parametric segment-A and
    segment-B scans once (both chunked, emitting a packed-y snapshot every
    SNAP_STEPS steps); compiles the genuine full leg lazily on the first 'full'
    task. When save_traj, each worker writes its OWN trajectory npz to disk
    (segment-A ramp-up+hold + per-hold segment-B ramp-down, downsampled to
    ~traj_points) — no large arrays cross the queue. Tasks:
      ('row',  r, tau_ramp, tau_holds) -> per-hold overlaps + dnorms (+ traj)
      ('full', tau_ramp, tau_hold, tag)-> single overlaps vector (+ traj)
      None                             -> shutdown
    """
    try:
        refsC, bs0, _ = load_reference_states()
        y0 = pack_Cb_to_y(refsC[j][None], bs0)
        assert y0.dtype == jnp.complex128

        num_snaps_a = int(round((ramp_max + hold_max) / SNAP_DT))
        num_snaps_b = int(np.ceil(np.ceil(ramp_max / DT) / SNAP_STEPS))

        def _chunked_scan(make_h, envelope_or_full, total_time_fn, num_snaps):
            """Build a chunked scan that integrates while t < total_time and
            emits y every SNAP_STEPS steps."""
            def run(param, y_init):
                solver = _build_solver(make_h(param))
                integ = _integrator(solver)
                total_time = total_time_fn(param)

                def step(carry, _):
                    t, y = carry
                    nc = jax.lax.cond(t < total_time,
                                      lambda: integ(t, y),
                                      lambda: (t + DT, y))
                    return nc, None

                def chunk(carry, _):
                    carry, _ = jax.lax.scan(step, carry, None, length=SNAP_STEPS)
                    return carry, carry[1]                 # emit packed y

                t0 = jnp.asarray(0.0, dtype=jnp.float64)
                (_, _), ys = jax.lax.scan(chunk, (t0, y_init), None,
                                          length=num_snaps)
                return ys                                  # (num_snaps, len_y)
            return run

        _run_segA = _chunked_scan(
            lambda tr: [_make_h_s_seg(s, tr, _envelope_up) for s in range(NUM_WELLS)],
            None, lambda tr: tr + hold_max, num_snaps_a)
        _run_segB = _chunked_scan(
            lambda tr: [_make_h_s_seg(s, tr, _envelope_down) for s in range(NUM_WELLS)],
            None, lambda tr: tr, num_snaps_b)

        tau_ex = jnp.asarray(0.5 * (ramp_max + 10.0), dtype=jnp.float64)
        t0 = time.perf_counter()
        compiled_A = jax.jit(_run_segA).lower(tau_ex, y0).compile()
        compiled_B = jax.jit(_run_segB).lower(tau_ex, y0).compile()
        t_compile = time.perf_counter() - t0

        solver_shapes = _build_solver(
            [_make_h_s_seg(s, float(tau_ex), _envelope_up) for s in range(NUM_WELLS)])

        # Lazy full-leg scan (genuine _envelope_param) for the consistency check.
        from qoct_gate_leg import _make_h_s_x64
        nsnaps_full = int(np.ceil(
            np.ceil((2.0 * ramp_max + hold_max) / DT) / SNAP_STEPS))
        compiled_full = {"fn": None}
        _run_full = _chunked_scan(
            lambda tau: [_make_h_s_x64(s, tau[0], tau[1]) for s in range(NUM_WELLS)],
            None, lambda tau: 2.0 * tau[0] + tau[1], nsnaps_full)

        result_q.put(("ready", j, t_compile))
    except Exception:
        result_q.put(("err", j, traceback.format_exc()))
        return

    def _norm2(y_np):
        C_T, bs_T = solver_shapes._y_to_Cb(jnp.asarray(y_np))
        return float(np.real(mctdh_overlap(C_T[0], bs_T, C_T[0], bs_T)))

    while True:
        task = task_q.get()
        if task is None:
            return
        try:
            if task[0] == "row":
                _, r, tau_ramp, tau_holds = task
                tau_ramp = float(tau_ramp)
                t_a = time.perf_counter()
                ys_A = np.asarray(compiled_A(
                    jnp.asarray(tau_ramp, dtype=jnp.float64), y0))
                jax.block_until_ready(ys_A)
                wall_a = time.perf_counter() - t_a
                # segment-A snapshots: prepend y0 -> times [0, SNAP_DT, ...].
                ys_full = np.concatenate([np.asarray(y0)[None], ys_A], axis=0)
                ts_a = np.arange(ys_full.shape[0]) * SNAP_DT
                n_hold = len(tau_holds)
                cols = np.full((n_hold, 6), np.nan, dtype=complex)
                dnormA = np.full(n_hold, np.nan)
                dnormB = np.full(n_hold, np.nan)
                finite = np.zeros(n_hold, dtype=bool)
                ys_b_all = np.full((n_hold, num_snaps_b, ys_full.shape[1]),
                                   np.nan, dtype=complex) if save_traj else None
                t_b = time.perf_counter()
                for k, th in enumerate(tau_holds):
                    idx = _checkpoint_index(tau_ramp, float(th))
                    y_cp = ys_full[idx]
                    dnormA[k] = abs(_norm2(y_cp) - 1.0) if np.all(
                        np.isfinite(y_cp)) else float("nan")
                    ys_B = np.asarray(compiled_B(
                        jnp.asarray(tau_ramp, dtype=jnp.float64),
                        jnp.asarray(y_cp)))
                    y_T = ys_B[-1]                          # frozen tail = t=tau_ramp
                    if np.all(np.isfinite(y_T)):
                        finite[k] = True
                        dnormB[k] = abs(_norm2(y_T) - 1.0)
                        cols[k] = _overlaps6(y_T, refsC, bs0, solver_shapes)
                    if save_traj:
                        ys_b_all[k] = ys_B
                wall_b = time.perf_counter() - t_b
                if save_traj:
                    ts_b = np.arange(1, num_snaps_b + 1) * SNAP_DT   # local time
                    sa = _downsample_stride(len(ts_a), traj_points)
                    sb = _downsample_stride(num_snaps_b, traj_points)
                    np.savez(
                        f"{traj_stem}_traj_r{r}_s{j}.npz",
                        readme=("full trajectory for hold k = segA_ys[segA_ts <= "
                                "checkpoint_times[k]] then segB_ys[k] (global "
                                "time = checkpoint_times[k] + segB_ts). packed y "
                                "-> unpack with solver._y_to_Cb."),
                        state_index=j, state_label=PAIR_LABELS[j],
                        tau_ramp=tau_ramp, tau_holds=np.asarray(tau_holds),
                        checkpoint_times=tau_ramp + np.asarray(tau_holds),
                        snap_dt=SNAP_DT, dt=DT,
                        segA_ts=ts_a[::sa], segA_ys=ys_full[::sa],
                        segB_ts=ts_b[::sb], segB_ys=ys_b_all[:, ::sb])
                result_q.put(("row", j, cols, dnormA, dnormB, finite,
                              wall_a, wall_b))

            elif task[0] == "full":
                _, tau_ramp, tau_hold, tag = task
                if compiled_full["fn"] is None:
                    compiled_full["fn"] = jax.jit(_run_full).lower(
                        jnp.asarray([tau_ramp, tau_hold], dtype=jnp.float64),
                        y0).compile()
                ys_F = np.asarray(compiled_full["fn"](
                    jnp.asarray([tau_ramp, tau_hold], dtype=jnp.float64), y0))
                jax.block_until_ready(ys_F)
                ys_full = np.concatenate([np.asarray(y0)[None], ys_F], axis=0)
                y_T = ys_full[int(round((2.0 * tau_ramp + tau_hold) / SNAP_DT))]
                col = (_overlaps6(y_T, refsC, bs0, solver_shapes)
                       if np.all(np.isfinite(y_T))
                       else np.full(6, np.nan, dtype=complex))
                if save_traj and tag:
                    ts_f = np.arange(ys_full.shape[0]) * SNAP_DT
                    sf = _downsample_stride(len(ts_f), traj_points)
                    np.savez(
                        f"{traj_stem}_traj_{tag}_s{j}.npz",
                        readme=("genuine full-leg trajectory (_envelope_param) "
                                "for glue cross-check. packed y."),
                        state_index=j, state_label=PAIR_LABELS[j],
                        tau_ramp=tau_ramp, tau_hold=tau_hold,
                        snap_dt=SNAP_DT, dt=DT,
                        ts=ts_f[::sf], ys=ys_full[::sf])
                result_q.put(("full", j, col))
            else:
                result_q.put(("err", j, f"unknown task {task[0]!r}"))
        except Exception:
            result_q.put(("err", j, traceback.format_exc()))


# ── Parent: scoring ───────────────────────────────────────────────────────────


def _score_point(M):
    """All per-grid-point scalars from a 4x4 gate matrix M."""
    Fbar = np.array([avg_gate_fidelity_up_to_localZ(SWEEP_REFS[r], M)
                     for r in REF_NAMES])
    Fxc = np.array([gate_fidelity_up_to_localZ(SWEEP_REFS[r], M)
                    for r in REF_NAMES])
    subspace = (np.abs(M) ** 2).sum(axis=0)        # per propagated column
    p01_10 = abs(M[1, 2]) ** 2
    p10_01 = abs(M[2, 1]) ** 2
    return Fbar, Fxc, subspace, p01_10, p10_01


def _assemble_row(results, tau_holds):
    """results[j] = (cols (n_hold,6), dnormA, dnormB, finite, wall_a, wall_b).
    Returns per-row arrays. doubles split is for the |11> propagated state
    (column j=3): p11 = |M[3,3]|^2, p20/p02 from that state's doubles overlaps."""
    n_hold = len(tau_holds)
    Ms = np.full((n_hold, 4, 4), np.nan, dtype=complex)
    doub11 = np.full((n_hold, 3), np.nan)          # [p11, p20, p02]
    dnormA = np.full((n_hold, 4), np.nan)
    dnormB = np.full((n_hold, 4), np.nan)
    finites = np.zeros((n_hold, 4), dtype=bool)
    for j in range(4):
        cols, dA, dB, fin, _, _ = results[j]
        Ms[:, :, j] = cols[:, :4]
        dnormA[:, j] = dA
        dnormB[:, j] = dB
        finites[:, j] = fin
        if j == 3:
            doub11[:, 0] = np.abs(cols[:, 3]) ** 2     # |<11|psi_11>|^2
            doub11[:, 1] = np.abs(cols[:, 4]) ** 2     # |<20|psi_11>|^2
            doub11[:, 2] = np.abs(cols[:, 5]) ** 2     # |<02|psi_11>|^2
    Fbar = np.full((n_hold, 3), np.nan)
    Fxc = np.full((n_hold, 3), np.nan)
    subspace = np.full((n_hold, 4), np.nan)
    transfer = np.full((n_hold, 2), np.nan)
    for k in range(n_hold):
        if np.all(np.isfinite(Ms[k])):
            fb, fx, sub, p1, p2 = _score_point(Ms[k])
            Fbar[k], Fxc[k], subspace[k] = fb, fx, sub
            transfer[k] = (p1, p2)
    return dict(Ms=Ms, Fbar=Fbar, Fxc=Fxc, subspace=subspace, transfer=transfer,
                doubles_pop_11=doub11, dnormA=dnormA, dnormB=dnormB,
                finites=finites)


# ── Parent: driver ────────────────────────────────────────────────────────────


def _spawn_workers(ramp_max, hold_max, traj_stem="", save_traj=False,
                   traj_points=TRAJ_POINTS_DEFAULT):
    ctx = mp.get_context("spawn")
    task_qs = [ctx.Queue() for _ in range(4)]
    result_q = ctx.Queue()
    workers = [ctx.Process(target=_worker,
                           args=(j, ramp_max, hold_max, traj_stem, save_traj,
                                 traj_points, task_qs[j], result_q),
                           daemon=True) for j in range(4)]
    for w in workers:
        w.start()
    t_compiles = np.full(4, np.nan)
    for _ in range(4):
        msg = result_q.get(timeout=COMPILE_TIMEOUT)
        if msg[0] == "err":
            for q in task_qs:
                q.put(None)
            raise RuntimeError(f"worker {msg[1]} ({PAIR_LABELS[msg[1]]}) failed "
                               f"to start:\n{msg[2]}")
        t_compiles[msg[1]] = msg[2]
    print(f"worker JIT compiles: {np.round(t_compiles, 1)} s", flush=True)
    return task_qs, result_q, workers, t_compiles


def _collect_row(task_qs, result_q, r, tau_ramp, tau_holds):
    for q in task_qs:
        q.put(("row", int(r), float(tau_ramp), np.asarray(tau_holds, dtype=float)))
    results = [None] * 4
    for _ in range(4):
        msg = result_q.get(timeout=EVAL_TIMEOUT)
        if msg[0] == "err":
            raise RuntimeError(f"worker {msg[1]} ({PAIR_LABELS[msg[1]]}) row "
                               f"error:\n{msg[2]}")
        _, j, cols, dA, dB, fin, wa, wb = msg
        results[j] = (cols, dA, dB, fin, wa, wb)
    return results


def _consistency_check(task_qs, result_q, refsC_info, cp, strict):
    """M_direct (genuine full leg) vs M_glued (A+B) at one snapped point."""
    cp_ramp, cp_hold = _snap([cp[0]], "cons tau_ramp")[0], \
        _snap([cp[1]], "cons tau_hold")[0]
    print(f"\n=== consistency check at (tau_ramp={cp_ramp:g}, "
          f"tau_hold={cp_hold:g}) ===", flush=True)
    for q in task_qs:
        q.put(("full", float(cp_ramp), float(cp_hold), "consistency"))
    M_direct = np.full((4, 4), np.nan, dtype=complex)
    for _ in range(4):
        msg = result_q.get(timeout=EVAL_TIMEOUT)
        if msg[0] == "err":
            raise RuntimeError(f"consistency full-leg worker {msg[1]} error:\n"
                               f"{msg[2]}")
        _, j, col = msg
        M_direct[:, j] = col[:4]
    results = _collect_row(task_qs, result_q, -1, cp_ramp, [cp_hold])
    M_glued = _assemble_row(results, [cp_hold])["Ms"][0]
    dM_max = float(np.nanmax(np.abs(M_glued - M_direct)))
    dM_fro = float(np.linalg.norm(np.nan_to_num(M_glued - M_direct)))
    dF = np.array([abs(avg_gate_fidelity_up_to_localZ(SWEEP_REFS[r], M_glued)
                       - avg_gate_fidelity_up_to_localZ(SWEEP_REFS[r], M_direct))
                   for r in REF_NAMES])
    print(f"  max|dM| = {dM_max:.3e}   ||dM||_F = {dM_fro:.3e}")
    print(f"  dF̄ per ref {REF_NAMES} = {np.array2string(dF, precision=3)}")
    if strict:
        assert dM_fro <= 1e-2, f"glue ||dM||_F {dM_fro:.2e} > 1e-2"
        assert float(np.max(dF)) <= 1e-3, f"glue dF̄ {np.max(dF):.2e} > 1e-3"
        print("  consistency PASS (asserted)")
    else:
        if dM_fro > 1e-2 or float(np.max(dF)) > 1e-3:
            print("  WARNING: glue error above test thresholds "
                  "(||dM||_F>1e-2 or dF̄>1e-3) — sweep continues, flagged")
    return dict(cons_point=np.array([cp_ramp, cp_hold]), M_direct=M_direct,
                M_glued=M_glued, dM_max=dM_max, dM_fro=dM_fro, dFbar=dF)


def run_sweep(tau_ramps, tau_hold_max, hold_stride, consistency, label,
              save_traj=True, traj_points=TRAJ_POINTS_DEFAULT):
    tau_ramps = _snap(tau_ramps, "tau_ramp")
    tau_holds = _snap(np.arange(0.0, tau_hold_max + 1e-9, hold_stride), "tau_hold")
    ramp_max, hold_max = float(np.max(tau_ramps)), float(np.max(tau_holds))
    # The worker's segment-A range must also cover the consistency point's
    # (tau_ramp, tau_hold) — its glued checkpoint sits at tau_ramp + tau_hold,
    # which can exceed the grid maxima (e.g. test grid hold_max=4 vs cons
    # hold=12.5). Snap the cons point the same way and widen the range.
    if consistency:
        cp_r = _snap([CONSISTENCY_POINT_DEFAULT[0]])[0]
        cp_h = _snap([CONSISTENCY_POINT_DEFAULT[1]])[0]
        ramp_max = max(ramp_max, float(cp_r))
        hold_max = max(hold_max, float(cp_h))
    print(f"=== sweep_map [{label}] ===")
    print(f"tau_ramps = {np.array2string(tau_ramps, precision=2)}")
    print(f"tau_holds = {np.array2string(tau_holds, precision=2)}  "
          f"(stride {hold_stride:g}, {len(tau_holds)} checkpoints)")
    print(f"refs = {REF_NAMES};  dt={DT}, caps[{VMF_MIN_TOL},{MAX_SOLVER_TOL}]"
          f"/{VMF_MAX_ITER}, newton={NUM_NEWTON}, norb={NUM_ORB}, x64, CPU",
          flush=True)

    refsC, bs0, info = load_reference_states()
    print("reference states (dominant determinant, never energy ordering):")
    for lab, k, w in info:
        print(f"  {lab}: state {k:2d}  weight {w:.4f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")   # microsec: no collision
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sweep_map_{label}_dt{DT:g}_{stamp}.npz")
    traj_stem = out_path[:-4]                           # strip .npz
    if save_traj:
        print(f"trajectories ON: ~{traj_points} pts/full-traj -> "
              f"{traj_stem}_traj_r*_s*.npz", flush=True)

    task_qs, result_q, workers, t_compiles = _spawn_workers(
        ramp_max, hold_max, traj_stem, save_traj, traj_points)

    cons = None
    if consistency:
        cons = _consistency_check(task_qs, result_q, info,
                                  CONSISTENCY_POINT_DEFAULT, strict=(label == "test"))

    rows = {k: [] for k in ("Ms", "Fbar", "Fxc", "subspace", "transfer",
                            "doubles_pop_11", "dnormA", "dnormB", "finites",
                            "walls_a", "walls_b")}

    def _save():
        out = dict(
            tau_ramps=tau_ramps, tau_holds=tau_holds, snap_dt=SNAP_DT, dt=DT,
            min_solver_tol=VMF_MIN_TOL, max_solver_tol=MAX_SOLVER_TOL,
            num_newton=NUM_NEWTON, max_solver_iter=VMF_MAX_ITER, num_orb=NUM_ORB,
            ref_names=np.array(REF_NAMES),
            ref_mats=np.stack([SWEEP_REFS[r] for r in REF_NAMES]),
            doubles_states=np.array([info[4][1], info[5][1]]),
            doubles_weights=np.array([info[4][2], info[5][2]]),
            t_compiles=t_compiles, n_rows_done=len(rows["Ms"]),
        )
        for k, v in rows.items():
            out[k] = np.array(v)
        if cons is not None:
            out.update({f"cons_{k}": v for k, v in cons.items()})
        np.savez(out_path, **out)

    try:
        for r, tr in enumerate(tau_ramps):
            t0 = time.perf_counter()
            results = _collect_row(task_qs, result_q, r, tr, tau_holds)
            row = _assemble_row(results, tau_holds)
            for k in ("Ms", "Fbar", "Fxc", "subspace", "transfer",
                      "doubles_pop_11", "dnormA", "dnormB", "finites"):
                rows[k].append(row[k])
            rows["walls_a"].append([results[j][4] for j in range(4)])
            rows["walls_b"].append([results[j][5] for j in range(4)])
            _save()
            best_k = {rn: int(np.nanargmax(row["Fbar"][:, i]))
                      for i, rn in enumerate(REF_NAMES)
                      if np.any(np.isfinite(row["Fbar"][:, i]))}
            nflag = int(np.sum((np.nan_to_num(row["dnormA"]) > 1e-6) |
                               (np.nan_to_num(row["dnormB"]) > 1e-6)))
            summ = "  ".join(
                f"{rn}:F̄={np.nanmax(row['Fbar'][:, i]):.3f}@th="
                f"{tau_holds[best_k[rn]]:g}"
                for i, rn in enumerate(REF_NAMES) if rn in best_k)
            print(f"row {r+1}/{len(tau_ramps)} tau_ramp={tr:g} "
                  f"({time.perf_counter()-t0:.0f}s): {summ}  "
                  f"[norm-flags {nflag}/{4*len(tau_holds)}]", flush=True)
    finally:
        for q in task_qs:
            q.put(None)
        for w in workers:
            w.join(timeout=30)

    print(f"\nSaved {out_path}")
    # Final summary: best per ref, and the identity-dominant (escape) region.
    Fbar = np.array(rows["Fbar"])                      # (R, n_hold, 3)
    for i, rn in enumerate(REF_NAMES):
        if np.any(np.isfinite(Fbar[..., i])):
            flat = np.nanargmax(Fbar[..., i])
            ri, ki = np.unravel_index(flat, Fbar[..., i].shape)
            print(f"  best F̄({rn}) = {Fbar[ri, ki, i]:.4f} at "
                  f"tau_ramp={tau_ramps[ri]:g}, tau_hold={tau_holds[ki]:g}")
    esc = np.array(rows["Fbar"])
    escape = np.sum(esc[..., 2] > esc[..., 0])         # identity beats sqiswap
    print(f"  adiabatic-escape grid points (F̄_identity > F̄_sqiswap): "
          f"{int(escape)}/{esc[..., 0].size}")
    return out_path


# ── verify mode ───────────────────────────────────────────────────────────────


def run_verify():
    verify_loss()                                       # existing analytic suite
    print("=== sweep-specific checks ===")
    refsC, bs0, info = load_reference_states()
    print("reference states (anchors then doubles):")
    for lab, k, w in info:
        print(f"  {lab}: state {k:2d}  dominant-det weight {w:.4f}")
    # Identity-ref analytic baselines (the escape detector).
    s2 = np.sqrt(2.0)
    checks = [
        ("identity ref, M=I", np.eye(4, dtype=complex), 1.0),
        ("identity ref, ideal sqiswap", np.asarray(M_REFS["sqiswap"]),
         (10.0 + 4.0 * s2) / 20.0),
        ("identity ref, ideal swap", np.asarray(M_REFS["swap"]), 0.4),
    ]
    for name, M, an in checks:
        fb = avg_gate_fidelity_up_to_localZ(SWEEP_REFS["identity"], M)
        print(f"[id-ref] {name}: F̄ = {fb:.6f}  (analytic {an:.6f})")
        assert abs(fb - an) <= 1e-6, f"identity-ref baseline {name}: {fb} != {an}"
    print("=== sweep verify PASSED ===")


# ── fit mode (frozen-H hold-axis fit; post-hoc, on a saved sweep npz) ──────────


def _matrix_pencil(signal, dt, n_modes):
    """Estimate complex exponents (i*omega) of a uniformly-sampled signal via
    the matrix-pencil method. signal[n] ~ sum_k c_k z_k^n, z_k = exp(-i omega_k
    dt). Returns omega_k. Used for the 'trace' frequency source (cross-check /
    fallback to the FCI plateau gaps)."""
    x = np.asarray(signal, dtype=complex)
    N = len(x)
    L = N // 2
    Y = np.array([x[i:i + L] for i in range(N - L)])    # Hankel
    Y0, Y1 = Y[:, :-1], Y[:, 1:]
    z = np.linalg.eigvals(np.linalg.pinv(Y0) @ Y1)
    z = z[np.argsort(-np.abs(z))][:n_modes]
    return (1j * np.log(z) / dt).real


def fit_M_of_hold(Ms_row, tau_holds, omegas):
    """Per-entry complex least squares M_ij(th) ~ sum_k A_k exp(-i omega_k th).
    Returns A (4,4,K) and per-entry residual (4,4)."""
    th = np.asarray(tau_holds, dtype=float)
    Phi = np.exp(-1j * np.outer(th, omegas))            # (n_hold, K)
    A = np.full((4, 4, len(omegas)), np.nan, dtype=complex)
    res = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(4):
            y = Ms_row[:, i, j]
            if np.all(np.isfinite(y)):
                a, *_ = np.linalg.lstsq(Phi, y, rcond=None)
                A[i, j] = a
                res[i, j] = float(np.linalg.norm(Phi @ a - y))
    return A, res


def _M_at(A, omegas, th):
    return np.einsum("ijk,k->ij", A, np.exp(-1j * omegas * th))


def run_fit(npz_path, omegas_arg, omegas_from, verify_optimum):
    d = np.load(npz_path, allow_pickle=True)
    tau_ramps, tau_holds = d["tau_ramps"], d["tau_holds"]
    Ms = d["Ms"]                                        # (R, n_hold, 4, 4)
    print(f"=== fit {os.path.basename(npz_path)} ===")
    print(f"{Ms.shape[0]} rows x {len(tau_holds)} holds; refs {list(d['ref_names'])}")

    if omegas_arg:
        omegas = np.array([float(x) for x in omegas_arg.split(",")])
        print(f"omegas (explicit): {np.array2string(omegas, precision=4)}")
    elif omegas_from == "trace":
        # Frequencies from the |11>-column overlap series along the hold of the
        # longest available row (most checkpoints): cross-check / fallback to
        # the FCI plateau gaps. Needs enough hold points to resolve ~6 modes.
        r = int(np.argmax(np.isfinite(Ms[:, :, 0, 0]).sum(axis=1)))
        sig = Ms[r, :, 3, 3]
        n_modes = min(6, len(tau_holds) // 2)
        omegas = _matrix_pencil(sig, float(tau_holds[1] - tau_holds[0]), n_modes)
        print(f"omegas (trace, matrix-pencil on row {r}, {n_modes} modes): "
              f"{np.array2string(np.sort(omegas), precision=4)}")
        if len(tau_holds) < 12:
            print("  WARNING: < 12 hold points cannot robustly constrain ~6 "
                  "frequencies — use the FCI plateau gaps (primary) for "
                  "production; this trace estimate is a plumbing check only.")
    else:                                               # "fci"
        raise NotImplementedError(
            "frequencies from the static two-well FCI at the target point are "
            "not implemented; pass --omegas a,b,.. or --omegas-from trace.")

    ref_mats = {n: d["ref_mats"][i] for i, n in enumerate(d["ref_names"])}
    th_dense = np.linspace(float(tau_holds[0]), float(tau_holds[-1]), 400)
    for r, tr in enumerate(tau_ramps):
        A, res = fit_M_of_hold(Ms[r], tau_holds, omegas)
        rmax = float(np.nanmax(res))
        Fb = np.array([[avg_gate_fidelity_up_to_localZ(ref_mats[n], _M_at(A, omegas, t))
                        for n in d["ref_names"]] for t in th_dense])
        line = []
        for i, n in enumerate(d["ref_names"]):
            ki = int(np.argmax(Fb[:, i]))
            line.append(f"{n}:F̄={Fb[ki, i]:.3f}@{th_dense[ki]:.2f}")
        flag = "  [FIT UNRELIABLE: residual>3e-2]" if rmax >= 3e-2 else ""
        print(f"row tau_ramp={tr:g}: max-resid={rmax:.2e}  "
              + "  ".join(line) + flag)

    if verify_optimum:
        print("\n--verify-optimum: re-propagating the best sqiswap prediction")
        # best over all rows for the sqiswap ref
        best = (-1.0, None)
        for r, tr in enumerate(tau_ramps):
            A, _ = fit_M_of_hold(Ms[r], tau_holds, omegas)
            for t in th_dense:
                f = avg_gate_fidelity_up_to_localZ(ref_mats["sqiswap"], _M_at(A, omegas, t))
                if f > best[0]:
                    best = (f, (float(tr), float(t)))
        f_pred, (tr, th) = best
        print(f"  predicted optimum: tau_ramp={tr:g}, tau_hold={th:.3f}, "
              f"F̄_pred={f_pred:.4f}")
        task_qs, result_q, workers, _ = _spawn_workers(tr, th + 1.0)
        try:
            for q in task_qs:
                q.put(("full", tr, float(_snap([th], "opt tau_hold")[0]), ""))
            M = np.full((4, 4), np.nan, dtype=complex)
            for _ in range(4):
                msg = result_q.get(timeout=EVAL_TIMEOUT)
                _, j, col = msg
                M[:, j] = col[:4]
        finally:
            for q in task_qs:
                q.put(None)
            for w in workers:
                w.join(timeout=30)
        f_prop = avg_gate_fidelity_up_to_localZ(ref_mats["sqiswap"], M)
        print(f"  propagated F̄ = {f_prop:.4f}   |dF̄| = {abs(f_prop-f_pred):.4f}")
        print("  (fit trusted only if |dF̄| is small; else fall back to grid)")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("verify")
    pt = sub.add_parser("test")
    pt.add_argument("--tau-ramp", type=float, default=22.0)
    pt.add_argument("--tau-hold-max", type=float, default=5.0)
    pt.add_argument("--consistency-point", type=str, default=None)
    pt.add_argument("--dt", type=float, default=None)
    ps = sub.add_parser("sweep")
    ps.add_argument("--tau-ramps", type=str,
                    default=",".join(f"{x:g}" for x in TAU_RAMPS_DEFAULT))
    ps.add_argument("--tau-hold-max", type=float, default=TAU_HOLD_MAX_DEFAULT)
    ps.add_argument("--hold-stride", type=float, default=HOLD_STRIDE_DEFAULT)
    ps.add_argument("--consistency-point", type=str, default=None)
    ps.add_argument("--no-consistency", action="store_true")
    ps.add_argument("--no-save-traj", action="store_true")
    ps.add_argument("--traj-points", type=int, default=TRAJ_POINTS_DEFAULT)
    ps.add_argument("--dt", type=float, default=None)
    pf = sub.add_parser("fit")
    pf.add_argument("npz")
    pf.add_argument("--omegas", type=str, default=None)
    pf.add_argument("--omegas-from", type=str, default="trace",
                    choices=["trace", "fci"])
    pf.add_argument("--verify-optimum", action="store_true")
    args = p.parse_args()

    global CONSISTENCY_POINT_DEFAULT, DT, SNAP_DT
    if getattr(args, "dt", None):
        os.environ["SWEEP_DT"] = repr(args.dt)         # inherited by spawn workers
        DT = args.dt
        SNAP_DT = SNAP_STEPS * DT                       # parent stays consistent
        print(f"dt override: dt={DT:g}, SNAP_DT={SNAP_DT:g}")
    if getattr(args, "consistency_point", None):
        a, b = args.consistency_point.split(",")
        CONSISTENCY_POINT_DEFAULT = (float(a), float(b))

    if args.mode == "verify":
        run_verify()
    elif args.mode == "test":
        run_sweep([args.tau_ramp], args.tau_hold_max, 2.0,
                  consistency=True, label="test",      # stride 2.0 -> 3 checkpoints
                  save_traj=True, traj_points=400)
    elif args.mode == "sweep":
        ramps = [float(x) for x in args.tau_ramps.split(",")]
        run_sweep(ramps, args.tau_hold_max, args.hold_stride,
                  consistency=not args.no_consistency, label="sweep",
                  save_traj=not args.no_save_traj, traj_points=args.traj_points)
    elif args.mode == "fit":
        run_fit(args.npz, args.omegas, args.omegas_from, args.verify_optimum)


if __name__ == "__main__":
    main()
