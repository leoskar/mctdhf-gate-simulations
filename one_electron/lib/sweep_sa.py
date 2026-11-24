"""State-averaged (tau_ramp, tau_hold) sweep of a gate leg by exact checkpoint reuse.

For a fixed tau_ramp (Sec. S5 D):

  segment A : one SA propagation with env_up = ramp-up then the plateau (the
              tau_hold -> inf limit of the envelope), to t = tau_ramp +
              th_max - Delta, saving a packed-y checkpoint every SNAP_DT.
  segment B : per tau_hold, restart from the checkpoint at
              t_c = tau_ramp + tau_hold - Delta and propagate the full
              envelope _envelope_param(t, tau_ramp, tau_hold) to
              T_total = 2*tau_ramp + tau_hold. One final SA state -> the
              gate matrix M.

The restart is exact: env_up(t) == _envelope_param(t, tau_ramp, th) to machine
precision for all t <= t_c, as long as the step-back Delta is large enough
that the ramp-down erf's leading tail at t_c is below eps.  The gap
down_mid - t_c = 0.5*tau_ramp + Delta does not depend on th, so one Delta
works for every hold; Delta = tau_ramp gives erfc(9) ~ 1e-37 (max|env_up -
env_param| on [0,t_c] is 0.0 in float64 for Delta >= 0.5*tau_ramp).
Identical envelopes give identical Hamiltonians on [0, t_c], so segment B
reproduces the full leg to integrator precision.

Propagation: SA (num_states=4, shared orbitals), norb=4, x64; VMF on the CPU
by default, CMF with SWEEP_INTEG=cmf (see below).  One SA process uses ~1.5
cores, so segment Bs run in parallel.

Subcommands:
    segA <tau_ramp> <th_max> [--delta D]
        propagate + checkpoint segment A; saves results/geom_2um/segA_tr<..>.npz
    segB <tau_ramp> <tau_hold> <segA_npz> [--delta D]
        one gate point from a checkpoint; saves results/geom_2um/sweep_pts/pt_tr<..>_th<..>.npz
    orchestrate --tau-ramps 30,40,50 --tau-holds 92,94,...,114 [--delta-frac 1.0]
                [--max-parallel 16] [--consistency 40,107]
        run the whole sweep in parallel; consistency point re-checked against the
        full-leg M (results/geom_2um/leg_sa_sqiswap_*.npz) before trusting it.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qoct_gate_leg import (
    DT_PROD,
    G_DICT_64,
    M_REFS,
    NUM_ORB,
    PAIR_CONFIGS,
    PAIR_LABELS,
    VMF_MAX_ITER,
    VMF_MIN_TOL,
    _make_h_s_x64,                 # GENUINE envelope (segment B)
    avg_gate_fidelity_up_to_localZ,
    gate_matrix,
    load_loss_anchors,
    mctdh_overlap,
)
from sweep_map import (
    SNAP_DT,
    SNAP_STEPS,
    _envelope_up,
    _make_h_s_seg,                 # env_up (segment A)
    DOUBLES_CONFIGS,
    DOUBLES_LABELS,
    identify_doubles_states,
)
from rte_step import (
    MAX_SOLVER_TOL,
    NUM_DVR,
    NUM_NEWTON,
    NUM_WELLS,
    C_anchor,
    bs_anchor,
)
from mctdhf import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.integrators import (get_vmf_integrator,
                                get_cmf_half_step_integrator)
from regauge import make_regauge
from gate_fidelity import gate_fidelity_up_to_localZ

assert NUM_WELLS == 4

# ── Active leg: w0w1 (adjacent, default) or w0w3 (long-range) ──────────────────
# Both reuse the SAME validated machinery; w0w3 only retargets (a) the resonance
# x_star_2 to the stage-5 w0w3 anchor, (b) the loss anchors to pair (0,3), and
# (c) the output subdir. The validated schedule builders close over their
# module's x_star_2_ghz (qoct_gate_leg._make_h_s_x64 / sweep_map._make_h_s_seg),
# and load_loss_anchors uses qoct_gate_leg.PAIR_CONFIGS — so we rebind those
# module globals rather than re-implement the builders. x_star_1 (idle) + the
# geometry integrals are leg-independent. (Set GATE_LEG=w0w3 in the env.)
import qoct_gate_leg as _gl
import sweep_map as _sm
GATE_LEG = os.environ.get("GATE_LEG", "w0w1")
if GATE_LEG == "w0w1":
    PAIR_CONFIGS_LEG = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)]
    DOUBLES_CFG_LEG = [(2, 0, 0, 0), (0, 2, 0, 0)]      # |20> well0, |02> well1
    _LEG_SUB, _LEG_PAIR = "", (0, 1)
elif GATE_LEG == "w0w3":
    from geometry_config import ANCHOR_PATHS as _AP
    _X5 = jnp.asarray(np.load(_AP[5])["best_x"], dtype=jnp.float64)   # w0w3 resonance
    _gl.x_star_2_ghz = _X5                              # segB genuine schedule
    _sm.x_star_2_ghz = _X5                              # segA env_up schedule
    PAIR_CONFIGS_LEG = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0, 1)]
    DOUBLES_CFG_LEG = [(2, 0, 0, 0), (0, 0, 0, 2)]      # |20> well0, |02> well3
    _LEG_SUB, _LEG_PAIR = "w0w3", (0, 3)
elif GATE_LEG == "w1w2":
    from geometry_config import ANCHOR_PATHS as _AP
    _X = jnp.asarray(np.load(_AP["3a"])["best_x"], dtype=jnp.float64)  # w1w2 resonance
    _gl.x_star_2_ghz = _X                               # segB genuine schedule
    _sm.x_star_2_ghz = _X                               # segA env_up schedule
    PAIR_CONFIGS_LEG = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0)]
    DOUBLES_CFG_LEG = [(0, 2, 0, 0), (0, 0, 2, 0)]      # |20> well1, |02> well2
    _LEG_SUB, _LEG_PAIR = "w1w2", (1, 2)
elif GATE_LEG == "w2w3":
    from geometry_config import ANCHOR_PATHS as _AP
    _X = jnp.asarray(np.load(_AP["4a"])["best_x"], dtype=jnp.float64)  # w2w3 resonance
    _gl.x_star_2_ghz = _X                               # segB genuine schedule
    _sm.x_star_2_ghz = _X                               # segA env_up schedule
    PAIR_CONFIGS_LEG = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1)]
    DOUBLES_CFG_LEG = [(0, 0, 2, 0), (0, 0, 0, 2)]      # |20> well2, |02> well3
    _LEG_SUB, _LEG_PAIR = "w2w3", (2, 3)
else:
    raise ValueError(f"GATE_LEG must be w0w1|w0w3|w1w2|w2w3, got {GATE_LEG!r}")
_gl.PAIR_CONFIGS = PAIR_CONFIGS_LEG                     # load_loss_anchors identity
_gl.ACTIVE_PAIR = _LEG_PAIR

# Which gate the sweep scores segB against (M_ref). Default "sqiswap" keeps every
# existing sqiswap sweep / aggregate / validate path byte-for-byte; "swap"
# retargets segB's Fbar/F_xc to the SWAP reference. The swept gate type also
# isolates the output subdir (below) so a SWAP sweep's checkpoints/points never
# collide with a prior sqiswap sweep's pt_ files in the same pair dir.
SWEEP_LEG_TYPE = os.environ.get("SWEEP_LEG_TYPE", "sqiswap")
assert SWEEP_LEG_TYPE in M_REFS, f"SWEEP_LEG_TYPE must be in {list(M_REFS)}"

# Mean-field spectator switch (MEANFIELD_SPEC=1): spectators at norb=1 (TDHF),
# active pair at NUM_ORB. Offset against norb=4 (spectator_norb_test.py):
# w2w3 SWAP -4.7e-4, w0w1 -4.4e-3, w1w2 -3.0e-3 (pessimistic).
# Role: cheap parameter OPTIMIZER — optima are confirmed at
# full norb=4 before reporting, and the mf-vs-norb4 gap is itself evidence that
# the orbital count matters. Initial anchors get their spectator modes projected
# to the leading SA natural orbital (build_reduced_init); scoring stays against
# the TRUE norb=4 anchors.
MEANFIELD_SPEC = os.environ.get("MEANFIELD_SPEC", "0") == "1"
NORB_LIST = [NUM_ORB if s in _LEG_PAIR else (1 if MEANFIELD_SPEC else NUM_ORB)
             for s in range(NUM_WELLS)]
print(f"[sweep_sa] GATE_LEG={GATE_LEG}  pair={_LEG_PAIR}  configs={PAIR_CONFIGS_LEG}  "
      f"sweep_leg={SWEEP_LEG_TYPE}  meanfield_spec={MEANFIELD_SPEC}  "
      f"norb={NORB_LIST}", flush=True)

# Output subdir: sqiswap uses the results root for w0w1, else the
# pair name); any other swept gate goes under '<pair>_<gate>' (e.g. w0w1_swap).
# Mean-field artifacts isolate under '<pair>_<gate>_mf' (checkpoint y-shapes
# differ from norb=4 — the layouts must never mix).
# ── integrator selection ──────────────────────────────────────────────────────
# Default: VMF at DT_PROD.
# SWEEP_INTEG=cmf switches to the configuration validated by the
# timestep check (gate_sweeps/timestep_check): CMF half-step, c/b propagator "diagonalize",
# SWEEP_DT_DIV x the production step, plus exact gauge re-orthonormalisation
# after every chunk.
#
# Why at all: VMF's implicit-midpoint phase error is O(dt^2) and these legs
# accumulate ~1e4 rad of dynamical phase, so the ~1 rad conditional phases that
# make the gate entangling come out of a four-digit cancellation. Two VMF runs
# differing only in dt agreed on population to 1e-05 and on the gate to only
# 6e-02. CMF with both substeps propagated exactly leaves only the O(dt^3)
# splitting error; measured on sqi12 it reproduces the dt fidelity to 6e-09 at
# 10dt while running 9x faster.
#
# Note dt only has to divide the SNAP_DT checkpoint grid -- segA/segB EXACT
# reuse is on that grid, so it is untouched by the step change. The per-chunk
# regauge is an exact gauge transformation, so regauged segA checkpoints remain
# exactly reusable: the EOM is gauge-covariant and the scoring is overlap-based
# and hence gauge-invariant.
SWEEP_INTEG = os.environ.get("SWEEP_INTEG", "vmf")
SWEEP_DT_DIV = float(os.environ.get("SWEEP_DT_DIV", "0.1"))   # 0.1 => 10 x dt
SWEEP_C_PROP = os.environ.get("SWEEP_C_PROP", "diagonalize")
SWEEP_B_PROP = os.environ.get("SWEEP_B_PROP", "diagonalize")
if SWEEP_INTEG == "cmf":
    SWEEP_DT = DT_PROD / SWEEP_DT_DIV
    SWEEP_STEPS = int(round(SNAP_DT / SWEEP_DT))
    assert SWEEP_STEPS >= 1 and abs(SWEEP_STEPS * SWEEP_DT - SNAP_DT) < 1e-12, (
        f"dt={SWEEP_DT} does not divide the SNAP_DT={SNAP_DT} checkpoint grid "
        f"that every existing artifact is keyed to")
else:
    SWEEP_DT, SWEEP_STEPS = DT_PROD, SNAP_STEPS
print(f"[sweep_sa] integrator={SWEEP_INTEG} dt={SWEEP_DT:g} "
      f"({SWEEP_STEPS} steps/chunk)"
      + (f" c={SWEEP_C_PROP} b={SWEEP_B_PROP} regauge=on"
         if SWEEP_INTEG == "cmf" else ""), flush=True)


_OUT_SUB = _LEG_SUB if SWEEP_LEG_TYPE == "sqiswap" else f"{GATE_LEG}_{SWEEP_LEG_TYPE}"
if MEANFIELD_SPEC:
    _OUT_SUB = f"{GATE_LEG}_{SWEEP_LEG_TYPE}_mf"
# CMF results go to their own root: a VMF segment-A checkpoint has the same
# shape as a CMF one, so a shared directory would let segment B silently
# resume from the wrong one.
if SWEEP_INTEG != "vmf":
    OUT_DIR = os.path.join(HERE, "results", "cmf_campaign",
                           _OUT_SUB or f"{GATE_LEG}_{SWEEP_LEG_TYPE}")
else:
    OUT_DIR = (os.path.join(HERE, "results", "geom_2um", _OUT_SUB) if _OUT_SUB
               else os.path.join(HERE, "results", "geom_2um"))
PTS_DIR = os.path.join(OUT_DIR, "sweep_pts")
SEGA_DIR = os.path.join(OUT_DIR, "segA")
for d in (OUT_DIR, PTS_DIR, SEGA_DIR):
    os.makedirs(d, exist_ok=True)


def _snap(t):
    """Snap a time to the nearest SNAP_DT multiple (checkpoint grid)."""
    return round(round(t / SNAP_DT) * SNAP_DT, 10)


def _sa_solver(h_list):
    return MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=G_DICT_64,
        num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=NORB_LIST,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=4, imag_time=False)


def _vmf(solver):
    return get_vmf_integrator(
        solver, DT_PROD, num_newton=NUM_NEWTON,
        min_solver_tol=VMF_MIN_TOL, max_solver_tol=MAX_SOLVER_TOL,
        max_solver_iter=VMF_MAX_ITER, precondition=True)


def _integrator(solver):
    if SWEEP_INTEG == "cmf":
        return get_cmf_half_step_integrator(
            solver, SWEEP_DT, c_propagator=SWEEP_C_PROP,
            b_propagator=SWEEP_B_PROP)
    if SWEEP_INTEG != "vmf":
        raise SystemExit(f"bad SWEEP_INTEG={SWEEP_INTEG} (vmf|cmf)")
    return _vmf(solver)


def _sa_y0():
    """Packed SA initial state (4 loss anchors, shared orbitals) + (C4, bs0).
    C4/bs0 are ALWAYS the full norb=4 anchors (scoring frame); under
    MEANFIELD_SPEC the packed state has its spectator modes projected to the
    leading SA natural orbital (build_reduced_init)."""
    C4, bs0 = load_loss_anchors()
    if MEANFIELD_SPEC:
        from spectator_norb_test import build_reduced_init
        spect = tuple(s for s in range(NUM_WELLS) if s not in _LEG_PAIR)
        C_red, bs_red, retained, ov = build_reduced_init(C4, bs0, 1, spect)
        print(f"[sweep_sa] meanfield init: spectators {spect} -> leading SA NO "
              f"(retained {[round(retained[s], 6) for s in spect]}, state-prep "
              f"overlaps {np.round(ov, 6)})", flush=True)
        C_init = jnp.asarray(C_red, dtype=jnp.complex128)
        bs_init = [jnp.asarray(b, dtype=jnp.complex128) for b in bs_red]
    else:
        C_init = jnp.stack([jnp.asarray(c, dtype=jnp.complex128) for c in C4])
        bs_init = [jnp.asarray(b, dtype=jnp.complex128) for b in bs0]
    return pack_Cb_to_y(C_init, bs_init), C4, bs0


def _chunked_scan(solver, integrator):
    """Compiled one-SNAP_DT chunk: (t,y) -> (t,y) after SWEEP_STEPS steps.

    Under CMF the chunk ends with an exact gauge re-orthonormalisation; see
    regauge.py for why that is mandatory rather than cosmetic.
    """
    def _step(carry, _):
        t, y = carry
        return integrator(t, y), None

    if SWEEP_INTEG == "cmf":
        regauge = make_regauge(solver)

        def _chunk(carry):
            t, y = jax.lax.scan(_step, carry, None, length=SWEEP_STEPS)[0]
            return (t, regauge(y))
    else:
        def _chunk(carry):
            carry, _ = jax.lax.scan(_step, carry, None, length=SWEEP_STEPS)
            return carry

    return jax.jit(_chunk)


# ── segment A : ramp-up + EXACT plateau, checkpointed ─────────────────────────


def run_segA(tau_ramp, th_max, delta):
    tau_ramp = _snap(tau_ramp)
    t_end = _snap(tau_ramp + th_max - delta)          # last checkpoint needed
    n_chunks = int(round(t_end / SNAP_DT))
    print(f"=== segment A  tau_ramp={tau_ramp}  th_max={th_max}  delta={delta} "
          f"=> checkpoint to t_end={t_end} ({n_chunks} snapshots @ SNAP_DT="
          f"{SNAP_DT}) ===", flush=True)

    h_list = [_make_h_s_seg(s, tau_ramp, _envelope_up) for s in range(NUM_WELLS)]
    solver = _sa_solver(h_list)
    y0, _, _ = _sa_y0()
    chunk = _chunked_scan(solver, _integrator(solver))

    t0 = time.perf_counter()
    carry = chunk((jnp.asarray(0.0), y0))
    jax.block_until_ready(carry)
    print(f"  compile+first chunk: {time.perf_counter()-t0:.1f}s", flush=True)

    ts = [0.0]
    ys = [np.asarray(y0)]
    carry = (jnp.asarray(0.0), y0)
    tic = time.perf_counter()
    for c in range(n_chunks):
        carry = chunk(carry)
        jax.block_until_ready(carry)
        t, y = carry
        ts.append(float(t))
        ys.append(np.asarray(y))
        if (c + 1) % 50 == 0:
            print(f"  segA chunk {c+1}/{n_chunks}  t={float(t):.1f}  "
                  f"finite={bool(np.all(np.isfinite(np.asarray(y))))}  "
                  f"({time.perf_counter()-tic:.0f}s)", flush=True)
    wall = time.perf_counter() - tic
    out = os.path.join(SEGA_DIR, f"segA_tr{tau_ramp:g}.npz")
    np.savez(out, tau_ramp=tau_ramp, th_max=th_max, delta=delta,
             snap_dt=SNAP_DT, ts=np.array(ts), ys=np.stack(ys), wall=wall)
    print(f"  segA done: {wall:.0f}s, {len(ts)} checkpoints -> {out}", flush=True)
    return out


# ── segment B : genuine ramp-down from a checkpoint -> one gate point ──────────


def _load_refs6(bs0):
    """Active-pair doubles leakage refs |20>,|02> as single determinants in the
    bs0 frame (pair-specific via DOUBLES_CFG_LEG; no bundle dependency, so it
    works for w0w3 where |0002> isn't a clean bundle state)."""
    refsC = []
    for cfg in DOUBLES_CFG_LEG:
        C = np.zeros((NUM_ORB,) * NUM_WELLS, dtype=complex)
        C[cfg] = 1.0
        C = jnp.asarray(C)
        n2 = float(jnp.real(mctdh_overlap(C, bs0, C, bs0)))
        refsC.append(C / jnp.sqrt(n2))
    return refsC                                  # [|20>, |02>]


# Save the segB ramp-down trajectory (packed-y at every TRAJ_STRIDE-th SNAP_DT
# chunk) alongside the endpoint M, so the FULL leg trajectory (segA prefix +
# segB) is reconstructible for overlap GIFs / any post-hoc analysis. The previous
# sweep_map saved these per-leg trajectories; do the same. The stored trajectory
# is downcast to complex64 (TRAJ_DTYPE) — purely a VISUALISATION/analysis copy,
# ~7 digits, well below the VMF residual ~1e-5 — which halves it to ~12 MB/leg
# (~1.25 GB for a 105-point sweep). The propagation stays complex128, the
# endpoint M is computed from the x64 carry, and the segA CHECKPOINTS stay
# complex128 (segB restarts from them — full precision for the EXACT reuse).
# Disable with SAVE_SEGB_TRAJ=0; coarsen with TRAJ_STRIDE>1.
SAVE_SEGB_TRAJ = os.environ.get("SAVE_SEGB_TRAJ", "1") != "0"
TRAJ_STRIDE = int(os.environ.get("TRAJ_STRIDE", "1"))
TRAJ_DTYPE = np.complex64


def run_segB(tau_ramp, tau_hold, segA_npz, delta):
    tau_ramp = _snap(tau_ramp)
    tau_hold = _snap(tau_hold)
    t_c = _snap(tau_ramp + tau_hold - delta)
    T_total = _snap(2.0 * tau_ramp + tau_hold)
    A = np.load(segA_npz)
    idx = int(round(t_c / A["snap_dt"]))
    assert abs(A["ts"][idx] - t_c) < 1e-6, (
        f"checkpoint time {A['ts'][idx]} != t_c {t_c} (segA t_end too short?)")
    y_c = jnp.asarray(A["ys"][idx])
    n_chunks = int(round((T_total - t_c) / SNAP_DT))
    print(f"=== segment B  tau_ramp={tau_ramp} tau_hold={tau_hold}  "
          f"t_c={t_c} -> T_total={T_total}  ({n_chunks} chunks) ===", flush=True)

    h_list = [_make_h_s_x64(s, tau_ramp, tau_hold) for s in range(NUM_WELLS)]
    solver = _sa_solver(h_list)
    chunk = _chunked_scan(solver, _integrator(solver))

    carry = (jnp.asarray(t_c), y_c)
    ts_traj = [float(t_c)]
    ys_traj = [np.asarray(y_c)] if SAVE_SEGB_TRAJ else None     # ramp-down trajectory
    tic = time.perf_counter()
    for c in range(n_chunks):
        carry = chunk(carry)
        jax.block_until_ready(carry)
        if SAVE_SEGB_TRAJ and ((c + 1) % TRAJ_STRIDE == 0 or c == n_chunks - 1):
            t, y = carry
            ts_traj.append(float(t))
            ys_traj.append(np.asarray(y))
    wall = time.perf_counter() - tic
    t_fin, y_fin = carry
    finite = bool(np.all(np.isfinite(np.asarray(y_fin))))

    # gate matrix (loss) + leakage diagnostics
    _, C4, bs0 = _sa_y0()
    C_T, bs_T = solver._y_to_Cb(y_fin)
    M = gate_matrix(C4, bs0, [(C_T[j], bs_T) for j in range(4)])
    refs6 = _load_refs6(bs0)
    doub = np.array([[complex(mctdh_overlap(Cr, bs0, C_T[j], bs_T)) for j in range(4)]
                     for Cr in refs6])              # (2 doubles, 4 states)
    Fbar = avg_gate_fidelity_up_to_localZ(M_REFS[SWEEP_LEG_TYPE], M)
    Fxc = gate_fidelity_up_to_localZ(M_REFS[SWEEP_LEG_TYPE], M)
    subspace = (np.abs(M) ** 2).sum(axis=0)
    transfer = 0.5 * (abs(M[1, 2]) ** 2 + abs(M[2, 1]) ** 2)

    out = os.path.join(PTS_DIR, f"pt_tr{tau_ramp:g}_th{tau_hold:g}.npz")
    traj = dict(segB_ts=np.array(ts_traj),
                segB_ys=np.stack(ys_traj).astype(TRAJ_DTYPE),   # complex64 (viz)
                traj_stride=TRAJ_STRIDE) if SAVE_SEGB_TRAJ else {}
    np.savez(out, tau_ramp=tau_ramp, tau_hold=tau_hold, t_c=t_c, T_total=T_total,
             delta=delta, M=M, doubles=doub, Fbar=Fbar, F_xc=Fxc,
             subspace=subspace, transfer=transfer, finite=finite, wall=wall,
             **traj)
    print(f"  (tr={tau_ramp:g},th={tau_hold:g}) Fbar={Fbar:.5f}  transfer="
          f"{transfer:.4f}  |11>sub={subspace[3]:.4f}  finite={finite}  "
          f"({wall:.0f}s)  -> {out}", flush=True)
    return out


# ── orchestration : segA per tau_ramp, then segB grid, in parallel ────────────


def _launch(args_list):
    return subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args_list,
                            cwd=os.path.join(HERE, ".."))


def _run_parallel(tasks, max_parallel, label):
    """tasks = list of arg-lists; run with a concurrency cap; return on all done."""
    print(f"\n--- {label}: {len(tasks)} tasks, max_parallel={max_parallel} ---",
          flush=True)
    running, pending, t0 = [], list(tasks), time.perf_counter()
    failed = []
    while pending or running:
        while pending and len(running) < max_parallel:
            a = pending.pop(0)
            running.append((a, _launch(a)))
        time.sleep(2.0)
        still = []
        for a, p in running:
            rc = p.poll()
            if rc is None:
                still.append((a, p))
            elif rc != 0:
                failed.append((a, rc))
                print(f"  TASK FAILED rc={rc}: {' '.join(a)}", flush=True)
        running = still
    print(f"--- {label} done in {time.perf_counter()-t0:.0f}s "
          f"({len(failed)} failures) ---", flush=True)
    return failed


def orchestrate(tau_ramps, tau_holds, delta_frac, max_parallel, consistency):
    tau_ramps = [_snap(r) for r in tau_ramps]
    tau_holds = [_snap(h) for h in tau_holds]
    th_max = max(tau_holds)
    print(f"=== SA checkpoint-reuse sweep ===\n  tau_ramps={tau_ramps}\n  "
          f"tau_holds={tau_holds}\n  delta_frac={delta_frac}  "
          f"max_parallel={max_parallel}  consistency={consistency}", flush=True)

    # 1) segment A per tau_ramp (parallel)
    segA_tasks, segA_files = [], {}
    for r in tau_ramps:
        delta = _snap(delta_frac * r)
        f = os.path.join(SEGA_DIR, f"segA_tr{r:g}.npz")
        segA_files[r] = (f, delta)
        segA_tasks.append(["segA", repr(r), repr(th_max), "--delta", repr(delta)])
    if _run_parallel(segA_tasks, max_parallel, "segment A"):
        sys.exit("ABORT: a segment-A task failed")

    # 2) segment B grid (parallel)
    segB_tasks = []
    for r in tau_ramps:
        f, delta = segA_files[r]
        for h in tau_holds:
            segB_tasks.append(["segB", repr(r), repr(h), f, "--delta", repr(delta)])
    if _run_parallel(segB_tasks, max_parallel, "segment B grid"):
        print("WARNING: some segment-B tasks failed (see above)", flush=True)

    # 3) aggregate
    rows = []
    for r in tau_ramps:
        for h in tau_holds:
            pt = os.path.join(PTS_DIR, f"pt_tr{r:g}_th{h:g}.npz")
            if not os.path.isfile(pt):
                print(f"  MISSING {pt}", flush=True)
                continue
            d = np.load(pt, allow_pickle=True)
            rows.append((r, h, float(d["Fbar"]), float(d["F_xc"]),
                         float(d["transfer"]), float(d["subspace"][3]),
                         bool(d["finite"])))
    rows.sort(key=lambda x: -x[2])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table = os.path.join(OUT_DIR, f"sweep_sa_{stamp}.npz")
    np.savez(table, tau_ramps=np.array(tau_ramps), tau_holds=np.array(tau_holds),
             rows=np.array([r[:6] for r in rows], dtype=float))
    print("\n=== SWEEP RESULTS (sorted by Fbar) ===")
    print("  tau_ramp  tau_hold   Fbar      F_xc    transfer  |11>sub  finite")
    for r, h, fb, fx, tr, s11, fin in rows:
        print(f"   {r:6.1f}   {h:7.1f}   {fb:.5f}  {fx:.5f}  {tr:.4f}   "
              f"{s11:.4f}  {fin}")

    # 4) consistency vs the full-leg M (proves end-to-end exactness)
    if consistency is not None:
        cr, ch = _snap(consistency[0]), _snap(consistency[1])
        pt = os.path.join(PTS_DIR, f"pt_tr{cr:g}_th{ch:g}.npz")
        legs = sorted(glob.glob(os.path.join(OUT_DIR, f"leg_sa_{SWEEP_LEG_TYPE}_*.npz")))
        if os.path.isfile(pt) and legs:
            M_sweep = np.load(pt, allow_pickle=True)["M"]
            d_leg = np.load(legs[-1], allow_pickle=True)
            if abs(float(d_leg["tau_ramp"]) - cr) < 1e-6 and abs(float(d_leg["tau_hold"]) - ch) < 1e-6:
                M_leg = d_leg["M"]
                dM = float(np.linalg.norm(M_sweep - M_leg))
                print(f"\n[consistency] sweep ({cr:g},{ch:g}) vs full leg "
                      f"{os.path.basename(legs[-1])}: ||dM||_F = {dM:.2e}  "
                      f"(exact reuse: ~1e-9)")
    print(f"\nSaved {table}")


# ── standalone aggregation + validation (for the SLURM array path) ────────────


def aggregate_glob(tag=""):
    """Build the sweep table from every per-point npz in PTS_DIR (grid-agnostic;
    the SLURM array writes the points, this collects them)."""
    rows = []
    for pt in sorted(glob.glob(os.path.join(PTS_DIR, "pt_tr*_th*.npz"))):
        d = np.load(pt, allow_pickle=True)
        rows.append((float(d["tau_ramp"]), float(d["tau_hold"]),
                     float(d["Fbar"]), float(d["F_xc"]), float(d["transfer"]),
                     float(d["subspace"][3]), bool(d["finite"])))
    rows.sort(key=lambda x: -x[2])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table = os.path.join(OUT_DIR, f"sweep_sa_{tag+'_' if tag else ''}{stamp}.npz")
    np.savez(table, rows=np.array([r[:6] for r in rows], dtype=float))
    print(f"=== SWEEP TABLE ({len(rows)} points, sorted by Fbar) ===")
    print("  tau_ramp  tau_hold   Fbar      F_xc    transfer  |11>sub  finite")
    for r, h, fb, fx, tr, s11, fin in rows:
        print(f"   {r:6.1f}   {h:7.1f}   {fb:.5f}  {fx:.5f}  {tr:.4f}   "
              f"{s11:.4f}  {fin}")
    print(f"\nSaved {table}")
    return table


def compare_leg(pt_npz, leg_npz, tol=1e-4):
    """Validate checkpoint reuse against a full leg.  The gate is defined only
    up to the idle dynamical phases (a local-Z(x)Z + global gauge, ~10^3 rad over
    a leg), so a raw ||M_segB - M_leg|| is phase-sensitive: a small endpoint
    difference between the two runs shows up as a per-state idle phase
    e^{-i E_j dt}, which F_bar gauges out.  The dynamics agree iff the moduli
    |M| and the gauge-corrected F_bar agree; gate on those, report the raw
    ||dM|| for context."""
    M_pt = np.load(pt_npz, allow_pickle=True)["M"]
    L = np.load(leg_npz, allow_pickle=True)
    M_leg = L["M"]
    raw = float(np.linalg.norm(M_pt - M_leg))
    mod = float(np.linalg.norm(np.abs(M_pt) - np.abs(M_leg)))      # phase-invariant
    Fp = avg_gate_fidelity_up_to_localZ(M_REFS[SWEEP_LEG_TYPE], M_pt)
    Fl = avg_gate_fidelity_up_to_localZ(M_REFS[SWEEP_LEG_TYPE], M_leg)
    dF = abs(Fp - Fl)
    print(f"[validate] checkpoint-reuse vs full leg "
          f"(tr={float(L['tau_ramp']):g}, th={float(L['tau_hold']):g}):")
    print(f"  || |M_segB|-|M_leg| ||_F = {mod:.2e}   (moduli; dynamics identical)")
    print(f"  |F_bar_segB - F_bar_leg| = {dF:.2e}   (F_bar {Fp:.6f} vs {Fl:.6f})")
    print(f"  raw ||dM||_F = {raw:.2e}   (phase-sensitive; benign idle-phase "
          f"gauge — reported, NOT gated)")
    if mod > tol or dF > tol:
        sys.exit(f"VALIDATION FAILED: moduli {mod:.2e} / dF_bar {dF:.2e} > {tol:.0e}")
    print(f"[validate] PASS — checkpoint reuse reproduces the gate "
          f"(moduli {mod:.1e}, dF_bar {dF:.1e}).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("segA"); a.add_argument("tau_ramp", type=float)
    a.add_argument("th_max", type=float); a.add_argument("--delta", type=float, required=True)
    b = sub.add_parser("segB"); b.add_argument("tau_ramp", type=float)
    b.add_argument("tau_hold", type=float); b.add_argument("segA_npz")
    b.add_argument("--delta", type=float, required=True)
    o = sub.add_parser("orchestrate")
    o.add_argument("--tau-ramps", required=True)
    o.add_argument("--tau-holds", required=True)
    o.add_argument("--delta-frac", type=float, default=1.0)
    o.add_argument("--max-parallel", type=int, default=16)
    o.add_argument("--consistency", default="40,107")
    g = sub.add_parser("aggregate"); g.add_argument("--tag", default="")
    c = sub.add_parser("compare"); c.add_argument("pt_npz"); c.add_argument("leg_npz")
    c.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    if args.cmd == "segA":
        run_segA(args.tau_ramp, args.th_max, args.delta)
    elif args.cmd == "segB":
        run_segB(args.tau_ramp, args.tau_hold, args.segA_npz, args.delta)
    elif args.cmd == "orchestrate":
        trs = [float(x) for x in args.tau_ramps.split(",")]
        ths = [float(x) for x in args.tau_holds.split(",")]
        cons = None if args.consistency.lower() == "none" else \
            tuple(float(x) for x in args.consistency.split(","))
        orchestrate(trs, ths, args.delta_frac, args.max_parallel, cons)
    elif args.cmd == "aggregate":
        aggregate_glob(args.tag)
    elif args.cmd == "compare":
        compare_leg(args.pt_npz, args.leg_npz, args.tol)


if __name__ == "__main__":
    # SWEEP_DEV=cuda runs the propagation on the GPU (CMF is throughput-bound and
    # gains 2.34x there, with CPU/GPU agreeing to strict 1 - 1.1e-16). Selection
    # MUST go through jax.default_device: this module and qoct_gate_leg pin
    # jax_platform_name='cpu' at import, so JAX_PLATFORMS=cuda would leave them
    # with no cpu backend and fail at import. Orchestration spawns one subprocess
    # per point and each inherits the env, so note that GPU concurrency is
    # pointless here (1/2/4 processes -> 1.00/0.97/1.11x aggregate).
    if os.environ.get("SWEEP_DEV", "cpu") == "cuda":
        with jax.default_device(jax.devices("cuda")[0]):
            main()
    else:
        main()
