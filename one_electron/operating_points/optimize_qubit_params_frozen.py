"""Frozen-spectator variant of `optimize_qubit_params.py`.

For the superposition stages (2/3/4) the two wells that are *not* part of the
superposition are FROZEN at their anchor-1 (stage-1) values, and only the
active pair's (alpha, beta) are optimised. The active alphas are box-bounded to
stay >= `FREEZE_GAP_GHZ` from each frozen neighbour while remaining free to be
mutually degenerate (the superposition resonance lives in that degeneracy).

Why
---
Letting all four wells float made the optimiser cram the whole alpha ladder near
the 15 GHz cap (the dressing terms reward tight wells), e.g. step3 landing on
[12.6, 14.4, 14.4, 14.4] -- the *high spectator* grazing the resonant pair at
0.02 GHz, and the lower wells frozen wherever the warm start left them. Pinning
the spectators to the idle (anchor-1) shape:

  * makes them static across *all* anchors, so anchor-to-anchor transitions move
    only the active wells -> no unnecessary level crossings (the reason the
    ascending property was wanted in the first place);
  * forces the spread the cramming destroyed -- stage3 -> [7.69, ~11, ~11,
    14.96], the high spectator a full ~3 GHz clear of the pair;
  * collapses the search to ~4 well-conditioned DOF, so BOBYQA converges
    reliably instead of wandering the flat spectator directions.

This file does NOT overwrite `optimize_qubit_params.py`; it imports and reuses
its machinery (the module-level solver, `evaluate_params`, bounds/IO helpers).
Precision/device are inherited unchanged from that module (no silent x64 switch
-- the landscape stays comparable to the existing anchor runs).

CLI
---
    optimize_qubit_params_frozen.py <run_name> <loss_name> <freeze_from>

  freeze_from : prior run stem under ITE_trajectories/ (or a path to an .npz).
                Its `best_x` supplies the frozen spectator (alpha, beta) AND the
                default x0 for the active wells. For the chain this is the
                stage-1 anchor, e.g. "<base>_step1_stage1_singles_ground".

As run by the chain: keep the good (ungapped) anchor
and freeze the spectators there; gap only the ACTIVE wells. Mode is per-stage,
from `ALPHA_ORDERING`:
  * "ascending" (stages 2/3/4): active pair held >= FREEZE_GAP_GHZ (2 GHz) from
    each frozen spectator -- keeps the pair clear of the high/low spectators
    while staying deep (the anchor's spectators are already well-separated, so
    the active windows sit ~10-14 GHz, never shallow).
  * "none" (stage 5, w0w3): active ends free over full bounds (they must cross
    the frozen middles to reach the long-range resonance).
The anchor itself is NOT gapped here (that is what pushed the ladder shallow and
broke the |...1> single-excitation identification); only the active wells are.
"""

from __future__ import annotations

import importlib
import os
import sys
import time

import numpy as np
import pybobyqa

# Make `mctdhf`, `utils`, `losses`, and the sibling module importable
# regardless of cwd. Importing `optimize_qubit_params` builds the module-level
# solver (MCTDHF) exactly once -- we reuse it via `evaluate_params`.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)

import optimize_qubit_params as oqp
from optimize_qubit_params import (
    ALPHA_BOUNDS_GHZ,
    BETA_BOUNDS_GHZ,
    CONFIGS,
    NUM_WELLS,
    _normalise_bounds,
    _resolve_warm_from,
    _save_trajectory,
    evaluate_params,
)

# ── Freeze configuration ──────────────────────────────────────────────────────

# Minimum alpha gap (GHz) between an active well and each frozen neighbour.
# Applied to the "ascending" stages (2/3/4) only; stage 5 (ordering="none") runs
# free. 2 GHz keeps the active pair clear of the frozen spectators without
# pushing it shallow (the good anchor's spectators are already well-separated, so
# the active windows stay deep, ~10-14 GHz -- unlike gapping the anchor itself).
FREEZE_GAP_GHZ = 2.0

# Freeze the spectator BETA too (full well-shape), not just alpha. True keeps the
# spectators genuinely static across anchors (only active wells move during the
# RTE morph -> the crossing-free property). Set False to freeze alpha only and
# let spectator beta re-optimise per stage.
FREEZE_BETA = True


def _active_wells_for_loss(loss_name):
    """Active (superposition) wells for a stage = (A-1, B-1).

    The stage modules expose module-level config indices `A, B` (in
    iter_bitstrings order, where config c in 1..4 is the single excitation of
    well c-1). The two superposition partners are therefore wells A-1 and B-1.
    """
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


# ── Optimisation (reduced to the active wells) ────────────────────────────────


def optimize_frozen(
    compute_loss,
    loss_name,
    active_wells,
    frozen_alpha,
    frozen_beta,
    x0_active_alpha,
    x0_active_beta,
    alpha_bounds_ghz=None,
    beta_bounds_ghz=None,
    gap=FREEZE_GAP_GHZ,
    apply_gap=True,
    freeze_beta=FREEZE_BETA,
    num_steps_cold=1000,
    num_steps_warm=200,
    maxiter=5000,
    rhobeg=0.1,
    rhoend=1e-3,
    seek_global_minimum=True,
    log_path=None,
):
    """PyBOBYQA over only the active wells' (alpha, beta); spectators frozen.

    `frozen_alpha`/`frozen_beta` are full length-NUM_WELLS arrays giving the
    pinned spectator values (entries at active positions are ignored and
    overwritten by the optimiser). The reduced variable vector is
    [alpha_active..., beta_active...] when `freeze_beta` is True the spectator
    betas are pinned too; otherwise all four betas are optimised.

    Trajectory/best are stored as full length-2*NUM_WELLS x-vectors so the
    existing `_save_trajectory` IO is reused unchanged.
    """
    alpha_b = _normalise_bounds(
        ALPHA_BOUNDS_GHZ if alpha_bounds_ghz is None else alpha_bounds_ghz,
        "alpha_bounds_ghz",
    )
    beta_b = _normalise_bounds(
        BETA_BOUNDS_GHZ if beta_bounds_ghz is None else beta_bounds_ghz,
        "beta_bounds_ghz",
    )

    active = list(active_wells)
    n_act = len(active)
    spectators = sorted(set(range(NUM_WELLS)) - set(active))

    frozen_alpha = np.asarray(frozen_alpha, dtype=float).copy()
    frozen_beta = np.asarray(frozen_beta, dtype=float).copy()

    # Which betas are optimised: active only (freeze_beta) or all four.
    beta_opt_wells = active if freeze_beta else list(range(NUM_WELLS))
    n_beta = len(beta_opt_wells)

    # Active alpha windows (ascending + frozen gap, or full bounds if no gap).
    a_lo, a_hi = _build_active_alpha_window(
        active, frozen_alpha, alpha_b, gap, apply_gap=apply_gap
    )
    # Beta bounds for the optimised betas.
    b_lo = beta_b[beta_opt_wells, 0]
    b_hi = beta_b[beta_opt_wells, 1]

    solver_lb = np.concatenate([a_lo, b_lo])
    solver_ub = np.concatenate([a_hi, b_hi])

    # x0 in reduced space: anchor-1 active alpha clamped into the window;
    # active (or all) betas clamped into their bounds.
    z0_alpha = np.clip(np.asarray(x0_active_alpha, dtype=float), a_lo, a_hi)
    if freeze_beta:
        z0_beta = np.clip(np.asarray(x0_active_beta, dtype=float), b_lo, b_hi)
    else:
        # Start all betas from the frozen/anchor-1 beta vector.
        z0_beta = np.clip(frozen_beta[beta_opt_wells], b_lo, b_hi)
    z0 = np.concatenate([z0_alpha, z0_beta])

    def _expand(z):
        """Reduced active vector -> full (alpha[NUM_WELLS], beta[NUM_WELLS])."""
        alpha = frozen_alpha.copy()
        beta = frozen_beta.copy()
        alpha[active] = z[:n_act]
        for j, w in enumerate(beta_opt_wells):
            beta[w] = z[n_act + j]
        return alpha, beta

    state = dict(
        y_prev=None, best_loss=np.inf, best_x=None, trajectory=[], eval_idx=0,
    )

    def loss_fn(z):
        state["eval_idx"] += 1
        i = state["eval_idx"]
        t0 = time.time()
        alpha_full, beta_full = _expand(z)
        x_full = np.concatenate([alpha_full, beta_full])  # length 2*NUM_WELLS

        num_steps = num_steps_cold if state["y_prev"] is None else num_steps_warm
        try:
            result = evaluate_params(
                alphas_ghz=alpha_full,
                betas_ghz=beta_full,
                compute_loss=compute_loss,
                y_prev=state["y_prev"],
                num_steps=num_steps,
            )
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            wall = time.time() - t0
            base_penalty = 1e6
            if state["best_x"] is not None:
                dist = np.linalg.norm(
                    (z - _reduce(state["best_x"])) / (solver_ub - solver_lb)
                )
                sentinel_loss = base_penalty * (1.0 + dist)
            else:
                sentinel_loss = base_penalty
            print(
                f"[{i:03d}] α={np.round(alpha_full, 3)}  β={np.round(beta_full, 3)}  "
                f"FAILED ({type(exc).__name__}: {exc}) "
                f"sentinel={sentinel_loss:.2e}  "
                f"warm={state['y_prev'] is not None}  {wall:.1f}s",
                flush=True,
            )
            state["y_prev"] = None
            state["trajectory"].append(
                dict(
                    eval=i, x=x_full.copy(), loss=sentinel_loss, min_fid=0.0,
                    fidelities=np.zeros(len(CONFIGS)),
                    matched_idx=np.zeros(len(CONFIGS), dtype=int),
                    energies=np.zeros(oqp.NUM_STATES), wall=wall, oob=True,
                )
            )
            return sentinel_loss

        loss = result["loss"]
        min_fid = float(result["fidelities"].min())
        wall = time.time() - t0
        worst_cfg = CONFIGS[int(np.argmin(result["fidelities"]))]
        print(
            f"[{i:03d}] α={np.round(alpha_full, 3)}  β={np.round(beta_full, 3)}  "
            f"loss={loss:.4f}  min_fid={min_fid:.4f} (cfg {worst_cfg})  "
            f"warm={state['y_prev'] is not None}  {wall:.1f}s",
            flush=True,
        )

        state["y_prev"] = result["y_final"]
        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_x"] = x_full.copy()
            state["best_result"] = result
        state["trajectory"].append(
            dict(
                eval=i, x=x_full.copy(), loss=loss, min_fid=min_fid,
                fidelities=result["fidelities"].copy(),
                matched_idx=result["matched_idx"].copy(),
                energies=result["energies"].copy(), wall=wall, oob=False,
            )
        )
        return loss

    def _reduce(x_full):
        """Full 8-vector -> reduced active z (for the sentinel distance)."""
        alpha = x_full[:NUM_WELLS]
        beta = x_full[NUM_WELLS:]
        return np.concatenate(
            [alpha[active], beta[beta_opt_wells]]
        )

    print("Starting frozen-spectator PyBOBYQA optimisation ...")
    print(f"  loss = {loss_name}")
    print(f"  active wells   = {active}")
    print(f"  frozen wells   = {spectators}")
    print(f"  frozen α (GHz) = {np.round(frozen_alpha[spectators], 4)} "
          f"at wells {spectators}")
    print(f"  frozen β (GHz) = {np.round(frozen_beta[spectators], 4)} "
          f"at wells {spectators}  (freeze_beta={freeze_beta})")
    if apply_gap:
        print(f"  min gap        = {gap} GHz (active alpha vs frozen neighbour)")
    else:
        print("  min gap        = none (active wells free over full bounds)")
    for j, w in enumerate(active):
        print(f"  active well {w}: α ∈ [{a_lo[j]:.3f}, {a_hi[j]:.3f}] GHz")
    print(f"  x0 active α (GHz) = {np.round(z0_alpha, 4)}")
    print(f"  x0 optimised β (GHz) = {np.round(z0_beta, 4)} at wells "
          f"{beta_opt_wells}")
    print(f"  num_steps cold/warm = {num_steps_cold}/{num_steps_warm}")
    print(f"  maxfun={maxiter}, rhobeg={rhobeg}, rhoend={rhoend}, "
          f"seek_global_minimum={seek_global_minimum}")

    t_start = time.time()
    soln = pybobyqa.solve(
        loss_fn,
        z0,
        bounds=(solver_lb, solver_ub),
        rhobeg=rhobeg,
        rhoend=rhoend,
        maxfun=maxiter,
        scaling_within_bounds=True,
        print_progress=False,
        seek_global_minimum=seek_global_minimum,
    )
    wall_total = time.time() - t_start

    print(f"\nOptimisation finished in {wall_total:.1f}s ({state['eval_idx']} evals).")
    print(f"Solver flag: {soln.flag} ({soln.msg})")
    print(f"Best loss = {state['best_loss']:.4f}")
    print(
        f"Best α (GHz) = {state['best_x'][:NUM_WELLS]}\n"
        f"Best β (GHz) = {state['best_x'][NUM_WELLS:]}"
    )

    if log_path is not None:
        # solver_x_physical is the full expanded best parameter vector.
        best_alpha, best_beta = _expand(np.asarray(soln.x))
        _save_trajectory(
            log_path, state, soln, wall_total, loss_name,
            alpha_bounds_ghz=alpha_b,
            beta_bounds_ghz=beta_b,
            x0_ghz=np.concatenate(_expand(z0)),
            x0_source=f"frozen:{loss_name}",
            alpha_ordering=(
                "frozen_spectators_gap" if apply_gap
                else "frozen_spectators_free"
            ),
            solver_x_physical=np.concatenate([best_alpha, best_beta]),
            seek_global_minimum=seek_global_minimum,
        )

    return state, soln


if __name__ == "__main__":
    if len(sys.argv) != 4 or not all(sys.argv[1:4]):
        sys.exit(
            "usage: optimize_qubit_params_frozen.py <run_name> <loss_name> "
            "<freeze_from>\n"
            "  loss_name   must match losses/<loss_name>.py and expose A, B "
            "(stages 2/3/4)\n"
            "  freeze_from prior run stem under ITE_trajectories/ (or .npz "
            "path);\n"
            "              its best_x supplies the frozen spectator (α,β) and "
            "the\n"
            "              active-well x0 (this is the stage-1 anchor)."
        )

    name = sys.argv[1]
    loss_name = sys.argv[2]
    freeze_from = sys.argv[3]

    spec = oqp.losses.load(loss_name)
    out_dir = HERE
    log_path = os.path.join(out_dir, "ITE_trajectories/" + name + "_" + loss_name + ".npz")

    traj_dir = os.path.join(out_dir, "ITE_trajectories")
    anchor_x, anchor_src = _resolve_warm_from(freeze_from, traj_dir)
    print(f"Freezing spectators from {anchor_src}: {anchor_x}")
    anchor_alpha = anchor_x[:NUM_WELLS]
    anchor_beta = anchor_x[NUM_WELLS:]

    active = _active_wells_for_loss(loss_name)

    # Keep the good (ungapped) anchor; gap only the ACTIVE wells against the
    # frozen spectators. "ascending" stages (2/3/4) get the FREEZE_GAP_GHZ window
    # (active pair held clear of the high/low frozen spectators); "none" stage 5
    # runs free (the ends must cross the frozen middles to reach the w0w3
    # resonance). The anchor itself is never gapped here, so no shallowing.
    apply_gap = spec.get("alpha_ordering", "none") == "ascending"

    state, soln = optimize_frozen(
        compute_loss=spec["compute"],
        loss_name=loss_name,
        active_wells=active,
        frozen_alpha=anchor_alpha,
        frozen_beta=anchor_beta,
        x0_active_alpha=anchor_alpha[active],
        x0_active_beta=anchor_beta[active],
        alpha_bounds_ghz=spec.get("alpha_bounds_ghz"),
        beta_bounds_ghz=spec.get("beta_bounds_ghz"),
        apply_gap=apply_gap,
        seek_global_minimum=spec.get("seek_global_minimum", True),
        maxiter=spec.get("maxiter", 5000),
        log_path=log_path,
    )
