"""Shared compute() factory for the superposition stages (stage2/3/4).

This module is imported by `losses/stageN_superpos_*.py`. Treat it as
**immutable once any stage referencing it has been submitted to Slurm**,
because every queued stage 2/3/4 job will pick up edits at job start.

`make_compute(A, B)` returns a closure with the standard loss-module
signature

    compute(*, C_final, bs_final, F, energies, num_wells) -> dict

With ascending alpha, single-excitation config index equals its
canonical eigenstate position (cfg 0 -> psi_0 = |0000>, cfg 1 -> psi_1 =
|1000>, ... cfg 4 -> psi_4 = |0001>). The factory penalises:
- the 2x2 sub-block F[[A,B], [A,B]] not equal to a balanced 50/50
  superposition (each of the four entries should be 0.5). This forces
  psi_A and psi_B to be maximally mixed between |cfg_A> and |cfg_B>;
  the (1 - F[A,A] - F[A,B]) form would be satisfied by clean uncoupled
  eigenstates too, which we don't want.
- the configs not in the superposition (ground + the 2 singles outside
  {A, B}) not landing on their canonical eigenstate positions
  (pinned-singles term, weighted by `WEIGHTS_BY_CFG_IDX`),
- doubles, triples, |1111> not Hungarian-aligning to eigenstates >= 5
  (higher term).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHA_BOUNDS_GHZ = np.tile(np.array([5.0, 15.0]), (4, 1))
ALPHA_ORDERING = "ascending"
SEEK_GLOBAL_MINIMUM = True
MAXITER = 5000

W_SUPER = 5.0
# Weights indexed by config slot in iter_bitstrings(4, 2) order
# (Hamming-weight ascending, then species-major).
WEIGHTS_BY_CFG_IDX = np.array([
    3.0,                           # 0:  |0000>
    1.0, 1.0, 1.0, 1.0,            # 1-4: singles
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # 5-10: doubles
    0.15, 0.15, 0.15, 0.15,        # 11-14: triples
    0.15,                          # 15: |1111>
])

PINNED_CANDIDATES = np.array([0, 1, 2, 3, 4])  # ground + singles
HIGHER_CFG_IDX = np.arange(5, 16)  # doubles + triples + |1111>


def make_compute(A: int, B: int):
    def compute(*, C_final, bs_final, F, energies, num_wells):
        n_states, n_cfg = F.shape

        # superposition lives at eigenstates A and B; force the 2x2 sub-block
        # F[[A,B], [A,B]] to a balanced 50/50 mix (all four entries == 0.5).
        super_loss = (
            (F[A, A] - 0.5) ** 2
            + (F[A, B] - 0.5) ** 2
            + (F[B, A] - 0.5) ** 2
            + (F[B, B] - 0.5) ** 2
        )

        # ground + singles not in the superposition: pinned to their
        # canonical eigenstate position (cfg index == eigenstate position).
        pinned_cfg_idx = PINNED_CANDIDATES[~np.isin(PINNED_CANDIDATES, (A, B))]
        pinned_weights = WEIGHTS_BY_CFG_IDX[pinned_cfg_idx]
        pinned_fids = F[pinned_cfg_idx, pinned_cfg_idx]
        pinned_loss = float(
            np.sum(pinned_weights) - np.sum(pinned_weights * pinned_fids)
        )

        # higher-Hamming-weight configs: Hungarian against eigenstates {5..}.
        higher_weights = WEIGHTS_BY_CFG_IDX[HIGHER_CFG_IDX]
        F_sub = F[5:, HIGHER_CFG_IDX]
        cost = -(F_sub * higher_weights[None, :])
        row_ind, col_ind = linear_sum_assignment(cost)
        order = np.argsort(col_ind)
        higher_state_for_cfg = (row_ind + 5)[order]
        higher_fids = F[higher_state_for_cfg, HIGHER_CFG_IDX]
        higher_loss = float(
            np.sum(higher_weights) - np.sum(higher_weights * higher_fids)
        )

        loss = float(W_SUPER * super_loss + pinned_loss + higher_loss)

        fidelities = np.zeros(n_cfg)
        matched_idx = np.full(n_cfg, -1, dtype=int)

        # superposition partners: report max-overlap eigenstate in {A, B}
        matched_idx[A] = A if F[A, A] >= F[B, A] else B
        fidelities[A] = float(F[matched_idx[A], A])
        matched_idx[B] = B if matched_idx[A] == A else A
        fidelities[B] = float(F[matched_idx[B], B])

        fidelities[pinned_cfg_idx] = pinned_fids
        matched_idx[pinned_cfg_idx] = pinned_cfg_idx
        fidelities[HIGHER_CFG_IDX] = higher_fids
        matched_idx[HIGHER_CFG_IDX] = higher_state_for_cfg

        return dict(loss=loss, fidelities=fidelities, matched_idx=matched_idx)

    return compute
