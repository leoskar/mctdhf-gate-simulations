"""Stage 1 of the chain: lowest 5 eigenstates = |0000>, |1000>, |0100>,
|0010>, |0001>. Other configs are still matched but at lower weight."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHA_BOUNDS_GHZ = np.tile(np.array([5.0, 15.0]), (4, 1))
ALPHA_ORDERING = "ascending"
SEEK_GLOBAL_MINIMUM = True
MAXITER = 5000

WEIGHTS = np.array(
    [3.0,
     3.0, 3.0, 3.0, 3.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     0.3, 0.3, 0.3, 0.3,
     0.3]
)

PINNED_CFG_IDX = np.arange(5)       # |0000>, singles → eigenstates 0..4
HIGHER_CFG_IDX = np.arange(5, 16)   # doubles, triples, |1111>


def compute(*, C_final, bs_final, F, energies, num_wells):
    F_np = np.asarray(F)

    higher_weights = WEIGHTS[HIGHER_CFG_IDX]
    F_sub = F_np[5:, HIGHER_CFG_IDX]
    row_ind, col_ind = linear_sum_assignment(-(F_sub * higher_weights[None, :]))
    order = np.argsort(col_ind)
    higher_state_for_cfg = (row_ind + 5)[order]

    fidelities = np.empty(len(WEIGHTS))
    fidelities[PINNED_CFG_IDX] = F_np[PINNED_CFG_IDX, PINNED_CFG_IDX]
    fidelities[HIGHER_CFG_IDX] = F_np[higher_state_for_cfg, HIGHER_CFG_IDX]

    matched_idx = np.empty(len(WEIGHTS), dtype=int)
    matched_idx[PINNED_CFG_IDX] = PINNED_CFG_IDX
    matched_idx[HIGHER_CFG_IDX] = higher_state_for_cfg

    loss = float(WEIGHTS.sum() - (WEIGHTS * fidelities).sum())
    return dict(loss=loss, fidelities=fidelities, matched_idx=matched_idx)
