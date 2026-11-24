"""Stage 5: psi superposition of |1000> and |0001> (configs A, B), w0/w3.

This stage runs with ALPHA_ORDERING = "none" (alphas are NOT forced ascending),
so we CANNOT assume "eigenstate index == config index". The shared
`_superpos.make_compute` bakes in exactly that ascending assumption -- pinned
singles via F[cfg, cfg], higher configs Hungarian-matched only against
eigenstates >= 5 -- so it is deliberately NOT reused here.

Instead the loss is fully assignment-based and order-agnostic:
- the 14 configs outside the superposition are weighted-Hungarian matched
  (weights = `WEIGHTS_BY_CFG_IDX`) against ALL eigenstates, each landing on a
  distinct eigenstate;
- the 2 superposition configs A, B live on the 2 leftover eigenstates carrying
  the most A/B character, whose 2x2 population sub-block is forced to a balanced
  50/50 mix (every entry == 0.5). The 50/50 form (rather than
  1 - F[A,A] - F[A,B]) rejects clean, unmixed eigenstates, which is the whole
  point of the superposition stage.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from losses._superpos import (
    ALPHA_BOUNDS_GHZ,
    SEEK_GLOBAL_MINIMUM,
    MAXITER,
    W_SUPER,
    WEIGHTS_BY_CFG_IDX,
)

ALPHA_ORDERING = "none"

A, B = 1, 4  # |1000>, |0001>


def compute(*, C_final, bs_final, F, energies, num_wells):
    n_states, n_cfg = F.shape

    super_cfg = np.array([A, B])
    non_super_cfg = np.setdiff1d(np.arange(n_cfg), super_cfg)
    non_super_w = WEIGHTS_BY_CFG_IDX[non_super_cfg]

    # Weighted Hungarian: each non-superposition config -> a distinct eigenstate,
    # chosen across ALL eigenstates (no ascending index == config assumption).
    cost = -(F[:, non_super_cfg] * non_super_w[None, :])  # (n_states, 14)
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind is a permutation of range(14); recover the eigenstate per config.
    order = np.argsort(col_ind)
    eig_for_non_super = row_ind[order]                    # aligned to non_super_cfg
    non_super_fids = F[eig_for_non_super, non_super_cfg]
    pinned_loss = float(
        np.sum(non_super_w) - np.sum(non_super_w * non_super_fids)
    )

    # Superposition: the 2 leftover eigenstates carrying the most A/B character,
    # forced to a balanced 50/50 2x2 sub-block (all four entries == 0.5).
    used = set(int(s) for s in eig_for_non_super)
    leftover = np.array([s for s in range(n_states) if s not in used])
    ab_char = F[leftover, A] + F[leftover, B]
    pair = leftover[np.argsort(-ab_char)[:2]]
    p, q = int(pair[0]), int(pair[1])
    super_loss = (
        (F[p, A] - 0.5) ** 2
        + (F[p, B] - 0.5) ** 2
        + (F[q, A] - 0.5) ** 2
        + (F[q, B] - 0.5) ** 2
    )

    loss = float(W_SUPER * super_loss + pinned_loss)

    fidelities = np.zeros(n_cfg)
    matched_idx = np.full(n_cfg, -1, dtype=int)

    fidelities[non_super_cfg] = non_super_fids
    matched_idx[non_super_cfg] = eig_for_non_super

    # superposition partners: report the max-overlap eigenstate of the pair.
    matched_idx[A] = p if F[p, A] >= F[q, A] else q
    fidelities[A] = float(F[matched_idx[A], A])
    matched_idx[B] = q if matched_idx[A] == p else p
    fidelities[B] = float(F[matched_idx[B], B])

    return dict(loss=loss, fidelities=fidelities, matched_idx=matched_idx)
