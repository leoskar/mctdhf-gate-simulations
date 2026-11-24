"""Pluggable loss functions for optimize_qubit_params.py.

Each loss lives in its own module `losses/<name>.py` exposing a `compute`
callable with the signature

    compute(*, C_final, bs_final, F, energies, num_wells) -> dict

returning at minimum {"loss": float, "fidelities": ndarray,
"matched_idx": ndarray}.

A loss module may additionally expose any subset of these module-level
attributes to override optimizer settings:

  ALPHA_BOUNDS_GHZ : array shape (NUM_WELLS, 2), per-well (lo, hi) for alpha.
                     A shape-(2,) value is also accepted and broadcast.
  BETA_BOUNDS_GHZ  : same convention as ALPHA_BOUNDS_GHZ, for beta.
  X0_GHZ           : length 2*NUM_WELLS, ordered [alpha_0..alpha_{K-1},
                     beta_0..beta_{K-1}].
  ALPHA_ORDERING   : "none" (default) or "ascending". When "ascending",
                     PyBOBYQA is reparametrised onto [alpha_0, gaps, beta]
                     so alpha_0 < alpha_1 < ... < alpha_{K-1} is enforced
                     by construction. See optimize_qubit_params.optimize.
  SEEK_GLOBAL_MINIMUM : bool, default True. Forwarded to pybobyqa.solve.
                     Set False for smooth/well-conditioned losses where
                     restart heuristics waste evaluations.
  MAXITER          : int, default 5000. Maximum number of loss evaluations
                     (pybobyqa `maxfun`).

Adding a new loss = adding a new file. Existing modules are never edited
after submission, so already-queued Slurm jobs are immune to new losses.
"""
from __future__ import annotations

import importlib


def load(name: str) -> dict:
    module = importlib.import_module(f"losses.{name}")
    spec: dict = {"compute": module.compute}
    for attr in (
        "ALPHA_BOUNDS_GHZ",
        "BETA_BOUNDS_GHZ",
        "X0_GHZ",
        "ALPHA_ORDERING",
        "SEEK_GLOBAL_MINIMUM",
        "MAXITER",
    ):
        if hasattr(module, attr):
            spec[attr.lower()] = getattr(module, attr)
    return spec
